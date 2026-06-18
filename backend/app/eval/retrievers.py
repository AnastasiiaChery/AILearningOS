"""Retrieval strategies under test — each returns a ranked list of chunk_ids.

Four strategies so the harness can isolate where quality comes from:
  dense_only    – semantic single-vector search alone.
  sparse_only   – BM25/keyword (sparse) search alone.
  hybrid        – dense+sparse fused with RRF (production, rerank off).
  hybrid_rerank – hybrid candidates re-scored by the cross-encoder (production).

`hybrid` / `hybrid_rerank` call the real `hybrid_search`, so they measure exactly
what the app serves. `dense_only` / `sparse_only` query Qdrant directly (the app
never exposes them) and exist purely as diagnostic baselines: they show how much
the keyword side and the fusion actually buy us.

The `*+hyde` strategies (Track C, exp.2) embed the dense side from a hypothetical
answer passage. All three HyDE strategies share ONE draft per question via
`_hypo` — so dense+hyde / hybrid+hyde / hybrid+hyde+rerank are compared on the
same hypothetical doc (and we pay one LLM call per question, not three).
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable

from qdrant_client.models import SparseVector

from app.core.config import settings
from app.services.rag.embedder import aembed_query
from app.services.rag.hyde import generate_hypothetical
from app.services.rag.qdrant_store import get_qdrant_client, _build_sparse_vector
from app.services.rag.retriever import hybrid_search

# A strategy returns a ranked chunk_id list, or (parent-document) a ranked list
# of blocks — each block the chunk_ids it covers. run_eval normalizes both.
Retriever = Callable[[str, int], Awaitable[list]]

# Per-run memo: question text → hypothetical doc. Sequential eval reuses the draft
# across the three HyDE strategies, keeping the comparison fair and cheap.
_hypo_cache: dict[str, str] = {}


async def _hypo(query: str) -> str:
    if query not in _hypo_cache:
        _hypo_cache[query] = await generate_hypothetical(query)
    return _hypo_cache[query]


async def dense_only(query: str, k: int) -> list[str]:
    client = get_qdrant_client()
    vec = await aembed_query(query)
    res = await client.query_points(
        collection_name=settings.qdrant_collection,
        query=vec,
        using="dense",
        limit=k,
        with_payload=["chunk_id"],
    )
    return [(p.payload or {}).get("chunk_id") for p in res.points]


async def sparse_only(query: str, k: int) -> list[str]:
    client = get_qdrant_client()
    sv = _build_sparse_vector(query)
    res = await client.query_points(
        collection_name=settings.qdrant_collection,
        query=SparseVector(indices=sv.indices, values=sv.values),
        using="sparse",
        limit=k,
        with_payload=["chunk_id"],
    )
    return [(p.payload or {}).get("chunk_id") for p in res.points]


async def hybrid(query: str, k: int) -> list[str]:
    chunks = await hybrid_search(query, top_k=k, rerank=False)
    return [c.get("chunk_id") for c in chunks]


async def hybrid_rerank(query: str, k: int) -> list[str]:
    chunks = await hybrid_search(query, top_k=k, rerank=True)
    return [c.get("chunk_id") for c in chunks]


async def dense_hyde(query: str, k: int) -> list[str]:
    """dense_only, but the search vector comes from the hypothetical doc."""
    client = get_qdrant_client()
    vec = await aembed_query(await _hypo(query))
    res = await client.query_points(
        collection_name=settings.qdrant_collection,
        query=vec,
        using="dense",
        limit=k,
        with_payload=["chunk_id"],
    )
    return [(p.payload or {}).get("chunk_id") for p in res.points]


async def hybrid_hyde(query: str, k: int) -> list[str]:
    chunks = await hybrid_search(query, top_k=k, rerank=False, hyde_doc=await _hypo(query))
    return [c.get("chunk_id") for c in chunks]


async def hybrid_hyde_rerank(query: str, k: int) -> list[str]:
    chunks = await hybrid_search(query, top_k=k, rerank=True, hyde_doc=await _hypo(query))
    return [c.get("chunk_id") for c in chunks]


# Parent-document strategies (Track C, exp.6). Same prod path (hybrid+rerank),
# but each hit is expanded into its parent block on the return path. These return
# a list of BLOCKS — each block the list of chunk_ids it covers — so the harness
# credits every covered chunk (block_* metrics). The ranking is hit-driven and
# unchanged; the measurable angle is coverage of multi-chunk gold (one section
# whose answer is split across chunks).
async def parent_section(query: str, k: int) -> list[list[str]]:
    blocks = await hybrid_search(query, top_k=k, rerank=True, parent="section")
    return [b.get("member_chunk_ids") or [b.get("chunk_id")] for b in blocks]


async def parent_window(query: str, k: int) -> list[list[str]]:
    blocks = await hybrid_search(query, top_k=k, rerank=True, parent="window")
    return [b.get("member_chunk_ids") or [b.get("chunk_id")] for b in blocks]


# Order matters: this is the column order in the results table. HyDE variants sit
# next to their non-HyDE twins so the до/после читается по строкам.
STRATEGIES: dict[str, Retriever] = {
    "dense": dense_only,
    "dense+hyde": dense_hyde,
    "sparse": sparse_only,
    "hybrid": hybrid,
    "hybrid+hyde": hybrid_hyde,
    "hybrid+rerank": hybrid_rerank,
    "hybrid+hyde+rerank": hybrid_hyde_rerank,
    "parent-sec": parent_section,
    "parent-win": parent_window,
}
