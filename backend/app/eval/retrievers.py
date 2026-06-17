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
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable

from qdrant_client.models import SparseVector

from app.core.config import settings
from app.services.rag.embedder import aembed_query
from app.services.rag.qdrant_store import get_qdrant_client, _build_sparse_vector
from app.services.rag.retriever import hybrid_search

Retriever = Callable[[str, int], Awaitable[list[str]]]


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


# Order matters: this is the column order in the results table.
STRATEGIES: dict[str, Retriever] = {
    "dense": dense_only,
    "sparse": sparse_only,
    "hybrid": hybrid,
    "hybrid+rerank": hybrid_rerank,
}
