"""Hybrid search retriever using RRF fusion."""
import logging

from qdrant_client.models import (
    FusionQuery,
    Fusion,
    Prefetch,
    SparseVector,
    Filter,
    FieldCondition,
    MatchAny,
)

from app.core.config import settings
from app.services.rag.qdrant_store import (
    get_qdrant_client,
    _build_sparse_vector,
    fetch_document_chunks,
)
from app.services.rag.embedder import aembed_query

logger = logging.getLogger(__name__)


def _heading_key(chunk: dict) -> tuple:
    """Identity of a section: a chunk's document + full heading-path.

    Heading titles are NOT unique across documents, so the key includes
    ``document_id`` — two docs with an identically-named "## Caching" section are
    different parents.
    """
    return (
        chunk.get("document_id"),
        chunk.get("h1_title"),
        chunk.get("h2_title"),
        chunk.get("h3_title"),
    )


def _block_from(hit: dict, members: list[dict]) -> dict:
    """Assemble a parent block: one return slot carrying several chunks.

    ``content`` is the members stitched by chunk_index (what the LLM reads);
    ``member_chunk_ids`` lets the eval harness credit every covered chunk. The
    block keeps the *primary* hit's id/score so ranking stays hit-driven.
    """
    members = sorted(members, key=lambda c: c.get("chunk_index", 0)) or [hit]
    return {
        "chunk_id": hit.get("chunk_id"),
        "member_chunk_ids": [c.get("chunk_id") for c in members],
        "document_id": hit.get("document_id"),
        "chunk_index": hit.get("chunk_index"),
        "filename": hit.get("filename"),
        "file_type": hit.get("file_type"),
        "content": "\n\n".join(c.get("content") or "" for c in members),
        "h1_title": hit.get("h1_title"),
        "h2_title": hit.get("h2_title"),
        "h3_title": hit.get("h3_title"),
        "page_number": hit.get("page_number"),
        "score": hit.get("score"),
    }


async def _expand_to_parents(hits: list[dict], mode: str, window: int) -> list[dict]:
    """Expand ranked hits into parent blocks, in hit order, de-duplicated.

    ``section`` gathers every sibling sharing the hit's heading-path; ``window``
    takes chunk_index ± ``window`` within the same document. A hit whose parent is
    already covered by an earlier block is dropped (so duplicate hits in one
    section collapse to a single block and free slots for other parents). The
    index is never touched — this is a pure return-path transform.
    """
    # Each involved document is scrolled once; tiny corpus, so this is cheap.
    docs: dict[str, list[dict]] = {}
    for h in hits:
        did = h.get("document_id")
        if did and did not in docs:
            docs[did] = await fetch_document_chunks(did)

    blocks: list[dict] = []
    seen_sections: set[tuple] = set()
    covered: set[str] = set()  # chunk_ids already inside some emitted block

    for h in hits:
        siblings = docs.get(h.get("document_id")) or [h]
        if mode == "section":
            key = _heading_key(h)
            if key in seen_sections:
                continue
            seen_sections.add(key)
            members = [c for c in siblings if _heading_key(c) == key] or [h]
        else:  # window
            if h.get("chunk_id") in covered:
                continue  # this hit already rode along inside an earlier window
            hi = h.get("chunk_index")
            members = (
                [
                    c
                    for c in siblings
                    if hi is not None
                    and c.get("chunk_index") is not None
                    and abs(c["chunk_index"] - hi) <= window
                ]
                if hi is not None
                else [h]
            ) or [h]
        for c in members:
            covered.add(c.get("chunk_id"))
        blocks.append(_block_from(h, members))
    return blocks


def _doc_filter(document_ids: list[str] | None) -> Filter | None:
    """Restrict retrieval to a set of documents (None → search the whole KB).

    ``document_id`` is stored in the payload as a string, so ids are stringified
    before matching. Used to scope a plan or quiz to user-selected files.
    """
    if not document_ids:
        return None
    return Filter(
        must=[FieldCondition(key="document_id", match=MatchAny(any=[str(d) for d in document_ids]))]
    )


async def hybrid_search(
    query: str,
    top_k: int | None = None,
    rerank: bool | None = None,
    hyde_doc: str | None = None,
    parent: str | None = None,
    document_ids: list[str] | None = None,
) -> list[dict]:
    """Two-stage retrieval: hybrid dense+sparse (RRF) → optional cross-encoder rerank.

    Stage 1 pulls a wider candidate set (rerank_candidates) for recall; stage 2
    re-scores those candidates and keeps top_k for precision.

    HyDE (``settings.hyde_enabled`` or a caller-supplied ``hyde_doc``): the dense
    side is embedded from a hypothetical answer passage instead of the raw query;
    the sparse side always uses the original query. Pass ``hyde_doc`` to inject a
    pre-generated passage (the eval harness does this to reuse one draft across
    strategies); leave it None and the flag on to generate on the fly.

    Parent-document (``settings.parent_retrieval`` or a caller-supplied
    ``parent``): when "section"/"window", the ranked hits are expanded into parent
    blocks (sibling chunks stitched together) on the return path — same search,
    fuller context. Each returned dict then carries ``member_chunk_ids`` and a
    stitched ``content``; ``top_k`` counts blocks, not chunks. See Module 11.
    """
    k = top_k or settings.retrieval_top_k
    do_rerank = settings.rerank_enabled if rerank is None else rerank
    parent_mode = settings.parent_retrieval if parent is None else parent
    parent_mode = None if parent_mode in (None, "", "off") else parent_mode
    # Stage-1 breadth: with rerank we want more candidates to re-score; without
    # it we just take k directly from RRF.
    candidate_n = settings.rerank_candidates if do_rerank else k
    client = get_qdrant_client()

    # Dense side may search by a hypothetical doc (HyDE); sparse stays on the query.
    dense_text = query
    if hyde_doc is not None:
        dense_text = hyde_doc
    elif settings.hyde_enabled:
        from app.services.rag.hyde import generate_hypothetical

        dense_text = await generate_hypothetical(query)

    dense_vec = await aembed_query(dense_text)
    sparse_vec = _build_sparse_vector(query)

    # Scope candidates to selected documents (if any). The filter goes on each
    # prefetch so both dense and sparse sides search only those docs before RRF.
    doc_filter = _doc_filter(document_ids)

    results = await client.query_points(
        collection_name=settings.qdrant_collection,
        prefetch=[
            Prefetch(query=dense_vec, using="dense", limit=candidate_n * 2, filter=doc_filter),
            Prefetch(
                query=SparseVector(indices=sparse_vec.indices, values=sparse_vec.values),
                using="sparse",
                limit=candidate_n * 2,
                filter=doc_filter,
            ),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=candidate_n,
        with_payload=True,
    )

    chunks = []
    for point in results.points:
        payload = point.payload or {}
        chunks.append(
            {
                "chunk_id": payload.get("chunk_id"),
                "document_id": payload.get("document_id"),
                "filename": payload.get("filename"),
                "file_type": payload.get("file_type"),
                "content": payload.get("content"),
                "h1_title": payload.get("h1_title"),
                "h2_title": payload.get("h2_title"),
                "h3_title": payload.get("h3_title"),
                "page_number": payload.get("page_number"),
                "chunk_index": payload.get("chunk_index"),
                "score": point.score,
            }
        )

    if do_rerank and chunks:
        from app.services.rag.reranker import arerank

        # Re-score ALL candidates (not just k): parent expansion below dedups
        # hits into blocks, so it needs a deeper ranked list to refill k slots.
        ranked = await arerank(query, chunks, len(chunks))
        logger.debug(
            "Hybrid+rerank for %r: %d candidates re-scored.", query, len(chunks)
        )
    else:
        ranked = chunks

    if parent_mode:
        blocks = await _expand_to_parents(ranked, parent_mode, settings.parent_window)
        logger.debug(
            "Parent(%s) for %r: %d hits → %d blocks (top %d).",
            parent_mode, query, len(ranked), len(blocks), k,
        )
        return blocks[:k]

    return ranked[:k]
