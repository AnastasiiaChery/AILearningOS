"""Hybrid search retriever using RRF fusion."""
import logging

from qdrant_client.models import (
    FusionQuery,
    Fusion,
    Prefetch,
    SparseVector,
)

from app.core.config import settings
from app.services.rag.qdrant_store import get_qdrant_client, _build_sparse_vector
from app.services.rag.embedder import aembed_query

logger = logging.getLogger(__name__)


async def hybrid_search(
    query: str, top_k: int | None = None, rerank: bool | None = None
) -> list[dict]:
    """Two-stage retrieval: hybrid dense+sparse (RRF) → optional cross-encoder rerank.

    Stage 1 pulls a wider candidate set (rerank_candidates) for recall; stage 2
    re-scores those candidates and keeps top_k for precision.
    """
    k = top_k or settings.retrieval_top_k
    do_rerank = settings.rerank_enabled if rerank is None else rerank
    # Stage-1 breadth: with rerank we want more candidates to re-score; without
    # it we just take k directly from RRF.
    candidate_n = settings.rerank_candidates if do_rerank else k
    client = get_qdrant_client()

    dense_vec = await aembed_query(query)
    sparse_vec = _build_sparse_vector(query)

    results = await client.query_points(
        collection_name=settings.qdrant_collection,
        prefetch=[
            Prefetch(query=dense_vec, using="dense", limit=candidate_n * 2),
            Prefetch(
                query=SparseVector(indices=sparse_vec.indices, values=sparse_vec.values),
                using="sparse",
                limit=candidate_n * 2,
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
                "score": point.score,
            }
        )

    if do_rerank and chunks:
        from app.services.rag.reranker import arerank

        reranked = await arerank(query, chunks, k)
        logger.debug(
            "Hybrid+rerank for %r: %d candidates → top %d.", query, len(chunks), len(reranked)
        )
        return reranked

    logger.debug("Hybrid search for %r returned %d chunks.", query, len(chunks))
    return chunks[:k]
