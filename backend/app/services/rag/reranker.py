"""Cross-encoder reranker (stage 2 of retrieval).

Stage 1 (hybrid dense+sparse) is a fast bi-encoder recall step that returns N
candidates. This cross-encoder re-scores each (query, chunk) pair *together* —
cross-attention lets query tokens attend to chunk tokens, which a bi-encoder
(separate vectors, compared by cosine) fundamentally cannot do. Accurate but
slow, so we only run it over the small candidate set, never the whole corpus.
"""
import asyncio
import logging
from functools import lru_cache

from sentence_transformers import CrossEncoder

from app.core.config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    logger.info("Loading reranker model: %s", settings.reranker_model)
    model = CrossEncoder(settings.reranker_model)
    logger.info("Reranker model loaded.")
    return model


def rerank_sync(query: str, chunks: list[dict], top_k: int) -> list[dict]:
    """Re-score candidates with the cross-encoder; return the top_k by score."""
    if not chunks:
        return []
    pairs = [(query, c.get("content") or "") for c in chunks]
    scores = get_reranker().predict(pairs)
    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = float(score)
    ranked = sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)
    return ranked[:top_k]


async def arerank(query: str, chunks: list[dict], top_k: int) -> list[dict]:
    """Async wrapper: predict() is blocking CPU work — offload off the loop."""
    return await asyncio.to_thread(rerank_sync, query, chunks, top_k)
