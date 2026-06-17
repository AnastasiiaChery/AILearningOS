"""Singleton SentenceTransformer embedder."""
import asyncio
import logging
from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.core.config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    logger.info("Loading embedding model: %s", settings.embedding_model)
    model = SentenceTransformer(settings.embedding_model)
    logger.info("Embedding model loaded.")
    return model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Synchronous encode. CPU-bound — do NOT call directly from async code."""
    model = get_embedder()
    # normalize_embeddings=True → unit-length vectors, so cosine == dot product,
    # which is what the Qdrant COSINE distance expects.
    embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    return embed_texts([query])[0]


# Async wrappers: model.encode is a blocking CPU-bound call. Running it directly
# in an async path would freeze the event loop (no other request progresses for
# the whole encode). asyncio.to_thread offloads it to a worker thread so the
# loop stays responsive.
async def aembed_texts(texts: list[str]) -> list[list[float]]:
    return await asyncio.to_thread(embed_texts, texts)


async def aembed_query(query: str) -> list[float]:
    return await asyncio.to_thread(embed_query, query)
