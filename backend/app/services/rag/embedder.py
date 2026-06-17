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


def embed_texts(texts: list[str], prefix: str = "") -> list[list[float]]:
    """Synchronous encode. CPU-bound — do NOT call directly from async code.

    ``prefix`` is prepended to every text before encoding. e5 models require a
    "query: " / "passage: " instruction prefix (see config); prefix-free models
    pass "".
    """
    model = get_embedder()
    if prefix:
        texts = [prefix + t for t in texts]
    # normalize_embeddings=True → unit-length vectors, so cosine == dot product,
    # which is what the Qdrant COSINE distance expects.
    embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return embeddings.tolist()


def embed_passages(texts: list[str]) -> list[list[float]]:
    """Embed documents/chunks for storage (e5 "passage: " prefix)."""
    return embed_texts(texts, settings.embedding_passage_prefix)


def embed_query(query: str) -> list[float]:
    """Embed a search query (e5 "query: " prefix).

    Asymmetric prefixing matters: a query and the passage that answers it are
    encoded with *different* prefixes, which is how e5 was trained.
    """
    return embed_texts([query], settings.embedding_query_prefix)[0]


# Async wrappers: model.encode is a blocking CPU-bound call. Running it directly
# in an async path would freeze the event loop (no other request progresses for
# the whole encode). asyncio.to_thread offloads it to a worker thread so the
# loop stays responsive.
async def aembed_passages(texts: list[str]) -> list[list[float]]:
    return await asyncio.to_thread(embed_passages, texts)


async def aembed_query(query: str) -> list[float]:
    return await asyncio.to_thread(embed_query, query)
