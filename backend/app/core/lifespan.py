import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    Modifier,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
)

from .config import settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up AI Learning OS...")

    # Initialize Qdrant collection
    client = AsyncQdrantClient(url=settings.qdrant_url)
    collections = await client.get_collections()
    existing = {c.name for c in collections.collections}

    if settings.qdrant_collection not in existing:
        logger.info("Creating Qdrant collection: %s", settings.qdrant_collection)
        await client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config={
                "dense": VectorParams(
                    size=settings.embedding_dimension,
                    distance=Distance.COSINE,
                )
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(
                    index=SparseIndexParams(on_disk=False),
                    # Qdrant computes IDF from real corpus document-frequency and
                    # applies it at query time. We store only TF — see qdrant_store.
                    modifier=Modifier.IDF,
                )
            },
        )
    await client.close()

    # Lazy-load embedder to avoid import-time model download
    from app.services.rag.embedder import get_embedder
    get_embedder()
    logger.info("Embedder loaded.")

    # Warm the cross-encoder reranker so the first query isn't slow
    if settings.rerank_enabled:
        from app.services.rag.reranker import get_reranker
        get_reranker()
        logger.info("Reranker loaded.")

    # Compile LangGraph agents and store on app.state
    from app.services.agents.mentor_agent import build_mentor_graph
    from app.services.agents.planner_agent import build_planner_graph
    app.state.mentor_graph = build_mentor_graph()
    app.state.planner_graph = build_planner_graph()
    logger.info("LangGraph agents compiled.")

    yield

    logger.info("Shutting down...")
