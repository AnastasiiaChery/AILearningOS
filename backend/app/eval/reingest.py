"""Recreate the Qdrant collection at the configured embedding dimension and
re-ingest the current corpus.

Required whenever the embedding model changes (Track C experiments are all
re-ingests): a new model means a new vector size AND a new tokenizer, so the
collection must be recreated and every chunk re-split + re-embedded.

  docker compose exec -T backend sh -c \
    'cd /app && PYTHONPATH=/app uv run python -m app.eval.reingest'

It re-ingests exactly the documents already present in the collection, so the
before/after corpus is identical — only the encoder (and its tokenization)
changes. If the collection is empty/absent it falls back to every document with
status='ready' in Postgres. Postgres DocumentChunk rows are refreshed to match
the new chunk boundaries.
"""
import argparse
import asyncio
import logging
import os
import uuid

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    Modifier,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
)
from sqlalchemy import delete, select

from app.core.config import settings

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("reingest")


async def _corpus_document_ids(client: AsyncQdrantClient) -> list[str] | None:
    """Distinct document_ids currently in the collection, or None if empty/absent."""
    if not await client.collection_exists(settings.qdrant_collection):
        return None
    ids: set[str] = set()
    offset = None
    while True:
        points, offset = await client.scroll(
            collection_name=settings.qdrant_collection,
            limit=256,
            with_payload=["document_id"],
            offset=offset,
        )
        for p in points:
            did = (p.payload or {}).get("document_id")
            if did:
                ids.add(did)
        if offset is None:
            break
    return sorted(ids) or None


async def _recreate_collection(client: AsyncQdrantClient) -> None:
    """Drop + create with the same config as app startup (lifespan), but at the
    dimension currently in settings."""
    if await client.collection_exists(settings.qdrant_collection):
        await client.delete_collection(settings.qdrant_collection)
    await client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config={
            "dense": VectorParams(
                size=settings.embedding_dimension, distance=Distance.COSINE
            )
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(
                index=SparseIndexParams(on_disk=False), modifier=Modifier.IDF
            )
        },
    )
    logger.info(
        "Recreated collection %s @ %dd (model=%s)",
        settings.qdrant_collection,
        settings.embedding_dimension,
        settings.embedding_model,
    )


async def _reingest_document(session, doc) -> int:
    from app.models.document import DocumentChunk
    from app.services.ingestion.markdown_loader import load_markdown
    from app.services.ingestion.pdf_loader import load_pdf
    from app.services.rag.qdrant_store import upsert_chunks

    path = os.path.join(settings.upload_dir, doc.filename)
    chunks = load_markdown(path) if doc.file_type == "markdown" else load_pdf(path)
    chunk_ids = [uuid.uuid4() for _ in chunks]

    # Refresh Postgres rows so they match the new chunk boundaries.
    await session.execute(
        delete(DocumentChunk).where(DocumentChunk.document_id == doc.id)
    )
    session.add_all(
        [
            DocumentChunk(
                id=chunk_ids[i],
                document_id=doc.id,
                qdrant_id=chunk_ids[i],
                chunk_index=c.chunk_index,
                content=c.content,
                h1_title=c.h1_title,
                h2_title=c.h2_title,
                h3_title=c.h3_title,
                page_number=c.page_number,
                token_count=c.token_count,
            )
            for i, c in enumerate(chunks)
        ]
    )
    await upsert_chunks(
        chunks, str(doc.id), doc.original_filename, doc.file_type, chunk_ids
    )
    doc.chunk_count = len(chunks)
    await session.commit()
    return len(chunks)


async def main() -> None:
    from app.core.database import AsyncSessionLocal
    from app.models.document import Document

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--filenames",
        default="",
        help="comma-separated original_filenames to ingest (deduped, first row "
        "per name). Use this to pin a reproducible corpus. Default: the docs "
        "currently in the collection, else every 'ready' document.",
    )
    args = ap.parse_args()
    want = [f.strip() for f in args.filenames.split(",") if f.strip()]

    client = AsyncQdrantClient(url=settings.qdrant_url)
    corpus_ids = await _corpus_document_ids(client)
    await _recreate_collection(client)

    async with AsyncSessionLocal() as session:
        if want:
            stmt = select(Document).where(
                Document.status == "ready", Document.original_filename.in_(want)
            )
        elif corpus_ids:
            stmt = select(Document).where(
                Document.id.in_([uuid.UUID(i) for i in corpus_ids])
            )
        else:
            logger.warning("Empty/absent collection — falling back to all 'ready' docs.")
            stmt = select(Document).where(Document.status == "ready")
        rows = (await session.execute(stmt.order_by(Document.created_at))).scalars().all()

        # Dedupe by original_filename (identical re-uploads exist), keep the first.
        seen: set[str] = set()
        docs = []
        for d in rows:
            if d.original_filename in seen:
                continue
            seen.add(d.original_filename)
            docs.append(d)

        total = 0
        for doc in docs:
            n = await _reingest_document(session, doc)
            total += n
            logger.info("  %-22s → %d chunks", doc.original_filename, n)
        logger.info("Re-ingested %d document(s), %d chunks total.", len(docs), total)

    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
