"""Qdrant vector store operations."""
import uuid
import logging
from collections.abc import Sequence

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    PointStruct,
    SparseVector,
    Filter,
    FieldCondition,
    MatchValue,
)

from app.core.config import settings
from app.services.ingestion.chunker import Chunk

logger = logging.getLogger(__name__)

_client: AsyncQdrantClient | None = None


def get_qdrant_client() -> AsyncQdrantClient:
    global _client
    if _client is None:
        _client = AsyncQdrantClient(url=settings.qdrant_url)
    return _client


import hashlib
import re
from collections import Counter

# Hashing-trick vocabulary size. 2^20 ≈ 1M columns → token collisions are rare.
_SPARSE_DIM = 1 << 20
# BM25 term-frequency saturation constant. We use the BM25 TF component with
# b=0 (no document-length normalization), because length norm needs corpus-wide
# avgdl which we don't track here. Saturation alone already curbs keyword spam.
_BM25_K1 = 1.5
_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _token_id(token: str) -> int:
    """Deterministic token → column id.

    Uses blake2b (stable across processes) instead of Python's built-in
    ``hash()``, whose string salt is randomized per process (PYTHONHASHSEED).
    With the old hash() a word indexed in one process landed in a DIFFERENT
    column after a restart, so stored chunks and queries stopped matching.
    """
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % _SPARSE_DIM


def _tf_weight(count: int) -> float:
    """BM25 TF saturation (b=0): count*(k1+1)/(count+k1). The 5th mention adds
    far less than the 2nd. IDF is NOT applied here — Qdrant's Modifier.IDF does
    it at query time using real corpus document-frequency."""
    return count * (_BM25_K1 + 1) / (count + _BM25_K1)


def _build_sparse_vector(text: str) -> SparseVector:
    """Build a TF-only sparse vector (Qdrant applies IDF via Modifier.IDF)."""
    counts = Counter(_tokenize(text))
    # Aggregate by column id so hash collisions never produce duplicate indices
    # (Qdrant requires unique indices within one sparse vector).
    agg: dict[int, float] = {}
    for token, count in counts.items():
        agg[_token_id(token)] = agg.get(_token_id(token), 0.0) + _tf_weight(count)
    indices = list(agg.keys())
    values = [float(v) for v in agg.values()]
    return SparseVector(indices=indices, values=values)


async def upsert_chunks(
    chunks: list[Chunk],
    document_id: str,
    filename: str,
    file_type: str,
    chunk_ids: list[uuid.UUID],
) -> None:
    from app.services.rag.embedder import aembed_texts

    client = get_qdrant_client()
    texts = [c.content for c in chunks]
    dense_vecs = await aembed_texts(texts)

    points = []
    for chunk, dense, chunk_id in zip(chunks, dense_vecs, chunk_ids):
        sparse = _build_sparse_vector(chunk.content)
        points.append(
            PointStruct(
                id=str(chunk_id),
                vector={"dense": dense, "sparse": sparse},
                payload={
                    "chunk_id": str(chunk_id),
                    "document_id": document_id,
                    "filename": filename,
                    "file_type": file_type,
                    "content": chunk.content,
                    "h1_title": chunk.h1_title,
                    "h2_title": chunk.h2_title,
                    "h3_title": chunk.h3_title,
                    "page_number": chunk.page_number,
                    "chunk_index": chunk.chunk_index,
                    "token_count": chunk.token_count,
                },
            )
        )

    await client.upsert(collection_name=settings.qdrant_collection, points=points)
    logger.info("Upserted %d points to Qdrant.", len(points))


async def delete_document_chunks(document_id: str) -> None:
    client = get_qdrant_client()
    await client.delete(
        collection_name=settings.qdrant_collection,
        points_selector=Filter(
            must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
        ),
    )
    logger.info("Deleted Qdrant points for document %s", document_id)
