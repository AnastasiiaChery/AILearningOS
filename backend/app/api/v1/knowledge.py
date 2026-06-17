"""Knowledge base management — upload, list, delete documents."""
import os
import uuid
import logging
from pathlib import Path

import aiofiles
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.config import settings
from app.core.database import get_db
from app.models.document import Document, DocumentChunk
from app.schemas.document import DocumentOut, DocumentChunkOut
from app.services.rag.qdrant_store import delete_document_chunks

router = APIRouter()
logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {".md", ".markdown", ".pdf"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB


async def _process_document(document_id: str, file_path: str, file_type: str) -> None:
    """Background task: parse, chunk, embed and store document."""
    from app.core.database import AsyncSessionLocal
    from app.models.document import Document, DocumentChunk
    from app.services.rag.qdrant_store import upsert_chunks

    async with AsyncSessionLocal() as session:
        doc = await session.get(Document, uuid.UUID(document_id))
        if not doc:
            return

        try:
            doc.status = "processing"
            await session.commit()

            if file_type == "markdown":
                from app.services.ingestion.markdown_loader import load_markdown
                chunks = load_markdown(file_path)
            else:
                from app.services.ingestion.pdf_loader import load_pdf
                chunks = load_pdf(file_path)

            chunk_ids = [uuid.uuid4() for _ in chunks]

            # Persist chunks to PostgreSQL
            db_chunks = [
                DocumentChunk(
                    id=chunk_ids[i],
                    document_id=uuid.UUID(document_id),
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
            session.add_all(db_chunks)

            # Upsert to Qdrant
            await upsert_chunks(chunks, document_id, doc.original_filename, file_type, chunk_ids)

            doc.status = "ready"
            doc.chunk_count = len(chunks)
            await session.commit()
            logger.info("Document %s processed: %d chunks.", document_id, len(chunks))

        except Exception as e:
            logger.exception("Error processing document %s: %s", document_id, e)
            doc.status = "error"
            doc.error_msg = str(e)
            await session.commit()


@router.post("/upload", response_model=DocumentOut, status_code=202)
async def upload_document(
    file: UploadFile,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type: {suffix}. Allowed: {', '.join(ALLOWED_EXTENSIONS)}")

    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(400, "File too large. Max 50 MB.")

    file_type = "markdown" if suffix in {".md", ".markdown"} else "pdf"
    doc_id = uuid.uuid4()
    safe_name = f"{doc_id}{suffix}"
    upload_path = os.path.join(settings.upload_dir, safe_name)

    os.makedirs(settings.upload_dir, exist_ok=True)
    async with aiofiles.open(upload_path, "wb") as f:
        await f.write(content)

    doc = Document(
        id=doc_id,
        filename=safe_name,
        original_filename=file.filename or safe_name,
        file_type=file_type,
        file_size=len(content),
        status="pending",
    )
    db.add(doc)
    await db.commit()
    await db.refresh(doc)

    background_tasks.add_task(_process_document, str(doc_id), upload_path, file_type)
    return doc


@router.get("/documents", response_model=list[DocumentOut])
async def list_documents(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Document).order_by(Document.created_at.desc()))
    return result.scalars().all()


@router.get("/documents/{doc_id}", response_model=DocumentOut)
async def get_document(doc_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    doc = await db.get(Document, doc_id)
    if not doc:
        raise HTTPException(404, "Document not found")
    return doc


@router.get("/documents/{doc_id}/chunks", response_model=list[DocumentChunkOut])
async def get_document_chunks(doc_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    doc = await db.get(Document, doc_id)
    if not doc:
        raise HTTPException(404, "Document not found")
    result = await db.execute(
        select(DocumentChunk)
        .where(DocumentChunk.document_id == doc_id)
        .order_by(DocumentChunk.chunk_index)
    )
    return result.scalars().all()


@router.delete("/documents/{doc_id}", status_code=204)
async def delete_document(doc_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    doc = await db.get(Document, doc_id)
    if not doc:
        raise HTTPException(404, "Document not found")
    await delete_document_chunks(str(doc_id))
    # Remove uploaded file
    file_path = os.path.join(settings.upload_dir, doc.filename)
    if os.path.exists(file_path):
        os.remove(file_path)
    await db.delete(doc)
    await db.commit()
