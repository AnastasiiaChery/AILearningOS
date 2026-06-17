"""Progress tracking API."""
from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.document import Document, DocumentChunk
from app.models.plan import LearningPlan, PlanTopic
from app.models.progress import ProgressEvent
from app.models.quiz import QuizAttempt
from app.schemas.progress import ProgressEventOut, ProgressSummary

router = APIRouter()


@router.get("/summary", response_model=ProgressSummary)
async def get_summary(db: AsyncSession = Depends(get_db)):
    total_docs = (await db.execute(select(func.count()).select_from(Document))).scalar_one()
    total_chunks = (await db.execute(select(func.count()).select_from(DocumentChunk))).scalar_one()
    total_plans = (await db.execute(select(func.count()).select_from(LearningPlan))).scalar_one()
    total_topics = (await db.execute(select(func.count()).select_from(PlanTopic))).scalar_one()
    completed_topics = (
        await db.execute(
            select(func.count()).select_from(PlanTopic).where(PlanTopic.status == "completed")
        )
    ).scalar_one()
    total_attempts = (await db.execute(select(func.count()).select_from(QuizAttempt))).scalar_one()
    avg_score_row = (await db.execute(select(func.avg(QuizAttempt.score)).select_from(QuizAttempt))).scalar_one()

    return ProgressSummary(
        total_documents=total_docs,
        total_chunks=total_chunks,
        total_plans=total_plans,
        completed_topics=completed_topics,
        total_topics=total_topics,
        total_quiz_attempts=total_attempts,
        avg_quiz_score=float(avg_score_row) if avg_score_row is not None else None,
    )


@router.get("/events", response_model=list[ProgressEventOut])
async def list_events(limit: int = 50, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(ProgressEvent).order_by(ProgressEvent.created_at.desc()).limit(limit)
    )
    return result.scalars().all()
