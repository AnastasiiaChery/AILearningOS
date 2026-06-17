import uuid
from datetime import datetime

from pydantic import BaseModel


class ProgressSummary(BaseModel):
    total_documents: int
    total_chunks: int
    total_plans: int
    completed_topics: int
    total_topics: int
    total_quiz_attempts: int
    avg_quiz_score: float | None


class ProgressEventOut(BaseModel):
    id: uuid.UUID
    event_type: str
    entity_id: uuid.UUID
    entity_type: str
    event_data: dict | None
    created_at: datetime

    model_config = {"from_attributes": True}
