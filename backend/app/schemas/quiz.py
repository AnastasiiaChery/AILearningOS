import uuid
from datetime import datetime

from pydantic import BaseModel


class GenerateQuizRequest(BaseModel):
    document_id: uuid.UUID | None = None
    topic_id: uuid.UUID | None = None
    question_count: int = 5


class QuizQuestionOut(BaseModel):
    id: uuid.UUID
    question_text: str
    question_type: str
    options: dict | None
    order_index: int

    model_config = {"from_attributes": True}


class QuizOut(BaseModel):
    id: uuid.UUID
    title: str
    topic_id: uuid.UUID | None
    document_id: uuid.UUID | None
    created_at: datetime

    model_config = {"from_attributes": True}


class QuizDetailOut(QuizOut):
    questions: list[QuizQuestionOut] = []


class SubmitAttemptRequest(BaseModel):
    answers: list[dict]  # [{question_id, answer}]


class AttemptResultOut(BaseModel):
    id: uuid.UUID
    score: float
    answers: list[dict]
    completed_at: datetime | None

    model_config = {"from_attributes": True}
