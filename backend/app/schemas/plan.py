import uuid
from datetime import datetime

from pydantic import BaseModel


class GeneratePlanRequest(BaseModel):
    goal: str
    # Optional: scope the plan to specific documents. Empty/None → whole KB.
    document_ids: list[uuid.UUID] | None = None


class UpdateTopicRequest(BaseModel):
    status: str  # not_started | in_progress | completed


class LessonCitation(BaseModel):
    chunk_id: str
    filename: str
    heading: str | None = None


class LessonRequest(BaseModel):
    language: str = "auto"     # auto | uk | en | ru
    regenerate: bool = False   # force a fresh lesson, ignoring the cache


class TopicLessonOut(BaseModel):
    topic_id: uuid.UUID
    topic_title: str
    lesson: str
    exercise: str
    language: str = "auto"
    citations: list[LessonCitation] = []


class GradeExerciseRequest(BaseModel):
    exercise: str
    answer: str
    language: str = "auto"


class GradeExerciseOut(BaseModel):
    verdict: str  # correct | partial | incorrect
    explanation: str


class TopicChatRequest(BaseModel):
    message: str
    history: list[dict] = []  # [{role: "user"|"assistant", content: str}]


class PlanTopicOut(BaseModel):
    id: uuid.UUID
    plan_id: uuid.UUID
    parent_id: uuid.UUID | None
    title: str
    description: str | None
    order_index: int
    status: str
    estimated_hours: float | None
    subtopics: list["PlanTopicOut"] = []

    model_config = {"from_attributes": True}


class LearningPlanOut(BaseModel):
    id: uuid.UUID
    title: str
    description: str | None
    goal: str
    status: str
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class LearningPlanDetailOut(LearningPlanOut):
    topics: list[PlanTopicOut] = []
