import uuid
from datetime import datetime

from pydantic import BaseModel


class GeneratePlanRequest(BaseModel):
    goal: str


class UpdateTopicRequest(BaseModel):
    status: str  # not_started | in_progress | completed


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
