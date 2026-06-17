"""Learning plans API."""
import uuid
import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.database import get_db
from app.models.plan import LearningPlan, PlanTopic
from app.schemas.plan import (
    GeneratePlanRequest,
    LearningPlanOut,
    LearningPlanDetailOut,
    PlanTopicOut,
    UpdateTopicRequest,
)
from app.services.agents.planner_agent import PlannerState

router = APIRouter()
logger = logging.getLogger(__name__)

VALID_STATUSES = {"not_started", "in_progress", "completed"}


def _topic_to_out(t: PlanTopic) -> PlanTopicOut:
    """Build PlanTopicOut from scalar columns only.

    Avoids ``model_validate(t)``, which would auto-read the lazy ``subtopics``
    relationship and trigger async IO outside a greenlet (MissingGreenlet).
    The tree is assembled manually from ``parent_id`` instead.
    """
    return PlanTopicOut(
        id=t.id,
        plan_id=t.plan_id,
        parent_id=t.parent_id,
        title=t.title,
        description=t.description,
        order_index=t.order_index,
        status=t.status,
        estimated_hours=t.estimated_hours,
        subtopics=[],
    )


def _build_topic_tree(topics: list[PlanTopic]) -> list[PlanTopicOut]:
    """Build nested topic tree from flat list."""
    by_id = {t.id: _topic_to_out(t) for t in topics}
    roots: list[PlanTopicOut] = []
    for topic in topics:
        topic_out = by_id[topic.id]
        if topic.parent_id and topic.parent_id in by_id:
            by_id[topic.parent_id].subtopics.append(topic_out)
        else:
            roots.append(topic_out)
    return roots


def _plan_to_detail(plan: LearningPlan) -> LearningPlanDetailOut:
    """Build the detail response explicitly.

    Avoids ``LearningPlanDetailOut.model_validate(plan)``, which would
    recursively validate ``plan.topics`` into PlanTopicOut and read each
    topic's lazy ``subtopics`` relationship → MissingGreenlet. We read only
    the eagerly-loaded ``plan.topics`` and assemble the tree from scalars.
    """
    return LearningPlanDetailOut(
        id=plan.id,
        title=plan.title,
        description=plan.description,
        goal=plan.goal,
        status=plan.status,
        created_at=plan.created_at,
        updated_at=plan.updated_at,
        topics=_build_topic_tree(plan.topics),
    )


@router.post("/generate", response_model=LearningPlanDetailOut, status_code=201)
async def generate_plan(body: GeneratePlanRequest, request: Request, db: AsyncSession = Depends(get_db)):
    planner_graph = request.app.state.planner_graph
    state = PlannerState(
        learning_goal=body.goal,
        kb_summary="",
        gap_analysis="",
        plan_json={},
        plan_id="",
        retry_count=0,
    )
    try:
        final_state = await planner_graph.ainvoke(state)
    except Exception as e:
        logger.exception("Planner agent failed: %s", e)
        raise HTTPException(500, "Failed to generate learning plan")

    plan_id = uuid.UUID(final_state["plan_id"])
    result = await db.execute(
        select(LearningPlan)
        .where(LearningPlan.id == plan_id)
        .options(selectinload(LearningPlan.topics))
    )
    plan = result.scalar_one_or_none()
    if not plan:
        raise HTTPException(500, "Plan was not persisted")

    return _plan_to_detail(plan)


@router.get("", response_model=list[LearningPlanOut])
async def list_plans(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(LearningPlan).order_by(LearningPlan.created_at.desc()))
    return result.scalars().all()


@router.get("/{plan_id}", response_model=LearningPlanDetailOut)
async def get_plan(plan_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(LearningPlan)
        .where(LearningPlan.id == plan_id)
        .options(selectinload(LearningPlan.topics))
    )
    plan = result.scalar_one_or_none()
    if not plan:
        raise HTTPException(404, "Plan not found")
    return _plan_to_detail(plan)


@router.patch("/{plan_id}/topics/{topic_id}", response_model=PlanTopicOut)
async def update_topic_status(
    plan_id: uuid.UUID,
    topic_id: uuid.UUID,
    body: UpdateTopicRequest,
    db: AsyncSession = Depends(get_db),
):
    if body.status not in VALID_STATUSES:
        raise HTTPException(400, f"Invalid status. Allowed: {', '.join(VALID_STATUSES)}")

    topic = await db.get(PlanTopic, topic_id)
    if not topic or topic.plan_id != plan_id:
        raise HTTPException(404, "Topic not found")

    topic.status = body.status
    await db.commit()
    await db.refresh(topic)

    # Log progress event
    if body.status == "completed":
        from app.models.progress import ProgressEvent
        event = ProgressEvent(
            event_type="topic_completed",
            entity_id=topic_id,
            entity_type="plan_topic",
            event_data={"plan_id": str(plan_id), "title": topic.title},
        )
        db.add(event)
        await db.commit()

    return _topic_to_out(topic)
