"""Learning plans API."""
import json
import uuid
import logging
from collections.abc import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, AIMessage
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
    LessonRequest,
    TopicLessonOut,
    LessonCitation,
    GradeExerciseRequest,
    GradeExerciseOut,
    TopicChatRequest,
)
from app.services.agents.mentor_agent import MentorState
from app.services.agents.planner_agent import PlannerState
from app.services.grader import grade_open_answer
from app.services.rag.retriever import hybrid_search
from app.services.tutor import generate_lesson

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
        document_ids=[str(d) for d in (body.document_ids or [])],
        kb_summary="",
        kb_context="",
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


async def _topic_and_plan(db: AsyncSession, plan_id: uuid.UUID, topic_id: uuid.UUID):
    topic = await db.get(PlanTopic, topic_id)
    if not topic or topic.plan_id != plan_id:
        raise HTTPException(404, "Topic not found")
    plan = await db.get(LearningPlan, plan_id)
    return plan, topic


def _topic_query(topic: PlanTopic) -> str:
    return f"{topic.title}. {topic.description}" if topic.description else topic.title


async def _retrieve_for_topic(plan: LearningPlan | None, topic: PlanTopic, top_k: int = 5) -> list[dict]:
    """Relevant chunks for a topic, scoped to the documents the plan was built on.

    rerank is skipped here: a lesson only needs a handful of on-topic chunks, and
    the cross-encoder pass is the slowest part of retrieval — dropping it shaves
    latency with no real quality loss for grounding.
    """
    doc_ids = None
    if plan is not None and isinstance(plan.raw_analysis, dict):
        doc_ids = plan.raw_analysis.get("document_ids") or None
    return await hybrid_search(_topic_query(topic), top_k=top_k, rerank=False, document_ids=doc_ids)


def _format_chunks(chunks: list[dict]) -> tuple[str, list[LessonCitation]]:
    parts: list[str] = []
    citations: list[LessonCitation] = []
    for i, c in enumerate(chunks, start=1):
        heading = " > ".join(
            s for s in [c.get("h1_title"), c.get("h2_title"), c.get("h3_title")] if s
        )
        src = c.get("filename") or "document"
        header = f"[{i}] {src}" + (f" — {heading}" if heading else "")
        parts.append(f"{header}\n{c.get('content', '')}")
        if c.get("chunk_id"):
            citations.append(LessonCitation(chunk_id=c["chunk_id"], filename=src, heading=heading or None))
    return "\n\n---\n\n".join(parts), citations


@router.post("/{plan_id}/topics/{topic_id}/lesson", response_model=TopicLessonOut)
async def topic_lesson(
    plan_id: uuid.UUID,
    topic_id: uuid.UUID,
    body: LessonRequest | None = None,
    db: AsyncSession = Depends(get_db),
):
    """Generate (or return cached) a grounded lesson + practice exercise for a topic."""
    req = body or LessonRequest()
    plan, topic = await _topic_and_plan(db, plan_id, topic_id)

    # Cache hit: same language, not forced to regenerate → return instantly.
    cache = topic.lesson_cache if isinstance(topic.lesson_cache, dict) else None
    if cache and not req.regenerate and cache.get("language") == req.language:
        return TopicLessonOut(
            topic_id=topic.id,
            topic_title=topic.title,
            lesson=cache.get("lesson", ""),
            exercise=cache.get("exercise", ""),
            language=req.language,
            citations=[LessonCitation(**c) for c in cache.get("citations", [])],
        )

    chunks = await _retrieve_for_topic(plan, topic)
    context, citations = _format_chunks(chunks)
    try:
        lesson = await generate_lesson(topic.title, topic.description, context, req.language)
    except Exception as e:
        logger.exception("Lesson generation failed: %s", e)
        raise HTTPException(500, "Failed to generate lesson")

    # Surface only the citations the lesson actually references (fall back to all).
    referenced = [c for i, c in enumerate(citations, start=1) if f"[{i}]" in lesson.lesson] or citations

    # Cache for instant reopen (one lesson per topic; keyed by language).
    topic.lesson_cache = {
        "language": req.language,
        "lesson": lesson.lesson,
        "exercise": lesson.exercise,
        "citations": [c.model_dump() for c in referenced],
    }
    await db.commit()

    return TopicLessonOut(
        topic_id=topic.id,
        topic_title=topic.title,
        lesson=lesson.lesson,
        exercise=lesson.exercise,
        language=req.language,
        citations=referenced,
    )


@router.post("/{plan_id}/topics/{topic_id}/grade", response_model=GradeExerciseOut)
async def grade_topic_exercise(
    plan_id: uuid.UUID,
    topic_id: uuid.UUID,
    body: GradeExerciseRequest,
    db: AsyncSession = Depends(get_db),
):
    """Grade a free-text answer to a topic's practice exercise (LLM judge)."""
    plan, topic = await _topic_and_plan(db, plan_id, topic_id)
    chunks = await _retrieve_for_topic(plan, topic)
    context, _ = _format_chunks(chunks)
    verdict = await grade_open_answer(body.exercise, context, body.answer, body.language)
    return GradeExerciseOut(verdict=verdict.verdict, explanation=verdict.explanation)


@router.post("/{plan_id}/topics/{topic_id}/chat")
async def topic_chat(
    plan_id: uuid.UUID,
    topic_id: uuid.UUID,
    body: TopicChatRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Streaming mentor chat anchored to one topic.

    Reuses the mentor agent but scoped: retrieval is restricted to the plan's
    documents and biased toward the topic, and the cached lesson (if any) is
    seeded into context. History is supplied by the client per request — this
    chat is ephemeral and not persisted as a session.
    """
    plan, topic = await _topic_and_plan(db, plan_id, topic_id)

    doc_ids = None
    if plan is not None and isinstance(plan.raw_analysis, dict):
        doc_ids = plan.raw_analysis.get("document_ids") or None
    lesson_context = ""
    if isinstance(topic.lesson_cache, dict):
        lesson_context = topic.lesson_cache.get("lesson") or ""

    history = []
    for m in body.history[-10:]:
        content = m.get("content", "")
        if m.get("role") == "user":
            history.append(HumanMessage(content=content))
        else:
            history.append(AIMessage(content=content))

    mentor_graph = request.app.state.mentor_graph

    async def stream_response() -> AsyncGenerator[str, None]:
        state: MentorState = {
            "session_id": str(topic_id),
            "user_message": body.message,
            "history": history,
            "search_query": "",
            "retrieved_chunks": [],
            "response_text": "",
            "cited_chunk_ids": [],
            "document_ids": doc_ids or [],
            "topic_hint": topic.title,
            "lesson_context": lesson_context,
        }
        final_state: dict = {}
        response_text = ""
        try:
            async for mode, chunk in mentor_graph.astream(state, stream_mode=["messages", "values"]):
                if mode == "messages":
                    msg, meta = chunk
                    if meta.get("langgraph_node") == "generate" and msg.content:
                        response_text += msg.content
                        yield f"data: {json.dumps({'type': 'token', 'text': msg.content})}\n\n"
                elif mode == "values":
                    final_state = chunk
        except Exception as e:
            logger.exception("Topic chat error: %s", e)
            yield f"data: {json.dumps({'type': 'error', 'text': 'Failed to generate response.'})}\n\n"
            return

        cited_ids = final_state.get("cited_chunk_ids", [])
        chunks = final_state.get("retrieved_chunks", [])
        citations = []
        seen = set()
        for c in chunks:
            cid = c.get("chunk_id")
            if cid and cid in cited_ids and cid not in seen:
                seen.add(cid)
                heading = " > ".join(
                    s for s in [c.get("h1_title"), c.get("h2_title"), c.get("h3_title")] if s
                )
                citations.append({"chunk_id": cid, "filename": c.get("filename", ""), "heading": heading or None})

        yield f"data: {json.dumps({'type': 'done', 'citations': citations})}\n\n"

    return StreamingResponse(stream_response(), media_type="text/event-stream")


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
