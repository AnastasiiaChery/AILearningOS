"""Planner LangGraph agent — generates personalized learning plans."""
import logging
import uuid
from typing import TypedDict

from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field
from qdrant_client.models import Filter, FieldCondition, MatchAny

from app.services.agents.llm_factory import get_llm
from app.services.rag.qdrant_store import get_qdrant_client
from app.services.rag.retriever import hybrid_search
from app.core.config import settings

logger = logging.getLogger(__name__)


class SubtopicModel(BaseModel):
    title: str
    description: str = ""
    estimated_hours: float = Field(default=0.5)


class TopicModel(BaseModel):
    title: str
    description: str = ""
    estimated_hours: float = Field(default=1.0)
    subtopics: list[SubtopicModel] = Field(default_factory=list)


class PlanModel(BaseModel):
    """Schema the LLM is forced to fill via with_structured_output."""

    title: str
    description: str = ""
    topics: list[TopicModel] = Field(default_factory=list)

ANALYZE_PROMPT = """You are an expert learning path designer. Analyze the user's
study material and their learning goal.

User's learning goal: {goal}

Their study material (outline of available documents, then the most relevant
excerpts for the goal):
{kb_context}

Identify:
1. What the material already covers toward the goal (cite concrete subjects seen above)
2. Key gaps the material does NOT cover but the goal needs
3. A recommended learning sequence

Be concise and specific. Ground every point in the material above where possible."""

PLAN_PROMPT = """Create a structured learning plan: a concise title, a short
description, and an ordered list of topics (each with subtopics and estimated hours).

Goal: {goal}

Gap analysis: {gap_analysis}

Study material to build the plan around:
{kb_context}

Rules:
- Prefer topics and subtopics that are actually grounded in the study material
  above — use its real subjects, terminology and structure, not generic filler.
- Order topics so material the user already has comes first (review), then the
  gaps. For a gap with no supporting material, say so in that topic's description
  (e.g. "not in your files — needs an external resource").
- Keep it realistic: 3-7 top-level topics."""


class PlannerState(TypedDict):
    learning_goal: str
    document_ids: list[str]
    kb_summary: str
    kb_context: str
    gap_analysis: str
    plan_json: dict
    plan_id: str
    retry_count: int


def _doc_scroll_filter(document_ids: list[str] | None) -> Filter | None:
    if not document_ids:
        return None
    return Filter(
        must=[FieldCondition(key="document_id", match=MatchAny(any=[str(d) for d in document_ids]))]
    )


async def summarize_knowledge(state: PlannerState) -> dict:
    """Ground the plan in the actual study material.

    Two complementary views, both scoped to ``document_ids`` when the user picked
    specific files (otherwise the whole KB):
    - an **outline** (per-file heading paths) from a Qdrant scroll → breadth of
      what's available;
    - the **most relevant excerpts** for the goal via hybrid search → real content
      the plan can be built on, not just titles.
    """
    document_ids = state.get("document_ids") or []
    client = get_qdrant_client()

    # --- Outline: filename → distinct heading paths ---
    outline = "No documents in knowledge base yet."
    try:
        scroll_result, _ = await client.scroll(
            collection_name=settings.qdrant_collection,
            scroll_filter=_doc_scroll_filter(document_ids),
            limit=400,
            with_payload=["filename", "h1_title", "h2_title", "h3_title"],
        )
        by_file: dict[str, set[str]] = {}
        for point in scroll_result:
            p = point.payload or {}
            fname = str(p.get("filename") or "document").strip()
            path = " > ".join(
                str(p[k]).strip() for k in ("h1_title", "h2_title", "h3_title") if p.get(k)
            )
            if path:
                by_file.setdefault(fname, set()).add(path)
        if by_file:
            outline = "\n".join(
                f"### {fname}\n" + "\n".join(f"- {h}" for h in sorted(paths))
                for fname, paths in sorted(by_file.items())
            )
    except Exception as e:
        logger.warning("Failed to build KB outline: %s", e)
        outline = "Could not access knowledge base."

    # --- Relevant excerpts for the goal (grounding) ---
    excerpts = ""
    try:
        hits = await hybrid_search(
            state["learning_goal"], top_k=10, document_ids=document_ids or None
        )
        parts = []
        for i, h in enumerate(hits, 1):
            heading = " > ".join(
                str(h[k]).strip() for k in ("h1_title", "h2_title", "h3_title") if h.get(k)
            )
            src = h.get("filename") or "document"
            label = f"{src} — {heading}" if heading else src
            body = (h.get("content") or "").strip().replace("\n", " ")[:500]
            parts.append(f"[{i}] ({label}) {body}")
        excerpts = "\n\n".join(parts)
    except Exception as e:
        logger.warning("Failed to retrieve grounding excerpts: %s", e)

    kb_context = f"## Document outline\n{outline}"
    if excerpts:
        kb_context += f"\n\n## Most relevant excerpts for the goal\n{excerpts}"

    return {"kb_summary": outline, "kb_context": kb_context}


async def analyze_gaps(state: PlannerState) -> dict:
    llm = get_llm()
    prompt = ANALYZE_PROMPT.format(kb_context=state["kb_context"], goal=state["learning_goal"])
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    return {"gap_analysis": response.content}


async def generate_plan(state: PlannerState) -> dict:
    """Produce the plan via structured output — no manual JSON parsing/retries.

    with_structured_output binds PlanModel as a tool/JSON schema, so the model is
    constrained to return a valid object (validated by Pydantic). On failure we
    fall back to a minimal one-topic plan instead of erroring.
    """
    prompt = PLAN_PROMPT.format(
        goal=state["learning_goal"],
        gap_analysis=state["gap_analysis"],
        kb_context=state["kb_context"],
    )
    try:
        llm = get_llm().with_structured_output(PlanModel)
        plan: PlanModel = await llm.ainvoke([HumanMessage(content=prompt)])
        return {"plan_json": plan.model_dump()}
    except Exception as e:
        logger.warning("Structured plan generation failed (%s); using fallback.", e)
        fallback = PlanModel(
            title=f"Learning Plan: {state['learning_goal'][:50]}",
            description="Auto-generated learning plan.",
            topics=[
                TopicModel(
                    title=state["learning_goal"],
                    description=state["gap_analysis"][:200],
                    estimated_hours=5.0,
                )
            ],
        )
        return {"plan_json": fallback.model_dump()}


async def persist_plan(state: PlannerState) -> dict:
    """Save plan to PostgreSQL."""
    from app.core.database import AsyncSessionLocal
    from app.models.plan import LearningPlan, PlanTopic

    plan_data = state["plan_json"]
    plan_id = str(uuid.uuid4())

    async with AsyncSessionLocal() as session:
        plan = LearningPlan(
            id=uuid.UUID(plan_id),
            title=plan_data.get("title", "Learning Plan"),
            description=plan_data.get("description"),
            goal=state["learning_goal"],
            raw_analysis={
                "gap_analysis": state["gap_analysis"],
                "kb_summary": state["kb_summary"],
                "document_ids": state.get("document_ids") or [],
            },
        )
        session.add(plan)

        for order, topic_data in enumerate(plan_data.get("topics", [])):
            topic = PlanTopic(
                plan_id=uuid.UUID(plan_id),
                title=topic_data["title"],
                description=topic_data.get("description"),
                order_index=order,
                estimated_hours=topic_data.get("estimated_hours"),
            )
            session.add(topic)
            await session.flush()

            for sub_order, sub_data in enumerate(topic_data.get("subtopics", [])):
                subtopic = PlanTopic(
                    plan_id=uuid.UUID(plan_id),
                    parent_id=topic.id,
                    title=sub_data["title"],
                    description=sub_data.get("description"),
                    order_index=sub_order,
                    estimated_hours=sub_data.get("estimated_hours"),
                )
                session.add(subtopic)

        await session.commit()

    return {"plan_id": plan_id}


def build_planner_graph():
    graph = StateGraph(PlannerState)
    graph.add_node("summarize", summarize_knowledge)
    graph.add_node("analyze", analyze_gaps)
    graph.add_node("generate", generate_plan)
    graph.add_node("persist", persist_plan)

    graph.set_entry_point("summarize")
    graph.add_edge("summarize", "analyze")
    graph.add_edge("analyze", "generate")
    graph.add_edge("generate", "persist")
    graph.add_edge("persist", END)
    return graph.compile()
