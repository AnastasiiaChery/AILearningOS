"""Planner LangGraph agent — generates personalized learning plans."""
import logging
import uuid
from typing import TypedDict

from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from app.services.agents.llm_factory import get_llm
from app.services.rag.qdrant_store import get_qdrant_client
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

ANALYZE_PROMPT = """You are an expert learning path designer. Analyze the user's existing knowledge base and their learning goal.

Knowledge base summary (topics found):
{kb_summary}

User's learning goal: {goal}

Identify:
1. What the user already knows (from their knowledge base)
2. Key gaps for achieving the goal
3. Recommended learning sequence

Be concise and specific."""

PLAN_PROMPT = """Based on this gap analysis, create a structured learning plan with
a concise title, a short description, and a list of topics (each with subtopics and
estimated hours).

Goal: {goal}
Gap Analysis: {gap_analysis}"""


class PlannerState(TypedDict):
    learning_goal: str
    kb_summary: str
    gap_analysis: str
    plan_json: dict
    plan_id: str
    retry_count: int


async def summarize_knowledge(state: PlannerState) -> dict:
    """Extract topic summary from Qdrant metadata."""
    client = get_qdrant_client()
    try:
        scroll_result, _ = await client.scroll(
            collection_name=settings.qdrant_collection,
            limit=200,
            with_payload=["filename", "h1_title", "h2_title"],
        )
        topics: set[str] = set()
        for point in scroll_result:
            p = point.payload or {}
            for key in ("h1_title", "h2_title", "filename"):
                val = p.get(key)
                if val:
                    topics.add(str(val).strip())
        summary = "\n".join(f"- {t}" for t in sorted(topics)) or "No documents in knowledge base yet."
    except Exception as e:
        logger.warning("Failed to summarize KB: %s", e)
        summary = "Could not access knowledge base."
    return {"kb_summary": summary}


async def analyze_gaps(state: PlannerState) -> dict:
    llm = get_llm()
    prompt = ANALYZE_PROMPT.format(kb_summary=state["kb_summary"], goal=state["learning_goal"])
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    return {"gap_analysis": response.content}


async def generate_plan(state: PlannerState) -> dict:
    """Produce the plan via structured output — no manual JSON parsing/retries.

    with_structured_output binds PlanModel as a tool/JSON schema, so the model is
    constrained to return a valid object (validated by Pydantic). On failure we
    fall back to a minimal one-topic plan instead of erroring.
    """
    prompt = PLAN_PROMPT.format(goal=state["learning_goal"], gap_analysis=state["gap_analysis"])
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
            raw_analysis={"gap_analysis": state["gap_analysis"], "kb_summary": state["kb_summary"]},
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
