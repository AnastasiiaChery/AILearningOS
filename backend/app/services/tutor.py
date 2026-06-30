"""Per-topic tutor: turn a plan topic + retrieved material into a short lesson
and a practice exercise.

Grounded in the user's knowledge base — the lesson must teach only from the
retrieved chunks and cite them [1], [2]; the exercise must be answerable from
what the lesson taught. Reuses the retrieval + structured-output patterns used
elsewhere (planner, grader).
"""
import logging

from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from app.services.agents.llm_factory import get_llm

logger = logging.getLogger(__name__)


class TopicLesson(BaseModel):
    """Schema the LLM is forced to fill via with_structured_output."""

    lesson: str       # markdown; cites the numbered context chunks as [1], [2]
    exercise: str     # one open-ended practice question grounded in the lesson


# How to phrase the language requirement. "auto" keeps the source language so a
# Ukrainian document is taught in Ukrainian and never silently translated.
LANG_INSTRUCTION = {
    "auto": "Write the lesson and exercise in the SAME language as the source context above. Do NOT translate the material into another language.",
    "uk": "Write the lesson and exercise in Ukrainian (українською мовою).",
    "en": "Write the lesson and exercise in English.",
    "ru": "Write the lesson and exercise in Russian (на русском языке).",
}


LESSON_PROMPT = """You are an expert tutor. Teach the topic below using ONLY the
numbered context from the learner's knowledge base.

Topic: {topic}
{description}

Context (numbered source chunks):
{context}

Language: {language_instruction}

Write two things:

1. lesson — a concise lesson in Markdown (about 200-350 words) that teaches this
   topic. Use short paragraphs, bold for key terms, and bullet lists where useful.
   Ground every claim in the context and cite sources inline as [1], [2] matching
   the chunk numbers. If the context is thin, teach what's there and say plainly
   what isn't covered — do not invent facts.

2. exercise — ONE open-ended practice question that checks whether the learner
   understood the lesson. It must be answerable in a few sentences from what you
   just taught (not a yes/no question, not requiring outside knowledge)."""


async def generate_lesson(
    topic_title: str, description: str | None, context: str, language: str = "auto"
) -> TopicLesson:
    """Produce a grounded lesson + practice exercise for one topic.

    Falls back to a minimal lesson (no exercise grading value lost — the caller
    still gets a usable question) if structured generation fails.
    """
    prompt = LESSON_PROMPT.format(
        topic=topic_title,
        description=f"What it covers: {description}" if description else "",
        context=context or "No relevant material found in the knowledge base.",
        language_instruction=LANG_INSTRUCTION.get(language, LANG_INSTRUCTION["auto"]),
    )
    try:
        llm = get_llm(temperature=0).with_structured_output(TopicLesson)
        return await llm.ainvoke([HumanMessage(content=prompt)])
    except Exception as e:
        logger.warning("Lesson generation failed (%s); using fallback.", e)
        return TopicLesson(
            lesson=(
                f"### {topic_title}\n\n"
                "Couldn't generate a lesson right now. Try the chat mentor or a "
                "quiz on this topic instead."
            ),
            exercise=f"In your own words, explain the key idea behind: {topic_title}.",
        )
