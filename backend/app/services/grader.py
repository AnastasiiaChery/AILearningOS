"""LLM-based grading for short-answer quiz questions.

multiple_choice / true_false are graded by exact string match in the API — the
correct answer is a single option key, so `==` is correct there. short_answer is
free text, where `==` against a terse reference answer is almost always a
false-negative (e.g. "It is compiled into bytecode" vs reference "Bytecode").

For short_answer we use an LLM judge constrained to a strict verdict schema via
with_structured_output. The judge is explicitly told to require *semantic
equivalence* (meaning, not wording) and to be strict, because LLM judges skew
lenient and would otherwise reward merely-topical answers.

On any LLM failure we degrade gracefully to the old exact-match behaviour so
grading never hard-fails an attempt submission.
"""
import logging
from typing import Literal

from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from app.services.agents.llm_factory import get_llm

logger = logging.getLogger(__name__)

# Score weight for a "partial" short-answer verdict.
PARTIAL_CREDIT = 0.5


class ShortAnswerVerdict(BaseModel):
    """Schema the judge LLM is forced to fill via with_structured_output."""

    verdict: Literal["correct", "partial", "incorrect"]
    explanation: str = ""


GRADER_PROMPT = """You are a strict but fair grader for one short-answer quiz question.

Question: {question}
Reference (model) answer: {reference}
Student's answer: {given}

Decide whether the student's answer is semantically equivalent to the reference
answer. Judge meaning, not wording — synonyms, paraphrases, and extra correct
detail are fine. Be strict: do NOT reward an answer that is only topically
related, vague, or that misses the key point the reference is testing.

Choose one verdict:
- "correct": conveys the same essential meaning as the reference answer.
- "partial": captures part of the key point but is incomplete or has a minor
  inaccuracy.
- "incorrect": wrong, off-topic, empty, or misses the key point.

Give a one-sentence justification of your verdict."""


async def grade_short_answer(
    question_text: str, reference: str, given: str
) -> ShortAnswerVerdict:
    """Grade a free-text answer with an LLM judge.

    Returns a ShortAnswerVerdict. Empty answers short-circuit to "incorrect"
    without an LLM call; LLM errors fall back to exact-match grading.
    """
    given = (given or "").strip()
    if not given:
        return ShortAnswerVerdict(verdict="incorrect", explanation="No answer provided.")

    prompt = GRADER_PROMPT.format(
        question=question_text, reference=reference, given=given
    )
    try:
        # temperature=0 → stable, repeatable verdicts.
        llm = get_llm(temperature=0).with_structured_output(ShortAnswerVerdict)
        return await llm.ainvoke([HumanMessage(content=prompt)])
    except Exception as e:
        logger.warning(
            "Short-answer grading failed (%s); falling back to exact match.", e
        )
        is_match = given.upper() == (reference or "").strip().upper()
        return ShortAnswerVerdict(
            verdict="correct" if is_match else "incorrect",
            explanation="Automated fallback grading (LLM judge unavailable).",
        )


def points_for_verdict(verdict: str) -> float:
    """Map a verdict to a score weight in [0, 1]."""
    return {"correct": 1.0, "partial": PARTIAL_CREDIT, "incorrect": 0.0}.get(verdict, 0.0)


OPEN_GRADER_PROMPT = """You are grading a learner's free-text answer to a practice
question from a lesson. There is no single canonical answer — judge the answer
against the reference material the question was based on.

Question: {question}

Reference material (what the lesson taught):
{context}

Learner's answer: {given}

Judge meaning, not wording. Be strict: do NOT reward vague, merely-topical, or
hand-wavy answers. Choose a verdict:
- "correct": demonstrates real understanding consistent with the material.
- "partial": on the right track but incomplete or with a minor error.
- "incorrect": wrong, off-topic, empty, or misses the key point.

Then give a short, encouraging explanation (2-3 sentences): what was right, what
was missing, and the key point to remember.

{language_instruction}"""

_GRADE_LANG = {
    "auto": "Write the explanation in the same language as the question and reference material.",
    "uk": "Write the explanation in Ukrainian (українською мовою).",
    "en": "Write the explanation in English.",
    "ru": "Write the explanation in Russian (на русском языке).",
}


async def grade_open_answer(
    question: str, context: str, given: str, language: str = "auto"
) -> ShortAnswerVerdict:
    """Grade a free-text answer to an open practice question.

    Unlike ``grade_short_answer`` there is no stored reference answer — the
    judge evaluates the answer against the lesson's source material (``context``)
    and returns formative feedback. On LLM failure we degrade to a neutral
    "partial" so the learner is never wrongly told they're wrong.
    """
    given = (given or "").strip()
    if not given:
        return ShortAnswerVerdict(verdict="incorrect", explanation="No answer provided.")

    prompt = OPEN_GRADER_PROMPT.format(
        question=question,
        context=context[:6000],
        given=given,
        language_instruction=_GRADE_LANG.get(language, _GRADE_LANG["auto"]),
    )
    try:
        llm = get_llm(temperature=0).with_structured_output(ShortAnswerVerdict)
        return await llm.ainvoke([HumanMessage(content=prompt)])
    except Exception as e:
        logger.warning("Open-answer grading failed (%s); returning neutral verdict.", e)
        return ShortAnswerVerdict(
            verdict="partial",
            explanation="Couldn't grade automatically right now — review the lesson material above.",
        )
