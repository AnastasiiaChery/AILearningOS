"""Quiz generation and attempts API."""
import json
import uuid
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.database import get_db
from app.models.document import Document, DocumentChunk
from app.models.plan import PlanTopic
from app.models.quiz import Quiz, QuizQuestion, QuizAttempt
from app.schemas.quiz import (
    GenerateQuizRequest,
    QuizOut,
    QuizDetailOut,
    SubmitAttemptRequest,
    AttemptResultOut,
)
from app.services.agents.llm_factory import get_llm
from app.services.grader import grade_short_answer, points_for_verdict
from app.services.rag.retriever import hybrid_search

router = APIRouter()
logger = logging.getLogger(__name__)

DIFFICULTY_GUIDE = {
    "easy": "Easy: test recall of explicitly stated facts and definitions. Plausible but clearly wrong distractors.",
    "medium": "Medium: test understanding and application, not just recall. Distractors should be plausible.",
    "hard": "Hard: test deeper reasoning, edge cases, and connections between ideas. Distractors must be subtle and tempting.",
}

QUIZ_PROMPT = """Generate {count} **{difficulty}** quiz questions based on the following content from a knowledge base.

Difficulty: {difficulty_guide}

Content:
{content}

Return a JSON array with this exact structure:
[
  {{
    "question_text": "...",
    "question_type": "multiple_choice",
    "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
    "correct_answer": "A",
    "explanation": "..."
  }}
]

Mix question types: multiple_choice (most common), true_false, short_answer.
For true_false: options = {{"A": "True", "B": "False"}}, correct_answer = "A" or "B".
For short_answer: options = null, correct_answer = brief expected answer.
Return only valid JSON array, no markdown."""


async def _generate_questions(content: str, count: int, difficulty: str = "medium") -> list[dict]:
    llm = get_llm()
    difficulty = difficulty if difficulty in DIFFICULTY_GUIDE else "medium"
    prompt = QUIZ_PROMPT.format(
        content=content[:6000],
        count=count,
        difficulty=difficulty,
        difficulty_guide=DIFFICULTY_GUIDE[difficulty],
    )

    from langchain_core.messages import HumanMessage
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    text = response.content.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text)


@router.post("/generate", response_model=QuizDetailOut, status_code=201)
async def generate_quiz(body: GenerateQuizRequest, db: AsyncSession = Depends(get_db)):
    if not body.document_id and not body.topic_id:
        raise HTTPException(400, "Provide document_id or topic_id")

    title = "Quiz"
    content_parts: list[str] = []

    if body.document_id:
        doc = await db.get(Document, body.document_id)
        if not doc:
            raise HTTPException(404, "Document not found")
        title = f"Quiz: {doc.original_filename}"
        result = await db.execute(
            select(DocumentChunk)
            .where(DocumentChunk.document_id == body.document_id)
            .order_by(DocumentChunk.chunk_index)
        )
        all_chunks = [c.content for c in result.scalars().all()]
        # Sample evenly across the WHOLE document, not just the first N chunks —
        # otherwise questions only ever cover the introduction. Aim for ~2 chunks
        # of material per question (bounded), spread end to end.
        budget = min(len(all_chunks), max(12, body.question_count * 2))
        if all_chunks and budget < len(all_chunks):
            step = len(all_chunks) / budget
            content_parts = [all_chunks[int(i * step)] for i in range(budget)]
        else:
            content_parts = all_chunks

    elif body.topic_id:
        topic = await db.get(PlanTopic, body.topic_id)
        if not topic:
            raise HTTPException(404, "Topic not found")
        title = f"Quiz: {topic.title}"
        chunks = await hybrid_search(topic.title, top_k=8)
        content_parts = [c["content"] for c in chunks]

    if not content_parts:
        raise HTTPException(400, "No content available to generate quiz")

    combined_content = "\n\n---\n\n".join(content_parts)
    if body.difficulty in DIFFICULTY_GUIDE and body.difficulty != "medium":
        title = f"{title} ({body.difficulty})"

    try:
        questions_data = await _generate_questions(combined_content, body.question_count, body.difficulty)
    except Exception as e:
        logger.exception("Quiz generation failed: %s", e)
        raise HTTPException(500, "Failed to generate quiz questions")

    quiz = Quiz(title=title, document_id=body.document_id, topic_id=body.topic_id)
    db.add(quiz)
    await db.flush()

    for i, q in enumerate(questions_data):
        question = QuizQuestion(
            quiz_id=quiz.id,
            question_text=q.get("question_text", ""),
            question_type=q.get("question_type", "multiple_choice"),
            options=q.get("options"),
            correct_answer=q.get("correct_answer", ""),
            explanation=q.get("explanation"),
            order_index=i,
        )
        db.add(question)

    await db.commit()

    result = await db.execute(
        select(Quiz).where(Quiz.id == quiz.id).options(selectinload(Quiz.questions))
    )
    return result.scalar_one()


@router.get("", response_model=list[QuizOut])
async def list_quizzes(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Quiz).order_by(Quiz.created_at.desc()))
    return result.scalars().all()


@router.get("/{quiz_id}", response_model=QuizDetailOut)
async def get_quiz(quiz_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Quiz).where(Quiz.id == quiz_id).options(selectinload(Quiz.questions))
    )
    quiz = result.scalar_one_or_none()
    if not quiz:
        raise HTTPException(404, "Quiz not found")
    return quiz


@router.post("/{quiz_id}/attempts", response_model=AttemptResultOut, status_code=201)
async def submit_attempt(
    quiz_id: uuid.UUID,
    body: SubmitAttemptRequest,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Quiz).where(Quiz.id == quiz_id).options(selectinload(Quiz.questions))
    )
    quiz = result.scalar_one_or_none()
    if not quiz:
        raise HTTPException(404, "Quiz not found")

    questions_by_id = {str(q.id): q for q in quiz.questions}
    scored_answers = []
    earned = 0.0
    correct_count = 0

    for ans in body.answers:
        q_id = ans.get("question_id")
        given = ans.get("answer", "")
        question = questions_by_id.get(q_id)
        if not question:
            continue

        if question.question_type == "short_answer":
            # Free text: an LLM judge decides semantic equivalence.
            v = await grade_short_answer(question.question_text, question.correct_answer, given)
            verdict = v.verdict
            points = points_for_verdict(verdict)
            explanation = v.explanation or question.explanation
        else:
            # MC / TF: the answer is a single option key — exact match is correct.
            verdict = "correct" if given.strip().upper() == question.correct_answer.strip().upper() else "incorrect"
            points = 1.0 if verdict == "correct" else 0.0
            explanation = question.explanation

        is_correct = verdict == "correct"
        earned += points
        if is_correct:
            correct_count += 1
        scored_answers.append({
            "question_id": q_id,
            "given_answer": given,
            "correct_answer": question.correct_answer,
            "correct": is_correct,
            "verdict": verdict,
            "points": points,
            "explanation": explanation,
        })

    total = len(quiz.questions)
    score = earned / total if total > 0 else 0.0

    attempt = QuizAttempt(
        quiz_id=quiz_id,
        score=score,
        answers=scored_answers,
        completed_at=datetime.now(timezone.utc),
    )
    db.add(attempt)

    # Log progress event
    from app.models.progress import ProgressEvent
    event = ProgressEvent(
        event_type="quiz_completed",
        entity_id=quiz_id,
        entity_type="quiz",
        event_data={"score": score, "correct": correct_count, "earned": earned, "total": total},
    )
    db.add(event)
    await db.commit()
    await db.refresh(attempt)
    return attempt
