"""Tests for short-answer LLM grading (P0 fix).

No network: we stub the LLM. The contracts under test:
- a semantically-equivalent answer that differs in wording is NOT marked wrong
  (the original `==` bug: "It is compiled into bytecode" vs "Bytecode");
- an empty answer short-circuits to "incorrect" with no LLM call;
- an LLM error degrades gracefully to exact-match instead of raising;
- verdicts map to the expected score weights.
"""
import asyncio

import app.services.grader as grader
from app.services.grader import ShortAnswerVerdict, points_for_verdict


class _FakeStructuredLLM:
    """Mimics get_llm(...).with_structured_output(Model)."""

    def __init__(self, verdict=None, raises=False):
        self._verdict = verdict
        self._raises = raises

    def with_structured_output(self, _model):
        return self

    async def ainvoke(self, _messages):
        if self._raises:
            raise RuntimeError("boom")
        return self._verdict


def _run(coro):
    return asyncio.run(coro)


def _patch_llm(monkeypatch, **kw):
    monkeypatch.setattr(grader, "get_llm", lambda **_: _FakeStructuredLLM(**kw))


def test_semantically_equivalent_answer_is_not_wrong(monkeypatch):
    # The acceptance criterion from the task: paraphrase must not be incorrect.
    _patch_llm(monkeypatch, verdict=ShortAnswerVerdict(verdict="correct", explanation="ok"))
    v = _run(grader.grade_short_answer(
        "What does the Python interpreter produce from source?",
        "Bytecode",
        "It is compiled into bytecode",
    ))
    assert v.verdict in ("correct", "partial")
    assert points_for_verdict(v.verdict) > 0


def test_clearly_wrong_answer_is_incorrect(monkeypatch):
    _patch_llm(monkeypatch, verdict=ShortAnswerVerdict(verdict="incorrect", explanation="off-topic"))
    v = _run(grader.grade_short_answer("...", "Bytecode", "The weather is nice"))
    assert v.verdict == "incorrect"
    assert points_for_verdict(v.verdict) == 0.0


def test_empty_answer_short_circuits_without_llm(monkeypatch):
    # If the LLM were called it would raise; an empty answer must not call it.
    _patch_llm(monkeypatch, raises=True)
    v = _run(grader.grade_short_answer("q", "ref", "   "))
    assert v.verdict == "incorrect"


def test_llm_error_falls_back_to_exact_match(monkeypatch):
    _patch_llm(monkeypatch, raises=True)
    match = _run(grader.grade_short_answer("q", "Bytecode", "bytecode"))
    assert match.verdict == "correct"
    miss = _run(grader.grade_short_answer("q", "Bytecode", "something else"))
    assert miss.verdict == "incorrect"


def test_points_for_verdict_mapping():
    assert points_for_verdict("correct") == 1.0
    assert points_for_verdict("partial") == grader.PARTIAL_CREDIT
    assert points_for_verdict("incorrect") == 0.0
    assert points_for_verdict("garbage") == 0.0
