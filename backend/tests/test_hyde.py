"""Tests for HyDE generation (Track C, exp.2).

No network: we stub the LLM. The contract under test is the graceful-degradation
guarantee — HyDE must never make retrieval worse than no-HyDE, so any empty
output or LLM error falls back to the original query.
"""
import asyncio

import app.services.rag.hyde as hyde


class _FakeResp:
    def __init__(self, content):
        self.content = content


class _FakeLLM:
    def __init__(self, content=None, raises=False):
        self._content = content
        self._raises = raises

    async def ainvoke(self, messages):
        if self._raises:
            raise RuntimeError("boom")
        return _FakeResp(self._content)


def _run(coro):
    return asyncio.run(coro)


def test_returns_generated_passage_stripped(monkeypatch):
    monkeypatch.setattr(hyde, "get_llm", lambda **kw: _FakeLLM(content="  a hypothetical answer  "))
    assert _run(hyde.generate_hypothetical("что такое RRF?")) == "a hypothetical answer"


def test_empty_output_falls_back_to_query(monkeypatch):
    monkeypatch.setattr(hyde, "get_llm", lambda **kw: _FakeLLM(content="   "))
    q = "edge case question"
    assert _run(hyde.generate_hypothetical(q)) == q


def test_none_content_falls_back_to_query(monkeypatch):
    monkeypatch.setattr(hyde, "get_llm", lambda **kw: _FakeLLM(content=None))
    q = "another question"
    assert _run(hyde.generate_hypothetical(q)) == q


def test_llm_error_falls_back_to_query(monkeypatch):
    monkeypatch.setattr(hyde, "get_llm", lambda **kw: _FakeLLM(raises=True))
    q = "question that breaks the llm"
    assert _run(hyde.generate_hypothetical(q)) == q


def test_uses_temperature_zero(monkeypatch):
    captured = {}

    def fake_get_llm(**kw):
        captured.update(kw)
        return _FakeLLM(content="ok")

    monkeypatch.setattr(hyde, "get_llm", fake_get_llm)
    _run(hyde.generate_hypothetical("q"))
    assert captured.get("temperature") == 0
