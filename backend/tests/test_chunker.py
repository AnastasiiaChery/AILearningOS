"""Regression tests for the token-accurate chunker (Module 1).

Locks in the fix: chunks must never exceed the embedding model's input limit,
and a long section must be split (covered in full), not truncated.
"""
import app.services.ingestion.chunker as chunker_mod
from app.core.config import settings
from app.services.ingestion.chunker import (
    chunk_markdown,
    count_tokens,
    _effective_max_tokens,
    _percentile,
    _semantic_split,
)


def test_chunks_never_exceed_model_limit():
    long_section = "# Title\n\n" + ("hybrid search fuses dense and sparse vectors. " * 200)
    chunks = chunk_markdown(long_section)
    assert len(chunks) > 1, "a long section must be split into multiple chunks"
    limit = 256  # all-MiniLM-L6-v2 hard truncation limit
    for c in chunks:
        assert count_tokens(c.content) <= limit
        assert c.token_count <= limit


def test_short_section_stays_single_chunk():
    chunks = chunk_markdown("# H1\n\nA short paragraph that easily fits.")
    assert len(chunks) == 1
    assert chunks[0].h1_title == "H1"


def test_effective_max_has_headroom():
    # never proposes a window at or above the raw model limit
    assert _effective_max_tokens() <= 256 - 2


def test_full_coverage_no_text_dropped():
    body = " ".join(f"word{i}" for i in range(1000))
    chunks = chunk_markdown(f"# Big\n\n{body}")
    joined = " ".join(c.content for c in chunks)
    # first and last tokens of the section must both survive somewhere
    assert "word0" in joined
    assert "word999" in joined


# --- Recursive boundary-aware splitting (Track C, exp.3) -------------------

def test_recursive_keeps_sentences_intact():
    """With recursive chunking every sentence survives whole in some chunk —
    no phrase is cut mid-sentence (the failure mode of the token window)."""
    sentences = [f"Речення номер {i} описує важливу ідею про бекенд." for i in range(40)]
    text = "# Тема\n\n" + " ".join(sentences)
    chunks = chunk_markdown(text)
    assert len(chunks) > 1, "should split into multiple chunks"
    for s in sentences:
        assert any(s in c.content for c in chunks), f"sentence broken: {s!r}"


def test_recursive_overlap_carries_whole_unit():
    """Overlap lands on a sentence edge: a boundary sentence reappears at the
    start of the next chunk, never a fragment."""
    sentences = [f"Унікальне речення {i} про окрему тему {i}." for i in range(60)]
    text = "# T\n\n" + " ".join(sentences)
    chunks = chunk_markdown(text)
    assert len(chunks) >= 2
    shared = any(
        s in a.content and s in b.content
        for a, b in zip(chunks, chunks[1:])
        for s in sentences
    )
    assert shared, "no whole sentence shared across a chunk boundary (overlap)"


def test_recursive_respects_model_limit():
    """The recursive path must honor the same hard token cap as the baseline."""
    long_section = "# Title\n\n" + ("гібридний пошук поєднує щільні та розріджені вектори. " * 200)
    chunks = chunk_markdown(long_section)
    assert len(chunks) > 1
    for c in chunks:
        assert count_tokens(c.content) <= 256


# --- Semantic boundary splitting (Track C, exp.4) --------------------------

def test_percentile_linear_interpolation():
    """Pure-function percentile matches numpy's default (linear interpolation)."""
    assert _percentile([], 95) == 0.0
    assert _percentile([0.5], 95) == 0.5
    # 50th of 1..5 is the median (3); 0th/100th are the extremes.
    assert _percentile([1, 2, 3, 4, 5], 50) == 3
    assert _percentile([1, 2, 3, 4, 5], 0) == 1
    assert _percentile([1, 2, 3, 4, 5], 100) == 5


def test_semantic_cuts_on_topic_shift(monkeypatch):
    """A boundary should fall on the seam between two clearly distinct topics,
    not mid-topic. Two coherent paragraphs about different subjects → the cut
    lands between them, so each chunk is single-topic."""
    monkeypatch.setattr(settings, "chunking_strategy", "semantic")
    backend = "Бекенд обробляє запити. Сервер відповідає клієнту. API повертає дані. "
    cooking = "Борщ варять з буряка. Тісто замішують з борошна. Суп солять за смаком. "
    chunks = chunk_markdown("# Тема\n\n" + backend + cooking)
    assert len(chunks) >= 2, "two distinct topics should produce at least two chunks"
    # the backend topic and the cooking topic should not share a chunk
    assert not any("Бекенд" in c.content and "Борщ" in c.content for c in chunks)


def test_semantic_respects_model_limit(monkeypatch):
    """An over-long coherent run must still be repacked under the hard cap."""
    monkeypatch.setattr(settings, "chunking_strategy", "semantic")
    long_section = "# Title\n\n" + ("гібридний пошук поєднує щільні та розріджені вектори. " * 200)
    chunks = chunk_markdown(long_section)
    assert len(chunks) > 1
    for c in chunks:
        assert count_tokens(c.content) <= 256


def test_semantic_single_unit_is_passthrough(monkeypatch):
    """One sentence has no neighbour to compare — return it unchanged, no embed."""
    def _boom(*a, **k):
        raise AssertionError("must not embed a single-unit section")

    monkeypatch.setattr(settings, "chunking_strategy", "semantic")
    monkeypatch.setattr(chunker_mod, "_sentence_units", lambda *a, **k: ["one unit only"])
    monkeypatch.setattr("app.services.rag.embedder.embed_passages", _boom)
    assert _semantic_split("anything", 256, 48) == ["one unit only"]


# --- Late chunking (Track C, exp.5) ----------------------------------------

def test_unit_spans_recover_verbatim_offsets():
    """Each unit span must index back to its exact substring of the section.

    Uses a tiny per-unit budget so the section atomizes into several sentence
    units (``_atomize`` only splits once a blob exceeds the budget)."""
    text = "Перше речення тут. Друге речення далі. Третє завершує думку."
    spans = chunker_mod._unit_spans(text, 8)
    assert len(spans) >= 3, "small budget should atomize into multiple units"
    for u, s, e in spans:
        assert text[s:e] == u, "offset does not recover the unit text"


def test_pack_spans_contiguous_no_overlap_within_cap():
    """Late spans are packed to the budget with NO overlap (clean seams) and
    every span still fits the model limit."""
    cap = _effective_max_tokens()
    body = " ".join(f"Речення номер {i} описує окрему ідею." for i in range(80))
    spans = chunker_mod._pack_spans(chunker_mod._unit_spans(body, cap), cap)
    assert len(spans) >= 2, "a long body must pack into multiple chunks"
    for (_s0, e0), (s1, _e1) in zip(spans, spans[1:]):
        assert s1 >= e0, "consecutive late chunks must not overlap"
    for s, e in spans:
        assert count_tokens(body[s:e]) <= cap


def test_late_chunking_attaches_pooled_vectors(monkeypatch):
    """The late path must hand the embedder one span per chunk and store the
    returned vector on each chunk — without re-embedding the text."""
    monkeypatch.setattr(settings, "chunking_strategy", "late")
    seen = {}

    def _fake_late(body, spans):
        seen["body"] = body
        seen["spans"] = spans
        return [[float(i), 0.0, 0.0, 1.0] for i in range(len(spans))]

    monkeypatch.setattr("app.services.rag.embedder.embed_spans_late", _fake_late)
    sentences = [f"Унікальне речення {i} про окрему ідею бекенду." for i in range(40)]
    chunks = chunk_markdown("# Тема\n\n" + " ".join(sentences))

    assert chunks, "late chunking produced no chunks"
    assert len(chunks) == len(seen["spans"]), "one span must map to one chunk"
    for c in chunks:
        assert c.dense_vector is not None and len(c.dense_vector) == 4
    joined = " ".join(c.content for c in chunks)
    assert sentences[0] in joined and sentences[-1] in joined, "text dropped"
