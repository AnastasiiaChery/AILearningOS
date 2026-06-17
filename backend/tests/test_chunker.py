"""Regression tests for the token-accurate chunker (Module 1).

Locks in the fix: chunks must never exceed the embedding model's input limit,
and a long section must be split (covered in full), not truncated.
"""
from app.services.ingestion.chunker import (
    chunk_markdown,
    count_tokens,
    _effective_max_tokens,
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
