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
