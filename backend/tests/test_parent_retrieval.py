"""Tests for parent-document expansion (Track C, exp.6).

No Qdrant: we stub ``fetch_document_chunks`` with an in-memory corpus. The
contract under test is the return-path transform — hits expand into parent
blocks, siblings ride along, and hits that map to the same parent collapse.
"""
import asyncio

import app.services.rag.retriever as retriever


def _run(coro):
    return asyncio.run(coro)


# One document, two sections. Section A = chunks 0,1,2 ; section B = chunk 3.
CORPUS = [
    {"chunk_id": "c0", "document_id": "d1", "chunk_index": 0, "h1_title": "Doc", "h2_title": "A", "h3_title": None, "content": "a0"},
    {"chunk_id": "c1", "document_id": "d1", "chunk_index": 1, "h1_title": "Doc", "h2_title": "A", "h3_title": None, "content": "a1"},
    {"chunk_id": "c2", "document_id": "d1", "chunk_index": 2, "h1_title": "Doc", "h2_title": "A", "h3_title": None, "content": "a2"},
    {"chunk_id": "c3", "document_id": "d1", "chunk_index": 3, "h1_title": "Doc", "h2_title": "B", "h3_title": None, "content": "b0"},
]
BY_ID = {c["chunk_id"]: c for c in CORPUS}


def _patch(monkeypatch):
    async def _fake_fetch(document_id):
        return [c for c in CORPUS if c["document_id"] == document_id]

    monkeypatch.setattr(retriever, "fetch_document_chunks", _fake_fetch)


def test_section_stitches_whole_section(monkeypatch):
    _patch(monkeypatch)
    # A hit on c1 expands to its whole section (c0,c1,c2), sorted by chunk_index.
    blocks = _run(retriever._expand_to_parents([BY_ID["c1"]], "section", 1))
    assert len(blocks) == 1
    assert blocks[0]["member_chunk_ids"] == ["c0", "c1", "c2"]
    assert blocks[0]["chunk_id"] == "c1"  # primary hit preserved
    assert blocks[0]["content"] == "a0\n\na1\n\na2"


def test_section_dedups_same_parent(monkeypatch):
    _patch(monkeypatch)
    # Two hits in the SAME section collapse to one block; a hit in another
    # section becomes its own block. Order follows the first hit of each parent.
    blocks = _run(
        retriever._expand_to_parents([BY_ID["c0"], BY_ID["c2"], BY_ID["c3"]], "section", 1)
    )
    assert [b["member_chunk_ids"] for b in blocks] == [["c0", "c1", "c2"], ["c3"]]


def test_window_takes_neighbors_within_radius(monkeypatch):
    _patch(monkeypatch)
    # Window ±1 around c1 = {c0,c1,c2}; crosses the section seam (it's index-based).
    blocks = _run(retriever._expand_to_parents([BY_ID["c1"]], "window", 1))
    assert blocks[0]["member_chunk_ids"] == ["c0", "c1", "c2"]


def test_window_dedups_already_covered_hit(monkeypatch):
    _patch(monkeypatch)
    # c0's window {c0,c1} already covers c1, so a later hit on c1 is dropped.
    blocks = _run(retriever._expand_to_parents([BY_ID["c0"], BY_ID["c1"]], "window", 1))
    assert len(blocks) == 1
    assert blocks[0]["member_chunk_ids"] == ["c0", "c1"]


def test_window_needs_chunk_index_on_hit(monkeypatch):
    # Regression: hybrid_search must populate chunk_index on its result dicts.
    # A hit lacking it silently collapses the window to a single chunk (the bug
    # this guards) — assert the window genuinely expands when the field is there.
    _patch(monkeypatch)
    hit_no_index = {k: v for k, v in BY_ID["c1"].items() if k != "chunk_index"}
    collapsed = _run(retriever._expand_to_parents([hit_no_index], "window", 1))
    assert collapsed[0]["member_chunk_ids"] == ["c1"]  # degenerate without index
    expanded = _run(retriever._expand_to_parents([BY_ID["c1"]], "window", 1))
    assert expanded[0]["member_chunk_ids"] == ["c0", "c1", "c2"]  # works with index
