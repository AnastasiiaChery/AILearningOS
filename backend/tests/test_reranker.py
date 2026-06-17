"""Test for the cross-encoder reranker (Module 3).

Verifies the stage-2 contract: candidates are re-scored, sorted by relevance,
truncated to top_k, and the clearly-relevant doc beats the clearly-irrelevant one.
"""
from app.services.rag.reranker import rerank_sync


def test_rerank_orders_by_relevance_and_truncates():
    query = "how does a cross-encoder rerank search results"
    chunks = [
        {"content": "Bananas are a good source of potassium and grow in bunches."},
        {"content": "A cross-encoder reads query and document together and scores "
                    "their relevance, fixing the order of retrieved results."},
        {"content": "The weather today is sunny with a light breeze."},
    ]
    out = rerank_sync(query, chunks, top_k=2)
    assert len(out) == 2  # truncated to top_k
    # the relevant chunk must rank first
    assert "cross-encoder" in out[0]["content"]
    # scores are present and sorted descending
    assert out[0]["rerank_score"] >= out[1]["rerank_score"]


def test_rerank_empty_input():
    assert rerank_sync("anything", [], top_k=5) == []
