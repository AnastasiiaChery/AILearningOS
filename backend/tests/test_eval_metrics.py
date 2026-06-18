"""Unit tests for the eval metrics — hand-computed expected values."""
import math

from app.eval import metrics

RANKED = ["a", "b", "c", "d"]


def test_hit_at_k():
    assert metrics.hit_at_k(RANKED, {"c"}, 1) == 0.0
    assert metrics.hit_at_k(RANKED, {"c"}, 3) == 1.0
    assert metrics.hit_at_k(RANKED, {"z"}, 4) == 0.0


def test_recall_at_k_single():
    # one relevant chunk → recall is hit-or-miss
    assert metrics.recall_at_k(RANKED, {"c"}, 2) == 0.0
    assert metrics.recall_at_k(RANKED, {"c"}, 3) == 1.0


def test_recall_at_k_multi():
    # two relevant: only "a" is in top-2 → 1/2
    assert metrics.recall_at_k(RANKED, {"a", "c"}, 2) == 0.5
    assert metrics.recall_at_k(RANKED, {"a", "c"}, 4) == 1.0


def test_reciprocal_rank():
    assert metrics.reciprocal_rank(RANKED, {"a"}, 10) == 1.0
    assert metrics.reciprocal_rank(RANKED, {"c"}, 10) == 1.0 / 3
    assert metrics.reciprocal_rank(RANKED, {"z"}, 10) == 0.0
    # cutoff excludes the hit
    assert metrics.reciprocal_rank(RANKED, {"c"}, 2) == 0.0


def test_ndcg_single_at_first():
    assert metrics.ndcg_at_k(RANKED, {"a"}, 5) == 1.0


def test_ndcg_single_at_third():
    # gain 1/log2(4) over ideal 1/log2(2) = 0.5
    assert math.isclose(metrics.ndcg_at_k(RANKED, {"c"}, 3), 0.5)


def test_ndcg_multi():
    # relevant {a,c}, top-2 holds only "a"
    # dcg@2 = 1/log2(2) = 1.0 ; idcg@2 = 1/log2(2) + 1/log2(3)
    idcg = 1.0 + 1.0 / math.log2(3)
    assert math.isclose(metrics.ndcg_at_k(RANKED, {"a", "c"}, 2), 1.0 / idcg)


def test_empty_relevant_is_zero():
    assert metrics.recall_at_k(RANKED, set(), 5) == 0.0
    assert metrics.ndcg_at_k(RANKED, set(), 5) == 0.0


# --- Block-aware metrics (parent-document, exp.6) ---------------------------

# Reduction: singleton blocks must reproduce the chunk-level metrics exactly, so
# non-parent strategies keep their old numbers. Mirror the chunk-level cases.
SINGLETON = [["a"], ["b"], ["c"], ["d"]]


def test_block_metrics_reduce_to_chunk_metrics():
    for rel in ({"c"}, {"a", "c"}, {"z"}):
        for k in (1, 2, 3, 4):
            assert metrics.block_hit_at_k(SINGLETON, rel, k) == metrics.hit_at_k(RANKED, rel, k)
            assert math.isclose(
                metrics.block_recall_at_k(SINGLETON, rel, k), metrics.recall_at_k(RANKED, rel, k)
            )
            assert math.isclose(
                metrics.block_ndcg_at_k(SINGLETON, rel, k), metrics.ndcg_at_k(RANKED, rel, k)
            )
        assert math.isclose(
            metrics.block_reciprocal_rank(SINGLETON, rel, 10),
            metrics.reciprocal_rank(RANKED, rel, 10),
        )


def test_block_recall_credits_covered_siblings():
    # Two gold chunks, both stitched into the FIRST block (one section). The
    # whole point of parent expansion: recall@1 jumps from 0.5 (only one gold
    # ranked) to 1.0 (the sibling rides along).
    blocks = [["c", "x", "y"], ["b"], ["z"]]
    assert metrics.block_recall_at_k(blocks, {"c", "y"}, 1) == 1.0
    # ... whereas with singleton chunks only "c" is in the top-1.
    assert metrics.recall_at_k(["c", "y", "b"], {"c", "y"}, 1) == 0.5


def test_block_ndcg_capped_at_one():
    # One block delivering two gold at rank 1 saturates (does not exceed 1.0).
    assert metrics.block_ndcg_at_k([["a", "b"], ["c"]], {"a", "b"}, 5) == 1.0
