"""Information-retrieval metrics for the eval harness.

All functions take:
  ranked   – list of chunk_ids the retriever returned, best first.
  relevant – set of chunk_ids that are correct for the query (≥1).
  k        – cutoff.

Relevance is binary (a chunk is correct or it isn't), which is the right model
for "did the retriever surface the chunk that answers this question". The three
metrics answer three different questions:

  Recall@k – of all correct chunks, what fraction landed in the top-k?
             (With one correct chunk this is just hit-or-miss = "Hit@k".)
  MRR      – how high was the FIRST correct chunk? 1/rank, 0 if absent.
             Rewards putting a right answer at position 1 over position 5.
  nDCG@k   – like MRR but credits EVERY correct chunk by its position, then
             normalizes by the best achievable ordering (so it's 0..1).
"""
from __future__ import annotations

import math
from collections.abc import Sequence


def hit_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    """1.0 if any correct chunk is in the top-k, else 0.0."""
    return 1.0 if relevant.intersection(ranked[:k]) else 0.0


def recall_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    """Fraction of the correct chunks that appear in the top-k."""
    if not relevant:
        return 0.0
    found = len(relevant.intersection(ranked[:k]))
    return found / len(relevant)


def reciprocal_rank(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    """1/rank of the first correct chunk within the top-k (0.0 if none)."""
    for i, cid in enumerate(ranked[:k], start=1):
        if cid in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    """Normalized DCG with binary gains.

    DCG  = Σ rel_i / log2(i+1)   over the returned top-k (rel_i ∈ {0,1}).
    IDCG = the same for the ideal ordering (all correct chunks first), so the
    score is 1.0 only when every correct chunk that *can* fit in k sits at the
    top in the best possible order.
    """
    if not relevant:
        return 0.0
    dcg = 0.0
    for i, cid in enumerate(ranked[:k], start=1):
        if cid in relevant:
            dcg += 1.0 / math.log2(i + 1)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0
