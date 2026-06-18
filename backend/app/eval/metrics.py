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

# --- Block-aware variants (Track C, exp.6: parent-document retrieval) -------
#
# Parent-document retrieval changes the retrieval *unit*: each return slot is no
# longer a single chunk but a parent BLOCK (a hit chunk stitched with its
# siblings). So a ranked item is a *set* of covered chunk_ids, and "@k" counts
# blocks (= primary retrieval slots), not chunks. The functions below score that
# block list, crediting every chunk a block covers.
#
# They reduce EXACTLY to the chunk-level functions above when every block is a
# singleton ``[chunk_id]`` — so non-parent strategies reproduce their old numbers
# bit-for-bit and run_eval can route all strategies through one code path. (See
# test_eval_metrics.py for the reduction test.)


def _covered(blocks: Sequence[Sequence[str]], k: int) -> set[str]:
    """Union of all chunk_ids carried by the top-k blocks."""
    out: set[str] = set()
    for b in blocks[:k]:
        out.update(b)
    return out


def block_hit_at_k(blocks: Sequence[Sequence[str]], relevant: set[str], k: int) -> float:
    """1.0 if any top-k block covers a correct chunk, else 0.0."""
    return 1.0 if relevant & _covered(blocks, k) else 0.0


def block_recall_at_k(blocks: Sequence[Sequence[str]], relevant: set[str], k: int) -> float:
    """Fraction of correct chunks covered by the union of the top-k blocks.

    This is where parent expansion can pay off: a gold chunk that wasn't itself
    ranked can still be *covered* by riding along as a sibling of a hit.
    """
    if not relevant:
        return 0.0
    return len(relevant & _covered(blocks, k)) / len(relevant)


def block_reciprocal_rank(blocks: Sequence[Sequence[str]], relevant: set[str], k: int) -> float:
    """1/rank of the first block that covers a correct chunk (0.0 if none)."""
    for i, b in enumerate(blocks[:k], start=1):
        if relevant & set(b):
            return 1.0 / i
    return 0.0


def block_ndcg_at_k(blocks: Sequence[Sequence[str]], relevant: set[str], k: int) -> float:
    """nDCG where each correct chunk is credited by the FIRST block delivering it.

    DCG  = Σ 1/log2(rank+1) over correct chunks, rank = first top-k block covering
           it. IDCG = the ideal staggered ordering (same as the chunk version), so
           singleton blocks reproduce ``ndcg_at_k`` exactly. Capped at 1.0: one
           block may deliver several correct chunks at once (its members share the
           block's rank), which can push DCG past the staggered ideal.
    """
    if not relevant:
        return 0.0
    dcg = 0.0
    for g in relevant:
        rank = next((i for i, b in enumerate(blocks[:k], start=1) if g in set(b)), None)
        if rank is not None:
            dcg += 1.0 / math.log2(rank + 1)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return min(1.0, dcg / idcg) if idcg else 0.0


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
