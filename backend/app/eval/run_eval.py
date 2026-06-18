"""Run the retrieval eval and print a metrics table.

  docker compose exec -T backend sh -c \
    'cd /app && PYTHONPATH=/app uv run python -m app.eval.run_eval'

Each strategy is queried ONCE per question at the deepest cutoff (MAX_K); every
@k metric is then computed by slicing that one ranked list — so dense/sparse/
hybrid/hybrid+rerank are compared on identical retrievals.

Flags:
  --k 1,3,5,10     cutoffs to report
  --only hybrid,…  restrict to some strategies
  --verbose        per-question first-hit rank for the deepest strategy
  --save NAME      write a JSON snapshot to eval/results/NAME.json (for до/после)
"""
from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from app.eval import metrics
from app.eval.resolve import fetch_chunks, load_goldset, resolve_all
from app.eval.retrievers import STRATEGIES

MAX_K = 10
DEFAULT_KS = [1, 3, 5, 10]
RESULTS_DIR = Path(__file__).parent / "results"


async def _run(ks: list[int], only: list[str] | None, verbose: bool) -> dict:
    goldset = load_goldset()
    chunks = await fetch_chunks()
    resolved = resolve_all(goldset, chunks)

    scorable = [r for r in resolved if r.relevant and not r.problems]
    skipped = [r for r in resolved if r not in scorable]
    if skipped:
        print(f"⚠ skipping {len(skipped)} unresolved question(s): "
              f"{', '.join(r.question.id for r in skipped)}\n")
    if not scorable:
        raise SystemExit("No scorable questions — fix goldset.toml anchors first "
                         "(run `python -m app.eval.resolve`).")

    strategies = {n: f for n, f in STRATEGIES.items() if not only or n in only}
    depth = max(MAX_K, max(ks))
    report: dict[str, dict] = {}

    for name, retrieve in strategies.items():
        agg = {f"{m}@{k}": 0.0 for m in ("hit", "recall", "ndcg") for k in ks}
        agg_mrr = 0.0
        per_q = []
        for r in scorable:
            ranked = await retrieve(r.question.question, depth)
            # A retriever may return a flat chunk_id list OR (parent-document,
            # exp.6) a list of blocks — each block a list of covered chunk_ids.
            # Normalize to blocks; a flat list is just singleton blocks, on which
            # the block metrics reduce exactly to the chunk-level ones.
            blocks = ranked if (ranked and isinstance(ranked[0], (list, tuple))) else [[c] for c in ranked]
            rank = next((i for i, b in enumerate(blocks, 1) if r.relevant & set(b)), None)
            per_q.append((r.question.id, rank))
            for k in ks:
                agg[f"hit@{k}"] += metrics.block_hit_at_k(blocks, r.relevant, k)
                agg[f"recall@{k}"] += metrics.block_recall_at_k(blocks, r.relevant, k)
                agg[f"ndcg@{k}"] += metrics.block_ndcg_at_k(blocks, r.relevant, k)
            agg_mrr += metrics.block_reciprocal_rank(blocks, r.relevant, depth)
        n = len(scorable)
        report[name] = {
            **{key: round(val / n, 4) for key, val in agg.items()},
            f"mrr@{depth}": round(agg_mrr / n, 4),
            "_per_q": per_q,
        }

    return {
        "n_questions": len(scorable),
        "n_skipped": len(skipped),
        "ks": ks,
        "depth": depth,
        "strategies": report,
        "verbose": verbose,
    }


def _print_table(result: dict) -> None:
    ks = result["ks"]
    depth = result["depth"]
    strat = result["strategies"]
    names = list(strat)
    namew = max(len(n) for n in names) + 1

    def block(metric: str, title: str) -> None:
        print(f"\n{title}")
        header = " " * namew + "".join(f"{f'@{k}':>9}" for k in ks)
        print(header)
        print("-" * len(header))
        for n in names:
            row = f"{n:<{namew}}" + "".join(f"{strat[n][f'{metric}@{k}']:>9.3f}" for k in ks)
            print(row)

    print(f"\n{'='*60}\nRetrieval eval · {result['n_questions']} questions "
          f"· cutoffs {ks}\n{'='*60}")
    block("hit", "Hit@k  (any correct chunk in top-k — found at all?)")
    block("recall", "Recall@k  (fraction of correct chunks in top-k)")
    block("ndcg", "nDCG@k  (correct chunks, position-weighted, 0..1)")
    print(f"\nMRR@{depth}  (1 / rank of first correct chunk)")
    print("-" * (namew + 9))
    for n in names:
        print(f"{n:<{namew}}{strat[n][f'mrr@{depth}']:>9.3f}")

    if result.get("verbose"):
        print(f"\nFirst-hit rank per question (— = miss within top-{depth}):")
        ref = names[-1]
        for qid, rank in strat[ref]["_per_q"]:
            print(f"  {qid:<22} {ref}: {'—' if rank is None else rank}")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", default=",".join(map(str, DEFAULT_KS)))
    ap.add_argument("--only", default="")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--save", default="")
    args = ap.parse_args()

    ks = sorted(int(x) for x in args.k.split(",") if x.strip())
    only = [s.strip() for s in args.only.split(",") if s.strip()] or None

    result = await _run(ks, only, args.verbose)
    _print_table(result)

    if args.save:
        RESULTS_DIR.mkdir(exist_ok=True)
        # strip per-question detail from saved snapshot; keep aggregates
        snap = {**result, "timestamp": datetime.now(timezone.utc).isoformat()}
        for s in snap["strategies"].values():
            s.pop("_per_q", None)
        path = RESULTS_DIR / f"{args.save}.json"
        path.write_text(json.dumps(snap, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nSaved snapshot → {path}")


if __name__ == "__main__":
    asyncio.run(main())
