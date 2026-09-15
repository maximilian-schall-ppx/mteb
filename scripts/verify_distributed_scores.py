"""Verify distributed retrieval scores match single-process on a real BEIR task.

Runs a model on a BEIR retrieval task twice — once single-process (baseline) and
once across ``--world-size`` ranks via :class:`mteb.models.DistributedSearchWrapper`
— then asserts the scores match. Uses the gloo backend on CPU, so it needs no GPU.

Examples::

    # dense, corpus-sharded
    python scripts/verify_distributed_scores.py \\
        --model sentence-transformers/all-MiniLM-L6-v2 --task NFCorpus \\
        --shard corpus --world-size 2

    # late-interaction (ColBERT/PyLate), corpus-sharded
    python scripts/verify_distributed_scores.py \\
        --model colbert-ir/colbertv2.0 --task NFCorpus --shard corpus --world-size 2

    # BM25, query-sharded
    python scripts/verify_distributed_scores.py \\
        --model mteb/baseline-bm25s --task NFCorpus --shard query --world-size 2
"""

from __future__ import annotations

import argparse
import logging
import os

logger = logging.getLogger("verify_distributed_scores")


def _extract_scores(model_result) -> dict[str, dict[str, float]]:
    """Return ``{split: {metric: value}}`` for the (single) task result."""
    task_result = model_result[0]
    out: dict[str, dict[str, float]] = {}
    for split, score_list in task_result.scores.items():
        # One subset for BEIR tasks; average defensively if more.
        agg: dict[str, float] = {}
        for scores in score_list:
            for k, v in scores.items():
                if isinstance(v, (int, float)):
                    agg.setdefault(k, 0.0)
                    agg[k] += float(v) / len(score_list)
        out[split] = agg
    return out


def run_baseline(model_name: str, task_name: str, batch_size: int) -> dict:
    import mteb

    model = mteb.get_model(model_name, device="cpu")
    task = mteb.get_task(task_name)
    results = mteb.evaluate(
        model,
        [task],
        cache=None,
        encode_kwargs={"batch_size": batch_size},
        co2_tracker=False,
    )
    return _extract_scores(results)


def _dist_worker(rank, world_size, model_name, task_name, shard, batch_size, q):
    import torch.distributed as dist

    import mteb
    from mteb.models import DistributedSearchWrapper

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29566")
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        model = mteb.get_model(model_name, device="cpu")
        search_model = DistributedSearchWrapper(model, shard=shard)
        task = mteb.get_task(task_name)
        results = mteb.evaluate(
            search_model,
            [task],
            cache=None,
            encode_kwargs={"batch_size": batch_size},
            co2_tracker=False,
        )
        if rank == 0:
            q.put(_extract_scores(results))
    finally:
        dist.destroy_process_group()


def run_distributed(
    model_name: str, task_name: str, shard: str, world_size: int, batch_size: int
) -> dict:
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    mp.spawn(
        _dist_worker,
        args=(world_size, model_name, task_name, shard, batch_size, q),
        nprocs=world_size,
        join=True,
    )
    return q.get(timeout=1800)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--task", default="NFCorpus")
    p.add_argument("--shard", choices=["corpus", "query"], default="corpus")
    p.add_argument("--world-size", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--tol", type=float, default=1e-4, help="Abs tolerance on scores.")
    p.add_argument(
        "--metrics",
        nargs="+",
        default=["ndcg_at_10", "recall_at_10", "main_score"],
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    logger.info("=== Baseline (single process) ===")
    baseline = run_baseline(args.model, args.task, args.batch_size)
    logger.info(
        "=== Distributed (world_size=%d, shard=%s) ===", args.world_size, args.shard
    )
    distributed = run_distributed(
        args.model, args.task, args.shard, args.world_size, args.batch_size
    )

    print(f"\nModel: {args.model}  Task: {args.task}  shard={args.shard}  "
          f"world_size={args.world_size}")
    ok = True
    for split in sorted(baseline):
        print(f"\n[{split}]")
        print(f"  {'metric':<18}{'baseline':>12}{'distributed':>14}{'|diff|':>12}")
        for m in args.metrics:
            b = baseline[split].get(m)
            d = distributed.get(split, {}).get(m)
            if b is None or d is None:
                continue
            diff = abs(b - d)
            flag = "" if diff <= args.tol else "  <-- MISMATCH"
            ok = ok and diff <= args.tol
            print(f"  {m:<18}{b:>12.6f}{d:>14.6f}{diff:>12.2e}{flag}")

    print("\nRESULT:", "PASS ✅" if ok else "FAIL ❌")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
