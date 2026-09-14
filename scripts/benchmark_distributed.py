"""Scaling benchmark for distributed single-task retrieval.

Runs one retrieval task under ``torch.distributed`` (one rank per GPU, launched
by ``srun``) and reports wall-clock timing so speedup can be measured across node
counts. Encoding runs on each rank's GPU; the small result all-gather uses gloo.

Prints one ``BENCHMARK {json}`` line from rank 0 with total and per-phase times.

Example (1 node x 8 GPUs)::

    srun --nodes=1 --ntasks-per-node=8 --gres=gpu:8 \\
        python scripts/benchmark_distributed.py --model sentence-transformers/all-MiniLM-L6-v2 \\
        --task MSMARCO --shard corpus --cap-top-k 100
"""

from __future__ import annotations

import argparse
import json
import logging
import time

import mteb
from mteb.distributed import barrier, cleanup, init_distributed_from_slurm
from mteb.models import DistributedSearchWrapper

logger = logging.getLogger("benchmark_distributed")


def _phase_durations(task_result) -> dict[str, float]:
    """Sum evaluation-phase durations (seconds) by phase name."""
    out: dict[str, float] = {}
    for ph in task_result.evaluation_phases or []:
        out[ph["name"]] = out.get(ph["name"], 0.0) + (ph["end"] - ph["start"])
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--task", default="MSMARCO")
    p.add_argument("--shard", choices=["corpus", "query"], default="corpus")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument(
        "--cap-top-k",
        type=int,
        default=None,
        help="Cap max(k_values) to keep the result all-gather cheap. Does not "
        "affect corpus encoding (the part being scaled).",
    )
    p.add_argument("--backend", choices=["auto", "nccl", "gloo"], default="gloo")
    p.add_argument(
        "--baseline",
        action="store_true",
        help="Run the raw model through the stock mteb.evaluate path (no wrapper, "
        "single process) to measure a 1-GPU baseline.",
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    import torch

    if args.baseline:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model = mteb.get_model(args.model, device=device)
        search_model = model  # default mteb path, no distributed wrapper
        world_size, is_main_rank = 1, True
    else:
        backend = None if args.backend == "auto" else args.backend
        info = init_distributed_from_slurm(backend=backend)
        n_gpu = max(1, torch.cuda.device_count())
        device = f"cuda:{info.local_rank % n_gpu}"
        model = mteb.get_model(args.model, device=device)
        search_model = DistributedSearchWrapper(model, shard=args.shard)
        world_size, is_main_rank = info.world_size, info.is_main

    task = mteb.get_task(args.task)
    if args.cap_top_k is not None:
        task.k_values = tuple(k for k in task.k_values if k <= args.cap_top_k)
        task._top_k = max(task.k_values)

    max_seq_length = getattr(getattr(model, "model", None), "max_seq_length", None)

    barrier()
    t0 = time.monotonic()
    results = mteb.evaluate(
        search_model,
        [task],
        cache=None,
        encode_kwargs={"batch_size": args.batch_size},
        co2_tracker=False,
    )
    barrier()
    total = time.monotonic() - t0

    if is_main_rank:
        tr = results[0]
        phases = _phase_durations(tr)
        split = next(iter(tr.scores))
        scores = tr.scores[split][0]
        payload = {
            "model": args.model,
            "task": args.task,
            "shard": None if args.baseline else args.shard,
            "path": "mteb-default" if args.baseline else "distributed-wrapper",
            "world_size": world_size,
            "batch_size": args.batch_size,
            "max_seq_length": max_seq_length,
            "top_k": max(task.k_values),
            "total_s": round(total, 2),
            "phases_s": {k: round(v, 2) for k, v in phases.items()},
            "ndcg_at_10": scores.get("ndcg_at_10"),
            "recall_at_100": scores.get("recall_at_100"),
        }
        print("BENCHMARK " + json.dumps(payload))

    barrier()
    cleanup()


if __name__ == "__main__":
    main()
