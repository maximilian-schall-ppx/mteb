"""Evaluate a model on a local (BEIR-format) retrieval dataset built from S3.

Builds a minimal ``AbsTaskRetrieval`` whose ``metadata.dataset`` points at a local
directory produced by ``build_hf_retrieval_dataset_from_s3.py`` (loaded through
MTEB's normal ``RetrievalDatasetLoader``), then runs ``mteb.evaluate``.

Single process by default; pass ``--distributed`` to shard across ranks with the
distributed search wrapper (launch via srun/torchrun).

Example::

    python scripts/run_local_retrieval.py --dataset-path ./local_q2d_web \\
        --model lightonai/mDenseOn --name Q2DWebLocal
"""

from __future__ import annotations

import argparse
import logging

import mteb
from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

logger = logging.getLogger("run_local_retrieval")


def make_task(
    path: str, name: str, split: str, eval_langs: list[str]
) -> AbsTaskRetrieval:
    """Build an AbsTaskRetrieval instance backed by the local dataset directory."""

    class LocalRetrieval(AbsTaskRetrieval):
        metadata = TaskMetadata(
            name=name,
            description=f"Local retrieval dataset at {path}",
            reference=None,
            type="Retrieval",
            category="t2t",
            eval_splits=[split],
            eval_langs=eval_langs,
            main_score="ndcg_at_10",
            dataset={"path": path, "revision": "local"},
            date=None,
            domains=None,
            task_subtypes=None,
            license=None,
            annotations_creators=None,
            dialect=None,
            sample_creation=None,
            bibtex_citation=None,
        )

    return LocalRetrieval()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-path", required=True, help="Local dataset directory.")
    p.add_argument("--model", required=True)
    p.add_argument("--name", default="LocalRetrieval")
    p.add_argument("--split", default="test")
    p.add_argument("--eval-langs", nargs="+", default=["eng-Latn"])
    p.add_argument("--output", default=None, help="Result cache dir (rank 0 only).")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--cap-top-k", type=int, default=None, help="Cap max(k_values).")
    p.add_argument(
        "--max-seq-length",
        type=int,
        default=None,
        help="Override the encoder's max_seq_length (truncates long docs).",
    )
    p.add_argument("--distributed", action="store_true")
    p.add_argument("--shard", choices=["corpus", "query"], default="corpus")
    p.add_argument("--backend", choices=["auto", "nccl", "gloo"], default="gloo")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    import torch

    task = make_task(args.dataset_path, args.name, args.split, args.eval_langs)
    if args.cap_top_k is not None:
        task.k_values = tuple(k for k in task.k_values if k <= args.cap_top_k)
        task._top_k = max(task.k_values)

    def _set_seq_len(m) -> None:
        st = getattr(m, "model", None)
        if args.max_seq_length and st is not None and hasattr(st, "max_seq_length"):
            st.max_seq_length = args.max_seq_length

    is_main_rank = True
    if args.distributed:
        from mteb.distributed import barrier, cleanup, init_distributed
        from mteb.models import DistributedSearchWrapper

        backend = None if args.backend == "auto" else args.backend
        info = init_distributed(backend=backend)
        n_gpu = max(1, torch.cuda.device_count())
        model = mteb.get_model(args.model, device=f"cuda:{info.local_rank % n_gpu}")
        _set_seq_len(model)
        model = DistributedSearchWrapper(model, shard=args.shard)
        is_main_rank = info.is_main
    else:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model = mteb.get_model(args.model, device=device)
        _set_seq_len(model)

    cache = mteb.ResultCache(args.output) if (args.output and is_main_rank) else None
    results = mteb.evaluate(
        model,
        [task],
        cache=cache,
        encode_kwargs={"batch_size": args.batch_size},
        co2_tracker=False,
        overwrite_strategy="always",
    )

    if is_main_rank:
        tr = results[0]
        split = next(iter(tr.scores))
        print("SCORES", {k: v for k, v in tr.scores[split][0].items()
                         if isinstance(v, (int, float))})

    if args.distributed:
        barrier()
        cleanup()


if __name__ == "__main__":
    main()
