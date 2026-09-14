"""Multi-node evaluation of a single MTEB retrieval task via ``torch.distributed``.

POC covering three model families through one wrapper
(:class:`mteb.models.DistributedSearchWrapper`):

* **dense** encoders            → ``--shard corpus`` (default)
* **late-interaction** (ColBERT) → ``--shard corpus``
* **BM25** / lexical            → ``--shard query`` (global corpus stats)

Launch one task per GPU with ``srun`` (see ``run_distributed_retrieval.sbatch``).
Every rank runs the same evaluation; the search work is sharded across ranks and
merged, so all ranks compute identical scores. Only rank 0 writes to disk.

Examples::

    # dense
    srun ... python scripts/run_distributed_retrieval.py \\
        --model intfloat/e5-base-v2 --task NFCorpus --output ./dist_results
    # late-interaction
    srun ... python scripts/run_distributed_retrieval.py \\
        --model colbert-ir/colbertv2.0 --task NFCorpus --output ./dist_results
    # BM25 (query-sharded)
    srun ... python scripts/run_distributed_retrieval.py \\
        --model mteb/baseline-bm25s --task NFCorpus --shard query --output ./dist_results
"""

from __future__ import annotations

import argparse
import logging

import mteb
from mteb.distributed import (
    barrier,
    cleanup,
    init_distributed_from_slurm,
    main_process_first,
)
from mteb.models import DistributedSearchWrapper

logger = logging.getLogger(__name__)


def _default_shard(model) -> str:
    """Pick a sharding mode: query for lexical/BM25 models, corpus otherwise.

    BM25 scores depend on global corpus statistics (IDF, average doc length),
    so its corpus cannot be sharded; dense and late-interaction scores are
    per-(query, doc) and shard cleanly.
    """
    meta = getattr(model, "mteb_model_meta", None)
    model_type = getattr(meta, "model_type", None) or []
    return "query" if "sparse" in model_type else "corpus"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Model name to evaluate.")
    parser.add_argument("--revision", default=None, help="Model revision.")
    parser.add_argument("--task", required=True, help="Retrieval task name.")
    parser.add_argument(
        "--output", default="./dist_results", help="Result cache directory."
    )
    parser.add_argument(
        "--shard",
        choices=["corpus", "query", "auto"],
        default="auto",
        help="Sharding mode. 'auto' picks query for BM25/lexical, else corpus.",
    )
    parser.add_argument(
        "--prediction-folder",
        default=None,
        help="Optional folder to save predictions.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--backend",
        choices=["auto", "nccl", "gloo"],
        default="auto",
        help="Process-group backend. 'gloo' is robust for the small all-gather "
        "of search results while models still encode on each rank's GPU.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    import torch

    backend = None if args.backend == "auto" else args.backend
    info = init_distributed_from_slurm(backend=backend)
    n_gpu = max(1, torch.cuda.device_count())
    device = f"cuda:{info.local_rank % n_gpu}"
    logger.info("Rank %d loading %s on %s", info.rank, args.model, device)

    # Serialise first-time downloads so ranks share the HF cache.
    with main_process_first():
        model = mteb.get_model(args.model, revision=args.revision, device=device)

    shard = _default_shard(model) if args.shard == "auto" else args.shard
    logger.info("Rank %d using shard mode: %s", info.rank, shard)
    search_model = DistributedSearchWrapper(model, shard=shard)

    task = mteb.get_task(args.task)

    # Gate all disk writes to rank 0; control flow stays identical on every rank.
    cache = mteb.ResultCache(args.output) if info.is_main else None
    prediction_folder = args.prediction_folder if info.is_main else None

    mteb.evaluate(
        search_model,
        [task],
        cache=cache,
        prediction_folder=prediction_folder,
        encode_kwargs={"batch_size": args.batch_size},
        overwrite_strategy="always",
    )

    barrier()
    cleanup()
    if info.is_main:
        logger.info("Done. Results written under %s", args.output)


if __name__ == "__main__":
    main()
