"""Bootstrap a ``torch.distributed`` process group from SLURM environment variables.

Used to run a single MTEB task evaluation across many ranks (SPMD): the same
evaluation script runs on every rank launched by ``srun`` (one rank per GPU), and
the ranks cooperate via collectives to share the work of that one task. See
``scripts/run_distributed_retrieval.py`` for a usage example.
"""

from __future__ import annotations

import contextlib
import logging
import os
import subprocess
from dataclasses import dataclass

logger = logging.getLogger(__name__)

DEFAULT_MASTER_PORT = "29500"


@dataclass(frozen=True)
class DistInfo:
    """Resolved distributed placement for this process."""

    rank: int
    world_size: int
    local_rank: int

    @property
    def is_main(self) -> bool:
        """Whether this is the coordinating (rank 0) process."""
        return self.rank == 0


def _resolve_master_addr() -> str:
    """First hostname of the SLURM allocation, used as the rendezvous host."""
    nodelist = os.environ.get("SLURM_NODELIST") or os.environ.get(
        "SLURM_JOB_NODELIST"
    )
    if not nodelist:
        return "127.0.0.1"
    hosts = subprocess.check_output(
        ["scontrol", "show", "hostnames", nodelist],
        text=True,
    ).splitlines()
    return hosts[0].strip()


def init_distributed_from_slurm(
    backend: str | None = None,
    master_port: str | None = None,
) -> DistInfo:
    """Initialise the process group from SLURM env vars and pin the local GPU.

    Reads ``SLURM_PROCID`` (global rank), ``SLURM_NTASKS`` (world size), and
    ``SLURM_LOCALID`` (local rank), derives ``MASTER_ADDR`` from the node list,
    then calls ``init_process_group`` and ``torch.cuda.set_device(local_rank)``.

    Args:
        backend: Process-group backend. Defaults to ``"nccl"`` when CUDA is
            available, otherwise ``"gloo"``.
        master_port: Rendezvous port. Defaults to ``$MASTER_PORT`` or 29500.

    Returns:
        The resolved :class:`DistInfo` for this process.
    """
    import torch
    import torch.distributed as dist

    rank = int(os.environ.get("SLURM_PROCID", "0"))
    world_size = int(os.environ.get("SLURM_NTASKS", "1"))
    local_rank = int(os.environ.get("SLURM_LOCALID", "0"))

    os.environ.setdefault("MASTER_ADDR", _resolve_master_addr())
    os.environ.setdefault(
        "MASTER_PORT", master_port or os.environ.get("MASTER_PORT", DEFAULT_MASTER_PORT)
    )
    # torch reads RANK / WORLD_SIZE from the environment for env:// rendezvous.
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(local_rank)

    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"

    if torch.cuda.is_available():
        # If SLURM exposes all node GPUs to every task, local_rank selects the
        # GPU; if it binds one GPU per task, only device 0 is visible. Modulo
        # keeps the index valid in both layouts.
        torch.cuda.set_device(local_rank % torch.cuda.device_count())

    if world_size > 1 and not dist.is_initialized():
        logger.info(
            "Initialising process group backend=%s rank=%d world_size=%d "
            "local_rank=%d master=%s:%s",
            backend,
            rank,
            world_size,
            local_rank,
            os.environ["MASTER_ADDR"],
            os.environ["MASTER_PORT"],
        )
        dist.init_process_group(
            backend=backend, rank=rank, world_size=world_size
        )

    return DistInfo(rank=rank, world_size=world_size, local_rank=local_rank)


def is_main() -> bool:
    """Whether this process is rank 0 (or distribution is inactive)."""
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        return True
    return dist.get_rank() == 0


def barrier() -> None:
    """Synchronise all ranks; a no-op when distribution is inactive."""
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        dist.barrier()


def cleanup() -> None:
    """Destroy the process group if one is active."""
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


@contextlib.contextmanager
def main_process_first():
    """Run the ``with`` body on rank 0 first, other ranks after a barrier.

    Useful to serialise first-time dataset/model downloads so ranks share the
    HuggingFace cache instead of racing to populate it.
    """
    if is_main():
        yield
        barrier()
    else:
        barrier()
        yield
