"""Bootstrap a ``torch.distributed`` process group for a single MTEB task (SPMD).

The same evaluation script runs on every rank (one rank per GPU) and the ranks
cooperate via collectives to share the work of one task. Two launchers are
supported and auto-detected by :func:`init_distributed`:

* ``srun`` with one task per GPU (``init_distributed_from_slurm``);
* ``torchrun`` / ``torchelastic`` with one launcher per node fanning out to the
  local GPUs (``init_distributed_from_torchrun``).

See ``scripts/run_distributed_retrieval.py`` and the ``.sbatch`` files for usage.
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


def _start_process_group(
    rank: int, world_size: int, local_rank: int, backend: str | None
) -> DistInfo:
    """Pin the local GPU and initialise the process group (env:// rendezvous)."""
    import torch
    import torch.distributed as dist

    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"

    if torch.cuda.is_available():
        # Modulo keeps the index valid whether all node GPUs or one per task are visible.
        torch.cuda.set_device(local_rank % torch.cuda.device_count())

    if world_size > 1 and not dist.is_initialized():
        logger.info(
            "Initialising process group backend=%s rank=%d world_size=%d "
            "local_rank=%d master=%s:%s",
            backend,
            rank,
            world_size,
            local_rank,
            os.environ.get("MASTER_ADDR"),
            os.environ.get("MASTER_PORT"),
        )
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    return DistInfo(rank=rank, world_size=world_size, local_rank=local_rank)


def init_distributed(backend: str | None = None) -> DistInfo:
    """Initialise the process group, auto-detecting the launcher.

    Uses ``torchrun``/env-var placement when present (``RANK`` + ``LOCAL_RANK``),
    otherwise falls back to SLURM ``srun`` placement.

    Args:
        backend: Backend override; defaults to ``"nccl"`` on CUDA else ``"gloo"``.

    Returns:
        The resolved :class:`DistInfo` for this process.
    """
    if "RANK" in os.environ and "LOCAL_RANK" in os.environ:
        return init_distributed_from_torchrun(backend)
    return init_distributed_from_slurm(backend)


def init_distributed_from_torchrun(backend: str | None = None) -> DistInfo:
    """Initialise from ``torchrun``/torchelastic env vars (``RANK``, ``LOCAL_RANK``).

    ``torchrun`` already exports ``MASTER_ADDR``/``MASTER_PORT``, so this only reads
    the placement and starts the group.
    """
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    return _start_process_group(rank, world_size, local_rank, backend)


def init_distributed_from_slurm(
    backend: str | None = None,
    master_port: str | None = None,
) -> DistInfo:
    """Initialise from SLURM env vars (one ``srun`` task per GPU) and pin the GPU.

    Reads ``SLURM_PROCID`` (rank), ``SLURM_NTASKS`` (world size), and
    ``SLURM_LOCALID`` (local rank), and derives ``MASTER_ADDR`` from the node list.

    Args:
        backend: Backend override; defaults to ``"nccl"`` on CUDA else ``"gloo"``.
        master_port: Rendezvous port. Defaults to ``$MASTER_PORT`` or 29500.

    Returns:
        The resolved :class:`DistInfo` for this process.
    """
    rank = int(os.environ.get("SLURM_PROCID", "0"))
    world_size = int(os.environ.get("SLURM_NTASKS", "1"))
    local_rank = int(os.environ.get("SLURM_LOCALID", "0"))

    os.environ.setdefault("MASTER_ADDR", _resolve_master_addr())
    os.environ.setdefault(
        "MASTER_PORT", master_port or os.environ.get("MASTER_PORT", DEFAULT_MASTER_PORT)
    )
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(local_rank)

    return _start_process_group(rank, world_size, local_rank, backend)


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
