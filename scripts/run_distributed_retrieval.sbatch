#!/bin/bash
#SBATCH --job-name=mteb-dist-retrieval
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=0
#SBATCH --output=mteb-dist-%j.out

# --mem=0 requests all node memory: with 8 ranks/node each materialising a
# corpus shard, a low memory limit OOM-kills ranks on large tasks (e.g. MS MARCO).

# Multi-node evaluation of a single MTEB retrieval task.
# One srun task per GPU; each becomes one torch.distributed rank.
# init_distributed_from_slurm() reads SLURM_PROCID / SLURM_NTASKS / SLURM_LOCALID
# and derives MASTER_ADDR from the allocation node list.

set -euo pipefail

MODEL=${MODEL:-intfloat/e5-base-v2}
TASK=${TASK:-NFCorpus}
OUTPUT=${OUTPUT:-./dist_results}

srun python scripts/run_distributed_retrieval.py \
    --model "$MODEL" \
    --task "$TASK" \
    --output "$OUTPUT" \
    --batch-size "${BATCH_SIZE:-32}"
