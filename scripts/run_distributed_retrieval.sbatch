#!/bin/bash
# Launcher A: one srun task per GPU (each task = one torch.distributed rank).
# Multi-node evaluation of a single MTEB retrieval task.
#SBATCH --job-name=mteb-dist-retrieval
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=0
#SBATCH --output=mteb-dist-%j.out

# --mem=0 requests all node memory: with 8 ranks/node each materialising a
# corpus shard, a low memory limit OOM-kills ranks on large tasks (e.g. MS MARCO).

set -euo pipefail

MODEL=${MODEL:-intfloat/e5-base-v2}
TASK=${TASK:-NFCorpus}
OUTPUT=${OUTPUT:-./dist_results}

srun python scripts/run_distributed_retrieval.py \
    --model "$MODEL" \
    --task "$TASK" \
    --output "$OUTPUT" \
    --backend gloo \
    --batch-size "${BATCH_SIZE:-32}"
