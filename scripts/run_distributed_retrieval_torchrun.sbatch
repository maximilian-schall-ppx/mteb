#!/bin/bash
# Launcher B: one srun task per node, torchrun fans out to all local GPUs.
# Same result as launcher A (one rank per GPU, world_size = nodes x gpus), but you
# only launch one process per node and torchrun spawns the per-GPU workers.
#SBATCH --job-name=mteb-dist-retrieval-torchrun
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --output=mteb-dist-%j.out

set -euo pipefail

MODEL=${MODEL:-intfloat/e5-base-v2}
TASK=${TASK:-NFCorpus}
OUTPUT=${OUTPUT:-./dist_results}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)

# torchrun exports RANK / LOCAL_RANK / WORLD_SIZE / MASTER_ADDR, which
# init_distributed() auto-detects (no SLURM_PROCID needed).
srun torchrun \
    --nnodes="$SLURM_NNODES" \
    --nproc_per_node="$GPUS_PER_NODE" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$MASTER_ADDR:29500" \
    scripts/run_distributed_retrieval.py \
    --model "$MODEL" \
    --task "$TASK" \
    --output "$OUTPUT" \
    --backend gloo \
    --batch-size "${BATCH_SIZE:-32}"
