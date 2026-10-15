#!/usr/bin/env bash
#SBATCH --job-name=eval_maps_v41
#SBATCH --account=s3292
#SBATCH --qos=alla100
#SBATCH --partition=gpu_a100
#SBATCH --constraint=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-gpu=110G
#SBATCH --time=12:00:00
#SBATCH --output=logs_v4_1/eval_maps_%A_%a.log
# One shard of the all-hours error-map pass (scripts/evaluate_maps_v4_1.py). Resumable:
# resubmit the same array to finish shards that hit the time limit.
set -euo pipefail
: "${EVAL:?Set EVAL to the frozen evaluation directory}"
: "${MAPS:?Set MAPS to the maps output directory}"
: "${SHARDS:?Set SHARDS to the array task count}"
PROJECT_DIR="${PROJECT_DIR:-/gpfsm/dnb10/projects/p311/ML_downscaling}"
ENV_DIR="${ENV_DIR:-/gpfsm/dnb10/projects/p311/ML_downscaling/env}"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${THREADS:-4}"
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_NODE
srun --cpu-bind=none "$ENV_DIR/bin/python" "${MAPS_SCRIPT:-$PROJECT_DIR/scripts/evaluate_maps_v4_1.py}" run \
  --evaluation "$EVAL" --output "$MAPS" --every "${EVERY:-3}" --members "${MEMBERS:-8}" \
  --shard-index "${SLURM_ARRAY_TASK_ID:-0}" --shards "$SHARDS" \
  --batch "${BATCH:-8}" --threads "${THREADS:-4}" --device cuda
