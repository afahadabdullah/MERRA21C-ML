#!/usr/bin/env bash
#SBATCH --job-name=eval_all_v41
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
#SBATCH --output=logs_v4_1/eval_all_%A_%a.log
# Submit through submit_evaluate_all_v4_1.sh after initializing a frozen run.
set -euo pipefail
: "${OUTPUT:?Set OUTPUT to an initialized evaluation directory}"
: "${SHARDS:?Set SHARDS to the array task count}"
PROJECT_DIR="${PROJECT_DIR:-/gpfsm/dnb10/projects/p311/ML_downscaling}"
ENV_DIR="${ENV_DIR:-/gpfsm/dnb10/projects/p311/ML_downscaling/env}"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${THREADS:-4}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/merraflow-all-${SLURM_JOB_ID:-local}"
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_NODE
srun --cpu-bind=none "$ENV_DIR/bin/python" -m merraflow.evaluate_all_v4_1 run --output "$OUTPUT" \
  --shard-index "${SLURM_ARRAY_TASK_ID:-0}" --shards "$SHARDS" \
  --batch "${BATCH:-8}" --threads "${THREADS:-4}" --device cuda
