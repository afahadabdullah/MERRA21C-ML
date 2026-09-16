#!/usr/bin/env bash
#SBATCH --job-name=flow_diag_v2
#SBATCH --account=s3292
#SBATCH --qos=alla100
#SBATCH --partition=gpu_a100
#SBATCH --constraint=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-gpu=48G
#SBATCH --time=12:00:00
#SBATCH --array=0-4%2
#SBATCH --output=logs_v2/flow_diag_%A_%a_v2.log
#SBATCH --error=logs_v2/flow_diag_%A_%a_v2.err
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/gpfsm/dnb10/projects/p311/ML_downscaling}"
ENV_DIR="${ENV_DIR:-/gpfsm/dnb10/projects/p311/ML_downscaling/env}"
set +u
source /discover/nobackup/projects/GEOS_MITgcm/afahad/conda/etc/profile.d/conda.sh
conda activate "$ENV_DIR"
set -u
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/merraflow-diag-v2-${SLURM_JOB_ID}"
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_NODE SLURM_CPUS_PER_GPU
read -r -a steps <<< "${STEPS:-24 48 96}"
read -r -a timestamps <<< "${TIMESTAMPS:-20260223_0530 20260209_1530 20260209_2030 20260305_1230 20260306_1830}"
args=(--config "${CONFIG:-configs/discover_annual_v2.yaml}"
      --checkpoint "${CHECKPOINT:?Set CHECKPOINT explicitly to the Copy1 flow checkpoint}"
      --output "${OUTPUT:?Set OUTPUT to a fresh diagnostic directory ending in v2}"
      --split "${SPLIT:-test}" --members "${MEMBERS:-5}"
      --timestamps "${timestamps[@]}" --steps "${steps[@]}")
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then args+=(--case-index "$SLURM_ARRAY_TASK_ID"); fi
srun python scripts/diagnose_flow_v2.py "${args[@]}"
