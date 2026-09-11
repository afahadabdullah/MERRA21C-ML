#!/usr/bin/env bash
#SBATCH --job-name=flow_test
#SBATCH --account=s3292
#SBATCH --qos=alla100
#SBATCH --partition=gpu_a100
#SBATCH --constraint=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/flow_test_%j.log
#SBATCH --error=logs/flow_test_%j.err
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/gpfsm/dnb10/projects/p311/ML_downscaling}"
ENV_DIR="${ENV_DIR:-/gpfsm/dnb10/projects/p311/ML_downscaling/env}"
# Conda's deactivation hooks are not nounset-safe when an interactive env is inherited.
set +u
source /discover/nobackup/projects/GEOS_MITgcm/afahad/conda/etc/profile.d/conda.sh
conda activate "$ENV_DIR"
set -u
cd "$PROJECT_DIR"
export OMP_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/merraflow-test-matplotlib-${SLURM_JOB_ID}"
extra_args=()
if [[ -n "${CHECKPOINT:-}" ]]; then extra_args+=(--checkpoint "$CHECKPOINT"); fi
if [[ -n "${OUTPUT:-}" ]]; then extra_args+=(--output "$OUTPUT"); fi
srun python scripts/test_best_model.py \
  --config "${CONFIG:-configs/discover.yaml}" \
  --samples "${SAMPLES:-3}" \
  --members "${MEMBERS:-5}" "${extra_args[@]}"
