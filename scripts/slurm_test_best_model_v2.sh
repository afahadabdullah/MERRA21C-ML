#!/usr/bin/env bash
#SBATCH --job-name=flow_test_v2
#SBATCH --account=s3292
#SBATCH --qos=alla100
#SBATCH --partition=gpu_a100
#SBATCH --constraint=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs_v2/flow_test_%j_v2.log
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
export MPLCONFIGDIR="${TMPDIR:-/tmp}/merraflow-test-v2-${SLURM_JOB_ID}"
# A job submitted from an interactive Slurm allocation can inherit its memory
# request alongside this job's --mem-per-gpu value. srun rejects multiple
# SLURM_MEM_PER_* variables even though sbatch accepted the new allocation.
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_NODE
args=(--config "${CONFIG:-configs/discover_annual_v2.yaml}"
      --split "${SPLIT:-test}" --samples "${SAMPLES:-3}" --members "${MEMBERS:-5}")
if [[ -n "${CHECKPOINT:-}" ]]; then args+=(--checkpoint "$CHECKPOINT"); fi
if [[ -n "${OUTPUT:-}" ]]; then args+=(--output "$OUTPUT"); fi
if [[ -n "${STEPS:-}" ]]; then args+=(--steps "$STEPS"); fi
if [[ -n "${NOISE_PADDING:-}" ]]; then args+=(--noise-padding "$NOISE_PADDING"); fi
case "${COMPARE_NOISE_PADDING:-0}" in
  1) args+=(--compare-noise-padding) ;;
  0) ;;
  *) echo 'COMPARE_NOISE_PADDING must be 0 or 1' >&2; exit 2 ;;
esac
if [[ -n "${INCLUDE_DATE:-}" ]]; then args+=(--include-date "$INCLUDE_DATE"); fi
if [[ -n "${TIMESTAMPS:-}" ]]; then
  read -r -a timestamp_args <<< "$TIMESTAMPS"
  args+=(--timestamps "${timestamp_args[@]}")
fi
srun python scripts/test_best_model_v2.py "${args[@]}"
