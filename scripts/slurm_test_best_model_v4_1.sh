#!/usr/bin/env bash
#SBATCH --job-name=eval_v4_1
#SBATCH --account=s3292
#SBATCH --qos=alla100
#SBATCH --partition=gpu_a100
#SBATCH --constraint=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-gpu=110G
#SBATCH --time=06:00:00
#SBATCH --output=logs_v4_1/eval_%j.log
# v4.1 held-out evaluation on one A100 (safe to run while training uses others).
#   CHECKPOINT=best (default) | latest | <epoch number> | <path>
#   SPLIT=test|val  SAMPLES=3  WETTEST=1  MEMBERS=8  ZOOMS=2  STEPS=<inference.steps>
#   TIMESTAMPS="20260223_0530 20260110_1230"  OUTPUT=<fresh dir>  SAVE_FIELDS=1
#   CARTOPY_DATA_DIR=<Natural Earth cache> for coastlines/states on offline nodes
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/gpfsm/dnb10/projects/p311/ML_downscaling}"
ENV_DIR="${ENV_DIR:-/gpfsm/dnb10/projects/p311/ML_downscaling/env}"
set +u
source /discover/nobackup/projects/GEOS_MITgcm/afahad/conda/etc/profile.d/conda.sh
conda activate "$ENV_DIR"
set -u
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4
export MPLCONFIGDIR="${TMPDIR:-/tmp}/merraflow-eval-v4_1-${SLURM_JOB_ID:-local}"
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_NODE
args=(--config "${CONFIG:-configs/discover_v4_1.yaml}" --checkpoint "${CHECKPOINT:-best}"
      --split "${SPLIT:-test}" --samples "${SAMPLES:-3}" --wettest "${WETTEST:-1}"
      --members "${MEMBERS:-8}" --zooms "${ZOOMS:-2}")
if [[ -n "${STEPS:-}" ]]; then args+=(--steps "$STEPS"); fi
if [[ -n "${OUTPUT:-}" ]]; then args+=(--output "$OUTPUT"); fi
if [[ -n "${CARTOPY_DATA_DIR:-}" ]]; then args+=(--cartopy-data-dir "$CARTOPY_DATA_DIR"); fi
if [[ "${SAVE_FIELDS:-0}" == 1 ]]; then args+=(--save-fields); fi
if [[ -n "${TIMESTAMPS:-}" ]]; then
  read -r -a timestamp_args <<< "$TIMESTAMPS"
  args+=(--timestamps "${timestamp_args[@]}")
fi
srun --cpu-bind=none python -m merraflow.evaluate_v4_1 "${args[@]}"
