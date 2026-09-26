#!/usr/bin/env bash
#SBATCH --job-name=sharp_v4_1
#SBATCH --account=s3292
#SBATCH --qos=alla100
#SBATCH --partition=gpu_a100
#SBATCH --constraint=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-gpu=110G
#SBATCH --time=03:00:00
#SBATCH --output=logs_v4_1/sharpness_%j.log
# v4.1 inference sharpness ablation on one A100.
# Compares 5 inference sharpening methods side-by-side with Ground Truth and Coarse inputs:
#   0. Baseline v4.1
#   1. Time-Step Warping (gamma=1.5)
#   2. Latent Residual Scaling (alpha=1.10)
#   3. Wet-Cutoff Thresholding (<0.1 mm/h -> 0)
#   4. Tukey Window Blending (alpha=0.3)
#   5. Combined Sharpness (all 4 methods together)
#
# Environment variable overrides:
#   CHECKPOINT=best (default) | latest | <epoch number> | <path>
#   SPLIT=test|val  SAMPLES=0  WETTEST=1  STEPS=<inference.steps>
#   WARP_GAMMA=1.5  RESIDUAL_SCALE=1.10  DRY_CUTOFF=0.1  TUKEY_ALPHA=0.3
#   TIMESTAMPS="20260223_0530"  OUTPUT=<fresh dir>
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
export MPLCONFIGDIR="${TMPDIR:-/tmp}/merraflow-sharpness-v4_1-${SLURM_JOB_ID:-local}"
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_NODE

args=(--config "${CONFIG:-configs/discover_v4_1.yaml}" --checkpoint "${CHECKPOINT:-best}"
      --split "${SPLIT:-test}" --samples "${SAMPLES:-0}" --wettest "${WETTEST:-1}")

if [[ -n "${STEPS:-}" ]]; then args+=(--steps "$STEPS"); fi
if [[ -n "${WARP_GAMMA:-}" ]]; then args+=(--warp-gamma "$WARP_GAMMA"); fi
if [[ -n "${RESIDUAL_SCALE:-}" ]]; then args+=(--residual-scale "$RESIDUAL_SCALE"); fi
if [[ -n "${DRY_CUTOFF:-}" ]]; then args+=(--dry-cutoff "$DRY_CUTOFF"); fi
if [[ -n "${TUKEY_ALPHA:-}" ]]; then args+=(--tukey-alpha "$TUKEY_ALPHA"); fi
if [[ -n "${OUTPUT:-}" ]]; then args+=(--output "$OUTPUT"); fi
if [[ -n "${CARTOPY_DATA_DIR:-}" ]]; then args+=(--cartopy-data-dir "$CARTOPY_DATA_DIR"); fi
if [[ -n "${TIMESTAMPS:-}" ]]; then
  read -r -a timestamp_args <<< "$TIMESTAMPS"
  args+=(--timestamps "${timestamp_args[@]}")
fi

srun --cpu-bind=none python -m merraflow.compare_sharpness_v4_1 "${args[@]}"
