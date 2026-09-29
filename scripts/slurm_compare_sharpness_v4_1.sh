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
# v4.1 inference-time sharpening ablation on one A100 (no retraining).
# Same case(s), same member noise, every method; see compare_sharpness_v4_1.py.
#   baseline      v4 inference (uniform Heun, Hann blending)
#   more_steps    2x Heun steps (control: does discretization limit sharpness?)
#   time_warp     finer ODE steps near the data end (gamma)
#   churn         EDM-style stochastic re-noising between Heun steps
#   autoguide     guidance by an earlier kept checkpoint of this run (Karras et al. 2024)
#   residual_scale  amplify departures from the frozen regression (rain in sqrt space)
#   wet_cutoff    drizzle below DRY_CUTOFF mm/h set to 0
#   tukey_window  flat-top tile blending (ablation)
#   hann2_window / hann3_window  centre-weighted (Hann², Hann³) tile blending
#   combined      the COMBINE methods together
# Cost per member ≈ 10 baseline solves with all methods (post-processing variants
# reuse the baseline solve; more_steps and autoguide cost 2x). ~1 wettest case x
# 4 members fits the 3 h limit comfortably.
#
# Environment variable overrides:
#   WEIGHTS=ema (default) | raw   (raw = optimizer weights stored in the checkpoint)
#   CHECKPOINT=best (default) | latest | <epoch number> | <path>
#   SPLIT=test|val  SAMPLES=0  WETTEST=1  MEMBERS=4  STEPS=<inference.steps>
#   METHODS=baseline,more_steps,time_warp,churn,autoguide,residual_scale,wet_cutoff,tukey_window,hann2_window,hann3_window,combined
#   COMBINE=churn,autoguide,residual_scale
#   WARP_GAMMA=1.5  CHURN=0.1  CHURN_RANGE="0.1 0.8"
#   GUIDE_CHECKPOINT=auto (kept epoch nearest 1/3 of the main one) | <epoch> | <path>  GUIDE_WEIGHT=1.5
#   RESIDUAL_SCALE=1.10  DRY_CUTOFF=0.1  TUKEY_ALPHA=0.3
#   DPI=300  NO_PDF=0 (every figure as PNG + PDF)
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
      --split "${SPLIT:-test}" --samples "${SAMPLES:-0}" --wettest "${WETTEST:-1}" --members "${MEMBERS:-4}")

if [[ -n "${WEIGHTS:-}" ]]; then args+=(--weights "$WEIGHTS"); fi
if [[ -n "${STEPS:-}" ]]; then args+=(--steps "$STEPS"); fi
if [[ -n "${METHODS:-}" ]]; then args+=(--methods "$METHODS"); fi
if [[ -n "${COMBINE:-}" ]]; then args+=(--combine "$COMBINE"); fi
if [[ -n "${WARP_GAMMA:-}" ]]; then args+=(--warp-gamma "$WARP_GAMMA"); fi
if [[ -n "${CHURN:-}" ]]; then args+=(--churn "$CHURN"); fi
if [[ -n "${CHURN_RANGE:-}" ]]; then read -r -a churn_range <<< "$CHURN_RANGE"; args+=(--churn-range "${churn_range[@]}"); fi
if [[ -n "${GUIDE_CHECKPOINT:-}" ]]; then args+=(--guide-checkpoint "$GUIDE_CHECKPOINT"); fi
if [[ -n "${GUIDE_WEIGHT:-}" ]]; then args+=(--guide-weight "$GUIDE_WEIGHT"); fi
if [[ -n "${RESIDUAL_SCALE:-}" ]]; then args+=(--residual-scale "$RESIDUAL_SCALE"); fi
if [[ -n "${DRY_CUTOFF:-}" ]]; then args+=(--dry-cutoff "$DRY_CUTOFF"); fi
if [[ -n "${TUKEY_ALPHA:-}" ]]; then args+=(--tukey-alpha "$TUKEY_ALPHA"); fi
if [[ -n "${DPI:-}" ]]; then args+=(--dpi "$DPI"); fi
if [[ "${NO_PDF:-0}" == 1 ]]; then args+=(--no-pdf); fi
if [[ -n "${OUTPUT:-}" ]]; then args+=(--output "$OUTPUT"); fi
if [[ -n "${CARTOPY_DATA_DIR:-}" ]]; then args+=(--cartopy-data-dir "$CARTOPY_DATA_DIR"); fi
if [[ -n "${TIMESTAMPS:-}" ]]; then
  read -r -a timestamp_args <<< "$TIMESTAMPS"
  args+=(--timestamps "${timestamp_args[@]}")
fi

srun --cpu-bind=none python -m merraflow.compare_sharpness_v4_1 "${args[@]}"
