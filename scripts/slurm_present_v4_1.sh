#!/usr/bin/env bash
#SBATCH --job-name=present_v4_1
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
#SBATCH --output=logs_v4_1/present_%j.log
# Presentation figures (see src/merraflow/present_v4_1.py): per case and variable, CONUS and zoom
# maps (GEOS-FP 0.25° | truth | member | ensemble mean), uncertainty and member-diversity maps,
# reveal stills + member-cycling GIFs, and skill figures over all cases (CRPS and RMSE vs GEOS-FP,
# scorecard, spectra, rank histograms, spread/skill, rain FSS / distribution / Q-Q / reliability).
# Full-domain sampling: ~1.5-2 h for 3 cases x 8 members.
#
# Environment variable overrides:
#   CHECKPOINT=latest (default) | best | <epoch> | <path>   CONFIG=configs/discover_v4_1.yaml  WEIGHTS=ema|raw
#   SPLIT=test (default) | val   SAMPLES=2 (random hours)  WETTEST=1  TIMESTAMPS="20260223_1730 20260115_2330"
#   MEMBERS=8  STEPS=64  ZOOMS=2  ZOOM_SIZE=320  POST=spectral|none  SEED=317
#   ANIMATE="precip t2m wind_speed" (reveal stills + member GIFs; "none" to skip)
#   DPI=300  NO_PDF=1  CARTOPY_DATA_DIR=<Natural Earth cache>  OUTPUT=<dir>
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
export MPLCONFIGDIR="${TMPDIR:-/tmp}/merraflow-present-v4_1-${SLURM_JOB_ID:-local}"
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_NODE
args=(--config "${CONFIG:-configs/discover_v4_1.yaml}" --checkpoint "${CHECKPOINT:-latest}" --split "${SPLIT:-test}"
      --samples "${SAMPLES:-2}" --wettest "${WETTEST:-1}" --members "${MEMBERS:-8}" --steps "${STEPS:-64}"
      --zooms "${ZOOMS:-2}" --zoom-size "${ZOOM_SIZE:-320}" --post "${POST:-spectral}" --dpi "${DPI:-300}")
opt() { if [[ -n "${!1:-}" ]]; then args+=("$2" "${!1}"); fi; }
opt WEIGHTS --weights; opt SEED --seed; opt OUTPUT --output; opt CARTOPY_DATA_DIR --cartopy-data-dir
if [[ -n "${TIMESTAMPS:-}" ]]; then read -r -a stamps <<< "$TIMESTAMPS"; args+=(--timestamps "${stamps[@]}"); fi
if [[ "${ANIMATE:-}" == none ]]; then args+=(--animate); elif [[ -n "${ANIMATE:-}" ]]; then
  read -r -a animate <<< "$ANIMATE"; args+=(--animate "${animate[@]}"); fi
if [[ "${NO_PDF:-0}" == 1 ]]; then args+=(--no-pdf); fi
srun --cpu-bind=none python -m merraflow.present_v4_1 "${args[@]}"
