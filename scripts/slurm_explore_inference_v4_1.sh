#!/usr/bin/env bash
#SBATCH --job-name=explore_v4_1
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
#SBATCH --output=logs_v4_1/explore_inference_%j.log
# Single-case search for the best inference recipe (sampling, tiling, guidance,
# member selection, post-processing) for crisp, calibrated members; see
# src/merraflow/explore_inference_v4_1.py for every recipe. Samples only the tiles
# around the front (REGION + MARGIN), all recipes from the same member noise.
# All recipes with 8 members (pool 16) ~ 2-4 h on one A100; use METHODS to run fewer.
#
# Environment variable overrides:
#   CHECKPOINT=latest (default) | best | <epoch> | <path>   CONFIG=configs/discover_v4_1.yaml
#   WEIGHTS=ema (default) | raw   SPLIT=val (default) | test   TIMESTAMP=20260115_2330
#   CENTER="ROW COL" or CENTER_LATLON="LAT LON"   ANYWHERE=1   REGION=384  MARGIN=96
#   MEMBERS=8  POOL=16  STEPS=64
#   METHODS=baseline,hann2,hann3,hard,shift_hard,shift_hann2,shift4_hard,time_warp,churn,langevin,sde,
#           restart,restart_shift,temp,hf_boost,autoguide,autoguide_hf,vguide,fk_steer,fk_edge,
#           select_clim,select_sharp,prescreen,spectral,vpost   (default: all)
#   VGUIDE_STRENGTHS="0.25 0.5 1"  VGUIDE_FIELDS="t2m q2m"  SDE_STRENGTHS="0.5 1 2"
#   FK_PARTICLES=4  FK_LAMBDA=10  LANGEVIN=0.3  CHURN=0.2  RESTART=2  RESTART_T=0.7  TEMP=1.1
#   GUIDE_CHECKPOINT=auto  GUIDE_WEIGHT=1.5   CLIM_COUNT=24  CLIM_DAYS=45
#   COMBINE="autoguide_hf+spectral,autoguide_hf+fk_steer+spectral"  (recipes joined by +; comma list)
#   Strength suffixes: fk_steer@4 / fk_edge@10 (FK lambda), vguide_0.5, sde_1, vpost_0.25
#   PHASE2=0 (skip the automatic best-sampler + selection/spectral combinations)  SPECTRAL_MAX_GAIN=1.5
#   CRPS_TOL=1  CRPS_TOL_MAX=3  BIAS_TOL=0.02   SAVE_MEMBERS=1  DPI=200  NO_PDF=1  OUTPUT=<dir>
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
export MPLCONFIGDIR="${TMPDIR:-/tmp}/merraflow-explore-v4_1-${SLURM_JOB_ID:-local}"
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_NODE
args=(--config "${CONFIG:-configs/discover_v4_1.yaml}" --checkpoint "${CHECKPOINT:-latest}"
      --split "${SPLIT:-val}" --members "${MEMBERS:-8}" --pool "${POOL:-16}")
opt() { if [[ -n "${!1:-}" ]]; then args+=("$2" "${!1}"); fi; }
multi() { if [[ -n "${!1:-}" ]]; then read -r -a values <<< "${!1}"; args+=("$2" "${values[@]}"); fi; }
opt WEIGHTS --weights; opt TIMESTAMP --timestamp; opt STEPS --steps; opt METHODS --methods
opt REGION --region; opt MARGIN --margin; opt CLIM_COUNT --clim-count; opt CLIM_DAYS --clim-days
opt FK_PARTICLES --fk-particles; opt FK_LAMBDA --fk-lambda; opt LANGEVIN --langevin; opt CHURN --churn
opt RESTART --restart; opt RESTART_T --restart-t; opt TEMP --temp
opt GUIDE_CHECKPOINT --guide-checkpoint; opt GUIDE_WEIGHT --guide-weight
opt CRPS_TOL --crps-tol; opt CRPS_TOL_MAX --crps-tol-max; opt BIAS_TOL --bias-tol
opt DPI --dpi; opt OUTPUT --output; opt COMBINE --combine; opt SPECTRAL_MAX_GAIN --spectral-max-gain
multi CENTER --center; multi CENTER_LATLON --center-latlon
multi VGUIDE_STRENGTHS --vguide-strengths; multi VGUIDE_FIELDS --vguide-fields; multi SDE_STRENGTHS --sde-strengths
if [[ "${ANYWHERE:-0}" == 1 ]]; then args+=(--anywhere); fi
if [[ "${PHASE2:-1}" == 0 ]]; then args+=(--no-phase2); fi
if [[ "${SAVE_MEMBERS:-0}" == 1 ]]; then args+=(--save-members); fi
if [[ "${NO_PDF:-0}" == 1 ]]; then args+=(--no-pdf); fi
srun --cpu-bind=none python -m merraflow.explore_inference_v4_1 "${args[@]}"
