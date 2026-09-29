#!/usr/bin/env bash
#SBATCH --job-name=ckpt_v4_1
#SBATCH --account=s3292
#SBATCH --qos=alla100
#SBATCH --partition=gpu_a100
#SBATCH --constraint=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-gpu=110G
#SBATCH --time=02:00:00
#SBATCH --output=logs_v4_1/compare_checkpoints_%j.log
# Same case, region, member noise and recipes (default baseline + spectral) for several
# checkpoints; one ranked table (CRPS, RMSE, MAE, bias, spread/skill, sharpness, edge step).
# ~10 min per checkpoint with 8 members; see src/merraflow/compare_checkpoints_v4_1.py.
#
# Environment variable overrides:
#   RUNS="main_best=configs/discover_v4_1.yaml:best main_latest=configs/discover_v4_1.yaml:latest
#         ft1_best=configs/discover_v4_1_ft_rollout.yaml:best ft2_best=configs/discover_v4_1_ft_rollout2.yaml:best"
#         (default; label=config:checkpoint, checkpoint = best | latest | <epoch> | <path>)
#   RECIPES=baseline,spectral  TIMESTAMP=20260115_2330  SPLIT=val  CENTER="ROW COL"  CENTER_LATLON="LAT LON"
#   MEMBERS=8  STEPS=64  REGION=384  SPECTRAL_MAX_GAIN=1.3  WEIGHTS=ema|raw  DPI=200  NO_PDF=1  OUTPUT=<dir>
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
export MPLCONFIGDIR="${TMPDIR:-/tmp}/merraflow-ckpt-v4_1-${SLURM_JOB_ID:-local}"
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_NODE
args=(--split "${SPLIT:-val}" --members "${MEMBERS:-8}" --recipes "${RECIPES:-baseline,spectral}")
opt() { if [[ -n "${!1:-}" ]]; then args+=("$2" "${!1}"); fi; }
multi() { if [[ -n "${!1:-}" ]]; then read -r -a values <<< "${!1}"; args+=("$2" "${values[@]}"); fi; }
opt TIMESTAMP --timestamp; opt WEIGHTS --weights; opt STEPS --steps; opt REGION --region
opt SPECTRAL_MAX_GAIN --spectral-max-gain; opt DPI --dpi; opt OUTPUT --output
multi CENTER --center; multi CENTER_LATLON --center-latlon; multi RUNS --runs
if [[ "${NO_PDF:-0}" == 1 ]]; then args+=(--no-pdf); fi
srun --cpu-bind=none python -m merraflow.compare_checkpoints_v4_1 "${args[@]}"
