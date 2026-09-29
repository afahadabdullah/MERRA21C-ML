#!/usr/bin/env bash
#SBATCH --job-name=storm_v4_1
#SBATCH --account=s3292
#SBATCH --qos=alla100
#SBATCH --partition=gpu_a100
#SBATCH --constraint=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-gpu=110G
#SBATCH --time=04:00:00
#SBATCH --output=logs_v4_1/storm_animation_%j.log
# Hour-by-hour storm GIFs (precip, t2m, 10 m wind speed, sea-level pressure): GEOS-FP 0.25°
# cells | 2 km truth | N members per checkpoint, imshow pixels (no interpolation). By default
# the window follows the storm (truth SLP minimum) and members get the spectral fix.
# ~1 h per checkpoint for 24 h x 4 members (64 steps, 512 px window); STEPS=32 halves it.
#
# Environment variable overrides:
#   TIMESTAMP=<peak hour, e.g. 20260223_1730> (default: wettest hour of SPLIT)  SPLIT=test|val
#   RUNS="main_latest=configs/discover_v4_1.yaml:latest" (default; more label=config:checkpoint
#        entries separated by spaces add one row of members each)
#   TRACK=1 (default: follow the low)  ZOOM=512 (px)   TRACK=0 REGION=768 (fixed window on the rain swath)
#   HOURS=24  MEMBERS=4  STEPS=64  POST=spectral|none  SEEDS=fixed|independent
#   FPS=2  DPI=90  FRAMES=1 (also every frame as PNG)  WEIGHTS=ema|raw  OUTPUT=<dir>
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
export MPLCONFIGDIR="${TMPDIR:-/tmp}/merraflow-storm-v4_1-${SLURM_JOB_ID:-local}"
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_NODE
args=(--split "${SPLIT:-test}" --hours "${HOURS:-24}" --members "${MEMBERS:-4}" --steps "${STEPS:-64}"
      --region "${REGION:-768}" --zoom "${ZOOM:-512}" --post "${POST:-spectral}" --seeds "${SEEDS:-fixed}")
if [[ "${TRACK:-1}" == 0 ]]; then args+=(--no-track); fi
opt() { if [[ -n "${!1:-}" ]]; then args+=("$2" "${!1}"); fi; }
opt TIMESTAMP --timestamp; opt WEIGHTS --weights; opt FPS --fps; opt DPI --dpi; opt OUTPUT --output
if [[ -n "${RUNS:-}" ]]; then read -r -a runs <<< "$RUNS"; args+=(--runs "${runs[@]}"); fi
if [[ "${FRAMES:-0}" == 1 ]]; then args+=(--frames); fi
srun --cpu-bind=none python -m merraflow.storm_animation_v4_1 "${args[@]}"
