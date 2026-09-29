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
#SBATCH --time=03:00:00
#SBATCH --output=logs_v4_1/storm_animation_%j.log
# Hour-by-hour storm GIFs (precip, t2m, 10 m wind speed, surface pressure) on the LCC map with
# coastlines/states: GEOS-FP 0.25° cells | 2 km truth | N members per checkpoint, imshow pixels
# (no interpolation), plus all fields saved as NetCDF (storm_fields.nc). By default the window
# follows the storm's pressure low (shifted north) and members get the spectral fix.
# ~40 min per checkpoint for 24 h x 2 members (64 steps, 394 px window).
#
# Environment variable overrides:
#   TIMESTAMP=<peak hour, e.g. 20260223_1730> (default: wettest hour of SPLIT)  SPLIT=test|val
#   RUNS="main_latest=configs/discover_v4_1.yaml:latest" (default; more label=config:checkpoint
#        entries separated by spaces add one row of members each)
#   TRACK=1 (default)  ZOOM=394 (px)  SHIFT_NORTH=0.25 (fraction of the window)
#   TRACK=0 REGION=768 (fixed window on the rain swath)
#   HOURS=24  MEMBERS=2  STEPS=64  POST=spectral|none  SEEDS=fixed|independent
#   MAPS=1 (Cartopy)  CARTOPY_DATA_DIR=<Natural Earth cache for offline nodes>  NETCDF=1
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
args=(--split "${SPLIT:-test}" --hours "${HOURS:-24}" --members "${MEMBERS:-2}" --steps "${STEPS:-64}"
      --region "${REGION:-768}" --zoom "${ZOOM:-394}" --shift-north "${SHIFT_NORTH:-0.25}"
      --post "${POST:-spectral}" --seeds "${SEEDS:-fixed}")
if [[ "${TRACK:-1}" == 0 ]]; then args+=(--no-track); fi
opt() { if [[ -n "${!1:-}" ]]; then args+=("$2" "${!1}"); fi; }
opt TIMESTAMP --timestamp; opt WEIGHTS --weights; opt FPS --fps; opt DPI --dpi; opt OUTPUT --output
if [[ -n "${RUNS:-}" ]]; then read -r -a runs <<< "$RUNS"; args+=(--runs "${runs[@]}"); fi
if [[ "${FRAMES:-0}" == 1 ]]; then args+=(--frames); fi
if [[ "${MAPS:-1}" == 0 ]]; then args+=(--no-maps); fi
if [[ "${NETCDF:-1}" == 0 ]]; then args+=(--no-netcdf); fi
if [[ -n "${CARTOPY_DATA_DIR:-}" ]]; then args+=(--cartopy-data-dir "$CARTOPY_DATA_DIR"); fi
srun --cpu-bind=none python -m merraflow.storm_animation_v4_1 "${args[@]}"
