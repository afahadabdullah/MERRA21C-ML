#!/usr/bin/env bash
#SBATCH --job-name=front_v4_1
#SBATCH --account=s3292
#SBATCH --qos=alla100
#SBATCH --partition=gpu_a100
#SBATCH --constraint=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-gpu=110G
#SBATCH --time=01:30:00
#SBATCH --output=logs_v4_1/front_diag_%j.log
# Front-sharpness diagnostic on one case (one A100, ~20-30 min with 4 members):
# full-domain tiled sampling at 32 and 64 steps vs the front's tile sampled alone
# (no blending, 64 steps). Answers: is the front blur from the tiling or the model?
#   CHECKPOINT=latest (default) | best | <epoch> | <path>   CONFIG=configs/discover_v4_1.yaml
#   TIMESTAMP=20260307_1930 (default: wettest hour)  SPLIT=test|val  MEMBERS=4
#   CENTER="ROW COL" (front point on the full grid; default: strongest truth q2m+t2m front)
#   DPI=200  NO_PDF=1  OUTPUT=<dir>
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
export MPLCONFIGDIR="${TMPDIR:-/tmp}/merraflow-front-v4_1-${SLURM_JOB_ID:-local}"
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_NODE
args=(--config "${CONFIG:-configs/discover_v4_1.yaml}" --checkpoint "${CHECKPOINT:-latest}"
      --split "${SPLIT:-test}" --members "${MEMBERS:-4}")
if [[ -n "${TIMESTAMP:-}" ]]; then args+=(--timestamp "$TIMESTAMP"); fi
if [[ -n "${CENTER:-}" ]]; then read -r -a center <<< "$CENTER"; args+=(--center "${center[@]}"); fi
if [[ -n "${DPI:-}" ]]; then args+=(--dpi "$DPI"); fi
if [[ "${NO_PDF:-0}" == 1 ]]; then args+=(--no-pdf); fi
if [[ -n "${OUTPUT:-}" ]]; then args+=(--output "$OUTPUT"); fi
srun --cpu-bind=none python -m merraflow.diag_front_v4_1 "${args[@]}"
