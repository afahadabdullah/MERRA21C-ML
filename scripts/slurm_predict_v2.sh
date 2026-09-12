#!/usr/bin/env bash
#SBATCH --job-name=predict_v2
#SBATCH --account=s3292
#SBATCH --qos=alla100
#SBATCH --partition=gpu_a100
#SBATCH --constraint=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs_v2/predict_%j_v2.log
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
export MPLCONFIGDIR="${TMPDIR:-/tmp}/merraflow-mpl-v2-${SLURM_JOB_ID}"
CONFIG="${CONFIG:-configs/discover_v2.yaml}"
args=(--config "$CONFIG" --checkpoint "${CHECKPOINT:?Set CHECKPOINT to a v2 checkpoint}" --split "${SPLIT:-val}")
if [[ -n "${LIMIT:-}" ]]; then args+=(--limit "$LIMIT"); fi
if [[ -n "${TIMESTAMP:-}" ]]; then args+=(--timestamp "$TIMESTAMP"); fi
srun python -m merraflow.cli_v2 predict "${args[@]}"
python -m merraflow.cli_v2 evaluate --config "$CONFIG" --split "${SPLIT:-val}"
python -m merraflow.cli_v2 plot --config "$CONFIG" --split "${SPLIT:-val}"
