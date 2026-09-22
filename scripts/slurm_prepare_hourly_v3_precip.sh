#!/usr/bin/env bash
# [fix 6] One month of trapezoid hourly-mean HWT precipitation targets (CPU array task),
# or the finalizer when FINALIZE=1. Submit through submit_prepare_hourly_v3_precip.sh.
#SBATCH --job-name=hourly_v3_precip
#SBATCH --account=s3292
#SBATCH --qos=allnccs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs_v3_precip/hourly_%A_%a.log
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
CONFIG="${CONFIG:-configs/discover_v3_precip.yaml}"
if [[ "${FINALIZE:-0}" == 1 ]]; then
  python -m merraflow.cli_v3_precip finalize-hourly --config "$CONFIG"
  exit 0
fi
: "${HOURLY_MONTHS:?Submit through scripts/submit_prepare_hourly_v3_precip.sh}"
: "${SLURM_ARRAY_TASK_ID:?Submit as an array job}"
read -r -a months <<< "$HOURLY_MONTHS"
if (( SLURM_ARRAY_TASK_ID < 1 || SLURM_ARRAY_TASK_ID > ${#months[@]} )); then
  echo 'Task index is outside the submitted month list' >&2
  exit 1
fi
MONTH="${months[SLURM_ARRAY_TASK_ID-1]}"
echo "Hourly-target task $SLURM_ARRAY_TASK_ID: $MONTH"
python -m merraflow.cli_v3_precip prepare-hourly --config "$CONFIG" --month "$MONTH"
