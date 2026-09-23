#!/usr/bin/env bash
#SBATCH --job-name=humidity_v4
#SBATCH --account=s3292
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs_v4/humidity_%A_%a.log
set -euo pipefail
cd "${PROJECT_DIR:-/gpfsm/dnb10/projects/p311/ML_downscaling}"
ENV_DIR="${ENV_DIR:-/gpfsm/dnb10/projects/p311/ML_downscaling/env}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
args=(--config "${CONFIG:-configs/discover_v4.yaml}")
if [[ "${STAGE:-prepare}" == finalize ]]; then
  "$ENV_DIR/bin/python" -m merraflow.cli_v4 finalize-humidity "${args[@]}"
else
  read -r -a months <<< "${MONTHS:?MONTHS is required}"
  "$ENV_DIR/bin/python" -m merraflow.cli_v4 prepare-humidity "${args[@]}" --month "${months[$SLURM_ARRAY_TASK_ID]}"
fi
