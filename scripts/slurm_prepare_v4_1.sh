#!/usr/bin/env bash
#SBATCH --job-name=pack_v4_1
#SBATCH --account=s3292
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs_v4_1/pack_%A_%a.log
# One-time CPU packing of the v4.1 training archive (restart-safe; resubmit
# the same month to finish it). STAGE=finalize writes the manifest and then
# runs the data-only preflight here, so training jobs start without checks.
set -euo pipefail
cd "${PROJECT_DIR:-/gpfsm/dnb10/projects/p311/ML_downscaling}"
ENV_DIR="${ENV_DIR:-/gpfsm/dnb10/projects/p311/ML_downscaling/env}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
args=(--config "${CONFIG:-configs/discover_v4_1.yaml}")
if [[ "${STAGE:-prepare}" == finalize ]]; then
  "$ENV_DIR/bin/python" -m merraflow.cli_v4_1 finalize-packed "${args[@]}"
  "$ENV_DIR/bin/python" -m merraflow.cli_v4_1 preflight "${args[@]}" --data-only
else
  read -r -a months <<< "${MONTHS:?MONTHS is required}"
  "$ENV_DIR/bin/python" -m merraflow.cli_v4_1 prepare-packed "${args[@]}" --month "${months[$SLURM_ARRAY_TASK_ID]}"
fi
