#!/usr/bin/env bash
#SBATCH --job-name=prepare_v2
#SBATCH --account=s3292
#SBATCH --qos=allnccs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs_v2/prepare_%A_%a_v2.log
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/gpfsm/dnb10/projects/p311/ML_downscaling}"
ENV_DIR="${ENV_DIR:-/gpfsm/dnb10/projects/p311/ML_downscaling/env}"
# Conda's proj4 deactivation hook reads optional variables. The submitted job
# inherits an active interactive environment, so deactivate it with nounset off.
set +u
source /discover/nobackup/projects/GEOS_MITgcm/afahad/conda/etc/profile.d/conda.sh
conda activate "$ENV_DIR"
set -u
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
: "${PREP_MONTHS:?Submit through submit_pipeline_v2.sh or submit_prepare_flow_v2.sh}"
: "${SLURM_ARRAY_TASK_ID:?Submit as an array job through submit_pipeline_v2.sh or submit_prepare_flow_v2.sh}"
read -r -a prep_months <<< "$PREP_MONTHS"
if (( SLURM_ARRAY_TASK_ID < 1 || SLURM_ARRAY_TASK_ID > ${#prep_months[@]} )); then
  echo 'Preparation task index is outside the submitted month list' >&2
  exit 1
fi
MONTH="${prep_months[SLURM_ARRAY_TASK_ID-1]}"
echo "Preparation task $SLURM_ARRAY_TASK_ID: $MONTH"
python -m merraflow.cli_v2 prepare --config "${CONFIG:-configs/discover_v2.yaml}" --month "$MONTH"
