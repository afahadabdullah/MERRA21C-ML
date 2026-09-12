#!/usr/bin/env bash
#SBATCH --job-name=prepare_v2
#SBATCH --account=s3292
#SBATCH --qos=allnccs
#SBATCH --array=1-12%14
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
printf -v MONTH '%s-%02d' "${PREP_YEAR:-2025}" "$SLURM_ARRAY_TASK_ID"
python -m merraflow.cli_v2 prepare --config "${CONFIG:-configs/discover_v2.yaml}" --month "$MONTH"
