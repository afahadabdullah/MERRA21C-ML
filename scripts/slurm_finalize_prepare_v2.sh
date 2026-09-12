#!/usr/bin/env bash
#SBATCH --job-name=finalize_v2
#SBATCH --account=s3292
#SBATCH --qos=allnccs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs_v2/finalize_%j_v2.log
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/gpfsm/dnb10/projects/p311/ML_downscaling}"
ENV_DIR="${ENV_DIR:-/gpfsm/dnb10/projects/p311/ML_downscaling/env}"
# Conda's deactivation hooks are not nounset-safe when an interactive env is inherited.
set +u
source /discover/nobackup/projects/GEOS_MITgcm/afahad/conda/etc/profile.d/conda.sh
conda activate "$ENV_DIR"
set -u
cd "$PROJECT_DIR"
export OMP_NUM_THREADS=1
python -m merraflow.cli_v2 prepare-finalize --config "${CONFIG:-configs/discover_v2.yaml}"
