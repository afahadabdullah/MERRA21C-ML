#!/usr/bin/env bash
#SBATCH --job-name=rain_edges_v2
#SBATCH --account=s3292
#SBATCH --qos=allnccs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs_v2/rain_edges_%j_v2.log
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
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU SLURM_CPUS_PER_GPU
srun python scripts/build_proposal_cache_v2.py \
  --config "${CONFIG:-configs/discover_rain_edges_v2.yaml}" \
  --kind structure --workers "${SLURM_CPUS_PER_TASK:-4}"
