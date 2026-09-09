#!/usr/bin/env bash
#SBATCH --job-name=merraflow
#SBATCH --account=s3292
#SBATCH --qos=allnccs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/flow_%j.log
#SBATCH --error=logs/flow_%j.err
# Submit from project root after mkdir -p logs.
# Select the current Discover A100 partition/resource name at submission.
# For four GPUs: sbatch --gres=gpu:4 --cpus-per-task=16 scripts/slurm_train_flow.sh
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/gpfsm/dnb10/projects/p311/ML_downscaling}"
ENV_DIR="${ENV_DIR:-/discover/nobackup/projects/GEOS_MITgcm/afahad/conda/envs/myenv}"
source /discover/nobackup/projects/GEOS_MITgcm/afahad/conda/etc/profile.d/conda.sh
conda activate "$ENV_DIR"
cd "$PROJECT_DIR"
export OMP_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/merraflow-matplotlib-${SLURM_JOB_ID}"
CONFIG="${CONFIG:-configs/discover.yaml}"
NPROC=$(python -c 'import torch; print(torch.cuda.device_count())')
if (( NPROC < 1 )); then
  echo 'No CUDA GPUs visible in this allocation' >&2
  exit 1
fi
args=(--config "$CONFIG")
if [[ -n "${RESUME:-}" ]]; then args+=(--resume "$RESUME"); fi
srun torchrun --standalone --nnodes=1 --nproc-per-node="$NPROC" -m merraflow.cli train "${args[@]}"
