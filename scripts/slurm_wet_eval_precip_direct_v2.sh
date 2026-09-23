#!/usr/bin/env bash
#SBATCH --job-name=wet_precip_direct_v2
#SBATCH --account=s3292
#SBATCH --qos=alla100
#SBATCH --partition=gpu_a100
#SBATCH --constraint=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs_precip_direct_v2/wet_%j.log
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/gpfsm/dnb10/projects/p311/ML_downscaling}"
ENV_DIR="${ENV_DIR:-/gpfsm/dnb10/projects/p311/ML_downscaling/env}"
set +u
source /discover/nobackup/projects/GEOS_MITgcm/afahad/conda/etc/profile.d/conda.sh
conda activate "$ENV_DIR"
set -u
cd "$PROJECT_DIR"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/wet-precip-direct-v2-${SLURM_JOB_ID}"
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_NODE
args=(--config "${CONFIG:?}" --checkpoint "${CHECKPOINT:?}" --output "${OUTPUT:?}"
      --split val --scan-hours "${SCAN_HOURS:-64}" --cases "${CASES:-6}"
      --min-wet-fraction "${MIN_WET_FRACTION:-0.1}" --members "${MEMBERS:-4}"
      --steps "${STEPS:-24}")
srun --ntasks=1 --cpu-bind=none python -m merraflow.evaluate_wet_precip_direct_v2 "${args[@]}"
