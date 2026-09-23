#!/usr/bin/env bash
#SBATCH --job-name=full_precip_direct_v2
#SBATCH --account=s3292
#SBATCH --qos=alla100
#SBATCH --partition=gpu_a100
#SBATCH --constraint=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs_precip_direct_v2/full_%j.log
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
export MPLCONFIGDIR="${TMPDIR:-/tmp}/full-precip-direct-${SLURM_JOB_ID}"
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_NODE SLURM_MEM_PER_GPU
args=(--config "${CONFIG:?}" --checkpoint "${CHECKPOINT:?}" --output "${OUTPUT:?}"
      --timestamp "${TIMESTAMP:-20260223_0530}" --split "${SPLIT:-test}"
      --members "${MEMBERS:-2}")
if [[ -n "${STEPS:-}" ]]; then args+=(--steps "$STEPS"); fi
srun --ntasks=1 --cpu-bind=none python -m merraflow.evaluate_full_precip_direct_v2 "${args[@]}"
