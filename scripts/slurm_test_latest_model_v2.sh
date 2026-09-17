#!/usr/bin/env bash
#SBATCH --job-name=flow_latest_test_v2
#SBATCH --account=s3292
#SBATCH --qos=alla100
#SBATCH --partition=gpu_a100
#SBATCH --constraint=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs_v2/flow_latest_test_%j_v2.log
#SBATCH --error=logs_v2/flow_latest_test_%j_v2.err
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/gpfsm/dnb10/projects/p311/ML_downscaling}"
ENV_DIR="${ENV_DIR:-$PROJECT_DIR/env}"
set +u
source /discover/nobackup/projects/GEOS_MITgcm/afahad/conda/etc/profile.d/conda.sh
conda activate "$ENV_DIR"
set -u
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_NODE
if [[ -z "${CHECKPOINT:-}" || -z "${OUTPUT:-}" ]]; then
  echo 'CHECKPOINT and OUTPUT must point to the frozen checkpoint and a fresh output directory' >&2
  exit 2
fi
args=(--config "${CONFIG:-configs/discover_rain_structure_v2.yaml}"
      --checkpoint "$CHECKPOINT" --output "$OUTPUT" --members "${MEMBERS:-10}"
      --timestamps 20260223_0530 20260209_1530 20260209_2030 20260305_1230 20260306_1830)
if [[ -n "${STEPS:-}" ]]; then args+=(--steps "$STEPS"); fi
srun python scripts/test_parallel_model_v2.py "${args[@]}"
