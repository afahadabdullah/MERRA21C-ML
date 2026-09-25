#!/usr/bin/env bash
# Four-GPU v4.1 training; each job queues its own continuation while epochs remain.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PROJECT_DIR="${PROJECT_DIR:-$PWD}"
export ENV_DIR="${ENV_DIR:-/gpfsm/dnb10/projects/p311/ML_downscaling/env}"
export CONFIG="${CONFIG:-configs/discover_v4_1.yaml}"
submit=(--parsable --export=ALL)
if [[ -n "${DEPENDENCY:-}" ]]; then
  submit+=(--dependency="afterok:${DEPENDENCY}" --kill-on-invalid-dep=yes)
fi
mkdir -p logs_v4_1
env -u SLURM_MEM_PER_CPU -u SLURM_MEM_PER_NODE -u SLURM_MEM_PER_GPU \
  sbatch "${submit[@]}" scripts/slurm_train_v4_1.sh
