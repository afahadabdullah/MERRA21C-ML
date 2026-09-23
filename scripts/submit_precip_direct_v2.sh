#!/usr/bin/env bash
# One flow training job; existing v2 regression is frozen, not retrained.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PROJECT_DIR="${PROJECT_DIR:-$PWD}"
export ENV_DIR="${ENV_DIR:-/gpfsm/dnb10/projects/p311/ML_downscaling/env}"
export CONFIG="${CONFIG:-configs/discover_precip_direct_v2.yaml}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python_bin="${PYTHON_BIN:-$ENV_DIR/bin/python}"
args=(--config "$CONFIG")
if [[ -n "${REGRESSION_CHECKPOINT:-}" ]]; then args+=(--regression-checkpoint "$REGRESSION_CHECKPOINT"); fi
if [[ -n "${RESUME:-}" ]]; then args+=(--resume "$RESUME"); fi
if [[ -n "${INITIALIZE_V2:-}" ]]; then args+=(--initialize-v2 "$INITIALIZE_V2"); fi
"$python_bin" -m merraflow.cli_precip_direct_v2 preflight "${args[@]}"
mkdir -p logs_precip_direct_v2
env -u SLURM_MEM_PER_CPU -u SLURM_MEM_PER_NODE -u SLURM_MEM_PER_GPU \
  sbatch --parsable --export=ALL scripts/slurm_train_precip_direct_v2.sh
