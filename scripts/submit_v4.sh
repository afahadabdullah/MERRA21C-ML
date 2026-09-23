#!/usr/bin/env bash
# Four-GPU v4 flow; one continuation at a time, no fixed job-chain length.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PROJECT_DIR="${PROJECT_DIR:-$PWD}"
export ENV_DIR="${ENV_DIR:-/gpfsm/dnb10/projects/p311/ML_downscaling/env}"
export CONFIG="${CONFIG:-configs/discover_v4.yaml}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python_bin="${PYTHON_BIN:-$ENV_DIR/bin/python}"
args=(--config "$CONFIG")
if [[ -n "${REGRESSION_CHECKPOINT:-}" ]]; then args+=(--regression-checkpoint "$REGRESSION_CHECKPOINT"); fi
if [[ -n "${HOURLY_TARGETS:-}" ]]; then args+=(--hourly-targets "$HOURLY_TARGETS"); fi
if [[ -n "${RESUME:-}" ]]; then args+=(--resume "$RESUME"); fi
submit=(--parsable --export=ALL)
if [[ -n "${DEPENDENCY:-}" ]]; then
  submit+=(--dependency="afterok:${DEPENDENCY}" --kill-on-invalid-dep=yes)
  echo 'Preflight will run inside the GPU job after its data dependency succeeds.' >&2
else
  "$python_bin" -m merraflow.cli_v4 preflight "${args[@]}"
fi
mkdir -p logs_v4
env -u SLURM_MEM_PER_CPU -u SLURM_MEM_PER_NODE -u SLURM_MEM_PER_GPU \
  sbatch "${submit[@]}" scripts/slurm_train_v4.sh
