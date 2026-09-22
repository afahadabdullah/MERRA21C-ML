#!/usr/bin/env bash
# Submit both four-GPU stages; diffusion starts only after regression succeeds.
# Run in the project environment from the repository root.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PROJECT_DIR="${PROJECT_DIR:-$PWD}"
export CONFIG="${CONFIG:-configs/discover_v3_precip.yaml}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
PREPARE_FIRST="${PREPARE_FIRST:-0}"
AFTEROK_JOB="${AFTEROK_JOB:-}"
REGRESSION_SEGMENTS="${REGRESSION_SEGMENTS:-2}"
DIFFUSION_SEGMENTS="${DIFFUSION_SEGMENTS:-8}"
if [[ "$PREPARE_FIRST" != 0 && "$PREPARE_FIRST" != 1 ]]; then
  echo 'PREPARE_FIRST must be 0 or 1' >&2; exit 2
fi
if [[ -n "$AFTEROK_JOB" && ! "$AFTEROK_JOB" =~ ^[0-9]+$ ]]; then
  echo 'AFTEROK_JOB must be one numeric Slurm job ID' >&2; exit 2
fi
if [[ "$PREPARE_FIRST" == 1 && -n "$AFTEROK_JOB" ]]; then
  echo 'Use PREPARE_FIRST=1 or AFTEROK_JOB, not both' >&2; exit 2
fi
for count in "$REGRESSION_SEGMENTS" "$DIFFUSION_SEGMENTS"; do
  if [[ ! "$count" =~ ^[1-9][0-9]*$ ]]; then
    echo 'REGRESSION_SEGMENTS and DIFFUSION_SEGMENTS must be positive integers' >&2; exit 2
  fi
done
if [[ "$PREPARE_FIRST" == 1 || -n "$AFTEROK_JOB" ]]; then
  # Full hourly-target audit runs in each GPU job after the finalizer succeeds.
  python -m merraflow.cli_v3_precip months --config "$CONFIG" > /dev/null
else
  python -m merraflow.cli_v3_precip audit --config "$CONFIG"
fi
REGRESSION_CHECKPOINT=$(python - "$CONFIG" <<'PY'
import sys
from pathlib import Path
from merraflow.config_v3_precip import load_config
root = Path(load_config(sys.argv[1])['train']['output'])
for stage in ('regression', 'diffusion'):
    path = root / (stage + '_v3_precip')
    if path.exists() and any(path.iterdir()):
        raise SystemExit(f'Existing run at {path}; use explicit --resume submission or fresh outputs')
print(root / 'regression_v3_precip' / 'best_v3_precip.pt')
PY
)
export REGRESSION_CHECKPOINT
# [fix 1] Training is split into chained ~12 h segments. Each segment stops cleanly at
# train.time_limit_hours and the next one auto-resumes; extra segments exit immediately.
mkdir -p logs_v3_precip
unset RESUME
export AUTO_RESUME=1
previous=""
if [[ "$PREPARE_FIRST" == 1 ]]; then
  previous=$(bash scripts/submit_prepare_hourly_v3_precip.sh --parsable)
  if [[ ! "$previous" =~ ^[0-9]+$ ]]; then
    echo 'Preparation did not return a valid finalizer job ID; training not submitted' >&2; exit 1
  fi
fi
if [[ -n "$AFTEROK_JOB" ]]; then
  previous="$AFTEROK_JOB"
  echo "Training chain starts after successful job $AFTEROK_JOB"
fi
jobs=()
for stage in regression diffusion; do
  if [[ "$stage" == regression ]]; then count=$REGRESSION_SEGMENTS; else count=$DIFFUSION_SEGMENTS; fi
  for ((i = 1; i <= count; i++)); do
    export STAGE=$stage
    if [[ -z "$previous" ]]; then
      job=$(sbatch --parsable --export=ALL scripts/slurm_train_v3_precip.sh)
    else
      job=$(sbatch --parsable --export=ALL --dependency="afterok:$previous" --kill-on-invalid-dep=yes scripts/slurm_train_v3_precip.sh)
    fi
    previous=${job%%;*}
    jobs+=("$stage[$i]=$previous")
  done
done
printf 'Submitted chain: %s\n' "${jobs[*]}"
echo 'Diffusion refuses to start until the regression run has completed all epochs.'
