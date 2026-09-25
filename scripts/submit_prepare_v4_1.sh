#!/usr/bin/env bash
# Pack the v4.1 archive: one CPU array task per month, then finalize (+ data
# preflight). SUBMIT_TRAIN=1 (default) queues training after the finalizer.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PROJECT_DIR="${PROJECT_DIR:-$PWD}"
export ENV_DIR="${ENV_DIR:-/gpfsm/dnb10/projects/p311/ML_downscaling/env}"
export CONFIG="${CONFIG:-configs/discover_v4_1.yaml}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python_bin="${PYTHON_BIN:-$ENV_DIR/bin/python}"
plan=$("$python_bin" -m merraflow.cli_v4_1 plan-packed --config "$CONFIG")
echo "$plan"
export MONTHS=$(printf '%s' "$plan" | "$python_bin" -c 'import json,sys; print(" ".join(json.load(sys.stdin)["months"]))')
read -r -a months <<< "$MONTHS"
mkdir -p logs_v4_1
array=$(env -u SLURM_MEM_PER_CPU -u SLURM_MEM_PER_NODE -u SLURM_MEM_PER_GPU \
  STAGE=prepare sbatch --parsable --export=ALL --array="0-$((${#months[@]}-1))%${MAX_PARALLEL:-8}" scripts/slurm_prepare_v4_1.sh)
array=${array%%;*}
final=$(env -u SLURM_MEM_PER_CPU -u SLURM_MEM_PER_NODE -u SLURM_MEM_PER_GPU \
  STAGE=finalize sbatch --parsable --export=ALL --time=01:00:00 --dependency="afterok:$array" \
  --kill-on-invalid-dep=yes scripts/slurm_prepare_v4_1.sh)
final=${final%%;*}
printf 'Packing array: %s (%s months); finalizer: %s\n' "$array" "${#months[@]}" "$final"
if [[ "${SUBMIT_TRAIN:-1}" == 1 ]]; then
  DEPENDENCY="$final" bash scripts/submit_v4_1.sh
fi
