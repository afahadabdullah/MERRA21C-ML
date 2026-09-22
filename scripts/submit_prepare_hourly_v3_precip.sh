#!/usr/bin/env bash
# [fix 6] Build hourly-mean HWT precipitation targets: monthly array + dependent finalizer.
# Restart-safe: completed hours are skipped. Run from the repository root on Discover.
set -euo pipefail
if [[ $# -gt 1 || ( $# -eq 1 && "${1:-}" != --parsable ) ]]; then
  echo 'Usage: bash scripts/submit_prepare_hourly_v3_precip.sh [--parsable]' >&2
  exit 2
fi
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PROJECT_DIR="${PROJECT_DIR:-$PWD}"
export CONFIG="${CONFIG:-configs/discover_v3_precip.yaml}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
HOURLY_MONTHS="$(python -m merraflow.cli_v3_precip months --config "$CONFIG" | tr '\n' ' ')"
export HOURLY_MONTHS
read -r -a months <<< "$HOURLY_MONTHS"
if (( ${#months[@]} == 0 )); then echo 'No preparation months found' >&2; exit 1; fi
mkdir -p logs_v3_precip
array_job="$(FINALIZE=0 sbatch --parsable --export=ALL --array="1-${#months[@]}%14" scripts/slurm_prepare_hourly_v3_precip.sh)"
array_job="${array_job%%;*}"
final_job="$(FINALIZE=1 sbatch --parsable --export=ALL --dependency="afterok:${array_job}" --kill-on-invalid-dep=yes \
  scripts/slurm_prepare_hourly_v3_precip.sh)"
final_job="${final_job%%;*}"
echo "Months: $HOURLY_MONTHS" >&2
echo "Hourly-target array: ${array_job} (${#months[@]} tasks); finalizer: ${final_job}" >&2
if [[ "${1:-}" == --parsable ]]; then echo "$final_job"; fi
