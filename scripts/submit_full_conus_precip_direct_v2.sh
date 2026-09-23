#!/usr/bin/env bash
# Generate the seven wet-evaluation panels for one full CONUS test hour.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PROJECT_DIR="${PROJECT_DIR:-$PWD}"
export ENV_DIR="${ENV_DIR:-/gpfsm/dnb10/projects/p311/ML_downscaling/env}"
case_root="runs/merraflow_precip_direct_v2/wet_evaluations/last_wet_KAF9LG"
timestamp="${TIMESTAMP:-20260223_0530}"
checkpoint="${CHECKPOINT:-$case_root/checkpoint_direct_v2.pt}"
config="${CONFIG:-$case_root/config.yaml}"
output="${OUTPUT:-$case_root/full_conus_${timestamp}}"
if [[ ! -f "$checkpoint" ]]; then echo "Missing direct checkpoint: $checkpoint" >&2; exit 1; fi
if [[ ! -f "$config" ]]; then echo "Missing saved direct config: $config" >&2; exit 1; fi
if [[ -f "$output/${timestamp}_full_conus_direct_v2.png" ]]; then
  echo "Whole-CONUS map already exists: $output/${timestamp}_full_conus_direct_v2.png"
  exit 0
fi
mkdir -p logs_precip_direct_v2
job="$(env -u SLURM_MEM_PER_CPU -u SLURM_MEM_PER_NODE -u SLURM_MEM_PER_GPU \
  PROJECT_DIR="$PROJECT_DIR" ENV_DIR="$ENV_DIR" CONFIG="$config" \
  CHECKPOINT="$checkpoint" OUTPUT="$output" \
  TIMESTAMP="$timestamp" SPLIT="${SPLIT:-test}" \
  MEMBERS="${MEMBERS:-2}" STEPS="${STEPS:-}" \
  sbatch --parsable --export=ALL scripts/slurm_full_conus_precip_direct_v2.sh)"
echo "Submitted full-CONUS direct-precipitation job ${job%%;*}; output $output; log logs_precip_direct_v2/full_${job%%;*}.log"
