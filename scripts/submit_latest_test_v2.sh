#!/usr/bin/env bash
# Freeze the most recent flow checkpoint before a queued evaluation can start.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs_v2
run_root="${RUN_ROOT:-runs/merraflow_rain_structure_v2}"
source_checkpoint="$run_root/flow_v2/last_v2.pt"
if [[ ! -f "$source_checkpoint" ]]; then
  echo "Missing latest flow checkpoint: $source_checkpoint" >&2
  exit 1
fi
mkdir -p "$run_root/evaluations_v2"
eval_root="$(mktemp -d "$run_root/evaluations_v2/latest_10member_v2_XXXXXX")"
checkpoint="$eval_root/last_snapshot_v2.pt"
cp "$source_checkpoint" "$checkpoint.tmp"
mv "$checkpoint.tmp" "$checkpoint"
python_bin="${PYTHON_BIN:-${ENV_DIR:-$PWD/env}/bin/python}"
"$python_bin" - "$checkpoint" <<'PY'
import sys
import torch
checkpoint = torch.load(sys.argv[1], map_location='cpu', weights_only=True)
if checkpoint.get('version') != 'v2' or checkpoint.get('stage') != 'flow':
    raise ValueError('Snapshot is not a v2 flow checkpoint')
print(f'Frozen flow epoch {checkpoint["epoch"]+1}: {sys.argv[1]}', flush=True)
PY
echo "Evaluation output: $eval_root/diagnostic_v2"
env -u SLURM_MEM_PER_CPU -u SLURM_MEM_PER_NODE -u SLURM_MEM_PER_GPU \
  CONFIG="${CONFIG:-configs/discover_rain_structure_v2.yaml}" \
  CHECKPOINT="$checkpoint" OUTPUT="$eval_root/diagnostic_v2" MEMBERS=10 \
  sbatch --export=ALL scripts/slurm_test_latest_model_v2.sh
