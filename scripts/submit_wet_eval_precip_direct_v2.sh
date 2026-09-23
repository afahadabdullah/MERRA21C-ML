#!/usr/bin/env bash
# Snapshot the current direct checkpoint; submit an independent wet-case audit.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PROJECT_DIR="${PROJECT_DIR:-$PWD}"
export ENV_DIR="${ENV_DIR:-/gpfsm/dnb10/projects/p311/ML_downscaling/env}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python_bin="${PYTHON_BIN:-$ENV_DIR/bin/python}"
kind="${CHECKPOINT_KIND:-last}"
case "$kind" in last|best) ;; *) echo 'CHECKPOINT_KIND must be last or best' >&2; exit 2 ;; esac
source_checkpoint="${CHECKPOINT:-runs/merraflow_precip_direct_v2/${kind}_direct_v2.pt}"
if [[ ! -f "$source_checkpoint" ]]; then echo "Missing checkpoint: $source_checkpoint" >&2; exit 1; fi
mkdir -p logs_precip_direct_v2
evaluation_root="${EVALUATION_ROOT:-runs/merraflow_precip_direct_v2/wet_evaluations}"
mkdir -p "$evaluation_root"
run_dir="$(mktemp -d "$evaluation_root/${kind}_wet_XXXXXX")"
snapshot="$run_dir/checkpoint_direct_v2.pt"
cp "$source_checkpoint" "$snapshot.tmp"
mv "$snapshot.tmp" "$snapshot"
"$python_bin" - "$snapshot" "$run_dir/config.yaml" "$source_checkpoint" <<'PY'
import json
import sys
from pathlib import Path
import torch
import yaml
from merraflow.precip_direct_v2 import VERSION
checkpoint, config, source = map(Path, sys.argv[1:4])
saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
if saved.get('version') != VERSION or saved.get('targets') != ['precip']:
    raise ValueError('Need a direct precipitation flow checkpoint')
config.write_text(yaml.safe_dump(saved['config'], sort_keys=False))
(checkpoint.parent/'snapshot.json').write_text(json.dumps(dict(
    source=str(source.resolve()), epoch=saved['epoch']+1,
    checkpoint=str(checkpoint.resolve())), indent=2)+'\n')
print(f'Frozen direct checkpoint epoch {saved["epoch"]+1}', flush=True)
PY
job="$(env -u SLURM_MEM_PER_CPU -u SLURM_MEM_PER_NODE -u SLURM_MEM_PER_GPU \
  PROJECT_DIR="$PROJECT_DIR" ENV_DIR="$ENV_DIR" CONFIG="$run_dir/config.yaml" \
  CHECKPOINT="$snapshot" OUTPUT="$run_dir/results" \
  sbatch --parsable --export=ALL scripts/slurm_wet_eval_precip_direct_v2.sh)"
printf '%s\n' "$job" > "$run_dir/submitted_job.txt"
echo "Submitted wet validation job ${job%%;*}; output $run_dir/results"
