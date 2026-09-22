#!/usr/bin/env bash
# Evaluate a frozen best or latest flow checkpoint while training may continue.
set -euo pipefail
cd "$(dirname "$0")/.."
run_root="${RUN_ROOT:-runs/merraflow_rain_edges_v2}"
checkpoint_kind="${CHECKPOINT_KIND:-best}"
case "$checkpoint_kind" in
  best|last) ;;
  *) echo 'CHECKPOINT_KIND must be best or last' >&2; exit 2 ;;
esac
source_checkpoint="$run_root/flow_v2/${checkpoint_kind}_v2.pt"
python_bin="${PYTHON_BIN:-${ENV_DIR:-$PWD/env}/bin/python}"
if [[ ! -f "$source_checkpoint" ]]; then
  echo "Missing $checkpoint_kind flow checkpoint: $source_checkpoint" >&2
  exit 1
fi
mkdir -p logs_v2 "$run_root/evaluations_v2"
eval_root="$(mktemp -d "$run_root/evaluations_v2/${checkpoint_kind}_v2_XXXXXX")"
checkpoint="$eval_root/${checkpoint_kind}_snapshot_v2.pt"
# Training atomically replaces checkpoints; this copy retains one complete save.
cp "$source_checkpoint" "$checkpoint.tmp"
mv "$checkpoint.tmp" "$checkpoint"
config="$eval_root/config_v2.yaml"
"$python_bin" - "$checkpoint" "$config" "$source_checkpoint" "$checkpoint_kind" <<'PY'
import hashlib
import json
from pathlib import Path
import sys
import torch
import yaml

path, config, source = map(Path, sys.argv[1:4])
kind = sys.argv[4]
ckpt = torch.load(path, map_location='cpu', weights_only=True)
if ckpt.get('version') != 'v2' or ckpt.get('stage') != 'flow':
    raise ValueError('Snapshot must be a v2 flow checkpoint')
# The checkpoint carries the configuration actually used for this run.
config.write_text(yaml.safe_dump(ckpt['config'], sort_keys=False))
digest = hashlib.sha256()
with path.open('rb') as stream:
    for block in iter(lambda: stream.read(1024*1024), b''):
        digest.update(block)
metadata = dict(source=str(source.resolve()), checkpoint=str(path.resolve()),
                epoch=ckpt['epoch']+1, sha256=digest.hexdigest(),
                checkpoint_kind=kind, selection_metric=ckpt.get('selection_metric'))
# A latest checkpoint carries the running best score, not its own epoch's score.
metadata['best_score' if kind == 'best' else 'best_score_so_far'] = ckpt.get('best')
(path.parent/'selection_v2.json').write_text(json.dumps(metadata, indent=2)+'\n')
print(json.dumps(metadata, indent=2), flush=True)
PY
echo "Evaluation output: $eval_root/diagnostic_v2"
job="$(env -u SLURM_MEM_PER_CPU -u SLURM_MEM_PER_NODE -u SLURM_MEM_PER_GPU \
  CONFIG="$config" CHECKPOINT="$checkpoint" OUTPUT="$eval_root/diagnostic_v2" \
  SPLIT=test MEMBERS="${MEMBERS:-5}" STEPS=24 \
  COMPARE_NOISE_PADDING=0 NOISE_PADDING= INCLUDE_DATE= \
  TIMESTAMPS='20260223_0530 20260209_1530 20260209_2030 20260305_1230 20260306_1830' \
  sbatch --parsable --export=ALL --job-name="flow_${checkpoint_kind}_test_v2" --gres=gpu:1 \
  --cpus-per-gpu=4 scripts/slurm_test_best_model_v2.sh)"
printf '%s\n' "$job" > "$eval_root/submitted_job.txt"
echo "Submitted $checkpoint_kind-checkpoint evaluation: ${job%%;*}"
echo "Log: logs_v2/flow_test_${job%%;*}_v2.log"
