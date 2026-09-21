#!/usr/bin/env bash
# Freeze a source flow checkpoint, then submit an isolated fine-tuning run.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
config="${CONFIG:-configs/discover_rain_edges_v2.yaml}"
source_checkpoint="${SOURCE_CHECKPOINT:-runs/merraflow_rain_structure_v2/flow_v2/best_v2.pt}"
python_bin="${PYTHON_BIN:-${ENV_DIR:-$PWD/env}/bin/python}"
gpus="${GPUS:-2}"
case "$gpus" in 2|4) ;; *) echo 'GPUS must be 2 (default) or 4' >&2; exit 2 ;; esac
if [[ ! -f "$source_checkpoint" ]]; then
  echo "Missing source flow checkpoint: $source_checkpoint" >&2
  exit 1
fi
run_root="$("$python_bin" - "$config" <<'PY'
import sys
from merraflow.config_v2 import load_config_v2
print(load_config_v2(sys.argv[1])['train']['output'])
PY
)"
if [[ -d "$run_root" ]]; then
  echo "Run already exists: $run_root. Resume its last checkpoint; do not submit a duplicate." >&2
  exit 1
fi
mkdir -p logs_v2 "$run_root"
mkdir "$run_root/initialization_v2"
snapshot="$run_root/initialization_v2/source_flow_v2.pt"
config_snapshot="$run_root/initialization_v2/config_v2.yaml"
cp "$source_checkpoint" "$snapshot.tmp"
mv "$snapshot.tmp" "$snapshot"
# Check checkpoint identity on the login node; archive/GPU checks run in Slurm.
"$python_bin" - "$config" "$snapshot" "$config_snapshot" "$gpus" <<'PY'
import sys
import torch
from pathlib import Path
import yaml
from merraflow.config_v2 import load_config_v2, validate_config_v2
from merraflow.train_v2 import architecture_v2, patch_geometry_v2
cfg = load_config_v2(sys.argv[1])
ckpt = torch.load(sys.argv[2], map_location='cpu', weights_only=True)
if ckpt.get('version') != 'v2' or ckpt.get('stage') != 'flow':
    raise ValueError('Source must be a v2 flow checkpoint')
if architecture_v2(cfg['model']) != architecture_v2(ckpt['config']['model']) or patch_geometry_v2(cfg['patch']) != patch_geometry_v2(ckpt['config']['patch']):
    raise ValueError('Use the matching rollout/control config; larger-context training starts from scratch')
old_world, world = cfg['train'].get('reference_world_size', 2), int(sys.argv[4])
for settings, key in ((cfg['train'], 'accumulate'), (cfg['train'], 'val_batches'),
                      (cfg['train']['generated_validation'], 'batches')):
    total = settings[key]*old_world
    if total % world or total < world:
        raise ValueError(f'Cannot preserve {key} sample count with {world} GPUs')
    settings[key] = total//world
cfg['train']['reference_world_size'] = world
validate_config_v2(cfg)
Path(sys.argv[3]).write_text(yaml.safe_dump(cfg, sort_keys=False))
print(f'Frozen source epoch {ckpt["epoch"]+1}: {sys.argv[2]}', flush=True)
print(f'{world} GPUs; effective batch {world*cfg["train"]["batch_size"]*cfg["train"]["accumulate"]}; frozen config: {sys.argv[3]}', flush=True)
PY
config="$config_snapshot"
submission_options=(--parsable --export=ALL --job-name=rain_rollout_v2 --gres="gpu:$gpus" --cpus-per-gpu=4)
edge_sampling="$("$python_bin" - "$config" <<'PY'
import sys
from merraflow.config_v2 import load_config_v2
print(int(load_config_v2(sys.argv[1])['patch'].get('structure_fraction', 0) > 0))
PY
)"
if [[ "$edge_sampling" == 1 && -z "${CACHE_JOB_ID:-}" ]]; then
  cache_job="$(env -u SLURM_MEM_PER_CPU -u SLURM_MEM_PER_NODE -u SLURM_MEM_PER_GPU \
    CONFIG="$config" sbatch --parsable --export=ALL scripts/slurm_cache_rain_edges_v2.sh)"
  CACHE_JOB_ID="${cache_job%%;*}"
  printf '%s\n' "$CACHE_JOB_ID" > "$run_root/initialization_v2/cache_job.txt"
  echo "Rain-edge proposal cache queued as $CACHE_JOB_ID"
fi
if [[ -n "${CACHE_JOB_ID:-}" ]]; then submission_options+=(--dependency="afterok:${CACHE_JOB_ID}"); fi
job="$(env -u SLURM_MEM_PER_CPU -u SLURM_MEM_PER_NODE -u SLURM_MEM_PER_GPU \
  -u TRAIN_BATCH_SIZE_OVERRIDE -u TRAIN_WORKERS_OVERRIDE \
  CONFIG="$config" STAGE=flow INITIALIZE_FLOW="$snapshot" RESUME= REGRESSION_CHECKPOINT= \
  ALLOW_WORLD_SIZE_CHANGE=0 RESET_FLOW_BEST_ON_MIGRATION=0 \
  TRAIN_PREFLIGHT=1 TEST_AFTER_TRAINING=1 PREFER_SKILL_CHECKPOINT=1 FLOW_PLOT_INTERVAL=1000 \
  sbatch "${submission_options[@]}" scripts/slurm_train_flow_v2.sh)"
printf '%s\n' "$job" > "$run_root/initialization_v2/submitted_job.txt"
echo "Submitted job ${job%%;*}; output: $run_root"
