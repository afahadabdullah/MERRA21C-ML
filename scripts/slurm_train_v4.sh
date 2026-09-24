#!/usr/bin/env bash
#SBATCH --job-name=v4
#SBATCH --account=s3292
#SBATCH --qos=alla100
#SBATCH --partition=gpu_a100
#SBATCH --constraint=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs_v4/train_%j.log
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/gpfsm/dnb10/projects/p311/ML_downscaling}"
ENV_DIR="${ENV_DIR:-/gpfsm/dnb10/projects/p311/ML_downscaling/env}"
CONFIG="${CONFIG:-configs/discover_v4.yaml}"
set +u
source /discover/nobackup/projects/GEOS_MITgcm/afahad/conda/etc/profile.d/conda.sh
conda activate "$ENV_DIR"
set -u
cd "$PROJECT_DIR"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/v4-mpl-${SLURM_JOB_ID}"
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_NODE
count=$(python -c 'import torch; print(torch.cuda.device_count())')
if [[ "$count" != 4 ]]; then echo "Require four GPUs; found $count" >&2; exit 1; fi
args=(--config "$CONFIG")
if [[ -n "${REGRESSION_CHECKPOINT:-}" ]]; then args+=(--regression-checkpoint "$REGRESSION_CHECKPOINT"); fi
if [[ -n "${RESUME:-}" ]]; then args+=(--resume "$RESUME"); fi
if [[ -n "${HOURLY_TARGETS:-}" ]]; then args+=(--hourly-targets "$HOURLY_TARGETS"); fi
if [[ -n "${RESUME:-}" ]]; then
  echo "Resuming $RESUME directly; using saved calibration and trainer compatibility checks."
else
  # First-run sample checks; `preflight --full` remains available as an audit.
  python -m merraflow.cli_v4 preflight "${args[@]}"
fi
srun --ntasks=1 --cpu-bind=none torchrun --standalone --nnodes=1 --nproc-per-node=4 \
  -m merraflow.cli_v4 train "${args[@]}"
# Queue only the continuation that is actually needed; no fixed segment count.
progress=$(python - "$CONFIG" <<'PY'
import sys
from pathlib import Path
import torch
from merraflow.v4 import load_config
cfg = load_config(sys.argv[1])
path = Path(cfg['train']['output'])/'last_v4.pt'
saved = torch.load(path, map_location='cpu', weights_only=True)
if saved['epoch']+1 < cfg['train']['epochs']:
    print(path)
PY
)
if [[ -n "$progress" ]]; then
  env -u SLURM_MEM_PER_CPU -u SLURM_MEM_PER_NODE -u SLURM_MEM_PER_GPU \
    CONFIG="$CONFIG" PROJECT_DIR="$PROJECT_DIR" ENV_DIR="$ENV_DIR" RESUME="$progress" \
    sbatch --parsable --export=ALL --dependency="afterok:${SLURM_JOB_ID}" --kill-on-invalid-dep=yes \
    scripts/slurm_train_v4.sh
fi
