#!/usr/bin/env bash
#SBATCH --job-name=merraflow_v2
#SBATCH --account=s3292
#SBATCH --qos=alla100
#SBATCH --partition=gpu_a100
#SBATCH --constraint=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs_v2/flow_%j_v2.log
#SBATCH --error=logs_v2/flow_%j_v2.err
# Submit from project root after mkdir -p logs_v2.
# The queue-friendly default is one A100. Request two at submission with
# ``sbatch --gres=gpu:2 scripts/slurm_train_flow_v2.sh``. NPROC below is derived
# from CUDA visibility and launches one DDP rank per allocated GPU.
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/gpfsm/dnb10/projects/p311/ML_downscaling}"
ENV_DIR="${ENV_DIR:-/gpfsm/dnb10/projects/p311/ML_downscaling/env}"
# Conda's deactivation hooks are not nounset-safe when an interactive env is inherited.
set +u
source /discover/nobackup/projects/GEOS_MITgcm/afahad/conda/etc/profile.d/conda.sh
conda activate "$ENV_DIR"
set -u
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/merraflow-matplotlib-${SLURM_JOB_ID}"
CONFIG="${CONFIG:-configs/discover_v2.yaml}"
if [[ "${STAGE:-regression}" == flow ]]; then
  # A new submission resumes the same flow run automatically.
  flow_dir="$(python - "$CONFIG" <<'PY'
import sys
from pathlib import Path
from merraflow.config_v2 import load_config_v2
print(Path(load_config_v2(sys.argv[1])['train']['output'])/'flow_v2')
PY
)"
  if [[ -z "${RESUME:-}" && -f "$flow_dir/last_v2.pt" ]]; then
    RESUME="$flow_dir/last_v2.pt"
  fi
  if [[ -z "${RESUME:-}" && ! -f "${REGRESSION_CHECKPOINT:-}" ]]; then
    echo 'Flow needs a completed regression checkpoint; none was found.' >&2
    exit 1
  fi
fi
NPROC=$(python -c 'import torch; print(torch.cuda.device_count())')
if (( NPROC < 1 )); then
  echo 'No CUDA GPUs visible in this allocation' >&2
  exit 1
fi
# Check both stages on a real prepared batch in the GPU allocation before a
# fresh regression run. This measures scratch steps, not full-run headroom.
if [[ "${STAGE:-regression}" == regression && -z "${RESUME:-}" ]]; then
  srun python scripts/benchmark_v2.py --config "$CONFIG" --steps 5
fi
args=(--config "$CONFIG" --stage "${STAGE:-regression}")
if [[ -n "${REGRESSION_CHECKPOINT:-}" ]]; then args+=(--regression-checkpoint "$REGRESSION_CHECKPOINT"); fi
if [[ -n "${RESUME:-}" ]]; then args+=(--resume "$RESUME"); fi
srun torchrun --standalone --nnodes=1 --nproc-per-node="$NPROC" -m merraflow.cli_v2 train "${args[@]}"
if [[ "${STAGE:-regression}" == flow ]]; then
  # Training has returned at an epoch boundary. Read the durable checkpoint,
  # not history.jsonl, which can be one record ahead after an interrupted save.
  progress="$(python - "$CONFIG" "$flow_dir/last_v2.pt" <<'PY'
import sys
import torch
from merraflow.config_v2 import load_config_v2
cfg = load_config_v2(sys.argv[1])
ckpt = torch.load(sys.argv[2], map_location='cpu', weights_only=True)
if ckpt['version'] != 'v2' or ckpt['stage'] != 'flow':
    raise ValueError('Expected a v2 flow checkpoint before continuation')
print(ckpt['epoch']+1, cfg['train']['flow_epochs'])
PY
)"
  read -r completed target <<< "$progress"
  if (( completed < target )); then
    next_job="$(env CONFIG="$CONFIG" STAGE=flow RESUME="$flow_dir/last_v2.pt" \
      REGRESSION_CHECKPOINT= sbatch --parsable --export=ALL --job-name=flow_v2 \
      --dependency="afterany:${SLURM_JOB_ID}" scripts/slurm_train_flow_v2.sh)"
    echo "Flow completed $completed/$target epochs; continuation queued as ${next_job%%;*}"
  else
    echo "Flow training complete at $completed/$target epochs; no continuation submitted"
  fi
fi
