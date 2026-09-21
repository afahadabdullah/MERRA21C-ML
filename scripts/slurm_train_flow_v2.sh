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
# Slurm sends stderr to --output when --error is omitted.
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
STAGE="${STAGE:-regression}"
case "$STAGE" in regression|flow) ;; *) echo 'Invalid STAGE' >&2; exit 2 ;; esac
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_NODE
stage_dir="$(python - "$CONFIG" "$STAGE" <<'PY'
import sys
from pathlib import Path
from merraflow.config_v2 import load_config_v2
print(Path(load_config_v2(sys.argv[1])['train']['output'])/f'{sys.argv[2]}_v2')
PY
)"
if [[ -z "${RESUME:-}" && -f "$stage_dir/last_v2.pt" ]]; then
  RESUME="$stage_dir/last_v2.pt"
fi
if [[ "$STAGE" == flow ]]; then
  if [[ -z "${RESUME:-}" && ! -f "${REGRESSION_CHECKPOINT:-}" && ! -f "${INITIALIZE_FLOW:-}" ]]; then
    echo 'Flow needs a regression checkpoint or INITIALIZE_FLOW; none was found.' >&2
    exit 1
  fi
fi
NPROC=$(python -c 'import torch; print(torch.cuda.device_count())')
if (( NPROC < 1 )); then
  echo 'No CUDA GPUs visible in this allocation' >&2
  exit 1
fi
if (( NPROC > 1 )); then
  export PYTHONFAULTHANDLER=1
  export TORCH_NCCL_TRACE_BUFFER_SIZE=2000
  export TORCH_NCCL_DUMP_ON_TIMEOUT=1
  export TORCH_NCCL_DESYNC_DEBUG=1
  echo "DDP preflight: ${NPROC} GPUs on $(hostname)"
  srun torchrun --standalone --nnodes=1 --nproc-per-node="$NPROC" -m merraflow.ddp_preflight_v2
fi
# Check both stages on a real prepared batch in the GPU allocation before a
# fresh regression run. This measures scratch steps, not full-run headroom.
if [[ "${STAGE:-regression}" == regression && -z "${RESUME:-}" ]]; then
  srun python scripts/benchmark_v2.py --config "$CONFIG" --steps 5
fi
if [[ "$STAGE" == flow && -z "${RESUME:-}" && "${TRAIN_PREFLIGHT:-0}" == 1 ]]; then
  benchmark_args=(--config "$CONFIG" --steps 2 --stage flow)
  if [[ -n "${INITIALIZE_FLOW:-}" ]]; then benchmark_args+=(--initialize-flow "$INITIALIZE_FLOW"); fi
  srun python scripts/benchmark_v2.py "${benchmark_args[@]}"
fi
args=(--config "$CONFIG" --stage "${STAGE:-regression}")
if [[ -n "${REGRESSION_CHECKPOINT:-}" ]]; then args+=(--regression-checkpoint "$REGRESSION_CHECKPOINT"); fi
if [[ -n "${RESUME:-}" ]]; then args+=(--resume "$RESUME"); fi
if [[ -n "${INITIALIZE_FLOW:-}" && -z "${RESUME:-}" ]]; then args+=(--initialize-flow "$INITIALIZE_FLOW"); fi
srun torchrun --standalone --nnodes=1 --nproc-per-node="$NPROC" -m merraflow.cli_v2 train "${args[@]}"
if [[ "$STAGE" == flow || "${TRAIN_FLOW_AFTER_REGRESSION:-0}" == 1 ]]; then
  # Training has returned at an epoch boundary. Read the durable checkpoint,
  # not history.jsonl, which can be one record ahead after an interrupted save.
  progress="$(python - "$CONFIG" "$stage_dir/last_v2.pt" "$STAGE" <<'PY'
import sys
import torch
from merraflow.config_v2 import load_config_v2
cfg = load_config_v2(sys.argv[1])
ckpt = torch.load(sys.argv[2], map_location='cpu', weights_only=True)
if ckpt['version'] != 'v2' or ckpt['stage'] != sys.argv[3]:
    raise ValueError('Expected a matching v2 checkpoint before continuation')
print(ckpt['epoch']+1, cfg['train'][f'{sys.argv[3]}_epochs'])
PY
)"
  read -r completed target <<< "$progress"
  if (( completed < target )); then
    next_job="$(env -u SLURM_MEM_PER_GPU CONFIG="$CONFIG" STAGE="$STAGE" RESUME="$stage_dir/last_v2.pt" \
      REGRESSION_CHECKPOINT= sbatch --parsable --export=ALL --job-name="${STAGE}_v2" --gres="gpu:$NPROC" \
      --dependency="afterany:${SLURM_JOB_ID}" scripts/slurm_train_flow_v2.sh)"
    echo "$STAGE completed $completed/$target epochs; continuation queued as ${next_job%%;*}"
  elif [[ "$STAGE" == regression ]]; then
    next_job="$(env -u SLURM_MEM_PER_GPU CONFIG="$CONFIG" STAGE=flow RESUME= \
      REGRESSION_CHECKPOINT="$stage_dir/best_v2.pt" sbatch --parsable --export=ALL \
      --job-name=flow_v2 --gres="gpu:$NPROC" --dependency="afterok:${SLURM_JOB_ID}" scripts/slurm_train_flow_v2.sh)"
    echo "Regression complete; flow queued as ${next_job%%;*}"
  else
    echo "Flow training complete at $completed/$target epochs; no continuation submitted"
    if [[ "${TEST_AFTER_TRAINING:-0}" == 1 ]]; then
      test_checkpoint="$stage_dir/best_v2.pt"
      if [[ "${PREFER_SKILL_CHECKPOINT:-0}" == 1 ]]; then
        if [[ -f "$stage_dir/best_skill_v2.pt" ]]; then
          test_checkpoint="$stage_dir/best_skill_v2.pt"
        else
          echo 'No checkpoint beat the coarse baseline on all three validation criteria; testing best_v2 for diagnosis.'
        fi
      fi
      next_job="$(env -u SLURM_MEM_PER_GPU CONFIG="$CONFIG" CHECKPOINT="$test_checkpoint" \
        OUTPUT="${stage_dir%/flow_v2}/trained_test_v2" SPLIT=test MEMBERS=5 STEPS=24 \
        COMPARE_NOISE_PADDING=0 NOISE_PADDING= INCLUDE_DATE= \
        TIMESTAMPS='20260223_0530 20260209_1530 20260209_2030 20260305_1230 20260306_1830' \
        sbatch --parsable --export=ALL --dependency="afterok:${SLURM_JOB_ID}" scripts/slurm_test_best_model_v2.sh)"
      echo "Five-case test queued as ${next_job%%;*}"
    fi
  fi
fi
