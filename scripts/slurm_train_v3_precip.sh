#!/usr/bin/env bash
#SBATCH --job-name=v3_precip
#SBATCH --account=s3292
#SBATCH --qos=alla100
#SBATCH --partition=gpu_a100
#SBATCH --constraint=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs_v3_precip/train_%j.log
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/gpfsm/dnb10/projects/p311/ML_downscaling}"
ENV_DIR="${ENV_DIR:-/gpfsm/dnb10/projects/p311/ML_downscaling/env}"
set +u
source /discover/nobackup/projects/GEOS_MITgcm/afahad/conda/etc/profile.d/conda.sh
conda activate "$ENV_DIR"
set -u
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/merraflow-v3-precip-mpl-${SLURM_JOB_ID}"
CONFIG="${CONFIG:-configs/discover_v3_precip.yaml}"
STAGE="${STAGE:-regression}"
case "$STAGE" in regression|diffusion) ;; *) echo 'Invalid STAGE' >&2; exit 2 ;; esac
python -m merraflow.cli_v3_precip audit --config "$CONFIG"
NPROC=$(python -c 'import torch; print(torch.cuda.device_count())')
if [[ "$NPROC" != 4 ]]; then
  echo "v3_precip requires 4 visible GPUs; found $NPROC" >&2
  exit 1
fi
args=(--config "$CONFIG" --stage "$STAGE")
# [fix 1] Chained segments: AUTO_RESUME=1 continues from this stage's last checkpoint if present.
if [[ -z "${RESUME:-}" && "${AUTO_RESUME:-0}" == 1 ]]; then
  LAST=$(python - "$CONFIG" "$STAGE" <<'PY'
import sys
from pathlib import Path
from merraflow.config_v3_precip import load_config
print(Path(load_config(sys.argv[1])['train']['output'])/(sys.argv[2]+'_v3_precip')/'last_v3_precip.pt')
PY
)
  if [[ -f "$LAST" ]]; then RESUME="$LAST"; echo "Auto-resuming from $LAST"; fi
fi
if [[ -n "${RESUME:-}" ]]; then args+=(--resume "$RESUME"); fi
if [[ "$STAGE" == diffusion && -z "${REGRESSION_CHECKPOINT:-}" ]]; then
  REGRESSION_CHECKPOINT=$(python - "$CONFIG" <<'PY'
import sys
from pathlib import Path
from merraflow.config_v3_precip import load_config
print(Path(load_config(sys.argv[1])['train']['output'])/'regression_v3_precip'/'best_v3_precip.pt')
PY
)
fi
if [[ -n "${REGRESSION_CHECKPOINT:-}" ]]; then args+=(--regression-checkpoint "$REGRESSION_CHECKPOINT"); fi
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_NODE
srun --ntasks=1 --cpu-bind=none torchrun --standalone --nnodes=1 --nproc-per-node=4 \
  -m merraflow.cli_v3_precip train "${args[@]}"
