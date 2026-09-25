#!/usr/bin/env bash
#SBATCH --job-name=v4.1
#SBATCH --account=s3292
#SBATCH --qos=alla100
#SBATCH --partition=gpu_a100
#SBATCH --constraint=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-gpu=110G
#SBATCH --time=12:00:00
#SBATCH --output=logs_v4_1/train_%j.log
# Whole A100 node (48 cores): 10 persistent loader workers per GPU. No
# preflight here -- packing's finalize job already validated the data.
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/gpfsm/dnb10/projects/p311/ML_downscaling}"
ENV_DIR="${ENV_DIR:-/gpfsm/dnb10/projects/p311/ML_downscaling/env}"
CONFIG="${CONFIG:-configs/discover_v4_1.yaml}"
set +u
source /discover/nobackup/projects/GEOS_MITgcm/afahad/conda/etc/profile.d/conda.sh
conda activate "$ENV_DIR"
set -u
cd "$PROJECT_DIR"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/v4_1-mpl-${SLURM_JOB_ID}"
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_NODE
count=$(nvidia-smi -L | wc -l)
if [[ "$count" != 4 ]]; then echo "Require four GPUs; found $count" >&2; exit 1; fi
args=(--config "$CONFIG")
if [[ -n "${RESUME:-}" ]]; then args+=(--resume "$RESUME"); fi
srun --ntasks=1 --cpu-bind=none torchrun --standalone --nnodes=1 --nproc-per-node=4 \
  -m merraflow.cli_v4_1 train "${args[@]}"
# Queue one continuation only if epochs remain (reads history.json; no torch import).
progress=$(python - "$CONFIG" <<'PY'
import json, sys
from pathlib import Path
import yaml
cfg = yaml.safe_load(Path(sys.argv[1]).read_text())
out = Path(cfg['train']['output'])
done = len(json.loads((out/'history.json').read_text())) if (out/'history.json').exists() else 0
if (out/'last_v4_1.pt').exists() and done < cfg['train']['epochs']:
    print(out/'last_v4_1.pt')
PY
)
if [[ -n "$progress" ]]; then
  env -u SLURM_MEM_PER_CPU -u SLURM_MEM_PER_NODE -u SLURM_MEM_PER_GPU \
    CONFIG="$CONFIG" PROJECT_DIR="$PROJECT_DIR" ENV_DIR="$ENV_DIR" RESUME="$progress" \
    sbatch --parsable --export=ALL --dependency="afterok:${SLURM_JOB_ID}" --kill-on-invalid-dep=yes \
    scripts/slurm_train_v4_1.sh
fi
