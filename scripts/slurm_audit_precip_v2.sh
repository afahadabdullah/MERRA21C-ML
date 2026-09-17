#!/usr/bin/env bash
#SBATCH --job-name=rain_audit_v2
#SBATCH --account=s3292
#SBATCH --qos=allnccs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs_v2/rain_audit_%j_v2.log
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
export MPLCONFIGDIR="${TMPDIR:-/tmp}/merraflow-audit-v2-${SLURM_JOB_ID}"
# Keep this CPU job's --mem allocation; remove inherited alternative modes.
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU SLURM_CPUS_PER_GPU
args=(--config "${CONFIG:-configs/discover_annual_v2.yaml}"
      --predictions "${PREDICTIONS:?Set PREDICTIONS to existing v2 member directory}"
      --split "${SPLIT:-test}")
if [[ -n "${OUTPUT:-}" ]]; then args+=(--output "$OUTPUT"); fi
if [[ -n "${MEMBERS:-}" ]]; then args+=(--members "$MEMBERS"); fi
case "${GROUP_BY_IDENTITY:-0}" in
  1) args+=(--group-by-identity) ;;
  0) ;;
  *) echo 'GROUP_BY_IDENTITY must be 0 or 1' >&2; exit 2 ;;
esac
if [[ -n "${TIMESTAMPS:-}" ]]; then
  read -r -a timestamp_args <<< "$TIMESTAMPS"
  args+=(--timestamps "${timestamp_args[@]}")
fi
srun python scripts/audit_precip_v2.py "${args[@]}"
