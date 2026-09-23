#!/usr/bin/env bash
# CPU-only, read-only audit + diagnostic figures for the v3 hourly rainfall targets used by v4.
#   sbatch scripts/slurm_diagnose_hourly_v4.sh
#   EXTRA="--time 2026-02-23T05:30:00 --checksum" sbatch scripts/slurm_diagnose_hourly_v4.sh
#SBATCH --job-name=diag_hourly_v4
#SBATCH --account=s3292
#SBATCH --qos=allnccs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs_v4/diagnose_hourly_%j.log
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/gpfsm/dnb10/projects/p311/ML_downscaling}"
ENV_DIR="${ENV_DIR:-/gpfsm/dnb10/projects/p311/ML_downscaling/env}"
cd "$PROJECT_DIR"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/diag-mpl-${SLURM_JOB_ID:-$$}"
# shellcheck disable=SC2086
"$ENV_DIR/bin/python" scripts/diagnose_hourly_targets_v4.py --config "${CONFIG:-configs/discover_v4.yaml}" \
  --verify --workers "${SLURM_CPUS_PER_TASK:-8}" ${EXTRA:-}
