#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/gpfsm/dnb10/projects/p311/ML_downscaling}"
ENV_DIR="${ENV_DIR:-/gpfsm/dnb10/projects/p311/ML_downscaling/env}"
set +u
source "${CONDA_SH:-/discover/nobackup/projects/GEOS_MITgcm/afahad/conda/etc/profile.d/conda.sh}"
conda activate "$ENV_DIR"
set -u
cd "$PROJECT_DIR"
export PROJECT_DIR ENV_DIR
export CONFIG="${CONFIG:-configs/discover_v2.yaml}"
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
PREP_MONTHS="$(python -m merraflow.cli_v2 prepare-months --config "$CONFIG")"
export PREP_MONTHS
read -r -a prep_months <<< "$PREP_MONTHS"
mkdir -p logs_v2
array_job="$(sbatch --parsable --export=ALL --array="1-${#prep_months[@]}%14" scripts/slurm_prepare_flow_v2.sh)"
array_job="${array_job%%;*}"
final_job="$(sbatch --parsable --export=ALL --dependency="afterok:${array_job}" --kill-on-invalid-dep=yes scripts/slurm_finalize_prepare_v2.sh)"
final_job="${final_job%%;*}"
echo "Months: $PREP_MONTHS"
echo "Monthly preparation array: ${array_job} (${#prep_months[@]} tasks, at most 14 concurrent)"
echo "Dependent finalization job: ${final_job}"
