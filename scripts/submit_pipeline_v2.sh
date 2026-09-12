#!/usr/bin/env bash
# Invoke with bash from either tcsh or bash. Submits a fresh 2025 v2 pipeline.
set -eo pipefail
PROJECT_DIR="${PROJECT_DIR:-/gpfsm/dnb10/projects/p311/ML_downscaling}"
ENV_DIR="${ENV_DIR:-/gpfsm/dnb10/projects/p311/ML_downscaling/env}"
# Conda hooks may read unset variables, including when an env is inherited.
set +u
source "${CONDA_SH:-/discover/nobackup/projects/GEOS_MITgcm/afahad/conda/etc/profile.d/conda.sh}"
conda activate "$ENV_DIR"
set -u
cd "$PROJECT_DIR"
export PROJECT_DIR ENV_DIR
export CONFIG="${CONFIG:-configs/discover_v2.yaml}"
export PREP_YEAR=2025
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
mkdir -p logs_v2

# Refuse to silently start over in an existing training run. Preparation itself
# can resume compatible monthly shards or reuse a compatible finalized archive.
run_dir="$(python - "$CONFIG" <<'PY'
import sys
from pathlib import Path
from merraflow.config_v2 import load_config_v2
cfg = load_config_v2(sys.argv[1])
if not (cfg['data']['start'].startswith('2025-') and cfg['data']['end'].startswith('2025-')):
    raise ValueError('This submission helper prepares the twelve months of 2025')
root = Path(cfg['train']['output'])
for stage in ('regression', 'flow'):
    path = root / f'{stage}_v2'
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f'{path} is not empty; use the runbook resume commands')
print(root)
PY
)"
submission_log="$(mktemp logs_v2/submission_v2.XXXXXX)"
echo "Submission record: $submission_log"
python -m merraflow.cli_v2 audit --config "$CONFIG" | tee "${submission_log}.audit.log"

submit() {
  local job_id
  job_id="$(sbatch --parsable --export=ALL "$@")" || return $?
  job_id="${job_id%%;*}"
  if [[ ! "$job_id" =~ ^[0-9]+$ ]]; then
    echo "Unexpected sbatch job ID: $job_id" >&2
    return 1
  fi
  printf '%s\n' "$job_id"
}

prep_job="$(submit scripts/slurm_prepare_flow_v2.sh)"
printf 'Preparation array: %s\n' "$prep_job" | tee -a "$submission_log"
final_job="$(submit --dependency="afterok:$prep_job" --kill-on-invalid-dep=yes scripts/slurm_finalize_prepare_v2.sh)"
printf 'Finalization: %s\n' "$final_job" | tee -a "$submission_log"
reg_job="$(STAGE=regression RESUME= REGRESSION_CHECKPOINT= submit --job-name=regression_v2 --dependency="afterok:$final_job" --kill-on-invalid-dep=yes scripts/slurm_train_flow_v2.sh)"
printf 'Regression: %s\n' "$reg_job" | tee -a "$submission_log"
flow_job="$(STAGE=flow RESUME= REGRESSION_CHECKPOINT="$run_dir/regression_v2/best_v2.pt" submit --job-name=flow_v2 --dependency="afterok:$reg_job" --kill-on-invalid-dep=yes scripts/slurm_train_flow_v2.sh)"
printf 'Flow: %s\n' "$flow_job" | tee -a "$submission_log"
echo 'Submitted. Follow squeue and logs_v2; failed dependencies cancel downstream jobs.'
