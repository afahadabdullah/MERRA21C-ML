#!/usr/bin/env bash
# Regrid the low-res predictors for the months a v2 config requests.
# Invoke with bash from either tcsh or bash. Submits nothing else.
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
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
mkdir -p logs_v2
# REGRID_MONTHS restricts the run to specific months, for example the months a
# coverage report shows as missing only regridded predictors: "2024-12 2026-01".
REGRID_MONTHS="${REGRID_MONTHS:-$(python -m merraflow.cli_v2 prepare-months --config "$CONFIG")}"
export REGRID_MONTHS
read -r -a regrid_months <<< "$REGRID_MONTHS"
WEIGHTS="${WEIGHTS_PATH:-$PROJECT_DIR/data/weights/regrid_weights_bilinear_conus.nc}"
export WEIGHTS_PATH="$WEIGHTS"
if [[ ! -f "$WEIGHTS" ]]; then
  echo "Cached regridding weights are missing: $WEIGHTS" >&2
  echo 'Parallel tasks must not generate them concurrently. Run one month first:' >&2
  echo "  python scripts/regrid_lowres.py --year ${regrid_months[0]%-*} --month ${regrid_months[0]#*-}" >&2
  exit 1
fi
echo "Months: $REGRID_MONTHS"
array_job="$(sbatch --parsable --export=ALL \
  --array="1-${#regrid_months[@]}%${REGRID_CONCURRENCY:-6}" scripts/slurm_regrid_months_v2.sh)"
array_job="${array_job%%;*}"
echo "Regridding array: ${array_job} (${#regrid_months[@]} tasks, at most ${REGRID_CONCURRENCY:-6} concurrent)"
echo 'Existing valid outputs are skipped. Re-run coverage before submitting the pipeline.'
