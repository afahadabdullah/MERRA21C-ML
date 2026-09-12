#!/usr/bin/env bash
#SBATCH --job-name=regrid_v2
#SBATCH --account=s3292
#SBATCH --qos=allnccs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs_v2/regrid_%A_%a_v2.log
# One task per calendar month. The month list comes from the submitting helper,
# so any year works; the fixed-2025 v1 regridding scripts are left unchanged.
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/gpfsm/dnb10/projects/p311/ML_downscaling}"
ENV_DIR="${ENV_DIR:-/gpfsm/dnb10/projects/p311/ML_downscaling/env}"
# Conda's deactivation hooks are not nounset-safe when an interactive env is inherited.
set +u
source "${CONDA_SH:-/discover/nobackup/projects/GEOS_MITgcm/afahad/conda/etc/profile.d/conda.sh}"
conda activate "$ENV_DIR"
set -u
cd "$PROJECT_DIR"
: "${REGRID_MONTHS:?Submit through submit_regrid_months_v2.sh}"
: "${SLURM_ARRAY_TASK_ID:?Submit as an array job through submit_regrid_months_v2.sh}"
read -r -a regrid_months <<< "$REGRID_MONTHS"
if (( SLURM_ARRAY_TASK_ID < 1 || SLURM_ARRAY_TASK_ID > ${#regrid_months[@]} )); then
  echo 'Regridding task index is outside the submitted month list' >&2
  exit 1
fi
MONTH="${regrid_months[SLURM_ARRAY_TASK_ID-1]}"
echo "Regridding task $SLURM_ARRAY_TASK_ID: $MONTH"
# Weights are shared and must already exist: concurrent tasks must not write them.
WEIGHTS="${WEIGHTS_PATH:-$PROJECT_DIR/data/weights/regrid_weights_bilinear_conus.nc}"
test -f "$WEIGHTS" || { echo "Missing cached weights $WEIGHTS" >&2; exit 1; }
python scripts/regrid_lowres.py \
    --lowres_root "${LOWRES_ROOT:-/gpfsm/dnb06/projects/p174/f5295_fp/diag}" \
    --year "${MONTH%-*}" \
    --month "${MONTH#*-}" \
    --highres_sample "${HIGHRES_SAMPLE:-/gpfsm/dnb08/projects/p38/HWT/CONUS02KM/Feature-c2160_L137/holding/hwt_30mn_slv_LCC/202501/Feature-c2160_L137.hwt_30mn_slv_LCC.20250131_2330z.nc4}" \
    --output_root "${OUTPUT_ROOT:-$PROJECT_DIR/data/lowres_lcc_1hr}" \
    --weights_path "$WEIGHTS" \
    --num_workers 16
echo "Regridding task $SLURM_ARRAY_TASK_ID finished: $MONTH"
