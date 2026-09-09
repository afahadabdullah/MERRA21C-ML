#!/bin/bash
#SBATCH --job-name=regrid_arr
#SBATCH --account=s3292
#SBATCH --qos=allnccs
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=1-31
#SBATCH --output=logs/regrid_day_%a_%A.log
#SBATCH --error=logs/regrid_day_%a_%A.err

# ==============================================================================
# SLURM Job Array: 31 Parallel Tasks (1 Task per Day of the Month)
# ==============================================================================

set -e

PROJECT_DIR="/gpfsm/dnb10/projects/p311/ML_downscaling"
ENV_DIR="/gpfsm/dnb10/projects/p311/ML_downscaling/env"

mkdir -p "${PROJECT_DIR}/logs"

source /discover/nobackup/projects/GEOS_MITgcm/afahad/conda/etc/profile.d/conda.sh || source ~/.bashrc
conda activate "${ENV_DIR}"

cd "${PROJECT_DIR}"

DAY=$(printf "%02d" $SLURM_ARRAY_TASK_ID)
DATE="202501${DAY}"

echo "=== Processing Day: ${DATE} on $(hostname) with 4 CPUs ==="

python scripts/regrid_lowres.py \
    --lowres_dir /gpfsm/dnb06/projects/p174/f5295_fp/diag/Y2025/M01 \
    --highres_sample /gpfsm/dnb08/projects/p38/HWT/CONUS02KM/Feature-c2160_L137/holding/hwt_30mn_slv_LCC/202501/Feature-c2160_L137.hwt_30mn_slv_LCC.20250131_2330z.nc4 \
    --output_dir "${PROJECT_DIR}/data/lowres_lcc_1hr/202501" \
    --weights_path "${PROJECT_DIR}/data/weights/regrid_weights_bilinear_conus.nc" \
    --date "${DATE}" \
    --num_workers 4

echo "=== Finished Day ${DATE} at $(date) ==="
