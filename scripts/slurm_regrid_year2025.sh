#!/bin/bash
#SBATCH --job-name=regrid2025
#SBATCH --account=s311
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --array=1-12
#SBATCH --output=logs/regrid_2025_m%a_%A.log
#SBATCH --error=logs/regrid_2025_m%a_%A.err

# ==============================================================================
# SLURM Job Array: 12 Parallel Tasks for Year 2025 (1 Task per Month)
# Automatically skips already processed & verified files (100% resumable)
# ==============================================================================

set -e

PROJECT_DIR="/gpfsm/dnb10/projects/p311/ML_downscaling"
ENV_DIR="/discover/nobackup/projects/GEOS_MITgcm/afahad/conda/envs/myenv"

mkdir -p "${PROJECT_DIR}/logs"
mkdir -p "${PROJECT_DIR}/data/weights"
mkdir -p "${PROJECT_DIR}/data/lowres_lcc_1hr"

# Activate user's conda environment
source /discover/nobackup/projects/GEOS_MITgcm/afahad/conda/etc/profile.d/conda.sh || source ~/.bashrc
conda activate "${ENV_DIR}"

cd "${PROJECT_DIR}"

MONTH_NUM=$SLURM_ARRAY_TASK_ID
MONTH_STR=$(printf "%02d" $MONTH_NUM)

echo "========================================================================"
echo "Job Array Task ${SLURM_ARRAY_TASK_ID} | Host: $(hostname) | Date: $(date)"
echo "Processing Year 2025, Month: ${MONTH_STR} (Y2025/M${MONTH_STR})"
echo "========================================================================"

python scripts/regrid_lowres.py \
    --lowres_root /gpfsm/dnb06/projects/p174/f5295_fp/diag \
    --year 2025 \
    --month "${MONTH_NUM}" \
    --highres_sample /gpfsm/dnb08/projects/p38/HWT/CONUS02KM/Feature-c2160_L137/holding/hwt_30mn_slv_LCC/202501/Feature-c2160_L137.hwt_30mn_slv_LCC.20250131_2330z.nc4 \
    --output_root "${PROJECT_DIR}/data/lowres_lcc_1hr" \
    --weights_path "${PROJECT_DIR}/data/weights/regrid_weights_bilinear_conus.nc" \
    --num_workers 16

echo "=== Month ${MONTH_STR} of 2025 Completed Successfully at $(date) ==="
