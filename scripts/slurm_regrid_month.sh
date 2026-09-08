#!/bin/bash
#SBATCH --job-name=regrid_mon
#SBATCH --account=s311
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=regrid_month_%j.log
#SBATCH --error=regrid_month_%j.err

# ==============================================================================
# Full-Month Resumable Batch Regridding on Discover (16 Parallel Workers)
# ==============================================================================

set -e

PROJECT_DIR="/gpfsm/dnb10/projects/p311/ML_downscaling"
ENV_DIR="/discover/nobackup/projects/GEOS_MITgcm/afahad/conda/envs/myenv"

echo "=== Host: $(hostname) | Date: $(date) ==="
echo "=== Loading Conda Environment ==="

source /discover/nobackup/projects/GEOS_MITgcm/afahad/conda/etc/profile.d/conda.sh || source ~/.bashrc
conda activate "${ENV_DIR}"

cd "${PROJECT_DIR}"

echo "=== Launching Parallel Regridding with 16 Workers ==="
python scripts/regrid_lowres.py \
    --lowres_dir /gpfsm/dnb06/projects/p174/f5295_fp/diag/Y2025/M01 \
    --highres_sample /gpfsm/dnb08/projects/p38/HWT/CONUS02KM/Feature-c2160_L137/holding/hwt_30mn_slv_LCC/202501/Feature-c2160_L137.hwt_30mn_slv_LCC.20250131_2330z.nc4 \
    --output_dir "${PROJECT_DIR}/data/lowres_lcc_1hr/202501" \
    --weights_path "${PROJECT_DIR}/data/weights/regrid_weights_bilinear_conus.nc" \
    --num_workers 16

echo "=== Finished Successfully at $(date) ==="
