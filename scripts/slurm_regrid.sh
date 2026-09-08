#!/bin/bash
#SBATCH --job-name=regrid_lowres
#SBATCH --account=s311
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=regrid_%j.log
#SBATCH --error=regrid_%j.err

# ==============================================================================
# SLURM script to batch regrid low-res GEOS-FP onto 3 km LCC grid
# ==============================================================================

set -e

PROJECT_DIR="/gpfsm/dnb10/projects/p311/ML_downscaling"
ENV_DIR="/discover/nobackup/projects/GEOS_MITgcm/afahad/conda/envs/myenv"

echo "=== Loading Conda Environment ==="
source /discover/nobackup/projects/GEOS_MITgcm/afahad/conda/etc/profile.d/conda.sh || source ~/.bashrc
conda activate "${ENV_DIR}"

LOWRES_DIR="/gpfsm/dnb06/projects/p174/f5295_fp/diag/Y2025/M01"
HIGHRES_SAMPLE="/gpfsm/dnb08/projects/p38/HWT/CONUS02KM/Feature-c2160_L137/holding/hwt_30mn_slv_LCC/202501/Feature-c2160_L137.hwt_30mn_slv_LCC.20250131_2330z.nc4"
OUTPUT_DIR="${PROJECT_DIR}/data/lowres_lcc_1hr/202501"
WEIGHTS_DIR="${PROJECT_DIR}/data/weights"

echo "=== Starting Regridding Job on $(hostname) ==="
python "${PROJECT_DIR}/scripts/regrid_lowres.py" \
    --lowres_dir "${LOWRES_DIR}" \
    --highres_sample "${HIGHRES_SAMPLE}" \
    --output_dir "${OUTPUT_DIR}" \
    --weights_dir "${WEIGHTS_DIR}"

echo "=== Regridding Completed Successfully ==="
