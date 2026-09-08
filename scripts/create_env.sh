#!/bin/bash
# ==============================================================================
# Script to create the Conda Environment on Discover
# Target Location: /gpfsm/dnb10/projects/p311/ML_downscaling/env
# ==============================================================================

set -e

ENV_DIR="/gpfsm/dnb10/projects/p311/ML_downscaling/env"
PROJECT_DIR="/gpfsm/dnb10/projects/p311/ML_downscaling"

echo "=== Setting up conda environment at: ${ENV_DIR} ==="

# Load Discover modules
source /usr/share/modules/init/bash || true
module purge || true
module load python/GEOSpyD/Min24.7.2-0_py3.11 || module load comp/gcc/11.2.0

# Create conda environment directly in target prefix
mkdir -p "${PROJECT_DIR}"
conda env create --prefix "${ENV_DIR}" -f "${PROJECT_DIR}/environment.yml"

echo "=== Environment successfully created! ==="
echo "To activate: conda activate ${ENV_DIR}"
