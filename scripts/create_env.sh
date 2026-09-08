#!/bin/bash
# ==============================================================================
# Script to create the Conda Environment on Discover
# Target Location: /gpfsm/dnb10/projects/p311/ML_downscaling/env
# ==============================================================================

set -e

PROJECT_DIR="/gpfsm/dnb10/projects/p311/ML_downscaling"
ENV_DIR="${PROJECT_DIR}/env"

echo "=== Setting up conda environment at: ${ENV_DIR} ==="

mkdir -p "${PROJECT_DIR}"
cd "${PROJECT_DIR}"

conda env create --prefix "${ENV_DIR}" -f environment.yml

echo "=== Environment successfully created! ==="
echo "To activate: conda activate ${ENV_DIR}"

