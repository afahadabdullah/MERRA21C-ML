#!/usr/bin/env bash
# Regenerate all paper figures from saved data; no GPU or model inference.
# OUTPUT=runs/... sbatch --export=ALL scripts/slurm_paper_figures_v4_1.sh
#SBATCH --job-name=paper_v4_1
#SBATCH --account=s3292
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs_v4_1/paper_v4_1_%j.log
set -euo pipefail
: "${OUTPUT:?Set completed evaluation OUTPUT directory}"
cd "${PROJECT_DIR:-/gpfsm/dnb10/projects/p311/ML_downscaling}"
ENV_DIR="${ENV_DIR:-/gpfsm/dnb10/projects/p311/ML_downscaling/env}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
"$ENV_DIR/bin/python" -m merraflow.paper_figures_v4_1 --evaluation "$OUTPUT" \
  --output "${FIGURE_OUTPUT:-$OUTPUT/paper_figures}" --dpi "${FIGURE_DPI:-600}"
