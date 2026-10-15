#!/usr/bin/env bash
#SBATCH --job-name=paper_interim_v41
#SBATCH --account=s3292
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs_v4_1/paper_interim_%j.log
# CPU-only snapshot/plotting/paper update; does not depend on the live array.
set -euo pipefail
: "${OUTPUT:?Set OUTPUT to the running full-test evaluation directory}"
PROJECT_DIR="${PROJECT_DIR:-/gpfsm/dnb10/projects/p311/ML_downscaling}"
ENV_DIR="${ENV_DIR:-/gpfsm/dnb10/projects/p311/ML_downscaling/env}"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/merraflow-interim-${SLURM_JOB_ID:-local}"
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU
snapshot="${SNAPSHOT_OUTPUT:-${OUTPUT}_interim_${SLURM_JOB_ID:?}}"
args=(--evaluation "$OUTPUT" --output "$snapshot"
      --manuscript "${MANUSCRIPT:-$PROJECT_DIR/manuscript/MERRA21C_ML_arxiv.tex}"
      --dpi "${FIGURE_DPI:-300}" --compile)
if [[ "${SKIP_MAPS:-0}" == 1 ]]; then args+=(--skip-maps); fi
"$ENV_DIR/bin/python" "${INTERIM_BUILDER:-$PROJECT_DIR/scripts/build_interim_paper_v4_1.py}" "${args[@]}"
printf 'Interim paper bundle: %s/MERRAFlow_interim_paper.zip\n' "$snapshot"
