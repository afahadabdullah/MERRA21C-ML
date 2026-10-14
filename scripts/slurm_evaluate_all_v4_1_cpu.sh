#!/usr/bin/env bash
#SBATCH --job-name=eval_all_cpu
#SBATCH --account=s3292
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs_v4_1/eval_all_cpu_%j.log
# CPU preparation and reduction; no data-intensive work on the login node.
set -euo pipefail
: "${OUTPUT:?Set evaluation OUTPUT directory}"
cd "${PROJECT_DIR:-/gpfsm/dnb10/projects/p311/ML_downscaling}"
ENV_DIR="${ENV_DIR:-/gpfsm/dnb10/projects/p311/ML_downscaling/env}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
if [[ "${STAGE:-init}" == aggregate ]]; then
  "$ENV_DIR/bin/python" -m merraflow.evaluate_all_v4_1 aggregate --output "$OUTPUT" \
    --block-hours "${BLOCK_HOURS:-72}" --draws "${BOOTSTRAP_DRAWS:-1000}"
  "$ENV_DIR/bin/python" -m merraflow.paper_figures_v4_1 --evaluation "$OUTPUT" \
    --output "${FIGURE_OUTPUT:-$OUTPUT/paper_figures}" --dpi "${FIGURE_DPI:-600}"
elif [[ "${STAGE:-init}" == init ]]; then
  "$ENV_DIR/bin/python" -m merraflow.evaluate_all_v4_1 init \
    --config "${CONFIG:-configs/discover_v4_1.yaml}" --checkpoint "${CHECKPOINT:-best}" \
    --output "$OUTPUT" --split "${SPLIT:-test}" --members "${MEMBERS:-16}" --steps "${STEPS:-64}" \
    --post "${POST:-both}" --weights "${WEIGHTS:-ema}" --spectral-count "${SPECTRAL_COUNT:-12}" \
    --save-cases "${SAVE_CASES:-auto}"
else
  echo 'STAGE must be init or aggregate' >&2; exit 2
fi
