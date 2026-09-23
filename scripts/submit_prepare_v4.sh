#!/usr/bin/env bash
# CPU-only monthly humidity extraction. Reuses the processed v3 hourly rainfall.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PROJECT_DIR="${PROJECT_DIR:-$PWD}"
export ENV_DIR="${ENV_DIR:-/gpfsm/dnb10/projects/p311/ML_downscaling/env}"
export CONFIG="${CONFIG:-configs/discover_v4.yaml}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python_bin="${PYTHON_BIN:-$ENV_DIR/bin/python}"
args=(--config "$CONFIG")
if [[ -n "${HOURLY_TARGETS:-}" ]]; then args+=(--hourly-targets "$HOURLY_TARGETS"); fi
"$python_bin" -m merraflow.cli_v4 inspect-data "${args[@]}"
export MONTHS=$($python_bin - "$CONFIG" <<'PY'
import sys,json
from pathlib import Path
from merraflow.v4 import load_config
cfg=load_config(sys.argv[1])
index=json.loads((Path(cfg['data']['prepared'])/'index_v2.json').read_text())
print(' '.join(sorted({e['time'][:7] for e in index['entries']})))
PY
)
read -r -a months <<< "$MONTHS"
mkdir -p logs_v4
# Short CPU requests can backfill; restart-safe extraction handles resubmission.
array=$(env -u SLURM_MEM_PER_CPU -u SLURM_MEM_PER_NODE -u SLURM_MEM_PER_GPU \
  STAGE=prepare sbatch --parsable --export=ALL --array="0-$((${#months[@]}-1))%${MAX_PARALLEL:-4}" scripts/slurm_prepare_v4.sh)
array=${array%%;*}
final=$(env -u SLURM_MEM_PER_CPU -u SLURM_MEM_PER_NODE -u SLURM_MEM_PER_GPU \
  STAGE=finalize sbatch --parsable --export=ALL --dependency="afterok:$array" --kill-on-invalid-dep=yes scripts/slurm_prepare_v4.sh)
final=${final%%;*}
printf 'Humidity array: %s; finalizer: %s\n' "$array" "$final"
if [[ "${SUBMIT_TRAIN:-1}" == 1 ]]; then
  DEPENDENCY="$final" bash scripts/submit_v4.sh
fi
