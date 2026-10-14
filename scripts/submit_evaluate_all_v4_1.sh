#!/usr/bin/env bash
# New run: OUTPUT=... bash scripts/submit_evaluate_all_v4_1.sh
# Resume:  OUTPUT=... RESUME=1 bash scripts/submit_evaluate_all_v4_1.sh
# A resume reads frozen settings, not current CHECKPOINT/MEMBERS/STEPS values.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PROJECT_DIR="${PROJECT_DIR:-$PWD}"
cd "$PROJECT_DIR"
export ENV_DIR="${ENV_DIR:-/gpfsm/dnb10/projects/p311/ML_downscaling/env}"
export OUTPUT="${OUTPUT:?Set a fresh evaluation OUTPUT directory (or RESUME=1)}"
export SHARDS="${SHARDS:-64}"
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
if ! [[ "$SHARDS" =~ ^[1-9][0-9]*$ && "${MAX_PARALLEL:-4}" =~ ^[1-9][0-9]*$ ]]; then
  echo 'SHARDS and MAX_PARALLEL must be positive integers' >&2; exit 2
fi
mkdir -p logs_v4_1
array_args=(--parsable --export=ALL --array="0-$((SHARDS-1))%${MAX_PARALLEL:-4}")
if [[ "${RESUME:-0}" != 1 ]]; then
  init=$(env -u SLURM_MEM_PER_CPU -u SLURM_MEM_PER_NODE -u SLURM_MEM_PER_GPU STAGE=init \
    sbatch --parsable --export=ALL scripts/slurm_evaluate_all_v4_1_cpu.sh)
  init=${init%%;*}
  array_args+=(--dependency="afterok:$init" --kill-on-invalid-dep=yes)
  printf 'CPU initialization: %s\n' "$init"
else
  test -f "$OUTPUT/manifest.json"
fi
array=$(env -u SLURM_MEM_PER_CPU -u SLURM_MEM_PER_NODE -u SLURM_MEM_PER_GPU \
  sbatch "${array_args[@]}" scripts/slurm_evaluate_all_v4_1.sh)
array=${array%%;*}
final=$(env -u SLURM_MEM_PER_CPU -u SLURM_MEM_PER_NODE -u SLURM_MEM_PER_GPU STAGE=aggregate \
  sbatch --parsable --export=ALL --time=02:00:00 --dependency="afterok:$array" \
  --kill-on-invalid-dep=yes scripts/slurm_evaluate_all_v4_1_cpu.sh)
final=${final%%;*}
printf 'Full-test array: %s; finalizer: %s; output: %s\n' "$array" "$final" "$OUTPUT"
printf 'Resume incomplete hours with OUTPUT=%q RESUME=1 bash scripts/submit_evaluate_all_v4_1.sh\n' "$OUTPUT"
