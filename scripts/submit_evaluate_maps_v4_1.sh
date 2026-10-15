#!/usr/bin/env bash
# All-hours per-pixel error maps (every EVERY-th test hour, MEMBERS members) for the paper.
#   EVAL=runs/merraflow_v4_1/evaluation/all_test_best_m16_s64 bash scripts/submit_evaluate_maps_v4_1.sh
# Defaults: EVERY=3 MEMBERS=8 SHARDS=8 MAX_PARALLEL=4 MAPS=${EVAL}_maps
# Reads the frozen evaluation (checkpoint, manifest, seeds); writes only to MAPS. Re-running resumes.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PROJECT_DIR="${PROJECT_DIR:-$PWD}"; export PROJECT_DIR
cd "$PROJECT_DIR"
export ENV_DIR="${ENV_DIR:-/gpfsm/dnb10/projects/p311/ML_downscaling/env}"
export EVAL="${EVAL:-runs/merraflow_v4_1/evaluation/all_test_best_m16_s64}"
if [[ "$EVAL" != /* ]]; then export EVAL="$PROJECT_DIR/$EVAL"; fi
test -f "$EVAL/manifest.json" || { echo "No evaluation manifest in $EVAL" >&2; exit 2; }
export MAPS="${MAPS:-${EVAL}_maps}" SHARDS="${SHARDS:-8}" EVERY="${EVERY:-3}" MEMBERS="${MEMBERS:-8}"
if [[ "$MAPS" != /* ]]; then export MAPS="$PROJECT_DIR/$MAPS"; fi
for number in "$SHARDS" "$EVERY" "$MEMBERS" "${MAX_PARALLEL:-4}"; do
  if [[ ! "$number" =~ ^[1-9][0-9]*$ ]]; then echo "Shard/member/stride/concurrency values must be positive integers" >&2; exit 2; fi
done
# Run a private copy of the script so later edits to scripts/ cannot change a running pass.
mkdir -p "$MAPS" logs_v4_1
maps_tools=$(mktemp -d "$MAPS/tools.XXXXXX")
cp scripts/evaluate_maps_v4_1.py "$maps_tools/evaluate_maps_v4_1.py"; export MAPS_SCRIPT="$maps_tools/evaluate_maps_v4_1.py"
array=$(env -u SLURM_MEM_PER_CPU -u SLURM_MEM_PER_NODE -u SLURM_MEM_PER_GPU \
  sbatch --parsable --export=ALL --array="0-$((SHARDS-1))%${MAX_PARALLEL:-4}" scripts/slurm_evaluate_maps_v4_1.sh)
array=${array%%;*}
combine=$(env -u SLURM_MEM_PER_CPU -u SLURM_MEM_PER_NODE -u SLURM_MEM_PER_GPU \
  sbatch --parsable --export=ALL --dependency="afterany:$array" --account=s3292 --partition=compute \
  --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=32G --time=01:00:00 --job-name=eval_maps_combine \
  --output=logs_v4_1/eval_maps_combine_%j.log \
  --wrap "cd '$PROJECT_DIR' && PYTHONPATH='$PROJECT_DIR/src' '$ENV_DIR/bin/python' '$MAPS_SCRIPT' combine --evaluation '$EVAL' --output '$MAPS'")
combine=${combine%%;*}
printf 'Maps array: %s (%s shards, every %sth hour, %s members); combine: %s\nOutput: %s/maps.npz\n' \
  "$array" "$SHARDS" "$EVERY" "$MEMBERS" "$combine" "$MAPS"
