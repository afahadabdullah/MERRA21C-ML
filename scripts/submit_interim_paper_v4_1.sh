#!/usr/bin/env bash
# Fetch origin/main first, then run this standalone launcher from the project root.
# It stages publishing tools only; src/merraflow stays frozen for live workers.
#   STORM=<storm_fields.nc>      add the storm case study (from scripts/slurm_storm_animation_v4_1.sh)
#   ERROR_MAPS=<maps dir|npz>    add error/skill maps; a maps directory uses the hours finished so far
#   PUBLISHING_REF=worktree      use checked-out scripts instead of origin/main
#   PLOTS_ONLY=1                generate plots and metric matrices without a manuscript
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$PWD}"
cd "$PROJECT_DIR"
export PROJECT_DIR
export OUTPUT="${OUTPUT:-runs/merraflow_v4_1/evaluation/all_test_best_m16_s64}"
if [[ "$OUTPUT" != /* ]]; then export OUTPUT="$PROJECT_DIR/$OUTPUT"; fi
if [[ ! -f "$OUTPUT/manifest.json" ]]; then
  echo "No evaluation manifest: $OUTPUT/manifest.json" >&2; exit 2
fi
if [[ -n "${STORM:-}" ]]; then
  if [[ "$STORM" != /* ]]; then STORM="$PROJECT_DIR/$STORM"; fi
  if [[ ! -f "$STORM" ]]; then echo "No storm file: $STORM" >&2; exit 2; fi
  export STORM
fi
if [[ -n "${ERROR_MAPS:-}" ]]; then
  if [[ "$ERROR_MAPS" != /* ]]; then ERROR_MAPS="$PROJECT_DIR/$ERROR_MAPS"; fi
  if [[ ! -e "$ERROR_MAPS" ]]; then echo "No error maps file or directory: $ERROR_MAPS" >&2; exit 2; fi
  export ERROR_MAPS
fi
publishing_ref="${PUBLISHING_REF:-origin/main}"
export MANUSCRIPT="${MANUSCRIPT:-$PROJECT_DIR/manuscript/MERRA21C_ML_arxiv.tex}"
if [[ "$MANUSCRIPT" != /* ]]; then export MANUSCRIPT="$PROJECT_DIR/$MANUSCRIPT"; fi
if [[ "${PLOTS_ONLY:-0}" != 1 ]]; then
  if [[ ! -f "$MANUSCRIPT" ]] || ! grep -q '^% BEGIN AUTO RESULTS$' "$MANUSCRIPT"; then
    echo "Copy your current local manuscript (with AUTO blocks) to $MANUSCRIPT, or set PLOTS_ONLY=1." >&2
    exit 2
  fi
fi
if [[ "$publishing_ref" == worktree ]]; then
  publishing_commit=worktree
  fetch() { cat "$PROJECT_DIR/$1"; }
else
  publishing_commit=$(git rev-parse --verify "$publishing_ref^{commit}")
  fetch() { git show "$publishing_commit:$1"; }
fi
mkdir -p "$PROJECT_DIR/runs" "$PROJECT_DIR/logs_v4_1"
publishing_tools=$(mktemp -d "$PROJECT_DIR/runs/interim_paper_tools.XXXXXX")
fetch scripts/build_interim_paper_v4_1.py > "$publishing_tools/build_interim_paper_v4_1.py"
fetch scripts/paper_plots_v4_1.py > "$publishing_tools/paper_plots_v4_1.py"
fetch scripts/evaluate_maps_v4_1.py > "$publishing_tools/evaluate_maps_v4_1.py"
fetch src/merraflow/assets/conus_boundaries_50m.json.gz > "$publishing_tools/conus_boundaries_50m.json.gz"
fetch scripts/slurm_interim_paper_v4_1.sh > "$publishing_tools/slurm_interim_paper_v4_1.sh"
export INTERIM_BUILDER="$publishing_tools/build_interim_paper_v4_1.py"
if [[ "${PLOTS_ONLY:-0}" != 1 ]]; then
  cp "$MANUSCRIPT" "$publishing_tools/manuscript_before_update.tex"
fi
printf '%s\n' "$publishing_commit" > "$publishing_tools/publishing_commit.txt"
job=$(env -u SLURM_MEM_PER_CPU -u SLURM_MEM_PER_NODE -u SLURM_MEM_PER_GPU \
  sbatch --parsable --export=ALL "$publishing_tools/slurm_interim_paper_v4_1.sh")
job=${job%%;*}
printf 'Interim CPU paper job: %s\n' "$job"
printf 'Completed-hour count is discovered when the CPU job starts; no inference is rerun.\n'
printf 'Publishing tools/backup: %s (%s)\n' "$publishing_tools" "$publishing_commit"
if [[ "${PLOTS_ONLY:-0}" != 1 ]]; then printf 'Manuscript to update: %s\n' "$MANUSCRIPT"; fi
if [[ -n "${STORM:-}" ]]; then printf 'Storm case study: %s\n' "$STORM"; fi
archive=MERRAFlow_paper.zip
if [[ "${PLOTS_ONLY:-0}" == 1 ]]; then archive=MERRAFlow_plots_metrics.zip; fi
printf 'Bundle: %s/%s\n' "${SNAPSHOT_OUTPUT:-${OUTPUT}_interim_${job}}" "$archive"
printf 'Log: %s/logs_v4_1/paper_interim_%s.log\n' "$PROJECT_DIR" "$job"
