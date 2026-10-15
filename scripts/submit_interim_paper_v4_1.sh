#!/usr/bin/env bash
# Fetch origin/main first, then run this standalone launcher from the project root.
# It stages publishing tools/template only; src/merraflow stays frozen for live workers.
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$PWD}"
cd "$PROJECT_DIR"
export PROJECT_DIR
export OUTPUT="${OUTPUT:-runs/merraflow_v4_1/evaluation/all_test_best_m16_s64}"
if [[ "$OUTPUT" != /* ]]; then export OUTPUT="$PROJECT_DIR/$OUTPUT"; fi
if [[ ! -f "$OUTPUT/manifest.json" ]]; then
  echo "No evaluation manifest: $OUTPUT/manifest.json" >&2; exit 2
fi
publishing_ref="${PUBLISHING_REF:-origin/main}"
publishing_commit=$(git rev-parse --verify "$publishing_ref^{commit}")
mkdir -p "$PROJECT_DIR/runs" "$PROJECT_DIR/logs_v4_1"
publishing_tools=$(mktemp -d "$PROJECT_DIR/runs/interim_paper_tools.XXXXXX")
git show "$publishing_commit:scripts/build_interim_paper_v4_1.py" > "$publishing_tools/build_interim_paper_v4_1.py"
git show "$publishing_commit:scripts/slurm_interim_paper_v4_1.sh" > "$publishing_tools/slurm_interim_paper_v4_1.sh"
export INTERIM_BUILDER="$publishing_tools/build_interim_paper_v4_1.py"
export MANUSCRIPT="${MANUSCRIPT:-$PROJECT_DIR/manuscript/MERRA21C_ML_arxiv.tex}"
if [[ "$MANUSCRIPT" != /* ]]; then export MANUSCRIPT="$PROJECT_DIR/$MANUSCRIPT"; fi
mkdir -p "$(dirname "$MANUSCRIPT")/figures"
# Keep the old source and asset before installing the current paper from Git.
# KEEP_MANUSCRIPT=1 uses an already current, locally edited template instead.
if [[ "${KEEP_MANUSCRIPT:-0}" != 1 ]]; then
  if [[ -f "$MANUSCRIPT" ]]; then cp "$MANUSCRIPT" "$publishing_tools/manuscript_before_template_update.tex"; fi
  git show "$publishing_commit:manuscript/MERRA21C_ML_arxiv.tex" > "$publishing_tools/current_template.tex"
  cp "$publishing_tools/current_template.tex" "$MANUSCRIPT"
fi
workflow_asset="$(dirname "$MANUSCRIPT")/figures/MERRA21C_ML_workflow.pdf"
if [[ -f "$workflow_asset" ]]; then cp "$workflow_asset" "$publishing_tools/workflow_before_update.pdf"; fi
git show "$publishing_commit:manuscript/figures/MERRA21C_ML_workflow.pdf" > "$workflow_asset"
printf '%s\n' "$publishing_commit" > "$publishing_tools/publishing_commit.txt"
job=$(env -u SLURM_MEM_PER_CPU -u SLURM_MEM_PER_NODE -u SLURM_MEM_PER_GPU \
  sbatch --parsable --export=ALL "$publishing_tools/slurm_interim_paper_v4_1.sh")
job=${job%%;*}
printf 'Interim CPU paper job: %s\n' "$job"
printf 'Completed-hour count is discovered when the CPU job starts; no inference is rerun.\n'
printf 'Publishing tools/backup: %s\n' "$publishing_tools"
printf 'Updated source: %s\n' "$MANUSCRIPT"
printf 'Bundle: %s/MERRAFlow_interim_paper.zip\n' "${SNAPSHOT_OUTPUT:-${OUTPUT}_interim_${job}}"
printf 'Log: %s/logs_v4_1/paper_interim_%s.log\n' "$PROJECT_DIR" "$job"
