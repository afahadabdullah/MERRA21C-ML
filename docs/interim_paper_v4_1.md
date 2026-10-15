# MERRA-Flow paper from whatever evaluation hours have finished

Run the interim CPU job while the full-test GPU array continues. It scans `cases/*.json`, freezes the IDs present at snapshot start, validates each against the original manifest, and aggregates those saved records. There is no fixed 156-hour requirement and no new inference. New completions after the ID list is frozen belong to a later snapshot.

The full-test aggregator and plotter retain their existing completeness checks. Interim output is a separate immutable sibling directory, carries `complete: false`, retains the original 815-hour manifest/run ID, and records its own snapshot ID, included hours, data/code hashes, and available predeclared example fields. Partial data are never presented as a complete test run. Confidence intervals and significance claims are withheld because completion order is not random sampling.

## Submit on Discover without modifying live evaluator code

From the existing project checkout on Discover:

```bash
cd /gpfsm/dnb10/projects/p311/ML_downscaling
git fetch origin main
git show origin/main:scripts/submit_interim_paper_v4_1.sh > /tmp/submit_interim_paper.sh
bash /tmp/submit_interim_paper.sh
```

`OUTPUT` defaults to `runs/merraflow_v4_1/evaluation/all_test_best_m16_s64`. Set it before the final command if your run directory differs. This submits a four-core, 32-GB CPU job with no GPU request or dependency on the live array. It discovers the completed-hour count when it starts. Output is `.../all_test_best_m16_s64_interim_JOBID`, printed by the launcher, with log `logs_v4_1/paper_interim_JOBID.log`.

The launcher extracts only publishing scripts and the current paper/template asset from the fetched commit. It does **not** check out, pull, or edit `src/merraflow`, so running evaluators keep their frozen code. It backs up an existing manuscript and workflow asset in the printed `runs/interim_paper_tools.*` directory, then updates the existing `manuscript/MERRA21C_ML_arxiv.tex` in place. Set `KEEP_MANUSCRIPT=1` to keep a locally edited, compatible current template instead of installing the latest template. Do not run two paper-building jobs against the same manuscript simultaneously.

For example, aggregate plots/tables only (no saved example arrays/maps):

```bash
SKIP_MAPS=1 bash /tmp/submit_interim_paper.sh
```

The environment needs the existing NumPy/SciPy/Matplotlib/Pillow dependencies. It does not require a model, a checkpoint load, Cartopy, or PyMuPDF for this pipeline. A PDF is exported if `tectonic` or `pdflatex` is installed. Otherwise the updated, self-contained TeX and all figures are still exported for compilation in the desktop editor. Author declarations requiring human information remain explicitly unresolved; they are not invented.

## Outputs

- `snapshot.json`: exact included IDs, parent identity, creation time, case/field/summary hashes, and completion-order scope.
- `manifest.json`: unchanged original full-test manifest.
- `cases/*.json`: byte-preserved per-hour scores; all six variables plus derived wind speed, all methods, and original member counts must validate.
- `fields/*.npz`: only available predeclared map examples and their saved grid; raw/spectral seeds and hashes are retained. Missing unfinished examples are not regenerated or selected by score.
- `summary.json`: pooled interim diagnostics with `scope: interim`, `complete: false`, and actual/expected hour counts.
- `paper_figures/`: temporal coverage, skill, calibration, spectra, ranks, reliability, FSS, sharpening, and available seven-variable case/spread maps in PDF/PNG/SVG. Every plot carries an interim subset banner. `figure_data.json` and `table01_all_variables.csv` save plotted/printed values with units.
- `manuscript_before_update.tex`: source backup before numerical population.
- `manuscript_update.json`: snapshot ID and populated manuscript path/counts.
- `pdf/`: compiled paper when a TeX engine is available.
- `MERRAFlow_interim_paper.zip`: lightweight updated source, figure PDFs, metrics JSON/CSV, included case JSON, provenance, and optional compiled paper. It omits the large NPZ arrays, which remain in the complete snapshot directory.

The updater fills all four data/settings/score/processing tables and replaces all four blank result/support figures. It adds the remaining aggregate diagnostic plots and temperature/precipitation examples when those predeclared fields are already available. It replaces the abstract, results, conclusions, and case-specific methodological statements using the actual interim summary. The workflow map illustration retains its explicitly separate development checkpoint; it is not relabeled as the quantitative validation-best run.

Do not infer complete-test or observational skill from this snapshot. Per-hour pooled RMSE, empirical versus fair CRPS, finite-member spread correction, temporal support, and model-reference limitations remain explicit in the paper. Representation/history/context ablations cannot be reconstructed from this run and are not invented; the fourth table reports the available paired raw/spectral effects instead.

## Regenerate without analysis or inference

Use the saved snapshot to restyle and repopulate the paper:

```bash
python scripts/build_interim_paper_v4_1.py \
  --snapshot runs/merraflow_v4_1/evaluation/all_test_best_m16_s64_interim_JOBID \
  --manuscript manuscript/MERRA21C_ML_arxiv.tex --dpi 600 --compile
```

This checks hashes, loads `summary.json` and saved example fields, redraws plots, and updates the existing source. It does not recompute verification or sampling. To include more completed hours later, submit a **new** snapshot. The original full-test finalizer remains separate and still requires all eligible hours.
