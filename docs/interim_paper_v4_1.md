# MERRA-Flow plots, score matrix, and local paper

The CPU job dynamically scans the running evaluation, freezes all completed valid hour records, and generates map comparisons, zooms, ensemble probabilities, gain maps, a skill matrix, temporal diagnostics, spectra, precipitation skill, and calibration. It never reruns model inference or modifies `src/merraflow`, whose Python files are hashed by live evaluators.

The entire `manuscript/` directory is ignored by Git. It stays on disk locally; source, figures, PDF exports, and the synthetic layout preview are not published. Existing Git history is unchanged. Plot generation works without a manuscript.

## Generate plots and metric matrices on Discover

From the existing evaluation checkout, fetch the publishing scripts without checking out new evaluator code:

```bash
cd /gpfsm/dnb10/projects/p311/ML_downscaling
git fetch origin main
git show origin/main:scripts/submit_interim_paper_v4_1.sh > /tmp/submit_interim_paper.sh
PLOTS_ONLY=1 FIGURE_DPI=300 bash /tmp/submit_interim_paper.sh
```

The launcher extracts the builder, plotter, boundary asset, and CPU job into a separate directory under `runs/`. It submits four CPUs, 32 GB, and two hours. It leaves running GPU workers and their source files unchanged. `OUTPUT` defaults to `runs/merraflow_v4_1/evaluation/all_test_best_m16_s64`. Override it to process another frozen evaluation. Set `FIGURE_DPI=600` for higher-resolution PNG exports; labels and line plots in PDF are vector graphics.

The launcher prints the actual snapshot and log paths. Outputs are under `<evaluation>_interim_<JOBID>/`:

- `paper_figures/fig_scorecard.pdf` and `.png`: annotated CRPS-skill / RMSE-reduction matrix and interval coverage.
- `paper_figures/score_matrix.csv`: the matrix values, percentages, and actual/expected hour counts.
- `paper_figures/table01_all_variables.csv`: physical-unit errors, CRPS, fair CRPS, bias, coverage, and corrected spread/error for every field and method.
- Other `paper_figures/fig_*.pdf` and `.png`: temporal, spectral, precipitation, calibration, and available predeclared case maps.
- `paper_figures/figures.json`: plotted diagnostics and map statistics; `figure_data.json`: pooled metrics; `plot_manifest.json`: data/code/figure hashes.
- `snapshot.json`, unchanged parent `manifest.json`, copied `cases/*.json`, and available predeclared `fields/*.npz`: reproducible frozen evidence. Later completed hours belong to a new snapshot.
- `MERRAFlow_plots_metrics.zip`: portable plots and metrics (large NPZ/NetCDF arrays remain in their saved directories).

The environment needs NumPy, SciPy, Matplotlib, and Pillow; optional storm NetCDF input also needs xarray and h5netcdf. Plotting needs no model load, GPU, Cartopy, or downloads. Existing saved map cases are used only if their hour has finished. Missing native GEOS-FP fields are labelled as interpolated inputs in the map metadata.

## Also update the local paper

Copy the current manuscript with `% BEGIN AUTO ...` blocks from the Mac to Discover once, and again whenever you revise its prose. Run this on the Mac, from the project folder:

```bash
scp manuscript/MERRA21C_ML_arxiv.tex \
  discover:/gpfsm/dnb10/projects/p311/ML_downscaling/manuscript/
scp manuscript/figures/MERRA21C_ML_workflow.pdf \
  discover:/gpfsm/dnb10/projects/p311/ML_downscaling/manuscript/figures/
```

The destination directories must already exist (`mkdir -p manuscript/figures` on Discover if needed). Then use the same fetched launcher on Discover, omitting `PLOTS_ONLY=1`:

```bash
bash /tmp/submit_interim_paper.sh
```

The launcher requires an existing compatible manuscript and preserves a backup. It never fetches manuscript files from Git or replaces local prose with an old template. The builder updates the AUTO blocks and copies figures into `manuscript/figures/auto/`. It creates `MERRAFlow_paper.zip`; `pdf/` is populated if Tectonic or pdfLaTeX is available. Otherwise compile the updated source in the desktop editor. `NO_INLINE=1` skips embedded PNG previews when only normal multi-file compilation is needed.

## Optional storm case (new GPU inference)

The already tracked storm job can generate the optional regional event; this is separate from the CPU plotting job:

```bash
cd /gpfsm/dnb10/projects/p311/ML_downscaling
EVAL=runs/merraflow_v4_1/evaluation/all_test_best_m16_s64
RUNS="MERRA-Flow=configs/discover_v4_1.yaml:$EVAL/checkpoint.pt" \
POST=none MEMBERS=3 HOURS=18 TRACK=0 REGION=512 \
OUTPUT=runs/merraflow_v4_1/evaluation/storm_case_paper \
sbatch --export=ALL scripts/slurm_storm_animation_v4_1.sh
```

`TIMESTAMP=YYYYMMDD_HHMM` selects a particular event; otherwise the storm tool selects the wettest test hour. Its runtime for this configuration has not been measured; the batch limit is three hours. When `storm_fields.nc` exists, include it in a CPU run:

```bash
STORM=runs/merraflow_v4_1/evaluation/storm_case_paper/storm_fields.nc \
PLOTS_ONLY=1 bash /tmp/submit_interim_paper.sh
```

Omit `PLOTS_ONLY=1` to also populate the local manuscript. The plot manifest records the NetCDF hash; retain that file for reproducibility. The existing storm NetCDF does not record a checkpoint hash, so preserve its `storm.json`, command, and frozen checkpoint separately. A fixed noise seed retains a member label across hours; it does not establish learned temporal coherence.

## Redraw a frozen snapshot

To install scripts into the checkout, extract only the publishing files; do not pull or change evaluator code while GPU workers are running:

```bash
for name in paper_plots_v4_1.py build_interim_paper_v4_1.py; do
  git show "origin/main:scripts/$name" > "scripts/$name"
done
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
./env/bin/python scripts/build_interim_paper_v4_1.py \
  --snapshot runs/merraflow_v4_1/evaluation/all_test_best_m16_s64_interim_JOBID \
  --plots-only --dpi 300
```

For the paper, replace `--plots-only` with `--manuscript manuscript/MERRA21C_ML_arxiv.tex --compile`. Add `--storm <saved storm_fields.nc>` when needed. Redrawing checks saved hashes and reads saved verification/fields without new inference. Spatial maps and hourly diagnostics are derived from those saved arrays/records. Submit a new snapshot to incorporate more completed hours.

Partial snapshots remain explicitly interim and completion-ordered, with counts in visible figure footers, captions, and PDF metadata. They carry `complete: false`; no score confidence intervals are inferred. Once every eligible hour is present, the paper describes all eligible hours, but this descriptive publishing path still does not replace the strict full-test finalizer or its block-bootstrap analysis. RMSE pools MSE; CRPS skill uses interpolated GEOS-FP as baseline. The reference is the high-resolution GEOS replay, not independent observations.

## All-hours error and skill maps

The full-test evaluator saves only domain-mean scores per hour, plus full fields for the three predeclared example hours. Per-pixel error maps over many hours therefore need the fields again, which `scripts/evaluate_maps_v4_1.py` regenerates for every `EVERY`-th test hour with the same frozen checkpoint, code, sampler and member seeds (first `MEMBERS` members). It accumulates per-pixel CRPS, GEOS-FP/regression MAE, squared errors, ensemble variance and 1 mm/h Brier scores, and writes only to its own directory.

Partial data is fine at every stage:

* Each shard saves its running sums after every hour, so a killed or time-limited job loses at most one hour; resubmitting the same command resumes.
* Keep `SHARDS`, `EVERY`, `MEMBERS`, and the maps script unchanged when resuming. The saved plan rejects incompatible changes, and a per-shard lock prevents simultaneous writers. Use a new `MAPS` directory for a different experiment.
* Give the paper builder the maps **directory** (`ERROR_MAPS=..._maps`). It combines whatever hours are finished at that moment, and the caption says "N of P planned hours finished".
* With no GPU pass at all, `python scripts/evaluate_maps_v4_1.py from-saved --evaluation EVAL --output EVAL_saved_maps` builds the same maps from whichever predeclared example hours are complete, using all their saved members. This sample has a separate scope from the deterministic maps-pass subsample.

Install only these publishing scripts on Discover before using the commands below. Fetching and extracting individual files preserves the live evaluator's source hash:

```bash
cd /gpfsm/dnb10/projects/p311/ML_downscaling
git fetch origin main
for f in evaluate_maps_v4_1.py submit_evaluate_maps_v4_1.sh slurm_evaluate_maps_v4_1.sh \
         build_interim_paper_v4_1.py paper_plots_v4_1.py \
         submit_interim_paper_v4_1.sh slurm_interim_paper_v4_1.sh; do
  git show "origin/main:scripts/$f" > "scripts/$f"
done
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
```

```bash
cd /gpfsm/dnb10/projects/p311/ML_downscaling
EVAL=runs/merraflow_v4_1/evaluation/all_test_best_m16_s64 EVERY=3 MEMBERS=8 SHARDS=8 \
  bash scripts/submit_evaluate_maps_v4_1.sh
# Any time later, with whatever hours are done:
ERROR_MAPS=runs/merraflow_v4_1/evaluation/all_test_best_m16_s64_maps \
  PUBLISHING_REF=worktree bash scripts/submit_interim_paper_v4_1.sh
```

The default throttle is four simultaneous GPU jobs across eight shards. Set `MAX_PARALLEL=8` on submission to allow all eight to run together. For 815 eligible hours, `EVERY=3` selects 272. At an assumed 100 s per member this is about 60 A100-hours: roughly 15 hours at four-way concurrency or 8 hours at eight-way concurrency, excluding queues and overhead. This maps-pass runtime has not been measured.

The paper job requires at least one completed map hour and an existing local manuscript with AUTO blocks; add `PLOTS_ONLY=1` if no manuscript is installed. It copies the consolidated map arrays into its own `error_maps.npz` with a hash and includes them in the results ZIP. Redraw an existing paper snapshot without `--maps` to reuse that exact map sample; submit a new snapshot to incorporate later map completions. Map captions disclose their separate hour and member counts. A partial map pass is not a full-test result even if the domain-score snapshot is complete.

The GPU pass verifies the frozen checkpoint, archive fingerprints, and evaluator source hash. It refuses changed source by default; preserve that check for the paper evaluation.
