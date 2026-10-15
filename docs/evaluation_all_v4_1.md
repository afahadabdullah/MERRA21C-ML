# Full-test CONUS evaluation and paper contribution framing

The earlier presentation evaluated one wettest hour and two randomly chosen hours. The new `merraflow.evaluate_all_v4_1` evaluates **every eligible hour in the selected split**, without ranking or selecting cases. It checks temperature, precipitation, surface pressure, both wind components, specific humidity, and wind speed derived separately for each member.

## Run on Discover

Install Cartopy in the existing project environment once with `python -m pip install -e '.[paper]'` . Initialization checks Cartopy before scheduling GPU evaluation. Production data/checkpoints are not present on the local Mac; these commands perform the real evaluation on Discover.

```bash
cd /gpfsm/dnb10/projects/p311/ML_downscaling
export PYTHONPATH="$PWD/src"

OUTPUT=runs/merraflow_v4_1/evaluation/all_test_best_m16_s64 \
CHECKPOINT=best MEMBERS=16 STEPS=64 POST=both SHARDS=64 MAX_PARALLEL=4 SAVE_CASES=auto \
bash scripts/submit_evaluate_all_v4_1.sh
```

The launcher queues a CPU initialization job to snapshot the checkpoint and save a training-only spectral reference, followed by a dependent one-GPU-per-task array and a CPU aggregation/figure job. It uses the existing project's Discover account, partition, environment, and 12-hour limits. `MAX_PARALLEL` limits concurrent GPUs. Sixty-four shards is a starting allocation, not a measured runtime guarantee; use the first completed hours to estimate cost. Initialization and aggregation use the compute partition with 32 GB RAM and no GPU request.

Initialization executes on a CPU compute node, not the login node. Do not initialize two processes into the same output directory. `best` is recommended for a final test; use `CHECKPOINT=latest` or an explicit epoch/path for a clearly labeled development comparison. The snapshot ensures later training updates to best/latest cannot change the evaluated weights. `POST=none` disables filtering; `POST=both` evaluates raw flow and the spectral variant using exactly the same generated members, with no second inference pass.

## Check progress without rerunning evaluation

The standalone standard-library checker reads the frozen manifest and saved per-hour JSON. It requires no GPU, model, NumPy, or original weather data, and writes nothing into the evaluation directory:

```bash
python scripts/check_evaluation_v4_1.py \
  --output runs/merraflow_v4_1/evaluation/all_test_best_m16_s64 \
  --shards 64
```

It reports completed/expected hours and percentage, remaining and invalid hours, unexpected files, excluded hours, lock owners (`--details`), per-shard totals (`--shards ... --details`), recent completions, and aggregation status. A completed hour must match the manifest's run ID and timestamp and contain every required method, all six variables plus derived wind speed, and the expected member counts. Saved example arrays must exist. `--verify-files` additionally hashes those arrays and the saved grid; this may read several GB and is unnecessary for routine progress checks. The authoritative final aggregation still performs its complete validation. A cached `completion.json` never overrides the observed case records.

For a running checkout, fetch and extract **only the checker** so the evaluator's frozen source files stay unchanged:

```bash
git fetch origin main
git show origin/main:scripts/check_evaluation_v4_1.py > /tmp/check_merraflow_evaluation.py
python /tmp/check_merraflow_evaluation.py \
  --output runs/merraflow_v4_1/evaluation/all_test_best_m16_s64 \
  --shards 64
```

Add `--watch 60` to refresh every minute, `--json` for machine-readable snapshots, or `--details` for missing IDs, invalid records, locks, and each shard. To check queued/running/failed array tasks as well, add `--job-id YOUR_NUMERIC_ARRAY_ID`; this runs read-only `squeue` and `sacct` queries and reports states such as `PENDING`, `RUNNING`, `TIMEOUT`, and `FAILED`. File progress alone cannot distinguish a queued job from a stopped job. Lock directories may belong to active workers or interrupted jobs; the checker never removes them or labels them stale based only on age.

`hours_saved` means every expected case record passed the selected checks; `aggregated` also requires a complete `summary.json` for the same run. Neither status proves the scheduler job is currently running. Missing `manifest.json` reports initialization pending/wrong path, with an unknown total rather than a misleading 0% claim. One-shot exit codes are 0 for a readable snapshot (including partial progress), 1 for invalid/unexpected records or a bad saved grid hash, and 2 for a read/manifest error. Ctrl-C stops watch mode.

## Increase parallelism on the existing array

The launcher defaults to `SHARDS=64 MAX_PARALLEL=4`: 64 disjoint hour assignments, with at most four GPU workers running at once. `MAX_PARALLEL` affects submission only. To raise an existing array's cap without restarting workers or changing shard assignments, use its **parent array job ID**:

```bash
scontrol show job ARRAY_JOB_ID
scontrol update JobId=ARRAY_JOB_ID ArrayTaskThrottle=16
squeue -r -j ARRAY_JOB_ID -o "%.20i %.10T %.35R"
```

Replace `ARRAY_JOB_ID` with the actual parent ID and confirm the job is the intended `eval_all_v41` array before updating. Slurm supports updating `ArrayTaskThrottle` on an existing array ([scontrol reference](https://slurm.schedmd.com/scontrol.html)). Increasing this cap leaves the frozen checkpoint/settings, completed records, live workers, shard assignment, and finalizer dependency in place. Availability, account/QoS GPU limits, and shared storage throughput determine how many extra tasks can actually start and the resulting speedup. Queue reasons distinguish an array throttle from resource/account limits. Start with 8 or 16 concurrent tasks and compare the observed completion rate.

At eight completed test hours per wall-clock hour with four workers, 663 remaining hours take about 83 hours. Ideal linear scaling gives about 41 hours at eight workers or 21 hours at sixteen, assuming all workers are scheduled and throughput per worker holds. These are estimates from a recent rate, not measured guarantees. Increasing workers reduces elapsed time; it does not reduce total GPU work and can increase contention.

Do not submit a second overlapping array into the same output while this array is live: the evaluator deliberately rejects competing locks. To change `SHARDS`, wait until the old array stops and use the documented resume procedure. Keep the frozen 16 members and 64 integration steps for this run; changing them would define a different experiment. The worker's runtime tile batch (`BATCH`, default 8) and CPU threads (`THREADS`, default 4) can be benchmarked for a later resume, but they are not changed by increasing the array throttle.

The checker also reports mean saved-hour total time versus ensemble sampling time. The remainder includes input preparation, regression, verification, filtering, and I/O; it is not purely CPU time. If sampling dominates, test GPU utilization and tile-batch memory headroom. If the remainder dominates, investigate scoring, data reads, and output compression before adding more GPUs. These recorded timings exclude job startup and unsaved/interrupted work, and do not identify a bottleneck by themselves.

## Resume evaluation

Resume after interrupted/failed array tasks:

```bash
OUTPUT=runs/merraflow_v4_1/evaluation/all_test_best_m16_s64 \
RESUME=1 SHARDS=64 MAX_PARALLEL=4 \
bash scripts/submit_evaluate_all_v4_1.sh
```

Completed hours are checked and skipped. An interrupted hour is regenerated with the same seeds. A hard-killed worker may leave a `cases/<hour>.lock/` directory: inspect its `owner.json`, confirm the corresponding job is no longer running, then remove that specific stale lock before resuming. Never clear locks belonging to live jobs. If a task repeatedly exceeds the walltime, increase `SHARDS` on resume **after the previous array has stopped**; assignment changes, saved hours remain valid. Runtime batch/threads may change, but code, data fingerprints, checkpoint, membership, and statistical settings may not. Do not edit the code while the array is running; workers reject code changes. Use a new output for a different experiment.

Manual equivalent:

```bash
python -m merraflow.evaluate_all_v4_1 init \
  --config configs/discover_v4_1.yaml --checkpoint best \
  --output runs/full_test --members 16 --steps 64 --post both
python -m merraflow.evaluate_all_v4_1 run \
  --output runs/full_test --shard-index 0 --shards 1 --device cuda
python -m merraflow.evaluate_all_v4_1 aggregate \
  --output runs/full_test --draws 1000 --block-hours 72
```

Initialization is CPU preparation and may read 12 full training reference fields for the spectral estimate. `run` is the GPU-intensive stage. `aggregate` needs only saved JSON and does not load the model or original meteorological fields. Compact per-hour diagnostics are saved for all hours. By default, lossless full-resolution fields are also saved for the first, middle, and last eligible hours, chosen before inference. `SAVE_CASES=none` omits these; `SAVE_CASES=all` saves every ensemble (substantial storage); `SAVE_CASES=20260209_2030,20260306_1730` requests explicit eligible IDs. This setting is frozen at initialization.

## Interim paper while evaluation is running

To populate the paper from whatever hours are already complete, use the separate [interim snapshot and paper builder](interim_paper_v4_1.md). It automatically scans and freezes completed records, generates all aggregate plots and available examples, and fills the paper with explicitly interim results. It never changes the live evaluator or marks a partial run complete.

## Generate publication figures without inference or evaluation

The final CPU job automatically generates the figures after successful complete aggregation. Figure 1 restores the README’s map-based design with native coarse temperature/wind maps, a matched regional native-versus-fine temperature zoom, all six generated fields, and the correct v4.1 targets/training/sampling. Its numerical-data version includes fixed raw temperature members. The checked-in README and paper use actual archived GEOS/v4.1 plots, with T2m as the matched coarse/reference comparison in the supplied four-panel workflow layout. Zooms use the same extent and color scale, and render the source cells without smoothing. Figures 2–8 cover complete-test skill, uncertainty, spectra, ranks, precipitation reliability/FSS and filtering. Figures 9–10 give all-variable maps and spread/error panels for each predeclared saved case. Value scales are shared across methods; pressure/humidity are converted to hPa/g kg⁻¹. Case maps retain native LCC array geometry and saved latitude/longitude contours. The workflow uses Cartopy and saved native/fine coordinates; boundary overlays use bundled public-domain Natural Earth geometry, with no downloads. Its `workflow_data.npz` and `workflow_provenance.json` preserve all numerical fields, zoom bounds and source labels for independent redraws.

Regenerate figures locally or on a CPU node after changing plot styling:

```bash
export PYTHONPATH="$PWD/src"
python -m merraflow.paper_figures_v4_1 \
  --evaluation runs/merraflow_v4_1/evaluation/all_test_best_m16_s64 \
  --output runs/merraflow_v4_1/evaluation/all_test_best_m16_s64/paper_figures --dpi 600
```

Submit the plotting stage alone on Discover (make `logs_v4_1` first):

```bash
mkdir -p logs_v4_1
OUTPUT=runs/merraflow_v4_1/evaluation/all_test_best_m16_s64 \
FIGURE_DPI=600 sbatch --export=ALL scripts/slurm_paper_figures_v4_1.sh
```

Regenerate the checked-in GEOS-FP/GEOS-replay/prediction workflow from the archived plot data:

```bash
python -m pip install -e '.[paper]'
python scripts/make_workflow_figure.py \
  --presentation present_last_v4_1 --case-id 20260307_1930 \
  --output output/figures/workflow_geos_v4_1 --dpi 300
```

This produces the supplied four-panel layout from original vector PDF panels. The paper and README use that same full figure; `fig01_workflow_paper.*` files are identical aliases rather than a separate layout. Their map detail, shared scales and extents are preserved. The provenance JSON records every source hash and clip. No HRRR fields, synthetic weather, or new metrics enter this figure. The selected event is a preliminary illustration, not a complete-test result.

For a subsequent redraw of a numerical GEOS workflow bundle:

```bash
python -m merraflow.paper_figures_v4_1 --workflow-only \
  --workflow-data runs/full_test/paper_figures/workflow_data.npz \
  --output output/figures/workflow_redraw --dpi 600
```

The plotting program loads only saved JSON/NPZ: it never loads a model/checkpoint, original weather archives, or recomputes verification. `--skip-maps` produces the aggregate figures from older complete evaluations without saved fields. `figure_data.json` contains every aggregated numeric diagnostic plotted; `table01_all_variables.csv` provides converted paper units. Keep the **entire evaluation directory**, especially manifest, all case JSON, fields, spectral reference and summary, to reproduce both analysis and figures. `plot_manifest.json` records code/library versions, hashes, DPI and run identity. The saved checkpoint is needed only to reproduce inference. Changing statistical settings requires reaggregation of case JSON; changing plot appearance only requires plotting.

Three saved 16-member examples require approximately **4.8 GB before compression** on the 1059×1799 grid with both raw/spectral ensembles plus reference/baselines. Saving every test hour may require terabytes. `SAVE_CASES=none` permits aggregate figures; supply a saved `--workflow-data` bundle for Figure 1. It cannot recover case maps later without inference. All real-data figures require a completed production run; local synthetic QA is not scientific evidence.

The source tests cover frozen checkpoint, field/hash provenance, resume, corruption rejection, complete-run enforcement, numerical metrics, scheduler dependencies and rendering the full figure suite. Synthetic plot QA establishes that exports work, not that the production plots demonstrate skill. Inspect actual rain-bin overflows, state scales and rare-event bin counts before selecting paper panels; captions describe statistical limits.

## Outputs and reproducibility

- `manifest.json`: complete eligible hour list, exclusion reasons for indexed split hours, model/configuration/source hashes, source archive fingerprints, spectral training dates, and fixed statistical settings. Missing hours absent from the original archive index must still be reviewed in the preprocessing missing-hour audit; the evaluator cannot discover unindexed source timestamps.
- `checkpoint.pt`: independent frozen copy of the selected checkpoint.
- `spectral_reference.npz`: fixed training-only spectrum and its manifest hash, when enabled. Reference selection is anchored to the first eligible evaluation date's calendar season, with no evaluation labels. This is one fixed reference for the run; the previous three-event presentation used a different first-case anchor. The helper's same-split fallback is explicitly forbidden here.
- `cases/<timestamp>.json`: atomic per-hour scores for coarse input, frozen regression, raw flow, and optional spectral flow; member seeds; units; checkpoint epoch; sampling and total time.
- `fields/grid.npz` and `fields/<timestamp>_{raw,spectral}.npz`: saved coordinates/areas and full six-variable example fields, baselines, reference, all members, seeds and run ID, plus available original native GEOS-FP arrays and their native coordinates. Missing native variables are recorded and interpolated fallbacks are explicitly labeled. Figures never manufacture native coarse blocks from interpolated fields. Hashes are checked on resume and plotting. Raw data are saved before filtering, so variants remain paired.
- `paper_figures/`: vector PDF/SVG, 600-dpi PNG, figure data, converted-unit table, captions and plotting provenance.
- `completion.json`: authoritative completeness status and any missing/unexpected hours.
- `summary.json`, `scorecard.csv`, `report.md`: written by successful complete aggregation. Keep these with the manifest and per-hour JSON.

Scores use internal physical units: K, mm/h, Pa, m/s, m/s, kg/kg, and m/s. Convert pressure errors by 0.01 for hPa and humidity errors by 1000 for g/kg in paper tables. No temperature offset is applied to errors.

## Metrics and aggregation

Every variable includes ensemble-mean MAE, bias, pooled RMSE, empirical CRPS, fair CRPS, mean member MAE/MSE, population and unbiased variance, corrected spread/error, empirical central 90% coverage, randomized-tie ranks, domain means, radial power spectra, and 99th-percentile gradient magnitude. Wind speed is derived before ensemble averaging. Spectra and gradients include **all** members, not a selected sharp subset. Gradients are in field units per grid pixel; spectra use cycles per pixel, with approximate physical spacing recorded separately. Pixel quantiles are index-weighted and reported as **means of hourly quantiles**, not pooled time-space extremes or area-weighted quantiles.

Precipitation adds thresholds 0.1, 1, 5, 10, and 25 mm/h: Brier score, exact member-count probability reliability bins, observed-pixel counts, area-weighted contingency counts, CSI/POD/FAR/frequency bias, and FSS at 1, 5, and 10 mm/h for 1/5/17/33/65-pixel windows. All-dry FSS is undefined, not perfect. FSS pools squared-difference and reference terms before division; member FSS pools all member terms. This differs from averaging per-hour FSS and must be labeled accordingly.

Grid areas weight spatial errors; hours receive equal weight. RMSE is the square root of pooled MSE, not the average of hourly RMSE (the latter is retained separately). Reliability and contingency/FSS denominators pool before ratios. Spread uses population-normalized variance, corrected consistently by `(M+1)/(M-1)` before comparing with ensemble-mean MSE. The legacy presentation used a different correction; do not combine its spread ratios with the new ones. Empirical CRPS scores the issued finite ensemble; fair CRPS addresses its finite-member pairwise term under independent/exchangeable draws. Central empirical 90% intervals with small ensembles do not have exactly 90% coverage under ideal calibration.

Paired bootstrap intervals use nonoverlapping **72-hour elapsed-time blocks**, retaining the full spatial dependence and pairing across methods. Report block count and test sensitivity to 48/96/120 hours using separate saved summaries. Intervals cover CRPS, MSE, and bias differences, including spectral minus raw. They are not adjusted for testing many fields/metrics, and fewer than five blocks produces an insufficient-blocks status. Gaps do not shorten the block duration. These intervals describe this held-out period, not all seasons.

The evaluation scores the complete LCC domain, including ocean. It does not yet create regional/land-only scores, object tracks, observational verification, joint thermodynamic/variogram scores, or temporal ensemble trajectories. The baseline regression is inherited and not capacity-matched to the flow. Do not interpret these outputs as a head-to-head benchmark against CorrDiff without training that comparator on matched data.

## What is distinctive, and what is shared with CorrDiff?

The strongest paper question is whether **variable-specific generative targets** improve conditional weather structure: direct square-root precipitation alongside regression residuals for temperature, pressure and winds, plus coarse-humidity residuals. The user reports better organized storms with direct precipitation in development experiments and more difficulty recovering sharp moisture features. These are worthwhile qualitative findings; the full raw-versus-spectral test measures filtering effects, but cannot alone prove that direct precipitation beats residual precipitation.

The original [CorrDiff paper](https://doi.org/10.1038/s43247-025-02042-5) establishes a regression-plus-residual diffusion framework. This project builds on that structure and replaces the generative stage with conditional flow matching while using different target coordinates by variable. The original CorrDiff study includes radar reflectivity as an output, so radar synthesis is not a direct precipitation-rate baseline. Claims should specify the comparator version and variables.

Generating ensembles is useful and shared with CorrDiff. Similarly, [NVIDIA's later CorrDiff implementation](https://developer.nvidia.com/blog/gen-ai-super-resolution-accelerates-weather-prediction-with-scalable-low-compute-models/) describes patch-based multidiffusion. Our synchronized tiled velocity sampler is an implementation choice for flow matching, not an established first use of overlap synchronization. The packed loader is an engineering contribution. Post-training bounded spectral adjustment is a practically useful experiment, not a new general filtering principle. No literature-wide priority or speed advantage has been established.

## Tests needed to substantiate the precipitation finding

Train a matched residual-rain ablation using the **same v4.1 data, hourly target, history, network, optimizer, sample budget, loss weights, and random seeds**. In one branch generate `g(P)`; in the other generate `g(P)-g(R_P)` with consistent decoding and clearly specified scaling. Choose checkpoints and filters on validation. Compare raw branches first, then apply the same filter protocol. Use more than one training seed where affordable. The earlier model generations change multiple factors and cannot isolate this effect; legacy accumulated-APCP experiments are not a valid reference.

Primary outcomes should combine precipitation CRPS and bias with FSS, spectra, gradients, wet area, and storm-object structure. Check that any gain in organization does not come from too much rain or merely amplified noise. Quantify humidity gradient/power deficits, CRPS, dry bias, and coverage alongside the same diagnostics for temperature/winds. Keep the mixed finding that precipitation CRPS improved while pointwise RMSE did not; do not replace it with a blanket improvement claim.

Global extension is future work only. This implementation and the manuscript revision concern CONUS.
