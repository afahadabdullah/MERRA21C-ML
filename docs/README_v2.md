# MERRAflow v2: regression plus residual flow

V2 is a separate experimental pipeline. Existing source files, CLI, configs,
prepared archives, training jobs and checkpoints are unchanged. Invoke
`python -m merraflow.cli_v2`; the existing `merraflow` entry point still runs v1.
New modules/configs/scripts and v2 output artifacts have `v2` names. V1 archives
and checkpoints cannot be loaded into v2. No cluster jobs are submitted by setup
or testing.

## What changes

| Component | V2 behavior |
|---|---|
| Targets | Original HWT midpoint T2M, PRECTOT, pressure and signed U/V; precipitation is never budget-adjusted |
| Baseline | Bilinear coarse precipitation, T2M, pressure and U/V; native precipitation is saved separately for audits |
| Static conditions | Existing elevation/coordinates/area plus ocean, land and lake fractions from one static file, lake-availability flag, grid-axis terrain slopes and signed distance to water |
| Dynamics | PRECTOT, U10M, V10M, TQV, QV2M, SLP and OMEGA500; full preparation rejects missing inputs |
| Regression | Multiscale conditional U-Net predicts the predictable transformed residual from the coarse baseline |
| Generation | A second U-Net learns flow matching on the remaining residual around the frozen best regression EMA |
| Architecture | Two residual blocks per scale, spatial conditions at every encoder scale, bilinear decoder upsampling, bottleneck self-attention and cross-attention to broader context |
| Local/context views | Default 128-pixel loss core, 32-pixel halo (192-pixel local input); a 576-pixel context view area-downsampled to 96 pixels |
| Sampling | 60% uniform / 40% rain/coast-rich proposal; exact inverse-proposal weights preserve the uniform candidate-patch objective |
| Inference | Shared full-field initial noise, overlapping tiles; optional single-owner stitching ablation; no mass projection or drizzle threshold |
| Verification | Original-HWT and regression-baseline scores, rainfall FSS/CSI/Brier/quantiles, CRPS, rank histograms, individual-member spectra, coastal errors and explicit budget discrepancies |

The decomposition is inspired by [CorrDiff](https://www.nature.com/articles/s43247-025-02042-5).
This is a conditional **flow matching** implementation, not a reproduction of
CorrDiff's EDM method. The attention implementation uses [PyTorch SDPA](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention).

## Surface fields: one ocean/land/lake file

All three production presets read the generated static NetCDF on the HWT LCC
grid. These are GSHHG-derived geographic fractions, not native GEOS surface
fractions. The PNG is a preview; preparation reads the full-resolution NetCDF.

```yaml
static:
  path: data/static_grid/hwt_surface_fractions_v2.nc
  ocean: FROCEAN
  lake: FRLAKE
  require_lake: true
  lat: lats
  lon: lons
```

The generator writes `FRLAND`, `FRLAKE` and `FROCEAN` in this one file. The loader
reads ocean and lake, derives land as `1 - FROCEAN - FRLAKE`, and checks finite
fractions in [0,1] and latitude/longitude agreement with HWT. Missing `FRLAKE`
now fails in the supplied presets. Mixed coastal cells retain fractional cover;
the generator defaults to 16 sub-cell samples per HWT cell. Keep the generated
file unchanged throughout preparation and training.

The loader still supports explicit alternative configurations with land/lake
fields, or optional lakes, but these are not the production defaults. Do not
pass category labels or percentages as fractions. Distance is measured in grid
pixels, and slopes are along LCC grid axes using geographic center spacing.

The [Discover runbook](runbook_discover_v2.md) includes the exact environment,
static-file checks, submission and recovery commands. The static NetCDF is
generated on Discover and is not downloaded by `git pull`.

## Losses and why they do not simply sharpen every image

Training targets are five standardized transformed residuals. Only precipitation
uses `log1p(P / precip_log_scale)`; U/V remain signed. Training-only statistics
normalize all five channels. The model estimates a transformed-space conditional
mean; decoding it is not exactly the conditional mean in physical rainfall units.

Stage 1 uses quadratic value error plus 0.1 times first-difference gradient error
and 0.1 times an average of 2x/4x pooled errors. Stage 2 keeps velocity matching as
the main objective and adds 0.05 times gradient error of the endpoint estimate
`x_hat_1 = x_t + (1-t) * v`. All terms are quadratic in errors. This gives spatial
structure additional emphasis while retaining the conditional-mean velocity
optimum; nonlinear spectral/perceptual penalties could change that optimum.
Spectral power is therefore assessed on actual generated members instead of
forcing each stochastic member to copy the exact target's rain-cell locations.

Flow residual RMS scales are calibrated on training patches after freezing the
best regression model and are stored in every flow checkpoint. Calibration
never uses validation/test data. Sampling importance weights correct toward
uniform **candidate patch locations**, not a claim of perfectly uniform pixel
coverage at domain edges. Validation uses uniform proposals and fixed locations,
noise and flow times. History includes separate value, gradient, multiscale and
five per-channel value losses. Regression and flow losses use different targets
and should not be compared numerically to one another.

## Run on Discover

For the complete copy/paste workflow, follow the
[Discover runbook](runbook_discover_v2.md). From the project root,
`bash scripts/submit_pipeline_v2.sh` activates the existing conda environment,
audits the inputs, then queues preparation, finalization, regression and flow
with `afterok` dependencies. All v2 jobs import `src` from the checked-out project.
The fresh regression job first benchmarks both stages for five scratch steps
on its allocated GPU, then trains. Failed dependencies cancel downstream jobs.

The commands below describe the individual stages. Use Bash and the existing
project environment (PyTorch >=2.3 and project dependencies). Set
`PYTHONPATH="$PWD/src"` when running the CLI directly.

First check representative inputs without writing prepared data:

```bash
python -m merraflow.cli_v2 audit --config configs/discover_v2.yaml
```

The default paths are separate:

```text
data/paired_hourly_v2/
runs/merraflow_v2/regression_v2/
runs/merraflow_v2/flow_v2/
runs/merraflow_v2/predictions_v2/
logs_v2/
```

Prepare serially, or use the v2 monthly array and dependent finalizer:

```bash
python -m merraflow.cli_v2 prepare --config configs/discover_v2.yaml
# Alternative to the serial command:
bash scripts/submit_prepare_flow_v2.sh
```

The array helper defaults to 2025, as do the supplied data ranges. Set PREP_YEAR
and the configuration together for another year; a multi-year archive needs an
array for each applicable year before finalization. Preparation resumes complete
shards and refuses incompatible metadata. Changes to external static-file bytes
invalidate resumable preparation. Full preparation checks predictors on all
paired hours. The read-only audit checks representative hours only.

The fresh regression batch job runs this benchmark automatically. To run it
separately, use a GPU allocation:

```bash
python scripts/benchmark_v2.py --config configs/discover_v2.yaml --steps 5
```

This measures scratch optimizer steps on an actual batch and reports CUDA peak
allocation. It is not a skill test or a guarantee of full-run memory headroom
(training also stores EMA/checkpoints). The 80 GB preset is an unbenchmarked
candidate, not a measured safe capacity.

Train the two stages in order:

```bash
python -m merraflow.cli_v2 train --config configs/discover_v2.yaml --stage regression
python -m merraflow.cli_v2 train --config configs/discover_v2.yaml --stage flow \
  --regression-checkpoint runs/merraflow_v2/regression_v2/best_v2.pt
```

For Slurm, create the log directory and submit regression, then flow after
regression has finished. Flow must read a completed, stable regression checkpoint:

```bash
mkdir -p logs_v2
STAGE=regression sbatch scripts/slurm_train_flow_v2.sh
# After regression finishes:
STAGE=flow REGRESSION_CHECKPOINT=runs/merraflow_v2/regression_v2/best_v2.pt \
  sbatch scripts/slurm_train_flow_v2.sh
```

No v1 training job is stopped or resumed by these scripts. Use the v2 `RESUME`
environment variable or CLI `--resume` for a v2 checkpoint. Exact resume requires
the same architecture, patch configuration, training/loss settings and DDP world
size. Runtime device/workers and output directory can differ. Checkpoints contain
optimizer, scheduler, scaler, RNG, EMA, frozen regression weights and residual
scales. Epoch recovery files are one-based (`epoch_0005_v2.pt`). Prefer resuming
in the same run directory to keep the earlier best checkpoint and complete history.

Generate a small fixed validation set first (eight members by default):

```bash
python -m merraflow.cli_v2 predict --config configs/discover_v2.yaml \
  --checkpoint runs/merraflow_v2/flow_v2/best_v2.pt --split val --limit 4
python -m merraflow.cli_v2 evaluate --config configs/discover_v2.yaml --split val
python -m merraflow.cli_v2 plot --config configs/discover_v2.yaml --split val
```

For Slurm, use `CHECKPOINT=... LIMIT=4 sbatch scripts/slurm_predict_v2.sh`.
Use a new v2 prediction directory for each checkpoint/step-count comparison.
Prediction refuses to overwrite existing members; evaluation rejects mixed
checkpoints/settings or incomplete ensembles and lists missing hours. Validation
and test predictions can share a directory because their timestamps are disjoint.
`best_v2.pt` is selected by validation training objective; use physical validation
metrics to decide which checkpoint is scientifically preferable. Never choose
hyperparameters by repeatedly tuning on the December held-out test event.

Unlabeled future hours are also supported: copy the v2 config, change date/source
paths and `data.prepared` to a fresh directory ending in v2, then run
`prepare-predict --reference-archive data/paired_hourly_v2`. Predict with
`--split predict`. The frozen static grid, fingerprint and normalization are
reused without opening HR labels. Evaluation requires actual held-out HR truth.

## Would smaller patches help?

Possibly, but a smaller field of view also removes storm organization. Three
independent candidate configs are supplied:

| Config | Loss core | Local input | Broad context | Purpose |
|---|---:|---:|---:|---|
| discover_v2.yaml | 128 | 192 | 576 → 96 | Starting point |
| small_core_v2.yaml | 64 | 192 | 576 → 96 | Smaller supervised core with context preserved |
| a100_80gb_v2.yaml | 256 | 320 | 960 → 128 | Larger context/capacity candidate |

All use the same prepared v2 archive. Their run/output paths differ. The small
core does **not** reduce local-network input size or promise memory savings; it
tests supervision/sampling geometry. At equal steps it sees fewer supervised
pixels. Report both steps and supervised-pixel exposure when comparing results.

Recommended ablations, one change per fresh run: `detail_fraction: 0`,
`loss.flow_gradient: 0`, smaller core, then larger model/context. Test 12/24/48
Heun steps with fixed seeds to separate solver error from model quality.
`inference.blend: owner` retains the tile with highest center weight per pixel,
avoiding sample averaging but potentially introducing seams. Compare it against
the default weighted blending before adopting it. Inspect individual members,
member-average spectra and neighborhood rainfall skill; ensemble-mean smoothness
alone is not failure of a probabilistic model.

## Limits and validation

The supplied split remains January–August training, September–mid-October
validation, late October–December test. More epochs cannot supply missing
seasonal regimes. Add real multi-year/all-season data through revised explicit
time ranges when available; no such data are invented here. HR midpoint snapshots
still approximate LR hourly means. Removing projection enables HWT bias correction
but does not eliminate timing/representativeness differences or guarantee physical
water-budget closure. Flow samples can deviate from native totals; reports show it.

V2 supports broad spatial context, not an autoregressive temporal model. Independent
patch generation plus shared noise is not a proof of full-domain stochastic
coherence. Richer atmospheric profiles and temporal conditioning are subsequent
data-dependent experiments, not claimed implemented features.

Run reproducible software checks independently of any production training:

```bash
python -m pytest -q
python -m merraflow.cli_v2 smoke --workdir /tmp/merraflow_smoke_v2
```

Choose a new smoke directory each time. Synthetic fixtures validate software only,
not atmospheric downscaling skill. CUDA/DDP performance and scientific improvement
must be measured on Discover and held-out data.
