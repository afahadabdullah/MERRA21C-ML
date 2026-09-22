# v3_precip: can we generate realistic fine-scale precipitation?

## Revision: fixes 1-7 (September 2026)

The first v3 draft was reviewed against the v1/v2 diagnostics (regression precip
RMSE ~= coarse RMSE; flow members with salt-and-pepper drizzle over dry areas and
visible tile seams). These changes are now implemented and the defaults in
`configs/discover_v3_precip.yaml`:

| # | Problem | Change |
| --- | --- | --- |
| 1 | ~12.8k optimizer updates, constant LR: undertrained denoiser (speckle) | `samples_per_epoch: 32768`, 40 regression / 250 diffusion epochs (~20k / ~128k updates at 64 patches/update); linear warmup (2000 updates) + cosine decay to 5%; LR schedule saved in checkpoints. Long runs are split into chained 12 h Slurm segments (`time_limit_hours`, auto-resume). Diffusion refuses an incomplete regression run. |
| 2 | 3.3M-parameter U-Net | `base_channels: 96`, `channel_mult: [1,2,3,4]`, `time_dim: 256`, 8 heads: 22M parameters. `peak_gpu_gb` is logged per epoch; drop to 64 channels (10.4M) if memory is tight. |
| 3 | Small positive noise over dry pixels decoded as drizzle | Dry-margin encoding: `P < 0.02 mm/h` maps to -0.25; wet rates use the square-root transform. This censors light rain and is an ablation, not demonstrated calibration. Evaluate occurrence, light-rain distribution, totals and CRPS together. |
| 4 | Uniform-patch and rain-emphasized objectives need an explicit choice | `patch.proposal: coarse` deliberately emphasizes rainy inputs using unit weights. This is not an unbiased estimate of uniform-patch risk. `proposal: truth` retains inverse-proposal weights. Validation remains uniform. |
| 5 | Independent-tile trajectories can leave seams | Synchronized EDM blends full-tile denoised outputs at every global Heun step; requires `loss_on_halo: true`. The Gaussian pointwise test verifies solver/blending algebra, not untiled equivalence or seam-free behavior of the neural network. |
| 6 | Target was one :30 HWT snapshot vs a GEOS hourly mean | `target_kind: hourly_mean_trapezoid`: `1/4 P(:00) + 1/2 P(:30) + 1/4 P(:00+1h)` from `hwt_30mn_slv_LCC`, prepared into a separate `data.hourly_targets` directory (v2 archive untouched). Each :30 snapshot is checked against the archived truth. Hours missing a snapshot are excluded, never filled. `midpoint_rate` remains available. |
| 7 | 64 patches x 4 members x 12 steps: noisy checkpoint ranking | 256 patches x 8 members x 18 steps, every 5 diffusion epochs (and the last); regression validated every epoch. |

These changes are hypotheses to evaluate on held-out data. Individual members
often have larger pointwise errors than an accurate conditional-mean baseline;
that is not guaranteed for an imperfect learned regression. Report CRPS,
ensemble-mean RMSE, member structure and both baselines without assuming success.

## Recommendation and evidence

Use **conditional residual diffusion with a precipitation-only output**, and
retain the atmospheric predictors. Treat this as a strong candidate to test,
not a universal best method. Predict an ensemble of plausible fine-scale rain
fields conditional on the resolved atmosphere; exact convective-cell positions
may be irreducibly uncertain. Assess spatial realism and probabilistic skill
separately from deterministic pixel error.

Literature reviewed September 2026:

| Method / primary source | What the evidence supports | Relevance here |
| --- | --- | --- |
| [Harris et al. (2022), stochastic precipitation downscaling](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2022MS003120) | Generative methods map hourly IFS atmospheric fields to hourly accumulated NIMROD rainfall and evaluate ensembles. | Direct hourly evidence; coarse atmospheric context and probabilistic verification matter. |
| [CorrDiff (2025)](https://www.nature.com/articles/s43247-025-02042-5) | Regression plus residual diffusion downscales 25 km atmospheric states to 2 km over Taiwan, including radar-channel synthesis. | A useful decomposition and spatial architecture; not proof of hourly rainfall-amount skill on GEOS/HWT. Temporal coherence remains a limitation. |
| [spateGAN-ERA5 (2025)](https://www.nature.com/articles/s41612-025-01103-y) | A conditional GAN downscales 24 km/hourly ERA5 to 2 km/10-minute precipitation, using multi-hour context and temporal convolutions. | Particularly relevant hourly/subhourly evidence and an important competitor. Do not infer that diffusion has universally beaten GANs. |
| [WassDiff (2025)](https://arxiv.org/html/2410.00381v3) | Diffusion with a Wasserstein distribution regularizer improves extreme precipitation in its experiments. Its MRMS reference is aggregated to **daily** precipitation. | Motivates a later tail-calibration ablation, not an established hourly winner or a reason to force each random member to match an observed storm exactly. |
| [Singh et al. (2026), generative intercomparison](https://gmd.copernicus.org/articles/19/7545/2026/) | Daily perfect-model experiments compare U-Net, WGAN and DDPM. Relative strengths differ for extremes, spatial structure, calibration and cost. | Coarsened truth is an easier problem than downscaling independently biased model data. Daily results cannot rank hourly GEOS/HWT methods. |
| [Karras et al. (2022), EDM](https://arxiv.org/abs/2206.00364) | Preconditioning, a continuous noise distribution and a second-order sampler provide a practical diffusion formulation. | Supplies the diffusion mathematics used here; not meteorological validation. |

My implementation choice is an inference from that evidence and the existing
repository: start with one-channel EDM, so changes can be interpreted against the
existing regression-plus-flow pipeline. Flow matching remains a credible baseline;
changing its objective does not itself guarantee better precipitation. A next
experiment could compare identical one-target backbones trained with flow and EDM.

## The target distinction that controls the scientific claim

The existing prepared v2 archive contains **HWT PRECTOT midpoint snapshots**, in
mm/h after conversion from kg m-2 s-1. GEOS inputs are hourly means. Multiplying an
instantaneous rate by one hour does not turn it into a measured hourly total.
The adapter supports `midpoint_rate` (`time: point`) and the default
`hourly_mean_trapezoid` (an explicitly approximate `time: mean`). It rejects
`hourly_accumulation` rather than silently substituting legacy APCP fields.

Three snapshots at :00, :30 and next :00 approximate the hourly mean with weights
1/4, 1/2 and 1/4. Brief convective bursts between snapshots can be missed. This
is not a measured accumulation. To claim accurate hourly amounts, use aligned
hourly accumulation labels: inspect source time bounds and accumulation/reset
semantics, convert units, difference cumulative totals only when documented,
and pair identical hourly windows. True accumulation labels still require an
additional audited archive adapter.

Hourly targets carry a persistent archive/method/weights contract. Per-hour
metadata records source paths, sizes and nanosecond modification times, plus a
SHA-256 checksum of the target array. Preparation checks these before skipping
an hour; finalization verifies target checksums and fingerprints the target set.
Training, checkpoints and prediction manifests retain that fingerprint. Source
identity is metadata-based, not a full hash of every large HWT NetCDF; preserve
source immutability. Changed source metadata, changed target metadata or an old
directory without provenance is rejected. Use a **fresh `data.hourly_targets`
directory** for legacy targets or intentional source/method changes. Finalized
targets and sources must remain unchanged at their recorded paths during a run.

HWT is also a high-resolution simulation, not radar/gauge truth. Agreement with
HWT establishes emulation of that simulation, not observational rainfall accuracy.

## Implemented experiment

- Target, diffusion state, output head and losses each have **one channel**:
  precipitation. Temperature, pressure and winds are never auxiliary targets.
- Current coarse meteorology, coarse baseline fields, terrain, surface fractions,
  coordinates and calendar features condition the model. The configured dynamic
  predictors at `t-2` and `t-1` add causal storm context. No preceding HR truth is
  an input. Only complete histories within the same train/val/test split qualify;
  missing hours are not filled or borrowed across split boundaries.
- Reuse v2's prepared arrays and training-only predictor normalization read-only.
  No second multi-terabyte archive is needed. The adapter slices the rain channel
  before reading target patches. Proposal scoring also reads precipitation only.
- Transform `z = sqrt(1 + P/s) - 1`, with fixed `s = 1 mm/h`. This is a deliberate
  testable choice that compresses heavy tails less than log1p, not a proven optimum.
  Decode `P = s * max(z, 0) * (max(z, 0) + 2)`. There is no arbitrary maximum-rate
  cap. The dry margin and optional decode threshold can affect dry-frequency
  calibration, which is explicitly evaluated. No separate wet/dry classifier is
  implemented.
- Regression predicts `z(HWT) - z(coarse)`. Its MSE optimum is in transformed
  space; the decoded regression is not the physical conditional mean. The
  physical ensemble mean is computed after decoding each member.
- Freeze regression EMA. Calibrate remaining residual RMS on training patches
  only, then train a one-channel preconditioned EDM denoiser of that normalized
  residual. Lognormal noise sampling and EDM-weighted denoising MSE are the
  stochastic objective. No GAN, Wasserstein penalty or flow rollout is hidden in
  this implementation.
- Default coarse proposals deliberately emphasize rainy inputs with unit weights.
  Truth proposals instead use exact inverse weights, and midpoint truth proposals
  can reuse the v2 cache. Coarse scores currently compute on demand, so benchmark
  loader throughput. Validation uses uniform proposals and fixed sampling seeds.
- Select checkpoints by **generated physical-unit validation CRPS**, against
  regression and coarse baselines. This is more relevant than comparing denoising
  losses alone. Small validation ensembles give a noisy ranking; repeat the final
  shortlist with more members and full held-out hours.
- Four-GPU DDP fp32/bf16 training, gradient accumulation, EMA, gradient clipping,
  epoch recovery and exact same-config/same-world-size resume are implemented.
  Single-device execution is retained for smoke tests. Warmup and cosine decay
  are saved with the optimizer. Checkpoints carry the
  dataset fingerprint, full config, frozen regression, residual scale, optimizer,
  per-rank RNG and history. Artifacts live under separate `v3_precip` paths.
- Validation scores are reduced across all ranks without padded duplicate
  validation samples. Training patch counts must divide world size. Residual RMS
  calibration also combines all ranks. Only rank zero writes plots/checkpoints.
- Every **5 completed epochs** of both stages, fixed rank-zero validation patches
  produce rainfall maps, spatial spectra, exceedance curves, rank histograms and
  global training/validation history plots. The same patch/noise seeds allow
  comparisons across epochs; these are patch diagnostics, not full-domain maps.
- Inference defaults to synchronized Heun sampling. Tile preparation disables
  autograd before caching regression outputs. Inspect neural tile-boundary
  artifacts; global solver state does not guarantee agreement of local contexts.
  Owner and weighted independent-tile sampling remain available as ablations.
- Previous-hour inputs improve conditioning but do not make this a video model.
  Independently generated hourly members have no guaranteed temporal coherence.
  Do not use their multi-hour sums as validated hydrologic trajectories.

The only shared-model change is a `target_channels=5` default parameter in
`UNetV2`; v2's default shapes and checkpoint keys remain unchanged. V3 supplies 1.

## Run on Discover

Use the project environment with PyTorch >=2.3 and the dependencies in
`pyproject.toml`. The following commands run from the project root. They do not
download source data. Preparation, if needed, remains the documented annual v2
workflow in `docs/runbook_discover_annual_v2.md`.

```bash
export PYTHONPATH="$PWD/src"
# CPU preparation first (or use the dependency-managed Slurm command below):
python -m merraflow.cli_v3_precip prepare-hourly --config configs/discover_v3_precip.yaml
python -m merraflow.cli_v3_precip finalize-hourly --config configs/discover_v3_precip.yaml
python -m merraflow.cli_v3_precip audit --config configs/discover_v3_precip.yaml

python -m merraflow.cli_v3_precip train \
  --config configs/discover_v3_precip.yaml --stage regression

python -m merraflow.cli_v3_precip train \
  --config configs/discover_v3_precip.yaml --stage diffusion \
  --regression-checkpoint runs/merraflow_v3_precip/regression_v3_precip/best_v3_precip.pt

# First validate a few frames; use a fresh prediction directory for each setup.
python -m merraflow.cli_v3_precip predict \
  --config configs/discover_v3_precip.yaml --split val --limit 4 \
  --checkpoint runs/merraflow_v3_precip/diffusion_v3_precip/best_v3_precip.pt
python -m merraflow.cli_v3_precip evaluate \
  --config configs/discover_v3_precip.yaml --split val
```

For a full evaluation, change `inference.output` to a fresh `...v3_precip`
directory, omit `--limit`, then evaluate. Use `--split test` once choices are
frozen. Predictions refuse overwrites and mixed checkpoints/settings. Evaluation
rejects partial ensembles; intentionally ungenerated hours and excluded temporal
histories are listed in the report. Predicting with a regression checkpoint is
supported, but repeated deterministic members do not form an uncertainty model.

Slurm, **four A100 GPUs per stage** (capacity and walltime must be measured on
the real archive). Activate the project environment, then submit both stages:

```bash
cd /gpfsm/dnb10/projects/p311/ML_downscaling
source /discover/nobackup/projects/GEOS_MITgcm/afahad/conda/etc/profile.d/conda.sh
conda activate /gpfsm/dnb10/projects/p311/ML_downscaling/env
PREPARE_FIRST=1 bash scripts/submit_v3_precip.sh
```

Hourly preparation now runs at most four months at once by default, which is
more schedulable than the earlier 14-way array. Override it only when the
compute queue has capacity, for example `PREPARE_ARRAY_CONCURRENCY=2`.

If the hourly-target finalizer is already queued, submit the GPU chain immediately and make its first regression segment wait for that finalizer.  For the currently queued run:

```bash
AFTEROK_JOB=58501027 bash scripts/submit_v3_precip.sh
```

This does not submit a second hourly-target array. Every training segment is chained with `afterok`; if the finalizer fails or is cancelled, Slurm cancels the training chain.

With `PREPARE_FIRST=1`, the helper first submits monthly CPU preparation and its
dependent finalizer, then chains training after that finalizer. Each GPU job
audits the completed target set before starting. If hourly targets already exist
with valid provenance, omit `PREPARE_FIRST=1` to submit training directly.
The helper creates the log directory and submits a
chain of segments: `REGRESSION_SEGMENTS` (default 2) then `DIFFUSION_SEGMENTS`
(default 8), each `afterok` on the previous one. Each segment estimates whether
another epoch fits within `train.time_limit_hours` and auto-resumes next job from `last_v3_precip.pt`;
segments after a stage finishes exit immediately. If 8 diffusion segments are not
enough, resubmit more with `AUTO_RESUME=1 STAGE=diffusion sbatch scripts/slurm_train_v3_precip.sh`.
Failed dependencies cancel the downstream job. The helper refuses occupied stage directories; use resume for
an existing run. Each job requests `--gres=gpu:4`, checks four GPUs are visible,
and uses `torchrun --nproc-per-node=4` within one Slurm task.

```bash
# Recover an interrupted diffusion job from the last completed epoch:
STAGE=diffusion \
RESUME=runs/merraflow_v3_precip/diffusion_v3_precip/last_v3_precip.pt \
sbatch scripts/slurm_train_v3_precip.sh
```

The default batch size is **8 per GPU**, with accumulation of 2: four GPUs produce
an effective batch of **64 patches per update**. `samples_per_epoch: 32768` is
the global patch count (8192 per rank), not 32768 per GPU. Loader workers are 4
per rank; the allocation reserves 4 CPUs and 32 GB host memory per GPU. The
four-CPU allocation matches Discover's working four-GPU test template; it avoids
the unavailable 64-CPU request from the first submission. Learning
rate is not automatically multiplied with GPU count. Exact resume requires the
same world size and training settings. A companion best checkpoint is required
only after a validation has selected one; early diffusion segments resume from
last alone. Last checkpoints are saved
each epoch, so an interrupted partial epoch is rerun. Start with a short pilot
using a copied config and fresh output paths; cosine LR, 40/250 epochs, 128-pixel
cores, 16 members and 32 sampling steps are unbenchmarked starting settings.

Training plots are written under each stage directory:

```text
runs/merraflow_v3_precip/regression_v3_precip/validation_plots/epoch_0005/
runs/merraflow_v3_precip/diffusion_v3_precip/validation_plots/epoch_0005/
```

Each epoch directory contains `fields.png`, `diagnostics.png`, `history.png`,
`samples.npz` and `metadata.json`. `train.validation_plot_interval: 5` controls
the cadence and `validation_plot_samples: 2` limits preview patches. Numerical
generated validation and plots run every fifth diffusion epoch (also the final
epoch); regression validation runs every epoch.
Regression plots contain deterministic predictions, so their spread is zero.
Maps share a fixed truth/coarse color scale with saturated values marked; the
saved arrays and exceedance diagnostics retain uncapped generated rates.

## What would establish realistic downscaling?

The evaluator writes `summary.json`, `per_hour.json` and a common-scale comparison
PNG under `evaluation_val_v3_precip` or `evaluation_test_v3_precip`.

1. Beat **both** interpolated coarse precipitation and regression on held-out
   physical CRPS. RMSE/MAE of the ensemble mean answer a different question and
   should be reported alongside CRPS. The report includes paired daily CRPS
   differences and a descriptive day-bootstrap interval when >=10 days exist.
   Storms spanning multiple days can invalidate independent-day intervals; use
   storm-block inference before publishing significance claims.
2. Inspect Brier scores, probability reliability bins, tie-randomized ranks and
   finite-ensemble coverage. Adequate spread is necessary; sharp samples alone
   do not imply calibrated uncertainty. The nominal 90% interval is estimated
   from few members and does not have exact 90% finite-ensemble coverage.
3. Check wet-area frequency, dry leakage, pooled 95/99/99.9th percentiles, per-hour
   maxima and exceedances at 0.1/1/5/10/25 mm/h. Heavy-rain events are rare: report
   their counts and eventwise uncertainty, not only an overall average.
4. Compare individual-member spectra and member FSS at 1/5/17/33-pixel windows.
   Score ensemble-mean FSS separately; averaging naturally smooths a distribution
   of displaced storm cells. Spectra use LCC index-space cycles per pixel.
5. Inspect native-cell rainfall budgets and dry-cell leakage without imposing
   biased GEOS totals on HWT. Also inspect tile seams and multiple extreme-event
   maps. Matching a spectrum or marginal distribution does not establish correct
   cell placement or water-budget closure.
6. Evaluate complete storms and seasons, with time-separated splits. The reused
   annual split has mainly cool-season validation/test; it cannot establish
   all-season convective skill. Confirm pair timing and coarse/HR large-scale
   agreement before attributing poor skill to architecture.

Useful controlled ablations: history `[0]` versus `[-2,-1,0]`; one-target versus
existing multivariate v2; owner versus weighted overlap; 16/32/64 sampling steps;
additional training seeds. Keep validation cases and member seeds fixed. Reserve
test events until all choices are settled. A future joint space-time model and
genuine hourly labels are needed for temporal rainfall-accumulation claims.

## Software verification versus scientific results

```bash
PYTHONPATH=src python -m pytest -q tests/test_v3_precip.py tests/test_pipeline_v2.py
PYTHONPATH=src python -m merraflow.cli_v3_precip smoke --workdir /tmp/merraflow_smoke_v3_precip
```

Use a new smoke directory. The synthetic run prepares input, trains both stages,
samples an ensemble, and produces the physical reports. Tests cover one-channel
supervision, history isolation, transform inversion, proposal weighting, an
analytical Gaussian diffusion sampler, finite gradients, bitwise CPU resume,
prediction provenance and incomplete-ensemble rejection. These are software
checks, not evidence that realistic GEOS-to-HWT precipitation has been achieved.
No production training or cluster submission is implied by these commands being
documented.

Current verification: **44 tests passed** across `test_v3_precip.py` (20),
`test_pipeline_v2.py` and `test_checkpoint_compat_v2.py`. This includes four CPU
DDP processes, resume before the first diffusion validation, gradient-free tile
caches, source/method mismatch rejection, target checksum corruption, target-set
checkpoint identity, epoch-5 plots and mocked preparation-to-training Slurm chains.
The local DDP test uses explicit loopback networking because automatic hostname
resolution failed on this Mac; it passed when rerun that way. Shell syntax and
diff whitespace checks pass. A separate CLI smoke run completed both stages,
synchronized prediction, physical evaluation and plots. No cluster jobs were
submitted. Four-A100 execution, memory/throughput and real-data skill remain
unmeasured; synthetic training is not a scientific realism result.
