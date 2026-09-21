# Next precipitation training experiment

The current rain-structure run produces grainy or diffuse precipitation across
checkpoints, and increasing inference from 24 to 48 steps does not correct it.
The next experiment therefore changes training. Success means lower physical
precipitation error than the coarse input **and** realistic member structure;
training-loss reduction or visually smoother ensemble means are insufficient.
No production improvement is claimed before this run is evaluated.

## Plan and implemented changes

1. **Fine-tune from a frozen copy of the existing flow EMA.** Reuse its embedded
   regression model, residual scale, square-root rainfall representation and
   correlated-noise prior. Reset the optimizer, learning-rate schedule, epoch
   counter and checkpoint selection. The old run and checkpoint are unchanged.
2. **Train through generated rainfall.** Retain the existing flow-matching loss.
   Every eighth local minibatch, sample four members on one training patch with
   a differentiable 24-step Heun trajectory, matching the configured validation
   step count. Decode to mm/h and backpropagate rainfall CRPS, spatial variogram,
   neighborhood wet-coverage and multiscale ensemble-mean error. This evaluates
   free samples; the target never enters the sampling trajectory.
3. **Expose more rain boundaries.** The recommended `rain_edges` configuration
   mixes 30% rainfall-edge proposals, 40% existing rain/coast detail proposals and
   30% uniform proposals. Edge scores use log-rain gradients and patches with
   both wet and dry pixels. Exact inverse proposal weights preserve the uniform
   candidate-grid objective. Validation proposals remain uniform. Time selection
   is still uniform over training hours; this is not storm-hour stratification.
4. **Select and assess generated skill.** Fixed validation patches now report
   coarse, regression and generated-rain errors. Keep a separate checkpoint that
   beats the coarse baseline on RMSE, CRPS versus coarse MAE, and variogram error.
   The final five-case domain test is still required: patch skill does not prove
   full-domain skill.
5. **Only then consider more spatial context.** A separate configuration uses a
   256-pixel core and 16 × 16 broad-context tokens instead of 128 and 8 × 8.
   It retrains both stages and cannot initialize from old checkpoints. It is a
   follow-up architecture experiment, not part of the first recommended run.

The existing dx/dy endpoint-error loss is retained. The new variogram loss
compares the ensemble's expected absolute rain increments to HWT at 1, 4, 16
and 32 pixels, horizontally, vertically and in both diagonal directions.
Coverage compares differentiable threshold fractions at 0.1, 1 and 5 mm/h over
1, 4, 16 and 32-pixel neighborhoods. Amount errors use the same neighborhood
sizes. No output blur, intensity cap or independent random wet-mask is applied.

The rainfall rate terms use 10 mm/h as their numerical unit; it is not a cap.
CRPS and squared expectation scores include finite-ensemble corrections to
avoid an incidental incentive to collapse spread. Individual corrected scores
can be negative; this is not an implementation error. The ensemble-mean error
term deliberately measures the actual finite ensemble. Coverage uses a smooth
threshold transition of 0.1 mm/h, and the same transition for target and samples.
These are initial modeling settings to validate, not established optimal values.

The auxiliary weight ramps from 0.2 to 1 over five epochs. Cadence is not
inverse-weighted: weight one applies on every eighth minibatch, so its average
contribution is smaller than applying it every step. The default run uses two
GPUs, batch size two per GPU, accumulation four, 8,192 patches per epoch and
30 fine-tuning epochs at initial learning rate 5e-5. Effective global batch is 16.
Sampling through 24 steps adds substantial compute; activation checkpointing
limits memory. The job benchmarks the complete auxiliary step before training.
Production GPU memory and wall time must be checked in that benchmark.

## Submit the recommended run

First sync the changed repository files to Discover. From the cluster project
directory, using bash:

```bash
cd /gpfsm/dnb10/projects/p311/ML_downscaling
bash scripts/submit_rain_rollout_v2.sh
```

The wrapper defaults to `configs/discover_rain_edges_v2.yaml` and initializes
from `runs/merraflow_rain_structure_v2/flow_v2/best_v2.pt`. To choose another
existing **rain-structure** flow checkpoint:

```bash
SOURCE_CHECKPOINT=runs/merraflow_rain_structure_v2/flow_v2/last_v2.pt \
  bash scripts/submit_rain_rollout_v2.sh
```

It freezes the source under `runs/merraflow_rain_edges_v2/initialization_v2`,
queues a CPU rain-edge proposal-cache job, and makes the two-GPU training job
depend on successful cache completion. The cache reads only training truth
and writes a separate proposal-cache file; it does not rewrite archive fields.
An existing matching cache is reused. No new regridding or normalization job is
needed. The helper refuses to submit into an existing run directory.

Use two GPUs by default: four GPUs are not assumed to provide worthwhile
scaling. For a **new** run where measured throughput justifies four GPUs, prefix
the same submission command with `GPUS=4`. The helper adjusts accumulation from
four to two and preserves 256 ordinary validation patches and 64 generated
validation patches. It freezes the effective configuration in
`initialization_v2/config_v2.yaml`; training and all continuations use that file.
The effective global batch remains 16. Random streams change with GPU count,
so the results are not bitwise identical. This option starts a new run; it does
not migrate a running two-GPU job.

The training wrapper automatically continues at completed-epoch boundaries
across 12-hour allocations. It then queues the existing five-member test for:
`20260223_0530`, `20260209_1530`, `20260209_2030`, `20260305_1230`,
`20260306_1830`. Final outputs are in
`runs/merraflow_rain_edges_v2/trained_test_v2`.

If a job fails after saving an epoch, fix the reported cause, confirm there is
no running or queued continuation for this run, and resume with:

```bash
env -u SLURM_MEM_PER_CPU -u SLURM_MEM_PER_NODE -u SLURM_MEM_PER_GPU \
  -u TRAIN_BATCH_SIZE_OVERRIDE -u TRAIN_WORKERS_OVERRIDE \
  CONFIG=runs/merraflow_rain_edges_v2/initialization_v2/config_v2.yaml STAGE=flow \
  RESUME=runs/merraflow_rain_edges_v2/flow_v2/last_v2.pt \
  INITIALIZE_FLOW= REGRESSION_CHECKPOINT= TEST_AFTER_TRAINING=1 \
  PREFER_SKILL_CHECKPOINT=1 FLOW_PLOT_INTERVAL=1000 \
  sbatch --export=ALL --gres=gpu:2 --cpus-per-gpu=4 scripts/slurm_train_flow_v2.sh
```

If submission/preflight failed before the first checkpoint, reuse the already
frozen source with that direct command, setting `RESUME=` and
`INITIALIZE_FLOW=runs/merraflow_rain_edges_v2/initialization_v2/source_flow_v2.pt`,
plus `TRAIN_PREFLIGHT=1`. For an incomplete cache, first resubmit
`scripts/slurm_cache_rain_edges_v2.sh` with the same CONFIG and add an
`--dependency=afterok:JOB_ID` to training. Do not delete a populated run to make
the submission helper run again.
For a run originally submitted with `GPUS=4`, use `--gres=gpu:4` when resuming
its frozen configuration. A mismatched GPU count is rejected.

## Evaluate the best checkpoint while training continues

After syncing `scripts/submit_best_test_v2.sh` to Discover, submit from the
cluster project directory:

```bash
bash scripts/submit_best_test_v2.sh
```

This freezes `runs/merraflow_rain_edges_v2/flow_v2/best_v2.pt` and its embedded
configuration in a unique `evaluations_v2/best_v2_*` directory. It prints the
selected epoch, score, SHA256, output directory and submitted job ID. Set
`RUN_ROOT` for another run. The evaluation requests one A100, five members,
24 integration steps and the same five test timestamps as the final test.
Training can continue independently; subsequent best-checkpoint updates do
not change this evaluation. The best composite score does not imply that the
checkpoint beats the coarse baseline.

The `diagnostic_v2` subdirectory contains full-domain and storm-zoom comparison
maps, `rmse_summary_v2.png`, `metrics_v2.json`, and ensemble NetCDF predictions.
Training-history plots are also written when history is available; these read
the run's history at evaluation time, whereas checkpoint weights are frozen.
Metrics include RMSE, MAE, bias, correlation, CRPS, spread and coverage, plus
rainfall CSI, POD, FAR, Brier score, reliability bins and neighborhood FSS.

## Read the results

- `flow_v2/history_v2.jsonl`: `rain_rollout_train` records auxiliary components
  and the number of sampled training patches. Existing `train.total` and
  `val.total` remain the ordinary flow objective for comparability.
  `timing` records training-plus-validation seconds and GPU-hours before saving
  the checkpoint. Compare epochs with the same validation cadence when deciding
  whether four GPUs offer useful speedup per GPU-hour.
- `generated_validation.rmse_skill_vs_coarse`: greater than zero means lower
  RMSE on those same validation patches. `beats_coarse` additionally requires
  lower CRPS than the deterministic coarse MAE and lower spatial variogram error.
- `flow_v2/best_v2.pt`: best composite generated-validation score, retained even
  if no checkpoint beats coarse. `last_v2.pt` is always the resume checkpoint.
- `flow_v2/best_skill_v2.pt`: best composite score among checkpoints satisfying
  all three coarse-baseline criteria. This file is absent if none qualify.
  The test job prefers it; otherwise it explicitly reports failure to qualify
  and tests `best_v2.pt` for diagnosis. Eligibility is not a significance test.
- `trained_test_v2/metrics_v2.json`: compare precipitation RMSE against the
  low-resolution baseline case by case, alongside individual-member rain maps.
  Use the existing precipitation audit for wet fractions, extremes, spectra,
  neighborhood skill and calibration. A five-case event sample does not establish
  skill over the full test distribution. Keep test cases out of loss tuning.
- `trained_test_v2/rain_training_v2.png`: generated validation RMSE, CRPS and
  spatial error plotted against coarse baselines across epochs. Unlike the
  ordinary flow-objective plot, this directly shows sampled-rainfall skill.

The source pairs still compare coarse hourly means to HWT midpoint snapshots.
That limits exact storm-band matching. The implementation tests verify software
behavior; they cannot establish meteorological improvement.

## Optional controlled comparisons

Use the **same frozen source** for these runs. They are not submitted by default:

```bash
SOURCE_CHECKPOINT=runs/merraflow_rain_edges_v2/initialization_v2/source_flow_v2.pt \
  CONFIG=configs/discover_rain_rollout_v2.yaml bash scripts/submit_rain_rollout_v2.sh
SOURCE_CHECKPOINT=runs/merraflow_rain_edges_v2/initialization_v2/source_flow_v2.pt \
  CONFIG=configs/discover_rain_control_v2.yaml bash scripts/submit_rain_rollout_v2.sh
```

`rain_rollout` uses the generated-rain objective with the original patch sampler;
`rain_control` uses the original objective and sampler with the same fine-tuning
schedule. Comparing edges versus rollout isolates patch selection. Comparing
rollout versus control tests the added objective. Auxiliary samples consume
additional random draws and compute, so these are not identical-noise or
equal-compute experiments.

If the new objective still fails to recover connected structures across
validation events, the separately configured larger-context run is:

```bash
env -u SLURM_MEM_PER_CPU -u SLURM_MEM_PER_NODE -u SLURM_MEM_PER_GPU \
  -u TRAIN_BATCH_SIZE_OVERRIDE -u TRAIN_WORKERS_OVERRIDE \
  CONFIG=configs/discover_rain_context_v2.yaml STAGE=regression \
  RESUME= INITIALIZE_FLOW= REGRESSION_CHECKPOINT= \
  TRAIN_FLOW_AFTER_REGRESSION=1 TEST_AFTER_TRAINING=1 \
  PREFER_SKILL_CHECKPOINT=1 FLOW_PLOT_INTERVAL=1000 \
  sbatch --export=ALL --gres=gpu:2 --cpus-per-gpu=4 scripts/slurm_train_flow_v2.sh
```

This run trains both stages from scratch. It changes initialization, patch area
and context as well as compute, so comparisons to fine-tuning are exploratory;
do not attribute differences exclusively to the context-token count.
