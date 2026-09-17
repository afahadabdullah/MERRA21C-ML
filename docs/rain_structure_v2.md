# Retraining for precipitation amount and spatial structure

The supplied February 23 case has precipitation ensemble-mean RMSE about
4.06 mm/h, versus 1.16 for the coarse input, and fragmented rain instead of
the target's organized bands. The previous noise-padding repair does not resolve
that failure. This configuration changes both training and inference.

## What changes

- **Rainfall representation:** learn residuals in `sqrt(1 + P/s) - 1`, with
  inverse `P = s * z * (z + 2)` for nonnegative `z`. This removes exponential
  amplification of positive errors. There is no upper rain cap. Negative decoded
  coordinates map to zero rain. `s` is the archive's existing precipitation scale.
- **Normalization:** square-root rain residuals explicitly use offset zero and
  scale one, not the old log-residual statistics. The second stage calibrates its
  residual RMS on training data, as before. Other variables retain their original
  normalization. The checkpoint config records the representation and prior;
  incompatible checkpoints are rejected.
- **Rainfall noise:** both flow training and sampling use a stationary correlated
  Gaussian rainfall prior with a two-grid-pixel Gaussian kernel standard deviation
  and unit marginal variance. Noise for other fields remains unchanged. This is
  a new train-time prior, not an inference filter or a smoothed final prediction.
  The chosen correlation length is an initial modeling setting to validate, not
  a demonstrated optimal value. The network can generate finer target structure.
- **Supervision:** flow loss covers the complete evolved patch, including halos.
  Its endpoint-error gradient and area-weighted multiscale losses supervise
  structure while retaining quadratic velocity objectives. Regression additionally
  sees physical rain-rate, regional-average and patch-average errors. Rainfall
  receives channel weight two; other fields retain weight one.
- **Sampling:** all overlapping tiles evaluate the same full-domain state at
  both stages of every Heun step. Their velocity estimates are blended before the
  next state is formed. Outside-domain halos are also evolved. This replaces
  separate tile trajectories with end-only blending in the new configuration.
- **Checkpoint selection:** every five epochs (also first and final), generate
  four members on 64 fixed validation patches and select the lowest sum of
  precipitation CRPS, ensemble-mean RMSE, and the square root of a spatial
  variogram score at 1, 4 and 16 pixels. All three terms have mm/h units. A falling
  velocity objective alone no longer replaces the best flow checkpoint.

The spatial term compares the ensemble's expected absolute pairwise differences
with the target's. It evaluates individual-member texture, including graininess
that ensemble averaging can hide. Variogram scores were developed to assess
multivariate dependence: [Scheuerer and Hamill (2015)](https://repository.library.noaa.gov/view/noaa/22327/).
The use of regression followed by a generative residual model is also an
established downscaling approach: [CorrDiff](https://www.nature.com/articles/s43247-025-02042-5).
Those studies do not validate this particular configuration or its skill on HWT.

## Submit training and the existing five-case test

Reuse `data/paired_hourly_annual_v2`. Its physical truth and coarse arrays are
already available; the new representation is computed while loading patches.
No archive rewrite, new normalization job, or additional diagnostic run is needed.
Both model stages train from scratch under `runs/merraflow_rain_structure_v2`.
Do not resume or initialize this run from the old `best_v2-Copy1.pt`.

From the cluster repository:

```bash
git pull --ff-only
mkdir -p logs_v2
env -u SLURM_MEM_PER_CPU -u SLURM_MEM_PER_NODE -u SLURM_MEM_PER_GPU \
    CONFIG=configs/discover_rain_structure_v2.yaml \
    STAGE=regression RESUME= REGRESSION_CHECKPOINT= \
    TRAIN_FLOW_AFTER_REGRESSION=1 TEST_AFTER_TRAINING=1 \
    FLOW_PLOT_INTERVAL=10 \
    sbatch --export=ALL --job-name=rain_structure_v2 scripts/slurm_train_flow_v2.sh
```

This queues regression first. The wrapper resumes interrupted stages from their
`last_v2.pt`, queues a continuation when training returns near the time limit,
starts flow only after all regression epochs finish, and finally queues the
existing test script for five members on these five cases:

`20260223_0530 20260209_1530 20260209_2030 20260305_1230 20260306_1830`

The test uses the newly selected flow checkpoint with its embedded regression
weights. Outputs go to `runs/merraflow_rain_structure_v2/trained_test_v2`.
No test data enter training or checkpoint selection. Do not submit the same run
twice concurrently. A failed command exits without queuing its successor.

## Continue an existing one-GPU flow run on four GPUs

The original checkpoint contains one rank's random state, so changing GPU count
requires an explicit migration. Do this only after a completed flow epoch has
written `flow_v2/last_v2.pt`. First identify the **flow training** job with
`squeue -u "$USER"`; do not cancel a `flow_latest_test_v2` evaluation job. Cancel
the flow training job, confirm it has left the queue, then submit from the project
root:

```bash
git pull --ff-only
scancel FLOW_TRAINING_JOB_ID
squeue -u "$USER"
env -u SLURM_MEM_PER_CPU -u SLURM_MEM_PER_NODE -u SLURM_MEM_PER_GPU \
    CONFIG=configs/discover_rain_structure_v2.yaml STAGE=flow \
    RESUME=runs/merraflow_rain_structure_v2/flow_v2/last_v2.pt \
    REGRESSION_CHECKPOINT= ALLOW_WORLD_SIZE_CHANGE=1 \
    RESET_FLOW_BEST_ON_MIGRATION=1 \
    TRAIN_BATCH_SIZE_OVERRIDE=2 TRAIN_WORKERS_OVERRIDE=3 \
    sbatch --export=ALL --job-name=flow_v2 --gres=gpu:4 \
    --cpus-per-gpu=4 scripts/slurm_train_flow_v2.sh
```

The four-GPU run keeps the effective batch at 16, the 512 optimizer updates per
epoch, the validation patch count and the learning-rate schedule. Model, EMA and
optimizer weights resume from the last completed epoch; the data order and the
extra GPUs' random streams change. This continuation is not bitwise identical
to one-GPU training. A new generated-validation score after the first resumed
epoch establishes the four-GPU best checkpoint. Full-domain progress plots are
kept out of DDP training so one rank cannot hold up the other ranks; use the
separate evaluation job for maps. The batch job tests GPU communication before
opening the archive. The old best is preserved as
`flow_v2/best_before_reset_v2.pt`. The wrapper carries these settings into later
12-hour continuation jobs. Numbered `epoch_XXXX_v2.pt` checkpoints are written
every two completed epochs; `last_v2.pt` is written every epoch. Four GPUs may
shorten an epoch, but the gain depends on data loading and DDP communication;
compare epoch wall times before assuming a fourfold speedup.

DDP is constructed with `find_unused_parameters=True`. The self-attention block
allocates a `context_norm` it never uses, so two parameters receive no gradient
in every iteration. One GPU never wrapped the model and never saw this; without
the flag the first four-GPU step stops with "Expected to have finished reduction
in the prior iteration" and names parameter indices 106 and 107. A resubmission
of the same migration command is safe after that failure: nothing was written,
so `flow_v2/last_v2.pt` still holds the last completed epoch and
`best_before_reset_v2.pt` is left as it was.

## What is verified and what still requires production training

Local tests exercise both stages, exact checkpoint resume, generated validation
selection, and synchronized full-domain prediction on small synthetic data.
They check transform round trips, matching prior covariance, actual halo gradients,
shared-state integration, sensitivity to shuffled spatial patterns, and batch-job
continuation/transition commands.

The production model has not been trained here. This is a tested implementation
of corrections aimed at the observed failures, not a claim of recovered HWT skill.
Changing the prior can affect fine-scale variability; square-root coordinates can
still generate excessive values, though growth is polynomial. Validate amount,
extremes and spatial structure together. The validation-patch score is a practical
checkpoint selector; the final domain test still matters because overlapping
full-domain integration differs from isolated patch evaluation.

The target is a high-resolution midpoint snapshot and coarse rain represents an
hourly mean. That existing temporal mismatch limits exact storm-band matching.
No output filtering or forced agreement with the coarse rain budget is applied.
Global integration increases inference I/O: CPU tile caching is bounded by
`inference.tile_cache_mb` (512 MB by default); global state arrays use additional
host memory. Old configurations retain their old representation and sampler.
