# Inference noise-padding correction

Training uses independent Gaussian initial noise throughout every patch, including
its halo. Previously, inference cropped a domain-sized noise field with edge
replication: the outside-domain halo repeated its nearest boundary value. This
creates spatially correlated noise that differs from the training prior.

V2 inference now defaults to `inference.noise_padding: independent_halo`. It draws
the original domain field first, then fills an extended halo with independent
Gaussian draws. Overlapping patches share the same extended coordinates. Every
in-domain noise value is identical to the legacy sampler for the same seed.

This is an inference-only change. Existing checkpoints, including
`best_v2-Copy1.pt`, require no retraining or conversion. Weights, normalization,
flow scale, input-field padding, and the ODE integrator are unchanged. Training
progress plots also use the corrected inference path; the training objective is
unchanged. Patch trajectories still evolve independently before final blending.

The correction addresses a verified boundary-noise mismatch. It has not yet been
shown to improve the production checkpoint's precipitation. It cannot directly
change interior cores whose patches never touch the outside-domain halo, so it
should not be expected to resolve all interior speckling.

## Controlled five-case compute job

The existing test script can compare legacy `replicate` and corrected
`independent_halo` padding with the same checkpoint, five members, seeds, and
24 Heun steps. The checkpoint is used read-only. Run this from the cluster repo:

```bash
git pull --ff-only
mkdir -p logs_v2
env -u SLURM_MEM_PER_CPU -u SLURM_MEM_PER_NODE -u SLURM_MEM_PER_GPU \
    CONFIG=configs/discover_annual_v2.yaml \
    CHECKPOINT=runs/merraflow_annual_v2/flow_v2/best_v2-Copy1.pt \
    OUTPUT=runs/merraflow_annual_v2/copy1_padding_compare_v2 \
    SPLIT=test MEMBERS=5 STEPS=24 COMPARE_NOISE_PADDING=1 \
    NOISE_PADDING= INCLUDE_DATE= \
    TIMESTAMPS="20260223_0530 20260209_1530 20260209_2030 20260305_1230 20260306_1830" \
    sbatch --export=ALL scripts/slurm_test_best_model_v2.sh
```

Use a fresh output directory. This performs ten case runs in total: five cases
for each padding mode. It does not retrain. After completion, inspect
`padding_comparison_v2.md` and `padding_comparison_v2.json` in the output root.
The table compares precipitation RMSE, bias, largest member value, and RMSE
separately for boundary-affected cores and the remaining interior. The JSON also
records regional area fractions, mean precipitation and target extremes.
Each mode's subdirectory contains its own member files, full/zoom maps,
per-variable metrics, and precipitation scores.

Judge whether boundary spikes and errors decrease without degradation elsewhere.
Interior and regression predictions should remain unchanged for this sampler.
These five cases are targeted comparisons; use validation cases for subsequent
model tuning, rather than selecting parameters on these test cases.

## Reproducibility

New NetCDF predictions record `noise_padding`. Evaluators and precipitation
audits reject mixed modes within an ensemble. Historical files missing the
attribute are interpreted as `replicate`. Set `inference.noise_padding: replicate`
in a config, or use `NOISE_PADDING=replicate COMPARE_NOISE_PADDING=0` with the test
wrapper, to reproduce the legacy sampler. The `--compare-noise-padding` option
cannot be combined with the single-mode `--noise-padding` option.
