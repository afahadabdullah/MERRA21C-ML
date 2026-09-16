# Trace Copy1 flow inference

This diagnostic generates new predictions directly from a specified checkpoint.
It needs the prepared archive and checkpoint, not any previous test outputs.
It uses the production Heun sampler with optional read-only observation hooks.
It holds weights, archive normalization, precision, tiling, blending and initial
noise fixed while comparing 24, 48 and 96 steps. No model settings are calibrated
or changed by this diagnostic.

## Submit on Discover

```bash
cd /gpfsm/dnb10/projects/p311/ML_downscaling
git pull --ff-only
mkdir -p logs_v2
env -u SLURM_MEM_PER_CPU -u SLURM_MEM_PER_NODE -u SLURM_MEM_PER_GPU \
  CONFIG=configs/discover_annual_v2.yaml \
  CHECKPOINT=runs/merraflow_annual_v2/flow_v2/best_v2-Copy1.pt \
  OUTPUT=runs/merraflow_annual_v2/copy1_flow_trace_v2 \
  SPLIT=test MEMBERS=5 STEPS="24 48 96" \
  TIMESTAMPS="20260223_0530 20260209_1530 20260209_2030 20260305_1230 20260306_1830" \
  sbatch --export=ALL scripts/slurm_diagnose_flow_v2.sh
```

The array contains five tasks, one timestamp per task, with at most two active
tasks. Each requests one A100, four CPUs, 48 GB host memory per GPU, and up to
12 hours. Logs: `logs_v2/flow_diag_<array-ID>_<task-index>_v2.log` and `.err`.
This performs seven times as many ODE steps as the original 24-step five-case
test, and adds CPU diagnostics and file writes. The walltime is a requested
limit, not a runtime estimate. Complete raw fields can occupy several GB.
Use a fresh OUTPUT on reruns. Change the sbatch array range if changing the
timestamp list; index 0 is February 23 and index 4 is March 6.

To investigate March 6 first, use the same command with
`sbatch --array=4 --export=ALL scripts/slurm_diagnose_flow_v2.sh`.
Remaining tasks can later use `--array=0-3%2` with the same output root, since
each task writes its own fresh subdirectory.

## Files to inspect

Each task writes `case_NN_v2/manifest_v2.json` and a timestamp subdirectory.
The manifest records the checkpoint path, SHA256, epoch, all five `flow_scale`
values, complete archive statistics, and the effective rainfall log multiplier
`flow_scale[precip] * residual_std[precip]`. Weights are loaded once per task;
hashes are checked before and after loading. Keep the copied checkpoint stable
throughout the array so each task loads the same file contents.

Within the timestamp directory:

- `report_v2.md`: compare RMSE, CRPS and the largest member rainfall at each
  step count. `summary_v2.json` also records same-member, same-seed changes
  between adjacent step counts in physical rainfall and raw endpoint space.
- `normalization_checks_v2.json`: verifies stored archive residuals against
  `transform(HWT) - transform(coarse)` and measures the HWT transform round trip.
  The inference loader also requires the checkpoint's statistics to match the
  archive exactly. A consistent transform does not establish good calibration.
- `reference_v2.nc`: HWT and coarse rainfall on the complete grid.
- `steps_NNN_v2/predictions_v2/*_mNNN_v2.nc`: physical rainfall, regression,
  normalized residuals, corrections before/after archive scaling, log rainfall
  before clipping/exponentiation, reconstructed blended raw endpoint, the
  endpoint required to reach HWT relative to blended regression, tile overlap
  counts, and tile disagreement in log-rainfall units. Lat/lon/area are included.
- `steps_NNN_v2/*_internal_v2.png`: maps of those stages over the entire domain,
  with percentile display limits and marked extreme pixels. The NetCDF fields
  and all metrics retain the extremes; only colors saturate.
- `steps_NNN_v2/*_trace_v2.json`: unweighted grid statistics for every tile's
  endpoint and each step of selected tile trajectories. The northwest, central,
  and wettest mean-HWT tiles are traced (deduplicated when equal).
- `steps_NNN_v2/*_trajectory_v2.npz`: raw state maps at the start, approximate
  quarter times, and endpoint for the traced tiles. Intermediate states are
  transport states, not valid precipitation forecasts.
- `steps_NNN_v2/audit_v2.json` and the standard audit PNGs: physical rainfall
  distributions, spectra, scores and calibration. Added `structure` metrics
  compare connected wet objects and isolated small objects at 0.1, 1 and 5 mm/h,
  plus neighboring-pixel differences at tile core edges versus elsewhere.

If sampling becomes nonfinite or decoding exceeds its existing guard, the task
fails rather than clipping the model's extremes. A failure JSON and available
trace data are retained; full pre-decoding fields are saved when frame assembly
has completed. These are diagnostic outputs, not a successful evaluation.

## Interpretation

The exact rainfall reconstruction is
`L = log1p(coarse/s) + rm + rs * (regression + flow_scale * endpoint)` followed
by `P = s * expm1(max(L, 0))`. The diagnostic observes this existing equation.

Large raw endpoints show excessive values before inverse-log decoding. Their
effect on rainfall depends on both scales and the coarse/regression background.
Decreasing differences from 24→48→96 steps support numerical convergence;
persistently bad structure after convergence points toward learned residuals,
conditioning or tiling rather than simply too few steps. This is evidence to
investigate, not automatic causal attribution.

Many small wet objects at comparable wet area support the visible speckling
diagnosis. Tile disagreement quantifies independently evolved predictions for
the same pixel before blending, but legitimate context dependence can also
produce disagreement. Core-edge gradients depend on weather location. Neither
is proof of a stitching bug. Thresholds, connectivity and grid resolution affect
object counts. Use validation cases before selecting changes to production.

## Local checks

Focused tests cover unchanged production samples with/without observation for
both blend modes, analytic Heun convergence, retained fields on decode overflow,
isolated pixels versus connected bands, a synthetic checkpoint-to-NetCDF/plots
run, fixed seeds across steps, checkpoint preservation, and array/memory argument
handling. These establish software behavior, not skill of the Discover model.
