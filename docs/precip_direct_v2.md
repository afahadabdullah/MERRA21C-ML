# Direct precipitation generation with the v2 archive

This experiment trains **only a new one-channel conditional flow model**. The
original v2 regression is frozen and supplies a predicted rainfall field as an
extra input. No regression retraining, hourly-target preparation, or changes to
the existing v2/v3 runs are required.

The target is the full `log1p(HWT rainfall / scale)` field. Rainfall is the only
supervised output. Noise flows to that field through a straight flow-matching
path. The coarse field and frozen regression prediction are conditions only:
neither is subtracted from the training target or added to the flow output.
The word "residual" in the U-Net's internal skip blocks does not change this
target definition.

The frozen v2 regression itself predicts a standardized correction. We convert
that correction to its full encoded rainfall prediction using its saved v2
statistics and coarse baseline **only when constructing the conditioning input**.
Its weights never enter the optimizer. The checkpoint stores a copy of those
weights, so prediction and resume do not need the original source file.

## Submit on Discover

The default source is the original annual run's regression checkpoint:
`runs/merraflow_annual_v2/regression_v2/best_v2.pt`. This is a configured path,
not a claim that the file has been checked on Discover. The submission helper
checks the archive and checkpoint before calling `sbatch`.

```bash
cd /gpfsm/dnb10/projects/p311/ML_downscaling
git pull --ff-only origin main
bash scripts/submit_precip_direct_v2.sh
```

If the original checkpoint is elsewhere:

```bash
REGRESSION_CHECKPOINT=/path/to/original_v2/regression_v2/best_v2.pt \
  bash scripts/submit_precip_direct_v2.sh
```

The helper invokes the project's Python directly, even from Conda base. It
submits one four-A100 job (4 CPUs and 32 GB host memory per GPU, 12 hours).
Training stops at a completed epoch near its 11-hour internal budget. If epochs
remain, the job submits one dependent continuation automatically. It does not
pre-submit eight segments. Failed training does not submit a continuation.

Training defaults mostly match the original annual v2 scale: base width 32,
100 epochs, 8,192 patches/epoch, batch 8 per GPU and accumulation 2 (effective
batch 64). Optimization is a quadratic full-field velocity loss including halo
pixels. Rain/coast patch proposals retain v2 inverse-proposal weights. No other
target variables or auxiliary rollout objectives are supervised.

## Validation and outputs

Every five epochs and at the final epoch, generate 4 members with 24 Heun steps
on 128 fixed uniform validation patches. All four ranks contribute metrics;
rank zero saves fixed-patch maps using a separate CPU coordination group. These
are patch validation plots, not whole-CONUS evaluations.

Outputs relative to the project root:

```text
runs/merraflow_precip_direct_v2/last_direct_v2.pt
runs/merraflow_precip_direct_v2/best_direct_v2.pt
runs/merraflow_precip_direct_v2/history.json
runs/merraflow_precip_direct_v2/validation_plots/epoch_0005/fields.png
runs/merraflow_precip_direct_v2/validation_plots/epoch_0005/history.png
runs/merraflow_precip_direct_v2/validation_plots/epoch_0005/samples.npz
runs/merraflow_precip_direct_v2/validation_plots/epoch_0005/metrics.json
logs_precip_direct_v2/train_<jobid>.log
```

Maps compare truth, coarse input, frozen regression input, two generated members,
ensemble mean and spread. Best checkpoints use rainfall CRPS. Logs also include
ensemble-mean RMSE, rainfall bias, spread and wet-area fractions (0.1 mm/h).
Validation estimates behavior on patches; it does not establish full-domain
skill. The original v2 targets are HWT **:30 snapshots**, not the separately
prepared v3 trapezoid hourly means.

### Targeted rainy-case check during training

The fixed validation previews are uniformly selected and may be almost dry.
An independent one-GPU job can select genuinely wet **validation** patches from
HWT truth, then compare four generated members with the coarse input and frozen
v2 regression on those patches. The selection reads truth only to choose and
score evaluation cases; truth never enters the model condition.

```bash
cd /gpfsm/dnb10/projects/p311/ML_downscaling
git pull --ff-only origin main
bash scripts/submit_wet_eval_precip_direct_v2.sh
```

The helper snapshots the current `last_direct_v2.pt` before submission so a
training checkpoint update cannot alter the evaluated epoch. Choose the best
checkpoint with `CHECKPOINT_KIND=best` or another path with `CHECKPOINT=...`.
Defaults scan 64 evenly spaced validation hours and choose six distinct hours,
each with a patch containing at least 10% HWT wet pixels (>=0.1 mm/h). Select
more cases with `SCAN_HOURS=128 CASES=12`; lower the threshold with
`MIN_WET_FRACTION=0.05` if too few cases qualify. The results directory is
printed by the helper and contains `report.json`, map PNGs and member arrays.

These metrics describe selected rainy cases and cannot replace the existing
uniform validation CRPS. In particular, a rainy-case gain does not establish
that the model has fixed false rain on dry cases.

### Spatial audit of an existing wet evaluation

The wet evaluation saves each case's truth, coarse field, frozen regression,
generated members, and area weights in `results/case_*.npz`. Score those saved
fields on CPU without resampling or using a GPU:

```bash
cd /gpfsm/dnb10/projects/p311/ML_downscaling
git pull --ff-only origin main
export PYTHONPATH="$PWD/src"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/precip-spatial-${USER}"
RESULTS=runs/merraflow_precip_direct_v2/wet_evaluations/last_wet_KAF9LG/results
env/bin/python -m merraflow.analyze_wet_precip_direct_v2 \
  --results "$RESULTS" \
  --history runs/merraflow_precip_direct_v2/history.json
```

Replace `RESULTS` with the directory printed by your wet evaluation submission
if it differs. The command writes `spatial_report.json` and `spatial_skill.png`
inside that results directory. It requires the saved `.npz` files, not just
`report.json` and PNGs. It also selects the **same epoch** from the uniform
validation history, refusing to compare another epoch by accident.

The plot compares mean individual-member, ensemble-mean, coarse, and frozen-v2
fraction skill scores (FSS) at 1, 5, and 10 mm/h. Neighborhood widths are 1,
5, 17, and 33 pixels (nominally 3, 15, 51, and 99 km). Scores rise toward 1
with better spatial agreement. If skill is poor at one pixel but improves at
larger neighborhoods, the rain features are close but displaced or miss fine
detail. If it stays poor even at 51–99 km, the broader event placement is wrong.
The selected wet cases remain distinct from the uniform validation scores;
both must be inspected before judging overall model skill.

### Whole-CONUS February 23 case

The CPU-only plotter reads already saved whole-domain NetCDF members for the
existing test hour `20260223_0530` (2026-02-23 05:30 UTC). It detects whether
the files are from the **direct full-rainfall flow** or the **original residual
v2 flow** and labels the figure accordingly. It never presents original v2
members as direct-flow output. This path needs no GPU when direct full-domain
members have already been saved.

With the project Conda environment activated on a compute node:

```bash
cd /gpfsm/dnb10/projects/p311/ML_downscaling
git pull --ff-only origin main
export PYTHONPATH="$PWD/src"
python -m merraflow.plot_saved_conus_precip_v2 --predictions runs --list
```

This lists all matching directories and model kinds. If one direct-flow
directory exists, the plotter selects it. If none exists, it can use one
original residual-v2 directory and states that clearly in the figure. If
multiple directories match, pass the desired exact directory using
`--predictions`. Plot from saved files with:

```bash
export MPLCONFIGDIR="${TMPDIR:-/tmp}/precip-conus-${USER}"
python -m merraflow.plot_saved_conus_precip_v2 --predictions runs --output runs/merraflow_precip_direct_v2/saved_conus_feb23
```

The fresh output directory contains a full-CONUS PNG and `report.json` with
the source NetCDF paths, area-weighted rainfall metrics, and neighborhood FSS.
The PNG shows HWT truth, coarse input, up to two members, ensemble mean, and
spread on a shared rain-rate scale. This selected case is a visual diagnostic,
not an overall test-set score. The plotter requires saved **member NetCDF**
files; a previously saved PNG alone cannot provide new model comparisons.

The September 2026 search on Discover listed **only residual-v2** February 23
members. The direct-flow wet evaluation saved 128-pixel patches, not a
whole-CONUS field. To make the same seven panels for the **direct** model at
epoch 40, submit one full-domain inference job from the immutable wet-evaluation
checkpoint:

```bash
cd /gpfsm/dnb10/projects/p311/ML_downscaling
git pull --ff-only origin main
bash scripts/submit_full_conus_precip_direct_v2.sh
```

The job uses one A100, two members, 24 steps, and the test hour
`20260223_0530`. It writes `HWT truth`, `Coarse input`, `Frozen v2 input`,
`Member 1`, `Member 2`, `Ensemble mean`, and `Ensemble spread` with one color
scale to `runs/merraflow_precip_direct_v2/wet_evaluations/last_wet_KAF9LG/full_conus_20260223_0530/20260223_0530_full_conus_direct_v2.png`.
The same folder holds a report and the full generated arrays. The helper
prints the Slurm job ID and log path. If the 12-hour allocation ends after a
member is saved, rerun the same submission command; completed members are
reused after checkpoint/settings validation. A saved whole-domain original-v2
residual field cannot replace the missing direct-flow output.

The current training sampler already devotes 40% of its proposal mixture to
rain/coast scores (`patch.detail_fraction: 0.4`). Its inverse-proposal weight
`1/(N*q)` returns the objective to uniform patch risk. Increasing rainy-case
emphasis by removing that weight would change the training objective and need
a separate run; it may worsen false rain on dry pixels, which is currently the
dominant observed error. The running job should keep its current settings.

Resume after a failed allocation (first verify no continuation is still queued):

```bash
RESUME=runs/merraflow_precip_direct_v2/last_direct_v2.pt \
  bash scripts/submit_precip_direct_v2.sh
```

Resume retains optimizer, scheduler, EMA, each rank's RNG and frozen regression.
The GPU count and training settings must match. An interrupted plot is recovered
from the completed epoch checkpoint. Outputs from the original v2 residual run
cannot be used as a direct-model resume checkpoint.

## Optional comparisons

Default flow weights start fresh. To transfer internal EMA weights from an
original v2 flow checkpoint, also set `INITIALIZE_V2=/path/to/flow_v2/best_v2.pt`
on the submission command. Input, conditioning and output layers are reset;
the optimizer and schedule start fresh. The source must match the archive,
statistics and model architecture. Known fine-tuned checkpoints are rejected.
This is an optional transfer experiment, not a lossless conversion of an old
residual flow model, and its benefit is unmeasured.

For a model with coarse/static inputs only, set `REGRESSION_CHECKPOINT=none`.
Use a separate `train.output` in a copied config for each comparison; the
helper rejects a nonempty run directory unless explicitly resuming it.

Full-domain inference uses one shared noise field and blends overlapping tile
velocities at every Heun step. Run inside a compute allocation:

```bash
export PYTHONPATH="$PWD/src"
/gpfsm/dnb10/projects/p311/ML_downscaling/env/bin/python \
  -m merraflow.cli_precip_direct_v2 predict \
  --config configs/discover_precip_direct_v2.yaml \
  --checkpoint runs/merraflow_precip_direct_v2/best_direct_v2.pt \
  --split val --limit 1
```

NetCDF members go to the configured `inference.output`. This inference path is
memory conservative and rereads tile conditions at each step; real throughput
and A100 memory requirements have not been measured.

## Existing v2 multi-GPU plots

The original residual v2 trainer now supports full-domain plots under DDP too.
All ranks participate in coordination; rank zero uses its unwrapped EMA model
to render, and the others wait on a separate Gloo group. Training's NCCL
timeout is unchanged. Plot errors reach every rank. `FLOW_PLOT_INTERVAL=5`
is the default; a resumed checkpoint at epoch 15 recreates a missing epoch-15
plot before training epoch 16. Running Python processes need a checkpoint
restart to load this fix; pulling code does not patch them in memory.
