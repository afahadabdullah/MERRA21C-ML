# Discover v2 runbook

For all 12 training months from December 2024 through November 2025, use the
[annual runbook](runbook_discover_annual_v2.md) and its separate config/output paths.
The commands below retain the original 2025 experiment.

The supplied setup uses the existing 2025 inputs, one static NetCDF for ocean,
land and lakes, and one A100 per training job. It prepares a separate v2 archive,
trains regression for 30 epochs, then trains flow for 100 epochs using the best
completed regression checkpoint. GPU jobs currently request 12 hours each;
resuming may be necessary. These resource settings come from the existing project
batch scripts; actual availability and performance must be checked on Discover.

## 1. Pull and activate the environment

Your Discover login prompt uses tcsh. First paste this command by itself:

```text
bash
```

Then paste the following into that Bash shell. Use this setup again after a new
login. Conda activation must happen before enabling `set -u`.

```bash
set -eo pipefail
set +u
source /discover/nobackup/projects/GEOS_MITgcm/afahad/conda/etc/profile.d/conda.sh
conda activate /gpfsm/dnb10/projects/p311/ML_downscaling/env
set -u
cd /gpfsm/dnb10/projects/p311/ML_downscaling
git pull --ff-only
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
mkdir -p logs_v2
```

## 2. Check the already-generated static file

There is no need to regenerate the file whose plot was inspected. This verifies
all three stored fractions, including their sum, before submission:

```bash
python - <<'PY'
import numpy as np
import xarray as xr
path = 'data/static_grid/hwt_surface_fractions_v2.nc'
with xr.open_dataset(path) as ds:
    names = ('FRLAND', 'FRLAKE', 'FROCEAN')
    arrays = [ds[name].values for name in names]
    assert arrays[0].shape == (1059, 1799), arrays[0].shape
    for name, values in zip(names, arrays):
        assert values.shape == arrays[0].shape
        assert np.isfinite(values).all(), name
        assert ((values >= 0) & (values <= 1)).all(), name
        assert values.max() > 0, f'{name} is empty'
        print(name, 'shape=', values.shape, 'min=', values.min(),
              'max=', values.max(), 'mean=', values.mean())
    np.testing.assert_allclose(sum(arrays), 1, rtol=0, atol=1e-6)
print('Static fractions PASS:', path)
PY
```

The shape above is from your reported HWT grid. The PNG preview is subsampled;
its axis lengths do not indicate a smaller training grid. The following audit
also checks geographic alignment against the actual HWT data.

## 3. Submit preparation and both training stages

Run this once for a fresh training run. The helper activates conda itself, so it
also works directly from tcsh when invoked with `bash`.

```bash
bash scripts/submit_pipeline_v2.sh
```

Before any `sbatch`, it runs the read-only v2 audit. The audit requires complete
input pairs with the current `strict_missing: true`, nonempty train/val/test
splits, representative predictor availability, and valid grid-aligned surface
fractions. Its JSON includes `lake_fraction_known: true` and fraction statistics.
It checks predictors on representative hours; preparation checks every hour.
An audit error stops submission. Resolve the reported input paths, dates or
predictors before retrying; disabling strict checks is not part of this runbook.

The helper prints and records these four job IDs in `logs_v2/submission_v2.*`:

1. Twelve monthly CPU preparation tasks for 2025.
2. Finalization after every month succeeds, creating the archive and training statistics.
3. Regression after finalization succeeds. Its GPU job benchmarks both stages
   for five scratch steps before starting training.
4. Flow after regression succeeds, loading `runs/merraflow_v2/regression_v2/best_v2.pt`.

Downstream jobs are canceled if their dependency fails. Submitting the chain
does not mean the audit has verified CUDA, every predictor file, or full-run
memory requirements. The short benchmark reports timing and peak allocation;
EMA/checkpoint memory can add to training requirements.

Compatible monthly shards and finalized archives can be reused. If preparation
reports incompatible metadata from an earlier static configuration, preserve
that archive and select a fresh `data.prepared` directory whose name ends in
`v2`. Do not mix old shards with the new mask. The helper refuses nonempty
training directories: use the resume commands below for an interrupted run.
Avoid submitting the helper twice while its jobs are pending or running.
If `sbatch` fails partway through submission, already accepted jobs remain
queued. Consult the submission record and `squeue` before retrying.

## 4. Monitor and locate outputs

```bash
squeue -u "$USER"
sacct -u "$USER" --starttime today --format=JobID,JobName,State,Elapsed,ExitCode
ls -lt logs_v2
```

Preparation logs are `logs_v2/prepare_<array-ID>_<month>_v2.log`. Both training
stages use `logs_v2/flow_<job-ID>_v2.log` and `.err`; match their IDs to the
submission record. A pending job with reason `Dependency` is waiting normally.

| Output | Location relative to project root |
|---|---|
| Finalized prepared archive | `data/paired_hourly_v2/index_v2.json` |
| Prepared static fields | `data/paired_hourly_v2/grid_v2.nc` |
| Training normalization | `data/paired_hourly_v2/stats_v2.json` |
| Regression checkpoint | `runs/merraflow_v2/regression_v2/best_v2.pt` |
| Flow checkpoint | `runs/merraflow_v2/flow_v2/best_v2.pt` |
| Epoch logs | `history_v2.jsonl` in each stage directory |
| Latest restart checkpoint | `last_v2.pt` in each stage directory |

Check job state `COMPLETED` and exit code `0:0` for each stage; `best_v2.pt` exists
before a run has necessarily finished. Keep configs, source code and the static
file stable while the chain is running.

## 5. Recover an interrupted job

After failed preparation, resolve the input error and rerun the pipeline helper
once its old jobs are no longer active. Compatible shards resume automatically.

If regression times out, wait until the old regression is stopped and its
dependent flow job is canceled. In the activated Bash shell, submit a regression
resume and a new dependent flow job:

```bash
test -f runs/merraflow_v2/regression_v2/last_v2.pt
reg_job=$(STAGE=regression RESUME=runs/merraflow_v2/regression_v2/last_v2.pt REGRESSION_CHECKPOINT= sbatch --parsable --export=ALL --job-name=regression_v2 scripts/slurm_train_flow_v2.sh)
reg_job=${reg_job%%;*}
STAGE=flow RESUME= REGRESSION_CHECKPOINT=runs/merraflow_v2/regression_v2/best_v2.pt sbatch --export=ALL --job-name=flow_v2 --dependency="afterok:$reg_job" --kill-on-invalid-dep=yes scripts/slurm_train_flow_v2.sh
```

If flow times out after regression completed, resume flow alone:

```bash
test -f runs/merraflow_v2/flow_v2/last_v2.pt
STAGE=flow RESUME=runs/merraflow_v2/flow_v2/last_v2.pt REGRESSION_CHECKPOINT= sbatch --export=ALL --job-name=flow_v2 scripts/slurm_train_flow_v2.sh
```

Resume retains the same config, GPU count and run directory. It restarts from
the last saved epoch, and a flow checkpoint already contains its frozen
regression weights. If no epoch checkpoint exists, these resume commands cannot
run; inspect the error before deciding how to restart the incomplete directory.

## 6. Validate the completed flow model

After flow finishes successfully, generate eight members for four validation
hours and evaluate/plot them on a GPU job:

```bash
CHECKPOINT=runs/merraflow_v2/flow_v2/best_v2.pt SPLIT=val LIMIT=4 sbatch --export=ALL scripts/slurm_predict_v2.sh
```

Inspect outputs under `runs/merraflow_v2/predictions_v2`. This is a small software
and visual check, not a full skill evaluation. Prediction refuses to overwrite
existing members; use a fresh v2 prediction directory in a copied config for
another checkpoint or a larger evaluation. Use validation for model selection
before evaluating the held-out test split.
