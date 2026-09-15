# Discover annual v2 runbook

`configs/discover_annual_v2.yaml`, all 12 calendar months in training.

| Split | Dates | Hours |
|---|---|---:|
| Training | 2024-12-01 – 2025-11-30 | 8,760 |
| Validation | 2025-12-03 – 2026-01-31 | 1,440 |
| Test | 2026-02-03 – 2026-03-08 | 816 |

- GEOS-FP ends 2026-03-08 23:30; the test split ends with it. HWT labels run past
  that date but cannot be paired. Other years are a later held-out experiment.
- Two-day gaps between splits. YAML endpoints are exclusive.
- Validation and test are mostly cool season: no warm-season held-out claim.
- Creates `data/paired_hourly_annual_v2` (~2.1 TiB, 213 MB per hour) and
  `runs/merraflow_annual_v2`. Neither replaces the 2025 run. Check quota first.
- Shards are keyed on predictors, roots, transforms, `stats_stride` and the
  surface bytes — not the calendar. A moved split boundary keeps them: re-run
  preparation only for months whose hours changed. Finalization revalidates each
  month's entry list and training-hour contribution before merging normalization.
- Old 2025 checkpoints cannot resume into this archive.

## 1. Coverage

From tcsh, enter `bash`, then:

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
python -m merraflow.cli_v2 coverage --config configs/discover_annual_v2.yaml | tee logs_v2/coverage_annual_v2.json
```

Required before proceeding:

- `all_requested_pairs_present: true`
- `paired_hours: 11016`
- `predictor_spot_check.all_sampled_months_have_predictors: true`

`missing_by_source` splits gaps into `lr` (regridded predictors), `hr` (HWT
labels) and `native` (GEOS precipitation). The spot check opens one regridded
file per month, because the regridder writes optional variables such as QV2M and
SLP only when the source has them. Missing `hr` or `native` hours cannot be
regenerated — find the source or revise the dates, never substitute hours.

## 2. Regrid months missing `lr`

```bash
env CONFIG=configs/discover_annual_v2.yaml \
  REGRID_MONTHS='2024-12 2026-01 2026-02 2026-03' \
  bash scripts/submit_regrid_months_v2.sh
```

Omit `REGRID_MONTHS` to sweep every configured month; valid outputs are skipped.
Requires `data/weights/regrid_weights_bilinear_conus.nc` to exist — concurrent
tasks must not write it, and the helper prints the single-month command that
creates it. Six tasks at a time, writing `data/lowres_lcc_1hr/YYYYMM`.

Regridding skips any output holding the v1 required variables, so a file written
before QV2M and SLP were emitted is skipped, not repaired:

```bash
python scripts/repair_lowres_predictors_v2.py \
  --config configs/discover_annual_v2.yaml \
  --report logs_v2/lowres_gaps_annual_v2.json
```

Re-run with `--move-aside` to rename incomplete files `*.incomplete_v2` (nothing
is deleted), regrid those months, then re-check coverage. Skip this whole section
when coverage is already clean. The `data/lowres_lcc_1hr` tree is shared with v1
and stores every optional variable the source has; v1 simply reads fewer.

## 3. Submit

```bash
env CONFIG=configs/discover_annual_v2.yaml bash scripts/submit_pipeline_v2.sh
```

Works from tcsh too. Audits first, then queues 16 preparation tasks,
finalization, 30 regression epochs and 100 flow epochs on `afterok` dependencies;
regression benchmarks five scratch steps first. Job IDs land in
`logs_v2/submission_v2.*`. Do not run it twice with jobs active.

Array index = position in the month list: task 1 is 2024-12, task 13 is 2025-12,
tasks 14–16 are 2026-01 through 2026-03. Each log prints its month.

Preparation and finalization only:

```bash
env CONFIG=configs/discover_annual_v2.yaml bash scripts/submit_prepare_flow_v2.sh
```

One month only (index maps into `PREP_MONTHS`):

```bash
CONFIG=configs/discover_annual_v2.yaml PREP_MONTHS='2026-03' \
  sbatch --export=ALL --array=1-1 scripts/slurm_prepare_flow_v2.sh
```

## 4. Normalization gate

Finalization merges per-channel moments from every training hour into one
`stats_v2.json`; `stats_stride: 8` subsamples grid points, never timestamps.
Validation and test fit nothing and reuse the frozen training normalization.

```bash
python - <<'PY'
import json
path = 'data/paired_hourly_annual_v2/stats_v2.json'
with open(path) as stream:
    stats = json.load(stream)
coverage = stats['training_coverage']
expected = ['2024-12'] + [f'2025-{month:02d}' for month in range(1, 12)]
assert sorted(coverage['hours_by_month']) == expected
assert coverage['split'] == 'train' and coverage['hours'] == 8760
assert sum(coverage['hours_by_month'].values()) == 8760
for group in ('condition', 'residual'):
    assert stats[group]['count_per_channel'] == 8760 * coverage['samples_per_hour_per_channel']
print(json.dumps(coverage, indent=2))
print('PASS: all 12 training months contributed; validation/test excluded.')
PY
```

Training starts only after finalization succeeds.

## 5. Proposal cache

Required before training, or the loader recomputes rain scores per sample and the
GPU sits idle:

```bash
python scripts/build_proposal_cache_v2.py \
  --config configs/discover_annual_v2.yaml --workers 16
```

Reads one channel per prepared hour, writes
`data/paired_hourly_annual_v2/_proposals_v2/size128_stride32.npy` (~74 MB for
11,016 hours). Rerun after adding hours, or with `--force`. A mismatched cache is
ignored rather than misread.

## 6. Monitor, resume, predict

```bash
squeue -u "$USER"
sacct -u "$USER" --starttime today --format=JobID,JobName,State,Elapsed,ExitCode
ls -lt logs_v2
```

One A100, 12 hours per stage. Flow starts after the regression job ends in any
state (`afterany`). It requires an existing `best_v2.pt` to start; a timed-out
regression job can still provide one. The running regression job is unchanged.
For the already queued jobs 58370700 and 58370701, deploy the updated code to
Discover, then replace the pending flow job before it starts. Slurm stores the
batch script at submission, so changing the file or dependency on 58370701
would not add automatic continuation to that queued job:

```bash
cd /gpfsm/dnb10/projects/p311/ML_downscaling
bash -c 'test "$(squeue -h -j 58370701 -o %T)" = PENDING && scancel 58370701 && env CONFIG=configs/discover_annual_v2.yaml STAGE=flow RESUME= REGRESSION_CHECKPOINT=runs/merraflow_annual_v2/regression_v2/best_v2.pt sbatch --export=ALL --job-name=flow_v2 --dependency=afterany:58370700 scripts/slurm_train_flow_v2.sh'
```

If regression needs another 12-hour session before flow starts, submit its
existing resume command and make a new flow job depend on that new job:

```bash
test -f runs/merraflow_annual_v2/regression_v2/last_v2.pt
reg_job=$(env CONFIG=configs/discover_annual_v2.yaml STAGE=regression RESUME=runs/merraflow_annual_v2/regression_v2/last_v2.pt REGRESSION_CHECKPOINT= sbatch --parsable --export=ALL --job-name=regression_v2 scripts/slurm_train_flow_v2.sh)
reg_job=${reg_job%%;*}
env CONFIG=configs/discover_annual_v2.yaml STAGE=flow RESUME= REGRESSION_CHECKPOINT=runs/merraflow_annual_v2/regression_v2/best_v2.pt sbatch --export=ALL --job-name=flow_v2 --dependency="afterany:$reg_job" --kill-on-invalid-dep=yes scripts/slurm_train_flow_v2.sh
```

If automatic submission fails, resume flow manually:

```bash
test -f runs/merraflow_annual_v2/flow_v2/last_v2.pt
env CONFIG=configs/discover_annual_v2.yaml STAGE=flow RESUME=runs/merraflow_annual_v2/flow_v2/last_v2.pt REGRESSION_CHECKPOINT= sbatch --export=ALL --job-name=flow_v2 scripts/slurm_train_flow_v2.sh
```

Flow checks Slurm's remaining wall time after every completed epoch. At 40
minutes or less, or when the last epoch suggests another would not fit safely,
it stops and the batch script submits one continuation with an `afterany`
dependency on the current job. The next job resumes `last_v2.pt`. The 40-minute
threshold can be changed with `FLOW_STOP_MINUTES`; no epoch count per job is
fixed. Every completed epoch is saved as `last_v2.pt`; epoch 5, 10, ... also
have durable recovery files and one-member, full-domain PNG comparisons in
`flow_v2/plots_v2/`.
The plots compare low-resolution input, regression, flow, HWT reference, and flow
error for the same validation timestamp, using the production sampler.

Keep config, GPU count and archive unchanged when resuming. Do not increase
`flow_epochs` between sessions: the checkpoint's optimizer schedule expects the
original 100-epoch target. After flow finishes:

```bash
env CONFIG=configs/discover_annual_v2.yaml CHECKPOINT=runs/merraflow_annual_v2/flow_v2/best_v2.pt SPLIT=val LIMIT=4 sbatch --export=ALL scripts/slurm_predict_v2.sh
```

Select on validation, then evaluate the untouched February–March test split. A
December 2025 event is validation here, not test. See the
[general runbook](runbook_discover_v2.md) for log naming, partial submission
failures and restart limits.

For a v1-style, independent multi-member diagnostic, use the dedicated v2 job
after choosing a flow checkpoint on validation. It samples three seeded,
previously unseen test hours by default and saves full-domain and rain-event
zoom maps, area-weighted RMSE for the coarse baseline, regression, one flow
member and ensemble mean, precipitation skill, and training curves:

```bash
mkdir -p logs_v2
env CONFIG=configs/discover_annual_v2.yaml \
    CHECKPOINT=runs/merraflow_annual_v2/flow_v2/best_v2.pt \
    SPLIT=test SAMPLES=3 MEMBERS=5 \
    sbatch --export=ALL scripts/slurm_test_best_model_v2.sh
```

The default output is `runs/merraflow_annual_v2/test_best_model_<checkpoint>_m5_<hash>_v2/`.
It contains `metrics_v2.json`, `rmse_summary_v2.png`, per-hour `*_full_v2.png`
and `*_zoom_v2.png`, and the generated NetCDF members. The selected timestamps
and checkpoint SHA-256 are recorded in JSON. To inspect the December 3 epoch
plot's hour with several members, run a **validation** diagnostic separately:

```bash
env CONFIG=configs/discover_annual_v2.yaml \
    CHECKPOINT=runs/merraflow_annual_v2/flow_v2/epoch_0025_v2.pt \
    SPLIT=val TIMESTAMPS=20251203_0030 MEMBERS=5 \
    OUTPUT=runs/merraflow_annual_v2/epoch25_validation_v2 \
    sbatch --export=ALL scripts/slurm_test_best_model_v2.sh
```

Use a fresh `OUTPUT` directory for another checkpoint or rerun; the script
refuses to mix or overwrite prediction members. The default `SPLIT=test` is
for final evaluation after validation-based checkpoint choice. A single
validation image or timestamp is not a held-out test result.

To include the 23 February 2026 event in a five-case test diagnostic, set
`SAMPLES=5 INCLUDE_DATE=2026-02-23`. The script selects that UTC day's hour
with the highest domain-area-weighted HWT precipitation, then draws four test
hours from other dates using the fixed seed. `metrics_v2.json` records the selected hour
and selection rule. Because one case is chosen using observed precipitation,
the five-case average is an event-focused diagnostic, not a random estimate of
overall test skill:

```bash
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_NODE SLURM_MEM_PER_GPU
env CONFIG=configs/discover_annual_v2.yaml \
    CHECKPOINT=runs/merraflow_annual_v2/flow_v2/best_v2.pt \
    SPLIT=test SAMPLES=5 MEMBERS=5 INCLUDE_DATE=2026-02-23 \
    OUTPUT=runs/merraflow_annual_v2/best_flow_feb23_five_cases_v2 \
    sbatch --export=ALL scripts/slurm_test_best_model_v2.sh
```

The `unset` makes submission safe from a login shell or an existing interactive
Slurm allocation; otherwise `--export=ALL` can carry that allocation's memory
variables into the new batch job and make its internal `srun` reject them.

For the completed five-case run, use the [precipitation audit](precip_audit_v2.md)
to inspect saved member amounts, wet areas, upper tails, spatial-scale errors,
spectra, calibration and possible displacement. It runs as a CPU compute job
and reuses `best_flow_feb23_five_cases_v2/predictions_v2`.
