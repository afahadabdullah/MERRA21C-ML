# Discover annual v2 runbook

Use `configs/discover_annual_v2.yaml` for a full annual training cycle. This
configuration assumes the required inputs exist through **March 31, 2026**;
that endpoint has not been verified from the local workspace.

| Split | Included dates | Expected hourly pairs |
|---|---|---:|
| Training | December 1, 2024–November 30, 2025 | 8,760 |
| Validation | December 3, 2025–January 31, 2026 | 1,440 |
| Test | February 3–March 31, 2026 | 1,368 |

December 1–2, 2025 and February 1–2, 2026 are excluded as two-day gaps between
splits. YAML split endpoints are exclusive. Training includes all 12 calendar
months; validation and test mainly cover the cool season, so they cannot establish
warm-season held-out skill. The existing 2025 configs keep their original ranges.

This run creates `data/paired_hourly_annual_v2` and `runs/merraflow_annual_v2`.
The archive stores 28 float32 channels per hour on the 1,059 x 1,799 HWT grid,
about 213 MB per paired hour, so 11,568 hours need roughly **2.3 TiB**. It does
not reuse or replace the 2025 archive; confirm the quota before submitting.
Use fresh annual training; the changed archive and normalization are incompatible
with resuming an old 2025 checkpoint. It reuses the same geographic surface file,
`data/static_grid/hwt_surface_fractions_v2.nc`. A changed HWT grid is rejected
during preparation rather than silently assigning the old mask to it.

## 1. Activate and check coverage on Discover

From your tcsh login prompt, first enter Bash:

```text
bash
```

Then paste:

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

This reports every requested month. `missing_by_source` separates missing
regridded predictors (`lr`), HWT labels (`hr`), and native GEOS precipitation
(`native`), and example missing paths are printed. A month existing in one archive
does not prove the other two are complete. `predictor_spot_check` additionally
opens one regridded file per month and lists any configured predictor it lacks,
because the regridder writes QV2M, SLP and OMEGA500 only when the GEOS source
provides them. Check `all_requested_pairs_present: true`, `paired_hours: 11568`
and `predictor_spot_check.all_sampled_months_have_predictors: true` before
proceeding. Preparation still validates every paired hour, not one per month.
The submission preflight additionally checks sample NetCDF contents and the mask.

If HWT (`hr`) or native GEOS precipitation (`native`) hours are missing, locate
those sources or revise the date ranges to the actual complete period. They
cannot be regenerated here. Do not silently replace missing hours.

## 2. Regrid the months that lack low-res predictors

Months reported as missing only `lr`, and months whose `predictor_spot_check`
lists missing variables, need regridding first. The fixed-2025 scripts do not
cover December 2024 or 2026, so use the month-list helper:

```bash
env CONFIG=configs/discover_annual_v2.yaml \
  REGRID_MONTHS='2024-12 2026-01 2026-02 2026-03' \
  bash scripts/submit_regrid_months_v2.sh
```

Omit `REGRID_MONTHS` to sweep every month in the config; already valid outputs
are skipped, so a sweep is safe but slower. The cached bilinear weights at
`data/weights/regrid_weights_bilinear_conus.nc` must already exist, because
concurrent tasks must not write them; the helper refuses to submit without them
and prints the single-month command that creates them. Tasks run at most six at
a time and write into `data/lowres_lcc_1hr/YYYYMM`.

Regridding skips any existing output that holds the v1 required variables
(`T2M`, `U10M`, `V10M`, `PS`, `TQV`, `PRECTOT`), so a file written before QV2M,
SLP and OMEGA500 were emitted is skipped rather than repaired. Staging those
files aside is what makes the array rebuild them:

```bash
python scripts/repair_lowres_predictors_v2.py \
  --config configs/discover_annual_v2.yaml \
  --report logs_v2/lowres_gaps_annual_v2.json
```

This opens every paired hour's regridded file and lists the ones missing a
configured predictor. Re-run it with `--move-aside` to rename them
`*.incomplete_v2` — nothing is deleted — then resubmit the regridding array for
the affected months. Re-run the coverage command afterwards and confirm that
`predictor_spot_check` reports no gaps.

Skip this section entirely when coverage already reports complete pairs and no
predictor gaps. The regridded archive itself is shared with v1: one writer, one
`data/lowres_lcc_1hr` tree, and every optional GEOS state variable present in
the source is written regardless of which config consumes it. V1 simply reads
four of them, so 2025 files written by the current regridder serve v2 unchanged.

## 3. Submit the annual run

```bash
env CONFIG=configs/discover_annual_v2.yaml bash scripts/submit_pipeline_v2.sh
```

This invocation also works directly in tcsh after changing to the project root:
the helper sources conda and activates the environment itself. It audits before
submitting anything, then queues **16 monthly preparation tasks**, finalization,
30 regression epochs, and 100 flow epochs. The stages use success dependencies;
regression first runs a short scratch GPU benchmark. Months and job IDs are saved
in `logs_v2/submission_v2.*`. Do not invoke it twice while jobs are active.

Array indices are positions in the saved month list: task 1 is December 2024,
task 2 is January 2025, task 13 is December 2025, and tasks 14–16 are January–March
2026. Each preparation log prints the actual month. Months come from the config;
there is no `PREP_YEAR` setting to maintain.

For preparation and finalization only, use this **instead of** the full pipeline:

```bash
env CONFIG=configs/discover_annual_v2.yaml bash scripts/submit_prepare_flow_v2.sh
```

## 4. Verify that normalization includes all training months

Finalization merges per-channel means and variances from **every training hour**
across December 2024–November 2025 into one `stats_v2.json`. It does not average
monthly standard deviations or normalize each month separately. The current
`stats_stride: 8` samples every eighth grid row/column in every training hour;
it does not skip timestamps. Validation and test contribute no fitted statistics
and use the same frozen training normalization. The ocean/land/lake file remains
a geographic input, independent of these atmospheric statistics.

After finalization completes, run:

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
print(path)
print(json.dumps(coverage, indent=2))
print('PASS: all 12 training months contributed; validation/test excluded.')
PY
```

The file records each month's training-hour count, the first and last fitted
timestamps, and spatial sampling. Finalization rejects an aggregate sample
count that does not match all training hours. Training begins only after
finalization succeeds.

## 5. Monitor, resume and validate

```bash
squeue -u "$USER"
sacct -u "$USER" --starttime today --format=JobID,JobName,State,Elapsed,ExitCode
ls -lt logs_v2
```

Jobs request one A100 and 12 hours per training stage. If regression times out,
wait until the old regression is stopped and its dependent flow job is canceled,
then run these commands in the activated Bash shell:

```bash
test -f runs/merraflow_annual_v2/regression_v2/last_v2.pt
reg_job=$(env CONFIG=configs/discover_annual_v2.yaml STAGE=regression RESUME=runs/merraflow_annual_v2/regression_v2/last_v2.pt REGRESSION_CHECKPOINT= sbatch --parsable --export=ALL --job-name=regression_v2 scripts/slurm_train_flow_v2.sh)
reg_job=${reg_job%%;*}
env CONFIG=configs/discover_annual_v2.yaml STAGE=flow RESUME= REGRESSION_CHECKPOINT=runs/merraflow_annual_v2/regression_v2/best_v2.pt sbatch --export=ALL --job-name=flow_v2 --dependency="afterok:$reg_job" --kill-on-invalid-dep=yes scripts/slurm_train_flow_v2.sh
```

If flow alone times out, resume it after its previous job stops:

```bash
test -f runs/merraflow_annual_v2/flow_v2/last_v2.pt
env CONFIG=configs/discover_annual_v2.yaml STAGE=flow RESUME=runs/merraflow_annual_v2/flow_v2/last_v2.pt REGRESSION_CHECKPOINT= sbatch --export=ALL --job-name=flow_v2 scripts/slurm_train_flow_v2.sh
```

Keep the config, GPU count and archive unchanged when resuming. After flow
finishes successfully, generate eight members for four validation hours:

```bash
env CONFIG=configs/discover_annual_v2.yaml CHECKPOINT=runs/merraflow_annual_v2/flow_v2/best_v2.pt SPLIT=val LIMIT=4 sbatch --export=ALL scripts/slurm_predict_v2.sh
```

Inspect `runs/merraflow_annual_v2/predictions_v2`. Use validation for model
selection, then evaluate the untouched February–March test split. A December
2025 event belongs to validation in this configuration, not the test split.
The [general runbook](runbook_discover_v2.md) describes log naming, partial
submission failures and restart limitations; use the annual config and paths above.
