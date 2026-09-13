# V2 local validation record

Validated 2026-09-12 in the existing temporary Python environment, PyTorch 2.14.0,
on CPU. This records software validation, not measured meteorological improvement.

- Full suite: **40 passed in 10.60 seconds**, including all pre-existing v1 tests.
- Synthetic end-to-end run: six hourly raw fixtures, v2 preparation, two regression
  epochs, two flow epochs, two generated validation members, physical evaluation,
  maps, spectra and training-history plots. Diagnostic map layout visually inspected.
- Exact epoch-boundary resume checked independently for both stages; resulting
  model weights match uninterrupted training bit-for-bit on CPU.
- Original HWT precipitation and signed wind targets verified; inference tested
  to allow positive rainfall inside native coarse-dry footprints.
- FROCEAN-only availability flag/warning, FROCEAN with FRLAKE, first-HR lookup,
  coordinate mismatch, invalid fractions and external-static mutation guards tested.
- Monthly shard resume/finalization and unlabeled inference with frozen training
  statistics tested. Detail-sampling inverse-proposal expectations verified.
- Default production-sized local forward pass: `(1, 5, 192, 192)`, finite.
  Default regression model: 3,304,741 parameters. Large preset: 7,016,309.
- Read-only preflight and scratch benchmark exercised on synthetic data.
- All v2 shell scripts pass `bash -n`.
- SHA-256 comparison confirms all 43 pre-existing source/config/script/test files
  are unchanged. All new maintained file basenames contain `v2`.

The generated GSHHG surface preview supplied from Discover was inspected at
domain scale: coastlines and the Great Lakes are visibly separated. The actual
Discover NetCDF has not been read locally. Production presets now require
FROCEAN and FRLAKE from that one file; the runbook checks stored fractions and
the preflight verifies alignment against HWT before submission.

After static-file integration, the full CPU suite passes **42 tests** (11.36 s),
including preflight rejection of missing pairs/empty splits and surface-fraction
reporting. Bash syntax checks pass for all v2 batch/submission scripts. The new
submission helper's four-job dependency chain, zero-submission behavior on
audit failure, and preservation of job IDs on partial submission failure were
checked using a mocked scheduler; no real jobs were queued.

Not validated locally: the generated Discover NetCDF's numerical contents,
CUDA/BF16/DDP execution, A100 memory/cost, or improvement on actual held-out HWT
events. The included preflight, benchmark and validation commands support those checks.
No remote training job was inspected, modified, stopped or submitted.

## Annual preset (December 2024–November 2025 training)

Added after the 42-test record above: `configs/discover_annual_v2.yaml`,
config-derived preparation months, `coverage`, `prepare-months`, the
training-coverage block in `stats_v2.json`, the month-list regridding helpers,
and the annual runbook.

- New tests: `tests/test_calendar_v2.py` (annual split boundaries, month
  schedule, overlap rejection) and, in `tests/test_pipeline_v2.py`, a 12-month
  monthly-statistics merge, coverage by missing source, and the predictor
  spot-check. The suite is **45 test functions**; the first four new tests were
  run in the session that added them, in a temporary CPU environment.
- Verified in this workspace, without a Python environment for the full suite:
  `bash -n` on every script; `py_compile` on the changed modules; the annual
  calendar replayed directly, giving 8,760 train / 1,440 val / 816 test hours,
  11,016 paired hours, 16 preparation months, both two-day gaps excluded and
  overlapping splits rejected; `coverage_v2` exercised against a stubbed
  manifest, producing valid JSON, correct per-month missing-source counts and
  one sampled entry per month with paired data; and the submission helper's
  prefixed `STAGE`/`RESUME`/`REGRESSION_CHECKPOINT` assignments confirmed to
  reach `sbatch --export=ALL` without leaking afterwards, using a stub `sbatch`.
- `scripts/repair_lowres_predictors_v2.py` was exercised against a stubbed
  manifest and gap list: report-only leaves files in place and exits nonzero,
  `--move-aside` renames only the incomplete outputs, and `--month` restricts
  the scan. It has no test in the suite and deletes nothing.
- Re-run `python -m pytest -q` in the project environment before relying on the
  combined 45-test count; it has not been executed in one workspace as a whole.

Not validated: the roughly 2.1 TiB the annual archive needs. The `coverage`
command reports the first two; the quota must be checked directly.

OMEGA500 was removed from every v2 preset after a scan of the regridded archive:
`repair_lowres_predictors_v2.py` found all 24 hours of January 8, 2025 lacking
only that variable, and preparation fails a whole month on any predictor gap.
The v2 dynamic set is now PRECTOT, U10M, V10M, TQV, QV2M and SLP, so the
condition tensor holds 30 channels instead of 31.

## Preparation is keyed on content, not on the calendar

Discover preparation showed the GEOS-FP source ending at 2026-03-08 23:30, which
shortened the test split to 816 hours after fifteen of sixteen months had already
been prepared. `preparation_signature` now covers only what stored arrays depend
on — predictors, roots, transforms, statistics stride and the surface bytes — and
`SCHEDULE_KEYS` (`start`, `end`, `splits`) are excluded, so a changed split
boundary reuses existing shards. Two checks replace the discarded hash equality:
finalization already compared each month's `entry_ids` against the current
manifest, and it now also requires each month's stored moments to cover exactly
that month's training hours under the current splits, backed by the existing
whole-archive count check in `finish_archive`.

Exercised directly against the real functions in this workspace (xarray and scipy
stubbed; neither is used by these code paths): a fresh record is written; a
split/end-only change keeps the signature stable and refreshes the stored record;
a changed predictor list is rejected; changed surface-file bytes are rejected; a
legacy record written before `static_sha256` was stored separately migrates
instead of failing; loose unversioned shards are still rejected. A legacy record
cannot be byte-checked against the surface file, which is stated in the code.
Two pytest cases cover the guard and a tampered month; neither has been run in a
full environment from here.


## Training throughput

The first annual regression job reached 2 epochs in 7.7 hours on one A100 while
its own benchmark measured 0.060 s per optimizer step, i.e. about 5 minutes of
GPU work per 512-step epoch. The loader was the bottleneck: `proposal()` read a
full precipitation field and ran a float64 double cumsum for nearly every sample,
because `DataLoader` workers are rebuilt each epoch and each kept a private
cache. Measured ~0.85 s per sample with two workers.

Changes: `scripts/build_proposal_cache_v2.py` precomputes the rain scores once per
(archive, patch size, sampling stride) and `PatchDatasetV2` memory-maps them;
`crop_v2` reads interior windows by slicing instead of fancy indexing, which
dominated the 576-pixel context view; `train.workers` 2 to 12 with
`--cpus-per-gpu=16`; batch 8 with `accumulate: 2` for an unchanged effective batch
of 16; `prefetch_factor` 4. Persistent workers were tried and rejected: epoch
patch locations come from `data.epoch` on the parent dataset, so persistent
workers would resample the first epoch's locations for the whole run.

Verified here against the real functions (torch, xarray and scipy stubbed; none
are used by these paths): `crop_v2` matches `crop` exactly on interior and
boundary-clipped windows; the cache loads as a memmap and is ignored when the
stride, grid or file is wrong; float32 storage costs 6e-8 relative error on a
sampling weight. Two pytest cases cover both. The end-to-end speedup has not been
measured on Discover — compare the first epoch's wall time against the 3.8 hours
above before trusting the fix.\n