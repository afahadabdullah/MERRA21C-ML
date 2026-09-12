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
  calendar replayed directly, giving 8,760 train / 1,440 val / 1,368 test hours,
  11,568 paired hours, 16 preparation months, both two-day gaps excluded and
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

Not validated: whether Discover actually holds paired inputs through March 31,
2026, whether the 2024-12 and 2026 months have been regridded with QV2M, SLP and
OMEGA500, and the roughly 2.3 TiB the annual archive needs. The `coverage`
command reports the first two; the quota must be checked directly.

