# Validation record

## PRECTOT target correction (2026-09-11)

- **25 tests passed** on macOS ARM64 CPU with Python 3.12.14 and PyTorch 2.14.0 in an isolated temporary environment.
- Regression coverage verifies exact same-time HWT surface `PRECTOT × 3600` truth values, no dependency on accumulated APCP files, rejection of missing/rate-unit-invalid/wrong-time HR fields, and rejection of legacy partial/completed archives. Monthly resume/finalization, train-only normalization, checkpoint resume, inference and evaluation also pass.
- Full synthetic CLI smoke run passed: six prepared hours, two training epochs, three generated members, evaluation and diagnostic PNG generation. Maximum relative wet-footprint budget error was 1.40e-8.
- Both production configurations load successfully; Python compilation, modified Slurm test-script syntax and whitespace checks pass.
- The Discover source files were not available locally. These tests establish software behavior, not real-data meteorological accuracy. The checked-in workflow bitmap was not regenerated; its old APCP timeline is explicitly marked obsolete in the README.
- The revised five-member test diagnostic was exercised end to end on the synthetic archive. It wrote five provenance-hashed NetCDF members, a five-column full-domain figure, an event-centered zoom with zoom-region RMSE, an ensemble-mean RMSE summary, training history and machine-readable full/zoom metrics using nearest-neighbor `imshow` rasterization.

## Earlier baseline validation

Local execution on 2026-09-09, macOS ARM64 CPU, Python 3.12.14, PyTorch 2.14.0. Dependencies were installed in an isolated temporary environment; the project was installed successfully in editable mode.

## Checks performed

- Numerical/integration tests: **16 passed**. Coverage includes area-weighted precipitation budgets (wet, dry, zero-proposal fallback, nonuniform areas), transform round trips, native-cell indexing, CRPS against its pairwise definition, perfect/undefined weather scores, randomized dry ties, month-boundary pairing, disjoint splits, missing files and predictor-schema gaps, train-only normalization, deterministic patches, network backward propagation, Heun integration, irregular-domain overlap coverage, exact epoch-boundary resume, label-free preparation, unit/time validation, ensemble NetCDF output and evaluation.
- Full synthetic CLI run: six paired hours across an August/September boundary; 36×44 grid; 16×16 cores with 4-pixel halo; two training epochs; three members with two Heun steps; evaluation and diagnostic PNGs generated successfully.
- Maximum relative wet-footprint precipitation budget error across the three smoke members: **1.65×10⁻⁸**. Dry-group leakage was zero. This verifies the discrete footprint constraint, not polygon-overlap conservation.
- Editable package installation and console entry point, Python compilation, shell syntax and whitespace checks passed.
- Field, spectral and precipitation skill figures were visually inspected for layout, axes and colorbars. Smoke outputs are under `runs/smoke_release/` (ignored by Git).

The synthetic model is deliberately tiny and barely trained. Its meteorological scores are not evidence of learned downscaling skill. A100 BF16 behavior, GPU peak memory, multi-GPU NCCL/DDP execution, Discover scheduler directives, production data availability/metadata and full-year throughput were not executed locally. Use `scripts/benchmark_a100.py` inside an allocation to measure memory before production.

The CPU resume test checks bitwise equality after resuming at an epoch boundary; cross-device or changed-world-size bitwise reproducibility is not claimed. The distributed implementation initializes EMA after DDP has broadcast the rank-zero model, so every rank starts with the same EMA weights.
