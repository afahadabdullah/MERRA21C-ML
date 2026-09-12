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

Not validated locally: Discover source FROCEAN location/availability, CUDA/BF16/DDP
execution, A100 memory/cost, or improvement on actual held-out HWT events. The
included preflight, benchmark and validation commands support those next checks.
No remote training job was inspected, modified, stopped or submitted.
