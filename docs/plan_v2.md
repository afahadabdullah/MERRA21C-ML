# MERRAflow v2 implementation plan

V1 source, configs, prepared data, checkpoints, and running jobs remain unchanged.
Every new maintained file and every v2 output artifact uses a `v2` name. V2 runs
through `python -m merraflow.cli_v2`, without changing the installed v1 entry point.

1. Prepare original HWT PRECTOT, T2M, pressure, and signed U/V targets. Never
   project precipitation during preparation or prediction. Retain budget audits.
   Support calendar-derived monthly arrays across years. The annual preset fits
   normalization on December 2024–November 2025 training hours and records monthly
   contributions; later validation/test never contribute to normalization.
2. Read coordinate-validated FROCEAN and required FRLAKE from one GSHHG-derived
   static file on the HWT grid; derive land as one minus ocean minus lake.
   Include elevation, terrain slopes, signed distance to water, coordinates and area.
   Add QV2M and SLP to the available coarse predictor set; OMEGA500 is excluded
   because the hourly GEOS-FP stream does not carry it for every hour.
3. Train a deterministic multiscale conditional U-Net, freeze its best EMA weights,
   calibrate residual scales on training patches, then train a residual flow U-Net.
   Use two residual blocks per scale, bottleneck self/cross attention and a wider
   low-resolution context view alongside the native-resolution local patch.
4. Use area-weighted quadratic value, gradient and multiscale losses for regression;
   retain velocity matching as the main generative objective with a small quadratic
   endpoint-gradient term. Do not force single stochastic members to reproduce the
   exact target spectrum or rain-cell phase. Diagnose spectra on generated members.
5. Sample a mixture of uniform and rain/coast-rich patch locations; carry exact
   inverse-proposal weights in the loss. Validation stays uniform with fixed seeds.
6. Stitch with shared noise and wide halos; export mean-stage fields, members,
   physical units, checkpoint identities, no-conservation metadata, and diagnostics.
7. Verify original-target fidelity, mask alignment, patch weights, attention/loss
   gradients, both training stages, exact resume, inference, evaluation and v1 tests.

Smaller patches are an ablation, not the default remedy: preserve the same broad
context and compare equal optimizer steps. More seasons/years require real source
data; never fabricate training coverage or mix held-out events into training.

Reference: [CorrDiff](https://www.nature.com/articles/s43247-025-02042-5) motivates
the regression/generative decomposition; this implementation uses flow matching,
not CorrDiff's EDM objective. [PyTorch SDPA](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention)
implements bottleneck attention. These choices require empirical validation.
