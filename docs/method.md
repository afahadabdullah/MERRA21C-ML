# Model and scientific choices

## Scope and variable mapping

The source information is GEOS-FP on a 0.25°×0.3125° grid. The existing bilinear regrid is conditioning preparation, not a resolution increase. The target is the HWT 1059×1799 LCC grid, nominally ~3 km. File names and roots come from the existing `data.md`, `plan.md`, and regrid scripts; no NASA files were accessible during local implementation.

| Output | LR collection/variable | HR collection/variable | Physical units |
|---|---|---|---|
| t2m | slv `T2M` | 30mn slv `TMP_2M` | K |
| precip | **native** flx `PRECTOT` | 30mn slv `PRECTOT` at the same :30 timestamp | mm/hour |
| ps | slv `PS` | 30mn slv `PRES_SFC` | Pa |
| wind10m | slv `hypot(U10M,V10M)` | 30mn slv `hypot(UGRD_10M,VGRD_10M)` | m/s |

Wind speed is scalar; this model does not generate wind direction or conserve momentum. The minimal dynamic predictor set is `PRECTOT`, `U10M`, `V10M`, and `TQV`. The four coarse baseline channels already supply `T2M`, native precipitation, `PS`, and wind speed; repeating temperature and pressure as dynamic channels adds no new field. U/V retain wind direction, while total-column water supplies moisture information not present in the baseline. `QV2M`, `SLP`, `OMEGA500`, `PRECCON`, and `PRECLSC` are excluded to reduce redundancy, missing-variable sensitivity, storage, and model width. This is a physically motivated minimal set, not a claim of measured feature importance; add predictors only through train/validation ablations. Static/temporal conditions are elevation, sine/cosine latitude and longitude, log area, annual phase accounting for leap years, UTC daily phase, and longitude-adjusted local solar phase. No HR dynamic predictors leak into conditioning.

## Flow formulation

Let `T` apply the precipitation/speed log transforms. The model learns the standardized transformed residual

```text
x1 = [T(HR_constrained) - T(coarse_baseline) - mu_residual] / sigma_residual
x0 ~ Normal(0, I)
t  ~ Uniform(0, 1)
xt = (1-t)*x0 + t*x1
loss = area-weighted mean |v_theta(xt, t, conditions) - (x1-x0)|²
```

This is independent-coupling conditional flow matching with straight conditional paths. It does **not** implement minibatch optimal-transport coupling, a DiT, or a separate deterministic regression pretrainer. The velocity model is a multiscale U-Net with GroupNorm, residual blocks, time-conditioned scale/shift, and concatenated spatial conditions. Avoiding global attention keeps the first implementation memory-efficient. The flow-matching objective follows [Lipman et al., Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) and the [Flow Matching Guide and Code](https://arxiv.org/abs/2412.06264).

At sampling, solve the ODE with Heun from a Gaussian initial field, decode the residual relative to the coarse baseline, then impose the physical precipitation constraint. Stochasticity comes from the initial field. Shared initial noise and overlapping context reduce tile disagreement but do not mathematically guarantee a globally coherent joint distribution. Inspect overlap seams and spectral artifacts; a patch model has limited context and does not guarantee long-range uncertainty coherence.

## Conservation definition and limits

Bilinear precipitation interpolation is not conservative. Budgets therefore come directly from native flux files, independently of the interpolated precipitation channel used for conditioning. Each LCC pixel is assigned to a native latitude/longitude cell by midpoint-defined native boundaries. Its actual `AREA` supplies the mass weight.

For group `g`, let `B_g = P_native,g * sum_i AREA_i`. Nonnegative proposed precipitation is scaled by `B_g / sum_i AREA_i*P_i`. When `B_g=0`, every HR pixel in that group becomes zero. When the proposal is completely dry and `B_g>0`, the native reference shape supplies a fallback. Projection after full-domain assembly avoids double-counted overlap budgets. The code validates every generated field to relative tolerance 2e-6; tests generally achieve much smaller errors.

The conserved quantity in the implementation has units mm·m²/hour. Dividing by 3600 gives kg/s because 1 mm of liquid-equivalent precipitation is 1 kg/m². The discrete represented-area budget is not the integral of the exact intersection of curvilinear HR polygons with native polygons. Implementing the latter requires trustworthy HR cell corners, an overlap matrix and a different constrained solve. At domain boundaries this implementation explicitly conserves the represented region, not the full outside-CONUS area of intersecting native cells.

Conservation of a biased source can harm comparison with original HR truth. The default projects HR training targets to the same budget and preserves original HR truth for independent evaluation. Native-dry/HR-wet disagreement is audited. Setting `conserve_training_precip: false` learns original HR targets but leaves the hard inference projection in place; this is a useful ablation, not a way to remove the source-budget incompatibility. Nonnegativity and log transforms do not themselves ensure mass conservation.

Related research demonstrates why physical constraints deserve explicit treatment: [Physically constrained generative adversarial networks for improving precipitation fields from Earth system models](https://www.nature.com/articles/s42256-022-00540-1). That paper is motivation, not a claim that this implementation reproduces its architecture or results.

## Temporal semantics

GEOS-FP `tavg1` labels are assumed to denote hourly midpoints. With `precip_source: hwt_30mn_slv_LCC.PRECTOT`, precipitation now uses the matched HR surface file at the same :30 timestamp and converts kg m-2 s-1 to mm/hour by multiplying by 3600, exactly as the legacy diagnostic does. There is no accumulated-APCP fallback. The previous assumption that hourly output meant one-hour accumulation was incorrect for the supplied experiment HISTORY (`ACCUMULATE`, `acc_interval: 1200000`). Legacy prepared archives/statistics must be rebuilt and models retrained.

All HR targets, including precipitation, use the midpoint surface snapshot. This approximates the corresponding LR hourly mean and is a declared representativeness mismatch, not an exact time-mean match. Prediction variables are labeled `time: point`; `lr_time_bounds` describes only the coarse conditioning window. Do not average arbitrary nearby snapshots and call it exact. If exact state averages become available, extend the manifest and unit-tested loader around their bounds.

## Configuration selection and evaluation

The supplied A100 presets are engineering starting points. No public paper determines the best hyperparameters for these exact paired archives. Select using **validation**, keeping held-out test untouched:

- Compare 128 and 256 cores, plus context/stride changes; measure peak VRAM and seconds per step.
- Compare precipitation log scales 0.1, 1, and 5 mm/hour; these require newly prepared train-only statistics.
- Compare 12, 24, and 48 Heun steps with fixed validation seeds to check integration convergence.
- Use 8 members for development, 16–32 for final uncertainty verification if feasible.
- Compare original versus budget-adjusted targets and report projection size, source-dry disagreements and extremes.
- Compare against the coarse baseline and, for a stronger experimental study, train a separate deterministic residual model.

The initial split is chronological with gaps. A single 2025 January–August training period omits part of the annual cycle. Prefer multiple training years and separate full-year validation/test once available; do not claim out-of-year/all-season generalization from the initial split. Random patches increase optimization samples but are not independent weather events. No patch is randomly split across train/validation/test.

Training statistics are unweighted per-pixel population moments on a configurable spatial subsample of **every training hour**. This is recorded in `stats.json`; these are not high-precision area-weighted climate estimates. Training loss and continuous evaluation scores use AREA weights. Rain quantiles and histogram densities are explicitly pixel-based. FSS uses unweighted neighborhood event fractions with AREA-weighted scoring and excludes incomplete border neighborhoods. Spectra are Hann-tapered radial diagnostics in cycles per LCC pixel, not exact physical isotropic spectra on a spatially varying map scale. Event-free FSS/CSI/POD/FAR are undefined (`null`), not silently reported as perfect.

The training implementation follows PyTorch's guidance for [mixed precision and accumulation](https://docs.pytorch.org/docs/main/notes/amp_examples.html) and [activation checkpointing](https://docs.pytorch.org/docs/stable/checkpoint). A100 BF16 avoids FP16 overflow/scaler sensitivity. CPU tests do not validate CUDA/DDP performance.
