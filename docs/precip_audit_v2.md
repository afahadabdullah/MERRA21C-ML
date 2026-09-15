# Audit saved v2 precipitation

This CPU job reads the existing five-case diagnostic members. It does not train,
generate new members, modify checkpoints, or submit another job. The code needs
NumPy, SciPy, xarray, h5netcdf, PyYAML and Matplotlib; the saved-field audit does
not import Torch. Use the existing project conda environment on Discover.

From the Discover login node, enter `bash` first if your prompt uses tcsh, then:

```bash
cd /gpfsm/dnb10/projects/p311/ML_downscaling
git pull --ff-only
mkdir -p logs_v2
env -u SLURM_MEM_PER_CPU -u SLURM_MEM_PER_NODE -u SLURM_MEM_PER_GPU \
  CONFIG=configs/discover_annual_v2.yaml \
  PREDICTIONS=runs/merraflow_annual_v2/best_flow_feb23_five_cases_v2/predictions_v2 \
  OUTPUT=runs/merraflow_annual_v2/best_flow_feb23_five_cases_v2/precip_audit_v2 \
  SPLIT=test MEMBERS=5 \
  sbatch --export=ALL scripts/slurm_audit_precip_v2.sh
```

It requests one CPU node, four CPUs and 32 GB for up to four hours with the
project's `allnccs` QoS. No GPU is requested. Logs are
`logs_v2/rain_audit_<job-ID>_v2.log` and `.err`.
For a rerun choose a fresh `OUTPUT` directory. To restrict the audit to the
storm case, add `TIMESTAMPS=20260223_0530` to the `env` arguments. Without that
argument every available prediction hour in the requested split is audited.

The corresponding direct command in an activated environment is:

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python scripts/audit_precip_v2.py \
  --config configs/discover_annual_v2.yaml \
  --predictions runs/merraflow_annual_v2/best_flow_feb23_five_cases_v2/predictions_v2 \
  --output runs/merraflow_annual_v2/best_flow_feb23_five_cases_v2/precip_audit_v2 \
  --split test --members 5
```

## Outputs

- `report_v2.md`: per-case summary and interpretation guide.
- `rainfall_scores_v2.csv`: HWT, coarse, regression, each flow member and the
  flow mean; physical-unit RMSE/bias, mean rate, amount ratio, wet fraction,
  area-weighted upper quantiles and maximum.
- `summary_v2.json`: checkpoint/sampler identity, case selection and mean
  per-hour deterministic scores. This is not a pooled RMSE or full-test score.
- `<timestamp>_audit_v2.json`: all profiles, metrics, conditional wet/dry scores,
  inferred tile-region scores, displacement and amplitude experiments.
- `<timestamp>_members_v2.png`: all rainfall members, HWT and deterministic
  baselines on one linear scale extending to the actual largest value.
- `<timestamp>_errors_v2.png`: regression and ensemble-mean errors plus spread.
- `<timestamp>_distribution_structure_v2.png`: exceedance curves, quantiles,
  precipitation amount, block-averaged errors, spectra and rank histogram.
- `<timestamp>_skill_calibration_v2.png`: neighborhood fractions skill scores,
  probability reliability, wet-area fractions and log-residual shrinkage sweep.

## Questions this audit answers

1. **Too much or too little precipitation?** Compare area-weighted mean rate,
   wet-area fractions at 0.1, 1, 5, 10 and 25 mm/h, and upper quantiles of every
   member against HWT. The volume output is a volume **rate** in m³/s. These
   fields cannot establish accumulated storm precipitation from one snapshot.
2. **Wrong only at fine scales, or wrong at storm scales too?** Errors are
   recalculated after area-weighted block averaging by 1, 4, 8, 16 and 32 grid
   pixels. Partial edge blocks retain all area and precipitation. Axes use
   grid pixels; no constant physical spacing is assumed for the projected grid.
3. **Could displacement explain part of the error?** On 8-pixel block means,
   a ±3-block offset scan compares every displacement over the same interior.
   It never wraps across domain boundaries and never changes saved forecasts.
   An error reduction suggests displacement but is not a forecast correction.
4. **Does rainfall have plausible variability?** Compare individual-member
   spectra and distributions, ensemble CRPS, rank frequencies, spread and
   reliability. Spectra are Hann-tapered radial spectra in grid-index space.
   Five-member calibration curves are noisy and have coarse probability steps.
5. **Does the output suggest a tiling issue?** RMSE/bias in overlapping versus
   single-tile regions are reported for coarse, regression and each flow field.
   These regions are inferred from the supplied config because existing NetCDFs
   omit tile geometry. Weather differs between those regions, so an error
   difference alone cannot establish a sampler defect.
6. **Is the rainfall residual amplitude suspect?** The output-only experiment
   uses `log1p(Pα/s) = log1p(Preg/s) + α[log1p(Pmember/s)-log1p(Preg/s)]`
   for α = 0, 0.25, 0.5, 0.75 and 1, with the archive's precipitation scale s.
   This is not a new ODE integration or an exact reconstruction of the latent
   residual: the decoder already clipped negative latent precipitation.
   The reported Jensen gap is an expected effect of exponentiation, not proof
   of excessive noise. Select any eventual calibration on validation cases.

## Integrity and limits

The loader checks archive fingerprint, grid coordinates, units, timestamps,
split, flow stage, member indices/counts, distinct seeds, common checkpoint and
sampler settings, and identical regression fields across members. All rainfall
must be finite and nonnegative. Use `MEMBERS=5` to require all expected members;
without an explicit count, a common count of at least two is inferred.

The source diagnostic's case-selection metadata is copied when its checkpoint
and split match. The February 23 hour was selected for high HWT precipitation,
so these five cases describe an event-focused sample. The audit does not infer
the cause of failure automatically or choose a model setting on these test cases.

This audit uses the prepared archive. It does not establish whether original
HWT snapshots and coarse hourly averages represent the same averaging window.
Comparing ODE step counts, synchronized tile evolution or other inference
settings requires generating additional predictions in separate directories.
Run this audit separately on each such directory; mixed runs are rejected.

## Local verification

Validated with 13 CPU tests in a temporary Python 3.12 environment: saved
NetCDF-to-JSON/CSV/Markdown/PNG workflow, unchanged input bytes, incomplete and
mixed-member rejection, area-weighted partial-block conservation, known spatial
offset recovery, dry cases, and batch argument/memory-environment handling.
The four synthetic plot layouts were visually inspected. Shell syntax and
Python compilation checks passed. This verifies the audit software; the actual
Discover checkpoint and storm members were not available locally.
