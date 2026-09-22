"""Physical rain realism and skill; never infer meteorological success from smoke tests."""
from collections import defaultdict
from pathlib import Path
import hashlib
import json
import numpy as np
import xarray as xr
from .dataset_v3_precip import PrecipArchive
from .metrics import continuous, precipitation, rank_histogram, radial_psd, weighted_mean
from .physics import budget_error


def field_metrics(ensemble, truth, area):
    frequencies, truth_psd = radial_psd(truth)
    return dict(continuous=continuous(ensemble, truth, area), precipitation=precipitation(ensemble, truth, area),
                rank_histogram=rank_histogram(ensemble, truth, area),
                wet_fraction_truth=weighted_mean(truth >= .1, area),
                wet_fraction_members=float(np.mean([weighted_mean(x >= .1, area) for x in ensemble])),
                max_truth=float(truth.max()), max_members=[float(x.max()) for x in ensemble],
                psd=dict(cycles_per_pixel=frequencies.tolist(), truth=truth_psd.tolist(),
                         members_mean=np.mean([radial_psd(x)[1] for x in ensemble], axis=0).tolist()))


def paired_day_interval(rows, baseline):
    days = defaultdict(list)
    for row in rows:
        days[row['time'][:10]].append(row[baseline]['continuous']['crps']-row['diffusion']['continuous']['crps'])
    values = np.array([np.mean(x) for x in days.values()])
    result = dict(days=len(values), mean_daily_crps_improvement=float(values.mean()), interval_95=None)
    if len(values) >= 10:
        rng = np.random.default_rng(891)
        # Resample days, not strongly dependent grid pixels. Multi-day storms
        # can still violate independence; this interval is descriptive only.
        means = [rng.choice(values, len(values)).mean() for _ in range(2000)]
        result['interval_95'] = np.quantile(means, [.025, .975]).tolist()
    return result


def evaluate(cfg, split='val'):
    if split not in ('val', 'test'):
        raise ValueError('Scientific evaluation requires val or test')
    archive = PrecipArchive(cfg)
    out = Path(cfg['inference']['output'])
    manifest = json.loads((out/'manifest_v3_precip.json').read_text())
    expected_inference = {k: v for k, v in cfg['inference'].items() if k != 'output'}
    if (manifest['fingerprint'] != archive.index['fingerprint'] or manifest['inference'] != expected_inference
            or manifest['history_hours'] != archive.lags or manifest['rain_scale_mm_h'] != archive.scale
            or manifest.get('target_kind') != archive.target_kind
            or manifest.get('hourly_fingerprint') != archive.hourly_fingerprint):
        raise ValueError('Evaluation configuration differs from prediction manifest')
    signature = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    rows, missing = [], []
    area = archive.static['area']
    eligible = archive.eligible(split)
    for entry in eligible:
        paths = [out/f'{entry["id"]}_m{i:03d}_v3_precip.nc' for i in range(cfg['inference']['members'])]
        if not any(p.exists() for p in paths):
            missing.append(entry['time'])
            continue
        if not all(p.exists() for p in paths):
            raise ValueError(f'Incomplete ensemble at {entry["time"]}')
        members, regression = [], None
        for i, path in enumerate(paths):
            with xr.open_dataset(path) as ds:
                if (ds.attrs['signature'] != signature or ds.attrs['split'] != split or ds.attrs['ensemble_member'] != i
                        or ds.time.values[0] != np.datetime64(entry['time'])):
                    raise ValueError(f'Mixed prediction provenance: {path}')
                value = ds.precip.values[0]
                current_regression = ds.regression_precip.values[0]
                if (value.shape != archive.shape or not np.isfinite(value).all() or np.any(value < 0)
                        or not np.isfinite(current_regression).all() or np.any(current_regression < 0)):
                    raise ValueError(f'Invalid rain prediction: {path}')
                if regression is not None and not np.array_equal(regression, current_regression):
                    raise ValueError('Regression baseline changed between members')
                regression = current_regression
                members.append(value)
        ensemble = np.stack(members)
        truth = np.asarray(archive.truth_field(entry)[0])
        coarse = np.asarray(archive.array(entry, 'baseline')[1])
        row = dict(time=entry['time'], diffusion=field_metrics(ensemble, truth, area),
                   regression=field_metrics(regression[None], truth, area), coarse=field_metrics(coarse[None], truth, area),
                   native_budget_members=[budget_error(member, archive.array(entry, 'native_reference')[0], area, archive.static['groups']) for member in ensemble])
        rows.append(row)
        if len(rows) == 1:
            preview = (truth, coarse, regression, ensemble)
    if not rows:
        raise ValueError('No complete prediction ensembles')
    summary = dict(version='v3_precip', split=split, target_kind=archive.target_kind, stage=manifest['stage'],
                   synthetic=bool(archive.index['data_config'].get('synthetic', False)),
                   evaluated_hours=len(rows), eligible_hours=len(eligible), missing_hours=missing,
                   excluded_history_hours=sum(e['split'] == split for e in archive.index['entries'])-len(eligible),
                   metrics={name: {key: float(np.mean([row[name]['continuous'][key] for row in rows]))
                                   for key in ('crps', 'rmse', 'mae', 'bias', 'spread', 'coverage_90')}
                            for name in ('diffusion', 'regression', 'coarse')},
                   paired_daily_comparisons={name: paired_day_interval(rows, name) for name in ('regression', 'coarse')},
                   interpretation='No automatic realism pass. Inspect reliability, extremes, member spectra, FSS and budgets. '
                                  'Independent hourly members do not establish temporal coherence or multi-hour accumulations.')
    report = out/('evaluation_'+split+'_v3_precip')
    report.mkdir(exist_ok=True)
    for name, value in [('per_hour', rows), ('summary', summary)]:
        (report/(name+'.json')).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    plot_preview(preview, report, rows[0]['time'], archive.target_kind)
    return report/'summary.json'


def plot_preview(fields, report, time, target_kind='midpoint_rate'):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    truth, coarse, regression, ensemble = fields
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    vmax = max(float(np.quantile(truth, .995)), float(np.quantile(ensemble, .995)), 1.)
    for ax, title, field in zip(axes.flat, [f'HWT truth ({target_kind})', 'Coarse baseline', 'Regression', 'Member 1', 'Ensemble mean', 'Ensemble spread'],
                                [truth, coarse, regression, ensemble[0], ensemble.mean(0), ensemble.std(0)]):
        artist = ax.imshow(field, origin='lower', vmin=0, vmax=vmax, cmap='Blues')
        ax.set_title(title)
    fig.colorbar(artist, ax=axes.ravel().tolist(), label='Rain rate (mm/h); 99.5th-percentile color scale')
    fig.suptitle(time+' — spatial rain-rate experiment ('+target_kind+')')
    fig.savefig(report/'precip_comparison.png', dpi=150)
    plt.close(fig)
