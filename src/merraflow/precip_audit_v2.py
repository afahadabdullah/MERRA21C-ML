"""CPU-only audit of saved v2 rainfall: amounts, calibration and spatial errors.

Reads prediction files without importing Torch or changing trained models.
All precipitation amounts below are rates, not accumulated storm totals.
"""
from pathlib import Path
import csv
import json
import re

import numpy as np
import xarray as xr

from .config import write_json
from .metrics import continuous, precipitation, radial_psd, rank_histogram, weighted_mean


THRESHOLDS = (.1, 1., 5., 10., 25.)
FACTORS = (1, 4, 8, 16, 32)
IDENTITY_KEYS = ('checkpoint_sha256', 'regression_sha256', 'checkpoint_epoch',
                 'ode_steps', 'blend', 'target_alignment', 'conservation')


def validate_fields(ensemble, regression, baseline, truth, area):
    if area.ndim != 2 or min(area.shape) < 4 or not np.isfinite(area).all() or np.any(area <= 0):
        raise ValueError('Require a finite positive 2D area grid, at least 4 by 4')
    if ensemble.ndim != 3 or ensemble.shape[0] < 2 or ensemble.shape[1:] != area.shape:
        raise ValueError('Require at least two rainfall members on the area grid')
    for name, values in [('members', ensemble), ('regression', regression), ('baseline', baseline), ('HWT', truth)]:
        if (name != 'members' and values.shape != area.shape) or not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError(f'{name}: require finite nonnegative rainfall on the area grid')


def deterministic_scores(field, truth, area):
    diff = np.asarray(field, dtype='float64')-truth
    return {'rmse': float(np.sqrt(weighted_mean(diff*diff, area))),
            'mae': weighted_mean(abs(diff), area), 'bias': weighted_mean(diff, area)}


def area_coarsen(field, area, factor):
    """Area-weighted block means, retaining partial edge blocks and all area."""
    if factor < 1:
        raise ValueError('Coarsening factor must be positive')
    yy, xx = np.arange(0, area.shape[0], factor), np.arange(0, area.shape[1], factor)
    def total(x):
        return np.add.reduceat(np.add.reduceat(x, yy, axis=-2), xx, axis=-1)
    weights = total(np.asarray(area, dtype='float64'))
    return total(np.asarray(field, dtype='float64')*area)/weights, weights


def rainfall_profile(field, area):
    values, weights = field.ravel(), area.ravel()
    order = np.argsort(values)
    ordered, mass = values[order], np.cumsum(weights[order], dtype='float64')
    probabilities = np.array([.5, .9, .95, .99, .999])
    quantiles = ordered[np.minimum(np.searchsorted(mass, probabilities*mass[-1]), len(ordered)-1)]
    levels = np.geomspace(.01, max(100., float(ordered[-1])), 70)
    indices = np.searchsorted(ordered, levels, side='left')
    below = np.r_[0., mass][indices]
    mean = weighted_mean(field, area)
    return {'mean_mm_h': mean, 'volume_rate_m3_s': mean*float(area.sum())/3.6e6,
            'max_mm_h': float(ordered[-1]),
            'area_quantiles_mm_h': dict(zip(('p50', 'p90', 'p95', 'p99', 'p999'), map(float, quantiles))),
            'wet_area_fraction': {str(t): weighted_mean(field >= t, area) for t in THRESHOLDS},
            'exceedance': {'threshold_mm_h': levels.tolist(), 'area_fraction': (1-below/mass[-1]).tolist()}}


def tile_regions(shape, size, stride):
    """Coverage inferred from config, not proof of the original sampler geometry."""
    if size > min(shape) or not 0 < stride <= size:
        raise ValueError('Invalid tile geometry for archive shape')
    counts = np.zeros(shape, dtype='int16')
    axes = [list(range(0, n-size+1, stride)) for n in shape]
    for axis, n in zip(axes, shape):
        if axis[-1] != n-size:
            axis.append(n-size)
    for y in axes[0]:
        for x in axes[1]:
            counts[y:y+size, x:x+size] += 1
    return {'single_tile': counts == 1, 'overlapping_tiles': counts > 1}


def lag_scan(prediction, truth, area, radius=3):
    """Compare offsets on the same interior, without wraparound or realignment."""
    h, w = truth.shape
    if min(h, w) <= 2*radius+2:
        return None
    target, weights = truth[radius:h-radius, radius:w-radius], area[radius:h-radius, radius:w-radius]
    candidates = []
    for dy in range(-radius, radius+1):
        for dx in range(-radius, radius+1):
            sample = prediction[radius+dy:h-radius+dy, radius+dx:w-radius+dx]
            candidates.append({'prediction_offset_y': dy, 'prediction_offset_x': dx,
                               **deterministic_scores(sample, target, weights)})
    # Prefer zero displacement in ties, including completely dry fields.
    best = min(candidates, key=lambda r: (r['rmse'], abs(r['prediction_offset_y'])+abs(r['prediction_offset_x'])))
    zero = next(r for r in candidates if r['prediction_offset_x'] == r['prediction_offset_y'] == 0)
    return {'unshifted_rmse': zero['rmse'], 'best': best, 'radius_blocks': radius,
            'interpretation': 'Prediction sampled at HWT coordinate plus offset; diagnostic only, common interior'}


def audit_case(ensemble, regression, baseline, truth, area, patch, precip_scale):
    validate_fields(ensemble, regression, baseline, truth, area)
    if not np.isfinite(precip_scale) or precip_scale <= 0:
        raise ValueError('Require positive finite precipitation transform scale')
    fields = {'HWT': truth, 'coarse': baseline, 'regression': regression, 'flow_mean': ensemble.mean(0)}
    fields.update({f'member_{i:03d}': value for i, value in enumerate(ensemble)})
    profiles = {name: rainfall_profile(value, area) for name, value in fields.items()}
    reference_amount = profiles['HWT']['mean_mm_h']
    for profile in profiles.values():
        profile['mean_ratio_to_HWT'] = profile['mean_mm_h']/reference_amount if reference_amount > 0 else None
    scores = {name: deterministic_scores(value, truth, area) for name, value in fields.items() if name != 'HWT'}
    probabilistic = continuous(ensemble, truth, area)
    masks = {'HWT_wet_ge_0.1': truth >= .1, 'HWT_dry_lt_0.1': truth < .1,
             **tile_regions(area.shape, patch['size'], patch['stride'])}
    regional = {}
    for name, mask in masks.items():
        regional[name] = {'area_fraction': weighted_mean(mask, area), 'scores': {
            key: deterministic_scores(value[mask], truth[mask], area[mask])
            for key, value in fields.items() if key != 'HWT'} if mask.any() else None}
    coarsened = {}
    for factor in FACTORS:
        if factor > min(area.shape):
            continue
        target, weights = area_coarsen(truth, area, factor)
        coarsened[str(factor)] = {name: deterministic_scores(area_coarsen(value, area, factor)[0], target, weights)
                                 for name, value in fields.items() if name != 'HWT'}
    spectra = {}
    for name, value in fields.items():
        freq, power = radial_psd(value)
        spectra[name] = power.tolist()
    spectra['member_average'] = np.mean([spectra[f'member_{i:03d}'] for i in range(len(ensemble))], axis=0).tolist()
    spectra['cycles_per_pixel'] = freq.tolist()
    rain_skill = {name: precipitation(values, truth, area, scales=(1, 5, 17, 33)) for name, values in
                  [('flow', ensemble), ('regression', regression[None]), ('coarse', baseline[None])]}
    target, weights = area_coarsen(truth, area, 8)
    displacement = {name: lag_scan(area_coarsen(fields[name], area, 8)[0], target, weights)
                    for name in ('coarse', 'regression', 'flow_mean')}
    # Output-only experiment: clipping at decode prevents reconstructing latent
    # negative values. This is not equivalent to resampling or changing flow_scale.
    log_members = np.log1p(ensemble/precip_scale)
    log_regression = np.log1p(regression/precip_scale)
    amplitude = {}
    for alpha in (0., .25, .5, .75, 1.):
        values = np.expm1(log_regression+alpha*(log_members-log_regression))*precip_scale
        amplitude[str(alpha)] = {'scores': continuous(values, truth, area),
                                'mean_mm_h': weighted_mean(values.mean(0), area)}
    decoded_log_mean = np.expm1(log_members.mean(0))*precip_scale
    gap = ensemble.mean(0)-decoded_log_mean
    return {'members': len(ensemble), 'profiles': profiles, 'scores': scores,
            'probabilistic': probabilistic, 'rank_histogram': rank_histogram(ensemble, truth, area),
            'regional': regional, 'coarsened': coarsened, 'spectra': spectra,
            'precipitation_skill': rain_skill, 'displacement_on_8_pixel_blocks': displacement,
            'postdecode_log_residual_shrinkage': amplitude,
            'jensen_gap': {'mean_mm_h': weighted_mean(gap, area),
                           'note': 'Arithmetic member mean minus decoded mean log rainfall; expected from convexity, not proof of excess variance'}}


def load_members(paths, entry, fingerprint, static, expected_members=None):
    """Load only rainfall; require common provenance, coordinates and regression."""
    if len(paths) < 2 or (expected_members is not None and len(paths) != expected_members):
        raise ValueError(f'{entry["id"]}: incomplete ensemble or fewer than two members')
    members, regression, identity, seeds = [], None, None, []
    for member, path in enumerate(paths):
        with xr.open_dataset(path) as ds:
            if (ds.attrs.get('version') != 'v2' or ds.attrs.get('stage') != 'flow'
                    or ds.attrs.get('dataset_fingerprint') != fingerprint
                    or ds.attrs.get('split') != entry['split'] or ds.attrs.get('ensemble_member') != member
                    or ds.sizes.get('time') != 1 or ds.time.values[0] != np.datetime64(entry['time'])):
                raise ValueError(f'{path}: incompatible v2 member/archive/time/split')
            if any(key not in ds.attrs for key in (*IDENTITY_KEYS, 'seed')):
                raise ValueError(f'{path}: missing sampler/checkpoint provenance')
            key = {name: ds.attrs[name].item() if isinstance(ds.attrs[name], np.generic) else ds.attrs[name]
                   for name in IDENTITY_KEYS}
            if identity is not None and key != identity:
                raise ValueError(f'{path}: mixed checkpoint or sampler settings')
            identity = key
            seeds.append(int(ds.attrs['seed']))
            for name in ('lat', 'lon'):
                if name not in ds or ds[name].shape != static[name].shape or not np.allclose(ds[name], static[name], rtol=0, atol=1e-5):
                    raise ValueError(f'{path}: {name} grid differs from archive')
            arrays = []
            for name in ('precip', 'regression_precip'):
                if name not in ds or ds[name].attrs.get('units') != 'mm h-1' or ds[name].dims != ('time', 'Ydim', 'Xdim'):
                    raise ValueError(f'{path}: invalid rainfall dimensions or units')
                arrays.append(np.asarray(ds[name].values[0], dtype='float32'))
            if regression is not None and not np.array_equal(regression, arrays[1]):
                raise ValueError(f'{path}: regression changed between members')
            regression = arrays[1]
            members.append(arrays[0])
    if len(set(seeds)) != len(seeds):
        raise ValueError('Repeated member seeds')
    return np.stack(members), regression, identity, seeds


def run_audit(cfg, predictions, output=None, split='test', timestamps=None, members=None, plots=True):
    if split not in ('val', 'test'):
        raise ValueError('Audit requires val or test')
    root, archive_root = Path(predictions), Path(cfg['data']['prepared'])
    index = json.loads((archive_root/'index_v2.json').read_text())
    stats = json.loads((archive_root/'stats_v2.json').read_text())
    if index.get('format') != 'v2':
        raise ValueError('Require a v2 archive')
    with np.load(archive_root/'static_v2.npz') as source:
        static = {name: source[name] for name in ('area', 'lat', 'lon')}
    available = {e['id']: e for e in index['entries'] if e['split'] == split}
    entries = [e for e in available.values() if list(root.glob(f'{e["id"]}_m*_v2.nc'))]
    if timestamps:
        entries = []
        for stamp in timestamps:
            matches = [e for e in available.values() if stamp in (e['id'], e['time'])]
            if len(matches) != 1:
                raise ValueError(f'No unique {split} timestamp: {stamp}')
            entries.append(matches[0])
        if len({e['id'] for e in entries}) != len(entries):
            raise ValueError('Repeated audit timestamps')
    if not entries:
        raise ValueError(f'No saved {split} predictions in {root}')
    out = Path(output or root.parent/'precip_audit_v2')
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'{out} is not empty; choose a fresh --output')
    out.mkdir(parents=True, exist_ok=True)
    reports, identity, count = [], None, members
    for entry in entries:
        paths = list(root.glob(f'{entry["id"]}_m*_v2.nc'))
        def number(path):
            match = re.fullmatch(re.escape(entry['id'])+r'_m(\d+)_v2.nc', path.name)
            if not match:
                raise ValueError(f'Invalid member filename: {path}')
            return int(match[1])
        paths.sort(key=number)
        if [number(p) for p in paths] != list(range(len(paths))):
            raise ValueError(f'{entry["id"]}: missing or repeated member numbers')
        ensemble, regression, key, seeds = load_members(paths, entry, index['fingerprint'], static, count)
        if identity is not None and key != identity:
            raise ValueError('Mixed checkpoint or sampler settings across timestamps')
        identity, count = key, len(ensemble)
        truth = np.asarray(np.load(archive_root/entry['id']/'truth_v2.npy', mmap_mode='r')[1])
        baseline = np.asarray(np.load(archive_root/entry['id']/'baseline_v2.npy', mmap_mode='r')[1])
        print(f'Auditing {entry["id"]}: {count} members', flush=True)
        report = audit_case(ensemble, regression, baseline, truth, static['area'], cfg['patch'], stats['precip_log_scale'])
        report.update(id=entry['id'], time=entry['time'], split=split, seeds=seeds,
                      prediction_files=[str(p.resolve()) for p in paths])
        write_json(out/f'{entry["id"]}_audit_v2.json', report)
        if plots:
            from .precip_plots_v2 import plot_case
            plot_case(out, report, ensemble, regression, baseline, truth)
        reports.append(report)
    selection_path = root.parent/'metrics_v2.json'
    selection = None
    if selection_path.is_file():
        previous = json.loads(selection_path.read_text())
        if previous.get('checkpoint_sha256') == identity['checkpoint_sha256'] and previous.get('split') == split:
            selection = previous.get('selection')
    # Recalculate each hour first; these are means of hour-level metrics, not
    # pooled spatial RMSE. Event case and random cases remain separately visible.
    summary = {'split': split, 'hours_audited': len(reports), 'members': count,
               'archive_fingerprint': index['fingerprint'], 'identity': identity,
               'config_patch_assumption': cfg['patch'], 'selection': selection,
               'timestamps': [r['id'] for r in reports], 'other_split_hours_not_audited': len(available)-len(reports),
               'aggregation': 'Equal-weight mean of per-hour scores; not pooled RMSE or full-test skill',
               'limits': ['Prepared HWT snapshots versus coarse hourly means; this audit cannot verify raw averaging windows.',
                          'Tile regions use supplied config; old predictions do not store tile geometry.',
                          'Offset and amplitude experiments diagnose existing outputs, not new model integrations.',
                          'Rank/coverage estimates with a small ensemble have limited resolution.'],
               'mean_hour_scores': {name: {metric: float(np.mean([r['scores'][name][metric] for r in reports]))
                                           for metric in ('rmse', 'mae', 'bias')}
                                    for name in ('coarse', 'regression', 'flow_mean')}}
    write_json(out/'summary_v2.json', summary)
    with open(out/'rainfall_scores_v2.csv', 'w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['id', 'field', 'rmse_mm_h', 'bias_mm_h', 'mean_mm_h', 'amount_ratio_to_HWT',
                         'wet_fraction_ge_0.1', 'p99_mm_h', 'p999_mm_h', 'max_mm_h'])
        for report in reports:
            for name, profile in report['profiles'].items():
                score = report['scores'].get(name, {})
                writer.writerow([report['id'], name, score.get('rmse', 0), score.get('bias', 0),
                                 profile['mean_mm_h'], profile['mean_ratio_to_HWT'], profile['wet_area_fraction']['0.1'],
                                 profile['area_quantiles_mm_h']['p99'], profile['area_quantiles_mm_h']['p999'], profile['max_mm_h']])
    lines = ['# V2 precipitation audit', '', f'{len(reports)} {split} hours; {count} members per hour.', '',
             'Amounts are precipitation rates, not accumulated storm totals.', '',
             '| Timestamp | Coarse RMSE | Regression RMSE | Flow-mean RMSE | Flow/HWT amount |',
             '|---|---:|---:|---:|---:|']
    for r in reports:
        ratio = r['profiles']['flow_mean']['mean_ratio_to_HWT']
        ratio_text = f'{ratio:.3f}' if ratio is not None else 'undefined (dry HWT)'
        lines.append(f'| {r["id"]} | {r["scores"]["coarse"]["rmse"]:.3f} | {r["scores"]["regression"]["rmse"]:.3f} | {r["scores"]["flow_mean"]["rmse"]:.3f} | {ratio_text} |')
    lines += ['', 'RMSE is in mm h-1. Inspect the individual members as well as their mean.', '',
              '## How to interpret the plots', '',
              '- Excess wet area or upper quantiles indicates a distribution mismatch; compare amounts and spatial errors too.',
              '- RMSE that persists after block averaging indicates errors beyond the smallest scales.',
              '- A lower error after an offset suggests displacement; it does not justify shifting forecasts after seeing HWT.',
              '- Overlap versus single-tile error differences also depend on where storms fall; they do not establish a stitching bug.',
              '- The log-residual shrinkage sweep is a post-decoding diagnostic. Do not choose a production amplitude on test cases.',
              '- Maps use a common linear scale covering the maximum across all displayed fields; extreme values are not clipped.', '',
              '## Limits', '', *[f'- {note}' for note in summary['limits']]]
    if selection:
        lines += ['', f'Original case selection: {selection}.']
    (out/'report_v2.md').write_text('\n'.join(lines)+'\n')
    return out
