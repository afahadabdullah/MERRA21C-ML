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
from .noise_v2 import saved_noise_padding_v2


THRESHOLDS = (.1, 1., 5., 10., 25.)
FACTORS = (1, 4, 8, 16, 32)
IDENTITY_KEYS = ('checkpoint_sha256', 'regression_sha256', 'checkpoint_epoch',
                 'ode_steps', 'blend', 'target_alignment', 'conservation', 'noise_padding')


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


def log_space_audit(fields, area, scale):
    """Compare decoded log corrections, including the correction HWT needs."""
    log_truth = np.log1p(np.asarray(fields['HWT'], dtype='float64')/scale)
    log_regression = np.log1p(np.asarray(fields['regression'], dtype='float64')/scale)
    records = {}
    for name, field in fields.items():
        log_field = np.log1p(np.asarray(field, dtype='float64')/scale)
        correction = log_field-log_regression
        peak = np.unravel_index(np.argmax(field), field.shape)
        records[name] = {
            'scores_vs_HWT_log_space': deterministic_scores(log_field, log_truth, area),
            'correction_from_regression': {
                'mean': weighted_mean(correction, area),
                'rms': float(np.sqrt(weighted_mean(correction**2, area))),
                'min': float(correction.min()), 'max': float(correction.max()),
                'area_fraction_ge': {str(t): weighted_mean(correction >= t, area) for t in (1., 2., 3.)}},
            'max_rainfall_pixel': {
                'y': int(peak[0]), 'x': int(peak[1]), 'rain_mm_h': float(field[peak]),
                'HWT_mm_h': float(fields['HWT'][peak]), 'regression_mm_h': float(fields['regression'][peak]),
                'log_correction_from_regression': float(correction[peak])}}
    return {'precip_log_scale_mm_h': float(scale), 'fields': records,
            'interpretation': 'At each pixel, (P + scale)/(Preg + scale) = exp(log correction). '
                              'HWT correction is the observed correction relative to regression.',
            'limits': 'Computed from decoded, nonnegative rainfall. Negative pre-clipping endpoints cannot be recovered. '
                      'Flow-mean log is log of the physical ensemble mean, not mean member log. '
                      'Saved rainfall alone cannot validate checkpoint flow_scale or raw ODE endpoints.'}


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
            'log_space': log_space_audit(fields, area, precip_scale),
            'probabilistic': probabilistic, 'rank_histogram': rank_histogram(ensemble, truth, area),
            'regional': regional, 'coarsened': coarsened, 'spectra': spectra,
            'precipitation_skill': rain_skill, 'displacement_on_8_pixel_blocks': displacement,
            'postdecode_log_residual_shrinkage': amplitude,
            'jensen_gap': {'mean_mm_h': weighted_mean(gap, area),
                           'note': 'Arithmetic member mean minus decoded mean log rainfall; expected from convexity, not proof of excess variance'}}


def identity_differences(reference, actual):
    return {name: {'reference': reference[name], 'actual': actual[name]}
            for name in IDENTITY_KEYS if reference[name] != actual[name]}


def describe_differences(reference, actual):
    return '; '.join(f'{name}: {values["reference"]!r} -> {values["actual"]!r}'
                     for name, values in identity_differences(reference, actual).items())


def member_identity(ds, path, entry, fingerprint, member):
    """Check lightweight metadata before loading any rainfall arrays."""
    if (ds.attrs.get('version') != 'v2' or ds.attrs.get('stage') != 'flow'
            or ds.attrs.get('dataset_fingerprint') != fingerprint
            or ds.attrs.get('split') != entry['split'] or ds.attrs.get('ensemble_member') != member
            or ds.sizes.get('time') != 1 or 'time' not in ds.coords
            or ds.time.values[0] != np.datetime64(entry['time'])):
        raise ValueError(f'{path}: incompatible v2 member/archive/time/split')
    if any(key not in ds.attrs for key in (*IDENTITY_KEYS, 'seed') if key != 'noise_padding'):
        raise ValueError(f'{path}: missing sampler/checkpoint provenance')
    return {**{name: ds.attrs[name].item() if isinstance(ds.attrs[name], np.generic) else ds.attrs[name]
               for name in IDENTITY_KEYS if name != 'noise_padding'},
            'noise_padding': saved_noise_padding_v2(ds.attrs)}


def preflight(entries, root, fingerprint, expected_members):
    """Check all member metadata and partition hours by exact saved identity."""
    plans, groups, count = [], [], expected_members
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
        if len(paths) < 2 or (count is not None and len(paths) != count):
            raise ValueError(f'{entry["id"]}: incomplete ensemble or fewer than two members')
        count = len(paths)
        identity, seeds = None, []
        for member, path in enumerate(paths):
            with xr.open_dataset(path) as ds:
                key = member_identity(ds, path, entry, fingerprint, member)
                if identity is not None and key != identity:
                    raise ValueError(f'{path}: mixed checkpoint or sampler settings within {entry["id"]}; '
                                     f'compared with {paths[0]}: {describe_differences(identity, key)}. '
                                     'Grouping cannot repair a mixed ensemble.')
                identity = key
                seeds.append(int(ds.attrs['seed']))
        if len(set(seeds)) != count:
            raise ValueError(f'{entry["id"]}: repeated member seeds')
        group = next((g for g in groups if g['identity'] == identity), None)
        if group is None:
            group = {'id': f'group_{len(groups)+1:03d}', 'identity': identity, 'timestamps': [],
                     'differences_from_first_group': identity_differences(groups[0]['identity'], identity) if groups else {}}
            groups.append(group)
        group['timestamps'].append(entry['id'])
        plans.append({'entry': entry, 'paths': paths, 'identity': identity, 'seeds': seeds, 'group': group['id']})
    return plans, groups, count


def load_members(paths, entry, fingerprint, static, expected_members=None):
    """Load only rainfall; require common provenance, coordinates and regression."""
    if len(paths) < 2 or (expected_members is not None and len(paths) != expected_members):
        raise ValueError(f'{entry["id"]}: incomplete ensemble or fewer than two members')
    members, regression, identity, seeds = [], None, None, []
    for member, path in enumerate(paths):
        with xr.open_dataset(path) as ds:
            key = member_identity(ds, path, entry, fingerprint, member)
            if identity is not None and key != identity:
                raise ValueError(f'{path}: mixed checkpoint or sampler settings; {describe_differences(identity, key)}')
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


def run_audit(cfg, predictions, output=None, split='test', timestamps=None, members=None, plots=True,
              group_by_identity=False):
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
    plans, groups, count = preflight(entries, root, index['fingerprint'], members)
    provenance = {'split': split, 'members': count, 'archive_fingerprint': index['fingerprint'],
                  'groups': groups, 'cases': [
                      {'id': p['entry']['id'], 'group': p['group'], 'seeds': p['seeds'],
                       'prediction_files': [str(path.resolve()) for path in p['paths']]} for p in plans]}
    write_json(out/'provenance_v2.json', provenance)
    differences = [f'{g["id"]} ({", ".join(g["timestamps"])}), compared with '
                   f'{groups[0]["id"]} ({", ".join(groups[0]["timestamps"])}): '
                   f'{describe_differences(groups[0]["identity"], g["identity"])}' for g in groups[1:]]
    print(f'Provenance preflight: {len(plans)} hours, {len(groups)} compatible group(s).', flush=True)
    for difference in differences:
        print(difference, flush=True)
    if len(groups) > 1 and not group_by_identity:
        raise ValueError('Mixed checkpoint or sampler settings across timestamps:\n' + '\n'.join(differences)
                         + f'\nDetails: {out}/provenance_v2.json. No case metrics computed. '
                         'Use --group-by-identity (batch: GROUP_BY_IDENTITY=1) with a fresh output directory '
                         'to audit groups separately, or select matching --timestamps.')
    reports = []
    for plan in plans:
        entry, paths = plan['entry'], plan['paths']
        ensemble, regression, key, seeds = load_members(paths, entry, index['fingerprint'], static, count)
        if key != plan['identity'] or seeds != plan['seeds']:
            raise ValueError(f'{entry["id"]}: member metadata changed after preflight; use stable prediction files')
        truth = np.asarray(np.load(archive_root/entry['id']/'truth_v2.npy', mmap_mode='r')[1])
        baseline = np.asarray(np.load(archive_root/entry['id']/'baseline_v2.npy', mmap_mode='r')[1])
        print(f'Auditing {entry["id"]}: {count} members, {plan["group"]}', flush=True)
        report = audit_case(ensemble, regression, baseline, truth, static['area'], cfg['patch'], stats['precip_log_scale'])
        report.update(id=entry['id'], time=entry['time'], split=split, seeds=seeds,
                      identity=key, group=plan['group'],
                      prediction_files=[str(p.resolve()) for p in paths])
        residual_stats = stats.get('residual')
        report['log_space']['archive_residual_normalization'] = (
            {'precip_mean': residual_stats['mean'][1], 'precip_std': residual_stats['std'][1]}
            if residual_stats else None)
        write_json(out/f'{entry["id"]}_audit_v2.json', report)
        if plots:
            from .precip_plots_v2 import plot_case
            plot_case(out, report, ensemble, regression, baseline, truth)
        reports.append(report)
    for group in groups:
        destination = out if len(groups) == 1 else out/'groups_v2'/group['id']
        destination.mkdir(parents=True, exist_ok=True)
        group_reports = [r for r in reports if r['group'] == group['id']]
        write_group_reports(destination, group_reports, root, cfg, split, count, index['fingerprint'],
                            group['identity'], len(available))
    if len(groups) > 1:
        write_json(out/'summary_v2.json', {
            **provenance, 'hours_audited': len(reports),
            'aggregation': 'No combined scores: checkpoint/sampler identities differ; see groups_v2/<group>/summary_v2.json'})
        lines = ['# V2 precipitation audit: separate provenance groups', '',
                 f'{len(reports)} {split} hours; {count} members per hour. No combined scores.', '',
                 'Case JSONs and plots are in this directory. Each group has its own report, CSV and summary.', '',
                 'Different groups may contain different weather cases; their averages are not a controlled model comparison.', '']
        for group in groups:
            lines += [f'- [{group["id"]}](groups_v2/{group["id"]}/report_v2.md): ' + ', '.join(group['timestamps'])]
        lines += ['', '## Saved metadata differences', '', *differences, '',
                  'Full identities and input files: [provenance_v2.json](provenance_v2.json).']
        (out/'report_v2.md').write_text('\n'.join(lines)+'\n')
    return out


def write_group_reports(out, reports, root, cfg, split, count, fingerprint, identity, available_count):
    """Summarize hours only after ensuring they share one exact identity."""
    selection_path = root.parent/'metrics_v2.json'
    selection = None
    if selection_path.is_file():
        previous = json.loads(selection_path.read_text())
        if previous.get('checkpoint_sha256') == identity['checkpoint_sha256'] and previous.get('split') == split:
            selection = previous.get('selection')
    # Recalculate each hour first; these are means of hour-level metrics, not
    # pooled spatial RMSE. Event case and random cases remain separately visible.
    summary = {'split': split, 'hours_audited': len(reports), 'members': count,
               'archive_fingerprint': fingerprint, 'identity': identity,
               'config_patch_assumption': cfg['patch'], 'selection': selection,
               'timestamps': [r['id'] for r in reports], 'other_split_hours_not_audited': available_count-len(reports),
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
              '- The companion errors_detail map uses percentile color limits, labels the saturated fraction and marks the largest absolute value.',
              '- The log_space JSON compares each decoded log correction with the HWT correction needed from regression; it also locates peak rainfall pixels.',
              '- Positive log corrections amplify P + scale exponentially. This explains amplification, not why the model generated the correction.', '',
              '## Limits', '', *[f'- {note}' for note in summary['limits']]]
    if selection:
        lines += ['', f'Original case selection: {selection}.']
    (out/'report_v2.md').write_text('\n'.join(lines)+'\n')
