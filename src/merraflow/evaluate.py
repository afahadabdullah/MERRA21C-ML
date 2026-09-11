from pathlib import Path
import json
import numpy as np
import xarray as xr
from . import TARGETS, UNITS
from .config import write_json
from .dataset import Archive
from .metrics import continuous, precipitation, rank_histogram
from .physics import budget_error


def load_members(paths, archive, entry):
    members, audits, ids, signatures = [], [], [], []
    for path in paths:
        with xr.open_dataset(path) as ds:
            if ds.attrs.get('dataset_fingerprint') != archive.index['fingerprint'] or ds.attrs.get('split') != entry['split']:
                raise ValueError(f'Prediction provenance mismatch: {path}')
            if ds.time.size != 1 or ds.time.values[0] != np.datetime64(entry['time']):
                raise ValueError(f'Prediction time mismatch: {path}')
            values = np.stack([ds[k].isel(time=0).values for k in TARGETS])
            if values.shape != (4, *archive.shape) or not np.isfinite(values).all():
                raise ValueError(f'Invalid prediction shape/values: {path}')
            if np.any(values[[1, 3]] < 0):
                raise ValueError(f'Negative precipitation/wind: {path}')
            members.append(values)
            audits.append(json.loads(ds.attrs['conservation_audit']))
            ids.append(ds.attrs['ensemble_member'])
            signatures.append(tuple(ds.attrs[k] for k in ('checkpoint', 'checkpoint_epoch', 'checkpoint_sha256',
                                                          'patch_size', 'patch_halo', 'patch_stride',
                                                          'ode_steps', 'dry_threshold_mm_h')))
    if len(set(ids)) != len(ids) or len(set(signatures)) != 1:
        raise ValueError('Duplicate ensemble member or mixed inference configurations')
    return np.stack(members), audits


def evaluate(cfg, predictions=None, split='test', output=None):
    if split not in ('val', 'test'):
        raise ValueError('Evaluate only held-out val/test data')
    archive = Archive(cfg['data']['prepared'])
    root = Path(predictions or cfg['inference']['output'])
    dest = Path(output or root/'evaluation')
    entries = [e for e in archive.index['entries'] if e['split'] == split]
    area, groups = archive.static['area'], archive.static['groups']
    reports, missing = [], []
    for entry in entries:
        paths = sorted(root.glob(f'{entry["id"]}_m*.nc'))
        if not paths:
            missing.append(entry['id'])
            continue
        if len(paths) != cfg['inference']['members']:
            raise ValueError(f'Incomplete ensemble for {entry["id"]}: {len(paths)} members')
        ensemble, audit = load_members(paths, archive, entry)
        truth, target, baseline = [archive.array(entry, k) for k in ('truth', 'target', 'baseline')]
        report = {'id': entry['id'], 'time': entry['time'], 'members': len(paths), 'original_hr': {}, 'training_target': {}, 'baseline': {},
                  'rank_histogram': {}, 'projection_audit': audit,
                  'mass': [budget_error(p[1], baseline[1], area, groups) for p in ensemble]}
        for i, name in enumerate(TARGETS):
            report['original_hr'][name] = continuous(ensemble[:, i], truth[i], area)
            report['training_target'][name] = continuous(ensemble[:, i], target[i], area)
            report['baseline'][name] = continuous(baseline[None, i], truth[i], area)
            report['rank_histogram'][name] = rank_histogram(ensemble[:, i], truth[i], area)
        report['precipitation'] = precipitation(ensemble[:, 1], truth[1], area)
        report['baseline_precipitation'] = precipitation(baseline[None, 1], truth[1], area)
        reports.append(report)
        print(f'Evaluated {entry["id"]}', flush=True)
    if not reports:
        raise ValueError('No predictions found for requested split')
    # Explicitly a mean of per-hour scores, not an incorrectly labeled pooled RMSE.
    summary = {'aggregation': 'equal-weight mean of per-hour, area-weighted scores; not pooled RMSE',
               'split': split, 'hours_evaluated': len(reports), 'missing_hours': missing,
               'units': dict(zip(TARGETS, UNITS)), 'conservation': archive.index['conservation'],
               'training_target_conserved': archive.index['data_config']['conserve_training_precip'],
               'baseline_definition': 'bilinear T2M/PS, magnitude of bilinear U/V; native footprint precipitation',
               'metrics': {}}
    for reference in ('original_hr', 'training_target', 'baseline'):
        summary['metrics'][reference] = {}
        for name in TARGETS:
            summary['metrics'][reference][name] = {}
            for metric in reports[0][reference][name]:
                values = [r[reference][name][metric] for r in reports if r[reference][name][metric] is not None]
                summary['metrics'][reference][name][metric] = float(np.mean(values)) if values else None
    write_json(dest/'per_hour.json', reports)
    write_json(dest/'summary.json', summary)
    return dest
