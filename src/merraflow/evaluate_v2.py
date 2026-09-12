"""Physical skill, rain structure, spectra and budget errors for v2 samples."""
from pathlib import Path
import json
import numpy as np
import xarray as xr
from .config import write_json
from .dataset_v2 import ArchiveV2
from .physics_v2 import TARGETS_V2
from .metrics import continuous, precipitation, radial_psd, rank_histogram


def load_members_v2(paths, archive, entry):
    members, means, audits, identity = [], [], [], None
    for member, path in enumerate(paths):
        with xr.open_dataset(path) as ds:
            if (ds.attrs.get('version') != 'v2' or ds.attrs['dataset_fingerprint'] != archive.index['fingerprint']
                    or ds.attrs['ensemble_member'] != member or ds.attrs['split'] != entry['split']
                    or ds.time.size != 1 or ds.time.values[0] != np.datetime64(entry['time'])):
                raise ValueError(f'Incompatible v2 prediction: {path}')
            key = tuple(ds.attrs[k] for k in ('checkpoint_sha256', 'regression_sha256', 'ode_steps', 'blend'))
            if identity is not None and key != identity:
                raise ValueError('Mixed v2 checkpoints or inference settings')
            identity = key
            members.append(np.stack([ds[n].values[0] for n in TARGETS_V2]))
            means.append(np.stack([ds['regression_'+n].values[0] for n in TARGETS_V2]))
            audits.append(json.loads(ds.attrs['budget_audit_v2']))
    if not members or not np.isfinite(members).all():
        raise ValueError('Empty or nonfinite v2 ensemble')
    if any(not np.array_equal(means[0], m) for m in means[1:]):
        raise ValueError('Regression prediction changed across ensemble members')
    return np.stack(members), means[0], audits, identity


def evaluate_v2(cfg, split='val', output=None):
    if split not in ('val', 'test'):
        raise ValueError('Evaluate v2 on val or test')
    archive = ArchiveV2(cfg['data']['prepared'])
    root = Path(cfg['inference']['output'])
    dest = Path(output or root/'evaluation_v2')
    reports, missing, identity = [], [], None
    area = archive.static['area']
    for entry in [e for e in archive.index['entries'] if e['split'] == split]:
        paths = sorted(root.glob(f'{entry["id"]}_m*_v2.nc'))
        if not paths:
            missing.append(entry['id'])
            continue
        if len(paths) != cfg['inference']['members']:
            raise ValueError(f'Incomplete ensemble: {entry["id"]}')
        ensemble, mean, audits, key = load_members_v2(paths, archive, entry)
        if identity is not None and key != identity:
            raise ValueError('Mixed v2 checkpoint/settings across timestamps')
        identity = key
        truth, baseline = [np.asarray(archive.array(entry, k)) for k in ('truth', 'baseline')]
        # Speed is derived per member before forming an ensemble mean.
        def speed(x):
            return np.concatenate([x, np.hypot(x[..., 3:4, :, :], x[..., 4:5, :, :])], axis=-3)
        ensemble, mean, truth, baseline = map(speed, (ensemble, mean, truth, baseline))
        report = dict(id=entry['id'], ensemble={}, regression={}, baseline={}, spectra={}, ranks={}, budget_audit=audits, coastal={})
        coast = abs(archive.static['features'][8]) < .2  # within ten grid pixels of water
        for i, name in enumerate((*TARGETS_V2, 'wind10m')):
            for label, values in [('ensemble', ensemble[:, i]), ('regression', mean[None, i]), ('baseline', baseline[None, i])]:
                report[label][name] = continuous(values, truth[i], area)
            if coast.any():
                report['coastal'][name] = continuous(ensemble[:, i, coast], truth[i, coast], area[coast])
            freq, power = radial_psd(truth[i])
            report['spectra'][name] = dict(cycles_per_pixel=freq.tolist(), truth=power.tolist(),
                                         member_mean=np.mean([radial_psd(x[i])[1] for x in ensemble], axis=0).tolist(),
                                         ensemble_mean=radial_psd(ensemble[:, i].mean(0))[1].tolist(),
                                         regression=radial_psd(mean[i])[1].tolist())
            report['ranks'][name] = rank_histogram(ensemble[:, i], truth[i], area)
        report['precipitation'] = precipitation(ensemble[:, 1], truth[1], area)
        report['baseline_precipitation'] = precipitation(baseline[None, 1], truth[1], area)
        reports.append(report)
    if not reports:
        raise ValueError('No v2 predictions found')
    summary = dict(version='v2', split=split, hours_evaluated=len(reports), missing_hours=missing,
                   aggregation='equal-weight mean of per-hour area-weighted scores; not pooled RMSE',
                   checkpoint_sha256=identity[0], conservation='none; report budget discrepancies', metrics={})
    for label in ('ensemble', 'regression', 'baseline'):
        summary['metrics'][label] = {}
        for name in reports[0][label]:
            summary['metrics'][label][name] = {}
            for metric in reports[0][label][name]:
                values = [r[label][name][metric] for r in reports if r[label][name][metric] is not None]
                summary['metrics'][label][name][metric] = float(np.mean(values)) if values else None
    write_json(dest/'per_hour_v2.json', reports)
    write_json(dest/'summary_v2.json', summary)
    return dest
