"""Plot a whole-CONUS rainfall case from existing v2 NetCDF members, CPU only."""
import argparse
import json
from pathlib import Path
import re

import numpy as np
import xarray as xr

from .analyze_wet_precip_direct_v2 import spatial_scores
from .metrics import crps_ensemble, weighted_mean


DEFAULT_TIMESTAMP = '20260223_0530'
KINDS = {'direct': 'Direct full-precipitation flow',
         'residual-v2': 'Original residual v2 flow'}


def archive_case(root, timestamp, split):
    root = Path(root)
    index = json.loads((root/'index_v2.json').read_text())
    entries = [entry for entry in index['entries'] if entry['split'] == split
               and timestamp in (entry['id'], entry['time'])]
    if len(entries) != 1:
        raise ValueError(f'Expected one {split} archive entry for {timestamp!r}, found {len(entries)}')
    entry = entries[0]
    truth = np.asarray(np.load(root/entry['id']/'truth_v2.npy', mmap_mode='r')[1], dtype='float32')
    coarse = np.asarray(np.load(root/entry['id']/'baseline_v2.npy', mmap_mode='r')[1], dtype='float32')
    with np.load(root/'static_v2.npz') as static:
        area = static['area'].copy()
    if truth.shape != coarse.shape or truth.shape != area.shape:
        raise ValueError('Archive truth, coarse, and area grid shapes differ')
    return entry, truth, coarse, area, index['fingerprint']


def available_member_groups(folder, entry_id):
    folder = Path(folder)
    groups = {}
    pattern = re.compile(rf'{re.escape(entry_id)}_m(\d+)(_direct)?_v2\.nc')
    for path in folder.rglob(f'{entry_id}_m*_v2.nc'):
        match = pattern.fullmatch(path.name)
        if match:
            kind = 'direct' if match.group(2) else 'residual-v2'
            groups.setdefault((kind, path.parent), []).append((int(match.group(1)), path))
    return groups


def member_paths(folder, entry_id, kind='auto'):
    groups = available_member_groups(folder, entry_id)
    if kind not in KINDS:
        if kind != 'auto':
            raise ValueError(f'Unknown saved-model kind: {kind}')
        # Prefer the direct experiment only when it is unambiguous. Never
        # silently substitute an original residual-v2 run for that model.
        kind = 'direct' if any(key[0] == 'direct' for key in groups) else 'residual-v2'
    matching = [(key, paths) for key, paths in groups.items() if key[0] == kind]
    if len(matching) != 1:
        found = [dict(kind=key[0], directory=str(key[1]), members=len(paths))
                 for key, paths in sorted(groups.items(), key=lambda item: str(item[0][1]))]
        raise ValueError(f'Expected one {kind} member directory for {entry_id}; found {found}. '
                         'Use --list, then set --predictions to the exact directory.')
    selected = sorted(matching[0][1])
    ids = [number for number, _ in selected]
    if ids != list(range(len(ids))):
        raise ValueError(f'Need complete member IDs 0..{len(ids)-1}; found {ids}')
    return kind, [path for _, path in selected]


def load_members(paths, kind, entry, shape, fingerprint):
    members, identity = [], None
    expected_version = 'v2_precip_direct' if kind == 'direct' else 'v2'
    for number, path in enumerate(paths):
        with xr.open_dataset(path) as ds:
            attrs = ds.attrs
            if attrs.get('version') != expected_version or attrs.get('split') != entry['split']:
                raise ValueError(f'Wrong model or split in {path}')
            if kind == 'direct' and attrs.get('target') != 'full precipitation':
                raise ValueError(f'Not a direct full-precipitation member: {path}')
            if kind == 'residual-v2' and attrs.get('dataset_fingerprint') != fingerprint:
                raise ValueError(f'Original v2 archive fingerprint differs: {path}')
            if int(attrs.get('ensemble_member', -1)) != number:
                raise ValueError(f'Wrong member number in {path}')
            if 'time' not in ds or ds.time.size != 1 or ds.time.values[0] != np.datetime64(entry['time']):
                raise ValueError(f'Wrong timestamp in {path}')
            if 'precip' not in ds or ds.precip.shape != (1, *shape):
                raise ValueError(f'Wrong rainfall field shape in {path}')
            key = (attrs.get('checkpoint_sha256'), attrs.get('checkpoint_epoch'),
                   attrs.get('sampler'), attrs.get('ode_steps'))
            if not key[0]:
                raise ValueError(f'Missing checkpoint hash in {path}')
            if identity is not None and key != identity:
                raise ValueError('Saved members mix checkpoints or inference settings')
            identity = key
            value = np.asarray(ds.precip.values[0], dtype='float32')
            if not np.isfinite(value).all() or np.any(value < 0):
                raise ValueError(f'Nonfinite or negative rainfall in {path}')
            members.append(value)
    return np.stack(members), identity


def scores(ensemble, truth, coarse, area):
    mean = ensemble.mean(0)
    return dict(crps_mm_h=weighted_mean(crps_ensemble(ensemble, truth), area),
                coarse_mae_mm_h=weighted_mean(abs(coarse-truth), area),
                ensemble_mean_rmse_mm_h=float(np.sqrt(weighted_mean((mean-truth)**2, area))),
                mean_bias_mm_h=weighted_mean(mean-truth, area),
                truth_mean_mm_h=weighted_mean(truth, area),
                generated_mean_mm_h=weighted_mean(mean, area),
                truth_wet_fraction=weighted_mean(truth >= .1, area),
                generated_wet_fraction=weighted_mean((ensemble >= .1).mean(0), area))


def plot_case(path, entry, kind, truth, coarse, ensemble, metrics):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from matplotlib.colors import PowerNorm

    mean = ensemble.mean(0)
    fields = [('HWT truth', truth), ('Coarse input', coarse)]
    fields += [(f'Member {number+1}', value) for number, value in enumerate(ensemble[:2])]
    if len(ensemble) > 1:
        fields += [('Ensemble mean', mean), ('Ensemble spread', ensemble.std(0))]
    vmax = max(1., float(np.quantile(np.stack([truth, coarse, mean]), .995)))
    norm = PowerNorm(gamma=.45, vmin=0, vmax=vmax)
    columns = 3
    rows = int(np.ceil(len(fields)/columns))
    fig, axes = plt.subplots(rows, columns, figsize=(6.3*columns, 4.5*rows),
                             squeeze=False, constrained_layout=True)
    try:
        for ax, (label, field) in zip(axes.flat, fields):
            image = ax.imshow(field, origin='lower', interpolation='nearest', cmap='Blues',
                              norm=norm, rasterized=True)
            ax.set(title=label, xticks=[], yticks=[])
        for ax in list(axes.flat)[len(fields):]:
            ax.axis('off')
        fig.colorbar(image, ax=axes.ravel().tolist(), label='Rain rate (mm/h); common scale',
                     extend='max', shrink=.85)
        fig.suptitle(f'{KINDS[kind]} · whole CONUS · {entry["time"]} UTC · '
                     f'CRPS {metrics["crps_mm_h"]:.3f} vs coarse MAE '
                     f'{metrics["coarse_mae_mm_h"]:.3f} mm/h')
        fig.savefig(path, dpi=140)
    finally:
        plt.close(fig)


def plot_saved(archive, predictions, output, timestamp=DEFAULT_TIMESTAMP,
               split='test', kind='auto'):
    entry, truth, coarse, area, fingerprint = archive_case(archive, timestamp, split)
    kind, paths = member_paths(predictions, entry['id'], kind)
    ensemble, identity = load_members(paths, kind, entry, truth.shape, fingerprint)
    metrics = scores(ensemble, truth, coarse, area)
    fss = spatial_scores(ensemble, truth, coarse, None, area)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Use a fresh output directory: {output}')
    output.mkdir(parents=True, exist_ok=True)
    report = dict(id=entry['id'], time=entry['time'], split=split, model_kind=kind,
                  model_label=KINDS[kind], checkpoint_sha256=identity[0],
                  saved_checkpoint_epoch=(int(identity[1]) if identity[1] is not None else None),
                  members=len(ensemble),
                  sources=[str(path.resolve()) for path in paths], metrics=metrics,
                  fss=fss, caveat='A selected test hour is a case study, not population test skill. '
                  'The v2 HWT target is a :30 snapshot, not an hourly accumulation.')
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    image = output/f'{entry["id"]}_{kind}_full_conus.png'
    plot_case(image, entry, kind, truth, coarse, ensemble, metrics)
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', default='data/paired_hourly_annual_v2')
    parser.add_argument('--predictions', default='runs',
                        help='Search root or exact folder containing saved NetCDF members')
    parser.add_argument('--output', help='Fresh directory for PNG and report.json')
    parser.add_argument('--timestamp', default=DEFAULT_TIMESTAMP)
    parser.add_argument('--split', choices=('val', 'test'), default='test')
    parser.add_argument('--kind', choices=('auto', *KINDS), default='auto')
    parser.add_argument('--list', action='store_true', help='List matching saved ensembles without plotting')
    args = parser.parse_args()
    if args.list:
        groups = available_member_groups(args.predictions, args.timestamp)
        print(json.dumps([dict(kind=key[0], directory=str(key[1]), members=len(paths))
                          for key, paths in sorted(groups.items(), key=lambda item: str(item[0][1]))], indent=2))
        return
    if not args.output:
        parser.error('--output is required unless --list is used')
    print(plot_saved(args.archive, args.predictions, args.output,
                     args.timestamp, args.split, args.kind))


if __name__ == '__main__':
    main()
