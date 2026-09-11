#!/usr/bin/env python3
"""Generate held-out ensemble diagnostics with full-domain and event-zoom maps."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, PowerNorm, TwoSlopeNorm
import numpy as np
import xarray as xr

try:
    import cartopy.crs as ccrs
except ImportError:  # Geographic lon/lat plotting remains available without Cartopy.
    ccrs = None

from merraflow import TARGETS
from merraflow.config import load_config, write_json
from merraflow.dataset import Archive
from merraflow.evaluate import load_members
from merraflow.inference import predict
from merraflow.metrics import continuous


DISPLAY = (
    ('2 m temperature', '°C', lambda x: x-273.15, 'coolwarm'),
    ('precipitation rate', 'mm h⁻¹', lambda x: x, 'YlGnBu'),
    ('surface pressure', 'hPa', lambda x: x/100, 'viridis'),
    ('10 m wind speed', 'm s⁻¹', lambda x: x, 'magma'),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/discover.yaml')
    parser.add_argument('--checkpoint', help='Default: <train.output>/best.pt')
    parser.add_argument('--output', help='Default: <train.output>/test_best_model_m<MEMBERS>')
    parser.add_argument('--samples', type=int, default=3, help='Number of held-out test timestamps')
    parser.add_argument('--members', type=int, default=5, help='Generated members per timestamp')
    parser.add_argument('--timestamps', nargs='*', help='Optional test IDs/timestamps instead of evenly spaced cases')
    parser.add_argument('--zoom-fraction', type=float, default=.4,
                        help='Fraction of each grid dimension shown around the strongest HR precipitation event')
    return parser.parse_args()


def select_entries(archive, count, requested=None):
    entries = [entry for entry in archive.index['entries'] if entry['split'] == 'test']
    if requested:
        selected = []
        for value in requested:
            matches = [entry for entry in entries if value in (entry['id'], entry['time'])]
            if len(matches) != 1:
                raise ValueError(f'Expected one held-out test entry for {value!r}, found {len(matches)}')
            selected.append(matches[0])
        if len({entry['id'] for entry in selected}) != len(selected):
            raise ValueError('Requested test timestamps must be unique')
        return selected
    if count < 1 or count > len(entries):
        raise ValueError(f'Request 1..{len(entries)} held-out test samples')
    indices = np.rint(np.linspace(0, len(entries)-1, count)).astype(int)
    return [entries[index] for index in indices]


def checkpoint_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as source:
        for block in iter(lambda: source.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def source_projection(archive):
    """Reconstruct the HWT LCC projection retained in grid.nc when available."""
    if ccrs is None:
        return None
    with xr.open_dataset(archive.root/'grid.nc') as grid:
        name = grid.attrs.get('grid_mapping_variable')
        attrs = dict(grid[name].attrs) if name in grid else {}
    if attrs.get('grid_mapping_name') != 'lambert_conformal_conic':
        return None
    parallels = np.atleast_1d(attrs['standard_parallel']).astype(float).tolist()
    if len(parallels) == 1:
        parallels *= 2
    return ccrs.LambertConformal(
        central_longitude=float(attrs['longitude_of_central_meridian']),
        central_latitude=float(attrs['latitude_of_projection_origin']),
        standard_parallels=tuple(parallels[:2]),
        false_easting=float(attrs.get('false_easting', 0.)),
        false_northing=float(attrs.get('false_northing', 0.)),
    )


def raster_geometry(archive):
    """Return projection and cell-edge extent for native-grid imshow rendering."""
    lon, lat = archive.static['lon'], archive.static['lat']
    projection = source_projection(archive)
    if projection is None:
        return None, (0, lon.shape[1], 0, lon.shape[0]), 'lower'
    points = projection.transform_points(ccrs.PlateCarree(), lon, lat)
    x, y = points[..., 0], points[..., 1]
    x_center, y_center = np.nanmedian(x, axis=0), np.nanmedian(y, axis=1)
    dx, dy = np.nanmedian(np.diff(x_center)), np.nanmedian(np.diff(y_center))
    if not np.isfinite([dx, dy]).all() or dx <= 0 or dy <= 0:
        raise ValueError('Cannot infer regular projected HWT cell spacing for imshow')
    x_edge = (x_center[0]-dx/2, x_center[-1]+dx/2)
    y_edge = (y_center[0]-dy/2, y_center[-1]+dy/2)
    extent = (min(x_edge), max(x_edge), min(y_edge), max(y_edge))
    return projection, extent, 'lower'


def map_axes(rows, columns, projection):
    if projection is None:
        return plt.subplots(rows, columns, figsize=(25, 16), constrained_layout=True)
    return plt.subplots(rows, columns, figsize=(25, 16), constrained_layout=True,
                        subplot_kw={'projection': projection})


def draw_map(ax, values, cmap, norm, extent, origin, projection, row, column):
    options = {'origin': origin, 'extent': extent, 'interpolation': 'nearest',
               'cmap': cmap, 'norm': norm, 'rasterized': True, 'aspect': 'auto'}
    if ccrs is not None and projection is not None:
        options['transform'] = projection
    image = ax.imshow(values, **options)
    if ccrs is not None and projection is not None:
        ax.set_extent(extent, crs=projection)
        grid = ax.gridlines(draw_labels=True, linewidth=.2, color='.35', alpha=.35, linestyle=':')
        grid.top_labels = False
        grid.right_labels = False
        grid.bottom_labels = row == 3
        grid.left_labels = column == 0
    else:
        ax.set_xlabel('HWT grid column')
        ax.set_ylabel('HWT grid row')
    return image


def event_window(precip, fraction):
    """Center a fixed-size zoom on the strongest broad HR precipitation feature."""
    h, w = precip.shape
    zh, zw = max(32, int(round(h*fraction))), max(32, int(round(w*fraction)))
    step = max(1, min(h, w)//150)
    coarse = precip[:h//step*step, :w//step*step]
    coarse = coarse.reshape(coarse.shape[0]//step, step, coarse.shape[1]//step, step).mean((1, 3))
    cy, cx = np.unravel_index(np.argmax(coarse), coarse.shape)
    cy, cx = int((cy+.5)*step), int((cx+.5)*step)
    y0 = int(np.clip(cy-zh//2, 0, h-zh))
    x0 = int(np.clip(cx-zw//2, 0, w-zw))
    return slice(y0, y0+zh), slice(x0, x0+zw)


def crop_extent(extent, shape, region):
    h, w = shape
    ys, xs = region
    x0, x1, y0, y1 = extent
    return (x0+(x1-x0)*xs.start/w, x0+(x1-x0)*xs.stop/w,
            y0+(y1-y0)*ys.start/h, y0+(y1-y0)*ys.stop/h)


def plot_maps(output, entry, archive, ensemble, checkpoint, scores, region=None):
    truth = np.asarray(archive.array(entry, 'truth'))
    baseline = np.asarray(archive.array(entry, 'baseline'))
    member, generated = ensemble[0], ensemble.mean(0)
    projection, full_extent, origin = raster_geometry(archive)
    if region is None:
        region = (slice(None), slice(None))
        extent, suffix, label = full_extent, '', 'full CONUS domain'
    else:
        extent, suffix, label = crop_extent(full_extent, truth.shape[1:], region), '_zoom', 'event-centered zoom'
    fig, axes = map_axes(4, 5, projection)
    columns = ('Coarse baseline', 'Original HWT target', 'Individual member 0',
               f'{len(ensemble)}-member ensemble mean', 'Ensemble mean − HWT')
    for row, (key, display) in enumerate(zip(TARGETS, DISPLAY)):
        title, unit, convert, cmap = display
        base_value, truth_value, member_value, generated_value = map(
            convert, (baseline[row], truth[row], member[row], generated[row]))
        pooled = np.concatenate([base_value.ravel(), truth_value.ravel(), member_value.ravel(), generated_value.ravel()])
        if key == 'precip':
            upper = max(float(np.quantile(pooled, .995)), .1)
            norm = PowerNorm(gamma=.45, vmin=0, vmax=upper)
        else:
            low, high = np.quantile(pooled, [.01, .99])
            if high <= low:
                high = low+1
            norm = Normalize(low, high)
        error = generated_value-truth_value
        error_limit = max(float(np.quantile(np.abs(error), .99)), 1e-6)
        error_norm = TwoSlopeNorm(vmin=-error_limit, vcenter=0, vmax=error_limit)
        fields = (base_value, truth_value, member_value, generated_value, error)
        for column, values in enumerate(fields):
            image = draw_map(axes[row, column], values[region],
                             'RdBu_r' if column == 4 else cmap,
                             error_norm if column == 4 else norm, extent, origin,
                             projection, row, column)
            axes[row, column].set_title(f'{title}\n{columns[column]}', fontsize=10, weight='semibold')
            fig.colorbar(image, ax=axes[row, column], orientation='horizontal', pad=.035,
                         shrink=.82, label=unit)
        score_label = 'Full-domain RMSE' if suffix == '' else 'Zoom RMSE'
        axes[row, 0].text(.02, .02, f'{score_label} {scores[key]["baseline"]["rmse"]:.3g} {unit}',
                          transform=axes[row, 0].transAxes, fontsize=8,
                          bbox={'facecolor': 'white', 'alpha': .78, 'edgecolor': 'none'})
        axes[row, 2].text(.02, .02, f'{score_label} {scores[key]["member_0"]["rmse"]:.3g} {unit}',
                          transform=axes[row, 2].transAxes, fontsize=8,
                          bbox={'facecolor': 'white', 'alpha': .78, 'edgecolor': 'none'})
        axes[row, 3].text(.02, .02, f'{score_label} {scores[key]["ensemble_mean"]["rmse"]:.3g} {unit}',
                          transform=axes[row, 3].transAxes, fontsize=8,
                          bbox={'facecolor': 'white', 'alpha': .78, 'edgecolor': 'none'})
        axes[row, 4].text(.02, .02, f'{score_label} {scores[key]["ensemble_mean"]["rmse"]:.3g} {unit}',
                          transform=axes[row, 4].transAxes, fontsize=8,
                          bbox={'facecolor': 'white', 'alpha': .78, 'edgecolor': 'none'})
    fig.suptitle(f'MERRA21C-ML held-out test diagnostic · {entry["time"]} UTC\n'
                 f'{checkpoint.name} · {len(ensemble)} members · {label}', fontsize=15, weight='bold')
    destination = output/f'test_{entry["id"]}{suffix}.png'
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def variable_scores(archive, ensemble, truth, baseline, region):
    ys, xs = region
    area = archive.static['area'][ys, xs]
    scores = {}
    for row, (key, display) in enumerate(zip(TARGETS, DISPLAY)):
        unit, convert = display[1:3]
        truth_value = convert(truth[row])[ys, xs]
        baseline_score = continuous(convert(baseline[None, row])[:, ys, xs], truth_value, area)
        member_score = continuous(convert(ensemble[:1, row])[:, ys, xs], truth_value, area)
        ensemble_score = continuous(convert(ensemble[:, row])[:, ys, xs], truth_value, area)
        scores[key] = {
            'unit': unit, 'baseline': baseline_score, 'member_0': member_score,
            'ensemble_mean': ensemble_score,
            'rmse_skill_percent': 100*(1-ensemble_score['rmse']/baseline_score['rmse'])
            if baseline_score['rmse'] else None,
        }
    return scores


def plot_case(output, entry, archive, ensemble, checkpoint, zoom_fraction):
    truth = np.asarray(archive.array(entry, 'truth'))
    baseline = np.asarray(archive.array(entry, 'baseline'))
    full_region = (slice(None), slice(None))
    zoom_region = event_window(truth[1], zoom_fraction)
    full_scores = variable_scores(archive, ensemble, truth, baseline, full_region)
    zoom_scores = variable_scores(archive, ensemble, truth, baseline, zoom_region)
    metrics = {'id': entry['id'], 'time': entry['time'], 'members': len(ensemble),
               'checkpoint': str(checkpoint.resolve()), 'checkpoint_sha256': checkpoint_sha256(checkpoint),
               'variables': full_scores,
               'zoom_grid_bounds': {'y': [zoom_region[0].start, zoom_region[0].stop],
                                    'x': [zoom_region[1].start, zoom_region[1].stop]},
               'zoom_variables': zoom_scores}
    plot_maps(output, entry, archive, ensemble, checkpoint, full_scores)
    plot_maps(output, entry, archive, ensemble, checkpoint, zoom_scores, zoom_region)
    return metrics


def plot_summary(output, rows):
    labels = [row['id'][:8] for row in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, (key, display) in zip(axes.ravel(), zip(TARGETS, DISPLAY)):
        title, unit = display[:2]
        baseline = [row['variables'][key]['baseline']['rmse'] for row in rows]
        generated = [row['variables'][key]['ensemble_mean']['rmse'] for row in rows]
        ax.bar(x-.19, baseline, width=.38, label='Coarse baseline', color='#8196a8')
        ax.bar(x+.19, generated, width=.38, label=f'{rows[0]["members"]}-member ensemble mean', color='#146c94')
        ax.set(xticks=x, xticklabels=labels, ylabel=f'Area-weighted RMSE ({unit})', title=title)
        ax.tick_params(axis='x', rotation=25)
        ax.grid(axis='y', alpha=.25)
        ax.legend(fontsize=8)
    fig.suptitle(f'Best-checkpoint performance on {len(rows)} held-out test cases', fontsize=14, weight='bold')
    fig.savefig(output/'rmse_summary.png', dpi=180)
    plt.close(fig)


def plot_training_history(output, training_root):
    history = training_root/'history.jsonl'
    if not history.exists():
        return
    rows = [json.loads(line) for line in history.read_text().splitlines() if line.strip()]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    ax.plot([row['epoch'] for row in rows], [row['train_loss'] for row in rows], label='Training')
    ax.plot([row['epoch'] for row in rows], [row['val_loss'] for row in rows], label='Validation EMA')
    best = min(rows, key=lambda row: row['val_loss'])
    ax.scatter([best['epoch']], [best['val_loss']], color='#b2182b', zorder=3,
               label=f'Best epoch {best["epoch"]}')
    ax.set(xlabel='Epoch', ylabel='Area-weighted flow loss', title='Training and validation history')
    ax.grid(alpha=.25)
    ax.legend()
    fig.savefig(output/'training_history.png', dpi=180)
    plt.close(fig)


def validate_reused_predictions(paths, checkpoint, cfg, archive):
    expected_checkpoint = str(checkpoint.resolve())
    expected = {'checkpoint': expected_checkpoint, 'ode_steps': cfg['inference']['steps'],
                'patch_size': cfg['patch']['size'], 'patch_halo': cfg['patch']['halo'],
                'patch_stride': cfg['patch']['stride'],
                'dry_threshold_mm_h': cfg['inference']['dry_threshold'],
                'checkpoint_sha256': checkpoint_sha256(checkpoint),
                'dataset_fingerprint': archive.index['fingerprint']}
    for path in paths:
        with xr.open_dataset(path) as dataset:
            actual = {name: dataset.attrs.get(name) for name in expected}
        if actual != expected:
            raise ValueError(f'{path}: existing prediction does not match requested checkpoint/sampler')


def main():
    args = parse_args()
    if args.members < 2 or not 0 < args.zoom_fraction <= 1:
        raise ValueError('members must be at least 2 and zoom-fraction must be in (0, 1]')
    cfg = load_config(args.config)
    checkpoint = Path(args.checkpoint) if args.checkpoint else Path(cfg['train']['output'])/'best.pt'
    if not checkpoint.is_file():
        raise FileNotFoundError(f'Best checkpoint not found: {checkpoint}')
    archive = Archive(cfg['data']['prepared'])
    selected = select_entries(archive, args.samples, args.timestamps)
    output = Path(args.output or Path(cfg['train']['output'])/f'test_best_model_m{args.members}')
    prediction_root = output/'predictions'
    output.mkdir(parents=True, exist_ok=True)
    prediction_root.mkdir(exist_ok=True)
    cfg['inference']['members'] = args.members
    cfg['inference']['output'] = str(prediction_root)
    reports = []
    for entry in selected:
        paths = sorted(prediction_root.glob(f'{entry["id"]}_m*.nc'))
        if paths and len(paths) != args.members:
            raise ValueError(f'{entry["id"]}: found {len(paths)} existing members, expected {args.members}')
        if not paths:
            predict(cfg, checkpoint, split='test', timestamp=entry['id'])
            paths = sorted(prediction_root.glob(f'{entry["id"]}_m*.nc'))
        validate_reused_predictions(paths, checkpoint, cfg, archive)
        ensemble, _ = load_members(paths, archive, entry)
        report = plot_case(output, entry, archive, ensemble, checkpoint, args.zoom_fraction)
        reports.append(report)
        print(f'Plotted held-out test case {entry["id"]}', flush=True)
    write_json(output/'metrics.json', {'selection': 'explicit' if args.timestamps else 'evenly spaced test timestamps',
                                      'samples': reports})
    plot_summary(output, reports)
    plot_training_history(output, Path(cfg['train']['output']))
    print(f'Best-model test diagnostics written to {output}', flush=True)


if __name__ == '__main__':
    main()
