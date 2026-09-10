#!/usr/bin/env python3
"""Generate three held-out test cases and deterministic diagnostic maps."""
import argparse
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
    parser.add_argument('--checkpoint', default='runs/cfm128/best.pt')
    parser.add_argument('--output', help='Default: <train.output>/test_best_model')
    parser.add_argument('--samples', type=int, default=3, help='Number of held-out test timestamps')
    parser.add_argument('--members', type=int, default=1, help='Generated members per timestamp')
    parser.add_argument('--timestamps', nargs='*', help='Optional test IDs/timestamps instead of evenly spaced cases')
    parser.add_argument('--plot-step', type=int, default=3, help='Map display subsampling only; inference stays full resolution')
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


def map_axes(rows, columns, lon, lat):
    if ccrs is None:
        return plt.subplots(rows, columns, figsize=(20, 16), constrained_layout=True)
    center_lon = float(np.nanmean(lon))
    center_lat = float(np.nanmean(lat))
    projection = ccrs.LambertConformal(central_longitude=center_lon, central_latitude=center_lat,
                                       standard_parallels=(30, 45))
    return plt.subplots(rows, columns, figsize=(20, 16), constrained_layout=True,
                        subplot_kw={'projection': projection})


def draw_map(ax, lon, lat, values, cmap, norm, row, column):
    options = {'shading': 'auto', 'cmap': cmap, 'norm': norm, 'rasterized': True}
    if ccrs is not None:
        options['transform'] = ccrs.PlateCarree()
    image = ax.pcolormesh(lon, lat, values, **options)
    if ccrs is not None:
        ax.set_extent([float(np.nanmin(lon)), float(np.nanmax(lon)),
                       float(np.nanmin(lat)), float(np.nanmax(lat))], crs=ccrs.PlateCarree())
        grid = ax.gridlines(draw_labels=True, linewidth=.25, color='.35', alpha=.45, linestyle=':')
        grid.top_labels = False
        grid.right_labels = False
        grid.bottom_labels = row == 3
        grid.left_labels = column == 0
    else:
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
    return image


def plot_case(output, entry, archive, ensemble, checkpoint, plot_step):
    truth = np.asarray(archive.array(entry, 'truth'))
    baseline = np.asarray(archive.array(entry, 'baseline'))
    generated = ensemble.mean(0)
    area = archive.static['area']
    lon = archive.static['lon'][::plot_step, ::plot_step]
    lat = archive.static['lat'][::plot_step, ::plot_step]
    fig, axes = map_axes(4, 4, lon, lat)
    metrics = {'id': entry['id'], 'time': entry['time'], 'members': len(ensemble),
               'checkpoint': str(checkpoint.resolve()), 'variables': {}}
    columns = ('Coarse baseline', 'Original HWT target',
               'Generated member' if len(ensemble) == 1 else 'Generated ensemble mean',
               'Generated − HWT')
    for row, (key, display) in enumerate(zip(TARGETS, DISPLAY)):
        title, unit, convert, cmap = display
        base_value, truth_value, generated_value = map(convert, (baseline[row], truth[row], generated[row]))
        pooled = np.concatenate([base_value.ravel(), truth_value.ravel(), generated_value.ravel()])
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
        fields = (base_value, truth_value, generated_value, error)
        for column, values in enumerate(fields):
            image = draw_map(axes[row, column], lon, lat, values[::plot_step, ::plot_step],
                             'RdBu_r' if column == 3 else cmap,
                             error_norm if column == 3 else norm, row, column)
            axes[row, column].set_title(f'{title}\n{columns[column]}', fontsize=10, weight='semibold')
            fig.colorbar(image, ax=axes[row, column], orientation='horizontal', pad=.035,
                         shrink=.82, label=unit)
        generated_score = continuous(convert(ensemble[:, row]), truth_value, area)
        baseline_score = continuous(convert(baseline[None, row]), truth_value, area)
        metrics['variables'][key] = {'unit': unit, 'generated': generated_score,
                                     'baseline': baseline_score,
                                     'rmse_skill_percent': 100*(1-generated_score['rmse']/baseline_score['rmse'])
                                     if baseline_score['rmse'] else None}
        axes[row, 2].text(.02, .02, f'RMSE {generated_score["rmse"]:.3g} {unit}',
                          transform=axes[row, 2].transAxes, fontsize=8,
                          bbox={'facecolor': 'white', 'alpha': .78, 'edgecolor': 'none'})
    fig.suptitle(f'MERRA21C-ML held-out test diagnostic · {entry["time"]} UTC\n'
                 f'{checkpoint.name} · {len(ensemble)} generated member(s)', fontsize=15, weight='bold')
    destination = output/f'test_{entry["id"]}.png'
    fig.savefig(destination, dpi=180)
    plt.close(fig)
    return metrics


def plot_summary(output, rows):
    labels = [row['id'][:8] for row in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, (key, display) in zip(axes.ravel(), zip(TARGETS, DISPLAY)):
        title, unit = display[:2]
        baseline = [row['variables'][key]['baseline']['rmse'] for row in rows]
        generated = [row['variables'][key]['generated']['rmse'] for row in rows]
        ax.bar(x-.19, baseline, width=.38, label='Coarse baseline', color='#8196a8')
        ax.bar(x+.19, generated, width=.38, label='Generated', color='#146c94')
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


def validate_reused_predictions(paths, checkpoint, cfg):
    expected_checkpoint = str(checkpoint.resolve())
    expected = {'checkpoint': expected_checkpoint, 'ode_steps': cfg['inference']['steps'],
                'patch_size': cfg['patch']['size'], 'patch_halo': cfg['patch']['halo'],
                'patch_stride': cfg['patch']['stride'],
                'dry_threshold_mm_h': cfg['inference']['dry_threshold']}
    for path in paths:
        with xr.open_dataset(path) as dataset:
            actual = {name: dataset.attrs.get(name) for name in expected}
        if actual != expected:
            raise ValueError(f'{path}: existing prediction does not match requested checkpoint/sampler')


def main():
    args = parse_args()
    if args.members < 1 or args.plot_step < 1:
        raise ValueError('members and plot-step must be positive')
    cfg = load_config(args.config)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f'Best checkpoint not found: {checkpoint}')
    archive = Archive(cfg['data']['prepared'])
    selected = select_entries(archive, args.samples, args.timestamps)
    output = Path(args.output or Path(cfg['train']['output'])/'test_best_model')
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
        validate_reused_predictions(paths, checkpoint, cfg)
        ensemble, _ = load_members(paths, archive, entry)
        report = plot_case(output, entry, archive, ensemble, checkpoint, args.plot_step)
        reports.append(report)
        print(f'Plotted held-out test case {entry["id"]}', flush=True)
    write_json(output/'metrics.json', {'selection': 'explicit' if args.timestamps else 'evenly spaced test timestamps',
                                      'samples': reports})
    plot_summary(output, reports)
    plot_training_history(output, Path(cfg['train']['output']))
    print(f'Best-model test diagnostics written to {output}', flush=True)


if __name__ == '__main__':
    main()
