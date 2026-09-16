#!/usr/bin/env python3
"""Generate independent v2 validation/test ensemble maps and physical scores."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, PowerNorm
import numpy as np
import torch

from merraflow.config import write_json
from merraflow.config_v2 import load_config_v2
from merraflow.dataset_v2 import ArchiveV2
from merraflow.evaluate_v2 import load_members_v2
from merraflow.inference_v2 import predict_v2
from merraflow.metrics import continuous, precipitation
from merraflow.physics_v2 import TARGETS_V2, UNITS_V2
from merraflow.train_v2 import file_hash_v2
from merraflow.noise_v2 import noise_padding_v2, NOISE_PADDING_MODES
from merraflow.inference import starts
from merraflow.precip_audit_v2 import deterministic_scores
from merraflow.metrics import weighted_mean


def select_entries(archive, split, count, timestamps, seed, include_date=None):
    entries = [e for e in archive.index['entries'] if e['split'] == split]
    if not entries:
        raise ValueError(f'No {split} entries in the v2 archive')
    if timestamps:
        if include_date:
            raise ValueError('--timestamps and --include-date cannot be combined')
        selected = []
        for stamp in timestamps:
            matches = [e for e in entries if stamp in (e['id'], e['time'])]
            if len(matches) != 1:
                raise ValueError(f'Expected one {split} entry for {stamp!r}, found {len(matches)}')
            selected.append(matches[0])
        if len({e['id'] for e in selected}) != len(selected):
            raise ValueError('Timestamps must be unique')
        return selected, {'strategy': 'explicit timestamps'}
    if not 1 <= count <= len(entries):
        raise ValueError(f'Choose 1..{len(entries)} {split} samples')
    event = None
    selection = {'strategy': 'seeded random', 'seed': seed}
    if include_date:
        candidates = [e for e in entries if e['time'].startswith(include_date+'T')]
        if not candidates:
            raise ValueError(f'No {split} entries on {include_date}')
        area = np.asarray(archive.static['area'], dtype='float64')
        area_total = area.sum()
        if area_total <= 0:
            raise ValueError('Archive has no positive grid area')
        def mean_rain(entry):
            rain = archive.array(entry, 'truth')[1]
            return float(np.sum(rain*area)/area_total)
        event = max(candidates, key=mean_rain)
        selection = {'strategy': 'highest area-weighted HWT precipitation on requested date, plus seeded random hours',
                     'seed': seed, 'event_date': include_date, 'event_id': event['id'],
                     'event_mean_precip_mm_h': mean_rain(event)}
    remaining = [e for e in entries if event is None or not e['time'].startswith(include_date+'T')]
    if count-int(event is not None) > len(remaining):
        raise ValueError(f'Not enough other {split} hours for {count} samples')
    rng = np.random.default_rng(seed)
    indices = sorted(rng.choice(len(remaining), size=count-int(event is not None), replace=False))
    return ([event] if event is not None else [])+[remaining[i] for i in indices], selection


def event_window(rain, fraction):
    h, w = rain.shape
    zh, zw = min(h, max(16, round(h*fraction))), min(w, max(16, round(w*fraction)))
    # A broad wet feature is more stable than the single largest rain pixel.
    step = max(1, min(h, w)//150)
    small = rain[:h//step*step, :w//step*step]
    small = small.reshape(small.shape[0]//step, step, small.shape[1]//step, step).mean((1, 3))
    cy, cx = np.unravel_index(np.argmax(small), small.shape)
    cy, cx = int((cy+.5)*step), int((cx+.5)*step)
    y0, x0 = int(np.clip(cy-zh//2, 0, h-zh)), int(np.clip(cx-zw//2, 0, w-zw))
    return slice(y0, y0+zh), slice(x0, x0+zw)


def scores_for_region(ensemble, regression, baseline, truth, area, region):
    ys, xs = region
    result = {}
    for i, name in enumerate(TARGETS_V2):
        actual, weights = truth[i, ys, xs], area[ys, xs]
        fields = {'baseline': baseline[None, i, ys, xs],
                  'regression': regression[None, i, ys, xs],
                  'member_0': ensemble[:1, i, ys, xs],
                  'ensemble_mean': ensemble[:, i, ys, xs]}
        result[name] = {key: continuous(value, actual, weights) for key, value in fields.items()}
    return result


def plot_maps(out, entry, ensemble, regression, baseline, truth, scores, region, suffix):
    mean = ensemble.mean(0)
    labels = ('LCC coarse baseline', 'Regression', 'Flow member 0',
              f'{len(ensemble)}-member mean', 'HWT reference', 'Ensemble mean − HWT')
    fig, axes = plt.subplots(5, 6, figsize=(24, 16), constrained_layout=True)
    try:
        for i, name in enumerate(TARGETS_V2):
            fields = (baseline[i], regression[i], ensemble[0, i], mean[i], truth[i])
            pool = np.stack(fields)
            lo, hi = np.quantile(pool, [.01, .99])
            if hi <= lo:
                hi = lo+1
            if name == 'precip':
                norm = PowerNorm(gamma=.45, vmin=0, vmax=max(float(np.quantile(pool, .995)), .1))
            else:
                norm = Normalize(lo, hi)
            difference = mean[i]-truth[i]
            bound = max(float(np.quantile(np.abs(difference[region]), .99)), 1e-4)
            for j, value in enumerate((*fields, difference)):
                ax = axes[i, j]
                image = ax.imshow(value[region], origin='lower', interpolation='nearest',
                                  cmap='RdBu_r' if j == 5 else ('YlGnBu' if i == 1 else 'viridis'),
                                  norm=Normalize(-bound, bound) if j == 5 else norm,
                                  rasterized=True)
                ax.set(title=f'{name} · {labels[j]}', xticks=[], yticks=[])
                key = ('baseline', 'regression', 'member_0', 'ensemble_mean', None, 'ensemble_mean')[j]
                if key:
                    rmse = scores[name][key]['rmse']
                    ax.text(.02, .03, f'RMSE {rmse:.3g} {UNITS_V2[i]}', transform=ax.transAxes,
                            fontsize=8, bbox={'facecolor': 'white', 'alpha': .8, 'edgecolor': 'none'})
                fig.colorbar(image, ax=ax, shrink=.64, label=UNITS_V2[i])
        fig.suptitle(f'V2 {entry["split"]} diagnostic · {entry["time"]} UTC · {len(ensemble)} flow members · {suffix or "full domain"}')
        fig.savefig(out/f'{entry["id"]}_{suffix or "full"}_v2.png', dpi=140)
    finally:
        plt.close(fig)


def plot_summary(out, reports, members, split):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    try:
        for ax, (i, name) in zip(axes.flat, enumerate(TARGETS_V2)):
            x = np.arange(len(reports))
            for offset, label, color in ((-.3, 'baseline', '#8b9aa8'),
                                         (-.1, 'regression', '#3b8c68'),
                                         (.1, 'member_0', '#d39a43'),
                                         (.3, 'ensemble_mean', '#2775a8')):
                ax.bar(x+offset, [r['full'][name][label]['rmse'] for r in reports],
                       width=.19, label=label, color=color)
            ax.set(title=name, ylabel=f'Area-weighted RMSE ({UNITS_V2[i]})',
                   xticks=x, xticklabels=[r['id'] for r in reports])
            ax.tick_params(axis='x', rotation=35)
            ax.grid(axis='y', alpha=.2)
        axes.flat[-1].axis('off')
        handles, labels = axes.flat[0].get_legend_handles_labels()
        axes.flat[-1].legend(handles, labels, loc='center')
        fig.suptitle(f'V2 {split}: {len(reports)} cases, {members} members per case')
        fig.savefig(out/'rmse_summary_v2.png', dpi=160)
    finally:
        plt.close(fig)


def plot_history(out, train_root):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    plotted = False
    for ax, stage in zip(axes, ('regression', 'flow')):
        path = train_root/f'{stage}_v2'/'history_v2.jsonl'
        if path.exists():
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            if rows:
                for label in ('train', 'val'):
                    ax.plot([r['epoch'] for r in rows], [r[label]['total'] for r in rows], label=label)
                ax.legend()
                plotted = True
        ax.set(title=f'{stage} objective', xlabel='Completed epoch', ylabel='Loss')
    if plotted:
        fig.savefig(out/'training_v2.png', dpi=160)
    plt.close(fig)


def boundary_affected_mask(shape, patch):
    """Union of cores whose input noise halo extends beyond the domain."""
    h, w = shape
    size, halo = patch['size'], patch['halo']
    affected = np.zeros(shape, dtype=bool)
    for y in starts(h, size, patch['stride']):
        for x in starts(w, size, patch['stride']):
            if y < halo or x < halo or y+size+halo > h or x+size+halo > w:
                affected[y:y+size, x:x+size] = True
    return affected


def padding_scores(ensemble, truth, area, patch):
    rain = ensemble[:, 1]
    pred = rain.mean(0)
    affected = boundary_affected_mask(truth.shape[-2:], patch)
    return {'mean_mm_h': weighted_mean(pred, area), 'HWT_mean_mm_h': weighted_mean(truth[1], area),
            'max_member_mm_h': float(rain.max()), 'max_ensemble_mean_mm_h': float(pred.max()),
            'HWT_max_mm_h': float(truth[1].max()),
            'regions': {name: {'area_fraction': weighted_mean(mask, area),
                               'scores': deterministic_scores(pred[mask], truth[1][mask], area[mask]) if mask.any() else None}
                        for name, mask in [('boundary_affected', affected), ('interior', ~affected)]}}


def write_padding_comparison(out, comparisons):
    old, new = comparisons['replicate'], comparisons['independent_halo']
    for key in ('checkpoint_sha256', 'members', 'ode_steps', 'inference_seed'):
        if old[key] != new[key]:
            raise ValueError(f'Uncontrolled padding comparison: {key} differs')
    if [r['id'] for r in old['samples']] != [r['id'] for r in new['samples']]:
        raise ValueError('Padding comparison timestamps differ')
    lines = ['# Noise-padding correction: controlled comparison', '',
             f'Checkpoint: `{old["checkpoint"]}`; {old["members"]} members; {old["ode_steps"]} steps.', '',
             'Every in-domain initial noise value is preserved. Only outside-domain noise padding differs.', '',
             '| Case | Padding | Rain RMSE | Rain bias | Largest member | Boundary-region RMSE | Interior RMSE |',
             '|---|---|---:|---:|---:|---:|---:|']
    for before, after in zip(old['samples'], new['samples']):
        if before['full']['precip']['regression'] != after['full']['precip']['regression']:
            raise ValueError('Regression changed in padding comparison')
        for mode, record in [('replicate', before), ('independent_halo', after)]:
            score, p = record['full']['precip']['ensemble_mean'], record['padding_check']
            def regional(name):
                scores = p['regions'][name]['scores']
                return f'{scores["rmse"]:.4g}' if scores else 'n/a'
            lines.append(f'| {record["id"]} | {mode} | {score["rmse"]:.4g} | {score["bias"]:.4g} | '
                         f'{p["max_member_mm_h"]:.4g} | {regional("boundary_affected")} | {regional("interior")} |')
    lines += ['', 'Rain units: mm/h. Boundary region is the union of cores whose halos cross the domain boundary.',
              'This correction does not change the model weights, normalization, conditioning, or patch evolution.',
              'Per-variable scores, precipitation calibration and full/zoom maps are saved separately under each padding directory.',
              'Improvement must be judged from these results; the software tests do not establish meteorological skill.']
    write_json(out/'padding_comparison_v2.json', comparisons)
    (out/'padding_comparison_v2.md').write_text('\n'.join(lines)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/discover_annual_v2.yaml')
    parser.add_argument('--checkpoint', help='Default: <train.output>/flow_v2/best_v2.pt')
    parser.add_argument('--output', help='Fresh output directory; default under <train.output>')
    parser.add_argument('--split', choices=('val', 'test'), default='test')
    parser.add_argument('--samples', type=int, default=3)
    parser.add_argument('--members', type=int, default=5)
    parser.add_argument('--timestamps', nargs='+', help='Exact IDs or ISO timestamps in the selected split')
    parser.add_argument('--include-date', help='Include the wettest HWT hour on this UTC date (YYYY-MM-DD), then sample remaining cases')
    parser.add_argument('--sample-seed', type=int, default=317)
    parser.add_argument('--zoom-fraction', type=float, default=.4)
    parser.add_argument('--noise-padding', choices=NOISE_PADDING_MODES, help='Default: independent_halo')
    parser.add_argument('--compare-noise-padding', action='store_true', help='Run legacy and corrected padding with identical in-domain noise')
    parser.add_argument('--steps', type=int, help='Override ODE steps for both sides of the comparison')
    args = parser.parse_args()
    if args.members < 2 or not 0 < args.zoom_fraction <= 1:
        parser.error('members must be >=2 and zoom-fraction must be in (0, 1]')
    cfg = load_config_v2(args.config)
    if args.steps is not None:
        if args.steps < 1:
            parser.error('--steps must be positive')
        cfg['inference']['steps'] = args.steps
    if args.noise_padding:
        if args.compare_noise_padding:
            parser.error('--noise-padding cannot be combined with --compare-noise-padding')
        cfg['inference']['noise_padding'] = args.noise_padding
    checkpoint = Path(args.checkpoint or Path(cfg['train']['output'])/'flow_v2'/'best_v2.pt')
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if ckpt['stage'] != 'flow':
        raise ValueError('Use a v2 flow checkpoint for this diagnostic')
    archive = ArchiveV2(cfg['data']['prepared'])
    selected, selection = select_entries(archive, args.split, args.samples, args.timestamps,
                                         args.sample_seed, args.include_date)
    digest = file_hash_v2(checkpoint)
    out = Path(args.output or Path(cfg['train']['output'])/f'test_best_model_{checkpoint.stem}_m{args.members}_{digest[:12]}_v2')
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'{out} is not empty; choose a fresh --output directory')
    out.mkdir(parents=True, exist_ok=True)
    cfg['inference']['members'] = args.members
    cfg['inference']['output'] = str(out/'predictions_v2')
    modes = ('replicate', 'independent_halo') if args.compare_noise_padding else (noise_padding_v2(cfg),)
    comparisons = {}
    for mode in modes:
        cfg['inference']['noise_padding'] = mode
        destination = out/f'{mode}_v2' if args.compare_noise_padding else out
        destination.mkdir(parents=True, exist_ok=True)
        cfg['inference']['output'] = str(destination/'predictions_v2')
        comparisons[mode] = run_selected(cfg, checkpoint, archive, destination, selected, selection, args, digest)
    if args.compare_noise_padding:
        write_padding_comparison(out, comparisons)
    print(f'V2 diagnostics written to {out}', flush=True)


def run_selected(cfg, checkpoint, archive, out, selected, selection, args, digest):
    reports = []
    for entry in selected:
        predict_v2(cfg, checkpoint, split=args.split, timestamp=entry['id'])
        paths = sorted((out/'predictions_v2').glob(f'{entry["id"]}_m*_v2.nc'))
        if len(paths) != args.members:
            raise RuntimeError(f'Incomplete diagnostic ensemble for {entry["id"]}')
        ensemble, regression, audits, identity = load_members_v2(paths, archive, entry)
        if identity[0] != digest:
            raise ValueError('Generated prediction checkpoint hash differs from requested checkpoint')
        truth = np.asarray(archive.array(entry, 'truth'))
        baseline = np.asarray(archive.array(entry, 'baseline'))
        region = event_window(truth[1], args.zoom_fraction)
        full = (slice(None), slice(None))
        full_scores = scores_for_region(ensemble, regression, baseline, truth, archive.static['area'], full)
        zoom_scores = scores_for_region(ensemble, regression, baseline, truth, archive.static['area'], region)
        plot_maps(out, entry, ensemble, regression, baseline, truth, full_scores, full, '')
        plot_maps(out, entry, ensemble, regression, baseline, truth, zoom_scores, region, 'zoom')
        reports.append({'id': entry['id'], 'time': entry['time'], 'split': args.split,
                        'padding_check': padding_scores(ensemble, truth, archive.static['area'], cfg['patch']),
                        'full': full_scores, 'zoom': zoom_scores,
                        'zoom_bounds': {'y': [region[0].start, region[0].stop],
                                        'x': [region[1].start, region[1].stop]},
                        'precipitation': precipitation(ensemble[:, 1], truth[1], archive.static['area']),
                        'budget_audit': audits})
        print(f'Wrote v2 diagnostic for {entry["id"]}', flush=True)
    metrics = {'version': 'v2', 'split': args.split,
               'checkpoint': str(checkpoint.resolve()),
               'checkpoint_sha256': digest, 'members': args.members,
               'selection': selection, 'samples': reports, 'noise_padding': noise_padding_v2(cfg),
               'ode_steps': cfg['inference']['steps'], 'inference_seed': cfg['inference']['seed']}
    write_json(out/'metrics_v2.json', metrics)
    plot_summary(out, reports, args.members, args.split)
    plot_history(out, Path(cfg['train']['output']))
    return metrics


if __name__ == '__main__':
    main()
