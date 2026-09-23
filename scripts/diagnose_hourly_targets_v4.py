"""Audit the v3 hourly-mean rainfall targets reused by v4, and compare them with
the coarse GEOS input and the old HWT :30 snapshot target.

Read-only. Nothing in the archive or the hourly target directory is modified.

Three rainfall fields per hour (all mm/h, same 1059 x 1799 LCC grid):
  coarse    GEOS-FP PRECTOT regridded, an hourly mean over HH:00-HH+1:00 (v2 baseline)
  snapshot  HWT PRECTOT at HH:30, the v2 / direct-v2 training target
  hourly    1/4 P(HH:00) + 1/2 P(HH:30) + 1/4 P(HH+1:00), the v3 / v4 target

Outputs under --out:
  audit.json            per-split / per-month completeness, stray or broken hours
  summary.json          sampled statistics behind the figures
  summary.png           domain-mean agreement, wet area, diurnal cycle, intensity
                        distribution, spatial agreement at GEOS scale, monthly means
  mean_maps.png         sampled-period mean coarse / snapshot / hourly / difference
  case_<id>.png         wettest sampled hours: coarse, the three HWT snapshots, the
                        hourly target, differences, GEOS-scale block means, zoom

Examples (Discover, CPU node):
  python scripts/diagnose_hourly_targets_v4.py --audit-only
  python scripts/diagnose_hourly_targets_v4.py --verify --workers 8
  python scripts/diagnose_hourly_targets_v4.py --every 7 --cases 4 --time 2026-02-23T05:30:00
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))

from merraflow.dataset_v2 import ArchiveV2  # noqa: E402
from merraflow.dataset_v3_precip import HOURLY_FILE, HOURLY_INDEX  # noqa: E402
from merraflow.prepare_v3_precip import (PROVENANCE, TARGET_META, contract, digest,  # noqa: E402
                                         read_rate, snapshot_path, target_valid, verify_target)

SPLITS = ('train', 'val', 'test')
WET = .1  # mm/h
BINS = np.r_[0., np.geomspace(.01, 300., 57)]
COLORS = dict(coarse='#2a78d6', snapshot='#eb6834', hourly='#1baf7a')
LABELS = dict(coarse='GEOS coarse (hourly mean)', snapshot='HWT :30 snapshot (old target)',
              hourly='HWT trapezoid hourly (v3/v4 target)')
FIELDS = ('coarse', 'snapshot', 'hourly')


# ---------------------------------------------------------------- audit
def check_entry(root, entry, shape):
    folder = root/entry['id']
    if not folder.is_dir():
        return 'no_directory'
    if list(folder.glob('*.tmp*')):
        tmp = 'leftover_tmp'
    else:
        tmp = None
    if not (folder/HOURLY_FILE).exists():
        return tmp or 'no_target'
    if not (folder/TARGET_META).exists():
        return 'no_provenance'
    if not target_valid(folder/HOURLY_FILE, shape):
        return 'bad_shape_or_dtype'
    return tmp or 'ok'


def _verify_chunk(args):
    cfg, entries, checksum = args
    archive = ArchiveV2(cfg['data']['prepared'])
    provenance = contract(cfg, archive)
    failures = []
    for entry in entries:
        try:
            verify_target(cfg, archive, entry, provenance, checksum=checksum)
        except Exception as error:  # report every failure, never stop at the first
            failures.append(dict(id=entry['id'], error=str(error)))
    return failures


def audit(cfg, archive, verify=False, checksum=False, workers=1):
    root = Path(cfg['data']['hourly_targets'])
    entries = archive.index['entries']
    status = {e['id']: check_entry(root, e, archive.shape) for e in entries}
    known = {e['id'] for e in entries}
    stray = sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith(('_', '.'))
                   and p.name not in known)
    by_split = {s: Counter() for s in SPLITS}
    by_month = defaultdict(Counter)
    for e in entries:
        by_split.setdefault(e['split'], Counter())[status[e['id']]] += 1
        by_month[e['time'][:7]][status[e['id']]] += 1
    problems = [dict(id=e['id'], time=e['time'], split=e['split'], status=status[e['id']])
                for e in entries if status[e['id']] != 'ok']
    # Hours that prepare-hourly reported as lacking one of the three HWT snapshots.
    months_dir = root/'_months'
    reported_missing, month_records = [], {}
    if months_dir.is_dir():
        for path in sorted(months_dir.glob('*.json')):
            record = json.loads(path.read_text())
            month_records[record['month']] = dict(written=record['written'], skipped=record['skipped'],
                                                  missing=len(record['missing']))
            reported_missing += record['missing']
    months_without_record = sorted(set(by_month)-set(month_records)) if 'all' not in month_records else []
    provenance_path = root/PROVENANCE
    expected = contract(cfg, archive)
    provenance_ok = provenance_path.exists() and json.loads(provenance_path.read_text()) == expected
    index_path = root/HOURLY_INDEX
    index = None
    if index_path.exists():
        payload = json.loads(index_path.read_text())
        index = dict(present=True, completed=len(payload['completed']),
                     missing_hours=len(payload['missing_hours']),
                     completed_by_split=payload['completed_by_split'],
                     fingerprint_ok=payload.get('fingerprint') == digest({k: v for k, v in payload.items()
                                                                           if k != 'fingerprint'}),
                     matches_directory=sorted(payload['completed']) == sorted(i for i, s in status.items()
                                                                               if s in ('ok', 'leftover_tmp')))
    else:
        index = dict(present=False)
    failures = None
    if verify:
        candidates = [e for e in entries if status[e['id']] in ('ok', 'leftover_tmp')]
        chunks = [(cfg, candidates[i::workers], checksum) for i in range(workers)]
        with Pool(workers) as pool:
            failures = [f for part in pool.map(_verify_chunk, chunks) for f in part]
    result = dict(
        hourly_targets=str(root.resolve()), archive_hours=len(entries),
        status_counts=dict(Counter(status.values())),
        by_split={s: dict(c) for s, c in by_split.items()},
        by_month={m: dict(c) for m, c in sorted(by_month.items())},
        stray_directories=stray, provenance_matches_archive=provenance_ok,
        index=index, month_records=month_records, months_without_record=months_without_record,
        reported_missing_snapshots=reported_missing, problems=problems[:500],
        verification=None if failures is None else dict(checksum=checksum, failures=len(failures),
                                                        examples=failures[:50]))
    result['complete'] = (result['status_counts'].get('ok', 0) == len(entries) and provenance_ok
                          and not stray and (failures is None or not failures))
    return result, status


# ---------------------------------------------------------------- statistics
def block_mean(field, k):
    h, w = (field.shape[0]//k)*k, (field.shape[1]//k)*k
    return field[:h, :w].reshape(h//k, k, w//k, k).mean((1, 3))


def corr(a, b):
    a, b = a.ravel()-a.mean(), b.ravel()-b.mean()
    d = np.sqrt((a*a).sum()*(b*b).sum())
    return float((a*b).sum()/d) if d > 0 else float('nan')


def load_fields(archive, root, entry):
    coarse = np.maximum(np.asarray(archive.array(entry, 'baseline')[1], dtype='float32'), 0)
    snapshot = np.asarray(archive.array(entry, 'truth')[1], dtype='float32')
    hourly = np.asarray(np.load(root/entry['id']/HOURLY_FILE, mmap_mode='r')[0], dtype='float32')
    return dict(coarse=coarse, snapshot=snapshot, hourly=hourly)


def _stats_chunk(args):
    cfg, entries, block, map_block = args
    archive = ArchiveV2(cfg['data']['prepared'])
    root = Path(cfg['data']['hourly_targets'])
    area = archive.static['area'].astype('float64')
    weight = area/area.sum()
    rows = []
    hist = {k: np.zeros(len(BINS)-1) for k in FIELDS}
    sums = None
    for entry in entries:
        f = load_fields(archive, root, entry)
        blocks = {k: block_mean(v, block) for k, v in f.items()}
        row = dict(id=entry['id'], time=entry['time'], split=entry['split'])
        for k, v in f.items():
            row[f'{k}_mean'] = float((v*weight).sum())
            row[f'{k}_wet'] = float(((v >= WET)*weight).sum())
            row[f'{k}_p99'] = float(np.quantile(v[::4, ::4], .99))
            hist[k] += np.histogram(np.minimum(v, BINS[-1]-1e-3), BINS)[0]
        for k in ('snapshot', 'hourly'):
            row[f'{k}_block_corr'] = corr(blocks['coarse'], blocks[k])
            row[f'{k}_block_rmse'] = float(np.sqrt(((blocks['coarse']-blocks[k])**2).mean()))
        row['hourly_minus_snapshot_mae'] = float((np.abs(f['hourly']-f['snapshot'])*weight).sum())
        rows.append(row)
        maps = {k: block_mean(v, map_block).astype('float64') for k, v in f.items()}
        if sums is None:
            sums = maps
        else:
            for k in maps:
                sums[k] += maps[k]
    return rows, hist, sums


def statistics(cfg, entries, block, map_block, workers):
    chunks = [(cfg, entries[i::workers], block, map_block) for i in range(workers)]
    with Pool(workers) as pool:
        parts = pool.map(_stats_chunk, [c for c in chunks if c[1]])
    rows = sorted([r for p in parts for r in p[0]], key=lambda r: r['time'])
    hist = {k: sum(p[1][k] for p in parts) for k in FIELDS}
    maps = {k: sum(p[2][k] for p in parts if p[2] is not None)/len(rows) for k in FIELDS}
    return rows, hist, maps


def summarize(rows):
    out = {}
    for k in FIELDS:
        mean = np.array([r[f'{k}_mean'] for r in rows])
        wet = np.array([r[f'{k}_wet'] for r in rows])
        out[k] = dict(domain_mean_mm_h=float(mean.mean()), wet_fraction=float(wet.mean()),
                      p99_median_mm_h=float(np.median([r[f'{k}_p99'] for r in rows])))
    coarse_mean = np.array([r['coarse_mean'] for r in rows])
    coarse_wet = np.array([r['coarse_wet'] for r in rows])
    for k in ('snapshot', 'hourly'):
        mean = np.array([r[f'{k}_mean'] for r in rows])
        wet = np.array([r[f'{k}_wet'] for r in rows])
        c = np.array([r[f'{k}_block_corr'] for r in rows])
        out[k].update(domain_mean_corr_with_coarse=float(np.corrcoef(mean, coarse_mean)[0, 1]),
                      domain_mean_bias_vs_coarse=float((mean-coarse_mean).mean()),
                      wet_fraction_corr_with_coarse=float(np.corrcoef(wet, coarse_wet)[0, 1]),
                      block_corr_median=float(np.nanmedian(c)),
                      block_rmse_mean=float(np.mean([r[f'{k}_block_rmse'] for r in rows])))
    better = np.array([r['hourly_block_corr'] > r['snapshot_block_corr'] for r in rows
                       if np.isfinite(r['hourly_block_corr']) and np.isfinite(r['snapshot_block_corr'])])
    out['fraction_hours_hourly_closer_to_coarse_pattern'] = float(better.mean()) if len(better) else None
    out['hours_sampled'] = len(rows)
    return out


# ---------------------------------------------------------------- figures
def style(plt):
    plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
                         'grid.color': '#e4e3df', 'grid.linewidth': .6, 'axes.edgecolor': '#8a8984',
                         'axes.labelcolor': '#52514e', 'xtick.color': '#52514e', 'ytick.color': '#52514e',
                         'axes.titlesize': 11, 'axes.titleweight': 'semibold', 'font.size': 9,
                         'legend.frameon': False, 'lines.linewidth': 2, 'figure.facecolor': 'white'})


def rain_norm():
    from matplotlib import colors
    levels = [.1, .25, .5, 1, 2, 4, 8, 16, 32, 64]
    cmap = colors.LinearSegmentedColormap.from_list(
        'rain', ['#e3eefb', '#a9cbf2', '#5c9ee6', '#2a78d6', '#1c55a3', '#123a73', '#0b2447'], N=len(levels)-1)
    cmap.set_under('#f7f7f5')
    cmap.set_over('#05142a')
    return cmap, colors.BoundaryNorm(levels, cmap.N)


def diff_norm(limit):
    from matplotlib import colors
    cmap = colors.LinearSegmentedColormap.from_list('diff', ['#a0522d', '#e2b48f', '#ebebe8', '#8fc3d8', '#1f6f8b'])
    return cmap, colors.TwoSlopeNorm(0, -limit, limit)


def summary_figure(rows, hist, stats, block, path):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    style(plt)
    fig, ax = plt.subplots(2, 3, figsize=(17, 10), constrained_layout=True)
    coarse = np.array([r['coarse_mean'] for r in rows])
    top = max(coarse.max(), *(max(r[f'{k}_mean'] for r in rows) for k in ('snapshot', 'hourly')))*1.05
    a = ax[0, 0]
    for k in ('snapshot', 'hourly'):
        y = np.array([r[f'{k}_mean'] for r in rows])
        s = stats[k]
        a.scatter(coarse, y, s=9, alpha=.45, color=COLORS[k], linewidths=0,
                  label=f"{LABELS[k]}  r={s['domain_mean_corr_with_coarse']:.3f}, bias={s['domain_mean_bias_vs_coarse']:+.4f}")
    a.plot([0, top], [0, top], color='#8a8984', lw=1, ls='--')
    a.set(xlim=(0, top), ylim=(0, top), xlabel='GEOS coarse domain mean (mm/h)', ylabel='HWT domain mean (mm/h)',
          title='Domain-mean rain rate vs coarse input')
    a.legend(loc='upper left', fontsize=8, markerscale=2)

    a = ax[0, 1]
    cw = np.array([r['coarse_wet'] for r in rows])
    for k in ('snapshot', 'hourly'):
        y = np.array([r[f'{k}_wet'] for r in rows])
        a.scatter(cw, y, s=9, alpha=.45, color=COLORS[k], linewidths=0,
                  label=f"{LABELS[k]}  mean={stats[k]['wet_fraction']:.3f}")
    lim = max(cw.max(), *(max(r[f'{k}_wet'] for r in rows) for k in ('snapshot', 'hourly')))*1.05
    a.plot([0, lim], [0, lim], color='#8a8984', lw=1, ls='--')
    a.set(xlim=(0, lim), ylim=(0, lim), xlabel=f'GEOS wet-area fraction (>= {WET} mm/h)',
          ylabel='HWT wet-area fraction', title=f"Wet area (coarse mean={stats['coarse']['wet_fraction']:.3f})")
    a.legend(loc='upper left', fontsize=8, markerscale=2)

    a = ax[0, 2]
    for k in FIELDS:
        by_hour = defaultdict(list)
        for r in rows:
            by_hour[datetime.fromisoformat(r['time']).hour].append(r[f'{k}_mean'])
        hours = sorted(by_hour)
        a.plot(hours, [np.mean(by_hour[h]) for h in hours], color=COLORS[k], label=LABELS[k], marker='o', ms=4)
    a.set(xlabel='UTC hour (window midpoint HH:30)', ylabel='Mean domain rain rate (mm/h)', xticks=range(0, 24, 3),
          title='Diurnal cycle')
    a.legend(fontsize=8)

    a = ax[1, 0]
    centers = np.sqrt(BINS[1:-1]*BINS[2:])
    for k in FIELDS:
        counts = hist[k]
        exceed = counts[::-1].cumsum()[::-1]/counts.sum()
        a.loglog(BINS[1:-1], exceed[1:], color=COLORS[k], label=LABELS[k])
    a.set(xlabel='Rain rate threshold (mm/h)', ylabel='Fraction of pixels exceeding', xlim=(.05, 200),
          title='Intensity distribution (all sampled pixels)')
    a.legend(fontsize=8)
    del centers

    a = ax[1, 1]
    cs = np.array([r['snapshot_block_corr'] for r in rows])
    ch = np.array([r['hourly_block_corr'] for r in rows])
    ok = np.isfinite(cs) & np.isfinite(ch)
    a.scatter(cs[ok], ch[ok], s=9, alpha=.45, color=COLORS['hourly'], linewidths=0)
    a.plot([-.2, 1], [-.2, 1], color='#8a8984', lw=1, ls='--')
    frac = stats['fraction_hours_hourly_closer_to_coarse_pattern']
    a.set(xlim=(-.2, 1), ylim=(-.2, 1), xlabel='corr(coarse, :30 snapshot)', ylabel='corr(coarse, hourly target)',
          title=f'Pattern agreement at ~{block}x{block}-pixel GEOS scale')
    a.text(.03, .95, f"hourly closer in {100*frac:.0f}% of hours\nmedian r: snapshot {stats['snapshot']['block_corr_median']:.3f}, "
           f"hourly {stats['hourly']['block_corr_median']:.3f}", transform=a.transAxes, va='top', fontsize=8.5,
           color='#0b0b0b')

    a = ax[1, 2]
    for k in FIELDS:
        by_month = defaultdict(list)
        for r in rows:
            by_month[r['time'][:7]].append(r[f'{k}_mean'])
        months = sorted(by_month)
        a.plot(range(len(months)), [np.mean(by_month[m]) for m in months], color=COLORS[k], label=LABELS[k],
               marker='o', ms=4)
    a.set_xticks(range(len(months)), months, rotation=60, ha='right')
    a.set(ylabel='Mean domain rain rate (mm/h)', title='Monthly mean')
    a.legend(fontsize=8)
    fig.suptitle(f'Hourly rainfall targets vs snapshot and coarse input — {len(rows)} sampled hours', fontsize=13,
                 fontweight='semibold')
    fig.savefig(path, dpi=130)
    plt.close(fig)


def mean_maps_figure(maps, count, path):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt, colors
    style(plt)
    fig, ax = plt.subplots(1, 4, figsize=(22, 4.6), constrained_layout=True)
    hi = float(np.quantile(np.stack([maps[k] for k in FIELDS]), .995))
    for a, k in zip(ax, FIELDS):
        im = a.imshow(maps[k], origin='lower', cmap='Blues', norm=colors.Normalize(0, hi))
        a.set(title=f'{LABELS[k]}\nmean {maps[k].mean():.4f} mm/h', xticks=[], yticks=[])
        a.grid(False)
    fig.colorbar(im, ax=ax[:3].tolist(), label='Mean rain rate (mm/h)', shrink=.85)
    diff = maps['hourly']-maps['snapshot']
    lim = float(np.quantile(np.abs(diff), .995)) or 1e-6
    cmap, norm = diff_norm(lim)
    im = ax[3].imshow(diff, origin='lower', cmap=cmap, norm=norm)
    ax[3].set(title='Hourly − snapshot mean', xticks=[], yticks=[])
    ax[3].grid(False)
    fig.colorbar(im, ax=ax[3], label='mm/h', shrink=.85)
    fig.suptitle(f'Sampled-period mean rainfall ({count} hours)', fontsize=13, fontweight='semibold')
    fig.savefig(path, dpi=120)
    plt.close(fig)


def case_figure(archive, root, entry, block, zoom, path, highres_root=None):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt, patches
    style(plt)
    f = load_fields(archive, root, entry)
    middle = datetime.fromisoformat(entry['time'])
    snaps = {}
    for label, minutes in (('HH:00', -30), ('HH+1:00', 30)):
        t = middle+timedelta(minutes=minutes)
        snaps[label] = read_rate(snapshot_path(entry, t, highres_root), t)
    # Consistency check: recompute the stored target from the three snapshots.
    rebuilt = .25*snaps['HH:00']+.5*f['snapshot']+.25*snaps['HH+1:00']
    rebuild_error = float(np.abs(rebuilt-f['hourly']).max())
    # Zoom on the wettest zoom x zoom window of the coarse field.
    h, w = f['coarse'].shape
    score = block_mean(f['coarse'], zoom//2)
    iy, ix = np.unravel_index(np.argmax(score), score.shape)
    y0 = int(np.clip(iy*(zoom//2)-zoom//4, 0, h-zoom))
    x0 = int(np.clip(ix*(zoom//2)-zoom//4, 0, w-zoom))
    window = (slice(y0, y0+zoom), slice(x0, x0+zoom))
    cmap, norm = rain_norm()
    fig, ax = plt.subplots(3, 4, figsize=(22, 13.5), constrained_layout=True)
    panels = [
        (ax[0, 0], f['coarse'], 'GEOS coarse (hourly mean)'),
        (ax[0, 1], f['snapshot'], f'HWT {middle:%H:%M} snapshot (old target)'),
        (ax[0, 2], f['hourly'], 'HWT trapezoid hourly (v3/v4 target)'),
        (ax[1, 0], snaps['HH:00'], f"HWT {middle-timedelta(minutes=30):%H:%M} snapshot (weight 1/4)"),
        (ax[1, 1], snaps['HH+1:00'], f"HWT {middle+timedelta(minutes=30):%H:%M} snapshot (weight 1/4)"),
        (ax[2, 0], f['coarse'][window], 'Zoom: GEOS coarse'),
        (ax[2, 1], f['snapshot'][window], 'Zoom: :30 snapshot'),
        (ax[2, 2], f['hourly'][window], 'Zoom: hourly target'),
    ]
    for a, field, title in panels:
        im = a.imshow(field, origin='lower', cmap=cmap, norm=norm, interpolation='nearest')
        a.set(title=f'{title}\nmean {field.mean():.3f}, max {field.max():.1f} mm/h', xticks=[], yticks=[])
        a.grid(False)
    for a in (ax[0, 0], ax[0, 1], ax[0, 2]):
        a.add_patch(patches.Rectangle((x0, y0), zoom, zoom, fill=False, ec='#e34948', lw=1.5))
    fig.colorbar(im, ax=ax[:, :3].ravel().tolist(), label='Rain rate (mm/h)', shrink=.6, extend='both',
                 location='left')
    diff = f['hourly']-f['snapshot']
    lim = max(float(np.quantile(np.abs(diff), .999)), .1)
    dcmap, dnorm = diff_norm(lim)
    for a, field, title in ((ax[0, 3], diff, 'Hourly − snapshot'), (ax[2, 3], diff[window], 'Zoom: hourly − snapshot')):
        im2 = a.imshow(field, origin='lower', cmap=dcmap, norm=dnorm, interpolation='nearest')
        a.set(title=title, xticks=[], yticks=[])
        a.grid(False)
    fig.colorbar(im2, ax=[ax[0, 3], ax[2, 3]], label='mm/h', shrink=.6, extend='both')
    # GEOS-scale comparison: block means against the coarse field.
    a = ax[1, 2]
    c, s, hr = (block_mean(f[k], block).ravel() for k in FIELDS)
    top = max(c.max(), s.max(), hr.max(), .1)*1.05
    a.scatter(c, s, s=5, alpha=.35, color=COLORS['snapshot'], linewidths=0,
              label=f'snapshot r={corr(c, s):.3f}')
    a.scatter(c, hr, s=5, alpha=.35, color=COLORS['hourly'], linewidths=0,
              label=f'hourly r={corr(c, hr):.3f}')
    a.plot([0, top], [0, top], color='#8a8984', lw=1, ls='--')
    a.set(xlim=(0, top), ylim=(0, top), xlabel=f'GEOS coarse, {block}x{block} block mean (mm/h)',
          ylabel='HWT block mean (mm/h)', title='GEOS-scale agreement')
    a.legend(fontsize=8, markerscale=3)
    a = ax[1, 3]
    for k in FIELDS:
        counts = np.histogram(np.minimum(f[k], BINS[-1]-1e-3), BINS)[0]
        exceed = counts[::-1].cumsum()[::-1]/counts.sum()
        a.loglog(BINS[1:-1], np.maximum(exceed[1:], 1e-9), color=COLORS[k], label=LABELS[k])
    a.set(xlim=(.05, 200), ylim=(1e-7, 1), xlabel='Rain rate threshold (mm/h)', ylabel='Fraction exceeding',
          title='Intensity distribution, this hour')
    a.legend(fontsize=7.5)
    fig.suptitle(f"{entry['time']} ({entry['split']}): rebuilt-from-snapshots max error {rebuild_error:.2e} mm/h",
                 fontsize=13, fontweight='semibold')
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return rebuild_error


# ---------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', default='configs/discover_v4.yaml')
    parser.add_argument('--hourly-targets')
    parser.add_argument('--out', default='plots/hourly_targets_v4')
    parser.add_argument('--audit-only', action='store_true')
    parser.add_argument('--verify', action='store_true',
                        help='Also run the finalizer provenance check (source file identity) on every hour')
    parser.add_argument('--checksum', action='store_true', help='With --verify, also SHA-256 every target')
    parser.add_argument('--every', type=int, default=7,
                        help='Sample every Nth archive hour for statistics (7 is coprime with 24, so all UTC hours appear)')
    parser.add_argument('--block', type=int, default=9, help='Pixels per side for GEOS-scale block means (~27 km)')
    parser.add_argument('--map-block', type=int, default=3)
    parser.add_argument('--cases', type=int, default=4, help='Wettest sampled hours to plot (distinct days)')
    parser.add_argument('--time', action='append', default=[], help='Extra case hour, e.g. 2026-02-23T05:30:00')
    parser.add_argument('--zoom', type=int, default=256)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.hourly_targets:
        cfg['data']['hourly_targets'] = args.hourly_targets
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    archive = ArchiveV2(cfg['data']['prepared'])
    root = Path(cfg['data']['hourly_targets'])

    report, status = audit(cfg, archive, args.verify, args.checksum, args.workers)
    (out/'audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(f"Archive hours: {report['archive_hours']}; status: {report['status_counts']}")
    for split in SPLITS:
        print(f'  {split}: {report["by_split"].get(split, {})}')
    print(f"Provenance matches archive: {report['provenance_matches_archive']}")
    print(f"Stray directories: {len(report['stray_directories'])}; months without a prepare record: "
          f"{report['months_without_record']}; snapshots reported missing: {len(report['reported_missing_snapshots'])}")
    if report['verification'] is not None:
        print(f"Provenance verification failures: {report['verification']['failures']}")
    if report['index']['present']:
        print(f"Index present: {report['index']}")
    else:
        print(f'MISSING {HOURLY_INDEX}: training cannot use these targets until finalize-hourly runs:\n'
              '  python -m merraflow.cli_v3_precip finalize-hourly --config configs/discover_v3_precip.yaml')
    print('ALL HOURS PROCESSED' if report['complete'] else 'NOT COMPLETE: see audit.json problems', flush=True)
    if args.audit_only:
        return 0 if report['complete'] else 1

    usable = [e for e in archive.index['entries'] if status[e['id']] in ('ok', 'leftover_tmp')]
    sample = usable[::args.every]
    if not sample:
        raise SystemExit('No processed hours to sample')
    print(f'Statistics over {len(sample)} hours ...', flush=True)
    rows, hist, maps = statistics(cfg, sample, args.block, args.map_block, args.workers)
    stats = summarize(rows)
    summary_figure(rows, hist, stats, args.block, out/'summary.png')
    mean_maps_figure(maps, len(rows), out/'mean_maps.png')

    by_id = {e['id']: e for e in usable}
    chosen, days = [], set()
    for r in sorted(rows, key=lambda r: -r['coarse_mean']):
        if len(chosen) >= args.cases:
            break
        if r['time'][:10] not in days:
            chosen.append(by_id[r['id']])
            days.add(r['time'][:10])
    for text in args.time:
        match = [e for e in usable if np.datetime64(e['time']) == np.datetime64(text)]
        if not match:
            print(f'Requested --time {text} has no processed hourly target; skipped')
        chosen += match
    stats['cases'] = {}
    for entry in chosen:
        path = out/f"case_{entry['id']}.png"
        try:
            error = case_figure(archive, root, entry, args.block, args.zoom, path, cfg['data'].get('highres_root'))
        except (OSError, ValueError, KeyError) as problem:
            print(f"Case {entry['id']} skipped: {problem}")
            continue
        stats['cases'][entry['id']] = dict(time=entry['time'], split=entry['split'], rebuild_max_error_mm_h=error)
        print(f'Wrote {path} (rebuild error {error:.2e} mm/h)', flush=True)
    (out/'summary.json').write_text(json.dumps(dict(summary=stats, hours=rows), indent=2)+'\n')
    print(json.dumps({k: v for k, v in stats.items() if k != 'cases'}, indent=2))
    print(f'Figures in {out.resolve()}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
