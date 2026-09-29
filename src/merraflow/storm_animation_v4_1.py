"""Hour-by-hour animation of a storm: GEOS-FP 0.25° input vs 2 km truth vs model members.

Takes --hours consecutive hours around a peak hour (--timestamp, e.g. from a documented
storm; default the wettest hour of the split). Two framing modes:

  --track (default)  a --zoom px window that follows the storm: centred each hour on the
                     truth's sea-level-pressure minimum (smoothed, and the track smoothed over
                     3 h); the low centre is marked on every panel
  --no-track         one fixed --region px window over the 24 h rain swath

For every hour and run (checkpoint) it samples --members members over that window (same
tiled sampler as inference; --post spectral applies the spectral fix), then writes one GIF
per variable:

  storm_precip.gif    rain rate (mm/h)
  storm_t2m.gif       2 m temperature (°C)
  storm_wind.gif      10 m wind speed (m/s)
  storm_slp.gif       sea-level pressure (hPa): truth and members reduced from surface pressure
                      with the 2 km elevation (standard hypsometric reduction with the member's
                      own 2 m temperature); the GEOS-FP panel shows GEOS-FP's own SLP

Panels: GEOS-FP native cells (each 0.25° cell mapped to its 2 km pixels: blocks, never
interpolated) | truth | member 1..N per run (one row per run). imshow(interpolation='nearest')
throughout: each grid point is a pixel. Colour scales are fixed over all hours (from truth).

Member noise: by default member k uses the SAME noise seed every hour (--seeds fixed), so the
animation does not flicker from re-drawn noise; --seeds independent draws fresh noise each hour.

Outputs (default <first run's train.output>/evaluation/storm_<peak>_<job>/):
  storm_<var>.gif, peak_<var>.png (peak hour still), track.png, frames/ (optional), storm.json
"""
import argparse
from datetime import datetime, timedelta
import gc
import os
from pathlib import Path
import time
import numpy as np
import torch
from scipy.ndimage import gaussian_filter

from .config import write_json
from .train import device_for
from .v4 import TARGETS
from .v4_1 import load_config
from .evaluate_v4_1 import (resolve_checkpoint, load_model, select_cases, member_seed, event_window,
                            native_fields, _rain_norm, DISPLAY)
from .explore_inference_v4_1 import RegionEngine, DEFAULTS, climatology, spectral_fix
from .compare_checkpoints_v4_1 import parse_runs

VARIABLES = ('precip', 't2m', 'wind', 'slp')
DEFAULT_RUNS = ('main_latest=configs/discover_v4_1.yaml:latest',)
G, RD, LAPSE = 9.80665, 287.05, .0065
STYLE = {'precip': ('Hourly rain rate', 'mm h⁻¹'), 't2m': ('2 m temperature', '°C'),
         'wind': ('10 m wind speed', 'm s⁻¹'), 'slp': ('Sea-level pressure', 'hPa')}


def _time(entry):
    return datetime.fromisoformat(entry['time'][:19])


def storm_hours(archive, split, timestamp=None, hours=24, log=print):
    """Consecutive hourly entries around the peak (``timestamp`` or the wettest hour)."""
    entries = sorted(archive.eligible(split), key=_time)
    peak = select_cases(archive, split, [timestamp] if timestamp else None, 0, 0 if timestamp else 1, 317, log)[0]['entry']
    i = next(k for k, e in enumerate(entries) if e['id'] == peak['id'])
    lo = hi = i   # grow the run of consecutive hours around the peak
    while hi-lo+1 < hours:
        grew = False
        if lo > 0 and _time(entries[lo])-_time(entries[lo-1]) == timedelta(hours=1) and (i-lo) <= (hi-i):
            lo, grew = lo-1, True
        elif hi+1 < len(entries) and _time(entries[hi+1])-_time(entries[hi]) == timedelta(hours=1):
            hi, grew = hi+1, True
        elif lo > 0 and _time(entries[lo])-_time(entries[lo-1]) == timedelta(hours=1):
            lo, grew = lo-1, True
        if not grew:
            break
    window = entries[lo:hi+1]
    if len(window) < hours:
        log(f'Only {len(window)} consecutive eligible hours around {peak["time"]} (asked for {hours})')
    return window, peak


def sea_level_pressure(ps, t2m, elevation):
    """Hypsometric reduction to sea level (Pa) with the mean temperature of a standard-lapse column."""
    z = np.asarray(elevation, dtype='float64')
    mean_t = np.asarray(t2m, dtype='float64')+LAPSE*z/2
    return (np.asarray(ps, dtype='float64')*np.exp(G*z/(RD*mean_t))).astype('float32')


def storm_track(archive, window, elevation, zoom, smooth=12., stride=2, log=print):
    """Per hour: centre (row, col) of the truth SLP minimum (smoothed field, 3-hour running mean
    of the positions), its value (hPa), and whether it sits at the domain edge (storm outside)."""
    h, w = archive.shape
    points = []
    for e in window:
        t = archive.physical_truth(e)
        slp = sea_level_pressure(t[TARGETS.index('ps'), ::stride, ::stride], t[TARGETS.index('t2m'), ::stride, ::stride],
                                 elevation[::stride, ::stride])
        s = gaussian_filter(slp, smooth/stride, mode='nearest')
        y, x = np.unravel_index(np.argmin(s), s.shape)
        points.append((y*stride, x*stride, float(s[y, x])/100.))
    rows = np.array([p[0] for p in points], dtype='float64')
    cols = np.array([p[1] for p in points], dtype='float64')
    if len(points) >= 3:   # 3-hour running mean, ends kept
        rows[1:-1] = (rows[:-2]+rows[1:-1]+rows[2:])/3
        cols[1:-1] = (cols[:-2]+cols[1:-1]+cols[2:])/3
    edge = max(4, int(3*smooth))
    track = []
    for (y, x, value), ry, rx in zip(points, rows, cols):
        at_edge = y < edge or x < edge or y >= h-edge or x >= w-edge
        r0 = int(np.clip(round(ry)-zoom//2, 0, max(0, h-zoom)))
        c0 = int(np.clip(round(rx)-zoom//2, 0, max(0, w-zoom)))
        track.append(dict(center=(int(y), int(x)), slp_hpa=value, at_edge=bool(at_edge),
                          region=(r0, min(h, r0+zoom), c0, min(w, c0+zoom))))
    return track


def rain_region(archive, window, size):
    """Square region covering the largest accumulated rain over the hours."""
    total = sum(np.asarray(archive.truth_field(e)[0], dtype='float64') for e in window)
    rows, cols = event_window(total, size)
    return rows.start, rows.stop, cols.start, cols.stop


def native_slp(entry, lat, lon):
    """GEOS-FP's own sea-level pressure (Pa) on its native grid, or None."""
    import xarray as xr
    if not entry.get('native'):
        return None
    flx = Path(entry['native'])
    path = flx.with_name(flx.name.replace('tavg1_2d_flx_Nx.', 'tavg1_2d_slv_Nx.'))
    if not path.is_file():
        return None
    with xr.open_dataset(path) as ds:
        if 'SLP' not in ds:
            return None
        ds = ds.assign_coords(lon=((ds.lon+180) % 360)-180).sortby('lon').sortby('lat')
        ds = ds.sel(lat=slice(float(np.nanmin(lat))-2, float(np.nanmax(lat))+2),
                    lon=slice(float(np.nanmin(lon))-2, float(np.nanmax(lon))+2))
        field = ds['SLP']
        for dim in [d for d in field.dims if d not in ('lat', 'lon')]:
            field = field.isel({dim: 0})
        glat, glon = np.asarray(ds.lat.values, dtype='float64'), np.asarray(ds.lon.values, dtype='float64')
        if glat.size < 2 or glon.size < 2:
            return None
        return dict(values=np.asarray(field.transpose('lat', 'lon').values, dtype='float32'), lat=glat, lon=glon,
                    dlat=float(np.median(np.diff(glat))), dlon=float(np.median(np.diff(glon))))


def _on_grid(f, lat, lon):
    iy = np.clip(np.rint((lat-f['lat'][0])/f['dlat']).astype(int), 0, len(f['lat'])-1)
    ix = np.clip(np.rint((lon-f['lon'][0])/f['dlon']).astype(int), 0, len(f['lon'])-1)
    return f['values'][iy, ix]


def native_on_grid(entry, lat, lon, log=print):
    """GEOS-FP native fields mapped to the fine grid by the containing cell (blocky, no
    interpolation). Returns {name: (R, C) array} for what is available."""
    fields, _ = native_fields(entry, lat, lon, log=log)
    out = {name: _on_grid(f, lat, lon) for name, f in fields.items()}
    try:
        slp = native_slp(entry, lat, lon)
    except Exception:  # noqa: BLE001 - optional panel content
        slp = None
    if slp is not None:
        out['slp'] = _on_grid(slp, lat, lon)
    return out


def display(var, fields, elevation):
    """(6, ...) physical fields -> the plotted quantity."""
    if var == 'precip':
        return fields[TARGETS.index('precip')]
    if var == 't2m':
        return fields[TARGETS.index('t2m')]-273.15
    if var == 'wind':
        return np.hypot(fields[TARGETS.index('u10m')], fields[TARGETS.index('v10m')])
    return sea_level_pressure(fields[TARGETS.index('ps')], fields[TARGETS.index('t2m')], elevation)/100.


def native_display(var, native, coarse, elevation):
    """Same quantity from the native GEOS-FP cells (SLP: GEOS-FP's own), else the regridded input."""
    def get(name):
        return native[name] if name in native else coarse[TARGETS.index(name)]
    if var == 'precip':
        return get('precip')
    if var == 't2m':
        return get('t2m')-273.15
    if var == 'wind':
        return np.hypot(get('u10m'), get('v10m'))
    if 'slp' in native:
        return native['slp']/100.
    return sea_level_pressure(coarse[TARGETS.index('ps')], coarse[TARGETS.index('t2m')], elevation)/100.


def run(runs, timestamp=None, split='test', hours=24, members=4, steps=64, region=768, margin=64, post='spectral',
        seeds='fixed', weights='ema', output=None, batch=32, threads=8, dpi=90, fps=2., frames=False, track=True,
        zoom=512, log=print, configs=None):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    started = time.monotonic()
    load = (lambda path: configs[path]) if configs else load_config
    job = os.environ.get('SLURM_JOB_ID') or time.strftime('%Y%m%d_%H%M%S')
    first_cfg = load(runs[0][1])
    device = device_for(first_cfg['train']['device'])
    archive = load_model(first_cfg, resolve_checkpoint(first_cfg, runs[0][2]), device, weights)[0]
    elevation_full = np.asarray(archive.static['elevation'], dtype='float64')
    lat_full = np.asarray(archive.static['lat'], dtype='float64')
    lon_full = np.asarray(archive.static['lon'], dtype='float64')
    window, peak = storm_hours(archive, split, timestamp, hours, log)
    h, w = archive.shape
    if track:
        size = min(zoom, h, w)
        path_ = storm_track(archive, window, elevation_full, size, log=log)
        regions = [p['region'] for p in path_]
    else:
        box = rain_region(archive, window, min(region, h, w))
        path_ = None
        regions = [box]*len(window)
    out = Path(output) if output else (Path(first_cfg['train']['output'])/'evaluation'/f'storm_{peak["id"]}_{job}')
    out.mkdir(parents=True, exist_ok=True)
    dx_km = float(np.sqrt(np.median(np.asarray(archive.static['area'], dtype='float64')))/1000)
    r0, r1, c0, c1 = regions[0]
    log(f'Storm peak {peak["time"]} ({split}); {len(window)} hours {window[0]["time"][:16]} → {window[-1]["time"][:16]}; '
        f'{"tracking the SLP minimum with" if track else "fixed"} {(r1-r0)*dx_km:.0f}×{(c1-c0)*dx_km:.0f} km window; '
        f'{len(runs)} run(s) × {members} members; seeds {seeds}; post {post}')
    if path_:
        for e, p in zip(window, path_):
            y, x = p['center']
            log(f'  {e["time"][:16]}: low {p["slp_hpa"]:.1f} hPa at {lat_full[y, x]:.2f}°N {lon_full[y, x]:.2f}°E'
                + ('  ⚠ at the domain edge: the low is probably outside the domain' if p['at_edge'] else ''))

    truth, native, found, centers = [], [], [], []
    for e, (a, b, c, d) in zip(window, regions):
        rows, cols = slice(a, b), slice(c, d)
        elev, lat, lon = elevation_full[rows, cols], lat_full[rows, cols], lon_full[rows, cols]
        t = np.asarray(archive.physical_truth(e), dtype='float32')[:, rows, cols]
        coarse = np.asarray(archive.coarse(e), dtype='float32')[:, rows, cols]
        nat = native_on_grid(e, lat, lon, log)
        found.append(all(k in nat for k in ('precip', 't2m', 'u10m', 'v10m')))
        truth.append({v: display(v, t, elev) for v in VARIABLES})
        native.append({v: native_display(v, nat, coarse, elev) for v in VARIABLES})
    native_label = ('GEOS-FP 0.25° cells' if all(found) else
                    'GEOS-FP 0.25° cells (some hours: regridded input)' if any(found) else 'GEOS-FP (regridded input)')
    if path_:
        centers = [(p['center'][0]-reg[0], p['center'][1]-reg[2]) for p, reg in zip(path_, regions)]

    members_by_run = {}
    for label, config, checkpoint in runs:
        cfg = load(config)
        ckpt = resolve_checkpoint(cfg, checkpoint)
        archive_r, model, conditioner, saved = load_model(cfg, ckpt, device, weights)
        spec = dict(DEFAULTS, id='baseline', steps=steps)
        peak_region = regions[next((k for k, e in enumerate(window) if e['id'] == peak['id']), 0)]
        clim = climatology(archive_r, peak, peak_region, log=log) if post == 'spectral' else None
        frames_run = []
        for k, (e, reg) in enumerate(zip(window, regions)):
            t0 = time.monotonic()
            engine = RegionEngine(model, conditioner, archive_r, e, cfg, device, reg, margin, batch, threads)
            elev = elevation_full[reg[0]:reg[1], reg[2]:reg[3]]
            anchor = peak if seeds == 'fixed' else e
            hour = []
            for m in range(members):
                fields = engine.sample(member_seed(cfg, anchor, m), spec)
                if clim is not None:
                    fields = spectral_fix(fields, clim, dx_km)
                hour.append({v: display(v, fields, elev) for v in VARIABLES})
            frames_run.append(hour)
            log(f'  {label} (epoch {saved["epoch"]+1}): hour {k+1}/{len(window)} {e["time"][:16]} · '
                f'{time.monotonic()-t0:.0f}s')
            del engine
        members_by_run[f'{label} (ep {saved["epoch"]+1}{", spectral" if post == "spectral" else ""})'] = frames_run
        del model, conditioner
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    info = dict(peak=peak['time'], split=split, hours=[e['time'] for e in window], regions=[list(r) for r in regions],
                grid_km=dx_km, runs=[dict(label=l, config=c, checkpoint=k) for l, c, k in runs], members=members,
                steps=steps, post=post, seeds=seeds, native=native_label, tracked=bool(track),
                track=[dict(time=e['time'], center=list(p['center']), lat=float(lat_full[p['center']]),
                            lon=float(lon_full[p['center']]), slp_hpa=p['slp_hpa'], at_edge=p['at_edge'])
                       for e, p in zip(window, path_)] if path_ else None)
    for var in VARIABLES:
        animate(var, window, truth, native, members_by_run, members, native_label, out, dpi, fps, frames, plt, info,
                centers)
    if path_:
        plot_track(info, out, dpi, plt)
    write_json(out/'storm.json', info)
    log(f'Storm animations written to {out} ({time.monotonic()-started:.0f}s)')
    return out


def _scale(var, truth):
    stack = np.stack([t[var] for t in truth])
    if var == 'precip':
        cmap, norm = _rain_norm()
        return dict(cmap=cmap, norm=norm)
    if var == 'slp':
        lo, hi = np.quantile(stack, [.005, .995])
        return dict(cmap='viridis', vmin=float(lo), vmax=float(hi))
    lo, hi = np.quantile(stack, [.005, .995])
    return dict(cmap=DISPLAY['t2m'][4] if var == 't2m' else 'magma', vmin=float(lo), vmax=float(hi))


def animate(var, window, truth, native, members_by_run, members, native_label, out, dpi, fps, frames, plt, info,
            centers=()):
    from matplotlib import animation
    runs = list(members_by_run)
    cols = 2+members
    rows = len(runs)
    h, w = truth[0][var].shape
    panel = 3.1
    fig, axes = plt.subplots(rows, cols, figsize=(panel*cols*w/max(h, w)+1.2, panel*rows*h/max(h, w)+1.1),
                             squeeze=False, constrained_layout=True)
    style = _scale(var, truth)
    images, markers = [], []
    for r, run in enumerate(runs):
        sources = [('native', native_label), ('truth', 'Truth 2 km')] if r == 0 else [None, None]
        sources += [(('member', run, m), f'{run}\nmember {m+1}') for m in range(members)]
        for c, src in enumerate(sources):
            ax = axes[r, c]
            ax.set_xticks([])
            ax.set_yticks([])
            if src is None:
                ax.set_visible(False)
                continue
            key, title = src
            images.append((key, ax.imshow(_frame(key, 0, var, truth, native, members_by_run), origin='lower',
                                          interpolation='nearest', **style)))
            if centers:
                markers.append(ax.plot([], [], marker='x', color='w', mec='k', ms=9, mew=2.2)[0])
            ax.set_title(title, fontsize=8)
    title, unit = STYLE[var]
    fig.colorbar(images[-1][1], ax=axes.ravel().tolist(), shrink=.8, pad=.01, label=f'{title} ({unit})')
    heading = fig.suptitle('', fontsize=10)

    def update(k):
        for key, image in images:
            image.set_data(_frame(key, k, var, truth, native, members_by_run))
        for marker in markers:
            marker.set_data([centers[k][1]], [centers[k][0]])
        low = ''
        if info.get('track'):
            p = info['track'][k]
            low = f' · truth low {p["slp_hpa"]:.0f} hPa (×){" at domain edge" if p["at_edge"] else ""}'
        heading.set_text(f'{title} · {window[k]["time"][:16].replace("T", " ")} UTC · hour {k+1}/{len(window)} · '
                         f'peak {info["peak"][:16].replace("T", " ")}{low}')
        return [image for _, image in images]+markers+[heading]
    peak_index = next((k for k, e in enumerate(window) if e['time'] == info['peak']), 0)
    update(peak_index)
    fig.savefig(out/f'peak_{var}.png', dpi=dpi*2)
    if frames:
        (out/'frames').mkdir(exist_ok=True)
        for k in range(len(window)):
            update(k)
            fig.savefig(out/'frames'/f'{var}_{k+1:02d}.png', dpi=dpi)
    anim = animation.FuncAnimation(fig, update, frames=len(window), blit=False)
    anim.save(out/f'storm_{var}.gif', writer=animation.PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)


def plot_track(info, out, dpi, plt):
    """Track of the truth low (lat/lon) with its central pressure."""
    t = info['track']
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    sc = a.scatter([p['lon'] for p in t], [p['lat'] for p in t], c=[p['slp_hpa'] for p in t], cmap='viridis', s=40)
    a.plot([p['lon'] for p in t], [p['lat'] for p in t], color='k', lw=.8)
    for p in t[::max(1, len(t)//6)]:
        a.annotate(p['time'][5:13].replace('T', ' '), (p['lon'], p['lat']), fontsize=7)
    fig.colorbar(sc, ax=a, label='Truth SLP minimum (hPa)')
    a.set_xlabel('Longitude (°E)')
    a.set_ylabel('Latitude (°N)')
    a.set_title('Track of the truth sea-level-pressure minimum', fontsize=10)
    b.plot(range(len(t)), [p['slp_hpa'] for p in t], marker='o', color='#2166ac')
    for k, p in enumerate(t):
        if p['at_edge']:
            b.plot(k, p['slp_hpa'], marker='o', color='#d73027')
    b.set_xticks(range(0, len(t), max(1, len(t)//8)), [t[k]['time'][5:13].replace('T', ' ')
                                                        for k in range(0, len(t), max(1, len(t)//8))], fontsize=7)
    b.set_ylabel('hPa')
    b.set_title('Central pressure in the domain (red = at domain edge)', fontsize=10)
    b.grid(True, ls='--', alpha=.4)
    fig.savefig(out/'track.png', dpi=dpi*2)
    plt.close(fig)


def _frame(key, k, var, truth, native, members_by_run):
    if key == 'native':
        return native[k][var]
    if key == 'truth':
        return truth[k][var]
    _, run, m = key
    return members_by_run[run][k][m][var]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--runs', nargs='+', default=list(DEFAULT_RUNS), help='label=config:checkpoint ...')
    parser.add_argument('--timestamp', help='Peak hour, e.g. 20260223_1730 (default: wettest hour of the split)')
    parser.add_argument('--split', choices=('val', 'test'), default='test')
    parser.add_argument('--hours', type=int, default=24)
    parser.add_argument('--members', type=int, default=4)
    parser.add_argument('--steps', type=int, default=64)
    parser.add_argument('--no-track', action='store_true', help='Fixed window over the rain swath instead of tracking')
    parser.add_argument('--zoom', type=int, default=512, help='Tracking window size in px (default 512 ≈ 965 km)')
    parser.add_argument('--region', type=int, default=768, help='Fixed window size in px with --no-track')
    parser.add_argument('--post', choices=('none', 'spectral'), default='spectral')
    parser.add_argument('--seeds', choices=('fixed', 'independent'), default='fixed')
    parser.add_argument('--weights', choices=('ema', 'raw'), default='ema')
    parser.add_argument('--fps', type=float, default=2.)
    parser.add_argument('--frames', action='store_true', help='Also write every frame as PNG')
    parser.add_argument('--output')
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--dpi', type=int, default=90)
    a = parser.parse_args()
    run(parse_runs(a.runs), a.timestamp, a.split, a.hours, a.members, a.steps, a.region, post=a.post, seeds=a.seeds,
        weights=a.weights, output=a.output, batch=a.batch, threads=a.threads, dpi=a.dpi, fps=a.fps, frames=a.frames,
        track=not a.no_track, zoom=a.zoom, log=lambda message: print(message, flush=True))


if __name__ == '__main__':
    main()
