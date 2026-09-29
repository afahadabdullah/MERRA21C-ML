"""Hour-by-hour animation of a storm: GEOS-FP 0.25° input vs 2 km truth vs model members.

Takes --hours consecutive hours around a peak hour (--timestamp, e.g. from a documented
storm; default the wettest hour of the split). Two framing modes:

  --track (default)  a --zoom px window that follows the storm: each hour it is centred on
                     the truth pressure low, shifted north by --shift-north x window size
                     (track smoothed over 3 h); the low centre is marked × on every panel.
                     The low is located on terrain-reduced (sea-level) truth pressure, used
                     ONLY to place the window: raw surface pressure has its minimum on the
                     highest terrain. Everything plotted is the model's own variables.
  --no-track         one fixed --region px window over the 24 h rain swath

For every hour and run (checkpoint) it samples --members members over that window (same
tiled sampler as inference; --post spectral applies the spectral fix) and writes:

  storm_precip.gif    rain rate (mm/h)
  storm_t2m.gif       2 m temperature (°C)
  storm_wind.gif      10 m wind speed (m/s)
  storm_ps.gif        surface pressure (hPa), the model's predicted variable
  storm_fields.nc     every hour's fields (NetCDF): truth, GEOS-FP cells on the 2 km grid,
                      and members (run x member), all six targets in physical units, with the
                      window's lat/lon/elevation and the track

Panels: GEOS-FP native cells (each 0.25° cell mapped to its 2 km pixels: blocks, never
interpolated) | truth | member 1..N per run (one row per run), on the native LCC grid with
Cartopy coastlines, borders and states. imshow(interpolation='nearest') throughout: each grid
point is a pixel. Colour scales are fixed over all hours (from truth).

Member noise: by default member k uses the SAME noise seed every hour (--seeds fixed), so the
animation does not flicker from re-drawn noise; --seeds independent draws fresh noise each hour.

Outputs (default <first run's train.output>/evaluation/storm_<peak>_<job>/):
  storm_<var>.gif, peak_<var>.png, storm_fields.nc, track.png, frames/ (optional), storm.json
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
from .v4 import TARGETS, UNITS
from .v4_1 import load_config
from .evaluate_v4_1 import (resolve_checkpoint, load_model, select_cases, member_seed, event_window,
                            native_fields, _rain_norm, DISPLAY, Canvas)
from .explore_inference_v4_1 import RegionEngine, DEFAULTS, climatology, spectral_fix
from .compare_checkpoints_v4_1 import parse_runs

VARIABLES = ('precip', 't2m', 'wind', 'ps')
DEFAULT_RUNS = ('main_latest=configs/discover_v4_1.yaml:latest',)
G, RD, LAPSE = 9.80665, 287.05, .0065
STYLE = {'precip': ('Hourly rain rate', 'mm h⁻¹'), 't2m': ('2 m temperature', '°C'),
         'wind': ('10 m wind speed', 'm s⁻¹'), 'ps': ('Surface pressure', 'hPa')}


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
    """Hypsometric reduction to sea level (Pa); used only to locate the low for tracking."""
    z = np.asarray(elevation, dtype='float64')
    mean_t = np.asarray(t2m, dtype='float64')+LAPSE*z/2
    return (np.asarray(ps, dtype='float64')*np.exp(G*z/(RD*mean_t))).astype('float32')


def storm_track(archive, window, elevation, zoom, shift_north=0., smooth=12., stride=2):
    """Per hour: the truth pressure-low centre (row, col; terrain-reduced for locating only,
    smoothed, 3-hour running mean of positions), its reduced value (hPa), whether it sits at the
    domain edge, and the zoom window centred on it and shifted north by shift_north*zoom rows."""
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
        cy = ry+shift_north*zoom   # rows increase northward (maps are drawn with origin='lower')
        r0 = int(np.clip(round(cy)-zoom//2, 0, max(0, h-zoom)))
        c0 = int(np.clip(round(rx)-zoom//2, 0, max(0, w-zoom)))
        track.append(dict(center=(int(y), int(x)), low_hpa=value, at_edge=bool(at_edge),
                          region=(r0, min(h, r0+zoom), c0, min(w, c0+zoom))))
    return track


def rain_region(archive, window, size):
    """Square region covering the largest accumulated rain over the hours."""
    total = sum(np.asarray(archive.truth_field(e)[0], dtype='float64') for e in window)
    rows, cols = event_window(total, size)
    return rows.start, rows.stop, cols.start, cols.stop


def native_on_grid(entry, lat, lon, log=print):
    """GEOS-FP native fields mapped to the fine grid by the containing cell (blocky, no
    interpolation). Returns {name: (R, C) array} for what is available."""
    fields, _ = native_fields(entry, lat, lon, log=log)
    out = {}
    for name, f in fields.items():
        iy = np.clip(np.rint((lat-f['lat'][0])/f['dlat']).astype(int), 0, len(f['lat'])-1)
        ix = np.clip(np.rint((lon-f['lon'][0])/f['dlon']).astype(int), 0, len(f['lon'])-1)
        out[name] = f['values'][iy, ix]
    return out


def display(var, fields):
    """(6, ...) physical fields -> the plotted quantity."""
    if var == 'precip':
        return fields[TARGETS.index('precip')]
    if var == 't2m':
        return fields[TARGETS.index('t2m')]-273.15
    if var == 'wind':
        return np.hypot(fields[TARGETS.index('u10m')], fields[TARGETS.index('v10m')])
    return fields[TARGETS.index('ps')]/100.


def geosfp_fields(native, coarse):
    """(6, R, C) GEOS-FP fields on the 2 km grid: native cells where available, else regridded input."""
    return np.stack([native[name] if name in native else coarse[c] for c, name in enumerate(TARGETS)]).astype('float32')


def run(runs, timestamp=None, split='test', hours=24, members=2, steps=64, region=768, margin=64, post='spectral',
        seeds='fixed', weights='ema', output=None, batch=32, threads=8, dpi=90, fps=2., frames=False, track=True,
        zoom=394, shift_north=.25, maps=True, save_netcdf=True, log=print, configs=None):
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
        path_ = storm_track(archive, window, elevation_full, min(zoom, h, w), shift_north)
        regions = [p['region'] for p in path_]
    else:
        path_ = None
        regions = [rain_region(archive, window, min(region, h, w))]*len(window)
    out = Path(output) if output else (Path(first_cfg['train']['output'])/'evaluation'/f'storm_{peak["id"]}_{job}')
    out.mkdir(parents=True, exist_ok=True)
    dx_km = float(np.sqrt(np.median(np.asarray(archive.static['area'], dtype='float64')))/1000)
    r0, r1, c0, c1 = regions[0]
    log(f'Storm peak {peak["time"]} ({split}); {len(window)} hours {window[0]["time"][:16]} → {window[-1]["time"][:16]}; '
        f'{"window following the low" if track else "fixed window"} {(r1-r0)*dx_km:.0f}×{(c1-c0)*dx_km:.0f} km'
        f'{f", shifted north {shift_north:.0%}" if track and shift_north else ""}; '
        f'{len(runs)} run(s) × {members} members; seeds {seeds}; post {post}')
    if path_:
        for e, p in zip(window, path_):
            y, x = p['center']
            log(f'  {e["time"][:16]}: low (terrain-reduced) {p["low_hpa"]:.1f} hPa at {lat_full[y, x]:.2f}°N '
                f'{lon_full[y, x]:.2f}°E' + ('  ⚠ at the domain edge: the low is probably outside the domain'
                                           if p['at_edge'] else ''))

    truth_raw, geos_raw, found = [], [], []
    for e, (a, b, c, d) in zip(window, regions):
        rows, cols = slice(a, b), slice(c, d)
        lat, lon = lat_full[rows, cols], lon_full[rows, cols]
        truth_raw.append(np.asarray(archive.physical_truth(e), dtype='float32')[:, rows, cols])
        coarse = np.asarray(archive.coarse(e), dtype='float32')[:, rows, cols]
        nat = native_on_grid(e, lat, lon, log)
        found.append(all(k in nat for k in ('precip', 't2m', 'u10m', 'v10m', 'ps')))
        geos_raw.append(geosfp_fields(nat, coarse))
    native_label = ('GEOS-FP 0.25° cells' if all(found) else
                    'GEOS-FP 0.25° cells (some hours: regridded input)' if any(found) else 'GEOS-FP (regridded input)')
    centers = [(p['center'][0]-reg[0], p['center'][1]-reg[2]) for p, reg in zip(path_, regions)] if path_ else []

    members_raw = {}   # run name -> [hour][member] (6, R, C)
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
            anchor = peak if seeds == 'fixed' else e
            hour = []
            for m in range(members):
                fields = engine.sample(member_seed(cfg, anchor, m), spec)
                if clim is not None:
                    fields = spectral_fix(fields, clim, dx_km)
                hour.append(fields)
            frames_run.append(hour)
            log(f'  {label} (epoch {saved["epoch"]+1}): hour {k+1}/{len(window)} {e["time"][:16]} · '
                f'{time.monotonic()-t0:.0f}s')
            del engine
        members_raw[f'{label} (ep {saved["epoch"]+1}{", spectral" if post == "spectral" else ""})'] = frames_run
        del model, conditioner
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    info = dict(peak=peak['time'], split=split, hours=[e['time'] for e in window], regions=[list(r) for r in regions],
                grid_km=dx_km, runs=[dict(label=l, config=c, checkpoint=k) for l, c, k in runs], members=members,
                steps=steps, post=post, seeds=seeds, native=native_label, tracked=bool(track), zoom=zoom,
                shift_north=shift_north,
                track=[dict(time=e['time'], center=list(p['center']), lat=float(lat_full[p['center']]),
                            lon=float(lon_full[p['center']]), low_hpa=p['low_hpa'], at_edge=p['at_edge'])
                       for e, p in zip(window, path_)] if path_ else None)
    if save_netcdf:
        write_netcdf(out/'storm_fields.nc', window, regions, truth_raw, geos_raw, members_raw, lat_full, lon_full,
                     elevation_full, info)
        log(f'Fields saved to {out/"storm_fields.nc"}')
    truth = [{v: display(v, t) for v in VARIABLES} for t in truth_raw]
    geos = [{v: display(v, g) for v in VARIABLES} for g in geos_raw]
    members_by_run = {run: [[{v: display(v, f) for v in VARIABLES} for f in hour] for hour in hours_]
                      for run, hours_ in members_raw.items()}
    canvas = Canvas(archive, use_cartopy=maps, features=maps, log=log) if maps else None
    for var in VARIABLES:
        animate(var, window, regions, truth, geos, members_by_run, members, native_label, out, dpi, fps, frames, plt,
                info, centers, canvas)
    if path_:
        plot_track(info, out, dpi, plt)
    write_json(out/'storm.json', info)
    log(f'Storm animations written to {out} ({time.monotonic()-started:.0f}s)')
    return out


def write_netcdf(path, window, regions, truth, geos, members_raw, lat_full, lon_full, elevation_full, info):
    """All hours' fields in physical units on the 2 km window (NetCDF4 via h5netcdf)."""
    import xarray as xr
    times = np.array([np.datetime64(e['time'][:19]) for e in window])
    crop = lambda full: np.stack([full[a:b, c:d] for a, b, c, d in regions]).astype('float32')
    runs = list(members_raw)
    members = len(next(iter(members_raw.values()))[0])
    data = {}
    for c, name in enumerate(TARGETS):
        attrs = {'units': UNITS[c]}
        data[f'truth_{name}'] = (('time', 'y', 'x'), np.stack([t[c] for t in truth]), attrs)
        data[f'geosfp_{name}'] = (('time', 'y', 'x'), np.stack([g[c] for g in geos]),
                                  dict(attrs, description='GEOS-FP 0.25° cells mapped to the 2 km grid (nearest cell)'))
        data[f'member_{name}'] = (('time', 'run', 'member', 'y', 'x'),
                                  np.stack([[[f[c] for f in hour] for hour in [members_raw[r][k] for r in runs]]
                                            for k in range(len(window))]), attrs)
    data['lat'] = (('time', 'y', 'x'), crop(lat_full), {'units': 'degrees_north'})
    data['lon'] = (('time', 'y', 'x'), crop(lon_full), {'units': 'degrees_east'})
    data['elevation'] = (('time', 'y', 'x'), crop(elevation_full), {'units': 'm'})
    data['window_row0'] = (('time',), np.array([r[0] for r in regions], dtype='int32'))
    data['window_col0'] = (('time',), np.array([r[2] for r in regions], dtype='int32'))
    if info.get('track'):
        data['low_lat'] = (('time',), np.array([p['lat'] for p in info['track']], dtype='float32'), {'units': 'degrees_north'})
        data['low_lon'] = (('time',), np.array([p['lon'] for p in info['track']], dtype='float32'), {'units': 'degrees_east'})
        data['low_reduced_pressure'] = (('time',), np.array([p['low_hpa'] for p in info['track']], dtype='float32'),
                                        {'units': 'hPa', 'description': 'terrain-reduced truth pressure at the tracked low'})
        data['low_at_domain_edge'] = (('time',), np.array([p['at_edge'] for p in info['track']], dtype='int8'))
    ds = xr.Dataset(data, coords=dict(time=times, run=np.array(runs, dtype=object), member=np.arange(1, members+1)),
                    attrs=dict(version='v4.1', peak=info['peak'], split=info['split'], post=info['post'],
                               seeds=info['seeds'], steps=info['steps'], grid_km=info['grid_km'],
                               note='Windows follow the storm: y/x are window-relative; see window_row0/col0, lat/lon.'))
    encoding = {k: dict(zlib=True, complevel=4) for k, v in data.items() if v[1].dtype.kind == 'f'}
    ds.to_netcdf(path, engine='h5netcdf', encoding=encoding)


def _scale(var, truth):
    stack = np.stack([t[var] for t in truth])
    if var == 'precip':
        cmap, norm = _rain_norm()
        return dict(cmap=cmap, norm=norm)
    lo, hi = np.quantile(stack, [.005, .995])
    cmap = {'t2m': DISPLAY['t2m'][4], 'wind': 'magma', 'ps': 'viridis'}[var]
    from matplotlib.colors import Normalize
    return dict(cmap=cmap, norm=Normalize(float(lo), float(hi)))


def animate(var, window, regions, truth, geos, members_by_run, members, native_label, out, dpi, fps, frames, plt,
            info, centers=(), canvas=None):
    from matplotlib import animation
    runs = list(members_by_run)
    cols = 2+members
    rows = len(runs)
    h, w = truth[0][var].shape
    panel = 3.3
    fig = plt.figure(figsize=(panel*cols*w/max(h, w)+1.3, panel*rows*h/max(h, w)+1.2), constrained_layout=True)
    grid = fig.add_gridspec(rows, cols)
    style = _scale(var, truth)
    mapped = canvas is not None and canvas.mode == 'lcc'
    origin = 'upper' if mapped and canvas.dy < 0 else 'lower'
    images, markers = [], []
    for r, run in enumerate(runs):
        sources = [('native', native_label), ('truth', 'Truth 2 km')] if r == 0 else [None, None]
        sources += [(('member', run, m), f'{run}\nmember {m+1}') for m in range(members)]
        for c, src in enumerate(sources):
            if src is None:
                continue
            key, title = src
            ax = canvas.axes(fig, grid[r, c]) if mapped else fig.add_subplot(grid[r, c])
            image = ax.imshow(_frame(key, 0, var, truth, geos, members_by_run), interpolation='nearest',
                              origin=origin, rasterized=True, **style,
                              **(dict(transform=canvas.proj) if mapped else {}))
            if mapped:
                canvas.decorate(ax, left=c == 0, bottom=r == rows-1)
            else:
                ax.set_xticks([])
                ax.set_yticks([])
            images.append((key, ax, image))
            if centers:
                markers.append(ax.plot([], [], marker='x', color='w', mec='k', ms=10, mew=2.4, zorder=6,
                                       **(dict(transform=canvas.proj) if mapped else {}))[0])
            ax.set_title(title, fontsize=8)
    title, unit = STYLE[var]
    bar = fig.colorbar(images[-1][2], ax=[a for _, a, _ in images], shrink=.8, pad=.01, label=f'{title} ({unit})')
    if var != 'precip':
        bar.formatter.set_useOffset(False)
        bar.update_ticks()
    heading = fig.suptitle('', fontsize=10)

    def place(k):
        """Data orientation and map extent of hour k's window."""
        a, b, c, d = regions[k]
        if not mapped:
            return None, False
        return canvas._extent(slice(a, b), slice(c, d)), canvas.dx < 0

    def update(k):
        extent, flip = place(k)
        for key, ax, image in images:
            data = _frame(key, k, var, truth, geos, members_by_run)
            image.set_data(data[:, ::-1] if flip else data)
            if extent is not None:
                image.set_extent(extent)
                ax.set_extent(extent, crs=canvas.proj)
        for marker in markers:
            y, x = centers[k]
            if mapped:
                a, _, c, _ = regions[k]
                marker.set_data([canvas.xc[c+x]], [canvas.yc[a+y]])
            else:
                marker.set_data([x], [y])
        low = ''
        if info.get('track'):
            p = info['track'][k]
            low = f' · × = pressure low{" (at domain edge)" if p["at_edge"] else ""}'
        heading.set_text(f'{title} · {window[k]["time"][:16].replace("T", " ")} UTC · hour {k+1}/{len(window)} · '
                         f'peak {info["peak"][:16].replace("T", " ")}{low}')
        return [image for _, _, image in images]+markers+[heading]
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
    """Track of the truth pressure low (lat/lon) with its terrain-reduced central pressure."""
    t = info['track']
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    sc = a.scatter([p['lon'] for p in t], [p['lat'] for p in t], c=[p['low_hpa'] for p in t], cmap='viridis', s=40)
    a.plot([p['lon'] for p in t], [p['lat'] for p in t], color='k', lw=.8)
    for p in t[::max(1, len(t)//6)]:
        a.annotate(p['time'][5:13].replace('T', ' '), (p['lon'], p['lat']), fontsize=7)
    fig.colorbar(sc, ax=a, label='Terrain-reduced pressure at the low (hPa)')
    a.set_xlabel('Longitude (°E)')
    a.set_ylabel('Latitude (°N)')
    a.set_title('Track of the truth pressure low', fontsize=10)
    b.plot(range(len(t)), [p['low_hpa'] for p in t], marker='o', color='#2166ac')
    for k, p in enumerate(t):
        if p['at_edge']:
            b.plot(k, p['low_hpa'], marker='o', color='#d73027')
    ticks = list(range(0, len(t), max(1, len(t)//8)))
    b.set_xticks(ticks, [t[k]['time'][5:13].replace('T', ' ') for k in ticks], fontsize=7)
    b.set_ylabel('hPa')
    b.set_title('Central pressure of the low in the domain (red = at domain edge)', fontsize=10)
    b.grid(True, ls='--', alpha=.4)
    fig.savefig(out/'track.png', dpi=dpi*2)
    plt.close(fig)


def _frame(key, k, var, truth, geos, members_by_run):
    if key == 'native':
        return geos[k][var]
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
    parser.add_argument('--members', type=int, default=2)
    parser.add_argument('--steps', type=int, default=64)
    parser.add_argument('--no-track', action='store_true', help='Fixed window over the rain swath instead of tracking')
    parser.add_argument('--zoom', type=int, default=394, help='Tracking window size in px (default 394 ≈ 745 km)')
    parser.add_argument('--shift-north', type=float, default=.25,
                        help='Move the tracking window north by this fraction of its size (default 0.25)')
    parser.add_argument('--region', type=int, default=768, help='Fixed window size in px with --no-track')
    parser.add_argument('--post', choices=('none', 'spectral'), default='spectral')
    parser.add_argument('--seeds', choices=('fixed', 'independent'), default='fixed')
    parser.add_argument('--weights', choices=('ema', 'raw'), default='ema')
    parser.add_argument('--no-maps', action='store_true', help='Skip Cartopy (plain pixel panels)')
    parser.add_argument('--cartopy-data-dir', help='Natural Earth cache for coastlines/states on offline nodes')
    parser.add_argument('--no-netcdf', action='store_true')
    parser.add_argument('--fps', type=float, default=2.)
    parser.add_argument('--frames', action='store_true', help='Also write every frame as PNG')
    parser.add_argument('--output')
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--dpi', type=int, default=90)
    a = parser.parse_args()
    if a.cartopy_data_dir:
        import cartopy
        cartopy.config['pre_existing_data_dir'] = a.cartopy_data_dir
    run(parse_runs(a.runs), a.timestamp, a.split, a.hours, a.members, a.steps, a.region, post=a.post, seeds=a.seeds,
        weights=a.weights, output=a.output, batch=a.batch, threads=a.threads, dpi=a.dpi, fps=a.fps, frames=a.frames,
        track=not a.no_track, zoom=a.zoom, shift_north=a.shift_north, maps=not a.no_maps,
        save_netcdf=not a.no_netcdf, log=lambda message: print(message, flush=True))


if __name__ == '__main__':
    main()
