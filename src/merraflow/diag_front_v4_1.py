"""Front-sharpness diagnostic for one case: is the blur in the model or in the tiling?

Samples the same member noise three ways and compares them with truth in one
192x192 tile window centred on a strong truth front:

  tiled@32     full-domain synchronized tiled Heun, 32 steps (v4 inference)
  tiled@64     the same with 2x steps (the best setting from the sharpness ablation)
  single@64    ONLY the tile containing the front, no neighbours, no blending,
               64 steps -- what the model does on a patch, as in training/validation

If single@64 has sharp fronts and tiled@64 does not, the tiling/blending is
blurring them. If both are smooth, the model itself is.

Scores (per member and for the ensemble mean, per field):
  * tail of the gradient magnitude, |grad f| quantiles p95/p99/p99.9 relative to truth
    (a smeared front lowers the tail even when the mean gradient looks fine);
  * cross-front profile along the truth-front normal, averaged over a band
    along the front, and its 10-90 % transition width in km.

Outputs (default <train.output>/evaluation/front_diag_<ckpt>_<case>_<job>/):
  fields_<var>.{png,pdf}   truth | members of each variant | ensemble means, profile line marked
  profiles.{png,pdf}       cross-front profiles for all fields
  gradient_tails.{png,pdf} |grad| quantile ratios to truth
  report.md, metrics.json
"""
import argparse
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from scipy.ndimage import gaussian_filter, map_coordinates

from .config import write_json
from .metrics import weighted_mean
from .train import device_for
from .train_v2 import file_hash_v2
from .v4 import TARGETS
from .v4_1 import load_config
from .evaluate_v4_1 import (resolve_checkpoint, load_model, DomainSampler, select_cases, member_seed,
                            _rain_norm, DISPLAY, _display)

FIELDS = ('q2m', 't2m', 'u10m', 'v10m', 'ps', 'precip')
QUANTILES = (.95, .99, .999)


def find_front(truth, shape, width, halo, center=None, smooth=3., region=None):
    """Tile origin (y, x) and front point/normal. The front point is the maximum of
    the combined, normalised q2m+t2m gradient (smoothed) inside ``region`` (rows, cols
    slices; default whole domain) unless ``center`` is given."""
    h, w = shape
    score = 0
    for name in ('q2m', 't2m'):
        f = gaussian_filter(np.asarray(truth[TARGETS.index(name)], dtype='float64'), smooth)
        gy, gx = np.gradient(f)
        g = np.hypot(gy, gx)
        score = score+g/max(float(np.quantile(g, .999)), 1e-30)
    if center is None:
        margin = width//2
        inner = np.full(score.shape, -np.inf)
        inner[margin:h-margin, margin:w-margin] = score[margin:h-margin, margin:w-margin]
        if region is not None:
            keep = np.zeros(score.shape, dtype=bool)
            keep[region] = True
            inner[~keep] = -np.inf
        cy, cx = np.unravel_index(np.argmax(inner), inner.shape)
    else:
        cy, cx = center
    # Normal: the q2m gradient direction at the front, smoothed over the front scale.
    f = gaussian_filter(np.asarray(truth[TARGETS.index('q2m')], dtype='float64'), 3*smooth)
    gy, gx = np.gradient(f)
    ny, nx = gy[cy, cx], gx[cy, cx]
    norm = np.hypot(ny, nx) or 1.
    # Tile origin in DomainSampler coordinates: padded-grid tile [y, y+width)
    # covers unpadded rows [y-halo, y-halo+width); centre the tile on the front.
    y = int(np.clip(cy-width//2+halo, halo, h+halo-width))
    x = int(np.clip(cx-width//2+halo, halo, w+halo-width))
    return (y, x), (int(cy), int(cx)), (ny/norm, nx/norm)


def profile(field, point, normal, length=60, band=24, step=1.):
    """Mean profile along the normal through ``point``, averaged over parallel
    lines within +-band px along the front. Returns offsets (px) and values."""
    cy, cx = point
    ny, nx = normal
    ty, tx = -nx, ny
    s = np.arange(-length, length+step, step)
    rows = []
    for b in np.arange(-band, band+1, 2):
        yy, xx = cy+b*ty+s*ny, cx+b*tx+s*nx
        rows.append(map_coordinates(np.asarray(field, dtype='float64'), [yy, xx], order=1, mode='nearest'))
    return s, np.mean(rows, axis=0)


def transition_width(offsets, values, dx_km, edge=.15):
    """10-90 % width (km) of the jump between the profile's two plateaus, measured
    around the steepest point (None if there is no clear jump)."""
    n = len(values)
    k = max(2, int(edge*n))
    lo, hi = float(np.mean(values[:k])), float(np.mean(values[-k:]))
    jump = hi-lo
    if abs(jump) < 1e-12:
        return None
    frac = (values-lo)/jump
    centre = int(np.argmax(np.abs(np.gradient(frac))))

    def crossing(level, direction):
        i = centre
        while 0 <= i < n and ((frac[i] > level) if direction < 0 else (frac[i] < level)):
            i += direction
        return None if not 0 <= i < n else offsets[i]
    a, b = crossing(.1, -1), crossing(.9, +1)
    if a is None or b is None:
        return None
    return float(abs(b-a)*dx_km)


def slope_width(offsets, values, dx_km, edge=.15):
    """Max-slope (equivalent) width: jump between the profile ends divided by the
    steepest slope. Measures the sharpest part of the transition, so a gradual
    ramp ahead of a sharp front does not inflate it."""
    k = max(2, int(edge*len(values)))
    jump = abs(float(np.mean(values[-k:])-np.mean(values[:k])))
    slope = float(np.max(np.abs(np.gradient(values, offsets))))
    if jump < 1e-12 or slope < 1e-12:
        return None
    return jump/slope*dx_km


def gradient_tail(field, quantiles=QUANTILES):
    gy, gx = np.gradient(np.asarray(field, dtype='float64'))
    return np.quantile(np.hypot(gy, gx), quantiles)


def single_tile(sampler, tile_index, seed, steps):
    """Integrate one tile alone (no neighbours, no blending) from the same noise
    the full-domain sampler uses there. Returns the flow-space core crop (6, width, width)."""
    y, x = sampler.tiles[tile_index]
    w = sampler.width
    noise = np.random.default_rng(seed).standard_normal(
        (len(TARGETS), sampler.h+2*sampler.halo, sampler.w+2*sampler.halo)).astype('float32')
    state = torch.from_numpy(noise[:, y:y+w, x:x+w].copy()[None]).to(sampler.device)
    cond, ctx = sampler.condition[tile_index:tile_index+1], sampler.context[tile_index:tile_index+1]
    mean = sampler.means[tile_index:tile_index+1]
    from .train import autocast
    with torch.no_grad():
        for i in range(steps):
            t0, t1 = i/steps, (i+1)/steps
            def v(s, t):
                with autocast(sampler.device, sampler.cfg['train']['precision']):
                    return sampler.model(s, torch.full((1,), float(t), device=sampler.device), cond, ctx, mean).float()
            first = v(state, t0)
            second = v(state+first/steps, t1)
            state = state+(first+second)/(2*steps)
    return state[0].cpu().numpy(), sampler.means[tile_index].float().cpu().numpy()


def decode_tile(sampler, core, mean, y, x):
    """Physical fields of a tile-shaped flow state (padded-grid origin y, x)."""
    w, halo = sampler.width, sampler.halo
    ys, xs = slice(y-halo, y-halo+w), slice(x-halo, x-halo+w)
    coarse = sampler.coarse[:, ys, xs]
    value = (core*sampler.scale+mean)*sampler.rs+sampler.rm+coarse
    z = np.maximum(core[1], 0)
    value[1] = sampler.archive.scale*z*(z+2)
    value[5] = np.clip(value[5], 0, 1)
    return value.astype('float32')


def run(cfg, checkpoint='latest', timestamp=None, split='test', members=4, center=None, output=None,
        batch=32, threads=8, dpi=200, pdf=True, profile_length=60, profile_band=24, center_latlon=None,
        anywhere=False, log=print):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    path = resolve_checkpoint(cfg, checkpoint)
    device = device_for(cfg['train']['device'])
    archive, model, conditioner, saved = load_model(cfg, path, device)
    cases = select_cases(archive, split, [timestamp] if timestamp else None, 0, 0 if timestamp else 1, 317, log)
    entry = cases[0]['entry']
    job = os.environ.get('SLURM_JOB_ID') or time.strftime('%Y%m%d_%H%M%S')
    out = Path(output) if output else (Path(cfg['train']['output'])/'evaluation'/
                                      f'front_diag_{path.stem}_{entry["id"]}_{job}')
    out.mkdir(parents=True, exist_ok=True)
    area = np.asarray(archive.static['area'], dtype='float64')
    dx_km = float(np.sqrt(np.median(area))/1000)
    truth_full = np.asarray(archive.physical_truth(entry), dtype='float32')
    started = time.monotonic()
    sampler = DomainSampler(model, conditioner, archive, entry, cfg, device, batch, threads)
    w, halo = sampler.width, sampler.halo
    if center is None and center_latlon is not None:
        lat, lon = np.asarray(archive.static['lat']), np.asarray(archive.static['lon'])
        dist = (lat-center_latlon[0])**2+((lon-center_latlon[1])*np.cos(np.deg2rad(center_latlon[0])))**2
        center = tuple(int(v) for v in np.unravel_index(np.argmin(dist), dist.shape))
    # By default look for the front near the strongest rain feature (storm-related
    # boundaries), not domain-wide, where static coastlines/terrain edges dominate.
    from .evaluate_v4_1 import event_window
    region = None if anywhere or center is not None else event_window(truth_full[1], 384)
    origin, point, normal = find_front(truth_full, archive.shape, w, halo, center, region=region)
    tile_index = sampler.tiles.index(origin) if origin in sampler.tiles else None
    if tile_index is None:
        # Build the inputs of this exact tile position (it need not be on the stride grid).
        sampler.tiles.append(origin)
        extra = archive.inputs_with_original(entry, *origin, cfg['patch'])
        from .dataset_v2 import crop_v2
        sampler.condition = torch.cat([sampler.condition, extra['condition'][None].to(device)])
        sampler.context = torch.cat([sampler.context, extra['context'][None].to(device)])
        coarse = torch.from_numpy(crop_v2(sampler.coarse, *origin, sampler.size, halo)[None]).to(device)
        with torch.no_grad():
            b = dict(original_condition=extra['original_condition'][None].to(device),
                     original_context=extra['original_context'][None].to(device), coarse=coarse)
            from .train import autocast
            with autocast(device, cfg['train']['precision']):
                sampler.means = torch.cat([sampler.means, conditioner(b).float()])
        tile_index = len(sampler.tiles)-1
        tiles_for_domain = sampler.tiles[:-1]
    else:
        tiles_for_domain = sampler.tiles
    y, x = origin
    ys, xs = slice(y-halo, y-halo+w), slice(x-halo, x-halo+w)
    local_point = (point[0]-(y-halo), point[1]-(x-halo))
    log(f'Case {entry["id"]} ({cases[0]["reason"]}); checkpoint {path.name} (epoch {saved["epoch"]+1}); '
        f'front at row {point[0]}, col {point[1]}; tile origin {origin}; {len(tiles_for_domain)} domain tiles; '
        f'grid {dx_km:.2f} km')

    # Domain sampling must use only the regular tiles: temporarily hide an extra tile.
    all_tiles = sampler.tiles
    variants = {'tiled@32': [], 'tiled@64': [], 'single@64': []}
    for m in range(members):
        seed = member_seed(cfg, entry, m)
        t0 = time.monotonic()
        sampler.tiles = tiles_for_domain
        for name, steps in (('tiled@32', 32), ('tiled@64', 64)):
            variants[name].append(sampler.sample(seed, steps)[:, ys, xs])
        sampler.tiles = all_tiles
        core, mean = single_tile(sampler, tile_index, seed, 64)
        variants['single@64'].append(decode_tile(sampler, core, mean, y, x))
        log(f'  member {m+1}/{members}: {time.monotonic()-t0:.0f}s')
    variants = {k: np.stack(v) for k, v in variants.items()}
    truth = truth_full[:, ys, xs]
    tile_area = area[ys, xs]
    coarse = sampler.coarse[:, ys, xs]

    # Scores
    metrics = dict(case=entry['id'], time=entry['time'], checkpoint=str(path), epoch=saved['epoch']+1,
                   front_point=list(point), normal=[float(v) for v in normal], tile_origin=list(origin),
                   grid_km=dx_km, members=members, fields={})
    profiles = {}
    for name in FIELDS:
        c = TARGETS.index(name) if name != 'precip' else 1
        t = truth[c]
        t_tail = gradient_tail(t)
        s, t_prof = profile(t, local_point, normal, profile_length, profile_band)
        entry_m = dict(truth=dict(width_km=transition_width(s, t_prof, dx_km), slope_width_km=slope_width(s, t_prof, dx_km),
                                  grad_tail=t_tail.tolist()))
        prof = dict(truth=t_prof, coarse=profile(coarse[c], local_point, normal, profile_length, profile_band)[1])
        for vname, ens in variants.items():
            f = ens[:, c]
            member_profiles = [profile(mm, local_point, normal, profile_length, profile_band)[1] for mm in f]
            member_widths = [transition_width(s, mp, dx_km) for mp in member_profiles]
            slopes = [v for v in (slope_width(s, mp, dx_km) for mp in member_profiles) if v is not None]
            member_tails = np.array([gradient_tail(mm) for mm in f])
            mean_prof = profile(f.mean(0), local_point, normal, profile_length, profile_band)[1]
            entry_m[vname] = dict(
                member_width_km=[None if v is None else float(v) for v in member_widths],
                member_width_km_median=float(np.median([v for v in member_widths if v is not None]))
                if any(v is not None for v in member_widths) else None,
                mean_width_km=transition_width(s, mean_prof, dx_km),
                member_slope_width_km_median=float(np.median(slopes)) if slopes else None,
                mean_slope_width_km=slope_width(s, mean_prof, dx_km),
                member_grad_tail_ratio=(member_tails.mean(0)/np.maximum(t_tail, 1e-30)).tolist(),
                mean_grad_tail_ratio=(gradient_tail(f.mean(0))/np.maximum(t_tail, 1e-30)).tolist(),
                member_rmse=float(np.mean([np.sqrt(weighted_mean((mm-t)**2, tile_area)) for mm in f])),
                mean_rmse=float(np.sqrt(weighted_mean((f.mean(0)-t)**2, tile_area))))
            prof[vname] = [profile(mm, local_point, normal, profile_length, profile_band)[1] for mm in f]
            prof[vname+' mean'] = mean_prof
        entry_m['coarse'] = dict(width_km=transition_width(s, prof['coarse'], dx_km),
                                 slope_width_km=slope_width(s, prof['coarse'], dx_km))
        metrics['fields'][name] = entry_m
        profiles[name] = (s, prof)
        log(f'  {name:>6}: 10-90% width truth {_km(entry_m["truth"]["width_km"])} · '
            + ' · '.join(f'{v} members {_km(entry_m[v]["member_width_km_median"])} / mean {_km(entry_m[v]["mean_width_km"])}'
                         for v in variants)
            + f' · p99 |∇| member ratio ' + ' / '.join(f'{entry_m[v]["member_grad_tail_ratio"][1]:.2f}' for v in variants))
    metrics['verdict'] = verdict(metrics)
    write_json(out/'metrics.json', json.loads(json.dumps(metrics, default=float)))

    def save(fig, stem):
        fig.savefig(out/f'{stem}.png', dpi=dpi)
        if pdf:
            fig.savefig(out/f'{stem}.pdf', dpi=dpi)
        plt.close(fig)

    heading = (f'v4.1 epoch {saved["epoch"]+1} ({path.stem}) · {split} {entry["time"][:16]} · front diagnostic · '
               f'tile origin {origin}, {w}×{w} px ({w*dx_km:.0f} km)')
    for name in FIELDS:
        save(plot_fields(name, truth, coarse, variants, local_point, normal, profile_length, heading, plt), f'fields_{name}')
    save(plot_profiles(profiles, metrics, dx_km, heading, plt), 'profiles')
    save(plot_tails(metrics, heading, plt), 'gradient_tails')
    write_report(out, metrics)
    log(f'Verdict: {metrics["verdict"]}')
    log(f'Front diagnostic written to {out} ({time.monotonic()-started:.0f}s)')
    return out


def _km(value):
    return 'n/a' if value is None else f'{value:.1f} km'


def verdict(metrics):
    """Compare single-tile and tiled member sharpness on q2m and t2m."""
    lines = []
    for name in ('q2m', 't2m', 'v10m'):
        f = metrics['fields'][name]
        tw, single, tiled = (f['truth']['slope_width_km'], f['single@64']['member_slope_width_km_median'],
                             f['tiled@64']['member_slope_width_km_median'])
        st, tt = f['single@64']['member_grad_tail_ratio'][1], f['tiled@64']['member_grad_tail_ratio'][1]
        if single is None or tiled is None:
            lines.append(f'{name}: no clear front profile')
            continue
        if tiled > 1.3*single and tt < .85*st:
            what = 'TILING blurs it (single tile clearly sharper)'
        elif single > 1.5*(tw or single) and abs(tiled-single) <= .3*single:
            what = 'the MODEL blurs it (single tile equally smooth)'
        else:
            what = 'no clear tiling effect'
        lines.append(f'{name}: truth {_km(tw)}, single tile {_km(single)}, tiled {_km(tiled)}, '
                     f'p99|∇| single {st:.2f} vs tiled {tt:.2f} → {what}')
    return lines


def _norm_for(name, pool):
    from matplotlib.colors import Normalize
    if name == 'precip':
        return _rain_norm()
    lo, hi = np.nanquantile(pool, [.01, .99])
    return DISPLAY[name][4], Normalize(float(lo), float(max(hi, lo+1e-9)))


def plot_fields(name, truth, coarse, variants, point, normal, length, heading, plt):
    c = TARGETS.index(name) if name != 'precip' else 1
    conv = (lambda f: f) if name == 'precip' else (lambda f: _display(name, f))
    members = min(3, len(next(iter(variants.values()))))
    columns = [('Truth', truth[c]), ('Coarse', coarse[c])]
    for vname, ens in variants.items():
        columns += [(f'{vname} · m{k+1}', ens[k, c]) for k in range(members)]
        columns.append((f'{vname} · mean', ens[:, c].mean(0)))
    cols = 2+len(variants)*(members+1)
    rows = 2
    per_row = (cols+1)//2
    fig, axes = plt.subplots(rows, per_row, figsize=(3.6*per_row, 3.9*rows), constrained_layout=True, squeeze=False)
    cmap, norm = _norm_for(name, np.concatenate([conv(truth[c]).ravel(), conv(coarse[c]).ravel()]))
    cy, cx = point
    ny, nx = normal
    image = None
    for ax, (label, field) in zip(axes.flat, columns):
        image = ax.imshow(conv(field), origin='lower', cmap=cmap, norm=norm, interpolation='nearest')
        ax.plot([cx-length*nx, cx+length*nx], [cy-length*ny, cy+length*ny], color='k', lw=.8, ls='--')
        ax.set_title(label, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in list(axes.flat)[len(columns):]:
        ax.set_visible(False)
    title, unit = DISPLAY[name][0], DISPLAY[name][1]
    fig.colorbar(image, ax=list(axes.flat), shrink=.8, label=f'{title} ({unit})')
    fig.suptitle(f'{heading}\n{title}: dashed line = cross-front profile; single = one tile alone, no blending',
                 fontsize=11)
    return fig


def plot_profiles(profiles, metrics, dx_km, heading, plt):
    fig, axes = plt.subplots(2, 3, figsize=(19, 10), constrained_layout=True)
    colors = {'tiled@32': '#74add1', 'tiled@64': '#2166ac', 'single@64': '#d6604d'}
    for ax, name in zip(axes.flat, FIELDS):
        s, prof = profiles[name]
        km = s*dx_km
        conv = (lambda f: f) if name == 'precip' else (lambda f: _display(name, f))
        ax.plot(km, conv(prof['truth']), color='k', lw=2.6, label='Truth', zorder=5)
        ax.plot(km, conv(prof['coarse']), color='0.55', lw=1.4, ls=':', label='Coarse')
        for vname, color in colors.items():
            for k, member in enumerate(prof[vname]):
                ax.plot(km, conv(member), color=color, lw=.7, alpha=.55, label=f'{vname} members' if k == 0 else None)
            ax.plot(km, conv(prof[vname+' mean']), color=color, lw=2, ls='--', label=f'{vname} mean')
        f = metrics['fields'][name]
        ax.set_title(f'{DISPLAY[name][0]} · 10–90% width: truth {_km(f["truth"]["width_km"])}, '
                     f'single {_km(f["single@64"]["member_width_km_median"])}, '
                     f'tiled {_km(f["tiled@64"]["member_width_km_median"])}\n'
                     f'max-slope width: truth {_km(f["truth"]["slope_width_km"])}, '
                     f'single {_km(f["single@64"]["member_slope_width_km_median"])}, '
                     f'tiled {_km(f["tiled@64"]["member_slope_width_km_median"])}', fontsize=9)
        ax.set_xlabel('Distance across front (km)')
        ax.set_ylabel(DISPLAY[name][1])
        ax.grid(True, ls='--', alpha=.4)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(f'{heading}\nCross-front profiles (band-averaged along the front)', fontsize=12)
    return fig


def plot_tails(metrics, heading, plt):
    fig, axes = plt.subplots(1, len(QUANTILES), figsize=(6*len(QUANTILES), 5), constrained_layout=True)
    variants = ('tiled@32', 'tiled@64', 'single@64')
    colors = ('#74add1', '#2166ac', '#d6604d')
    x = np.arange(len(FIELDS))
    for q, ax in enumerate(axes):
        for k, (vname, color) in enumerate(zip(variants, colors)):
            values = [metrics['fields'][n][vname]['member_grad_tail_ratio'][q] for n in FIELDS]
            ax.bar(x+(k-1)*.27, values, .27, color=color, label=f'{vname} members')
        ax.axhline(1, color='k', lw=1.2)
        ax.set_xticks(x, FIELDS)
        ax.set_title(f'|∇| p{QUANTILES[q]*100:g} ratio to truth (1 = truth-like edges)')
        ax.grid(True, axis='y', ls='--', alpha=.4)
    axes[0].legend(fontsize=8)
    fig.suptitle(f'{heading}\nSharpest gradients in the tile: members vs truth', fontsize=12)
    return fig


def write_report(out, metrics):
    L = [f'# Front diagnostic · case `{metrics["case"]}` · epoch {metrics["epoch"]}', '',
         f'Tile origin {metrics["tile_origin"]}, front point {metrics["front_point"]}, grid {metrics["grid_km"]:.2f} km, '
         f'{metrics["members"]} members.', '', '## Verdict', '']
    L += [f'- {line}' for line in metrics['verdict']]
    L += ['', '## 10–90 % cross-front transition width (km) and p99 |∇| ratio to truth (members)', '',
          '| Field | Truth | Coarse | tiled@32 members | tiled@64 members | single@64 members | tiled@64 mean | single@64 mean | p99 tiled@32 | p99 tiled@64 | p99 single@64 |',
          '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name, f in metrics['fields'].items():
        cells = [_km(f['truth']['width_km']), _km(f['coarse']['width_km'])]
        cells += [_km(f[v]['member_width_km_median']) for v in ('tiled@32', 'tiled@64', 'single@64')]
        cells += [_km(f[v]['mean_width_km']) for v in ('tiled@64', 'single@64')]
        cells += [f'{f[v]["member_grad_tail_ratio"][1]:.2f}' for v in ('tiled@32', 'tiled@64', 'single@64')]
        L.append(f'| {name} | ' + ' | '.join(cells) + ' |')
    L += ['', '## Max-slope front width (km): jump / steepest slope (robust to gradual ramps)', '',
          '| Field | Truth | Coarse | tiled@32 members | tiled@64 members | single@64 members | tiled@64 mean | single@64 mean |',
          '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name, f in metrics['fields'].items():
        cells = [_km(f['truth']['slope_width_km']), _km(f['coarse']['slope_width_km'])]
        cells += [_km(f[v]['member_slope_width_km_median']) for v in ('tiled@32', 'tiled@64', 'single@64')]
        cells += [_km(f[v]['mean_slope_width_km']) for v in ('tiled@64', 'single@64')]
        L.append(f'| {name} | ' + ' | '.join(cells) + ' |')
    L += ['', 'Member widths close to truth with a wider ensemble-mean width = sharp members at slightly different '
          'positions (expected). Member widths far above truth = the members themselves are smooth.']
    (out/'report.md').write_text('\n'.join(L)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', default='configs/discover_v4_1.yaml')
    parser.add_argument('--checkpoint', default='latest', help='best | latest (default) | <epoch> | <path>')
    parser.add_argument('--timestamp', help='Case ID or ISO time (default: wettest hour of the split)')
    parser.add_argument('--split', choices=('val', 'test'), default='test')
    parser.add_argument('--members', type=int, default=4)
    parser.add_argument('--center', type=int, nargs=2, metavar=('ROW', 'COL'),
                        help='Front point on the full grid (default: strongest truth q2m+t2m front)')
    parser.add_argument('--center-latlon', type=float, nargs=2, metavar=('LAT', 'LON'),
                        help='Front point as latitude/longitude (e.g. 35.5 -90.5)')
    parser.add_argument('--anywhere', action='store_true',
                        help='Search the whole domain for the front (default: near the strongest rain feature)')
    parser.add_argument('--output')
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--dpi', type=int, default=200)
    parser.add_argument('--no-pdf', action='store_true')
    args = parser.parse_args()
    run(load_config(args.config), args.checkpoint, args.timestamp, args.split, args.members,
        tuple(args.center) if args.center else None, args.output, args.batch, args.threads, args.dpi,
        not args.no_pdf, center_latlon=tuple(args.center_latlon) if args.center_latlon else None,
        anywhere=args.anywhere, log=lambda message: print(message, flush=True))


if __name__ == '__main__':
    main()
