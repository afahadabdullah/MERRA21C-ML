"""v4.1 held-out evaluation: full-CONUS and zoomed ensemble maps plus diagnostics.

Checkpoint: best (default), ``--latest``, an epoch number (``--checkpoint 22``
picks ``checkpoints/epoch_0022_*``) or an explicit path.

Ensembles use v4's synchronized tiled Heun sampler with the same member seeds
as ``cli_v4_1 predict``. Tile inputs are built once per case and tiles are
batched on the GPU, so a member takes about a minute instead of tens of minutes.

Output (fresh directory, default under ``<train.output>/evaluation/``):

cases/<id>/
  conus_precip.png   coarse, frozen regression, truth, members, ensemble mean,
                     P(rain >= 1 mm/h), spread; zoom boxes marked
  conus_states.png   t2m, ps, u10m, v10m, q2m, wind speed: coarse, truth,
                     ensemble mean, mean - truth, spread
  zoom_event.png     all fields around the strongest rain feature
  zoom_random_K.png  all fields in seeded random windows
summary_scores.png       coarse / regression / member / ensemble MAE and CRPS
summary_precip.png       intensity distribution, Q-Q, reliability, FSS vs scale,
                         categorical scores, per-case domain-mean rain
summary_spectra.png      radial power spectra vs wavelength (km)
summary_calibration.png  rank histograms and spread-skill
summary_maps.png         case-mean bias maps and precip CRPS skill map
metrics_v4_1.json, report.md
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
import json
import socket
import time
import numpy as np
import torch
from .config import write_json
from .dataset_v2 import crop_v2
from .inference import starts, blend_window
from .metrics import continuous, precipitation, fss, rank_histogram, radial_psd, weighted_mean, crps_ensemble
from .train import device_for, autocast
from .train_v2 import file_hash_v2
from .v4 import ArchiveV4, TARGETS, UNITS, make_model, FrozenRegression
from .v4_1 import load_config, base_config, check_checkpoint
from .validation_v4_1 import RAIN_EDGES, RAIN_LEVELS, _rain_norm

# name: (title, display unit, offset, factor, colormap, symmetric)
DISPLAY = {'t2m': ('2 m temperature', '°C', -273.15, 1., 'RdYlBu_r', False),
           'precip': ('Hourly rain rate', 'mm h⁻¹', 0., 1., None, False),
           'ps': ('Surface pressure', 'hPa', 0., .01, 'cividis', False),
           'u10m': ('10 m zonal wind', 'm s⁻¹', 0., 1., 'RdBu_r', True),
           'v10m': ('10 m meridional wind', 'm s⁻¹', 0., 1., 'RdBu_r', True),
           'q2m': ('2 m specific humidity', 'g kg⁻¹', 0., 1000., 'YlGnBu', False),
           'wind_speed': ('10 m wind speed', 'm s⁻¹', 0., 1., 'viridis', False)}
THRESHOLDS = (.1, 1., 5., 10., 25.)
FSS_THRESHOLDS = (1., 5., 10.)
FSS_SCALES = (1, 5, 9, 17, 33, 65, 129)
QUANTILES = (.5, .75, .9, .95, .975, .99, .995, .999, .9995, .9999)


# ----------------------------------------------------------------------------
# Checkpoint, model, cases
# ----------------------------------------------------------------------------

def resolve_checkpoint(cfg, spec='best'):
    """best | latest/last | <epoch number> | <path>."""
    out = Path(cfg['train']['output'])
    spec = str(spec or 'best')
    if spec == 'best':
        path = out/'best_v4_1.pt'
    elif spec in ('latest', 'last'):
        path = out/'last_v4_1.pt'
    elif spec.isdigit():
        matches = sorted((out/'checkpoints').glob(f'epoch_{int(spec):04d}_*v4_1.pt'))
        if len(matches) != 1:
            raise FileNotFoundError(f'Expected one kept checkpoint for epoch {int(spec)} in {out/"checkpoints"}, '
                                    f'found {[m.name for m in matches]}')
        path = matches[0]
    else:
        path = Path(spec)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_model(cfg, checkpoint, device):
    base = base_config(cfg)
    archive = ArchiveV4(base, verify_files=False)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    check_checkpoint(saved, cfg, archive)
    model = make_model(archive.channels, base).to(device).eval()
    model.load_state_dict(saved['ema'])
    conditioner = FrozenRegression(saved['regression_condition'], archive.index['condition_channels'],
                                   archive.stats, archive.scale, cfg['data']['humidity_scale_kg_kg']).to(device)
    conditioner.flow_scale.copy_(saved['flow_scale'].to(device))
    return archive, model, conditioner, saved


def member_seed(cfg, entry, member):
    """Identical to inference_v4.predict, so members are reproducible there."""
    return int(np.random.SeedSequence([cfg['inference']['seed'], int(entry['id'].replace('_', '')),
                                       member]).generate_state(1)[0])


def _hours(entry):
    return datetime.fromisoformat(entry['time']).timestamp()/3600


def select_cases(archive, split, timestamps=None, samples=3, wettest=1, seed=317, log=print):
    entries = archive.eligible(split)
    if timestamps:
        chosen = []
        for stamp in timestamps:
            matches = [e for e in entries if stamp in (e['id'], e['time'], e['time'][:16])]
            if len(matches) != 1:
                raise ValueError(f'Expected one eligible {split} hour for {stamp!r}, found {len(matches)}')
            chosen.append(dict(entry=matches[0], reason='requested'))
        return chosen
    chosen = []
    if wettest:
        log(f'Scanning {len(entries)} {split} hours for the wettest cases (1/64 subsample)...')
        scores = np.array([float(np.asarray(archive.truth_field(e)[0, ::8, ::8], dtype='float64').mean())
                           for e in entries])
        for j in np.argsort(-scores, kind='stable'):
            if all(abs(_hours(entries[j])-_hours(c['entry'])) >= 12 for c in chosen):
                chosen.append(dict(entry=entries[j], reason=f'wettest (domain-mean ≈ {scores[j]:.3f} mm/h)'))
            if len(chosen) == wettest:
                break
    taken = {c['entry']['id'] for c in chosen}
    rest = [e for e in entries if e['id'] not in taken]
    if samples and rest:
        rng = np.random.default_rng(seed)
        for j in sorted(rng.choice(len(rest), size=min(samples, len(rest)), replace=False)):
            chosen.append(dict(entry=rest[j], reason='random'))
    if not chosen:
        raise ValueError('No cases selected')
    return chosen


# ----------------------------------------------------------------------------
# Full-domain sampler (v4 math, batched tiles)
# ----------------------------------------------------------------------------

class DomainSampler:
    """inference_v4.sample_frame with per-case inputs cached and tiles batched."""

    def __init__(self, model, conditioner, archive, entry, cfg, device, batch=32, threads=8):
        p = cfg['patch']
        self.model, self.conditioner, self.archive, self.cfg, self.device = model, conditioner, archive, cfg, device
        self.h, self.w = archive.shape
        self.size, self.halo = p['size'], p['halo']
        self.width = self.size+2*self.halo
        self.batch = batch
        self.tiles = [(y, c) for y in starts(self.h, self.size, p['stride'])
                      for c in starts(self.w, self.size, p['stride'])]
        self.window = torch.from_numpy(blend_window(self.width)).to(device)
        self.weight = torch.zeros((1, self.h+2*self.halo, self.w+2*self.halo), device=device)
        for y, c in self.tiles:
            self.weight[:, y:y+self.width, c:c+self.width] += self.window
        if bool((self.weight <= 0).any()):
            raise ValueError('Uncovered tile pixels')
        self.coarse = np.asarray(archive.coarse(entry), dtype='float32')
        first = archive.inputs_with_original(entry, *self.tiles[0], p)  # warm the map cache serially
        with ThreadPoolExecutor(max(1, threads)) as pool:
            rest = list(pool.map(lambda t: archive.inputs_with_original(entry, t[0], t[1], p), self.tiles[1:]))
        inputs = [first, *rest]
        self.channels = archive.index['condition_channels']
        self.condition = torch.stack([i['condition'] for i in inputs]).to(device)
        self.context = torch.stack([i['context'] for i in inputs]).to(device)
        coarse = torch.from_numpy(np.stack([crop_v2(self.coarse, y, c, self.size, self.halo) for y, c in self.tiles]))
        means = []
        with torch.no_grad():
            for s in range(0, len(self.tiles), batch):
                b = dict(original_condition=self.condition[s:s+batch, :self.channels],
                         original_context=self.context[s:s+batch, :self.channels],
                         coarse=coarse[s:s+batch].to(device))
                with autocast(device, cfg['train']['precision']):
                    means.append(conditioner(b).float())
        self.means = torch.cat(means)
        blended = torch.zeros((len(TARGETS), self.h+2*self.halo, self.w+2*self.halo), device=device)
        for k, (y, c) in enumerate(self.tiles):
            blended[:, y:y+self.width, c:c+self.width] += self.means[k]*self.window
        self.mean = (blended/self.weight)[:, self.halo:self.halo+self.h, self.halo:self.halo+self.w].cpu().numpy()
        self.rs = conditioner.rs[0].cpu().numpy()
        self.rm = conditioner.rm[0].cpu().numpy()
        self.scale = conditioner.flow_scale[0].cpu().numpy()
        regression = self.mean*self.rs+self.rm+self.coarse
        z = np.maximum(self.mean[1], 0)
        regression[1] = archive.scale*z*(z+2)
        self.regression = regression.astype('float32')

    @torch.no_grad()
    def velocity(self, state, time_value):
        result = torch.zeros_like(state)
        n, w = len(self.tiles), self.width
        for s in range(0, n, self.batch):
            tiles = self.tiles[s:s+self.batch]
            x = torch.stack([state[:, y:y+w, c:c+w] for y, c in tiles])
            t = torch.full((len(tiles),), float(time_value), device=self.device)
            with autocast(self.device, self.cfg['train']['precision']):
                value = self.model(x, t, self.condition[s:s+len(tiles)], self.context[s:s+len(tiles)],
                                   self.means[s:s+len(tiles)]).float()
            value = value*self.window
            for k, (y, c) in enumerate(tiles):
                result[:, y:y+w, c:c+w] += value[k]
        return result/self.weight

    @torch.no_grad()
    def sample(self, seed, steps=None):
        steps = steps or self.cfg['inference']['steps']
        noise = np.random.default_rng(seed).standard_normal(
            (len(TARGETS), self.h+2*self.halo, self.w+2*self.halo)).astype('float32')
        x = torch.from_numpy(noise).to(self.device)
        for i in range(steps):
            first = self.velocity(x, i/steps)
            second = self.velocity(x+first/steps, (i+1)/steps)
            x = x+(first+second)/(2*steps)
            if not bool(torch.isfinite(x).all()):
                raise FloatingPointError('Nonfinite full-field trajectory')
        core = x[:, self.halo:self.halo+self.h, self.halo:self.halo+self.w].cpu().numpy()
        value = (core*self.scale+self.mean)*self.rs+self.rm+self.coarse
        z = np.maximum(core[1], 0)
        value[1] = self.archive.scale*z*(z+2)  # direct rain, no add-back
        value[5] = np.clip(value[5], 0, 1)
        if not np.isfinite(value).all():
            raise FloatingPointError('Nonfinite physical output')
        return value.astype('float32')


# ----------------------------------------------------------------------------
# Maps
# ----------------------------------------------------------------------------

def _lcc(grid_path, ccrs):
    import xarray as xr
    if not Path(grid_path).exists():
        return None
    with xr.open_dataset(grid_path) as grid:
        name = grid.attrs.get('grid_mapping_variable')
        attrs = dict(grid[name].attrs) if name and name in grid else {}
    if attrs.get('grid_mapping_name') != 'lambert_conformal_conic':
        return None
    parallels = np.atleast_1d(attrs['standard_parallel']).astype(float).tolist()
    parallels = (parallels*2)[:2] if len(parallels) == 1 else parallels[:2]
    radius = attrs.get('earth_radius')
    globe = ccrs.Globe(semimajor_axis=float(radius), semiminor_axis=float(radius)) if radius else None
    return ccrs.LambertConformal(central_longitude=float(attrs['longitude_of_central_meridian']),
                                 central_latitude=float(attrs['latitude_of_projection_origin']),
                                 standard_parallels=tuple(parallels),
                                 false_easting=float(attrs.get('false_easting', 0.)),
                                 false_northing=float(attrs.get('false_northing', 0.)), globe=globe)


class Canvas:
    """Native-grid maps: LCC imshow when the grid is regular in projected
    coordinates, lon/lat pcolormesh otherwise, plain index plots without Cartopy."""

    def __init__(self, archive, use_cartopy=True, features=True, log=print):
        self.lat, self.lon = np.asarray(archive.static['lat']), np.asarray(archive.static['lon'])
        self.h, self.w = self.lat.shape
        self.mode, self.proj, self.ccrs = 'index', None, None
        self.features = False
        if use_cartopy:
            try:
                import cartopy.crs as ccrs
                self.ccrs = ccrs
            except ImportError:
                log('Cartopy unavailable: plotting on grid indices')
        if self.ccrs is None:
            return
        ccrs = self.ccrs
        proj = _lcc(archive.root/'grid_v2.nc', ccrs)
        if proj is not None:
            points = proj.transform_points(ccrs.PlateCarree(), self.lon, self.lat)
            x, y = points[..., 0], points[..., 1]
            xc, yc = np.nanmedian(x, axis=0), np.nanmedian(y, axis=1)
            dx, dy = np.nanmedian(np.diff(xc)), np.nanmedian(np.diff(yc))
            regular = (np.isfinite([dx, dy]).all() and dx != 0 and dy != 0
                       and np.nanmax(abs(x-xc[None])) < .5*abs(dx) and np.nanmax(abs(y-yc[:, None])) < .5*abs(dy))
            if regular:
                self.mode, self.proj = 'lcc', proj
                self.xc, self.yc, self.dx, self.dy = xc, yc, dx, dy
        if self.mode == 'index':
            self.mode = 'mesh'
            self.proj = ccrs.LambertConformal(central_longitude=float(np.nanmedian(self.lon)),
                                              central_latitude=float(np.nanmedian(self.lat)),
                                              standard_parallels=(33, 45))
        self.features = features and self._probe(log)
        log(f'Maps: {self.mode} projection; coastlines/states {"on" if self.features else "off"}')

    @staticmethod
    def _probe(log):
        previous = socket.getdefaulttimeout()
        socket.setdefaulttimeout(10)
        try:
            from cartopy.io import shapereader
            for category, name in (('physical', 'coastline'), ('cultural', 'admin_1_states_provinces_lakes'),
                                   ('cultural', 'admin_0_boundary_lines_land')):
                shapereader.natural_earth(resolution='50m', category=category, name=name)
            return True
        except Exception as exc:  # offline compute node without a Natural Earth cache
            log(f'Natural Earth shapes unavailable ({type(exc).__name__}); maps keep lat/lon gridlines only. '
                'Pass --cartopy-data-dir with a cache to draw coastlines and states.')
            return False
        finally:
            socket.setdefaulttimeout(previous)

    def axes(self, fig, spec):
        return fig.add_subplot(spec, projection=self.proj) if self.mode != 'index' else fig.add_subplot(spec)

    @staticmethod
    def region(region, shape):
        if region is None:
            return slice(0, shape[0]), slice(0, shape[1])
        return region

    def _extent(self, ys, xs):
        xa = (self.xc[xs.start]-self.dx/2, self.xc[xs.stop-1]+self.dx/2)
        ya = (self.yc[ys.start]-self.dy/2, self.yc[ys.stop-1]+self.dy/2)
        return (min(xa), max(xa), min(ya), max(ya))

    def show(self, ax, field, region=None, cmap=None, norm=None, left=False, bottom=False):
        ys, xs = self.region(region, field.shape)
        data = field[ys, xs]
        options = dict(cmap=cmap, norm=norm, interpolation='nearest', rasterized=True)
        if self.mode == 'lcc':
            if self.dx < 0:
                data = data[:, ::-1]
            extent = self._extent(ys, xs)
            image = ax.imshow(data, origin='lower' if self.dy > 0 else 'upper', extent=extent,
                              transform=self.proj, **options)
            ax.set_extent(extent, crs=self.proj)
        elif self.mode == 'mesh':
            step = max(1, max(data.shape)//900)
            pc = self.ccrs.PlateCarree()
            lon, lat = self.lon[ys, xs][::step, ::step], self.lat[ys, xs][::step, ::step]
            image = ax.pcolormesh(lon, lat, data[::step, ::step], transform=pc, cmap=cmap, norm=norm,
                                  shading='auto', rasterized=True)
            ax.set_extent([float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())], crs=pc)
        else:
            image = ax.imshow(data, origin='lower', aspect='auto', **options)
            ax.set_xticks([])
            ax.set_yticks([])
        self.decorate(ax, left, bottom)
        return image

    def decorate(self, ax, left=False, bottom=False):
        if self.mode == 'index':
            return
        if self.features:
            import cartopy.feature as cfeature
            ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=.6, edgecolor='#222222', zorder=3)
            ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth=.35, edgecolor='#444444',
                           linestyle=':', zorder=3)
            ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=.6, edgecolor='#222222', zorder=3)
        # Edge labels only: LCC otherwise gets inline labels scattered over the map.
        grid = ax.gridlines(draw_labels=left or bottom, linewidth=.3, color='0.35', alpha=.5, linestyle=':',
                            x_inline=False, y_inline=False, rotate_labels=False)
        if left or bottom:
            grid.top_labels = grid.right_labels = False
            grid.left_labels, grid.bottom_labels = left, bottom
            grid.xlabel_style = grid.ylabel_style = {'size': 7}

    def contour(self, ax, mask, region=None, **kwargs):
        ys, xs = self.region(region, mask.shape)
        data = mask[ys, xs].astype(float)
        if self.mode == 'lcc':
            xx = self.xc[xs] if self.dx > 0 else self.xc[xs]
            X, Y = np.meshgrid(xx, self.yc[ys])
            ax.contour(X, Y, data, levels=[.5], transform=self.proj, **kwargs)
        elif self.mode == 'mesh':
            ax.contour(self.lon[ys, xs], self.lat[ys, xs], data, levels=[.5],
                       transform=self.ccrs.PlateCarree(), **kwargs)
        else:
            ax.contour(data, levels=[.5], **kwargs)

    def box(self, ax, window, label, color='k'):
        from matplotlib.patches import Rectangle
        ys, xs = window
        if self.mode == 'lcc':
            x0, x1, y0, y1 = self._extent(ys, xs)
            ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, fill=False, ec=color, lw=1.3, transform=self.proj, zorder=5))
            ax.text(x0, y1, f' {label}', transform=self.proj, fontsize=8, va='bottom', color=color, zorder=5,
                    bbox=dict(fc='white', ec='none', alpha=.7, pad=1))
        elif self.mode == 'mesh':
            lon, lat = self.lon[ys, xs], self.lat[ys, xs]
            edge = np.concatenate([np.c_[lon[0], lat[0]], np.c_[lon[:, -1], lat[:, -1]],
                                   np.c_[lon[-1, ::-1], lat[-1, ::-1]], np.c_[lon[::-1, 0], lat[::-1, 0]]])
            ax.plot(edge[:, 0], edge[:, 1], color=color, lw=1.3, transform=self.ccrs.PlateCarree(), zorder=5)
            ax.text(lon[-1, 0], lat[-1, 0], f' {label}', transform=self.ccrs.PlateCarree(), fontsize=8, color=color)
        else:
            ax.add_patch(Rectangle((xs.start, ys.start), xs.stop-xs.start, ys.stop-ys.start, fill=False, ec=color, lw=1.3))
            ax.text(xs.start, ys.stop, f' {label}', fontsize=8, color=color)


def event_window(rain, size):
    """Window centred on the strongest broad rain feature (as in the v2 test)."""
    h, w = rain.shape
    zh, zw = min(h, size), min(w, size)
    step = max(1, min(h, w)//150)
    small = rain[:h//step*step, :w//step*step]
    small = small.reshape(small.shape[0]//step, step, small.shape[1]//step, step).mean((1, 3))
    cy, cx = np.unravel_index(np.argmax(small), small.shape)
    cy, cx = int((cy+.5)*step), int((cx+.5)*step)
    y0, x0 = int(np.clip(cy-zh//2, 0, h-zh)), int(np.clip(cx-zw//2, 0, w-zw))
    return slice(y0, y0+zh), slice(x0, x0+zw)


def random_windows(shape, size, count, seed):
    h, w = shape
    zh, zw = min(h, size), min(w, size)
    rng = np.random.default_rng(seed)
    result = []
    for _ in range(count):
        y0, x0 = int(rng.integers(0, h-zh+1)), int(rng.integers(0, w-zw+1))
        result.append((slice(y0, y0+zh), slice(x0, x0+zw)))
    return result


# ----------------------------------------------------------------------------
# Per-case metrics
# ----------------------------------------------------------------------------

def speed(fields):
    return np.hypot(fields[..., 3, :, :], fields[..., 4, :, :])


def categorical(pred, truth, area, threshold):
    p, t = pred >= threshold, truth >= threshold
    hit, miss, false = [weighted_mean(v, area) for v in (p & t, ~p & t, p & ~t)]
    return dict(csi=hit/(hit+miss+false) if hit+miss+false else None, pod=hit/(hit+miss) if hit+miss else None,
                far=false/(hit+false) if hit+false else None,
                frequency_bias=weighted_mean(p, area)/weighted_mean(t, area) if t.any() else None)


def case_metrics(ensemble, truth, coarse, regression, area, rng):
    report, diagnostics = {}, {}
    sources = dict(coarse=coarse[None], regression=regression[None], member_0=ensemble[:1], ensemble=ensemble)
    for c, name in enumerate(TARGETS):
        report[name] = {key: continuous(value[:, c], truth[c], area) for key, value in sources.items()}
    report['wind_speed'] = {key: continuous(speed(value), speed(truth), area) for key, value in sources.items()}
    rain, rain_truth = ensemble[:, 1], truth[1]
    report['precip_probabilistic'] = precipitation(rain, rain_truth, area)
    report['precip_categorical'] = {
        str(thr): {key: categorical(field, rain_truth, area, thr)
                   for key, field in (('ensemble_mean', rain.mean(0)), ('member_0', rain[0]),
                                      ('coarse', coarse[1]), ('regression', regression[1]))}
        for thr in THRESHOLDS}
    members = rain[:min(4, len(rain))]
    report['fss'] = {str(thr): {
        'ensemble_mean': [fss(rain.mean(0), rain_truth, thr, s, area) for s in FSS_SCALES],
        'members': [_mean([fss(m, rain_truth, thr, s, area) for m in members]) for s in FSS_SCALES],
        'coarse': [fss(coarse[1], rain_truth, thr, s, area) for s in FSS_SCALES],
        'regression': [fss(regression[1], rain_truth, thr, s, area) for s in FSS_SCALES]} for thr in FSS_THRESHOLDS}
    report['fss_scales_px'] = list(FSS_SCALES)
    report['domain_mean_rain_mm_h'] = dict(truth=weighted_mean(rain_truth, area), ensemble=weighted_mean(rain.mean(0), area),
                                           coarse=weighted_mean(coarse[1], area), regression=weighted_mean(regression[1], area))
    # Diagnostics kept for the summary figures (not in JSON).
    bins = len(RAIN_EDGES)-1
    def histogram(field):
        index = np.digitize(field, RAIN_EDGES[1:-1])
        return np.bincount(index.ravel(), np.broadcast_to(area, field.shape).ravel(), minlength=bins)/area.sum()
    diagnostics['hist'] = dict(truth=histogram(rain_truth), ensemble=histogram(rain)/len(rain),
                               coarse=histogram(coarse[1]), regression=histogram(regression[1]))
    diagnostics['ranks'] = np.array([rank_histogram(ensemble[:, c], truth[c], area, seed=c) for c in range(len(TARGETS))])
    spectra = {}
    for c, name in enumerate(TARGETS):
        freq, t_psd = radial_psd(truth[c])
        spectra[name] = dict(freq=freq, truth=t_psd,
                             members=np.mean([radial_psd(m)[1] for m in ensemble[:min(4, len(ensemble)), c]], axis=0),
                             ensemble_mean=radial_psd(ensemble[:, c].mean(0))[1],
                             coarse=radial_psd(coarse[c])[1], regression=radial_psd(regression[c])[1])
    diagnostics['spectra'] = spectra
    reliability = {}
    for thr in (1., 5.):
        probability = np.rint((rain >= thr).mean(0)*len(rain)).astype(int)
        observed = rain_truth >= thr
        reliability[str(thr)] = dict(area=np.bincount(probability.ravel(), area.ravel(), minlength=len(rain)+1),
                                     observed=np.bincount(probability.ravel(), (area*observed).ravel(), minlength=len(rain)+1))
    diagnostics['reliability'] = reliability
    pick = rng.choice(rain_truth.size, size=min(200_000, rain_truth.size), replace=False)
    diagnostics['qq'] = dict(truth=rain_truth.ravel()[pick], members=rain.reshape(len(rain), -1)[:, pick].ravel(),
                             coarse=coarse[1].ravel()[pick], regression=regression[1].ravel()[pick])
    return report, diagnostics


def _mean(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------

def _display(name, field, difference=False):
    _, _, offset, factor, _, _ = DISPLAY[name]
    return field*factor if difference else (field+offset)*factor


def _norm(name, pool):
    from matplotlib.colors import Normalize
    lo, hi = np.nanquantile(pool, [.005, .995])
    if DISPLAY[name][5]:
        hi = max(abs(lo), abs(hi))
        lo = -hi
    return Normalize(float(lo), float(max(hi, lo+1e-6)))


def _stamp(ax, text):
    ax.text(.02, .02, text, transform=ax.transAxes, fontsize=7.5, va='bottom', ha='left', zorder=6,
            bbox=dict(boxstyle='round,pad=.25', fc='white', ec='none', alpha=.85))


def plot_conus_precip(canvas, item, path, plt):
    from matplotlib.gridspec import GridSpec
    cmap, norm = _rain_norm()
    area = item['area']
    rain = item['ensemble'][:, 1]
    panels = [('Coarse input (GEOS-FP)', item['coarse'][1]), ('Frozen v2 regression', item['regression'][1]),
              ('Truth (HWT hourly)', item['truth'][1]), ('Ensemble mean', rain.mean(0)),
              ('Member 1', rain[0]), ('Member 2', rain[min(1, len(rain)-1)])]
    fig = plt.figure(figsize=(26, 10.2), constrained_layout=True)
    grid = GridSpec(2, 4, figure=fig)
    spots = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1)]
    image = None
    for (label, field), (r, c) in zip(panels, spots):
        ax = canvas.axes(fig, grid[r, c])
        image = canvas.show(ax, field, cmap=cmap, norm=norm, left=c == 0, bottom=r == 1)
        ax.set_title(label, fontsize=11)
        _stamp(ax, f'mean {weighted_mean(field, area):.3f} · max {float(field.max()):.1f} mm/h · '
                   f'wet {weighted_mean(field >= .1, area):.0%}')
        if label.startswith('Truth'):
            for window, name in item['windows']:
                canvas.box(ax, window, name)
    fig.colorbar(image, ax=fig.axes, location='bottom', shrink=.45, pad=.01, extend='both', ticks=RAIN_LEVELS,
                 label='Rain rate (mm h⁻¹)')
    ax = canvas.axes(fig, grid[1, 2])
    image = canvas.show(ax, (rain >= 1).mean(0), cmap='PuBuGn', norm=plt.Normalize(0, 1), bottom=True)
    canvas.contour(ax, item['truth'][1] >= 1, colors='k', linewidths=.5)
    ax.set_title('P(rain ≥ 1 mm/h) · black: observed ≥ 1', fontsize=11)
    fig.colorbar(image, ax=ax, shrink=.7, label='Probability')
    ax = canvas.axes(fig, grid[1, 3])
    spread = rain.std(0)
    image = canvas.show(ax, spread, cmap='magma_r', norm=plt.Normalize(0, max(float(np.quantile(spread, .998)), .1)),
                        bottom=True)
    ax.set_title('Ensemble spread (std)', fontsize=11)
    fig.colorbar(image, ax=ax, shrink=.7, label='mm h⁻¹', extend='max')
    m = item['metrics']['precip']
    fig.suptitle(f'{item["heading"]}\nRain: CRPS {m["ensemble"]["crps"]:.3f} vs coarse MAE {m["coarse"]["mae"]:.3f} '
                 f'vs regression MAE {m["regression"]["mae"]:.3f} mm/h · {len(rain)} members · boxes = zoom windows',
                 fontsize=13)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_conus_states(canvas, item, path, plt):
    from matplotlib.gridspec import GridSpec
    names = ['t2m', 'ps', 'u10m', 'v10m', 'q2m', 'wind_speed']
    fig = plt.figure(figsize=(26, 4.3*len(names)), constrained_layout=True)
    grid = GridSpec(len(names), 5, figure=fig)
    for r, name in enumerate(names):
        title, unit, *_ , cmap, _ = DISPLAY[name]
        if name == 'wind_speed':
            coarse, truth, ens = speed(item['coarse']), speed(item['truth']), speed(item['ensemble'])
        else:
            c = TARGETS.index(name)
            coarse, truth, ens = item['coarse'][c], item['truth'][c], item['ensemble'][:, c]
        mean = ens.mean(0)
        shown = [_display(name, v) for v in (coarse, truth, mean)]
        norm = _norm(name, np.stack(shown)[:, ::4, ::4])
        for j, (label, field) in enumerate(zip(('Coarse input', 'Truth', 'Ensemble mean'), shown)):
            ax = canvas.axes(fig, grid[r, j])
            image = canvas.show(ax, field, cmap=cmap, norm=norm, left=j == 0, bottom=r == len(names)-1)
            ax.set_title(f'{title} · {label}', fontsize=10)
        fig.colorbar(image, ax=fig.axes[-3:], shrink=.85, label=unit, pad=.01)
        error = _display(name, mean-truth, difference=True)
        bound = max(float(np.quantile(abs(error), .995)), 1e-6)
        ax = canvas.axes(fig, grid[r, 3])
        image = canvas.show(ax, error, cmap='RdBu_r', norm=plt.Normalize(-bound, bound), bottom=r == len(names)-1)
        stats = item['metrics'][name]
        factor = DISPLAY[name][3]
        _stamp(ax, f'RMSE {stats["ensemble"]["rmse"]*factor:.3g} · coarse {stats["coarse"]["rmse"]*factor:.3g} · '
                   f'bias {stats["ensemble"]["bias"]*factor:+.2g} {unit}')
        ax.set_title(f'{title} · mean − truth', fontsize=10)
        fig.colorbar(image, ax=ax, shrink=.85, label=unit)
        spread = _display(name, ens.std(0), difference=True)
        ax = canvas.axes(fig, grid[r, 4])
        image = canvas.show(ax, spread, cmap='magma_r', norm=plt.Normalize(0, max(float(np.quantile(spread, .995)), 1e-6)),
                            bottom=r == len(names)-1)
        ax.set_title(f'{title} · spread', fontsize=10)
        fig.colorbar(image, ax=ax, shrink=.85, label=unit, extend='max')
    fig.suptitle(f'{item["heading"]} · states (midpoint snapshot) and 10 m wind speed', fontsize=13)
    fig.savefig(path, dpi=95)
    plt.close(fig)


def plot_zoom(canvas, item, window, label, path, plt):
    from matplotlib.gridspec import GridSpec
    rain_cmap, rain_norm = _rain_norm()
    ys, xs = window
    columns = ('Coarse input', 'Frozen regression', 'Truth', 'Member 1', 'Ensemble mean', 'Mean − truth', 'Spread')
    names = list(TARGETS)
    fig = plt.figure(figsize=(29, 3.9*len(names)), constrained_layout=True)
    grid = GridSpec(len(names), len(columns), figure=fig)
    area = item['area'][ys, xs]
    for r, name in enumerate(names):
        c = TARGETS.index(name)
        title, unit, *_, cmap, _ = DISPLAY[name]
        ens = item['ensemble'][:, c]
        fields = [item['coarse'][c], item['regression'][c], item['truth'][c], ens[0], ens.mean(0)]
        bottom = r == len(names)-1
        if name == 'precip':
            cmap, norm = rain_cmap, rain_norm
            shown = fields
        else:
            shown = [_display(name, f) for f in fields]
            norm = _norm(name, np.stack([s[ys, xs] for s in shown]))
        for j, field in enumerate(shown):
            ax = canvas.axes(fig, grid[r, j])
            image = canvas.show(ax, field, window, cmap=cmap, norm=norm, left=j == 0, bottom=bottom)
            ax.set_title(f'{name} · {columns[j]}', fontsize=9.5)
            if name == 'precip':
                _stamp(ax, f'mean {weighted_mean(field[ys, xs], area):.2f} · max {float(field[ys, xs].max()):.1f}')
        fig.colorbar(image, ax=fig.axes[-5:], shrink=.85, pad=.01, label=unit,
                     **({'ticks': RAIN_LEVELS, 'extend': 'both'} if name == 'precip' else {}))
        error = _display(name, ens.mean(0)-item['truth'][c], difference=True)
        bound = max(float(np.quantile(abs(error[ys, xs]), .995)), 1e-6)
        ax = canvas.axes(fig, grid[r, 5])
        image = canvas.show(ax, error, window, cmap='RdBu_r', norm=plt.Normalize(-bound, bound), bottom=bottom)
        rmse = float(np.sqrt(weighted_mean((error[ys, xs])**2, area)))
        coarse_rmse = float(np.sqrt(weighted_mean(_display(name, item['coarse'][c]-item['truth'][c], True)[ys, xs]**2, area)))
        _stamp(ax, f'RMSE {rmse:.3g} (coarse {coarse_rmse:.3g}) {unit}')
        ax.set_title(f'{name} · {columns[5]}', fontsize=9.5)
        fig.colorbar(image, ax=ax, shrink=.85, label=unit)
        spread = _display(name, ens.std(0), difference=True)
        ax = canvas.axes(fig, grid[r, 6])
        image = canvas.show(ax, spread, window, cmap='magma_r',
                            norm=plt.Normalize(0, max(float(np.quantile(spread[ys, xs], .995)), 1e-6)), bottom=bottom)
        ax.set_title(f'{name} · {columns[6]}', fontsize=9.5)
        fig.colorbar(image, ax=ax, shrink=.85, label=unit, extend='max')
    cy, cx = (ys.start+ys.stop)//2, (xs.start+xs.stop)//2
    fig.suptitle(f'{item["heading"]}\n{label} · rows {ys.start}-{ys.stop}, cols {xs.start}-{xs.stop} · '
                 f'centre {float(canvas.lat[cy, cx]):.2f}°N {float(canvas.lon[cy, cx]):.2f}°E', fontsize=13)
    fig.savefig(path, dpi=90)
    plt.close(fig)


def plot_summary(reports, diagnostics, out, dx_km, members, plt):
    names = list(TARGETS)+['wind_speed']
    colors = dict(coarse='#9e9e9e', regression='#4575b4', member_0='#fdae61', ensemble='#d73027',
                  truth='k', members='#d73027', ensemble_mean='#f46d43')
    # 1. Scores
    fig, axes = plt.subplots(2, 4, figsize=(22, 9), constrained_layout=True)
    for ax, name in zip(axes.flat, names):
        factor = DISPLAY[name][3]
        keys = ('coarse', 'regression', 'member_0', 'ensemble')
        mae = [np.mean([r['metrics'][name][k]['mae'] for r in reports])*factor for k in keys]
        crps = np.mean([r['metrics'][name]['ensemble']['crps'] for r in reports])*factor
        bars = ax.bar(range(4), mae, color=[colors[k] for k in keys])
        ax.bar(4, crps, color='#67001f')
        ax.set_xticks(range(5), ['coarse\nMAE', 'regression\nMAE', 'member\nMAE', 'ens-mean\nMAE', 'ensemble\nCRPS'], fontsize=8)
        ratio = np.mean([r['metrics'][name]['ensemble']['spread']/max(r['metrics'][name]['ensemble']['rmse'], 1e-12)
                         for r in reports])
        ax.set_title(f'{DISPLAY[name][0]} ({DISPLAY[name][1]})\nCRPS skill vs coarse {1-crps/max(mae[0], 1e-12):+.0%} · '
                     f'spread/RMSE {ratio:.2f}', fontsize=10)
        ax.bar_label(bars, fmt='%.3g', fontsize=7)
    axes.flat[-1].axis('off')
    fig.suptitle(f'v4.1 evaluation · mean over {len(reports)} cases · {members} members · lower is better', fontsize=13)
    fig.savefig(out/'summary_scores.png', dpi=110)
    plt.close(fig)
    # 2. Precipitation
    fig, axes = plt.subplots(2, 3, figsize=(21, 12), constrained_layout=True)
    ax = axes[0, 0]
    labels = [f'{a:g}–{b:g}' if np.isfinite(b) else f'≥{a:g}' for a, b in zip(RAIN_EDGES[:-1], RAIN_EDGES[1:])]
    labels[0] = '<0.1'
    for key, style in (('truth', dict(color='k', lw=2.4)), ('ensemble', dict(color='#d73027', lw=2)),
                       ('regression', dict(color='#4575b4', lw=1.5, ls='--')), ('coarse', dict(color='#9e9e9e', lw=1.5, ls=':'))):
        share = np.mean([d['hist'][key] for d in diagnostics], axis=0)
        ax.step(range(len(labels)), np.where(share > 0, share, np.nan), where='mid', label=key, **style)
    ax.set(yscale='log', xticks=range(len(labels)), xlabel='Rain rate (mm h⁻¹)', ylabel='Fraction of CONUS area',
           title='Rain-intensity distribution')
    ax.set_xticklabels(labels, rotation=35, fontsize=8)
    ax.legend()
    ax = axes[0, 1]
    pooled = {k: np.concatenate([d['qq'][k] for d in diagnostics]) for k in ('truth', 'members', 'coarse', 'regression')}
    tq = np.quantile(pooled['truth'], QUANTILES)
    for key, style in (('members', dict(color='#d73027', marker='o')), ('regression', dict(color='#4575b4', marker='s')),
                       ('coarse', dict(color='#9e9e9e', marker='^'))):
        ax.plot(tq, np.quantile(pooled[key], QUANTILES), label=key, ms=5, **style)
    top = max(float(tq.max()), 1.)
    ax.plot([0, top], [0, top], 'k--', lw=.8)
    for q, x in zip(QUANTILES, tq):
        if q >= .99:
            ax.annotate(f'{q:g}', (x, x), fontsize=7, textcoords='offset points', xytext=(3, -10))
    ax.set(xlabel='Truth quantile (mm h⁻¹)', ylabel='Model quantile (mm h⁻¹)', title='Rain Q-Q (tails: 99th-99.99th pct)')
    ax.legend()
    ax = axes[0, 2]
    for thr, color in (('1.0', '#2166ac'), ('5.0', '#b2182b')):
        area_sum = np.sum([d['reliability'][thr]['area'] for d in diagnostics], axis=0)
        observed = np.sum([d['reliability'][thr]['observed'] for d in diagnostics], axis=0)
        k = np.arange(len(area_sum))/(len(area_sum)-1)
        valid = area_sum > 0
        ax.plot(k[valid], observed[valid]/area_sum[valid], marker='o', color=color, label=f'≥ {float(thr):g} mm/h')
    ax.plot([0, 1], [0, 1], 'k--', lw=.8)
    ax.set(xlabel='Forecast probability (member fraction)', ylabel='Observed frequency', xlim=(0, 1), ylim=(0, 1),
           title='Reliability (area-weighted)')
    ax.legend()
    scales_km = np.array(FSS_SCALES)*dx_km
    for ax, thr in zip((axes[1, 0], axes[1, 1]), ('1.0', '5.0')):
        for key, style in (('ensemble_mean', dict(color='#f46d43', lw=2)), ('members', dict(color='#d73027')),
                           ('regression', dict(color='#4575b4', ls='--')), ('coarse', dict(color='#9e9e9e', ls=':'))):
            values = [[r['metrics']['fss'][thr][key][i] for r in reports] for i in range(len(FSS_SCALES))]
            ax.plot(scales_km, [_mean(v) if _mean(v) is not None else np.nan for v in values], marker='o', label=key, **style)
        ax.set(xscale='log', xlabel='Neighbourhood (km)', ylabel='FSS', ylim=(0, 1),
               title=f'Fractions skill score, rain ≥ {float(thr):g} mm/h')
        ax.legend()
    ax = axes[1, 2]
    x = np.arange(len(reports))
    for offset, key, color in ((-.3, 'truth', 'k'), (-.1, 'ensemble', '#d73027'), (.1, 'regression', '#4575b4'),
                               (.3, 'coarse', '#9e9e9e')):
        ax.bar(x+offset, [r['metrics']['domain_mean_rain_mm_h'][key] for r in reports], width=.19, color=color, label=key)
    ax.set_xticks(x, [r['id'] for r in reports], rotation=30, fontsize=8)
    ax.set(ylabel='CONUS mean rain (mm h⁻¹)', title='Domain-mean rain per case')
    ax.set_ylim(0, ax.get_ylim()[1]*1.18)
    ax.legend(loc='upper center', ncol=4, fontsize=8)
    fig.suptitle('v4.1 evaluation · precipitation diagnostics', fontsize=13)
    fig.savefig(out/'summary_precip.png', dpi=110)
    plt.close(fig)
    # 3. Categorical scores vs threshold (appended into the precip figure's sibling)
    fig, axes = plt.subplots(1, 4, figsize=(22, 5), constrained_layout=True)
    for ax, score in zip(axes, ('csi', 'pod', 'far', 'frequency_bias')):
        for key, color in (('ensemble_mean', '#f46d43'), ('member_0', '#d73027'), ('regression', '#4575b4'), ('coarse', '#9e9e9e')):
            ax.plot(THRESHOLDS, [_mean([r['metrics']['precip_categorical'][str(t)][key][score] for r in reports])
                                 or np.nan for t in THRESHOLDS], marker='o', color=color, label=key)
        if score == 'frequency_bias':
            ax.axhline(1, color='k', lw=.8, ls='--')
        ax.set(xscale='log', xlabel='Threshold (mm h⁻¹)', title=score.upper().replace('_', ' '))
    axes[0].legend()
    fig.suptitle('v4.1 evaluation · rain categorical scores (mean over cases)', fontsize=13)
    fig.savefig(out/'summary_categorical.png', dpi=110)
    plt.close(fig)
    # 4. Spectra
    fig, axes = plt.subplots(2, 3, figsize=(21, 11), constrained_layout=True)
    for ax, name in zip(axes.flat, TARGETS):
        freq = diagnostics[0]['spectra'][name]['freq']
        wavelength = dx_km/np.maximum(freq, 1e-9)
        valid = freq > 0
        for key, style in (('truth', dict(color='k', lw=2.4)), ('members', dict(color='#d73027', lw=1.8)),
                           ('ensemble_mean', dict(color='#f46d43', lw=1.2, ls='-.')),
                           ('regression', dict(color='#4575b4', lw=1.3, ls='--')), ('coarse', dict(color='#9e9e9e', lw=1.3, ls=':'))):
            psd = np.mean([d['spectra'][name][key] for d in diagnostics], axis=0)
            ax.loglog(wavelength[valid], np.maximum(psd[valid], 1e-30), label=key, **style)
        ax.invert_xaxis()
        ax.set(xlabel='Wavelength (km)', ylabel=f'PSD ({UNITS[TARGETS.index(name)]})²', title=f'{name}: radial power spectrum')
        ax.legend(fontsize=8)
    fig.suptitle('v4.1 evaluation · full-CONUS spectra (members should track truth at small scales)', fontsize=13)
    fig.savefig(out/'summary_spectra.png', dpi=110)
    plt.close(fig)
    # 5. Calibration
    fig, axes = plt.subplots(1, 2, figsize=(18, 6), constrained_layout=True)
    ranks = np.mean([d['ranks'] for d in diagnostics], axis=0)
    for c, name in enumerate(TARGETS):
        axes[0].plot(np.arange(ranks.shape[1]), ranks[c]*ranks.shape[1], marker='o', ms=3, label=name)
    axes[0].axhline(1, color='k', lw=.8, ls='--')
    axes[0].set(xlabel='Rank of truth among members', ylabel='Relative frequency (flat = 1)',
                title='Rank histograms (U: under-dispersed, ∩: over-dispersed, slope: bias)')
    axes[0].legend(ncol=3)
    for name in names:
        factor = DISPLAY[name][3]
        rmse = [r['metrics'][name]['ensemble']['rmse']/max(r['metrics'][name]['coarse']['rmse'], 1e-12) for r in reports]
        spread = [r['metrics'][name]['ensemble']['spread']/max(r['metrics'][name]['coarse']['rmse'], 1e-12) for r in reports]
        axes[1].scatter(rmse, spread, label=name, s=30)
    lim = axes[1].get_xlim()[1]
    axes[1].plot([0, lim], [0, lim], 'k--', lw=.8)
    axes[1].set(xlabel='Ensemble-mean RMSE / coarse RMSE', ylabel='Spread / coarse RMSE',
                title='Spread-skill per case (on the diagonal = calibrated)')
    axes[1].legend(ncol=2)
    fig.savefig(out/'summary_calibration.png', dpi=110)
    plt.close(fig)


def plot_case_mean_maps(canvas, sums, count, out, plt):
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(24, 12.5), constrained_layout=True)
    grid = GridSpec(2, 3, figure=fig)
    panels = []
    for name in ('t2m', 'precip', 'q2m'):
        panels.append((f'{DISPLAY[name][0]}: mean(ensemble − truth)', _display(name, sums['bias'][name]/count, True),
                       'RdBu_r', True, DISPLAY[name][1]))
    crps, coarse = sums['crps_precip']/count, sums['coarse_ae_precip']/count
    skill = 1-crps/np.maximum(coarse, 1e-3)
    panels += [('Rain CRPS (mean over cases)', crps, 'magma_r', False, 'mm h⁻¹'),
               ('Rain coarse-input MAE (mean over cases)', coarse, 'magma_r', False, 'mm h⁻¹'),
               ('Rain CRPS skill vs coarse (1 − CRPS/MAE)', np.where(coarse > .01, skill, np.nan), 'PiYG', True, '')]
    for k, (title, field, cmap, symmetric, unit) in enumerate(panels):
        ax = canvas.axes(fig, grid[k//3, k % 3])
        finite = field[np.isfinite(field)]
        if symmetric:
            bound = max(float(np.quantile(abs(finite), .99)) if finite.size else 1., 1e-6)
            if title.startswith('Rain CRPS skill'):
                bound = 1.
            norm = plt.Normalize(-bound, bound)
        else:
            norm = plt.Normalize(0, max(float(np.quantile(finite, .995)) if finite.size else 1., 1e-6))
        image = canvas.show(ax, field, cmap=cmap, norm=norm, left=k % 3 == 0, bottom=k // 3 == 1)
        ax.set_title(title, fontsize=11)
        fig.colorbar(image, ax=ax, shrink=.75, label=unit)
    fig.suptitle(f'v4.1 evaluation · case-mean maps over {count} cases (few cases: indicative only)', fontsize=13)
    fig.savefig(out/'summary_maps.png', dpi=100)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------

def write_report(out, metrics):
    rows = []
    for name in list(TARGETS)+['wind_speed']:
        factor, unit = DISPLAY[name][3], DISPLAY[name][1]
        mean = lambda k, s: np.mean([c['metrics'][name][k][s] for c in metrics['cases']])*factor
        crps, coarse = mean('ensemble', 'crps'), mean('coarse', 'mae')
        rows.append(f'| {name} ({unit}) | {coarse:.4g} | {mean("regression", "mae"):.4g} | {mean("ensemble", "mae"):.4g} | '
                    f'{crps:.4g} | {1-crps/max(coarse, 1e-12):+.1%} | {mean("ensemble", "spread")/max(mean("ensemble", "rmse"), 1e-12):.2f} |')
    lines = [f'# v4.1 evaluation — {metrics["checkpoint_label"]}', '',
             f'Checkpoint `{metrics["checkpoint"]}` (epoch {metrics["checkpoint_epoch"]}, sha256 {metrics["checkpoint_sha256"][:12]}), '
             f'split **{metrics["split"]}**, {metrics["members"]} members, {metrics["ode_steps"]} Heun steps.', '',
             '| Variable | Coarse MAE | Regression MAE | Ens-mean MAE | Ensemble CRPS | CRPS skill vs coarse | Spread/RMSE |',
             '|---|---:|---:|---:|---:|---:|---:|', *rows, '', '## Cases', '']
    for case in metrics['cases']:
        rain = case['metrics']['domain_mean_rain_mm_h']
        lines.append(f'- `{case["id"]}` ({case["reason"]}): CONUS-mean rain truth {rain["truth"]:.3f}, '
                     f'ensemble {rain["ensemble"]:.3f}, coarse {rain["coarse"]:.3f} mm/h → `cases/{case["id"]}/`')
    lines += ['', 'Figures: `summary_scores.png`, `summary_precip.png`, `summary_categorical.png`, `summary_spectra.png`, '
              '`summary_calibration.png`, `summary_maps.png`; per case `conus_precip.png`, `conus_states.png`, `zoom_*.png`.',
              '', 'A handful of hours is a case study, not population skill; use more `--samples` for stable averages.']
    (out/'report.md').write_text('\n'.join(lines)+'\n')


def save_fields(path, archive, item):
    import xarray as xr
    data = {}
    for c, name in enumerate(TARGETS):
        for key, value in (('truth', item['truth'][c]), ('coarse', item['coarse'][c]), ('regression', item['regression'][c]),
                           ('ensemble_mean', item['ensemble'][:, c].mean(0)), ('ensemble_spread', item['ensemble'][:, c].std(0))):
            data[f'{name}_{key}'] = (('Ydim', 'Xdim'), value.astype('float32'), {'units': UNITS[c]})
    data['lat'] = (('Ydim', 'Xdim'), np.asarray(archive.static['lat'], dtype='float32'), {'units': 'degrees_north'})
    data['lon'] = (('Ydim', 'Xdim'), np.asarray(archive.static['lon'], dtype='float32'), {'units': 'degrees_east'})
    ds = xr.Dataset(data, attrs=dict(version='v4.1', time=item['time'], members=len(item['ensemble'])))
    ds.to_netcdf(path, engine='h5netcdf', encoding={k: dict(zlib=True) for k in data})


def evaluate(cfg, checkpoint='best', output=None, split='test', samples=3, wettest=1, timestamps=None,
             members=8, steps=None, zooms=2, zoom_size=256, seed=317, batch=32, threads=8,
             use_cartopy=True, map_features=True, save=False, log=print):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from .validation_v4_1 import _style
    _style(plt)
    if members < 2:
        raise ValueError('Use at least two members')
    path = resolve_checkpoint(cfg, checkpoint)
    device = device_for(cfg['train']['device'])
    archive, model, conditioner, saved = load_model(cfg, path, device)
    digest = file_hash_v2(path)
    label = checkpoint if not Path(str(checkpoint)).is_file() else path.stem
    out = Path(output) if output else (Path(cfg['train']['output'])/'evaluation'/
                                      f'{path.stem}_{split}_m{members}_{digest[:12]}')
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'{out} is not empty; pass a fresh --output')
    out.mkdir(parents=True, exist_ok=True)
    steps = steps or cfg['inference']['steps']
    log(f'Checkpoint {path} (epoch {saved["epoch"]+1}); device {device}; {members} members × {steps} steps; output {out}')
    cases = select_cases(archive, split, timestamps, samples, wettest, seed, log)
    log('Cases: '+', '.join(f'{c["entry"]["id"]} ({c["reason"]})' for c in cases))
    canvas = Canvas(archive, use_cartopy, map_features, log)
    area = np.asarray(archive.static['area'], dtype='float64')
    dx_km = float(np.sqrt(np.median(area))/1000)
    reports, diagnostics = [], []
    sums = dict(bias={n: np.zeros(archive.shape) for n in ('t2m', 'precip', 'q2m')},
                crps_precip=np.zeros(archive.shape), coarse_ae_precip=np.zeros(archive.shape))
    rng = np.random.default_rng(seed)
    for number, case in enumerate(cases, 1):
        entry = case['entry']
        started = time.monotonic()
        sampler = DomainSampler(model, conditioner, archive, entry, cfg, device, batch, threads)
        log(f'[{number}/{len(cases)}] {entry["id"]}: {len(sampler.tiles)} tiles ready in {time.monotonic()-started:.0f}s')
        ensemble = []
        for m in range(members):
            t0 = time.monotonic()
            ensemble.append(sampler.sample(member_seed(cfg, entry, m), steps))
            log(f'  member {m+1}/{members}: {time.monotonic()-t0:.0f}s')
        ensemble = np.stack(ensemble)
        truth = np.asarray(archive.physical_truth(entry), dtype='float32')
        coarse, regression = sampler.coarse, sampler.regression
        del sampler
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        report, diag = case_metrics(ensemble, truth, coarse, regression, area, rng)
        for c, name in ((0, 't2m'), (1, 'precip'), (5, 'q2m')):
            sums['bias'][name] += ensemble[:, c].mean(0)-truth[c]
        sums['crps_precip'] += crps_ensemble(ensemble[:, 1], truth[1])
        sums['coarse_ae_precip'] += abs(coarse[1]-truth[1])
        windows = [(event_window(truth[1], zoom_size), 'event')]
        windows += [(w, f'random {k+1}') for k, w in enumerate(random_windows(
            archive.shape, zoom_size, zooms, np.random.SeedSequence([seed, int(entry['id'].replace('_', ''))])))]
        item = dict(truth=truth, coarse=coarse, regression=regression, ensemble=ensemble, area=area,
                    time=entry['time'], metrics=report,
                    windows=[(w, 'E' if name == 'event' else f'Z{name.split()[-1]}') for w, name in windows],
                    heading=f'v4.1 · epoch {saved["epoch"]+1} ({label}) · {split} {entry["time"].replace("T", " ")[:16]} UTC · '
                            f'{case["reason"]}')
        folder = out/'cases'/entry['id']
        folder.mkdir(parents=True, exist_ok=True)
        plot_conus_precip(canvas, item, folder/'conus_precip.png', plt)
        plot_conus_states(canvas, item, folder/'conus_states.png', plt)
        for window, name in windows:
            file = 'zoom_event.png' if name == 'event' else f'zoom_random_{name.split()[-1]}.png'
            plot_zoom(canvas, item, window, f'{"Rain-event" if name == "event" else "Random"} zoom ({name})', folder/file, plt)
        if save:
            save_fields(folder/'fields.nc', archive, item)
        reports.append(dict(id=entry['id'], time=entry['time'], reason=case['reason'], metrics=report,
                            windows={name: dict(rows=[w[0].start, w[0].stop], cols=[w[1].start, w[1].stop])
                                     for w, name in windows}, seconds=round(time.monotonic()-started, 1)))
        diagnostics.append(diag)
        log(f'[{number}/{len(cases)}] {entry["id"]} done in {time.monotonic()-started:.0f}s → {folder}')
        del ensemble, item
    plot_summary(reports, diagnostics, out, dx_km, members, plt)
    plot_case_mean_maps(canvas, sums, len(reports), out, plt)
    metrics = dict(version='v4.1', checkpoint=str(path.resolve()), checkpoint_label=str(label),
                   checkpoint_sha256=digest, checkpoint_epoch=saved['epoch']+1, split=split, members=members,
                   ode_steps=steps, inference_seed=cfg['inference']['seed'], map_mode=canvas.mode, grid_km=dx_km,
                   cases=reports)
    write_json(out/'metrics_v4_1.json', json.loads(json.dumps(metrics, default=float)))
    write_report(out, metrics)
    log(f'v4.1 evaluation written to {out}')
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', default='configs/discover_v4_1.yaml')
    parser.add_argument('--checkpoint', default='best', help='best (default) | latest | <epoch number> | <path>')
    parser.add_argument('--latest', action='store_true', help='Shortcut for --checkpoint latest')
    parser.add_argument('--output', help='Fresh output directory (default under <train.output>/evaluation/)')
    parser.add_argument('--split', choices=('val', 'test'), default='test')
    parser.add_argument('--samples', type=int, default=3, help='Seeded random hours')
    parser.add_argument('--wettest', type=int, default=1, help='Also include the N wettest hours (≥12 h apart)')
    parser.add_argument('--timestamps', nargs='+', help='Exact IDs or ISO times instead of --samples/--wettest')
    parser.add_argument('--members', type=int, default=8)
    parser.add_argument('--steps', type=int, help='Heun steps (default: inference.steps)')
    parser.add_argument('--zooms', type=int, default=2, help='Random zoom windows per case (plus one rain-event zoom)')
    parser.add_argument('--zoom-size', type=int, default=256, help='Zoom window edge in grid pixels')
    parser.add_argument('--seed', type=int, default=317, help='Case and zoom selection seed')
    parser.add_argument('--batch', type=int, default=32, help='Tiles per GPU forward pass')
    parser.add_argument('--threads', type=int, default=8, help='CPU threads for building tile inputs')
    parser.add_argument('--cartopy-data-dir', help='Natural Earth cache for offline coastlines/states')
    parser.add_argument('--no-cartopy', action='store_true', help='Plot on grid indices')
    parser.add_argument('--no-map-features', action='store_true', help='Skip coastlines/states')
    parser.add_argument('--save-fields', action='store_true', help='Write truth/coarse/regression/mean/spread NetCDF per case')
    args = parser.parse_args()
    if args.cartopy_data_dir:
        import cartopy
        cartopy.config['pre_existing_data_dir'] = args.cartopy_data_dir
    cfg = load_config(args.config)
    evaluate(cfg, 'latest' if args.latest else args.checkpoint, args.output, args.split, args.samples, args.wettest,
             args.timestamps, args.members, args.steps, args.zooms, args.zoom_size, args.seed, args.batch, args.threads,
             not args.no_cartopy, not args.no_map_features, args.save_fields,
             log=lambda message: print(message, flush=True))


if __name__ == '__main__':
    main()
