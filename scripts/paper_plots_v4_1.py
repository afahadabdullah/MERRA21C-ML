"""Publication figures for the MERRA-Flow v4.1 paper: maps first, downscaling gain over GEOS-FP.

Lives in scripts/ (not src/merraflow) on purpose: the running full-test evaluator
hashes every src/merraflow/*.py file, so new plotting code must stay outside it.

Inputs are only what the frozen evaluation already saved:
  * summary.json / cases/*.json        aggregate and per-hour scores
  * fields/<id>_{raw,spectral}.npz     predeclared example hours (truth, coarse,
                                       regression, members, native GEOS-FP cells)
  * fields/grid.npz                    lat / lon / cell area
  * optional storm_fields.nc           from merraflow.storm_animation_v4_1
No model, checkpoint, GPU, Cartopy or network access is needed.

Map conventions
  * Every map is drawn with imshow(interpolation='nearest') on the native HWT
    Lambert-conformal array, so each grid cell is a pixel and the projection is the
    grid itself. State/coast lines are projected into grid-index space.
  * GEOS-FP panels show the native 0.25 x 0.3125 degree cells: each fine pixel takes
    the value of the native cell that contains it (no interpolation), so the coarse
    blocks stay visible. Zoom panels outline the native cells.
  * Full-CONUS panels are block-averaged by DISPLAY_BLOCK (2 x 2 -> ~6 km) for display
    and file size only; zoom panels use every native 3 km pixel.
  * Zoom boxes are fixed before plotting: a fixed terrain box, and a box centred on
    the heaviest smoothed GEOS-FP (input) precipitation. Neither the reference nor any
    model output or score is used to choose what is shown.
"""
import gzip
import json
import re
from pathlib import Path
import struct

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
import matplotlib.patheffects as pe
import matplotlib.ticker
import numpy as np

FIELDS = ('t2m', 'precip', 'ps', 'u10m', 'v10m', 'q2m', 'wind_speed')
SHORT = ('Temperature', 'Precipitation', 'Surface pressure', 'Zonal wind', 'Meridional wind',
         'Specific humidity', 'Wind speed')
UNITS = ('K', r'mm h$^{-1}$', 'hPa', r'm s$^{-1}$', r'm s$^{-1}$', r'g kg$^{-1}$', r'm s$^{-1}$')
FACTORS = (1., 1., .01, 1., 1., 1000., 1.)
METHODS = ('coarse', 'regression', 'raw', 'spectral')
NAMES = dict(coarse='GEOS-FP (interp.)', regression='Regression', raw='MERRA-Flow',
             spectral='MERRA-Flow + spectral', reference='GEOS-CAM (reference)')
COLORS = dict(coarse='#8a8f98', regression='#c7882b', raw='#1d6fb8', spectral='#cc3d5a', reference='#111111')
INK, MUTED, GRID = '#1f2933', '#5f6b7a', '#e4e8ee'
DISPLAY_BLOCK = 2
NATIVE_LABEL = 'GEOS-FP (~25 km cells)'
TERRAIN_BOX = (-110.0, -103.5, 36.0, 41.5)   # Colorado Rockies / Front Range (lon0, lon1, lat0, lat1)
ZOOM_PX = 200                                 # ~600 km zoom for the input-centred rain box

# Display definitions for the map variables. c: channel in the saved arrays (6 = wind speed).
VARS = {
    't2m': dict(c=0, title='2-m temperature', unit='°C', offset=-273.15, factor=1.),
    'wind_speed': dict(c=6, title='10-m wind speed', unit=r'm s$^{-1}$', offset=0., factor=1.),
    'precip': dict(c=1, title='Precipitation rate', unit=r'mm h$^{-1}$', offset=0., factor=1.),
    'q2m': dict(c=5, title='2-m specific humidity', unit=r'g kg$^{-1}$', offset=0., factor=1000.),
    'ps': dict(c=2, title='Surface pressure', unit='hPa', offset=0., factor=.01),
    'u10m': dict(c=3, title='10-m zonal wind', unit=r'm s$^{-1}$', offset=0., factor=1.),
    'v10m': dict(c=4, title='10-m meridional wind', unit=r'm s$^{-1}$', offset=0., factor=1.),
}
RAIN_TICK_STEP = 5.
RAIN_DISPLAY_QUANTILE = .98
RAIN_WET_THRESHOLD = .1
RAIN_COLORS = ['#d7ecfb', '#a6d3f2', '#5fb0e5', '#2c7fc9', '#1f9e5a', '#7cc644', '#f2d43a', '#f29434',
               '#de3b2f', '#a8166f']
# Display-only neutral bands, in physical display units or percentage points.
GAIN_WHITE_BAND = dict(t2m=.1, wind_speed=.1, precip=.1, q2m=.05)
SKILL_WHITE_BAND = 5.
SKILL_COLOR_LIMIT = 80.


# ----------------------------------------------------------------------------- style / export

def style():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 7.5, 'axes.titlesize': 8, 'axes.labelsize': 7.5,
        'legend.fontsize': 7, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'axes.linewidth': .55,
        'axes.edgecolor': '#4a5560', 'axes.labelcolor': INK, 'xtick.color': '#4a5560', 'ytick.color': '#4a5560',
        'text.color': INK, 'axes.spines.top': False, 'axes.spines.right': False,
        'xtick.major.width': .5, 'ytick.major.width': .5, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
        'lines.linewidth': 1.4, 'legend.frameon': False, 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'svg.fonttype': 'none', 'savefig.dpi': 300, 'figure.dpi': 100, 'mathtext.default': 'regular'})


def export(fig, out, stem, dpi=300, subject=None):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    meta = {'Title': f'MERRA-Flow {stem}', 'Subject': subject or 'MERRA-Flow v4.1'}
    if subject and '(interim, completion-ordered)' in subject:
        count = re.search(r'\d+/\d+', subject).group()
        bottom = min([ax.get_position().y0 for ax in fig.axes] or [0])
        fig.text(.5, min(bottom-.1, -.025), f'INTERIM: {count} held-out hours; completion-ordered subset',
                 ha='center', va='top', fontsize=6, color=MUTED)
    fig.savefig(out/f'{stem}.pdf', dpi=dpi, bbox_inches='tight', pad_inches=.02, facecolor='white', metadata=meta)
    fig.savefig(out/f'{stem}.png', dpi=dpi, bbox_inches='tight', pad_inches=.02, facecolor='white')
    plt.close(fig)
    return stem


def letter(ax, i, text=None, inside=True, color=INK):
    s = f'{chr(97+i)}' if text is None else f'{chr(97+i)}  {text}'
    if inside:
        ax.text(.025, .965, chr(97+i), transform=ax.transAxes, ha='left', va='top', fontsize=8.5,
                weight='bold', color=color, zorder=20,
                path_effects=[pe.withStroke(linewidth=2.2, foreground='white')])
    else:
        ax.set_title(s, loc='left', fontsize=8, weight='bold', pad=4)


def precip_scale(*fields, signed=False, step=RAIN_TICK_STEP):
    """Cover every finite displayed value with a linear range and uniform ticks."""
    peak = 0.
    for field in fields:
        values = np.asanyarray(field)
        finite = values[np.isfinite(values)]
        if finite.size:
            peak = max(peak, float(np.max(np.abs(finite) if signed else finite)))
    tick_step = step*max(1, np.ceil(peak/(10*step)))
    maximum = tick_step*max(1, np.ceil(peak/tick_step))
    minimum = -maximum if signed else 0.
    return mcolors.Normalize(minimum, maximum), np.arange(minimum, maximum+tick_step/2, tick_step)


def precip_ticks(norm, step=RAIN_TICK_STEP):
    if hasattr(norm, '_rain_ticks'):
        return norm._rain_ticks.copy()
    tick_step = step*max(1, np.ceil(norm.vmax/(10*step)))
    return np.arange(norm.vmin, norm.vmax+tick_step/2, tick_step)


def rain_display_scale(*fields):
    """Shared linear display range: largest per-panel 98th percentile of wet pixels.

    Rare extremes use the colorbar's upper extension; values and scores are unchanged.
    Excluding dry pixels prevents a mostly dry CONUS panel from collapsing the scale.
    """
    upper = 1.
    for field in fields:
        values = np.asanyarray(field)
        wet = values[np.isfinite(values) & (values >= RAIN_WET_THRESHOLD)]
        if wet.size:
            upper = max(upper, float(np.quantile(wet, RAIN_DISPLAY_QUANTILE)))
    locator = matplotlib.ticker.MaxNLocator(nbins=8, steps=[1, 2, 2.5, 5, 10])
    ticks = locator.tick_values(0, upper)
    ticks = ticks[ticks >= 0]
    norm = mcolors.Normalize(0, ticks[-1])
    norm._rain_ticks = ticks
    return norm, ticks


def rain_cmap(*fields):
    cmap = mcolors.LinearSegmentedColormap.from_list('rain_linear', ['white', *RAIN_COLORS])
    norm, _ = rain_display_scale(*fields)
    return cmap.with_extremes(under='white', over=RAIN_COLORS[-1]), norm


def precip_error_scale(limit, signed=False):
    """Dynamic linear precipitation error/difference scale; symmetric when signed."""
    return precip_scale(limit, signed=signed)


def difference_scale(vmin, vmax, white_band, cmap='RdBu'):
    """Keep a linear numeric range; paint small signed changes white without altering data."""
    norm = mcolors.Normalize(vmin, vmax)
    base = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
    values = np.linspace(vmin, vmax, 4097)
    colors = base(np.interp(values, [vmin, 0., vmax], [0., .5, 1.]))
    # Include the LUT cell touching each boundary so +/-white_band also stays white.
    colors[np.abs(values) <= white_band+(vmax-vmin)/4096.] = (1., 1., 1., 1.)
    result = mcolors.ListedColormap(colors, name='white_band_'+base.name)
    return result.with_extremes(bad=base.get_bad(), under=base.get_under(), over=base.get_over()), norm


def scale_for(name, reference):
    """Linear colour scale; precipitation map callers supply every compared panel."""
    if name == 'precip':
        cmap, norm = rain_cmap(reference); return cmap, norm, 'max'
    ref = np.asarray(reference, dtype='float64')
    lo, hi = np.nanquantile(ref, [.01, .99])
    if name in ('u10m', 'v10m'):
        m = max(abs(lo), abs(hi)); return 'RdBu_r', mcolors.Normalize(-m, m), 'both'
    if name == 'wind_speed':
        return wind_cmap(), mcolors.Normalize(0, max(np.nanquantile(ref, .995), 1)), 'max'
    if name == 'q2m':
        return 'BrBG', mcolors.Normalize(lo, hi), 'both'
    if name == 'ps':
        return 'cividis', mcolors.Normalize(lo, hi), 'both'
    return temp_cmap(), mcolors.Normalize(lo, hi), 'both'


def temp_cmap():
    return mcolors.LinearSegmentedColormap.from_list('temp', [
        '#2b1d6b', '#3b4cc0', '#5f8fd8', '#9cc7e8', '#e9f1f2', '#f6d48a', '#f19a4b', '#d8492f', '#8f1a26'])


def wind_cmap():
    return mcolors.LinearSegmentedColormap.from_list('wind', [
        '#f4f6f8', '#c9e3e8', '#86c5cf', '#4a9fbd', '#2f6fa8', '#3b3f8f', '#6a2a7a', '#a4245c'])


def spread_cmap():
    return mcolors.LinearSegmentedColormap.from_list('spread', ['#ffffff', '#efe3f3', '#c9a3d6', '#9061b3', '#5b2d8e', '#2a0f4f'])


def error_cmap():
    return mcolors.LinearSegmentedColormap.from_list('err', ['#ffffff', '#fde8c8', '#f7b267', '#e0603a', '#9e1f3d', '#3d0a2a'])


# ----------------------------------------------------------------------------- geography

def _wkb_lines(hexstring):
    """Minimal WKB decoder for (Multi)LineString/(Multi)Polygon; avoids a shapely dependency."""
    data = bytes.fromhex(hexstring); out = []

    def read(offset):
        order = '<' if data[offset] == 1 else '>'
        kind = struct.unpack_from(order+'I', data, offset+1)[0] % 1000; offset += 5
        if kind == 2:
            n = struct.unpack_from(order+'I', data, offset)[0]; offset += 4
            out.append(np.frombuffer(data, dtype=order+'f8', count=2*n, offset=offset).reshape(n, 2)); offset += 16*n
        elif kind == 3:
            rings = struct.unpack_from(order+'I', data, offset)[0]; offset += 4
            for _ in range(rings):
                n = struct.unpack_from(order+'I', data, offset)[0]; offset += 4
                out.append(np.frombuffer(data, dtype=order+'f8', count=2*n, offset=offset).reshape(n, 2)); offset += 16*n
        elif kind in (5, 6, 7):
            parts = struct.unpack_from(order+'I', data, offset)[0]; offset += 4
            for _ in range(parts):
                offset = read(offset)
        else:
            raise ValueError(f'Unsupported WKB geometry {kind}')
        return offset
    read(0)
    return out


def load_boundaries(extra=()):
    """Natural Earth coast/borders/states shipped with the repo (public domain)."""
    here = Path(__file__).resolve().parent
    for path in [*map(Path, extra), here/'conus_boundaries_50m.json.gz',
                 here.parent/'src'/'merraflow'/'assets'/'conus_boundaries_50m.json.gz']:
        if path.is_file():
            with gzip.open(path, 'rt') as f:
                geoms = json.load(f)['geometries']
            return {k: [line for g in v for line in _wkb_lines(g)] for k, v in geoms.items()}
    return {}


class Projector:
    """lon/lat -> fractional (col, row) on the HWT LCC array, without a projection library."""

    def __init__(self, lat, lon, stride=6):
        from scipy.interpolate import LinearNDInterpolator
        lat, lon = np.asarray(lat, dtype='float64'), np.asarray(lon, dtype='float64')
        rows, cols = np.mgrid[0:lat.shape[0]:stride, 0:lat.shape[1]:stride]
        # Include the last row/column so the hull covers the whole domain.
        rows = np.unique(np.r_[rows[:, 0], lat.shape[0]-1]); cols = np.unique(np.r_[cols[0], lat.shape[1]-1])
        rr, cc = np.meshgrid(rows, cols, indexing='ij')
        points = np.c_[lon[rr, cc].ravel(), lat[rr, cc].ravel()]
        self.f = LinearNDInterpolator(points, np.c_[cc.ravel(), rr.ravel()].astype('float64'))
        self.shape = lat.shape

    def __call__(self, lonlat):
        return self.f(np.asarray(lonlat, dtype='float64'))

    def segments(self, lines, box=None):
        """Projected polylines (NaN breaks outside the domain), clipped to an index box."""
        out = []
        for line in lines:
            xy = self(line)
            ok = np.isfinite(xy).all(1)
            if box is not None:
                r0, r1, c0, c1 = box
                ok &= (xy[:, 0] >= c0-2) & (xy[:, 0] <= c1+1) & (xy[:, 1] >= r0-2) & (xy[:, 1] <= r1+1)
            if ok.sum() < 2:
                continue
            breaks = np.flatnonzero(np.diff(ok.astype(int)) != 0)+1
            for part, good in zip(np.split(xy, breaks), np.split(ok, breaks)):
                if good[0] and len(part) > 1:
                    out.append(part)
        return out


class Geo:
    """Cached projected boundaries for one grid."""

    def __init__(self, lat, lon, boundaries):
        self.lat, self.lon = lat, lon
        self.proj = Projector(lat, lon) if boundaries else None
        self.full = {k: self.proj.segments(v) for k, v in boundaries.items()} if boundaries else {}

    def draw(self, ax, box=None, color='#2b2f36', scale=1.):
        if not self.full:
            return
        r0, r1, c0, c1 = box if box is not None else (0, self.lat.shape[0], 0, self.lat.shape[1])
        for name, width, alpha in (('states', .32, .55), ('borders', .45, .8), ('coast', .5, .85)):
            segs = [s for s in self.full.get(name, ()) if
                    (s[:, 0].max() >= c0 and s[:, 0].min() <= c1 and s[:, 1].max() >= r0 and s[:, 1].min() <= r1)]
            if segs:
                ax.add_collection(LineCollection(segs, colors=color, linewidths=width*scale, alpha=alpha, zorder=6))

    def box_of(self, lon0, lon1, lat0, lat1):
        """Index box (r0, r1, c0, c1) enclosing a lon/lat rectangle."""
        if self.proj is None:
            mask = (self.lon >= lon0) & (self.lon <= lon1) & (self.lat >= lat0) & (self.lat <= lat1)
            rr, cc = np.nonzero(mask)
            return int(rr.min()), int(rr.max())+1, int(cc.min()), int(cc.max())+1
        lo = np.linspace(lon0, lon1, 20); la = np.linspace(lat0, lat1, 20)
        edge = np.r_[np.c_[lo, np.full(20, lat0)], np.c_[lo, np.full(20, lat1)],
                     np.c_[np.full(20, lon0), la], np.c_[np.full(20, lon1), la]]
        xy = self.proj(edge); xy = xy[np.isfinite(xy).all(1)]
        h, w = self.lat.shape
        return (max(int(xy[:, 1].min()), 0), min(int(np.ceil(xy[:, 1].max())), h),
                max(int(xy[:, 0].min()), 0), min(int(np.ceil(xy[:, 0].max())), w))


# ----------------------------------------------------------------------------- data helpers

def channel(arr, c):
    arr = np.asarray(arr)
    return np.hypot(arr[..., 3, :, :], arr[..., 4, :, :]) if c == 6 else arr[..., c, :, :]


def physical(name, values):
    v = VARS[name]
    return np.asarray(values, dtype='float32')*v['factor']+v['offset']


def native_index(lat, lon, glat, glon):
    """Index of the native lat/lon cell containing every fine pixel (nearest centre)."""
    glat, glon = np.asarray(glat, dtype='float64'), np.asarray(glon, dtype='float64')
    dlat, dlon = np.median(np.diff(glat)), np.median(np.diff(glon))
    iy = np.clip(np.rint((lat-glat[0])/dlat).astype(int), 0, len(glat)-1)
    ix = np.clip(np.rint((lon-glon[0])/dlon).astype(int), 0, len(glon)-1)
    return iy, ix


def block(a, k):
    """k x k block mean for display (trailing partial blocks trimmed)."""
    if k <= 1:
        return a
    h, w = a.shape[-2]//k*k, a.shape[-1]//k*k
    a = a[..., :h, :w]
    return a.reshape(*a.shape[:-2], h//k, k, w//k, k).mean((-3, -1))


class Case:
    """One saved evaluation hour, with every displayed quantity in physical display units."""

    def __init__(self, raw_path, spectral_path, grid_path, time, case_id):
        self.id, self.time = case_id, time
        with np.load(grid_path, allow_pickle=False) as g:
            self.lat, self.lon = np.asarray(g['lat'], dtype='float64'), np.asarray(g['lon'], dtype='float64')
            self.area = np.asarray(g['area'], dtype='float64') if 'area' in g.files else np.ones_like(self.lat)
        with np.load(raw_path, allow_pickle=False) as f:
            self.truth, self.coarse, self.regression = f['truth'], f['coarse'], f['regression']
            self.ensemble = f['ensemble']
            self.native = {}
            for name in ('t2m', 'precip', 'ps', 'u10m', 'v10m', 'q2m'):
                if f'native_{name}_values' in f.files:
                    self.native[name] = {k: f[f'native_{name}_{k}'] for k in ('values', 'lat', 'lon')}
        self.spectral = None
        if spectral_path is not None and Path(spectral_path).is_file():
            with np.load(spectral_path, allow_pickle=False) as f:
                self.spectral = f['ensemble']
        self._idx = {}

    @property
    def members(self):
        return len(self.ensemble)

    def get(self, name, which, member=0):
        c = VARS[name]['c']
        if which == 'truth':
            return physical(name, channel(self.truth, c))
        if which == 'coarse':
            return physical(name, channel(self.coarse, c))
        if which == 'regression':
            return physical(name, channel(self.regression, c))
        if which == 'member':
            return physical(name, channel(self.ensemble[member], c))
        if which == 'spectral_member':
            return physical(name, channel((self.spectral if self.spectral is not None else self.ensemble)[member], c))
        if which == 'mean':
            return physical(name, channel(self.ensemble, c).mean(0)) if name != 'wind_speed' else \
                physical(name, channel(self.ensemble, c).mean(0))
        if which == 'spread':
            return channel(self.ensemble, c).std(0)*VARS[name]['factor']
        if which == 'native':
            return self.native_field(name)
        raise KeyError(which)

    def has_native(self, name):
        return name in self.native or (name == 'wind_speed' and {'u10m', 'v10m'} <= set(self.native))

    def native_field(self, name):
        """Native GEOS-FP cells on the fine grid (nearest containing cell).

        When the native file was not archived, the interpolated input is averaged back onto
        the GEOS-FP 0.25 x 0.3125 degree cells, so the coarse grid is still shown honestly."""
        if name == 'wind_speed' and self.has_native(name):
            u, v = self.native_field('u10m'), self.native_field('v10m')
            return np.hypot(u, v)
        if name not in self.native:
            return self.cell_mean(self.get(name, 'coarse'))
        n = self.native[name]
        key = (len(n['lat']), float(n['lat'][0]), len(n['lon']), float(n['lon'][0]))
        if key not in self._idx:
            self._idx[key] = native_index(self.lat, self.lon, n['lat'], n['lon'])
        iy, ix = self._idx[key]
        return physical(name, n['values'][iy, ix])

    def geosfp_cells(self):
        """Index of the GEOS-FP 0.25 x 0.3125 degree cell (global grid from -90, -180) for every pixel."""
        if 'geosfp' not in self._idx:
            iy = np.rint((self.lat+90.)/.25).astype(int)
            ix = np.rint((((self.lon+180.) % 360.))/.3125).astype(int)
            self._idx['geosfp'] = (iy, ix)
        return self._idx['geosfp']

    def cell_mean(self, values):
        """Area-weighted mean of a fine field over each GEOS-FP cell, painted back on the fine grid."""
        iy, ix = self.geosfp_cells()
        key = (iy-iy.min())*(ix.max()-ix.min()+1)+(ix-ix.min())
        _, inverse = np.unique(key.ravel(), return_inverse=True)
        w = self.area.ravel()
        total = np.bincount(inverse, weights=np.asarray(values, dtype='float64').ravel()*w)
        return (total/np.bincount(inverse, weights=w))[inverse].reshape(values.shape).astype('float32')

    def native_cells(self, name):
        """(iy, ix) native-cell index arrays for outlining cells."""
        src = 'u10m' if name == 'wind_speed' else name
        if src not in self.native:
            return self.geosfp_cells()
        n = self.native[src]
        return native_index(self.lat, self.lon, n['lat'], n['lon'])


# ----------------------------------------------------------------------------- map drawing

def draw_map(ax, data, cmap, norm, box=None, geo=None, k=1, cells=None, frame=True, interpolation='nearest'):
    """imshow on grid-index coordinates. box=(r0,r1,c0,c1) at full resolution; k=display block."""
    h, w = data.shape
    r0, r1, c0, c1 = box if box is not None else (0, h, 0, w)
    sub = np.asarray(data[r0:r1, c0:c1], dtype='float32')
    shown = block(sub, k)
    extent = (c0-.5, c0-.5+shown.shape[1]*k, r0-.5, r0-.5+shown.shape[0]*k)
    im = ax.imshow(shown, origin='lower', cmap=cmap, norm=norm, interpolation=interpolation,
                   extent=extent, rasterized=True, zorder=1)
    if cells is not None:
        iy, ix = cells[0][r0:r1, c0:c1], cells[1][r0:r1, c0:c1]
        ys, xs = np.mgrid[r0:r1, c0:c1]
        for idx in (iy, ix):
            levels = np.arange(idx.min(), idx.max())+.5
            if 0 < len(levels) < 400:
                ax.contour(xs, ys, idx, levels=levels, colors='white', linewidths=.35, alpha=.55, zorder=5)
    if geo is not None:
        geo.draw(ax, (r0, r1, c0, c1), scale=1.4 if box is not None else 1.)
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(frame); s.set_linewidth(.5); s.set_color('#55606b')
    return im


def colorbar(fig, im, cax, label, extend='neither', ticks=None, orientation='vertical'):
    cb = fig.colorbar(im, cax=cax, orientation=orientation, extend=extend, ticks=ticks,
                      format=matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:g}') if ticks is not None else None)
    cb.outline.set_linewidth(.4); cb.ax.tick_params(labelsize=6.2, length=2, width=.4, pad=1.5)
    cb.set_label(label, fontsize=6.8, labelpad=2)
    return cb


def map_grid(nrows, ncols, aspect, width=7.2, cbar=True, hspace=.06, wspace=.03, top_pad=.22, left_pad=.18,
             cbar_w=.10, extra=None):
    """Figure with nrows x ncols map axes (fixed aspect h/w) and a colour-bar column per row.
    extra={col: inches} widens the gap after a column (e.g. for an inner colour bar)."""
    extra = extra or {}
    gap_w = wspace*width/ncols
    usable = width-left_pad-(cbar_w+.42 if cbar else 0)-sum(extra.values())
    pw = (usable-(ncols-1)*gap_w)/ncols; ph = pw*aspect
    xs = [left_pad+c*(pw+gap_w)+sum(v for k, v in extra.items() if k < c) for c in range(ncols)]
    gap_h = hspace+top_pad
    height = nrows*ph+(nrows-1)*gap_h+top_pad+.08
    fig = plt.figure(figsize=(width, height))
    axes, caxes = [], []
    for r in range(nrows):
        y = height-top_pad-(r+1)*ph-r*gap_h
        row = [fig.add_axes([xs[c]/width, y/height, pw/width, ph/height]) for c in range(ncols)]
        axes.append(row)
        if cbar:
            x = xs[-1]+pw+.08
            caxes.append(fig.add_axes([x/width, (y+.06*ph)/height, cbar_w/width, .88*ph/height]))
    return fig, np.array(axes), caxes


def col_titles(axes, titles, size=7.6):
    for ax, t in zip(axes[0], titles):
        ax.set_title(t, fontsize=size, pad=3.5, color=INK, weight='bold')


def row_label(ax, text):
    ax.text(-.03, .5, text, transform=ax.transAxes, rotation=90, ha='right', va='center', fontsize=7.2,
            color=INK, weight='bold', linespacing=1.15)


def tag(ax, text):
    ax.text(.98, .03, text, transform=ax.transAxes, ha='right', va='bottom', fontsize=5.8, color=MUTED, zorder=20,
            bbox=dict(boxstyle='round,pad=.2', fc='white', ec='none', alpha=.85))


def mark_box(ax, box, label, color='#111111'):
    r0, r1, c0, c1 = box
    ax.add_patch(Rectangle((c0, r0), c1-c0, r1-r0, fill=False, lw=1.0, ec=color, zorder=12))
    ax.text(c0, r1+4, label, fontsize=6.6, weight='bold', color=color, ha='left', va='bottom', zorder=12,
            path_effects=[pe.withStroke(linewidth=2, foreground='white')])


# ----------------------------------------------------------------------------- zoom selection

def rain_box(case, geo, size=ZOOM_PX):
    """Box centred on the heaviest smoothed GEOS-FP *input* precipitation (no reference or output used)."""
    from scipy.ndimage import uniform_filter
    rain = case.get('precip', 'coarse')
    h, w = rain.shape
    s = uniform_filter(rain.astype('float64'), size=min(size//2, h, w), mode='constant')
    margin = size//2+4
    if h <= 2*margin or w <= 2*margin:
        return (0, h, 0, w), False
    inner = s[margin:h-margin, margin:w-margin]
    if not np.isfinite(inner).any() or np.nanmax(inner) < .05:
        return terrain_box(geo, h, w), False
    r, c = np.unravel_index(np.nanargmax(inner), inner.shape)
    r, c = r+margin, c+margin
    return (r-size//2, r+size//2, c-size//2, c+size//2), True


def terrain_box(geo, h=None, w=None):
    h = h or geo.lat.shape[0]; w = w or geo.lat.shape[1]
    lon0, lon1, lat0, lat1 = TERRAIN_BOX
    if geo.lon.min() < lon0 and geo.lon.max() > lon1 and geo.lat.min() < lat0 and geo.lat.max() > lat1:
        box = geo.box_of(lon0, lon1, lat0, lat1)
        if box[1]-box[0] > 20 and box[3]-box[2] > 20:
            return box
    # Small or non-CONUS grids: central 40%.
    return (int(.3*h), int(.7*h), int(.3*w), int(.7*w))


# ----------------------------------------------------------------------------- case-map figures

OVERVIEW_ROWS = ('t2m', 'wind_speed', 'precip', 'q2m')


def fig_overview(case, geo, boxes, out, dpi, stem, subject=None, rows=OVERVIEW_ROWS):
    """CONUS: native GEOS-FP cells | MERRA-Flow member | GEOS replay reference."""
    h, w = case.lat.shape
    fig, axes, caxes = map_grid(len(rows), 3, h/w, width=7.2, hspace=.05, top_pad=.2)
    native_ok = []
    for r, name in enumerate(rows):
        ref = case.get(name, 'truth')
        native_ok.append(case.has_native(name))
        panels = [case.get(name, 'native'), case.get(name, 'member', 0), ref]
        cmap, norm, extend = scale_for(name, ref)
        if name == 'precip':
            cmap, norm = rain_cmap(*(block(data, DISPLAY_BLOCK) for data in panels))
        for c, data in enumerate(panels):
            im = draw_map(axes[r, c], data, cmap, norm, geo=geo, k=DISPLAY_BLOCK,
                          interpolation='nearest')
            letter(axes[r, c], 3*r+c)
        if not native_ok[-1]:
            tag(axes[r, 0], 'cell means of input')
        for label, b in boxes.items():
            if (label == 'A' and name in ('t2m', 'wind_speed')) or (label == 'B' and name in ('precip', 'q2m')):
                for c in range(3):
                    mark_box(axes[r, c], b, label)
        row_label(axes[r, 0], VARS[name]['title'])
        colorbar(fig, im, caxes[r], VARS[name]['unit'], extend,
                 ticks=precip_ticks(norm) if name == 'precip' else None)
    col_titles(axes, [NATIVE_LABEL, 'MERRA-Flow (member 1, ~3 km)',
                      'GEOS-CAM reference (~3 km)'])
    export(fig, out, stem, dpi, subject)
    return dict(stem=stem, native=all(native_ok))


ZOOM_ROWS = (('t2m', 'A'), ('wind_speed', 'A'), ('precip', 'B'), ('q2m', 'B'))


def fig_zoom(case, geo, boxes, out, dpi, stem, subject=None):
    """Resolution ladder inside the two zoom boxes."""
    cols = ['native', 'coarse', 'regression', 'member', 'truth']
    titles = [NATIVE_LABEL, 'Bilinear input', 'Regression', 'MERRA-Flow member', 'Reference']
    if case.spectral is not None:
        cols.insert(4, 'spectral_member'); titles.insert(4, '+ spectral')
    b = boxes['A']; aspect = (b[1]-b[0])/(b[3]-b[2])
    fig, axes, caxes = map_grid(len(ZOOM_ROWS), len(cols), aspect, width=7.2, hspace=.04, top_pad=.2, cbar_w=.09)
    for r, (name, key) in enumerate(ZOOM_ROWS):
        box = boxes[key]
        ref = case.get(name, 'truth')
        sub = ref[box[0]:box[1], box[2]:box[3]]
        cmap, norm, extend = scale_for(name, sub)
        panels = [case.get(name, which) for which in cols]
        if name == 'precip':
            cmap, norm = rain_cmap(*(data[box[0]:box[1], box[2]:box[3]] for data in panels))
        cells = case.native_cells(name)
        for c, (which, data) in enumerate(zip(cols, panels)):
            im = draw_map(axes[r, c], data, cmap, norm, box=box, geo=geo, cells=cells if which == 'native' else None)
            letter(axes[r, c], len(cols)*r+c)
        row_label(axes[r, 0], f'{VARS[name]["title"]}\nbox {key}')
        if not case.has_native(name):
            tag(axes[r, 0], 'cell means of input')
        colorbar(fig, im, caxes[r], VARS[name]['unit'], extend,
                 ticks=precip_ticks(norm) if name == 'precip' else None)
    col_titles(axes, titles, size=7.0)
    export(fig, out, stem, dpi, subject)
    return dict(stem=stem)


def fig_ensemble(case, geo, box, out, dpi, stem, name='precip', subject=None, threshold=None):
    """Plausible realizations: GEOS-FP, three members, reference; mean, spread, exceedance probability.

    The exceedance threshold defaults to 1 mm/h, or 0.25 mm/h when the GEOS-FP *input* in the box is
    light (90th percentile of its wet pixels below 1 mm/h); the reference is never used to choose it."""
    if threshold is None:
        coarse = case.get(name, 'coarse')[box[0]:box[1], box[2]:box[3]]
        wet = coarse[coarse >= .1]
        threshold = 1. if wet.size and np.quantile(wet, .9) >= 1 else .25
    aspect = (box[1]-box[0])/(box[3]-box[2])
    fig, axes, _ = map_grid(2, 4, aspect, width=7.2, cbar=False, hspace=.05, top_pad=.2)
    ref = case.get(name, 'truth'); cmap, norm, extend = scale_for(name, ref[box[0]:box[1], box[2]:box[3]])
    cells = case.native_cells(name)
    top = [('native', 0, NATIVE_LABEL)] + [('member', m, f'Member {m+1}') for m in range(min(3, case.members))]
    top_fields = [case.get(name, which, m) for which, m, _ in top]
    mean = case.get(name, 'mean')
    if name == 'precip':
        cmap, norm = rain_cmap(*(data[box[0]:box[1], box[2]:box[3]] for data in [*top_fields, ref, mean]))
    for c, ((which, m, title), data) in enumerate(zip(top, top_fields)):
        im = draw_map(axes[0, c], data, cmap, norm, box=box, geo=geo,
                      cells=cells if which == 'native' else None)
        axes[0, c].set_title(title, fontsize=7.4, pad=3, weight='bold'); letter(axes[0, c], c)
    im_ref = draw_map(axes[1, 0], ref, cmap, norm, box=box, geo=geo)
    axes[1, 0].set_title('Reference', fontsize=7.4, pad=3, weight='bold'); letter(axes[1, 0], 4)
    draw_map(axes[1, 1], mean, cmap, norm, box=box, geo=geo)
    axes[1, 1].set_title(f'Ensemble mean ({case.members} members)', fontsize=7.4, pad=3, weight='bold')
    letter(axes[1, 1], 5)
    spread = case.get(name, 'spread')
    shown_spread = spread[box[0]:box[1], box[2]:box[3]]
    smax = max(float(np.nanmax(shown_spread) if name == 'precip' else np.nanquantile(shown_spread, .995)), 1e-6)
    spread_norm, spread_ticks = rain_display_scale(shown_spread) if name == 'precip' else (mcolors.Normalize(0, smax), None)
    im_s = draw_map(axes[1, 2], spread, spread_cmap(), spread_norm, box=box, geo=geo)
    axes[1, 2].set_title('Ensemble spread (s.d.)', fontsize=7.4, pad=3, weight='bold'); letter(axes[1, 2], 6)
    c = VARS[name]['c']
    prob = (channel(case.ensemble, c)*VARS[name]['factor']+VARS[name]['offset'] >= threshold).mean(0)
    pc = mcolors.LinearSegmentedColormap.from_list('p', ['#ffffff', '#d9ecf2', '#8cc5d6', '#3f8fb3', '#1b4f80', '#14234d'])
    im_p = draw_map(axes[1, 3], prob, pc, mcolors.Normalize(0, 1), box=box, geo=geo)
    r0, r1, c0, c1 = box
    ys, xs = np.mgrid[r0:r1, c0:c1]
    axes[1, 3].contour(xs, ys, ref[r0:r1, c0:c1] >= threshold, levels=[.5], colors='#d1495b', linewidths=.7, zorder=8)
    axes[1, 3].set_title(f'P(rate ≥ {threshold:g} mm h$^{{-1}}$)', fontsize=7.4, pad=3, weight='bold')
    letter(axes[1, 3], 7)
    # Horizontal colour bars under the bottom row.
    for ax, im, lab, ticks in ((axes[1, 0], im_ref, VARS[name]['unit'], precip_ticks(norm) if name == 'precip' else None),
                               (axes[1, 2], im_s, f'Spread ({VARS[name]["unit"]})', spread_ticks),
                               (axes[1, 3], im_p, 'Probability (red: reference ≥ threshold)', [0, .25, .5, .75, 1])):
        pos = ax.get_position(); cax = fig.add_axes([pos.x0+.1*pos.width, pos.y0-.05, .8*pos.width, .013])
        colorbar(fig, im, cax, lab, 'max' if ax is axes[1, 0] or (name == 'precip' and ax is axes[1, 2]) else 'neither', ticks, 'horizontal')
    export(fig, out, stem, dpi, subject)
    result = dict(stem=stem, members=case.members, threshold=float(threshold))
    if name == 'precip':
        result['rain_display'] = dict(upper_quantile=RAIN_DISPLAY_QUANTILE, wet_threshold=RAIN_WET_THRESHOLD,
                                     vmax=float(norm.vmax), ticks=precip_ticks(norm).tolist(),
                                     spread_vmax=float(spread_norm.vmax), over_range='colorbar upper extension')
    return result


def pixel_crps(ens, truth):
    m = len(ens)
    srt = np.sort(ens.astype('float64'), axis=0)
    coeff = (2*np.arange(1, m+1)-m-1).reshape((m,)+(1,)*(ens.ndim-1))
    return np.mean(np.abs(ens-truth), axis=0)-np.sum(coeff*srt, axis=0)/(m*m)


GAIN_ROWS = ('t2m', 'wind_speed', 'precip', 'q2m')


def fig_gain(case, geo, out, dpi, stem, subject=None, k=4):
    """Where downscaling helps: GEOS-FP absolute error, MERRA-Flow CRPS, and their difference."""
    h, w = case.lat.shape
    fig, axes, caxes = map_grid(len(GAIN_ROWS), 3, h/w, width=7.2, hspace=.05, top_pad=.2, extra={1: .5})
    numbers = {}
    for r, name in enumerate(GAIN_ROWS):
        c = VARS[name]['c']; f = VARS[name]['factor']
        truth = channel(case.truth, c).astype('float64')
        coarse_err = np.abs(channel(case.coarse, c)-truth)*f
        crps = pixel_crps(channel(case.ensemble, c), truth)*f
        gain = coarse_err-crps
        a = block(case.area, k)
        ce, cr, g = block(coarse_err, k), block(crps, k), block(gain, k)
        wet = (ce > 0) | (cr > 0)
        fine_wet = (coarse_err > 0) | (crps > 0)
        numbers[name] = dict(coarse_mae=float(np.sum(coarse_err*case.area)/case.area.sum()),
                             crps=float(np.sum(crps*case.area)/case.area.sum()),
                             improved_area=float(np.sum(case.area*(gain > 0)*fine_wet)/
                                                 max(np.sum(case.area*fine_wet), 1e-12)))
        numbers[name]['skill'] = 1-numbers[name]['crps']/numbers[name]['coarse_mae'] if numbers[name]['coarse_mae'] > 0 else None
        vmax = max(float(np.nanquantile(ce, .99)), 1e-6)
        n1, error_ticks = precip_scale(ce, cr) if name == 'precip' else (mcolors.Normalize(0, vmax), None)
        im1 = draw_map(axes[r, 0], coarse_err, error_cmap(), n1, geo=geo, k=k)
        draw_map(axes[r, 1], crps, error_cmap(), n1, geo=geo, k=k)
        gm = max(float(np.nanquantile(np.abs(g), .99)), 1e-6)
        gain_norm, gain_ticks = precip_scale(g, signed=True) if name == 'precip' else (mcolors.Normalize(-gm, gm), None)
        gain_cmap, gain_norm = difference_scale(gain_norm.vmin, gain_norm.vmax, GAIN_WHITE_BAND[name])
        im3 = draw_map(axes[r, 2], gain, gain_cmap, gain_norm, geo=geo, k=k)
        for cc in range(3):
            letter(axes[r, cc], 3*r+cc)
        row_label(axes[r, 0], VARS[name]['title'])
        pos = axes[r, 1].get_position()
        cax = fig.add_axes([pos.x1+.008, pos.y0+.06*pos.height, .012, .88*pos.height])
        colorbar(fig, im1, cax, VARS[name]['unit'], 'max', error_ticks)
        colorbar(fig, im3, caxes[r], 'Gain ' + VARS[name]['unit'], 'both', gain_ticks)
        sk = numbers[name]['skill']
        axes[r, 2].text(.98, .04, f'CRPS skill {100*sk:.0f}%' if sk is not None else '', transform=axes[r, 2].transAxes,
                        ha='right', va='bottom', fontsize=6.2, color=INK,
                        bbox=dict(boxstyle='round,pad=.25', fc='white', ec='none', alpha=.85), zorder=20)
    # Narrow the middle column gap where the inner colour bar sits.
    col_titles(axes, ['GEOS-FP absolute error', 'MERRA-Flow CRPS', 'Gain (GEOS-FP error − CRPS)'])
    export(fig, out, stem, dpi, subject)
    return dict(stem=stem, numbers=numbers, block_km=None)


# ----------------------------------------------------------------------------- aggregate figures

def methods(summary):
    return [m for m in METHODS if m in summary['sources']]


def fig_scorecard(summary, out, dpi, stem, subject=None):
    """Annotated skill heat map (relative to GEOS-FP) and calibration dot plot."""
    ms = [m for m in methods(summary) if m != 'coarse']
    src = summary['sources']
    crpss = np.array([[np.nan if src[m]['fields'][f]['crpss_vs_coarse'] is None else
                      100*src[m]['fields'][f]['crpss_vs_coarse'] for m in ms] for f in FIELDS])
    rmse = np.array([[100*(1-src[m]['fields'][f]['rmse_pooled']/src['coarse']['fields'][f]['rmse_pooled'])
                      if src['coarse']['fields'][f]['rmse_pooled'] > 0 else np.nan for m in ms] for f in FIELDS])
    data = np.c_[crpss, rmse]
    fig = plt.figure(figsize=(7.2, 2.7))
    ax = fig.add_axes([.165, .2, .50, .66])
    cmap = mcolors.LinearSegmentedColormap.from_list('skill', ['#b2182b', '#ef8a62', '#fbe3d6', '#f7f7f7',
                                                               '#d1e5f0', '#67a9cf', '#2166ac', '#0b3b70'])
    cmap, norm = difference_scale(-SKILL_COLOR_LIMIT, SKILL_COLOR_LIMIT, SKILL_WHITE_BAND, cmap)
    im = ax.imshow(data, cmap=cmap, norm=norm, aspect='auto')
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if np.isfinite(v):
                rgb = cmap(norm(v))[:3]; dark = .299*rgb[0]+.587*rgb[1]+.114*rgb[2] < .5
                ax.text(j, i, f'{v:+.0f}' if abs(v) >= 9.5 else f'{v:+.1f}', ha='center', va='center',
                        fontsize=6.8, color='white' if dark else INK)
    labels = {'regression': 'Regr.', 'raw': 'Flow', 'spectral': 'Flow+sp.'}
    ax.set_xticks(range(data.shape[1]), [labels[m] for m in ms]*2, fontsize=6.6)
    ax.set_yticks(range(len(FIELDS)), SHORT, fontsize=7)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.axvline(len(ms)-.5, color='white', lw=3)
    ax.set_xticks(np.arange(-.5, data.shape[1]), minor=True); ax.set_yticks(np.arange(-.5, len(FIELDS)), minor=True)
    ax.grid(which='minor', color='white', lw=1.2); ax.tick_params(which='minor', length=0)
    ax.text((len(ms)-1)/2, -.95, 'CRPS skill vs GEOS-FP (%)', ha='center', va='bottom', fontsize=7.4, weight='bold')
    ax.text(len(ms)+(len(ms)-1)/2, -.95, 'RMSE reduction vs GEOS-FP (%)', ha='center', va='bottom', fontsize=7.4,
            weight='bold')
    ax.text(-.30, 1.06, 'a', transform=ax.transAxes, fontsize=9, weight='bold', va='bottom')
    cax = fig.add_axes([.215, .045, .40, .03])
    colorbar(fig, im, cax, '', 'both', orientation='horizontal')
    # b: calibration
    bx = fig.add_axes([.72, .2, .26, .66])
    y = np.arange(len(FIELDS))
    for k, m in enumerate([m for m in ('raw', 'spectral') if m in src]):
        cov = [100*src[m]['fields'][f]['coverage_90'] for f in FIELDS]
        bx.scatter(cov, y+(k-.5)*.22, s=16, color=COLORS[m], label=NAMES[m], zorder=3, edgecolor='white', lw=.4)
    bx.axvline(90, color=INK, lw=.7, ls=(0, (3, 2)))
    bx.set_yticks(y, []); bx.set_ylim(len(FIELDS)-.5, -.5)
    bx.grid(axis='x', color=GRID, lw=.5); bx.spines['left'].set_visible(False); bx.tick_params(axis='y', length=0)
    for i in y:
        bx.axhline(i, color=GRID, lw=.4, zorder=0)
    bx.legend(loc='upper center', bbox_to_anchor=(.5, -.2), ncol=2, fontsize=6.2, handletextpad=.1, columnspacing=.6)
    bx.set_title('Central 90% interval coverage', fontsize=7.4, weight='bold', pad=6)
    bx.set_xlabel('%', labelpad=1)
    bx.text(-.08, 1.06, 'b', transform=bx.transAxes, fontsize=9, weight='bold', va='bottom')
    export(fig, out, stem, dpi, subject)
    return dict(stem=stem, crpss=dict(zip(FIELDS, crpss[:, ms.index('raw')] if 'raw' in ms else [np.nan]*7)))


SPECTRA_ROWS = ('t2m', 'wind_speed', 'precip', 'q2m')


def fig_spectra(summary, out, dpi, stem, subject=None, coarse_km=27., dx_km=None):
    dx = float(dx_km or summary.get('dx_km_approx') or 3.)
    fig = plt.figure(figsize=(7.2, 3.6))
    gs = fig.add_gridspec(2, 4, height_ratios=(2.1, 1), hspace=.08, wspace=.32, left=.07, right=.99, top=.87, bottom=.12)
    src = summary['sources']; result = {}
    for j, name in enumerate(SPECTRA_ROWS):
        top, bot = fig.add_subplot(gs[0, j]), fig.add_subplot(gs[1, j])
        s0 = src['coarse']['fields'][name]['spectrum']
        f = np.asarray(s0['frequency_cycles_pixel'], dtype='float64'); ok = f > 0
        lam = dx/f[ok]
        ref = np.asarray(s0['reference'])[ok]
        top.loglog(lam, ref, color=COLORS['reference'], lw=1.8, label=NAMES['reference'], zorder=5)
        for m in methods(summary):
            p = np.asarray(src[m]['fields'][name]['spectrum']['members'])[ok]
            top.loglog(lam, p, color=COLORS[m], lw=1.25, label=NAMES[m], ls='--' if m == 'spectral' else '-')
            ratio = np.divide(p, ref, out=np.full_like(p, np.nan), where=ref > 0)
            bot.semilogx(lam, ratio, color=COLORS[m], lw=1.25, ls='--' if m == 'spectral' else '-')
            band = (lam >= 2*dx) & (lam <= 2*coarse_km)
            result.setdefault(name, {})[m] = float(np.nanmean(ratio[band])) if band.any() else None
        for ax in (top, bot):
            ax.axvspan(lam.min()*.9, 2*coarse_km, color='#eef2f6', zorder=0, lw=0)
            ax.set_xlim(lam.max()*1.05, lam.min()*.95)
            ax.grid(which='major', color=GRID, lw=.4)
        top.axvline(2*coarse_km, color=MUTED, lw=.6, ls=':')
        if j == 0:
            top.text(2*coarse_km*.92, .04, 'below GEOS-FP\nresolution', transform=top.get_xaxis_transform(), fontsize=5.8,
                     color=MUTED, ha='left', va='bottom')
        bot.axhline(1, color=INK, lw=.6)
        bot.set_ylim(0, 1.6); bot.set_yticks([0, .5, 1, 1.5])
        top.set_xticklabels([]); top.set_title(VARS[name]['title'], fontsize=7.8, weight='bold', pad=3)
        top.text(.03, .04, chr(97+j), transform=top.transAxes, fontsize=9, weight='bold')
        bot.set_xlabel('Wavelength (km)')
        if j == 0:
            top.set_ylabel('Power spectral density'); bot.set_ylabel('Ratio to\nreference')
        top.tick_params(labelsize=6); bot.tick_params(labelsize=6)
    handles, labels = top.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=5, bbox_to_anchor=(.53, 1.0), fontsize=6.8, handlelength=1.8)
    export(fig, out, stem, dpi, subject)
    return dict(stem=stem, ratio=result)


def fig_precip(summary, out, dpi, stem, subject=None, dx_km=None):
    """Neighbourhood skill and intensity tails for precipitation."""
    dx = float(dx_km or summary.get('dx_km_approx') or 3.)
    src = summary['sources']
    fig = plt.figure(figsize=(7.2, 2.35))
    gs = fig.add_gridspec(1, 4, wspace=.38, left=.07, right=.99, top=.80, bottom=.2)
    result = {}
    for j, th in enumerate(('1.0', '5.0', '10.0')):
        ax = fig.add_subplot(gs[0, j])
        for m in methods(summary):
            fss = src[m]['precipitation'].get(th, {}).get('fss', {})
            if not fss:
                continue
            w = sorted(fss, key=int)
            x = np.array([int(s) for s in w])*dx
            yv = np.array([np.nan if fss[s]['ensemble_mean'] is None else fss[s]['ensemble_mean'] for s in w])
            ax.plot(x, yv, color=COLORS[m], marker='o', ms=2.6, lw=1.2, label=NAMES[m], ls='--' if m == 'spectral' else '-')
            if m in ('raw', 'spectral'):
                ym = [np.nan if fss[s]['members'] is None else fss[s]['members'] for s in w]
                ax.plot(x, ym, color=COLORS[m], lw=.8, ls=':', alpha=.9)
            result.setdefault(th, {})[m] = dict(zip([int(s) for s in w], yv.tolist()))
        ax.set_xscale('log'); ax.set_ylim(0, 1.02)
        ax.set_xlabel('Neighbourhood width (km)')
        if j == 0:
            ax.set_ylabel('Fractions skill score')
        ax.set_title(f'≥ {float(th):g} mm h$^{{-1}}$', fontsize=7.8, weight='bold', pad=3)
        ax.grid(color=GRID, lw=.4)
        ax.text(.03, .93, chr(97+j), transform=ax.transAxes, fontsize=9, weight='bold', va='top')
    ax = fig.add_subplot(gs[0, 3])
    q = summary.get('pixel_quantile_probabilities', [.5, .95, .99, .999])
    ref = src['raw']['fields']['precip'].get('reference_pixel_quantiles_hour_mean')
    xs = np.arange(len(q))
    if ref is not None:
        ax.plot(xs, ref, color=COLORS['reference'], marker='s', ms=3.2, lw=1.6, label='Reference')
    tails = {}
    for m in methods(summary):
        v = src[m]['fields']['precip'].get('pixel_quantiles_hour_mean')
        if v is not None:
            ax.plot(xs, v, color=COLORS[m], marker='o', ms=2.6, lw=1.1, ls='--' if m == 'spectral' else '-')
            tails[m] = v
    ax.set_yscale('linear'); ax.set_ylim(0, 5.*max(1, np.ceil(ax.get_ylim()[1]/5.)))
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5))
    ax.set_xticks(xs, [f'{100*p:g}' for p in q])
    ax.set_xlabel('Pixel percentile'); ax.set_ylabel(r'Rate (mm h$^{-1}$)')
    ax.set_title('Intensity distribution', fontsize=7.8, weight='bold', pad=3)
    ax.grid(color=GRID, lw=.4); ax.text(.03, .93, 'd', transform=ax.transAxes, fontsize=9, weight='bold', va='top')
    h, l = fig.axes[0].get_legend_handles_labels()
    h2, l2 = ax.get_legend_handles_labels()
    fig.legend(h+h2, l+l2, loc='upper center', ncol=5, bbox_to_anchor=(.53, 1.02), fontsize=6.8)
    export(fig, out, stem, dpi, subject)
    return dict(stem=stem, fss=result, tails=dict(tails, reference=ref), quantiles=q)


def per_hour(cases_dir, ids, method='raw'):
    rows = []
    for cid in ids:
        rec = json.loads((Path(cases_dir)/f'{cid}.json').read_text())
        s = rec['sources']
        sk = []
        for f in FIELDS:
            base = s['coarse']['fields'][f]['crps']
            sk.append(1-s[method]['fields'][f]['crps']/base if base > 0 else np.nan)
        rows.append((np.datetime64(rec['time'][:16]), sk))
    rows.sort(key=lambda r: r[0])
    return np.array([r[0] for r in rows]), np.array([r[1] for r in rows])*100


def fig_temporal(manifest, ids, cases_dir, out, dpi, stem, subject=None):
    """Hour-by-hour robustness of the downscaling gain over the held-out period."""
    times, skill = per_hour(cases_dir, ids)
    expected = np.array([np.datetime64(e['time'][:16]) for e in manifest['entries']])
    days = np.arange(expected.min().astype('datetime64[D]'), expected.max().astype('datetime64[D]')+np.timedelta64(1, 'D'))
    dkey = times.astype('datetime64[D]')
    grid = np.full((len(FIELDS), len(days)), np.nan); count = np.zeros(len(days), int)
    for d, day in enumerate(days):
        sel = dkey == day; count[d] = sel.sum()
        if sel.any():
            grid[:, d] = np.nanmean(skill[sel], 0)
    expected_count = np.array([(expected.astype('datetime64[D]') == d).sum() for d in days])
    palette = ['#c0392b', '#1d6fb8', '#7b4fa0', '#2a9d8f', '#8ab17d', '#e9a03b', '#264653']
    fig = plt.figure(figsize=(7.2, 4.3))
    top = fig.add_gridspec(2, 2, height_ratios=(.22, 1), width_ratios=(1, .025), hspace=.06, wspace=.02,
                           left=.13, right=.93, top=.96, bottom=.53)
    cov = fig.add_subplot(top[0, 0]); hm = fig.add_subplot(top[1, 0], sharex=cov); cax = fig.add_subplot(top[1, 1])
    x = np.arange(len(days))
    cov.bar(x, expected_count, color='#e3e8ee', width=.86, label='eligible hours')
    cov.bar(x, count, color='#7a8794', width=.86, label='evaluated hours')
    cov.set_yticks([]); cov.spines['left'].set_visible(False); cov.tick_params(axis='x', labelbottom=False, length=0)
    cov.legend(loc='upper right', bbox_to_anchor=(1.0, 1.55), ncol=2, fontsize=6, handlelength=1, handleheight=.7)
    cov.text(-.135, 1.0, 'a', transform=cov.transAxes, fontsize=9, weight='bold', va='bottom')
    cmap = mcolors.LinearSegmentedColormap.from_list('skill', ['#b2182b', '#f4a582', '#f7f7f7', '#92c5de', '#2166ac', '#0b3b70'])
    cmap, norm = difference_scale(-SKILL_COLOR_LIMIT, SKILL_COLOR_LIMIT, SKILL_WHITE_BAND, cmap)
    im = hm.imshow(grid, aspect='auto', cmap=cmap, norm=norm, interpolation='nearest')
    hm.set_yticks(range(len(FIELDS)), SHORT, fontsize=6.6)
    step = max(1, int(np.ceil(len(days)/9)))
    hm.set_xticks(x[::step], [str(d)[5:].replace('-', '/') for d in days[::step]], fontsize=6.2)
    hm.set_xlabel('Valid date (month/day, UTC)', labelpad=2)
    hm.tick_params(length=0)
    for sp in hm.spines.values():
        sp.set_visible(False)
    colorbar(fig, im, cax, 'Daily-mean CRPS skill\nvs GEOS-FP (%)', 'both')
    bottom = fig.add_gridspec(1, 3, width_ratios=(1.25, 1, 1), wspace=.42, left=.08, right=.985, top=.38, bottom=.1)
    dc = fig.add_subplot(bottom[0, 0])
    hours = (times.astype('datetime64[h]').astype(int) % 24)
    hh = sorted(set(hours))
    for i, f in enumerate(FIELDS):
        dc.plot(hh, [np.nanmean(skill[hours == h, i]) for h in hh], color=palette[i], lw=1.1, label=SHORT[i])
    dc.axhline(0, color=INK, lw=.6)
    dc.set_xlim(0, 23); dc.set_xticks([0, 6, 12, 18]); dc.set_xlabel('Hour (UTC)')
    dc.set_ylabel('CRPS skill (%)'); dc.grid(color=GRID, lw=.4)
    dc.set_title('Diurnal composite', fontsize=7.4, weight='bold', pad=3)
    dc.text(-.25, 1.02, 'b', transform=dc.transAxes, fontsize=9, weight='bold', va='bottom')
    bx = fig.add_subplot(bottom[0, 1])
    groups = [skill[np.isfinite(skill[:, i]), i] for i in range(len(FIELDS))]
    groups = [g if len(g) else np.array([np.nan, np.nan]) for g in groups]
    kw = dict(positions=range(len(FIELDS)), showextrema=False, showmedians=True, widths=.8)
    try:
        parts = bx.violinplot(groups, orientation='horizontal', **kw)
    except TypeError:   # Matplotlib < 3.10
        parts = bx.violinplot(groups, vert=False, **kw)
    for body, col in zip(parts['bodies'], palette):
        body.set_facecolor(col); body.set_edgecolor('none'); body.set_alpha(.75)
    parts['cmedians'].set_color(INK); parts['cmedians'].set_linewidth(.8)
    bx.axvline(0, color=INK, lw=.6); bx.set_yticks(range(len(FIELDS)), SHORT, fontsize=6.2); bx.invert_yaxis()
    bx.set_xlabel('Hourly CRPS skill (%)'); bx.tick_params(axis='y', length=0); bx.grid(axis='x', color=GRID, lw=.4)
    bx.set_title('Distribution over hours', fontsize=7.4, weight='bold', pad=3)
    bx.text(-.52, 1.02, 'c', transform=bx.transAxes, fontsize=9, weight='bold', va='bottom')
    fb = fig.add_subplot(bottom[0, 2])
    valid_count = np.isfinite(skill).sum(axis=0)
    frac = np.divide(100*((skill > 0) & np.isfinite(skill)).sum(axis=0), valid_count,
                     out=np.full(len(FIELDS), np.nan), where=valid_count > 0)
    fb.barh(np.arange(len(FIELDS)), frac, color=palette, height=.62)
    for i, v in enumerate(frac):
        fb.text(v-2 if v > 30 else v+2, i, f'{v:.0f}%', va='center', ha='right' if v > 30 else 'left', fontsize=6.2,
                color='white' if v > 30 else INK)
    fb.set_yticks(range(len(FIELDS)), []); fb.invert_yaxis(); fb.set_xlim(0, 100)
    fb.set_xlabel('Hours with skill > 0 (%)'); fb.tick_params(axis='y', length=0)
    fb.set_title('Consistency', fontsize=7.4, weight='bold', pad=3)
    fb.text(-.08, 1.02, 'd', transform=fb.transAxes, fontsize=9, weight='bold', va='bottom')
    export(fig, out, stem, dpi, subject)
    return dict(stem=stem, positive_fraction=dict(zip(FIELDS, frac.tolist())), hours=len(times),
                median=dict(zip(FIELDS, np.nanmedian(skill, 0).tolist())),
                p10=dict(zip(FIELDS, np.nanquantile(skill, .1, axis=0).tolist())),
                daily_positive=dict(zip(FIELDS, [float(np.mean(row[np.isfinite(row)] > 0)) if np.isfinite(row).any()
                                                 else None for row in grid])),
                diurnal={f: {str(h): float(np.nanmean(skill[hours == h, i])) for h in hh} for i, f in enumerate(FIELDS)})


def fig_calibration(summary, out, dpi, stem, subject=None):
    """Appendix: rank histograms as heat maps, spread-error, rain reliability and Brier skill."""
    src = summary['sources']; members = summary['settings']['members']
    ms = [m for m in ('raw', 'spectral') if m in src]
    fig = plt.figure(figsize=(7.2, 4.9))
    gs = fig.add_gridspec(2, 3, hspace=.62, wspace=.5, left=.13, right=.98, top=.94, bottom=.09, width_ratios=(1.15, 1.15, 1))
    for k, m in enumerate(ms):
        ax = fig.add_subplot(gs[0, k])
        ranks = np.array([np.asarray(src[m]['fields'][f]['rank_histogram_hour_mean'] or [np.nan]*(members+1)) for f in FIELDS])
        ranks = ranks*(members+1)
        im = ax.imshow(ranks, aspect='auto', cmap='PuOr_r', norm=mcolors.TwoSlopeNorm(1, 0, max(3, np.nanmax(ranks))),
                       interpolation='nearest')
        ax.set_yticks(range(len(FIELDS)), SHORT if k == 0 else [], fontsize=6.4)
        ax.set_xticks([0, members//2, members], ['1', str(members//2+1), str(members+1)])
        ax.set_xlabel('Rank of reference', labelpad=1); ax.tick_params(length=0)
        ax.set_title(NAMES[m], fontsize=7.2, weight='bold', pad=3)
        ax.text(-.04 if k else -.62, 1.03, chr(97+k), transform=ax.transAxes, fontsize=9, weight='bold', va='bottom')
        for sp in ax.spines.values():
            sp.set_visible(False)
    pos0 = fig.axes[0].get_position(); pos1 = fig.axes[len(ms)-1].get_position()
    cax = fig.add_axes([pos0.x0+.15*(pos1.x1-pos0.x0), pos0.y0-.12, .7*(pos1.x1-pos0.x0), .014])
    colorbar(fig, im, cax, f'Relative rank frequency (1 = uniform, {members} members)', 'max', orientation='horizontal')
    ax = fig.add_subplot(gs[0, 2])
    y = np.arange(len(FIELDS))
    for k, m in enumerate(ms):
        ax.scatter([src[m]['fields'][f]['spread_skill_corrected'] for f in FIELDS], y+(k-.5)*.22, s=14,
                   color=COLORS[m], label=NAMES[m], edgecolor='white', lw=.4, zorder=3)
    ax.axvline(1, color=INK, lw=.7, ls=(0, (3, 2))); ax.set_yticks(y, []); ax.set_ylim(len(FIELDS)-.5, -.5)
    ax.set_xlabel('Spread / RMSE (corrected)', labelpad=1); ax.grid(axis='x', color=GRID, lw=.4)
    ax.set_title('Spread–error ratio', fontsize=7.2, weight='bold', pad=3)
    ax.text(-.06, 1.03, 'c', transform=ax.transAxes, fontsize=9, weight='bold', va='bottom')
    ax.legend(fontsize=5.6, loc='upper center', bbox_to_anchor=(.5, -.2), ncol=1)
    for j, th in enumerate(('1.0', '5.0')):
        ax = fig.add_subplot(gs[1, j])
        for m in ms:
            r = src[m]['precipitation'].get(th)
            if not r:
                continue
            freq = np.array([np.nan if v is None else v for v in r['reliability_frequency']])
            ax.plot(r['probability_bins'], freq, color=COLORS[m], marker='o', ms=2.6, lw=1.1, label=NAMES[m])
        ax.plot([0, 1], [0, 1], color=INK, lw=.6, ls=':')
        ax.set(xlim=(-.02, 1.02), ylim=(-.02, 1.02))
        ax.set_xlabel('Forecast probability', labelpad=1); ax.set_ylabel('Observed frequency' if j == 0 else '')
        ax.grid(color=GRID, lw=.4)
        ax.set_title(f'Reliability, ≥ {float(th):g} mm h$^{{-1}}$', fontsize=7.2, weight='bold', pad=3)
        ax.text(-.2 if j == 0 else -.08, 1.03, chr(100+j), transform=ax.transAxes, fontsize=9, weight='bold', va='bottom')
        r = src['raw']['precipitation'].get(th)
        if r:
            ins = ax.inset_axes([.56, .1, .4, .26])
            mass = np.asarray(r['reliability_area'], dtype='float64'); mass = mass/max(mass.sum(), 1e-12)
            ins.bar(r['probability_bins'], np.where(mass > 0, mass, np.nan), width=.8/len(mass), color=COLORS['raw'])
            ins.set_yscale('log'); ins.set_xticks([]); ins.set_yticks([])
            ins.set_title('issued area (log)', fontsize=5.4, pad=1)
    ax = fig.add_subplot(gs[1, 2])
    thresholds = [t for t in src['coarse']['precipitation']]
    tv = [float(t) for t in thresholds]
    for m in methods(summary):
        if m == 'coarse':
            continue
        bss = [1-src[m]['precipitation'][t]['brier']/src['coarse']['precipitation'][t]['brier']
               if src['coarse']['precipitation'][t]['brier'] > 0 else np.nan for t in thresholds]
        ax.plot(range(len(tv)), 100*np.array(bss), marker='o', ms=2.8, lw=1.1, color=COLORS[m], label=NAMES[m],
                ls='--' if m == 'spectral' else '-')
    ax.axhline(0, color=INK, lw=.6); ax.set_xticks(range(len(tv)), [f'{v:g}' for v in tv])
    ax.set_xlabel(r'Threshold (mm h$^{-1}$)', labelpad=1); ax.set_ylabel('Brier skill vs GEOS-FP (%)')
    ax.grid(color=GRID, lw=.4); ax.set_title('Exceedance skill', fontsize=7.2, weight='bold', pad=3)
    ax.text(-.32, 1.03, 'f', transform=ax.transAxes, fontsize=9, weight='bold', va='bottom')
    export(fig, out, stem, dpi, subject)
    return dict(stem=stem)


# ----------------------------------------------------------------------------- storm case study

class Storm:
    """storm_fields.nc written by merraflow.storm_animation_v4_1 (window may follow the low)."""

    TARGETS = ('t2m', 'precip', 'ps', 'u10m', 'v10m', 'q2m')

    def __init__(self, path):
        import xarray as xr
        with xr.open_dataset(path) as ds:
            self.times = np.asarray(ds['time'].values).astype('datetime64[m]')
            self.attrs = dict(ds.attrs)
            self.runs = [str(r) for r in ds['run'].values]
            g = lambda k: np.asarray(ds[k].values)
            self.truth = {n: g(f'truth_{n}') for n in self.TARGETS if f'truth_{n}' in ds}
            self.geos = {n: g(f'geosfp_{n}') for n in self.TARGETS if f'geosfp_{n}' in ds}
            self.member = {n: g(f'member_{n}')[:, 0] for n in self.TARGETS if f'member_{n}' in ds}  # first run
            self.lat, self.lon, self.elev = g('lat'), g('lon'), g('elevation')
            self.row0 = g('window_row0') if 'window_row0' in ds else np.zeros(len(self.times), int)
            self.col0 = g('window_col0') if 'window_col0' in ds else np.zeros(len(self.times), int)
            self.low = None
            if 'low_lat' in ds:
                self.low = dict(lat=g('low_lat'), lon=g('low_lon'), hpa=g('low_reduced_pressure'))
        self.post = str(self.attrs.get('post', 'none'))
        self.members = self.member['precip'].shape[1]
        self.peak = int(np.argmax([np.nanmax(self.truth['precip'][k]) for k in range(len(self.times))])
                        if 'peak' not in self.attrs else
                        np.argmin(np.abs(self.times-np.datetime64(str(self.attrs['peak'])[:16]))))

    def field(self, name, source, k, m=0):
        pick = {'truth': self.truth, 'geos': self.geos, 'member': self.member}[source]
        if name == 'wind_speed':
            u, v = (pick['u10m'][k], pick['v10m'][k]) if source != 'member' else (pick['u10m'][k, m], pick['v10m'][k, m])
            return np.hypot(u, v)
        a = pick[name][k] if source != 'member' else pick[name][k, m]
        return physical(name, a)

    def slp(self, source, k, m=0):
        """Terrain-reduced pressure (hPa) for contours, as in storm_animation_v4_1."""
        ps = self.field('ps', source, k, m)*100.
        t = self.field('t2m', source, k, m)+273.15
        z = self.elev[k]
        return ps*(1-.0065*z/(t+.0065*z))**(-9.80665/(287.05*.0065))/100.


def _smooth(a, s=3):
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(np.asarray(a, dtype='float64'), s)


def fig_storm_evolution(storm, boundaries, out, dpi, stem, subject=None, n=4):
    """Precipitation and sea-level pressure through the storm: GEOS-FP | members | reference."""
    T = len(storm.times)
    ks = sorted(set(np.clip(np.round(np.linspace(max(storm.peak-9, 0), min(storm.peak+6, T-1), n)).astype(int), 0, T-1)))
    if storm.peak not in ks:
        ks[int(np.argmin(np.abs(np.array(ks)-storm.peak)))] = storm.peak
    cols = [('geos', 0, NATIVE_LABEL)] + [('member', m, f'MERRA-Flow member {m+1}') for m in range(min(2, storm.members))] + \
           [('truth', 0, 'Reference')]
    h, w = storm.lat.shape[1:]
    fig, axes, caxes = map_grid(len(ks), len(cols), h/w, width=7.2, hspace=.04, top_pad=.2, cbar=False)
    cmap, norm = rain_cmap(*(storm.field('precip', src, k, m) for k in ks for src, m, _ in cols))
    numbers = []
    for r, k in enumerate(ks):
        geo = Geo(storm.lat[k], storm.lon[k], boundaries)
        for c, (src, m, title) in enumerate(cols):
            ax = axes[r, c]
            rain = storm.field('precip', src, k, m)
            im = draw_map(ax, rain, cmap, norm, geo=geo)
            try:
                p = _smooth(storm.slp(src, k, m), 10)
                p = np.where(_smooth(storm.elev[k], 4) > 1000, np.nan, p)   # reduction unreliable over high terrain
                lv = np.arange(np.floor(np.nanmin(p)/4)*4, np.nanmax(p)+4, 4)
                cs = ax.contour(np.arange(w), np.arange(h), p, levels=lv, colors='#2b2f36', linewidths=.5, zorder=7)
                ax.clabel(cs, cs.levels[::2], fmt='%d', fontsize=4.8, inline=True, inline_spacing=1)
            except Exception:
                pass
            letter(ax, len(cols)*r+c)
            if r == 0:
                ax.set_title(title, fontsize=7.2, pad=3, weight='bold')
        stamp = str(storm.times[k]).replace('T', ' ')+' UTC' + ('  (peak)' if k == storm.peak else '')
        row_label(axes[r, 0], stamp)
        numbers.append(dict(time=str(storm.times[k]), geos_max=float(np.nanmax(storm.field('precip', 'geos', k))),
                            truth_max=float(np.nanmax(storm.field('precip', 'truth', k))),
                            member_max=float(np.nanmax(storm.field('precip', 'member', k, 0)))))
    pos = axes[-1, 0].get_position(); last = axes[-1, -1].get_position()
    cax = fig.add_axes([pos.x0+.15*(last.x1-pos.x0), pos.y0-.045, .7*(last.x1-pos.x0), .012])
    colorbar(fig, im, cax, r'Precipitation rate (mm h$^{-1}$); contours: sea-level pressure every 4 hPa', 'max',
             precip_ticks(norm), 'horizontal')
    export(fig, out, stem, dpi, subject)
    return dict(stem=stem, hours=numbers)


def fig_storm_peak(storm, boundaries, out, dpi, stem, subject=None):
    """Storm structure at the peak hour for wind, temperature and humidity."""
    k = storm.peak
    geo = Geo(storm.lat[k], storm.lon[k], boundaries)
    rows = [n for n in ('wind_speed', 't2m', 'q2m') if (n == 'wind_speed' or n in storm.truth)]
    cols = [('geos', 0, NATIVE_LABEL)] + [('member', m, f'Member {m+1}') for m in range(min(2, storm.members))] + \
           [('truth', 0, 'Reference')]
    h, w = storm.lat.shape[1:]
    fig, axes, caxes = map_grid(len(rows), len(cols), h/w, width=7.2, hspace=.04, top_pad=.2)
    for r, name in enumerate(rows):
        ref = storm.field(name, 'truth', k)
        cmap, norm, extend = scale_for(name, ref)
        for c, (src, m, title) in enumerate(cols):
            im = draw_map(axes[r, c], storm.field(name, src, k, m), cmap, norm, geo=geo)
            letter(axes[r, c], len(cols)*r+c)
        row_label(axes[r, 0], VARS[name]['title'])
        colorbar(fig, im, caxes[r], VARS[name]['unit'], extend)
    col_titles(axes, [t for _, _, t in cols], size=7.2)
    export(fig, out, stem, dpi, subject)
    return dict(stem=stem, time=str(storm.times[k]))


def fig_storm_series(storm, out, dpi, stem, subject=None, threshold=5.):
    """Window statistics through the event: rain area, peak rate, and wind maximum."""
    T = len(storm.times); hrs = (storm.times-storm.times[storm.peak]).astype('timedelta64[m]').astype(float)/60
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.15), layout='constrained')
    stats = {'geos': [], 'truth': [], 'member': []}
    spec = [('Area ≥ %g mm h$^{-1}$ (10$^3$ km$^2$)' % threshold,
             lambda a: (a >= threshold).sum()*float(storm.attrs.get('grid_km', 3.))**2/1e3, 'precip'),
            (r'99.9th percentile rate (mm h$^{-1}$)', lambda a: np.nanquantile(a, .999), 'precip'),
            (r'99th percentile wind speed (m s$^{-1}$)', lambda a: np.nanquantile(a, .99), 'wind_speed')]
    for j, (label, fn, name) in enumerate(spec):
        ax = axs[j]
        g = [fn(storm.field(name, 'geos', k)) for k in range(T)]
        t = [fn(storm.field(name, 'truth', k)) for k in range(T)]
        mem = np.array([[fn(storm.field(name, 'member', k, m)) for k in range(T)] for m in range(storm.members)])
        ax.plot(hrs, t, color=COLORS['reference'], lw=1.7, label='Reference')
        ax.plot(hrs, g, color=COLORS['coarse'], lw=1.4, label=NATIVE_LABEL)
        if storm.members > 1:
            ax.fill_between(hrs, mem.min(0), mem.max(0), color=COLORS['raw'], alpha=.18, lw=0)
        ax.plot(hrs, mem.mean(0), color=COLORS['raw'], lw=1.4, label=f'MERRA-Flow ({storm.members} members, range)')
        ax.axvline(0, color=MUTED, lw=.6, ls=':')
        ax.set_xlabel('Hours from peak'); ax.set_title(label, fontsize=7.2, weight='bold', pad=3)
        ax.grid(color=GRID, lw=.4); ax.text(.03, .93, chr(97+j), transform=ax.transAxes, fontsize=9, weight='bold', va='top')
        stats['geos'].append(float(np.nanmax(g))); stats['truth'].append(float(np.nanmax(t)))
        stats['member'].append(float(np.nanmax(mem.mean(0))))
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc='outside upper center', ncol=3, fontsize=6.6)
    export(fig, out, stem, dpi, subject)
    return dict(stem=stem, threshold=threshold, maxima=stats)


def storm_totals(storm):
    """Event accumulation on the intersection of all windows (mm)."""
    r0, c0 = storm.row0.astype(int), storm.col0.astype(int)
    h, w = storm.lat.shape[1:]
    R0, R1, C0, C1 = r0.max(), (r0+h).min(), c0.max(), (c0+w).min()
    if R1-R0 < 32 or C1-C0 < 32:
        return None
    acc = {}
    for src in ('geos', 'truth', 'member'):
        total = 0
        for k in range(len(storm.times)):
            sl = (slice(R0-r0[k], R1-r0[k]), slice(C0-c0[k], C1-c0[k]))
            a = storm.field('precip', src, k) if src != 'member' else storm.member['precip'][k][:, sl[0], sl[1]]
            total = total+(a[sl] if src != 'member' else a)
        acc[src] = total
    k0 = 0
    sl = (slice(R0-r0[k0], R1-r0[k0]), slice(C0-c0[k0], C1-c0[k0]))
    return acc, storm.lat[k0][sl], storm.lon[k0][sl]


def fig_storm_totals(storm, boundaries, out, dpi, stem, subject=None):
    res = storm_totals(storm)
    if res is None:
        return None
    acc, lat, lon = res
    geo = Geo(lat, lon, boundaries)
    cmap = mcolors.ListedColormap(['#e0f0fa', '#b3dbf1', '#7cc0e6', '#3f9ad6', '#2a6fb5', '#2c9c5c', '#8cc63f',
                                   '#f2d43a', '#f29434', '#de3b2f', '#a8166f']).with_extremes(under='white', over='#5b0f4a')
    panels = [(acc['geos'], NATIVE_LABEL), (acc['member'][0], 'MERRA-Flow member 1'),
              (acc['member'].mean(0), f'Ensemble mean ({storm.members})'), (acc['truth'], 'Reference')]
    norm, levels = precip_scale(*(data for data, _ in panels), step=25.)
    h, w = lat.shape
    fig, axes, caxes = map_grid(1, 4, h/w, width=7.2, top_pad=.2, cbar=True)
    for c, (a, t) in enumerate(panels):
        im = draw_map(axes[0, c], a, cmap, norm, geo=geo); letter(axes[0, c], c)
        axes[0, c].set_title(t, fontsize=7.2, pad=3, weight='bold')
    colorbar(fig, im, caxes[0], f'{len(storm.times)}-h total (mm)', 'max', levels)
    export(fig, out, stem, dpi, subject)
    return dict(stem=stem, max=dict(geos=float(acc['geos'].max()), member=float(acc['member'][0].max()),
                                    truth=float(acc['truth'].max())))


SKILL_MAP_ROWS = ('t2m', 'wind_speed', 'q2m', 'precip')


def fig_skill_maps(maps_path, out, dpi, stem, subject=None, boundaries=None, k=4):
    """Time-mean error and skill maps over all hours of the maps pass (scripts/evaluate_maps_v4_1.py)."""
    with np.load(maps_path, allow_pickle=False) as f:
        meta = json.loads(str(f['meta']))
        data = {key: f[key] for key in f.files if key != 'meta'}
    lat, lon = data.get('lat'), data.get('lon')
    area = data.get('area', np.ones_like(data['t2m_crps'], dtype='float64')).astype('float64')
    geo = Geo(lat, lon, load_boundaries() if boundaries is None else boundaries) if lat is not None else None
    h, w = data['t2m_crps'].shape
    fig, axes, caxes = map_grid(len(SKILL_MAP_ROWS), 4, h/w, width=7.2, hspace=.05, top_pad=.2, extra={1: .5})
    skill_cmap = mcolors.LinearSegmentedColormap.from_list('sk', ['#8e0e2c', '#d6604d', '#f4a582', '#f7f7f7',
                                                                  '#92c5de', '#4393c3', '#2166ac', '#0b3b70'])
    skill_cmap = skill_cmap.with_extremes(bad='#e6e8eb')
    skill_cmap, skill_norm = difference_scale(-SKILL_COLOR_LIMIT, SKILL_COLOR_LIMIT, SKILL_WHITE_BAND, skill_cmap)
    short = dict(t2m='Temperature', wind_speed='Wind speed', q2m='Humidity', precip='Precipitation')
    numbers = {}
    a = block(area, k)
    for r, name in enumerate(SKILL_MAP_ROWS):
        fct = VARS[name]['factor']
        mae, crps = data[f'{name}_mae_coarse']*fct, data[f'{name}_crps']*fct
        bm, bc = block(mae.astype('float64'), k), block(crps.astype('float64'), k)
        # Skill is undefined where GEOS-FP is (almost) never wrong, e.g. rain over deserts: mask those blocks.
        floor = .01 if name == 'precip' else 0.
        valid = bm > floor
        skill = np.where(valid, 100*(1-bc/np.where(valid, bm, 1)), np.nan)
        rm = np.where(valid, 100*(1-np.sqrt(block(data[f'{name}_se_mean'].astype('float64'), k))/
                                  np.sqrt(np.maximum(block(data[f'{name}_se_coarse'].astype('float64'), k), 1e-30))),
                      np.nan)
        top = valid & (bm >= np.nanquantile(bm[valid], .9)) if valid.any() else np.zeros_like(valid)
        def relative(numerator, denominator):
            return float(1-numerator/denominator) if denominator > 0 else None
        numbers[name] = dict(
            skill=relative(np.sum(crps*area), np.sum(mae*area)),
            positive_area=float(np.sum(a*(skill > 0)*valid)/np.sum(a*valid)) if valid.any() else None,
            skill_top_decile=relative(np.sum((bc*a)[top]), np.sum((bm*a)[top])),
            skill_rest=relative(np.sum((bc*a)[valid & ~top]), np.sum((bm*a)[valid & ~top])),
            rmse_reduction=relative(np.sqrt(np.sum(data[f'{name}_se_mean']*area)),
                                    np.sqrt(np.sum(data[f'{name}_se_coarse']*area))))
        vmax = max(float(np.nanquantile(bm, .99)), 1e-9)
        en, error_ticks = precip_scale(bm, bc) if name == 'precip' else (mcolors.Normalize(0, vmax), None)
        im1 = draw_map(axes[r, 0], mae, error_cmap(), en, geo=geo, k=k)
        draw_map(axes[r, 1], crps, error_cmap(), en, geo=geo, k=k)
        im3 = axes[r, 2].imshow(skill, origin='lower', cmap=skill_cmap, norm=skill_norm, interpolation='nearest',
                                extent=(-.5, skill.shape[1]*k-.5, -.5, skill.shape[0]*k-.5), rasterized=True)
        axes[r, 3].imshow(rm, origin='lower', cmap=skill_cmap, norm=skill_norm, interpolation='nearest',
                          extent=(-.5, rm.shape[1]*k-.5, -.5, rm.shape[0]*k-.5), rasterized=True)
        for c in (2, 3):
            ax = axes[r, c]
            if geo is not None:
                geo.draw(ax)
            ax.set_xlim(-.5, skill.shape[1]*k-.5); ax.set_ylim(-.5, skill.shape[0]*k-.5)
            ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(.5); sp.set_color('#55606b')
        for c in range(4):
            letter(axes[r, c], 4*r+c)
        row_label(axes[r, 0], short[name])
        pos = axes[r, 1].get_position()
        cax = fig.add_axes([pos.x1+.008, pos.y0+.06*pos.height, .012, .88*pos.height])
        colorbar(fig, im1, cax, VARS[name]['unit'], 'max', error_ticks)
        colorbar(fig, im3, caxes[r], 'Skill (%)', 'both', ticks=np.arange(-SKILL_COLOR_LIMIT, SKILL_COLOR_LIMIT+1, 20))
        score = numbers[name]['skill']
        axes[r, 2].text(.98, .04, f"{100*score:.0f}%" if score is not None else 'undefined', transform=axes[r, 2].transAxes, ha='right',
                        va='bottom', fontsize=6.4, color=INK, zorder=20,
                        bbox=dict(boxstyle='round,pad=.25', fc='white', ec='none', alpha=.85))
    col_titles(axes, ['GEOS-FP mean abs. error', 'MERRA-Flow mean CRPS', 'CRPS skill vs GEOS-FP', 'RMSE reduction'],
               size=7.2)
    export(fig, out, stem, dpi, subject)
    return dict(stem=stem, numbers=numbers, hours=meta['n_hours'], every=meta.get('every'), members=meta['members'],
                complete=meta['complete'], planned=len(meta['hours']), source=meta.get('source', 'maps_pass'),
                block_km=None)


# ----------------------------------------------------------------------------- driver

def load_case(root, manifest, case_id):
    rec = json.loads((Path(root)/'cases'/f'{case_id}.json').read_text())
    sf = rec['saved_fields']
    return Case(Path(root)/sf['raw']['path'], Path(root)/sf['spectral']['path'] if 'spectral' in sf else None,
                Path(root)/manifest['saved_grid']['path'], rec['time'], case_id)


def coarse_wet_fraction(root, manifest, case_id, threshold=1.):
    """Input-only ranking of saved cases (GEOS-FP wet area); never uses reference or output."""
    rec = json.loads((Path(root)/'cases'/f'{case_id}.json').read_text())
    with np.load(Path(root)/rec['saved_fields']['raw']['path'], allow_pickle=False) as f:
        return float((f['coarse'][1] >= threshold).mean())


def grid_spacing_km(root, manifest):
    """Median great-circle spacing of the saved grid (km). The evaluator's dx_km_approx comes from the
    archived cell-area field, which is not the 3-km geometric spacing, so it is used only as a fallback."""
    item = manifest.get('saved_grid')
    path = Path(root)/item['path'] if item else None
    if not path or not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as g:
        lat, lon = np.deg2rad(g['lat'].astype('float64')), np.deg2rad(g['lon'].astype('float64'))
    a = np.sin((lat[:, 1:]-lat[:, :-1])/2)**2+np.cos(lat[:, 1:])*np.cos(lat[:, :-1])*np.sin((lon[:, 1:]-lon[:, :-1])/2)**2
    return float(np.median(2*6371.*np.arcsin(np.sqrt(a))))


def render_all(root, manifest, summary, case_ids, out, dpi=300, storm_path=None, subject=None, boundaries=None,
               maps_path=None):
    """Draw every paper figure available from saved data. Returns a dict of figure metadata."""
    style(); out = Path(out); out.mkdir(parents=True, exist_ok=True)
    boundaries = load_boundaries() if boundaries is None else boundaries
    figs = {}
    figs['scorecard'] = fig_scorecard(summary, out, dpi, 'fig_scorecard', subject)
    dx_km = grid_spacing_km(root, manifest)
    figs['grid_km'] = dx_km
    figs['spectra'] = fig_spectra(summary, out, dpi, 'fig_spectra', subject, dx_km=dx_km)
    figs['precip'] = fig_precip(summary, out, dpi, 'fig_precip_skill', subject, dx_km=dx_km)
    figs['temporal'] = fig_temporal(manifest, case_ids, Path(root)/'cases', out, dpi, 'fig_temporal', subject)
    figs['calibration'] = fig_calibration(summary, out, dpi, 'fig_calibration', subject)
    figs['skill_maps'] = None
    if maps_path:
        figs['skill_maps'] = fig_skill_maps(maps_path, out, dpi, 'fig_skill_maps', subject, boundaries)
        figs['skill_maps']['block_km'] = 4*float(dx_km or summary.get('dx_km_approx') or 3.)
    maps = [c for c in manifest['settings'].get('saved_cases', []) if c in case_ids]
    figs['cases'] = []
    if maps:
        order = sorted(maps, key=lambda c: -coarse_wet_fraction(root, manifest, c))
        for i, cid in enumerate(order):
            case = load_case(root, manifest, cid)
            geo = Geo(case.lat, case.lon, boundaries)
            bA = terrain_box(geo); bB, centred = rain_box(case, geo)
            boxes = {'A': bA, 'B': bB}
            tag = f'{cid}'
            item = dict(id=cid, time=case.time, main=i == 0, members=case.members, rain_box_from_input=centred,
                        boxes={k: list(map(int, v)) for k, v in boxes.items()},
                        overview=fig_overview(case, geo, boxes, out, dpi, f'fig_case_{tag}_overview', subject))
            if i == 0:
                item['zoom'] = fig_zoom(case, geo, boxes, out, dpi, f'fig_case_{tag}_zoom', subject)
                item['ensemble'] = fig_ensemble(case, geo, bB, out, dpi, f'fig_case_{tag}_ensemble', subject=subject)
                item['gain'] = fig_gain(case, geo, out, dpi, f'fig_case_{tag}_gain', subject)
                item['gain']['block_km'] = 4*float(dx_km or summary.get('dx_km_approx') or 3.)
            figs['cases'].append(item)
            del case
    figs['storm'] = None
    if storm_path:
        storm = Storm(storm_path)
        figs['storm'] = dict(path=str(storm_path), peak=str(storm.times[storm.peak]), hours=len(storm.times),
                             members=storm.members, post=storm.post, start=str(storm.times[0]), end=str(storm.times[-1]),
                             tracked=bool(storm.low is not None),
                             evolution=fig_storm_evolution(storm, boundaries, out, dpi, 'fig_storm_evolution', subject),
                             peak_fields=fig_storm_peak(storm, boundaries, out, dpi, 'fig_storm_peak', subject),
                             series=fig_storm_series(storm, out, dpi, 'fig_storm_series', subject),
                             totals=fig_storm_totals(storm, boundaries, out, dpi, 'fig_storm_totals', subject))
    return figs
