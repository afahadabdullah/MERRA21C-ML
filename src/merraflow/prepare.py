"""Pair archive files and write memory-mappable, one-timestamp shards."""
from datetime import datetime, timedelta
from pathlib import Path
import hashlib
import json
import os
import shutil
import numpy as np
import xarray as xr
from .config import write_json
from .physics import native_groups, project_precip, transform_target, budget_error

PRECIP_VARS = {'PRECTOT', 'PRECCON', 'PRECLSC', 'PRECANV', 'PRECSNO'}


def field(ds, name):
    da = ds[name]
    for dim in tuple(da.dims):
        if dim not in ('Ydim', 'Xdim', 'lat', 'lon', 'y', 'x'):
            if da.sizes[dim] != 1:
                raise ValueError(f'{name}: expected singleton {dim}, got {da.sizes[dim]}')
            da = da.isel({dim: 0})
    dims = ('lat', 'lon') if 'lat' in da.dims else ('Ydim', 'Xdim') if 'Ydim' in da.dims else ('y', 'x')
    a = da.transpose(*dims).values.astype('float32')
    if a.ndim != 2 or not np.isfinite(a).all():
        raise ValueError(f'{name}: require finite 2D data (missing/fill values are not silently replaced)')
    return a


def unit_text(da):
    u = da.attrs.get('units', '').lower().replace(' ', '').replace('**', '^')
    # HWT LCC metadata writes positive exponents with an explicit plus sign.
    # These exact aliases change spelling only; they do not convert dimensions.
    return {'m+2': 'm2', 'm+2s-2': 'm2s-2'}.get(u, u)


def units(ds, name, kind):
    u = unit_text(ds[name])
    allowed = {
        'rate': {'kgm-2s-1', 'kgm^-2s^-1', 'kg/m2/s', 'kg/m^2/s'},
        'accum': {'mm', 'kgm-2', 'kgm^-2', 'kg/m2', 'kg/m^2'},
        'temperature': {'k', 'kelvin'}, 'pressure': {'pa', 'pascal', 'pascals'},
        'wind': {'ms-1', 'ms^-1', 'm/s'}, 'area': {'m2', 'm^2'},
    }
    if u not in allowed[kind]:
        raise ValueError(f'{name}: unsupported {kind} units {u!r}; audit metadata rather than guessing')


def assert_time(ds, expected, path):
    if 'time' not in ds or ds.time.size != 1:
        raise ValueError(f'{path}: expected one decoded timestamp')
    actual = np.asarray(ds.time.values).ravel()[0]
    if not np.issubdtype(type(actual), np.datetime64) or abs((actual-np.datetime64(expected))/np.timedelta64(1, 's')) > 1:
        raise ValueError(f'{path}: data time {actual} != filename/pair time {expected}')


def split_for(t, splits):
    hits = [key for key, (start, end) in splits.items() if datetime.fromisoformat(start) <= t < datetime.fromisoformat(end)]
    if len(hits) > 1:
        raise ValueError(f'Overlapping time splits at {t}')
    return hits[0] if hits else None


def manifest(cfg):
    d = cfg['data']
    # Validate ranges independently of which files happen to exist.
    ranges = sorted((datetime.fromisoformat(v[0]), datetime.fromisoformat(v[1]), k) for k, v in d['splits'].items())
    if set(d['splits']) != {'train', 'val', 'test'} or any(a >= b for a, b, _ in ranges):
        raise ValueError('Provide nonempty train, val and test half-open ranges')
    if any(ranges[i][1] > ranges[i+1][0] for i in range(len(ranges)-1)):
        raise ValueError('Split ranges overlap')
    entries, missing = [], []
    start, end = datetime.fromisoformat(d['start']), datetime.fromisoformat(d['end'])
    if start.minute != 30 or end.minute != 30 or start > end:
        raise ValueError('Inclusive data start/end must be hourly midpoint timestamps (:30)')
    t = start
    while t <= end:
        split = split_for(t, d['splits'])
        if split:
            tag, endtag = t.strftime('%Y%m%d_%H%M'), (t+timedelta(minutes=30)).strftime('%Y%m%d_%H%M')
            hrroot = Path(d['highres_root'])
            paths = {
                'lr': Path(d['lowres_lcc_root'])/t.strftime('%Y%m')/f'f5295_fp.lowres_lcc_1hr.{tag}z.nc4',
                'hr': hrroot/'hwt_30mn_slv_LCC'/t.strftime('%Y%m')/f'Feature-c2160_L137.hwt_30mn_slv_LCC.{tag}z.nc4',
                'acc': hrroot/'hwt_01hr_acc_LCC'/endtag[:6]/f'Feature-c2160_L137.hwt_01hr_acc_LCC.{endtag}z.nc4',
                'native': Path(d['native_root'])/t.strftime('Y%Y/M%m')/f'f5295_fp.tavg1_2d_flx_Nx.{tag}z.nc4',
            }
            absent = [str(p) for p in paths.values() if not p.is_file()]
            if absent:
                missing.append({'time': t.isoformat(), 'missing': absent})
            else:
                entries.append({'time': t.isoformat(), 'id': tag, 'split': split, **{k: str(v.resolve()) for k, v in paths.items()}})
        t += timedelta(hours=1)
    return entries, missing


def sorted_native(ds):
    return ds.assign_coords(lon=((ds.lon+180) % 360)-180).sortby('lon').sortby('lat')


def make_static(hr, native):
    lat, lon, area = field(hr, 'lats'), ((field(hr, 'lons')+180) % 360)-180, field(hr, 'AREA')
    units(hr, 'AREA', 'area')
    if np.any(area <= 0):
        raise ValueError('AREA must be strictly positive')
    z = field(hr, 'HGT_SFC')
    u = unit_text(hr.HGT_SFC)
    if u in {'m2s-2', 'm^2s^-2', 'm2/s2', 'm^2/s^2'}:
        z /= 9.80665
    elif u not in {'m', 'meter', 'meters'}:
        raise ValueError(f'Unknown HGT_SFC units: {u!r}')
    groups, source = native_groups(native.lat.values, native.lon.values, lat, lon)
    radlat, radlon = np.deg2rad(lat), np.deg2rad(lon)
    features = np.stack([z/2000, np.sin(radlat), np.cos(radlat), np.sin(radlon), np.cos(radlon), np.log(area/9e6)])
    return dict(lat=lat, lon=lon, area=area, elevation=z, groups=groups, source_flat=source,
                native_lat=native.lat.values, native_lon=native.lon.values, features=features.astype('float32'))


def time_features(time, lon):
    t = datetime.fromisoformat(time)
    year_start = datetime(t.year, 1, 1)
    year_days = (datetime(t.year+1, 1, 1)-year_start).days
    phase = 2*np.pi*(t-year_start).total_seconds()/(year_days*86400)
    utc = 2*np.pi*(t.hour+t.minute/60)/24
    local = utc + np.deg2rad(lon)
    ones = np.ones_like(lon)
    return np.stack([ones*np.sin(phase), ones*np.cos(phase), ones*np.sin(utc), ones*np.cos(utc), np.sin(local), np.cos(local)]).astype('float32')


def transformed_predictors(ds, names, scale):
    fields = []
    for name in names:
        a = field(ds, name)
        if name in PRECIP_VARS:
            units(ds, name, 'rate')
            if np.min(a) < -1e-10:
                raise ValueError(f'Negative precipitation in {name}')
            a = np.log1p(np.maximum(a, 0)*3600/scale)
        fields.append(a)
    return np.stack(fields)


class Moments:
    def __init__(self):
        self.n, self.mean, self.m2 = 0, None, None

    def add(self, arr):
        x = arr.astype('float64').reshape(arr.shape[0], -1)
        n, mu, m2 = x.shape[1], x.mean(1), ((x-x.mean(1)[:, None])**2).sum(1)
        if self.n == 0:
            self.n, self.mean, self.m2 = n, mu, m2
        else:
            delta = mu-self.mean
            self.m2 += m2 + delta**2*self.n*n/(self.n+n)
            self.mean += delta*n/(self.n+n)
            self.n += n

    def result(self):
        if not self.n:
            raise ValueError('No training samples for normalization')
        return {'mean': self.mean.tolist(), 'std': np.maximum(np.sqrt(self.m2/self.n), 1e-4).tolist(), 'count_per_channel': self.n}


def prepare(cfg):
    d = cfg['data']
    root = Path(d['prepared'])
    root.mkdir(parents=True, exist_ok=True)
    if (root/'index.json').exists():
        raise FileExistsError(f'{root} already prepared; use a new directory to prevent stale shards/statistics')
    entries, missing = manifest(cfg)
    write_json(root/'missing.json', missing)
    if missing and d['strict_missing']:
        raise FileNotFoundError(f'{len(missing)} incomplete hourly pairs; see {root}/missing.json. Adjust dates or explicitly disable strict_missing.')
    if not entries or any(not any(e['split'] == s for e in entries) for s in ('train', 'val', 'test')):
        raise ValueError('Need at least one complete hour in each train/val/test split')
    with xr.open_dataset(entries[0]['hr']) as hr, xr.open_dataset(entries[0]['native']) as raw:
        static = make_static(hr, sorted_native(raw))
        # Preserve available projection and grid coordinate metadata in a small NetCDF.
        geo = xr.Dataset({k: (('Ydim', 'Xdim'), static[k], {'units': u}) for k, u in [('lat', 'degrees_north'), ('lon', 'degrees_east'), ('area', 'm2'), ('elevation', 'm')]})
        for coord in ('Xdim', 'Ydim'):
            if coord in hr:
                geo = geo.assign_coords({coord: hr[coord].load()})
        mappings = [k for k in hr if 'grid_mapping_name' in hr[k].attrs]
        for k in mappings:
            geo[k] = hr[k].load()
        if mappings:
            geo.attrs['grid_mapping_variable'] = mappings[0]
        geo.attrs.update({'grid': 'HWT LCC; coordinates preserved from source', 'state_alignment': d['state_alignment']})
        geo.to_netcdf(root/'grid.nc', engine='h5netcdf')
    np.savez(root/'static.npz', **static)
    cond_mom, residual_mom = Moments(), Moments()
    stride = d['stats_stride']
    if stride < 1 or d['precip_log_scale'] <= 0 or d['wind_log_scale'] <= 0:
        raise ValueError('Statistics stride and transform scales must be positive')
    audit = []
    for i, e in enumerate(entries):
        t = datetime.fromisoformat(e['time'])
        with xr.open_dataset(e['lr']) as lr, xr.open_dataset(e['hr']) as hr, xr.open_dataset(e['acc']) as acc, xr.open_dataset(e['native']) as raw:
            native = sorted_native(raw)
            for ds, when, key in [(lr, t, 'lr'), (hr, t, 'hr'), (acc, t+timedelta(minutes=30), 'acc'), (native, t, 'native')]:
                assert_time(ds, when, e[key])
            for key, expected in [('native_lat', native.lat.values), ('native_lon', native.lon.values)]:
                if not np.array_equal(static[key], expected):
                    raise ValueError('Native grid changed within archive')
            for name, saved in [('lats', 'lat'), ('AREA', 'area')]:
                if not np.allclose(field(hr, name), static[saved], rtol=1e-6, atol=1e-5):
                    raise ValueError(f'HR grid {name} changed within archive')
            if not np.allclose(((field(hr, 'lons')+180) % 360)-180, static['lon'], atol=1e-5):
                raise ValueError('HR longitude grid changed')
            # If bounds exist, do not accept a different accumulation interval.
            bounds = acc.time.attrs.get('bounds')
            if bounds and bounds in acc:
                b = acc[bounds].values.ravel()
                expected = [np.datetime64(t-timedelta(minutes=30)), np.datetime64(t+timedelta(minutes=30))]
                if len(b) != 2 or not np.array_equal(b, expected):
                    raise ValueError('APCP time bounds do not match the LR hourly window')
            for ds, name, kind in [(native, 'PRECTOT', 'rate'), (acc, 'APCP', 'accum'), (hr, 'TMP_2M', 'temperature'), (hr, 'PRES_SFC', 'pressure'), (lr, 'T2M', 'temperature'), (lr, 'PS', 'pressure')]:
                units(ds, name, kind)
            for ds, names in [(hr, ['UGRD_10M', 'VGRD_10M']), (lr, ['U10M', 'V10M'])]:
                for name in names:
                    units(ds, name, 'wind')
            source_pr = field(native, 'PRECTOT').ravel()[static['source_flat']]*3600
            if source_pr.min() < -1e-7:
                raise ValueError('Negative native precipitation')
            reference = np.maximum(source_pr, 0)[static['groups']]
            precip = field(acc, 'APCP')/d['accumulation_hours']
            if precip.min() < -1e-7:
                raise ValueError('Negative APCP')
            target = np.stack([field(hr, 'TMP_2M'), np.maximum(precip, 0), field(hr, 'PRES_SFC'), np.hypot(field(hr, 'UGRD_10M'), field(hr, 'VGRD_10M'))])
            baseline = np.stack([field(lr, 'T2M'), reference, field(lr, 'PS'), np.hypot(field(lr, 'U10M'), field(lr, 'V10M'))])
            cond = transformed_predictors(lr, d['predictors'], d['precip_log_scale'])
            if target.shape != baseline.shape or cond.shape[1:] != static['area'].shape:
                raise ValueError('Regridded predictors and HR targets must share the LCC grid')
            raw_target = target.copy()
            before = budget_error(target[1], reference, static['area'], static['groups'])
            if d['conserve_training_precip']:
                target[1] = project_precip(target[1], reference, static['area'], static['groups'])
            audit.append({'id': e['id'], 'raw_hr_vs_native_budget': before,
                          'precip_adjustment_mae_mm_h': float(np.mean(np.abs(raw_target[1]-target[1]))),
                          'raw_hr_wet_in_native_dry_fraction': float(np.mean((raw_target[1] > .1) & (reference == 0)))})
            kw = (d['precip_log_scale'], d['wind_log_scale'])
            residual = transform_target(target, *kw)-transform_target(baseline, *kw)
            folder = root/e['id']
            tmp = root/(e['id']+'.tmp')
            if tmp.exists():
                shutil.rmtree(tmp)
            tmp.mkdir()
            for name, array in [('condition', cond), ('target', target), ('truth', raw_target), ('baseline', baseline), ('residual', residual)]:
                np.save(tmp/f'{name}.npy', array.astype('float32'))
            if folder.exists():
                shutil.rmtree(folder)
            os.replace(tmp, folder)
            if e['split'] == 'train':
                cond_mom.add(cond[:, ::stride, ::stride])
                residual_mom.add(residual[:, ::stride, ::stride])
        if i % 24 == 0:
            print(f'Prepared {i+1}/{len(entries)}: {e["id"]}', flush=True)
    stats = {'condition': cond_mom.result(), 'residual': residual_mom.result(), 'predictors': d['predictors'],
             'precip_log_scale': d['precip_log_scale'], 'wind_log_scale': d['wind_log_scale']}
    write_json(root/'stats.json', stats)
    write_json(root/'precip_audit.json', audit)
    fingerprint = hashlib.sha256(json.dumps({'data': d, 'entries': entries, 'stats': stats}, sort_keys=True).encode()).hexdigest()
    write_json(root/'index.json', {'entries': entries, 'data_config': d, 'fingerprint': fingerprint, 'condition_channels': len(d['predictors'])+16,
                                  'conservation': 'native cell center-assigned footprint; represented LCC AREA; not polygon overlap'})
    return root


def prepare_predict(cfg, reference_archive):
    """Prepare new LR hours without HR labels, reusing the training grid and stats."""
    from .dataset import Archive
    reference_archive = Archive(reference_archive)
    d, root = cfg['data'], Path(cfg['data']['prepared'])
    if root.resolve() == reference_archive.root.resolve() or (root/'index.json').exists():
        raise FileExistsError('Use a new data.prepared directory for unlabeled inference')
    for key in ('predictors', 'precip_log_scale', 'wind_log_scale'):
        if d[key] != reference_archive.stats[key]:
            raise ValueError(f'Inference preprocessing {key} differs from checkpoint training data')
    root.mkdir(parents=True, exist_ok=True)
    static = reference_archive.static
    entries, missing = [], []
    t, end = datetime.fromisoformat(d['start']), datetime.fromisoformat(d['end'])
    if t.minute != 30 or end.minute != 30 or t > end:
        raise ValueError('Use inclusive :30 start/end timestamps')
    while t <= end:
        tag = t.strftime('%Y%m%d_%H%M')
        lrpath = Path(d['lowres_lcc_root'])/t.strftime('%Y%m')/f'f5295_fp.lowres_lcc_1hr.{tag}z.nc4'
        nativepath = Path(d['native_root'])/t.strftime('Y%Y/M%m')/f'f5295_fp.tavg1_2d_flx_Nx.{tag}z.nc4'
        if not lrpath.exists() or not nativepath.exists():
            missing.append({'time': t.isoformat(), 'missing': [str(p) for p in (lrpath, nativepath) if not p.exists()]})
        else:
            entries.append({'time': t.isoformat(), 'id': tag, 'split': 'predict', 'lr': str(lrpath.resolve()), 'native': str(nativepath.resolve())})
        t += timedelta(hours=1)
    write_json(root/'missing.json', missing)
    if missing and d['strict_missing']:
        raise FileNotFoundError(f'Incomplete inference hours; see {root}/missing.json')
    if not entries:
        raise ValueError('No complete inference hours')
    for entry in entries:
        t = datetime.fromisoformat(entry['time'])
        with xr.open_dataset(entry['lr']) as lr, xr.open_dataset(entry['native']) as source:
            native = sorted_native(source)
            assert_time(lr, t, entry['lr'])
            assert_time(native, t, entry['native'])
            if not np.array_equal(native.lat, static['native_lat']) or not np.array_equal(native.lon, static['native_lon']):
                raise ValueError('Inference native grid differs from trained grid')
            units(native, 'PRECTOT', 'rate')
            for name, kind in [('T2M', 'temperature'), ('PS', 'pressure'), ('U10M', 'wind'), ('V10M', 'wind')]:
                units(lr, name, kind)
            pr = field(native, 'PRECTOT').ravel()[static['source_flat']]*3600
            if pr.min() < -1e-7:
                raise ValueError('Negative native precipitation')
            ref = np.maximum(pr, 0)[static['groups']]
            baseline = np.stack([field(lr, 'T2M'), ref, field(lr, 'PS'), np.hypot(field(lr, 'U10M'), field(lr, 'V10M'))])
            cond = transformed_predictors(lr, d['predictors'], d['precip_log_scale'])
            if baseline.shape[1:] != reference_archive.shape or cond.shape[1:] != reference_archive.shape:
                raise ValueError('Inference LCC grid shape mismatch')
            folder = root/entry['id']
            folder.mkdir(exist_ok=True)
            np.save(folder/'condition.npy', cond.astype('float32'))
            np.save(folder/'baseline.npy', baseline.astype('float32'))
    for name in ('static.npz', 'grid.nc', 'stats.json'):
        shutil.copyfile(reference_archive.root/name, root/name)
    write_json(root/'index.json', {**reference_archive.index, 'entries': entries, 'data_config': d,
                                  'reference_archive': str(reference_archive.root.resolve()),
                                  'inference_only': True})
    return root
