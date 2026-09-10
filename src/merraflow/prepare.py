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
BASELINE_VARS = {'T2M', 'PS', 'U10M', 'V10M'}


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


def manifest(cfg, month=None):
    d = cfg['data']
    # Validate ranges independently of which files happen to exist.
    ranges = sorted((datetime.fromisoformat(v[0]), datetime.fromisoformat(v[1]), k) for k, v in d['splits'].items())
    if set(d['splits']) != {'train', 'val', 'test'} or any(a >= b for a, b, _ in ranges):
        raise ValueError('Provide nonempty train, val and test half-open ranges')
    if any(ranges[i][1] > ranges[i+1][0] for i in range(len(ranges)-1)):
        raise ValueError('Split ranges overlap')
    if month is not None:
        try:
            parsed_month = datetime.strptime(month, '%Y-%m')
        except ValueError as exc:
            raise ValueError('Month must use YYYY-MM') from exc
        if parsed_month.strftime('%Y-%m') != month:
            raise ValueError('Month must use zero-padded YYYY-MM')
    entries, missing = [], []
    start, end = datetime.fromisoformat(d['start']), datetime.fromisoformat(d['end'])
    if start.minute != 30 or end.minute != 30 or start > end:
        raise ValueError('Inclusive data start/end must be hourly midpoint timestamps (:30)')
    t = start
    while t <= end:
        split = split_for(t, d['splits'])
        if split and (month is None or t.strftime('%Y-%m') == month):
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


def predictor_gaps(entries, predictors):
    """Return unreadable/incomplete regridded files before expensive preparation."""
    gaps = []
    required = set(predictors) | BASELINE_VARS
    for entry in entries:
        try:
            with xr.open_dataset(entry['lr'], decode_times=False) as ds:
                absent = sorted(required-set(ds.data_vars))
            if absent:
                gaps.append({'id': entry['id'], 'path': entry['lr'], 'missing': absent})
        except Exception as exc:
            gaps.append({'id': entry['id'], 'path': entry['lr'],
                         'error': f'{type(exc).__name__}: {exc}'})
    return gaps


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

    def _merge(self, n, mean, m2):
        if not n:
            return
        mean, m2 = np.asarray(mean, dtype='float64'), np.asarray(m2, dtype='float64')
        if self.n == 0:
            self.n, self.mean, self.m2 = int(n), mean.copy(), m2.copy()
            return
        delta = mean-self.mean
        total = self.n+n
        self.m2 += m2 + delta**2*self.n*n/total
        self.mean += delta*n/total
        self.n = int(total)

    def add(self, arr):
        x = arr.astype('float64').reshape(arr.shape[0], -1)
        self._merge(x.shape[1], x.mean(1), ((x-x.mean(1)[:, None])**2).sum(1))

    def merge(self, state):
        self._merge(state['n'], state['mean'], state['m2'])

    def state(self):
        return {'n': self.n,
                'mean': None if self.mean is None else self.mean.tolist(),
                'm2': None if self.m2 is None else self.m2.tolist()}

    def result(self):
        if not self.n:
            raise ValueError('No training samples for normalization')
        return {'mean': self.mean.tolist(), 'std': np.maximum(np.sqrt(self.m2/self.n), 1e-4).tolist(), 'count_per_channel': self.n}


SHARD_NAMES = ('condition', 'target', 'truth', 'baseline', 'residual')
PREPARATION_FORMAT = 2


def preparation_signature(cfg):
    payload = {'format': PREPARATION_FORMAT, 'data': cfg['data']}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def ensure_work_signature(root, cfg):
    path = root/'_preparation.json'
    expected = {'format': PREPARATION_FORMAT, 'signature': preparation_signature(cfg),
                'data_config': cfg['data']}
    if path.exists():
        with open(path) as source:
            actual = json.load(source)
        if actual != expected:
            raise ValueError(f'{root} contains shards from a different preparation configuration; use a new data.prepared directory')
    else:
        write_json(path, expected)


def static_for(entry):
    with xr.open_dataset(entry['hr']) as hr, xr.open_dataset(entry['native']) as raw:
        return make_static(hr, sorted_native(raw))


def grid_signature(static):
    digest = hashlib.sha256()
    for name in ('lat', 'lon', 'area', 'elevation', 'native_lat', 'native_lon'):
        value = np.ascontiguousarray(static[name])
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape, dtype='int64').tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def grid_mismatches(reference, candidate):
    """Report material grid changes while tolerating NetCDF round-off."""
    tolerances = {
        'lat': (0., 2e-5), 'lon': (0., 2e-5),
        'area': (1e-6, 1e-2), 'elevation': (1e-6, 1e-3),
        'native_lat': (0., 1e-7), 'native_lon': (0., 1e-7),
    }
    mismatches = []
    for name, (rtol, atol) in tolerances.items():
        left, right = np.asarray(reference[name]), np.asarray(candidate[name])
        if left.shape != right.shape:
            mismatches.append(f'{name} shape {left.shape} != {right.shape}')
        elif not np.allclose(left, right, rtol=rtol, atol=atol):
            mismatches.append(f'{name} max_abs_difference={float(np.max(np.abs(left-right))):.6g}')
    for name in ('groups', 'source_flat'):
        left, right = np.asarray(reference[name]), np.asarray(candidate[name])
        if left.shape != right.shape or not np.array_equal(left, right):
            mismatches.append(f'{name} mapping changed')
    return mismatches


def write_static(root, entry, static, data_config):
    with xr.open_dataset(entry['hr']) as hr:
        geo = xr.Dataset({k: (('Ydim', 'Xdim'), static[k], {'units': u}) for k, u in
                          [('lat', 'degrees_north'), ('lon', 'degrees_east'),
                           ('area', 'm2'), ('elevation', 'm')]})
        for coord in ('Xdim', 'Ydim'):
            if coord in hr:
                geo = geo.assign_coords({coord: hr[coord].load()})
        mappings = [k for k in hr if 'grid_mapping_name' in hr[k].attrs]
        for name in mappings:
            geo[name] = hr[name].load()
        if mappings:
            geo.attrs['grid_mapping_variable'] = mappings[0]
        geo.attrs.update({'grid': 'HWT LCC; coordinates preserved from source',
                          'state_alignment': data_config['state_alignment']})
        temporary = root/'grid.tmp.nc'
        geo.to_netcdf(temporary, engine='h5netcdf')
        os.replace(temporary, root/'grid.nc')
    temporary = root/'static.tmp.npz'
    np.savez(temporary, **static)
    os.replace(temporary, root/'static.npz')


def shard_complete(folder, static, predictor_count):
    expected = {'condition': (predictor_count, *static['area'].shape)}
    expected.update({name: (4, *static['area'].shape) for name in SHARD_NAMES if name != 'condition'})
    try:
        for name, shape in expected.items():
            array = np.load(folder/f'{name}.npy', mmap_mode='r')
            if array.shape != shape or array.dtype != np.dtype('float32'):
                return False
    except (FileNotFoundError, OSError, ValueError):
        return False
    return True


def audit_arrays(entry_id, truth, target, baseline, static):
    return {'id': entry_id,
            'raw_hr_vs_native_budget': budget_error(truth[1], baseline[1], static['area'], static['groups']),
            'precip_adjustment_mae_mm_h': float(np.mean(np.abs(truth[1]-target[1]))),
            'raw_hr_wet_in_native_dry_fraction': float(np.mean((truth[1] > .1) & (baseline[1] == 0)))}


def build_arrays(cfg, entry, static):
    d, t = cfg['data'], datetime.fromisoformat(entry['time'])
    with xr.open_dataset(entry['lr']) as lr, xr.open_dataset(entry['hr']) as hr, \
            xr.open_dataset(entry['acc']) as acc, xr.open_dataset(entry['native']) as raw:
        native = sorted_native(raw)
        for ds, when, key in [(lr, t, 'lr'), (hr, t, 'hr'),
                              (acc, t+timedelta(minutes=30), 'acc'), (native, t, 'native')]:
            assert_time(ds, when, entry[key])
        for key, expected in [('native_lat', native.lat.values), ('native_lon', native.lon.values)]:
            if not np.array_equal(static[key], expected):
                raise ValueError(f'{entry["native"]}: native grid changed within archive')
        for name, saved in [('lats', 'lat'), ('AREA', 'area')]:
            if not np.allclose(field(hr, name), static[saved], rtol=1e-6, atol=1e-5):
                raise ValueError(f'{entry["hr"]}: HR grid {name} changed within archive')
        if not np.allclose(((field(hr, 'lons')+180) % 360)-180, static['lon'], atol=1e-5):
            raise ValueError(f'{entry["hr"]}: HR longitude grid changed')
        bounds = acc.time.attrs.get('bounds')
        if bounds and bounds in acc:
            values = acc[bounds].values.ravel()
            expected = [np.datetime64(t-timedelta(minutes=30)), np.datetime64(t+timedelta(minutes=30))]
            if len(values) != 2 or not np.array_equal(values, expected):
                raise ValueError(f'{entry["acc"]}: APCP time bounds do not match the LR hourly window')
        for ds, name, kind in [(native, 'PRECTOT', 'rate'), (acc, 'APCP', 'accum'),
                               (hr, 'TMP_2M', 'temperature'), (hr, 'PRES_SFC', 'pressure'),
                               (lr, 'T2M', 'temperature'), (lr, 'PS', 'pressure')]:
            units(ds, name, kind)
        for ds, names in [(hr, ['UGRD_10M', 'VGRD_10M']), (lr, ['U10M', 'V10M'])]:
            for name in names:
                units(ds, name, 'wind')
        source_pr = field(native, 'PRECTOT').ravel()[static['source_flat']]*3600
        if source_pr.min() < -1e-7:
            raise ValueError(f'{entry["native"]}: negative native precipitation')
        reference = np.maximum(source_pr, 0)[static['groups']]
        precip = field(acc, 'APCP')/d['accumulation_hours']
        if precip.min() < -1e-7:
            raise ValueError(f'{entry["acc"]}: negative APCP')
        target = np.stack([field(hr, 'TMP_2M'), np.maximum(precip, 0),
                           field(hr, 'PRES_SFC'), np.hypot(field(hr, 'UGRD_10M'), field(hr, 'VGRD_10M'))])
        baseline = np.stack([field(lr, 'T2M'), reference, field(lr, 'PS'),
                             np.hypot(field(lr, 'U10M'), field(lr, 'V10M'))])
        condition = transformed_predictors(lr, d['predictors'], d['precip_log_scale'])
        if target.shape != baseline.shape or condition.shape[1:] != static['area'].shape:
            raise ValueError(f'{entry["id"]}: regridded predictors and HR targets must share the LCC grid')
        truth = target.copy()
        if d['conserve_training_precip']:
            target[1] = project_precip(target[1], reference, static['area'], static['groups'])
        scales = (d['precip_log_scale'], d['wind_log_scale'])
        residual = transform_target(target, *scales)-transform_target(baseline, *scales)
    return {'condition': condition, 'target': target, 'truth': truth,
            'baseline': baseline, 'residual': residual}


def write_shard(root, entry_id, arrays):
    folder, temporary = root/entry_id, root/(entry_id+'.tmp')
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    for name in SHARD_NAMES:
        np.save(temporary/f'{name}.npy', arrays[name].astype('float32'))
    if folder.exists():
        shutil.rmtree(folder)
    os.replace(temporary, folder)


def process_entries(cfg, entries, static, label='all'):
    d, stride = cfg['data'], cfg['data']['stats_stride']
    if stride < 1 or d['precip_log_scale'] <= 0 or d['wind_log_scale'] <= 0:
        raise ValueError('Statistics stride and transform scales must be positive')
    root = Path(d['prepared'])
    condition_moments, residual_moments = Moments(), Moments()
    audit, skipped, written = [], 0, 0
    for index, entry in enumerate(entries):
        folder = root/entry['id']
        if shard_complete(folder, static, len(d['predictors'])):
            arrays = {name: np.load(folder/f'{name}.npy', mmap_mode='r') for name in SHARD_NAMES}
            skipped += 1
        else:
            arrays = build_arrays(cfg, entry, static)
            write_shard(root, entry['id'], arrays)
            written += 1
        if entry['split'] == 'train':
            condition_moments.add(arrays['condition'][:, ::stride, ::stride])
            residual_moments.add(arrays['residual'][:, ::stride, ::stride])
        audit.append(audit_arrays(entry['id'], arrays['truth'], arrays['target'], arrays['baseline'], static))
        if index % 24 == 0:
            print(f'{label}: verified {index+1}/{len(entries)} {entry["id"]} '
                  f'(written={written}, skipped={skipped})', flush=True)
    return {'condition': condition_moments.state(), 'residual': residual_moments.state(),
            'audit': audit, 'written': written, 'skipped': skipped}


def finish_archive(cfg, entries, missing, gaps, static, condition_moments, residual_moments, audit):
    d, root = cfg['data'], Path(cfg['data']['prepared'])
    write_static(root, entries[0], static, d)
    stats = {'condition': condition_moments.result(), 'residual': residual_moments.result(),
             'predictors': d['predictors'], 'precip_log_scale': d['precip_log_scale'],
             'wind_log_scale': d['wind_log_scale']}
    write_json(root/'missing.json', missing)
    write_json(root/'predictor_gaps.json', gaps)
    write_json(root/'stats.json', stats)
    write_json(root/'precip_audit.json', audit)
    fingerprint = hashlib.sha256(json.dumps({'data': d, 'entries': entries, 'stats': stats}, sort_keys=True).encode()).hexdigest()
    write_json(root/'index.json', {'entries': entries, 'data_config': d, 'fingerprint': fingerprint,
                                  'condition_channels': len(d['predictors'])+16,
                                  'conservation': 'native cell center-assigned footprint; represented LCC AREA; not polygon overlap'})
    return root


def validate_manifest(cfg, entries, missing, require_splits):
    d = cfg['data']
    if missing and d['strict_missing']:
        raise FileNotFoundError(f'{len(missing)} incomplete hourly pairs; adjust dates or explicitly disable strict_missing')
    if not entries:
        raise ValueError('No complete hours to prepare')
    if require_splits and any(not any(e['split'] == split for e in entries) for split in ('train', 'val', 'test')):
        raise ValueError('Need at least one complete hour in each train/val/test split')


def prepare(cfg):
    d, root = cfg['data'], Path(cfg['data']['prepared'])
    root.mkdir(parents=True, exist_ok=True)
    if (root/'index.json').exists():
        raise FileExistsError(f'{root} already prepared; use a new directory to prevent stale shards/statistics')
    ensure_work_signature(root, cfg)
    entries, missing = manifest(cfg)
    write_json(root/'missing.json', missing)
    validate_manifest(cfg, entries, missing, require_splits=True)
    gaps = predictor_gaps(entries, d['predictors'])
    write_json(root/'predictor_gaps.json', gaps)
    if gaps:
        raise ValueError(f'{len(gaps)} regridded files lack required predictors or are unreadable; see {root}/predictor_gaps.json')
    static = static_for(entries[0])
    result = process_entries(cfg, entries, static)
    condition_moments, residual_moments = Moments(), Moments()
    condition_moments.merge(result['condition'])
    residual_moments.merge(result['residual'])
    return finish_archive(cfg, entries, missing, gaps, static, condition_moments,
                          residual_moments, result['audit'])


def prepare_month(cfg, month):
    d, root = cfg['data'], Path(cfg['data']['prepared'])
    root.mkdir(parents=True, exist_ok=True)
    if (root/'index.json').exists():
        with open(root/'index.json') as source:
            index = json.load(source)
        if index['data_config'] != d:
            raise ValueError(f'{root} is complete for a different data configuration')
        print(f'{root} is already complete; month {month} skipped', flush=True)
        return root/'index.json'
    ensure_work_signature(root, cfg)
    entries, missing = manifest(cfg, month=month)
    part = root/'_monthly'
    part.mkdir(exist_ok=True)
    write_json(part/f'{month}.issues.json', {'missing': missing})
    validate_manifest(cfg, entries, missing, require_splits=False)
    gaps = predictor_gaps(entries, d['predictors'])
    write_json(part/f'{month}.issues.json', {'missing': missing, 'predictor_gaps': gaps})
    if gaps:
        raise ValueError(f'{len(gaps)} regridded files in {month} lack required inputs; see {part}/{month}.issues.json')
    static = static_for(entries[0])
    result = process_entries(cfg, entries, static, label=month)
    metadata = {'month': month, 'signature': preparation_signature(cfg),
                'grid_signature': grid_signature(static),
                'entry_ids': [entry['id'] for entry in entries],
                'missing': missing, **result}
    destination = part/f'{month}.json'
    write_json(destination, metadata)
    print(f'{month} complete: written={result["written"]}, skipped={result["skipped"]}', flush=True)
    return destination


def finalize_prepare(cfg):
    d, root = cfg['data'], Path(cfg['data']['prepared'])
    if (root/'index.json').exists():
        with open(root/'index.json') as source:
            index = json.load(source)
        if index['data_config'] != d:
            raise ValueError(f'{root} is complete for a different data configuration')
        return root
    ensure_work_signature(root, cfg)
    entries, missing = manifest(cfg)
    validate_manifest(cfg, entries, missing, require_splits=True)
    static = static_for(entries[0])
    condition_moments, residual_moments = Moments(), Moments()
    audit, gaps = [], []
    for month in sorted({entry['time'][:7] for entry in entries}):
        path = root/'_monthly'/f'{month}.json'
        if not path.exists():
            raise FileNotFoundError(f'Month {month} is incomplete: {path} does not exist')
        with open(path) as source:
            metadata = json.load(source)
        expected_ids = [entry['id'] for entry in entries if entry['time'].startswith(month)]
        if metadata['signature'] != preparation_signature(cfg) or metadata['entry_ids'] != expected_ids:
            raise ValueError(f'{path} does not match the current configuration/manifest')
        month_entry = next(entry for entry in entries if entry['time'].startswith(month))
        month_static = static_for(month_entry)
        if metadata['grid_signature'] != grid_signature(month_static):
            raise ValueError(f'{path}: source grid changed after this month was prepared')
        mismatches = grid_mismatches(static, month_static)
        if mismatches:
            raise ValueError(f'{path}: spatial grid differs materially: {"; ".join(mismatches)}')
        for entry_id in expected_ids:
            if not shard_complete(root/entry_id, static, len(d['predictors'])):
                raise ValueError(f'{root/entry_id}: monthly metadata exists but shard is incomplete')
        condition_moments.merge(metadata['condition'])
        residual_moments.merge(metadata['residual'])
        audit.extend(metadata['audit'])
        gaps.extend(metadata.get('predictor_gaps', []))
    if len(audit) != len(entries):
        raise ValueError('Monthly precipitation audit count does not match the manifest')
    return finish_archive(cfg, entries, missing, gaps, static, condition_moments,
                          residual_moments, audit)


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
