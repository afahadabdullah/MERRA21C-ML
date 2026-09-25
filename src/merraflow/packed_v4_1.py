"""v4.1 packed training archive: tiled hourly fields plus precomputed broad context.

Why this exists
---------------
The v4 loader cropped a 576x576 broad context from three full-resolution
memory maps per sample (condition, baseline, previous-hour condition), then
area-averaged it 6x down to 96x96. That is ~23 MB of strided GPFS page faults
per sample, and for most patch positions the crop crosses the domain edge and
falls back to element-wise fancy indexing. Training was >95% data-wait.

v4.1 does the expensive, sample-independent work once per hour:

* ``cond``  raw condition predictors, tiled ``(tiles_y, tiles_x, C, T, T)``
* ``rest``  raw baseline (5) + physical truth (t2m, hourly rain, ps, u, v, q2m)
* ``ctx``   the broad context's per-hour channels (normalized dynamic
            predictors and encoded baseline), already area-pooled on a
            replicate-padded grid, tiled the same way
* ``score`` the input-only coarse-rain proposal score for every candidate

A local crop is a handful of contiguous ``pread`` calls into one file per hour,
and the broad context is a 96x96 crop of a small pooled grid.

Exactness contract
------------------
Candidate patch origins are restricted to multiples of the context pooling
factor ``f = broad/context_size`` (6 in production). The pooled grid's blocks
are then exactly the blocks v4 averaged for those origins, including the
replicated rows/columns outside the domain, so every tensor a v4.1 sample
contains equals what v4's ``DatasetV4`` produces for the same hour and origin.
``tests/test_v4_1.py`` checks this sample by sample.
"""
from datetime import datetime
from pathlib import Path
import json
import os
import numpy as np
import torch
from torch.nn import functional as F
from .config import write_json
from .dataset_v2 import box_means_v2
from .prepare_v3_precip import digest, file_identity

FORMAT = 'v4.1-packed-1'
INDEX = 'index_v4_1.json'
SCORES = 'scores_v4_1.npy'
STATIC = 'static_v4_1'
REST_CHANNELS = ('baseline_t2m', 'baseline_precip', 'baseline_ps', 'baseline_u10m', 'baseline_v10m',
                 't2m', 'precip_hourly', 'ps', 'u10m', 'v10m', 'q2m')
BASELINE = slice(0, 5)
TRUTH = slice(5, 11)


# ----------------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------------

def candidates_v4_1(shape, size, stride, factor):
    """Candidate origins on multiples of ``factor``; the last origin per axis is
    snapped down so it stays aligned (at most factor-1 edge pixels move from
    the patch interior to its halo)."""
    if min(shape) < size:
        raise ValueError('Patch exceeds archive grid')
    if stride % factor:
        raise ValueError(f'sampling_stride must be a multiple of the context pooling factor {factor}')
    axes = [np.unique(np.r_[np.arange(0, n-size+1, stride), (n-size)//factor*factor]) for n in shape]
    yy, xx = np.meshgrid(*axes, indexing='ij')
    return yy.ravel().astype(int), xx.ravel().astype(int)


def _section(offset, channels, rows, cols, tile, itemsize=4):
    nbytes = rows*cols*channels*tile*tile*itemsize
    return dict(offset=offset, channels=channels, tile=tile, tiles_y=rows, tiles_x=cols, nbytes=nbytes)


def geometry(cfg, shape, predictors):
    """All derived sizes and byte offsets; identical for every hour.

    ``predictors`` is the number of raw dynamic predictor channels (6)."""
    p, pk = cfg['patch'], cfg['packed']
    size, halo = p['size'], p['halo']
    width = size+2*halo
    broad = width*p['context_scale']
    if broad % p['context_size']:
        raise ValueError('Broad context width must be an integer multiple of context_size')
    factor = broad//p['context_size']
    offset = (broad-size)//2
    yy, xx = candidates_v4_1(shape, size, p['sampling_stride'], factor)
    h, w = shape
    pooled = ((int(yy.max())+broad)//factor, (int(xx.max())+broad)//factor)
    tile, ctile = pk['tile'], pk['context_tile']
    ty, tx = -(-h//tile), -(-w//tile)
    cy, cx = -(-pooled[0]//ctile), -(-pooled[1]//ctile)
    ctx_channels = predictors+5
    sections = {}
    at = 0
    for name, channels, rows, cols, t in (('cond', predictors, ty, tx, tile),
                                           ('rest', len(REST_CHANNELS), ty, tx, tile),
                                           ('ctx', ctx_channels, cy, cx, ctile)):
        sections[name] = _section(at, channels, rows, cols, t)
        at += sections[name]['nbytes']
    sections['score'] = dict(offset=at, count=len(yy), nbytes=len(yy)*8)
    at += sections['score']['nbytes']
    return dict(shape=[h, w], size=size, halo=halo, window=width, broad=broad, factor=factor,
                offset=offset, context_size=p['context_size'], sampling_stride=p['sampling_stride'],
                candidates=len(yy), pooled=list(pooled), sections=sections, file_bytes=at)


def hour_path(root, entry):
    return Path(root)/entry['time'][:7].replace('-', '')/f'{entry["id"]}.v4_1.bin'


# ----------------------------------------------------------------------------
# Writing (one-time preparation)
# ----------------------------------------------------------------------------

def _tiles(arr, tile, rows, cols):
    """(C,H,W) -> contiguous (rows, cols, C, tile, tile); edge-replicated padding."""
    arr = np.asarray(arr, dtype='float32')
    c, h, w = arr.shape
    iy = np.minimum(np.arange(rows*tile), h-1)
    ix = np.minimum(np.arange(cols*tile), w-1)
    padded = arr[:, iy][:, :, ix]
    return np.ascontiguousarray(padded.reshape(c, rows, tile, cols, tile).transpose(1, 3, 0, 2, 4))


def pooled_indices(geo):
    """Source rows/cols (replicate-clipped) of the padded grid that is pooled."""
    h, w = geo['shape']
    f, off = geo['factor'], geo['offset']
    iy = np.clip(np.arange(geo['pooled'][0]*f)-off, 0, h-1)
    ix = np.clip(np.arange(geo['pooled'][1]*f)-off, 0, w-1)
    return iy, ix


def area_pool(stack, geo):
    """Area pooling identical to v4's F.interpolate(..., mode='area') on aligned blocks."""
    tensor = torch.from_numpy(np.ascontiguousarray(stack, dtype='float32'))[None]
    return F.interpolate(tensor, size=tuple(geo['pooled']), mode='area')[0].numpy()


def dynamic_inputs(archive, condition):
    """Per-pixel normalized predictors, exactly as ArchiveV2.condition computes them."""
    return (np.asarray(condition, dtype='float32')-archive.cm)/archive.cs


def baseline_inputs(archive, baseline):
    """Per-pixel encoded baseline, exactly as ArchiveV2.condition computes it."""
    base = archive.encode(np.asarray(baseline, dtype='float32'))
    base[0] = (base[0]-280)/20
    base[2] = (base[2]-90000)/15000
    base[3:] /= 10
    return base


def rain_score(baseline_rain, geo, yy, xx):
    """Input-only coarse rain score, as v4's PrecipDataset.rain_score('coarse')."""
    rain = np.asarray(baseline_rain)
    return box_means_v2(np.log1p(np.maximum(rain, 0)), yy, xx, geo['size'])


def pack_hour(archive, geo, entry, targets):
    """Return the bytes-ready sections for one hour (all float32 except score)."""
    s = geo['sections']
    condition = np.asarray(archive.array(entry, 'condition'), dtype='float32')
    baseline = np.asarray(archive.array(entry, 'baseline'), dtype='float32')
    if not (np.isfinite(condition).all() and np.isfinite(baseline).all()):
        raise ValueError(f'{entry["id"]}: nonfinite condition/baseline input')
    if targets:
        truth = np.array(archive.array(entry, 'truth'), dtype='float32', copy=True)
        rain = np.asarray(archive.truth_field(entry), dtype='float32')
        if not np.isfinite(rain).all() or np.any(rain < 0):
            raise ValueError(f'{entry["id"]}: invalid hourly truth precipitation')
        truth[1:2] = rain
        humidity = np.asarray(archive.humidity(entry), dtype='float32')
        truth = np.concatenate([truth, humidity])
        if not np.isfinite(truth).all():
            raise ValueError(f'{entry["id"]}: nonfinite truth')
    else:
        # History-only hour: never supervised. NaN makes accidental use loud.
        truth = np.full((6, *condition.shape[-2:]), np.nan, dtype='float32')
    rest = np.concatenate([baseline, truth])
    iy, ix = pooled_indices(geo)
    stack = np.concatenate([dynamic_inputs(archive, condition)[:, iy][:, :, ix],
                            baseline_inputs(archive, baseline)[:, iy][:, :, ix]])
    ctx = area_pool(stack, geo)
    del stack
    yy, xx = candidates_v4_1(geo['shape'], geo['size'], geo['sampling_stride'], geo['factor'])
    score = rain_score(baseline[1], geo, yy, xx) if targets else np.full(len(yy), np.nan)
    return dict(cond=_tiles(condition, s['cond']['tile'], s['cond']['tiles_y'], s['cond']['tiles_x']),
                rest=_tiles(rest, s['rest']['tile'], s['rest']['tiles_y'], s['rest']['tiles_x']),
                ctx=_tiles(ctx, s['ctx']['tile'], s['ctx']['tiles_y'], s['ctx']['tiles_x']),
                score=np.asarray(score, dtype='<f8'))


def write_hour(path, geo, sections):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    try:
        with open(tmp, 'wb') as stream:
            for name in ('cond', 'rest', 'ctx', 'score'):
                value = sections[name]
                if value.nbytes != geo['sections'][name]['nbytes']:
                    raise ValueError(f'{path}: {name} has {value.nbytes} bytes, expected {geo["sections"][name]["nbytes"]}')
                value.astype('<f4' if name != 'score' else '<f8', copy=False).tofile(stream)
        if tmp.stat().st_size != geo['file_bytes']:
            raise ValueError(f'{path}: unexpected packed size')
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


# ----------------------------------------------------------------------------
# Plan, contract, prepare, finalize
# ----------------------------------------------------------------------------

def plan(cfg, archive):
    """Hours to pack: eligible hours of the configured splits plus their history."""
    wanted = {}
    for split in cfg['packed']['splits']:
        for entry in archive.eligible(split):
            history = archive.history(entry)[:-1]
            wanted[entry['id']] = dict(id=entry['id'], time=entry['time'], split=split, targets=True,
                                       history=[h['id'] for h in history])
            for h in history:
                wanted.setdefault(h['id'], dict(id=h['id'], time=h['time'], split=h['split'],
                                                targets=False, history=[]))
    order = {e['id']: i for i, e in enumerate(archive.index['entries'])}
    return sorted(wanted.values(), key=lambda e: order[e['id']])


def contract(cfg, archive):
    geo = geometry(cfg, archive.shape, len(archive.cm))
    return dict(format=FORMAT, archive_fingerprint=archive.index['fingerprint'], stats=digest(archive.stats),
                hourly_fingerprint=archive.hourly_fingerprint, humidity_fingerprint=archive.humidity_fingerprint,
                condition_channels=int(archive.index['condition_channels']), predictors=len(archive.cm),
                history_hours=list(cfg['data']['history_hours']), splits=list(cfg['packed']['splits']),
                rest_channels=list(REST_CHANNELS), geometry=geo)


def _sources(archive, entry, targets):
    names = ['condition', 'baseline'] + (['truth'] if targets else [])
    paths = [archive.root/entry['id']/f'{n}_v2.npy' for n in names]
    if targets:
        paths += [archive.hourly_root/entry['id']/'truth_hourly_v3_precip.npy',
                  archive.humidity_root/entry['id']/'q2m_v4.npy']
    return [file_identity(p) for p in paths]


def _sidecar(path):
    return path.with_suffix('.json')


def _complete(path, sidecar, geo, key):
    try:
        meta = json.loads(sidecar.read_text())
        return meta.get('contract') == key and path.stat().st_size == geo['file_bytes']
    except (OSError, ValueError):
        return False


def prepare(cfg, archive, month=None, limit=None):
    """Restart-safe packing of one month (or everything). Returns counts."""
    root = Path(cfg['packed']['root'])
    terms = contract(cfg, archive)
    key, geo = digest(terms), terms['geometry']
    entries = [e for e in plan(cfg, archive) if month is None or e['time'].startswith(month)]
    if month is not None and not entries:
        raise ValueError(f'No v4.1 hours to pack for {month}')
    written = skipped = 0
    for entry in entries[:limit]:
        path = hour_path(root, entry)
        sidecar = _sidecar(path)
        if _complete(path, sidecar, geo, key):
            skipped += 1
            continue
        source = archive.by_time[datetime.fromisoformat(entry['time'])]
        before = _sources(archive, source, entry['targets'])
        sections = pack_hour(archive, geo, source, entry['targets'])
        if _sources(archive, source, entry['targets']) != before:
            raise ValueError(f'{entry["id"]}: source files changed while packing')
        write_hour(path, geo, sections)
        write_json(sidecar, dict(contract=key, id=entry['id'], time=entry['time'], targets=entry['targets'],
                                 sources=before, bytes=geo['file_bytes']))
        written += 1
        print(f'Packed {entry["id"]} ({"targets" if entry["targets"] else "history"})', flush=True)
    return dict(written=written, skipped=skipped, bytes_per_hour=geo['file_bytes'])


def finalize(cfg, archive):
    """Check every planned hour and write the manifest plus the proposal score table."""
    root = Path(cfg['packed']['root'])
    terms = contract(cfg, archive)
    key, geo = digest(terms), terms['geometry']
    entries = plan(cfg, archive)
    missing, records = [], {}
    for entry in entries:
        path, sidecar = hour_path(root, entry), _sidecar(hour_path(root, entry))
        if not _complete(path, sidecar, geo, key):
            missing.append(entry['id'])
            continue
        records[entry['id']] = digest(json.loads(sidecar.read_text()))
    if missing:
        raise FileNotFoundError(f'{len(missing)} v4.1 hours are not packed (first: {missing[:5]}); '
                                'rerun prepare-packed for their months')
    supervised = [e for e in entries if e['targets']]
    scores = np.lib.format.open_memmap(root/f'.{SCORES}.{os.getpid()}.tmp', mode='w+', dtype='<f8',
                                       shape=(len(supervised), geo['candidates']))
    s = geo['sections']['score']
    for row, entry in enumerate(supervised):
        with open(hour_path(root, entry), 'rb') as stream:
            stream.seek(s['offset'])
            value = np.frombuffer(stream.read(s['nbytes']), dtype='<f8')
        if value.shape != (geo['candidates'],) or not np.isfinite(value).all():
            raise ValueError(f'{entry["id"]}: invalid packed proposal score')
        scores[row] = value
        entry['row'] = row
    scores.flush()
    del scores
    os.replace(root/f'.{SCORES}.{os.getpid()}.tmp', root/SCORES)
    statics = write_statics(root, archive, geo, entries)
    payload = dict(contract=terms, contract_digest=key, entries=entries, records=records,
                   scores=dict(file=SCORES, rows=len(supervised)), statics=statics)
    payload['fingerprint'] = digest(payload)
    write_json(root/INDEX, payload)
    total = geo['file_bytes']*len(entries)
    return dict(hours=len(entries), supervised=len(supervised), fingerprint=payload['fingerprint'],
                terabytes=round(total/1e12, 3))


def write_statics(root, archive, geo, entries):
    """Static inputs as .npy files that loader workers memory-map (page-cache
    shared) instead of receiving ~130 MB of pickled arrays each at startup.

    Also precomputes the pooled static features and the pooled time-of-day
    fields, so neither the trainer nor any worker computes them."""
    from .dataset_v2 import time_features
    folder = Path(root)/STATIC
    folder.mkdir(parents=True, exist_ok=True)
    iy, ix = pooled_indices(geo)
    lon_padded = np.ascontiguousarray(np.asarray(archive.static['lon'])[iy][:, ix])
    times = sorted({e['time'][11:16]: e['time'] for e in entries}.items())
    arrays = {k: np.asarray(archive.static[k]) for k in ('features', 'lon', 'area', 'land_fraction')}
    arrays['pooled_features'] = area_pool(np.asarray(archive.static['features'])[:, iy][:, :, ix], geo)
    arrays['pooled_daily'] = np.stack([area_pool(time_features(t, lon_padded)[4:6], geo) for _, t in times])
    arrays['lon_padded'] = lon_padded
    for name, value in arrays.items():
        tmp = folder/f'.{name}.{os.getpid()}.tmp.npy'
        np.save(tmp, value, allow_pickle=False)
        os.replace(tmp, folder/f'{name}.npy')
    return dict(folder=STATIC, daily_keys=[k for k, _ in times],
                shapes={k: list(v.shape) for k, v in arrays.items()})


def load_manifest(cfg, archive, verify_files=False):
    root = Path(cfg['packed']['root'])
    path = root/INDEX
    if not path.exists():
        raise FileNotFoundError(f'{path} missing: run cli_v4_1 prepare-packed then finalize-packed')
    manifest = json.loads(path.read_text())
    if manifest.get('fingerprint') != digest({k: v for k, v in manifest.items() if k != 'fingerprint'}):
        raise ValueError('v4.1 packed manifest was modified')
    if manifest['contract'] != contract(cfg, archive):
        raise ValueError('v4.1 packed archive was built for different data, geometry or splits; '
                         'repack into a fresh packed.root')
    if verify_files:
        geo = manifest['contract']['geometry']
        for entry in manifest['entries']:
            path, sidecar = hour_path(root, entry), _sidecar(hour_path(root, entry))
            if not _complete(path, sidecar, geo, manifest['contract_digest']):
                raise ValueError(f'{path}: packed hour missing or truncated')
            if digest(json.loads(sidecar.read_text())) != manifest['records'][entry['id']]:
                raise ValueError(f'{path}: packed hour changed after finalization')
    return manifest


# ----------------------------------------------------------------------------
# Reading
# ----------------------------------------------------------------------------

def _read_exact(fd, buffer, offset):
    view = memoryview(buffer).cast('B')
    done = 0
    while done < len(view):
        if hasattr(os, 'preadv'):
            count = os.preadv(fd, [view[done:]], offset+done)
        else:  # pragma: no cover - platforms without preadv
            chunk = os.pread(fd, len(view)-done, offset+done)
            view[done:done+len(chunk)] = chunk
            count = len(chunk)
        if count <= 0:
            raise EOFError(f'Short read at byte {offset+done}')
        done += count


class HourFiles:
    """Per-process LRU of open packed-hour descriptors plus tiled block reads."""

    def __init__(self, root, geo, cache_size=64):
        self.root, self.geo, self.cache_size = Path(root), geo, max(1, int(cache_size))
        self._fds = {}

    def __getstate__(self):
        return dict(self.__dict__, _fds={})

    def close(self):
        for fd in self._fds.values():
            os.close(fd)
        self._fds.clear()

    def fd(self, entry):
        key = entry['id']
        if key in self._fds:
            self._fds[key] = self._fds.pop(key)  # most recent last
            return self._fds[key]
        fd = os.open(hour_path(self.root, entry), os.O_RDONLY)
        self._fds[key] = fd
        while len(self._fds) > self.cache_size:
            os.close(self._fds.pop(next(iter(self._fds))))
        return fd

    def block(self, entry, name, r0, r1, c0, c1):
        """Rows [r0,r1) x cols [c0,c1) of a tiled section, all channels, (C, r, c)."""
        s = self.geo['sections'][name]
        t, ch = s['tile'], s['channels']
        ty0, ty1, tx0, tx1 = r0//t, (r1-1)//t, c0//t, (c1-1)//t
        tile_bytes = ch*t*t*4
        buf = np.empty((ty1-ty0+1, tx1-tx0+1, ch, t, t), dtype='<f4')
        fd = self.fd(entry)
        for j, ty in enumerate(range(ty0, ty1+1)):
            # Tiles along a row are contiguous: one read per tile row.
            _read_exact(fd, buf[j], s['offset']+(ty*s['tiles_x']+tx0)*tile_bytes)
        full = buf.transpose(2, 0, 3, 1, 4).reshape(ch, (ty1-ty0+1)*t, (tx1-tx0+1)*t)
        return full[:, r0-ty0*t:r1-ty0*t, c0-tx0*t:c1-tx0*t]

    def crop(self, entry, name, y0, x0, n):
        """Same values as dataset.crop/crop_v2 of the source field (edge replicate)."""
        h, w = self.geo['shape']
        if y0 >= 0 and x0 >= 0 and y0+n <= h and x0+n <= w:
            return np.ascontiguousarray(self.block(entry, name, y0, y0+n, x0, x0+n))
        iy = np.clip(np.arange(y0, y0+n), 0, h-1)
        ix = np.clip(np.arange(x0, x0+n), 0, w-1)
        value = self.block(entry, name, int(iy.min()), int(iy.max())+1, int(ix.min()), int(ix.max())+1)
        return np.ascontiguousarray(value[:, iy-iy.min()][:, :, ix-ix.min()])
