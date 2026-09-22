"""Hourly-mean HWT precipitation targets for v3_precip (fix 6).

GEOS-FP PRECTOT at HH:30 is a mean over HH:00-HH+1:00. The v2 archive pairs it
with a single HWT snapshot at HH:30. Here each hour instead gets the trapezoid
mean of the three HWT 30-minute snapshots bounding that window:

    P_hour = 1/4 P(HH:00) + 1/2 P(HH:30) + 1/4 P(HH+1:00)

This is still an approximation (three instantaneous rates, not a time integral),
and can still miss short convective bursts. The v2 archive stays
read-only: targets go to ``data.hourly_targets`` as one float32 (1, H, W) array
per hour, and the archived :30 truth is checked against the :30 snapshot so a
mis-paired file cannot slip in.

Workflow (Discover): ``prepare-hourly --month YYYY-MM`` per month (Slurm array),
then ``finalize-hourly`` writes the index that training requires.
"""
from datetime import datetime, timedelta
from pathlib import Path
import json
import os
import hashlib
import fcntl
import numpy as np
import xarray as xr
from .config import write_json
from .dataset_v2 import ArchiveV2
from .dataset_v3_precip import HOURLY_FILE, HOURLY_INDEX
from .prepare_v2 import field, units, assert_time

METHOD = 'trapezoid_00_30_00'
WEIGHTS = (.25, .5, .25)
PROVENANCE = 'provenance_v3_precip.json'
TARGET_META = 'provenance_hourly_v3_precip.json'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_digest(path):
    value = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            value.update(chunk)
    return value.hexdigest()


def file_identity(path):
    stat = path.stat()
    return dict(path=str(path.resolve()), size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def contract(cfg, archive):
    root = cfg['data'].get('highres_root')
    return dict(format=1, archive_fingerprint=archive.index['fingerprint'], method=METHOD,
                weights=list(WEIGHTS), shape=list(archive.shape),
                highres_root=str(Path(root).resolve()) if root else None)


def ensure_provenance(cfg, archive, create=False):
    out = Path(cfg['data']['hourly_targets'])
    expected = contract(cfg, archive)
    path = out/PROVENANCE
    if create:
        out.mkdir(parents=True, exist_ok=True)
        # Monthly array tasks can reach the first manifest simultaneously.
        with open(out/'.provenance.lock', 'a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not path.exists():
                if any(out.glob('*/'+HOURLY_FILE)) or (out/HOURLY_INDEX).exists():
                    raise ValueError('Unverified legacy hourly targets: use a fresh hourly_targets directory')
                write_json(path, expected)
    if not path.exists() or json.loads(path.read_text()) != expected:
        raise ValueError('Hourly source/method provenance mismatch: use a fresh hourly_targets directory')
    return expected


def source_identity(entry, highres_root=None):
    middle = datetime.fromisoformat(entry['time'])
    return [file_identity(snapshot_path(entry, middle+timedelta(minutes=m), highres_root)) for m in (-30, 0, 30)]


def verify_target(cfg, archive, entry, provenance, checksum=False):
    dest = Path(cfg['data']['hourly_targets'])/entry['id']/HOURLY_FILE
    meta_path = dest.parent/TARGET_META
    if not meta_path.exists():
        raise ValueError(f'{dest}: missing target provenance; use a fresh hourly_targets directory')
    meta = json.loads(meta_path.read_text())
    try:
        matches = (meta['contract_sha256'] == digest(provenance)
                   and meta['sources'] == source_identity(entry, cfg['data'].get('highres_root'))
                   and meta['target'] == file_identity(dest))
    except (KeyError, OSError):
        matches = False
    if not matches or not target_valid(dest, archive.shape):
        raise ValueError(f'{dest}: hourly target/source provenance mismatch; use a fresh directory')
    if checksum and file_digest(dest) != meta['target_sha256']:
        raise ValueError(f'{dest}: hourly target checksum mismatch')
    return digest(meta)


def validate_hourly_index(cfg, archive, index):
    provenance = ensure_provenance(cfg, archive)
    payload = {k: v for k, v in index.items() if k != 'fingerprint'}
    if (index.get('fingerprint') != digest(payload)
            or index.get('contract_sha256') != digest(provenance)):
        raise ValueError('Hourly index fingerprint mismatch; re-finalize verified targets')
    entries = {e['id']: e for e in archive.index['entries']}
    if len(set(index['completed'])) != len(index['completed']):
        raise ValueError('Duplicate hourly target IDs')
    for entry_id in index['completed']:
        if entry_id not in entries or verify_target(cfg, archive, entries[entry_id], provenance) != index['target_fingerprints'].get(entry_id):
            raise ValueError(f'Hourly target changed after finalization: {entry_id}')


def snapshot_path(entry, time, highres_root=None):
    """HWT 30-min file for ``time``, following the archived :30 file's naming."""
    source = Path(entry['hr'])
    middle = datetime.fromisoformat(entry['time'])
    root = Path(highres_root)/'hwt_30mn_slv_LCC' if highres_root else source.parents[1]
    name = source.name.replace(middle.strftime('%Y%m%d_%H%M'), time.strftime('%Y%m%d_%H%M'))
    if name == source.name and time != middle:
        raise ValueError(f'Cannot derive HWT snapshot names from {source.name}')
    return root/time.strftime('%Y%m')/name


def read_rate(path, time):
    with xr.open_dataset(path) as ds:
        assert_time(ds, time, path)
        units(ds, 'PRECTOT', 'rate')
        rate = field(ds, 'PRECTOT')*3600
    if not np.isfinite(rate).all() or np.any(rate < 0):
        raise ValueError(f'{path}: nonfinite or negative PRECTOT')
    return rate


def hourly_target(entry, archive, highres_root=None):
    middle = datetime.fromisoformat(entry['time'])
    if middle.minute != 30:
        raise ValueError(f'Archive hour {entry["time"]} is not an HH:30 midpoint')
    times = (middle-timedelta(minutes=30), middle, middle+timedelta(minutes=30))
    paths = [snapshot_path(entry, t, highres_root) for t in times]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        return None, missing
    rates = [read_rate(p, t) for p, t in zip(paths, times)]
    if any(r.shape != archive.shape for r in rates):
        raise ValueError(f'{entry["time"]}: HWT grid {rates[0].shape} != archive {archive.shape}')
    archived = np.asarray(archive.array(entry, 'truth')[1])
    if not np.allclose(rates[1], archived, rtol=1e-5, atol=1e-6):
        raise ValueError(f'{entry["time"]}: :30 snapshot differs from archived truth; pairing/grid mismatch')
    target = sum(w*r for w, r in zip(WEIGHTS, rates)).astype('float32')
    return target[None], []


def entries_for(archive, month=None):
    if month is not None:
        datetime.strptime(month, '%Y-%m')
    return [e for e in archive.index['entries'] if month is None or e['time'][:7] == month]


def target_valid(path, shape):
    try:
        value = np.load(path, mmap_mode='r')
        return value.shape == (1, *shape) and value.dtype == np.float32
    except (OSError, ValueError):
        return False


def prepare_hourly(cfg, month=None):
    d = cfg['data']
    archive = ArchiveV2(d['prepared'])
    out = Path(d['hourly_targets'])
    provenance = ensure_provenance(cfg, archive, create=True)
    written, skipped, missing = 0, 0, []
    for entry in entries_for(archive, month):
        dest = out/entry['id']/HOURLY_FILE
        if dest.exists():
            verify_target(cfg, archive, entry, provenance)
            skipped += 1
            continue
        try:
            sources = source_identity(entry, d.get('highres_root'))
        except FileNotFoundError:
            sources = None
        target, absent = hourly_target(entry, archive, d.get('highres_root'))
        if target is None:
            missing.append(dict(time=entry['time'], missing=absent))
            continue
        if sources != source_identity(entry, d.get('highres_root')):
            raise ValueError('HWT sources changed during preparation; retry with stable inputs')
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name+'.tmp.npy')
        np.save(tmp, target)
        identity = file_identity(tmp)
        identity['path'] = str(dest.resolve())
        write_json(dest.parent/TARGET_META, dict(contract_sha256=digest(provenance), sources=sources,
                                               target=identity, target_sha256=file_digest(tmp)))
        os.replace(tmp, dest)
        written += 1
    label = month or 'all'
    write_json(out/'_months'/f'{label}.json', dict(month=label, written=written, skipped=skipped, missing=missing,
                                                   archive_fingerprint=archive.index['fingerprint']))
    return dict(month=label, written=written, skipped=skipped, missing=len(missing))


def finalize_hourly(cfg):
    d = cfg['data']
    archive = ArchiveV2(d['prepared'])
    out = Path(d['hourly_targets'])
    provenance = ensure_provenance(cfg, archive)
    completed, missing, fingerprints = [], [], {}
    for entry in archive.index['entries']:
        if (out/entry['id']/HOURLY_FILE).exists():
            fingerprints[entry['id']] = verify_target(cfg, archive, entry, provenance, checksum=True)
            completed.append(entry['id'])
        else:
            missing.append(entry['time'])
    if not completed:
        raise ValueError(f'No hourly targets under {out}; run prepare-hourly first')
    completed_ids = set(completed)
    by_split = {s: sum(e['split'] == s and e['id'] in completed_ids for e in archive.index['entries'])
                for s in ('train', 'val', 'test')}
    index = dict(method=METHOD, weights=list(WEIGHTS), archive_fingerprint=archive.index['fingerprint'],
                 contract_sha256=digest(provenance), target_fingerprints=fingerprints,
                 source='hwt_30mn_slv_LCC.PRECTOT x 3600 (mm/h)', completed=completed,
                 missing_hours=missing, completed_by_split=by_split,
                 note='Hours without all three snapshots are excluded from training/evaluation, never filled.')
    index['fingerprint'] = digest(index)
    write_json(out/HOURLY_INDEX, index)
    return dict(completed=len(completed), missing=len(missing), by_split=by_split)


def months(cfg):
    archive = ArchiveV2(cfg['data']['prepared'])
    return sorted({e['time'][:7] for e in archive.index['entries']})
