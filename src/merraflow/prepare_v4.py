"""Add HWT 2 m specific humidity targets without rewriting v2 or v3 data."""
from datetime import datetime
from pathlib import Path
import json
import os
import re
import numpy as np
import xarray as xr
from .config import write_json
from .dataset_v2 import ArchiveV2
from .prepare_v2 import field, assert_time
from .prepare_v3_precip import digest, file_identity, file_digest

INDEX = 'index_q2m_v4.json'


def contract(cfg, archive):
    return dict(format='v4_q2m', archive_fingerprint=archive.index['fingerprint'],
                shape=list(archive.shape), variable=cfg['data']['humidity_target_variable'],
                units='kg kg-1', temporal_support='midpoint_snapshot')


def read_q2m(cfg, archive, entry):
    name = cfg['data']['humidity_target_variable']
    with xr.open_dataset(entry['hr']) as ds:
        assert_time(ds, datetime.fromisoformat(entry['time']), entry['hr'])
        if name not in ds:
            raise KeyError(f'{entry["hr"]}: missing {name}; available variables: {list(ds.data_vars)}')
        units = re.sub(r'[\s*^()]', '', str(ds[name].attrs.get('units', '')).lower())
        if units in ('kgkg-1', 'kg/kg', '1'):
            factor = 1.
        elif units in ('gkg-1', 'g/kg'):
            factor = .001
        else:
            raise ValueError(f'{entry["hr"]}: unsupported specific humidity units {units!r}')
        for var, expected in [('lats', archive.static['lat']), ('lons', archive.static['lon'])]:
            value = field(ds, var)
            if var == 'lons':
                value = ((value+180) % 360)-180
            if value.shape != expected.shape or not np.allclose(value, expected, atol=2e-5, rtol=0):
                raise ValueError(f'{entry["hr"]}: humidity grid mismatch')
        value = field(ds, name)*factor
        if value.shape != archive.shape or not np.isfinite(value).all() or value.min() < 0 or value.max() > 1:
            raise ValueError(f'{entry["hr"]}: invalid specific humidity')
    # QV2M was cached as a raw dynamic predictor by v2; ensure kg/kg and that
    # the archived input is the same field, never a mislabeled g/kg array.
    with xr.open_dataset(entry['lr']) as ds:
        assert_time(ds, datetime.fromisoformat(entry['time']), entry['lr'])
        units = re.sub(r'[\s*^()]', '', str(ds['QV2M'].attrs.get('units', '')).lower())
        if units not in ('kgkg-1', 'kg/kg', '1'):
            raise ValueError(f'{entry["lr"]}: archived QV2M must be kg/kg, got {units!r}')
        coarse = field(ds, 'QV2M')
        channel = archive.stats['predictors'].index('QV2M')
        cached = archive.array(entry, 'condition')[channel]
        if (coarse.shape != archive.shape or not np.isfinite(coarse).all()
                or coarse.min() < 0 or coarse.max() > 1
                or not np.allclose(coarse, cached, rtol=1e-6, atol=1e-9)):
            raise ValueError(f'{entry["lr"]}: invalid or mismatched archived QV2M')
    return value[None].astype('float32')


def verify(cfg, archive, entry, checksum=False):
    folder = Path(cfg['data']['humidity_targets'])/entry['id']
    path = folder/'q2m_v4.npy'
    meta = json.loads((folder/'provenance.json').read_text())
    if (meta['contract'] != digest(contract(cfg, archive)) or meta['time'] != entry['time']
            or meta['source'] != file_identity(Path(entry['hr'])) or meta['target'] != file_identity(path)):
        raise ValueError(f'{folder}: humidity provenance changed; use a fresh cache')
    arr = np.load(path, mmap_mode='r')
    if arr.shape != (1, *archive.shape) or arr.dtype != np.float32:
        raise ValueError(f'{path}: invalid humidity cache shape/type')
    if checksum and file_digest(path) != meta['sha256']:
        raise ValueError(f'{path}: humidity checksum mismatch')
    return digest(meta)


def prepare_humidity(cfg, month=None):
    archive = ArchiveV2(cfg['data']['prepared'])
    if 'QV2M' not in archive.stats['predictors']:
        raise ValueError('Existing v2 archive must contain QV2M as a predictor')
    entries = [e for e in archive.index['entries'] if month is None or e['time'].startswith(month)]
    if not entries:
        raise ValueError(f'No archive entries for {month}')
    written = skipped = 0
    for entry in entries:
        folder = Path(cfg['data']['humidity_targets'])/entry['id']
        path = folder/'q2m_v4.npy'
        if path.exists() and (folder/'provenance.json').exists():
            verify(cfg, archive, entry, checksum=True)
            skipped += 1
            continue
        source = file_identity(Path(entry['hr']))
        value = read_q2m(cfg, archive, entry)
        if file_identity(Path(entry['hr'])) != source:
            raise ValueError('Humidity source changed while reading')
        folder.mkdir(parents=True, exist_ok=True)
        tmp = folder/f'q2m.{os.getpid()}.tmp'
        with open(tmp, 'wb') as stream:
            np.save(stream, value)
        os.replace(tmp, path)
        write_json(folder/'provenance.json', dict(contract=digest(contract(cfg, archive)), time=entry['time'],
                    source=source, target=file_identity(path), sha256=file_digest(path)))
        written += 1
        print(f'Humidity {entry["id"]}', flush=True)
    return dict(written=written, skipped=skipped)


def finalize_humidity(cfg):
    archive = ArchiveV2(cfg['data']['prepared'])
    records = {e['id']: verify(cfg, archive, e, checksum=True) for e in archive.index['entries']}
    payload = dict(contract=contract(cfg, archive), records=records)
    payload['fingerprint'] = digest(payload)
    write_json(Path(cfg['data']['humidity_targets'])/INDEX, payload)
    return dict(hours=len(records), fingerprint=payload['fingerprint'])


def load_index(cfg, archive):
    path = Path(cfg['data']['humidity_targets'])/INDEX
    if not path.exists():
        raise FileNotFoundError(f'{path}: run cli_v4 prepare-humidity and finalize-humidity')
    index = json.loads(path.read_text())
    if (index['contract'] != contract(cfg, archive)
            or index['fingerprint'] != digest({k:v for k,v in index.items() if k != 'fingerprint'})):
        raise ValueError('Humidity index provenance mismatch')
    for entry in archive.index['entries']:
        if index['records'].get(entry['id']) != verify(cfg, archive, entry):
            raise ValueError(f'Humidity changed after finalization: {entry["id"]}')
    return index
