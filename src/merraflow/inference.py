from pathlib import Path
import hashlib
import json
import os
import numpy as np
import torch
import xarray as xr
from . import TARGETS, UNITS
from .dataset import Archive, crop
from .model import VelocityUNet, integrate
from .physics import transform_target, inverse_target, project_precip, budget_error
from .train import device_for, autocast


def starts(length, size, stride):
    if length < size or not 0 < stride <= size:
        raise ValueError('Domain must contain the patch and stride must avoid gaps')
    values = list(range(0, length-size+1, stride))
    if values[-1] != length-size:
        values.append(length-size)
    return values


def blend_window(size):
    # Positive edges cover domain boundaries; overlap receives smoothly varying weights.
    w = np.hanning(size+2)[1:-1]
    return np.maximum(np.outer(w, w), 1e-4).astype('float32')


@torch.no_grad()
def sample_frame(model, archive, entry, cfg, device, seed):
    p = cfg['patch']
    size, halo = p['size'], p['halo']
    h, w = archive.shape
    rng = np.random.default_rng(seed)
    # A common noise field for all overlapping patches of this ensemble member.
    noise = rng.standard_normal((4, h, w), dtype=np.float32)
    accum, denom = np.zeros_like(noise), np.zeros((h, w), dtype='float32')
    window = blend_window(size)
    model.eval()
    for y in starts(h, size, p['stride']):
        for x in starts(w, size, p['stride']):
            c = torch.from_numpy(archive.condition(entry, y, x, size, halo)[None]).to(device)
            z = torch.from_numpy(crop(noise, y, x, size, halo)[None]).to(device)
            with autocast(device, cfg['train']['precision']):
                sample = integrate(model, z, c, cfg['inference']['steps'])
            core = sample[0, :, halo:halo+size, halo:halo+size].float().cpu().numpy()
            accum[:, y:y+size, x:x+size] += core*window
            denom[y:y+size, x:x+size] += window
    if np.any(denom == 0):
        raise RuntimeError('Inference tiling left uncovered pixels')
    residual = (accum/denom)*archive.rs+archive.rm
    kw = (archive.stats['precip_log_scale'], archive.stats['wind_log_scale'])
    base = archive.array(entry, 'baseline')
    raw = inverse_target(transform_target(base, *kw)+residual, *kw)
    if not np.isfinite(raw).all():
        raise FloatingPointError('Nonfinite decoded sample')
    result = raw.copy()
    result[1] = project_precip(raw[1], base[1], archive.static['area'], archive.static['groups'], cfg['inference']['dry_threshold'])
    audit = {'before': budget_error(raw[1], base[1], archive.static['area'], archive.static['groups']),
             'after': budget_error(result[1], base[1], archive.static['area'], archive.static['groups']),
             'projection_mae_mm_h': float(np.mean(np.abs(result[1]-raw[1])))}
    if audit['after']['max_relative_wet'] > 2e-6 or audit['after']['dry_group_leakage_mm_m2_per_hour'] > 0:
        raise RuntimeError(f'Conservation check failed: {audit}')
    return result, raw[1], audit


def predict(cfg, checkpoint, split='test', limit=None, timestamp=None):
    archive = Archive(cfg['data']['prepared'])
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=True)
    digest = hashlib.sha256()
    with open(checkpoint, 'rb') as source:
        for block in iter(lambda: source.read(8*1024*1024), b''):
            digest.update(block)
    checkpoint_hash = digest.hexdigest()
    if ckpt['fingerprint'] != archive.index['fingerprint'] or ckpt['stats'] != archive.stats:
        raise ValueError('Checkpoint and prepared dataset/statistics do not match')
    device = device_for(cfg['train']['device'])
    model = VelocityUNet(archive.index['condition_channels'], **ckpt['config']['model']).to(device)
    model.load_state_dict(ckpt['ema'])
    entries = [e for e in archive.index['entries'] if e['split'] == split and (timestamp is None or timestamp in (e['time'], e['id']))]
    if limit is not None:
        if limit < 1:
            raise ValueError('limit must be positive')
        entries = entries[:limit]
    if not entries or cfg['inference']['members'] < 1:
        raise ValueError('No matching entries or invalid ensemble size')
    out = Path(cfg['inference']['output'])
    out.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        for member in range(cfg['inference']['members']):
            # Stable across limits/order, reproducible for a timestamp and member.
            seed = int(np.random.SeedSequence([cfg['inference']['seed'], int(entry['id'].replace('_', '')), member]).generate_state(1)[0])
            dest = out/f'{entry["id"]}_m{member:03d}.nc'
            if dest.exists():
                raise FileExistsError(f'{dest} exists; choose a new output directory')
            result, raw_pr, audit = sample_frame(model, archive, entry, cfg, device, seed)
            with xr.open_dataset(archive.root/'grid.nc') as source:
                ds = source.load().copy()
            ds = ds.expand_dims(time=[np.datetime64(entry['time'])])
            # Static fields stay 2D, scalar grid mapping stays scalar.
            for k in list(ds.data_vars):
                ds[k] = ds[k].isel(time=0, drop=True)
            for i, (name, unit) in enumerate(zip(TARGETS, UNITS)):
                ds[name] = (('time', 'Ydim', 'Xdim'), result[i][None], {'units': unit, 'coordinates': 'lat lon',
                            'cell_methods': 'time: point'})
                if 'grid_mapping_variable' in ds.attrs:
                    ds[name].attrs['grid_mapping'] = ds.attrs['grid_mapping_variable']
            ds['precip_unconstrained'] = (('time', 'Ydim', 'Xdim'), raw_pr[None], {'units': 'mm h-1', 'long_name': 'Nonnegative generated precipitation before dry-threshold and budget projection'})
            # These describe the LR conditioning window, not the HR snapshot target.
            ds['lr_time_bounds'] = (('time', 'bounds'), np.array([[np.datetime64(entry['time'])-np.timedelta64(30, 'm'), np.datetime64(entry['time'])+np.timedelta64(30, 'm')]]),
                                    {'long_name': 'Hourly averaging window of coarse conditioning fields'})
            ds.attrs.update({'ensemble_member': member, 'seed': seed, 'checkpoint': str(Path(checkpoint).resolve()),
                             'checkpoint_epoch': ckpt['epoch'], 'checkpoint_sha256': checkpoint_hash,
                             'dataset_fingerprint': archive.index['fingerprint'],
                             'precip_source': archive.index['data_config']['precip_source'],
                             'target_alignment': 'Matched :30 HR snapshot approximates LR hourly mean; precipitation budget projection applied',
                             'conservation': archive.index['conservation'], 'conservation_audit': json.dumps(audit),
                             'split': split, 'patch_size': cfg['patch']['size'], 'patch_halo': cfg['patch']['halo'],
                             'patch_stride': cfg['patch']['stride'], 'ode_steps': cfg['inference']['steps'],
                             'dry_threshold_mm_h': cfg['inference']['dry_threshold']})
            ds.time.encoding.update(units='minutes since 1970-01-01', calendar='proleptic_gregorian')
            ds.lr_time_bounds.encoding.update(units='minutes since 1970-01-01', calendar='proleptic_gregorian')
            encoding = {name: {'zlib': True, 'complevel': 2, 'dtype': 'float32'} for name in (*TARGETS, 'precip_unconstrained')}
            tmp = str(dest)+'.tmp'
            ds.to_netcdf(tmp, engine='h5netcdf', encoding=encoding)
            ds.close()
            os.replace(tmp, dest)
            print(f'Wrote {dest}; max wet budget error {audit["after"]["max_relative_wet"]:.3g}', flush=True)
    return out
