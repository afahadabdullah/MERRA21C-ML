"""V2 ensemble generation; no rainfall budget projection or dry threshold."""
from pathlib import Path
import json
import os
import numpy as np
import torch
import xarray as xr
from .config_v2 import validate_config_v2
from .dataset import crop
from .dataset_v2 import ArchiveV2
from .inference import starts, blend_window
from .model_v2 import UNetV2, regression_v2, integrate_v2
from .physics_v2 import TARGETS_V2, UNITS_V2, transform_v2, inverse_v2
from .physics import budget_error
from .train import device_for, autocast
from .train_v2 import check_checkpoint_v2, file_hash_v2


def load_models_v2(cfg, checkpoint, archive, device):
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=True)
    check_checkpoint_v2(ckpt, archive, cfg)
    nc = archive.index['condition_channels']
    mean = UNetV2(nc, **cfg['model']).to(device).eval()
    flow, scale = None, None
    if ckpt['stage'] == 'regression':
        mean.load_state_dict(ckpt['ema'])
    else:
        mean.load_state_dict(ckpt['regression_ema'])
        flow = UNetV2(nc, **cfg['model'], mean_condition=True).to(device).eval()
        flow.load_state_dict(ckpt['ema'])
        scale = ckpt['flow_scale'].to(device)
    return mean, flow, scale, ckpt


@torch.no_grad()
def sample_frame_v2(mean_model, flow, flow_scale, archive, entry, cfg, device, seed):
    p, (h, w) = cfg['patch'], archive.shape
    size, halo = p['size'], p['halo']
    noise = np.random.default_rng(seed).standard_normal((5, h, w), dtype=np.float32)
    accum, mean_accum = np.zeros_like(noise), np.zeros_like(noise)
    denom = np.zeros((h, w), dtype='float32')
    window = blend_window(size)
    mode = cfg['inference'].get('blend', 'weighted')
    if mode not in ('weighted', 'owner'):
        raise ValueError('V2 blend must be weighted or owner')
    for y in starts(h, size, p['stride']):
        for x in starts(w, size, p['stride']):
            local, broad = archive.inputs(entry, y, x, p)
            z = torch.from_numpy(crop(noise, y, x, size, halo)[None]).to(device)
            b = dict(target=z, condition=local[None].to(device), context=broad[None].to(device))
            with autocast(device, cfg['train']['precision']):
                mean = regression_v2(mean_model, b)
                prediction = mean if flow is None else mean+flow_scale*integrate_v2(flow, z, b['condition'], b['context'], mean, cfg['inference']['steps'])
            core = prediction[0, :, halo:halo+size, halo:halo+size].float().cpu().numpy()
            mc = mean[0, :, halo:halo+size, halo:halo+size].float().cpu().numpy()
            region = np.s_[y:y+size, x:x+size]
            if mode == 'weighted':
                accum[:, region[0], region[1]] += core*window
                mean_accum[:, region[0], region[1]] += mc*window
                denom[region] += window
            else:
                # Keep one coherent local sample at each pixel; may expose seams.
                take = window > denom[region]
                accum[:, region[0], region[1]][:, take] = core[:, take]
                mean_accum[:, region[0], region[1]][:, take] = mc[:, take]
                denom[region][take] = window[take]
    if np.any(denom <= 0):
        raise RuntimeError('V2 tiling left uncovered pixels')
    if mode == 'weighted':
        accum /= denom
        mean_accum /= denom
    baseline = transform_v2(archive.array(entry, 'baseline'), archive.stats['precip_log_scale'])
    result = inverse_v2(baseline+accum*archive.rs+archive.rm, archive.stats['precip_log_scale'])
    deterministic = inverse_v2(baseline+mean_accum*archive.rs+archive.rm, archive.stats['precip_log_scale'])
    audit = budget_error(result[1], archive.array(entry, 'native_reference')[0], archive.static['area'], archive.static['groups'])
    return result, deterministic, audit


def predict_v2(cfg, checkpoint, split='val', limit=None, timestamp=None):
    validate_config_v2(cfg)
    archive = ArchiveV2(cfg['data']['prepared'])
    device = device_for(cfg['train']['device'])
    mean, flow, scale, ckpt = load_models_v2(cfg, checkpoint, archive, device)
    entries = [e for e in archive.index['entries'] if e['split'] == split and
               (timestamp is None or timestamp in (e['id'], e['time']))]
    if limit is not None:
        if limit < 1:
            raise ValueError('limit must be positive')
        entries = entries[:limit]
    if not entries:
        raise ValueError('No matching v2 entries')
    out = Path(cfg['inference']['output'])
    out.mkdir(parents=True, exist_ok=True)
    digest = file_hash_v2(checkpoint)
    for entry in entries:
        for member in range(cfg['inference']['members']):
            dest = out/f'{entry["id"]}_m{member:03d}_v2.nc'
            if dest.exists():
                raise FileExistsError(f'{dest} exists; use a fresh v2 prediction directory')
            seed = int(np.random.SeedSequence([cfg['inference']['seed'], int(entry['id'].replace('_', '')), member]).generate_state(1)[0])
            result, deterministic, audit = sample_frame_v2(mean, flow, scale, archive, entry, cfg, device, seed)
            with xr.open_dataset(archive.root/'grid_v2.nc') as source:
                ds = source.load().copy()
            ds = ds.assign_coords(time=[np.datetime64(entry['time'])])
            for prefix, values in [('', result), ('regression_', deterministic)]:
                for i, (name, unit) in enumerate(zip(TARGETS_V2, UNITS_V2)):
                    ds[prefix+name] = (('time', 'Ydim', 'Xdim'), values[i][None],
                                       {'units': unit, 'coordinates': 'lat lon', 'cell_methods': 'time: point'})
                    if 'grid_mapping_variable' in ds.attrs:
                        ds[prefix+name].attrs['grid_mapping'] = ds.attrs['grid_mapping_variable']
                ds[prefix+'wind10m'] = (('time', 'Ydim', 'Xdim'), np.hypot(values[3], values[4])[None], {'units': 'm s-1'})
            ds['lr_time_bounds_v2'] = (('time', 'bounds'), np.array([[np.datetime64(entry['time'])-np.timedelta64(30, 'm'),
                                                                    np.datetime64(entry['time'])+np.timedelta64(30, 'm')]]))
            ds.attrs.update(version='v2', stage=ckpt['stage'], checkpoint=str(Path(checkpoint).resolve()),
                            checkpoint_epoch=ckpt['epoch']+1, checkpoint_sha256=digest,
                            regression_sha256=ckpt.get('regression_sha256') or digest,
                            dataset_fingerprint=archive.index['fingerprint'], ensemble_member=member, seed=seed,
                            split=split, ode_steps=cfg['inference']['steps'], blend=cfg['inference'].get('blend', 'weighted'),
                            target_alignment='HR midpoint snapshot approximates coarse hourly mean',
                            conservation='none; audit only', budget_audit_v2=json.dumps(audit))
            encoding = {name: {'zlib': True, 'complevel': 2, 'dtype': 'float32'} for name in ds.data_vars if name != 'lr_time_bounds_v2' and ds[name].ndim == 3}
            ds.time.encoding.update(units='minutes since 1970-01-01', calendar='proleptic_gregorian')
            ds.lr_time_bounds_v2.encoding.update(units='minutes since 1970-01-01', calendar='proleptic_gregorian')
            tmp = str(dest)+'.tmp'
            ds.to_netcdf(tmp, engine='h5netcdf', encoding=encoding)
            ds.close()
            os.replace(tmp, dest)
            print(f'Wrote {dest}', flush=True)
    return out
