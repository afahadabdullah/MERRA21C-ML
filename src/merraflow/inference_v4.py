"""Synchronized full-domain six-field flow inference with hybrid decoding."""
from pathlib import Path
import os
import numpy as np
import torch
import xarray as xr
from .v4 import ArchiveV4, TARGETS, UNITS, make_model, FrozenRegression, check_checkpoint, VERSION
from .dataset_v2 import crop_v2
from .inference import starts, blend_window
from .train import device_for, autocast
from .train_v2 import file_hash_v2


@torch.no_grad()
def sample_frame(model, conditioner, archive, entry, cfg, device, seed):
    p = cfg['patch']
    h, w = archive.shape
    size, halo, width = p['size'], p['halo'], p['size']+2*p['halo']
    tiles = [(y, col) for y in starts(h, size, p['stride']) for col in starts(w, size, p['stride'])]
    x = np.random.default_rng(seed).standard_normal((len(TARGETS), h+2*halo, w+2*halo)).astype('float32')
    window = blend_window(width)
    weight = np.zeros((1, *x.shape[-2:]), dtype='float32')
    means = {}
    coarse_field = archive.coarse(entry)
    for y, col in tiles:
        weight[:, y:y+width, col:col+width] += window
    if np.any(weight <= 0):
        raise ValueError('Uncovered tile pixels')
    def velocity(state, time):
        result = np.zeros_like(state)
        for y, col in tiles:
            inputs = archive.inputs_with_original(entry, y, col, p)
            b = {k:v[None].to(device) for k,v in inputs.items()}
            tensor = torch.from_numpy(state[:, y:y+width, col:col+width].copy()[None]).to(device)
            with autocast(device, cfg['train']['precision']):
                if (y, col) not in means:
                    coarse = crop_v2(coarse_field, y, col, size, halo).copy()
                    b['coarse'] = torch.from_numpy(coarse[None]).to(device)
                    means[(y, col)] = conditioner(b).float().cpu()
                mean = means[(y, col)].to(device)
                value = model(tensor, torch.tensor([time], device=device), b['condition'], b['context'], mean)
            result[:, y:y+width, col:col+width] += value[0].float().cpu().numpy()*window
        return result/weight
    steps = cfg['inference']['steps']
    for i in range(steps):
        first = velocity(x, i/steps)
        second = velocity(x+first/steps, (i+1)/steps)
        x += (first+second)/(2*steps)
        if not np.isfinite(x).all():
            raise FloatingPointError('Nonfinite full-field trajectory')
    mean = np.zeros_like(x)
    for (y, col), value in means.items():
        mean[:, y:y+width, col:col+width] += value[0].numpy()*window
    mean /= weight
    core = x[:, halo:halo+h, halo:halo+w]
    mean = mean[:, halo:halo+h, halo:halo+w]
    # Decode on CPU to avoid allocating full-domain six-channel tensors on GPU.
    rs = conditioner.rs[0].cpu().numpy()
    rm = conditioner.rm[0].cpu().numpy()
    scale = conditioner.flow_scale[0].cpu().numpy()
    value = (core*scale+mean)*rs+rm+coarse_field
    z = np.maximum(core[1], 0)
    value[1] = archive.scale*z*(z+2)  # Rain never receives a baseline add-back.
    value[5] = np.clip(value[5], 0, 1)
    if not np.isfinite(value).all():
        raise FloatingPointError('Nonfinite physical output')
    return value


def predict(cfg, checkpoint, split='test', limit=1, timestamp=None):
    archive = ArchiveV4(cfg)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    check_checkpoint(saved, cfg, archive)
    device = device_for(cfg['train']['device'])
    model = make_model(archive.channels, cfg).to(device).eval()
    model.load_state_dict(saved['ema'])
    conditioner = FrozenRegression(saved['regression_condition'], archive.index['condition_channels'],
        archive.stats, archive.scale, cfg['data']['humidity_scale_kg_kg']).to(device)
    conditioner.flow_scale.copy_(saved['flow_scale'].to(device))
    out = Path(cfg['inference']['output'])
    out.mkdir(parents=True, exist_ok=True)
    fingerprint = file_hash_v2(checkpoint)
    entries = archive.eligible(split)
    if timestamp:
        entries = [e for e in entries if np.datetime64(e['time']) == np.datetime64(timestamp)]
        if not entries:
            raise ValueError('Requested time has no complete targets/history in this split')
    for entry in entries[:limit]:
        for member in range(cfg['inference']['members']):
            dest = out/f'{entry["id"]}_m{member:03d}_v4.nc'
            if dest.exists():
                raise FileExistsError(f'Use a fresh inference output: {dest}')
            seed = int(np.random.SeedSequence([cfg['inference']['seed'],
                int(entry['id'].replace('_', '')), member]).generate_state(1)[0])
            value = sample_frame(model, conditioner, archive, entry, cfg, device, seed)
            with xr.open_dataset(archive.root/'grid_v2.nc') as grid:
                ds = grid.load().copy()
            time = np.datetime64(entry['time'])
            ds = ds.assign_coords(time=[time])
            ds['rainfall_time_bounds'] = (('time', 'bounds'), [[time-np.timedelta64(30,'m'), time+np.timedelta64(30,'m')]])
            for c, (name, unit) in enumerate(zip(TARGETS, UNITS)):
                attrs = dict(units=unit, coordinates='lat lon', cell_methods='time: mean' if c == 1 else 'time: point')
                if c == 1:
                    attrs.update(temporal_support='approximate trapezoidal hourly mean', time_bounds='rainfall_time_bounds')
                ds[name] = (('time', 'Ydim', 'Xdim'), value[c][None], attrs)
            ds.attrs.update(version=VERSION, checkpoint_sha256=fingerprint, checkpoint_epoch=saved['epoch']+1,
                hourly_fingerprint=archive.hourly_fingerprint, humidity_fingerprint=archive.humidity_fingerprint,
                targets='direct sqrt rainfall; residual t2m ps u10m v10m q2m', ensemble_member=member,
                seed=seed, sampler='synchronized tiled Heun velocities', split=split)
            tmp = str(dest)+'.tmp'
            ds.to_netcdf(tmp, engine='h5netcdf', encoding={n:dict(zlib=True, dtype='float32') for n in TARGETS})
            ds.close()
            os.replace(tmp, dest)
            print(f'Wrote {dest}', flush=True)
    return out
