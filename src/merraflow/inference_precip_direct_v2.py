"""Synchronized tiled full-field velocity integration, with no baseline add-back."""
from pathlib import Path
import os
import numpy as np
import torch
import xarray as xr
from .precip_direct_v2 import (DirectPrecipDataset, make_model, FrozenRegression,
                              decode_rain, check_checkpoint, VERSION)
from .dataset_v2 import crop_v2
from .inference import starts, blend_window
from .train import device_for, autocast
from .train_v2 import file_hash_v2


@torch.no_grad()
def sample_frame(model, conditioner, archive, entry, cfg, device, seed):
    p = cfg['patch']
    h, w = archive.shape
    size, halo, width = p['size'], p['halo'], p['size']+2*p['halo']
    tiles = [(y, x) for y in starts(h, size, p['stride']) for x in starts(w, size, p['stride'])]
    x = np.random.default_rng(seed).standard_normal((1, h+2*halo, w+2*halo)).astype('float32')
    window = blend_window(width)
    weight = np.zeros_like(x)
    means = {}
    for y, col in tiles:
        weight[:, y:y+width, col:col+width] += window
    if np.any(weight <= 0):
        raise ValueError('Uncovered tile pixels')
    def velocity(state, time):
        result = np.zeros_like(state)
        for y, col in tiles:
            condition, context = archive.inputs(entry, y, col, p)
            condition, context = condition[None].to(device), context[None].to(device)
            tensor = torch.from_numpy(state[:, y:y+width, col:col+width].copy()[None]).to(device)
            with autocast(device, cfg['train']['precision']):
                mean = None
                if conditioner is not None:
                    if (y, col) not in means:
                        coarse = crop_v2(archive.array(entry, 'baseline')[1:2], y, col, size, halo).copy()
                        means[(y, col)] = conditioner(dict(condition=condition, context=context,
                            coarse=torch.from_numpy(coarse[None]).to(device))).float().cpu()
                    mean = means[(y, col)].to(device)
                value = model(tensor, torch.tensor([time], device=device), condition, context, mean)
            result[:, y:y+width, col:col+width] += value[0].float().cpu().numpy()*window
        return result/weight
    steps = cfg['inference']['steps']
    for i in range(steps):
        first = velocity(x, i/steps)
        second = velocity(x+first/steps, (i+1)/steps)
        x += (first+second)/(2*steps)
        if not np.isfinite(x).all():
            raise FloatingPointError('Nonfinite full-field trajectory')
    # This is the whole generated encoded field; neither coarse nor mean is added.
    return decode_rain(x[0, halo:halo+h, halo:halo+w], archive.stats['precip_log_scale'])


def predict(cfg, checkpoint, split='val', limit=1):
    archive = DirectPrecipDataset(cfg, split, 1, cfg['train']['seed']).archive
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    check_checkpoint(saved, cfg, archive)
    device = device_for(cfg['train']['device'])
    model = make_model(archive.index['condition_channels'], cfg).to(device).eval()
    model.load_state_dict(saved['ema'])
    bundle = saved.get('regression_condition')
    conditioner = FrozenRegression(bundle, archive.index['condition_channels'], archive.stats).to(device) if bundle else None
    out = Path(cfg['inference']['output'])
    out.mkdir(parents=True, exist_ok=True)
    fingerprint = file_hash_v2(checkpoint)
    entries = [e for e in archive.index['entries'] if e['split'] == split][:limit]
    for entry in entries:
        for member in range(cfg['inference']['members']):
            dest = out/f'{entry["id"]}_m{member:03d}_direct_v2.nc'
            if dest.exists():
                raise FileExistsError(f'Use a fresh inference output: {dest}')
            seed = int(np.random.SeedSequence([cfg['inference']['seed'],
                int(entry['id'].replace('_', '')), member]).generate_state(1)[0])
            value = sample_frame(model, conditioner, archive, entry, cfg, device, seed)
            with xr.open_dataset(archive.root/'grid_v2.nc') as grid:
                ds = grid.load().copy()
            ds = ds.assign_coords(time=[np.datetime64(entry['time'])])
            ds['precip'] = (('time', 'Ydim', 'Xdim'), value[None],
                           dict(units='mm h-1', cell_methods='time: point', coordinates='lat lon'))
            ds.attrs.update(version=VERSION, target='full precipitation', target_kind='midpoint_rate',
                            checkpoint_sha256=fingerprint, checkpoint_epoch=saved['epoch']+1,
                            frozen_regression_input=int(bundle is not None), ensemble_member=member,
                            seed=seed, sampler='synchronized velocities', split=split)
            tmp = str(dest)+'.tmp'
            ds.to_netcdf(tmp, engine='h5netcdf', encoding={'precip': dict(zlib=True, dtype='float32')})
            ds.close()
            os.replace(tmp, dest)
            print(f'Wrote {dest}', flush=True)
    return out
