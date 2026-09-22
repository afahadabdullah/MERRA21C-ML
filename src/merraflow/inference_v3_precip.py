"""Reproducible one-target ensembles with provenance and explicit rate semantics."""
from pathlib import Path
import hashlib
import json
import os
import numpy as np
import torch
import xarray as xr
from .config_v3_precip import validate_config
from .dataset_v3_precip import PrecipArchive
from .model_v3_precip import make_regression, PrecipEDM, sample_edm, sigma_schedule, heun_edm, core
from .model_v2 import regression_v2
from .noise_v2 import padded_noise_v2
from .inference import starts, blend_window
from .train import device_for, autocast
from .train_v3_precip import check_checkpoint, file_hash


class TileField:
    """Per-tile inputs and regression means for one frame, computed once and reused."""

    @torch.no_grad()
    def __init__(self, mean_model, archive, entry, cfg, device):
        p = cfg['patch']
        self.size, self.halo = p['size'], p['halo']
        self.width = self.size+2*self.halo
        h, w = archive.shape
        self.tiles = [(y, x) for y in starts(h, self.size, p['stride']) for x in starts(w, self.size, p['stride'])]
        self.batch = cfg['inference'].get('tile_batch', 8)
        self.device, self.precision = device, cfg['train']['precision']
        self.inputs, self.means = [], []
        for y, x in self.tiles:
            condition, context = archive.inputs(entry, y, x, p)
            batch = dict(target=torch.zeros((1, 1, self.width, self.width), device=device),
                         condition=condition[None].to(device), context=context[None].to(device))
            with autocast(device, self.precision):
                mean = regression_v2(mean_model, batch)
            self.inputs.append((condition, context))
            self.means.append(mean[0].detach().float().cpu())

    def chunks(self):
        for i in range(0, len(self.tiles), self.batch):
            index = range(i, min(i+self.batch, len(self.tiles)))
            yield (index, torch.stack([self.inputs[j][0] for j in index]).to(self.device),
                   torch.stack([self.inputs[j][1] for j in index]).to(self.device),
                   torch.stack([self.means[j] for j in index]).to(self.device))


def blend_means(field, shape):
    """Core-window blend of the per-tile regression means onto the domain."""
    h, w = shape
    size, halo = field.size, field.halo
    window = blend_window(size)
    total, weight = np.zeros((h, w), 'float32'), np.zeros((h, w), 'float32')
    for (y, x), mean in zip(field.tiles, field.means):
        total[y:y+size, x:x+size] += core(mean, halo, size)[0].numpy()*window
        weight[y:y+size, x:x+size] += window
    return total/weight


@torch.no_grad()
def synchronized_residual(diffusion, field, noise, cfg, steps=None):
    """One global EDM trajectory over the halo-padded domain.

    At every denoiser evaluation all overlapping tiles read the SAME current
    global state, their full-tile denoised estimates are blended with a smooth
    window, and one Heun step is taken globally. No tile runs its own
    trajectory, avoiding independent-sample stitching. Neural tile context can
    still disagree; this does not guarantee seam-free or untiled equivalence.
    Requires a denoiser trained with patch.loss_on_halo (halo outputs are used).
    """
    width = field.width
    window = torch.from_numpy(blend_window(width))
    weight = torch.zeros(noise.shape[-2:])
    for y, x in field.tiles:
        weight[y:y+width, x:x+width] += window
    if torch.any(weight <= 0):
        raise RuntimeError('Synchronized tiling left uncovered pixels')

    def denoise(state, sigma):
        result = torch.zeros_like(state)
        for index, condition, context, mean in field.chunks():
            x = torch.stack([state[0, field.tiles[j][0]:field.tiles[j][0]+width,
                                   field.tiles[j][1]:field.tiles[j][1]+width] for j in index])[:, None].to(field.device)
            with autocast(field.device, field.precision):
                value = diffusion(x, sigma.to(field.device).expand(len(x)), condition, context, mean)
            value = value.float().cpu()[:, 0]*window
            for k, j in enumerate(index):
                y, xx = field.tiles[j]
                result[0, y:y+width, xx:xx+width] += value[k]
        return result/weight

    sigmas = sigma_schedule(cfg, steps or cfg['inference']['steps'], torch.device('cpu'))
    return heun_edm(denoise, torch.from_numpy(noise), sigmas)[0].numpy()


@torch.no_grad()
def sample_frame(mean_model, diffusion, scale, archive, entry, cfg, device, seed, field=None):
    p = cfg['patch']
    h, w = archive.shape
    size, halo = p['size'], p['halo']
    field = field or TileField(mean_model, archive, entry, cfg, device)
    mean = blend_means(field, (h, w))
    baseline = archive.encode_rain(archive.array(entry, 'baseline')[1])
    if diffusion is None:
        return archive.decode_rain(baseline+mean), archive.decode_rain(baseline+mean)
    blend = cfg['inference']['blend']
    noise = padded_noise_v2(seed, (1, h, w), halo)
    if blend == 'synchronized':
        residual = synchronized_residual(diffusion, field, noise, cfg)[halo:halo+h, halo:halo+w]
        value = mean+scale*residual
    else:  # Legacy independent-tile ablations: 'owner' or 'weighted'.
        value, denominator = np.zeros((h, w), 'float32'), np.zeros((h, w), 'float32')
        window = blend_window(size)
        for (y, x), (condition, context), tile_mean in zip(field.tiles, field.inputs, field.means):
            z = torch.from_numpy(noise[:, y:y+size+2*halo, x:x+size+2*halo].copy()[None]).to(device)
            m = tile_mean[None].to(device)
            with autocast(device, cfg['train']['precision']):
                sample = m+scale*sample_edm(diffusion, z, condition[None].to(device), context[None].to(device), m, cfg)
            sample = core(sample, halo, size)[0, 0].float().cpu().numpy()
            region = np.s_[y:y+size, x:x+size]
            if blend == 'weighted':
                value[region] += window*sample
                denominator[region] += window
            else:
                take = window > denominator[region]
                value[region][take] = sample[take]
                denominator[region][take] = window[take]
        if blend == 'weighted':
            value /= denominator
    return archive.decode_rain(baseline+value), archive.decode_rain(baseline+mean)


def predict(cfg, checkpoint, split='val', limit=None):
    validate_config(cfg)
    archive = PrecipArchive(cfg)
    entries = archive.eligible(split)
    if limit is not None:
        if limit < 1:
            raise ValueError('limit must be positive')
        entries = entries[:limit]
    device = device_for(cfg['train']['device'])
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    check_checkpoint(saved, cfg, archive)
    mean = make_regression(archive.channels, cfg).to(device).eval()
    diffusion = None
    if saved['stage'] == 'regression':
        mean.load_state_dict(saved['ema'])
    else:
        mean.load_state_dict(saved['regression_ema'])
        diffusion = PrecipEDM(archive.channels, cfg).to(device).eval()
        diffusion.load_state_dict(saved['ema'])
    out = Path(cfg['inference']['output'])
    out.mkdir(parents=True, exist_ok=True)
    contract = dict(version='v3_precip', fingerprint=archive.index['fingerprint'], checkpoint_sha256=file_hash(checkpoint),
                    hourly_fingerprint=archive.hourly_fingerprint,
                    stage=saved['stage'], inference={k: v for k, v in cfg['inference'].items() if k != 'output'},
                    target_kind=archive.target_kind, history_hours=archive.lags, rain_scale_mm_h=archive.scale,
                    rain_codec=dict(dry_threshold_mm_h=archive.dry_threshold, dry_offset=archive.dry_offset,
                                    wet_threshold_z=archive.wet_threshold))
    signature = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    manifest_path = out/'manifest_v3_precip.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != contract:
        raise ValueError('Mixed checkpoints/settings: use a fresh prediction directory')
    manifest_path.write_text(json.dumps(contract, indent=2)+'\n')
    if archive.target_kind == 'midpoint_rate':
        cell_methods, alignment = 'time: point', 'HWT :30 rain-rate snapshot; NOT an hourly mean or accumulation'
    else:
        cell_methods = 'time: mean (trapezoid of HWT 30-min PRECTOT snapshots :00, :30, :00+1h)'
        alignment = 'Approximate hourly-mean rate of the HWT simulation (trapezoid rule); not observed rainfall'
    for entry in entries:
        tiles = TileField(mean, archive, entry, cfg, device)  # Inputs/means shared by all members.
        for member in range(cfg['inference']['members']):
            dest = out/f'{entry["id"]}_m{member:03d}_v3_precip.nc'
            if dest.exists():
                raise FileExistsError(f'{dest} already exists')
            timestamp_seed = int(hashlib.sha256(entry['time'].encode()).hexdigest()[:8], 16)
            seed = int(np.random.SeedSequence([cfg['inference']['seed'], timestamp_seed, member]).generate_state(1)[0])
            value, deterministic = sample_frame(mean, diffusion, saved['residual_scale'], archive, entry, cfg, device, seed, tiles)
            with xr.open_dataset(archive.root/'grid_v2.nc') as grid:
                ds = grid.load().copy()
            ds = ds.assign_coords(time=[np.datetime64(entry['time'])])
            for name, field in [('precip', value), ('regression_precip', deterministic)]:
                ds[name] = (('time', 'Ydim', 'Xdim'), field[None], dict(units='mm h-1', cell_methods=cell_methods, coordinates='lat lon'))
                if 'grid_mapping_variable' in ds.attrs:
                    ds[name].attrs['grid_mapping'] = ds.attrs['grid_mapping_variable']
            ds.attrs.update(version='v3_precip', signature=signature, split=split, ensemble_member=member, seed=seed,
                            checkpoint_sha256=contract['checkpoint_sha256'], target_kind=archive.target_kind, target_alignment=alignment,
                            conservation='none; coarse bias is allowed and audited')
            tmp = str(dest)+'.tmp'
            ds.to_netcdf(tmp, engine='h5netcdf', encoding={k: dict(zlib=True, complevel=2) for k in ('precip', 'regression_precip')})
            ds.close()
            os.replace(tmp, dest)
            print(f'Wrote {dest}', flush=True)
    return out
