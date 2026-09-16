"""One global flow state, with overlapping patch velocities combined at each RHS.

The outer halo is part of the evolved state. This path requires models trained
with full-patch flow supervision; legacy checkpoints use independent sampling.
"""
import numpy as np
import torch
from .inference import starts, blend_window
from .model_v2 import regression_v2
from .train import autocast


def integrate_global_v2(initial, rhs, steps):
    """Heun with synchronization before BOTH velocity evaluations."""
    if steps < 1:
        raise ValueError('ODE steps must be positive')
    state, dt = initial.copy(), 1/steps
    for i in range(steps):
        k1 = rhs(state, i*dt)
        k2 = rhs(state+dt*k1, (i+1)*dt)
        state += (dt/2)*(k1+k2)
        if not np.isfinite(state).all():
            raise FloatingPointError('Nonfinite synchronized flow state')
    return state


@torch.no_grad()
def synchronized_frame_v2(mean_model, flow, flow_scale, archive, entry, cfg, device, noise):
    patch = cfg['patch']
    size, halo = patch['size'], patch['halo']
    width = size+2*halo
    h, w = archive.shape
    tiles = [(y, x) for y in starts(h, size, patch['stride']) for x in starts(w, size, patch['stride'])]
    full_window, core_window = blend_window(width), blend_window(size)
    denominator = np.zeros(noise.shape[-2:], dtype=np.float32)
    core_denominator = np.zeros((h, w), dtype=np.float32)
    mean_field = np.zeros((5, h, w), dtype=np.float32)
    cache, cached_bytes = {}, 0
    cache_limit = int(cfg['inference'].get('tile_cache_mb', 512)*1024**2)

    def inputs(y, x):
        nonlocal cached_bytes
        if (y, x) in cache:
            return cache[y, x]
        local, broad = archive.inputs(entry, y, x, patch)
        condition, context = local[None].to(device), broad[None].to(device)
        with autocast(device, cfg['train']['precision']):
            mean = regression_v2(mean_model, dict(target=torch.zeros((1, 5, width, width), device=device),
                                                   condition=condition, context=context))
        value = (local, broad, mean.cpu())
        nbytes = sum(v.numel()*v.element_size() for v in value)
        # Retain a bounded fixed subset instead of repeatedly evicting every
        # tile during sequential sweeps when the full field exceeds the cache.
        if cached_bytes+nbytes <= cache_limit:
            cache[y, x] = value
            cached_bytes += nbytes
        return value

    for y, x in tiles:
        denominator[y:y+width, x:x+width] += full_window
        _, _, mean = inputs(y, x)
        mean_field[:, y:y+size, x:x+size] += mean[0, :, halo:halo+size, halo:halo+size].numpy()*core_window
        core_denominator[y:y+size, x:x+size] += core_window
    if np.any(denominator <= 0) or np.any(core_denominator <= 0):
        raise RuntimeError('Synchronized tiling left uncovered pixels')
    mean_field /= core_denominator

    def rhs(state, time):
        velocity = np.zeros_like(state)
        t = torch.tensor([time], device=device)
        for y, x in tiles:
            local, broad, mean = inputs(y, x)
            # Every tile reads the SAME current state. No local ODE is run.
            z = torch.from_numpy(state[:, y:y+width, x:x+width].copy())[None].to(device)
            with autocast(device, cfg['train']['precision']):
                value = flow(z, t, local[None].to(device), broad[None].to(device), mean.to(device)).float()
            velocity[:, y:y+width, x:x+width] += value[0].cpu().numpy()*full_window
        return velocity/denominator

    endpoint = integrate_global_v2(noise, rhs, cfg['inference']['steps'])
    endpoint = endpoint[:, halo:halo+h, halo:halo+w]
    scale = flow_scale.detach().cpu().numpy()[0]
    return mean_field+scale*endpoint, mean_field
