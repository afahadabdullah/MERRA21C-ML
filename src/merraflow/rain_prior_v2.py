"""Stationary correlated Gaussian rainfall prior, shared by training and sampling.

Correlation is introduced in the initial noise, never by filtering predictions.
Unit marginal variance keeps the calibrated residual scale interpretable.
"""
import numpy as np
import torch
from torch.nn import functional as F
from scipy.ndimage import convolve1d


def rain_noise_sigma_v2(cfg):
    return float(cfg.get('representation', {}).get('rain_noise_sigma_pixels', 0.))


def rain_noise_kernel_v2(sigma):
    radius = int(np.ceil(3*sigma))
    offsets = np.arange(-radius, radius+1, dtype=np.float32)
    kernel = np.exp(-.5*(offsets/sigma)**2)
    return kernel/np.sqrt(np.sum(kernel**2))


def training_noise_v2(shape, device, generator, cfg):
    noise = torch.randn(shape, device=device, generator=generator)
    sigma = rain_noise_sigma_v2(cfg)
    if not sigma:
        return noise
    kernel = torch.from_numpy(rain_noise_kernel_v2(sigma)).to(device)
    radius = len(kernel)//2
    b, _, h, w = shape
    rain = torch.randn((b, 1, h+2*radius, w+2*radius), device=device, generator=generator)
    # No zero/replicate padding: extra random support gives stationary covariance.
    # Keep prior construction independent of model mixed precision.
    with torch.autocast(device_type=device.type, enabled=False):
        rain = F.conv2d(rain, kernel[None, None, None, :])
        rain = F.conv2d(rain, kernel[None, None, :, None])
    noise[:, 1:2] = rain
    return noise


def inference_rain_noise_v2(noise, seed, cfg):
    sigma = rain_noise_sigma_v2(cfg)
    if not sigma:
        return noise
    kernel = rain_noise_kernel_v2(sigma)
    radius = len(kernel)//2
    h, w = noise.shape[-2:]
    rng = np.random.default_rng(np.random.SeedSequence([seed, 72419]))
    rain = rng.standard_normal((h+2*radius, w+2*radius), dtype=np.float32)
    rain = convolve1d(convolve1d(rain, kernel, axis=0), kernel, axis=1)
    # Discard all filter-boundary values, including beyond the evolved halo.
    noise[1] = rain[radius:radius+h, radius:radius+w]
    return noise
