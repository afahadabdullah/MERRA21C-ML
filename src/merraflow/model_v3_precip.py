"""One-channel conditional EDM residual diffusion with deterministic Heun sampling.

Karras et al. (2022) preconditioning; regression/residual decomposition inspired
by CorrDiff. This is an experimental implementation, not a paper reproduction.
"""
import torch
from torch import nn
from .model_v2 import UNetV2, regression_v2


def make_regression(channels, cfg):
    return UNetV2(channels, **cfg['model'], target_channels=1)


class PrecipEDM(nn.Module):
    def __init__(self, channels, cfg):
        super().__init__()
        self.net = UNetV2(channels, **cfg['model'], target_channels=1, mean_condition=True)
        self.sigma_data = cfg['diffusion']['sigma_data']

    def forward(self, noisy, sigma, condition, context, mean):
        sigma = sigma.to(noisy).reshape(-1, 1, 1, 1)
        sd = self.sigma_data
        norm = (sigma.square()+sd**2).sqrt()
        cskip = sd**2/norm.square()
        cout = sigma*sd/norm
        # UNetV2's TimeEmbedding multiplies by 1000 internally.
        embedding_time = sigma.flatten().log()/4000
        value = self.net(noisy/norm, embedding_time, condition, context, mean).float()
        return cskip*noisy+cout*value


def core(x, halo, size):
    return x[..., halo:halo+size, halo:halo+size]


def weighted_loss(error, batch, patch, full=False):
    """Area-weighted mean over the core (default) or the whole halo-padded patch.

    ``full=True`` trains the denoiser on halo pixels too, which synchronized
    tiled sampling needs: every overlapping tile's full output is blended.
    """
    if full:
        area = batch['area_full'][:, None]
    else:
        error = core(error, patch['halo'], patch['size'])
        area = batch['area'][:, None]
    per_sample = (error*area).sum((1, 2, 3))/area.sum((1, 2, 3))
    # Do not self-normalize proposal weights: that biases a minibatch estimator.
    return (per_sample*batch['importance']).mean()


def objective(model, batch, cfg, regression=None, residual_scale=1., generator=None):
    if regression is None:
        return weighted_loss((regression_v2(model, batch)-batch['target']).square(), batch, cfg['patch'],
                             full=cfg['patch'].get('loss_on_halo', False))
    with torch.no_grad():
        mean = regression_v2(regression, batch)
    clean = (batch['target']-mean)/residual_scale
    ed = cfg['diffusion']
    sigma = (torch.randn((len(clean),), device=clean.device, generator=generator)*ed['p_std']+ed['p_mean']).exp()
    noise = torch.randn(clean.shape, device=clean.device, generator=generator)
    denoised = model(clean+sigma[:, None, None, None]*noise, sigma, batch['condition'], batch['context'], mean)
    weight = (sigma.square()+ed['sigma_data']**2)/(sigma*ed['sigma_data']).square()
    return weighted_loss((denoised-clean).square()*weight[:, None, None, None], batch, cfg['patch'],
                         full=cfg['patch'].get('loss_on_halo', False))


def sigma_schedule(cfg, steps, device):
    if steps < 2:
        raise ValueError('EDM requires at least two sampling steps')
    ed = cfg['diffusion']
    ramp = torch.linspace(0, 1, steps, device=device)
    sigmas = (ed['sigma_max']**(1/ed['rho'])+ramp*(ed['sigma_min']**(1/ed['rho'])-ed['sigma_max']**(1/ed['rho'])))**ed['rho']
    return torch.cat([sigmas, sigmas.new_zeros(1)])


def heun_edm(denoise, noise, sigmas):
    """Deterministic EDM Heun solver for any denoiser D(x, sigma) (tensor or tiled)."""
    x = noise.float()*sigmas[0]
    for i in range(len(sigmas)-1):
        sigma, next_sigma = sigmas[i], sigmas[i+1]
        derivative = (x-denoise(x, sigma))/sigma
        proposal = x+(next_sigma-sigma)*derivative
        if i < len(sigmas)-2:
            next_d = (proposal-denoise(proposal, next_sigma))/next_sigma
            x = x+(next_sigma-sigma)*(derivative+next_d)/2
        else:
            x = proposal  # Terminal Euler step avoids evaluating log(0).
    if not torch.isfinite(x).all():
        raise FloatingPointError('Nonfinite EDM trajectory')
    return x


@torch.no_grad()
def sample_edm(model, noise, condition, context, mean, cfg, steps=None):
    sigmas = sigma_schedule(cfg, steps or cfg['inference']['steps'], noise.device)
    return heun_edm(lambda x, sigma: model(x, sigma.expand(len(x)), condition, context, mean), noise, sigmas)
