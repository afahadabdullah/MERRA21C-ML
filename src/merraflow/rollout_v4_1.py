"""v4.1 sample-based fine-tune: score the model's own members, not its velocities.

Why: the flow-matching loss (and any linear reweighting of its velocity error,
e.g. the gradient term) has the conditional-mean velocity as its optimum, so it
cannot change *how often* the model draws sharp fronts. Diagnostics showed that
v4.1 can draw crisp fronts for some noise draws but usually smears them. Scores
on generated members change the sample distribution directly.

Each training step (on ``patches`` patches of the batch, every ``interval`` steps):

1. Draw ``members`` independent members per patch from pure noise (the truth never
   enters the trajectory) with a ``steps``-step Heun ODE. Only the last
   ``grad_steps`` steps are differentiated (truncated backprop, as in DRaFT-K,
   Clark et al. 2024); fronts and fine detail form there.
2. Score members against the target in flow space (states: normalized residuals;
   rain: sqrt1p z), per channel, with
   * afCRPS: "almost fair" CRPS, alpha*fair + (1-alpha)*standard (AIFS-CRPS,
     Lang et al. 2024). Proper; rewards calibrated spread, not smoothing.
   * multiscale afCRPS: the same on area-averaged fields at 2-16 px, so large-scale
     placement and amplitude stay right (scale-aware scoring).
   * variogram (edge) score: for lags of 1-8 px in four directions, the fair
     squared error between the members' expected |increment| and the truth's,
     relative to the mean of both squared increments (bounded, ~2 at most, so
     dry or flat patches cannot dominate). Too-smooth members have
     too-small increments across fronts and are penalized; the finite-ensemble
     correction means extra noise does not lower the score.
   * bias guard: fair squared error of the members' patch-mean against the truth's
     patch-mean for every field, plus the physical rain amount (mm/h), so sharper
     members cannot come with a drifting mean or a worse rain bias.
3. total = flow loss (all patches; keeps the model on its trained distribution)
         + strength * weight * sum_k w_k * score_k.

All model calls live inside one module forward so DDP sees a single forward per
backward (as in rain_rollout_v2).
"""
import torch
from torch import nn
from torch.nn import functional as F

from .v4_1 import objective

SCORES = ('crps', 'multiscale_crps', 'variogram', 'bias')
DEFAULTS = dict(weight=1.0, patches=4, members=2, steps=16, grad_steps=3, interval=1,
                warmup_epochs=0, ramp_epochs=1, alpha=0.95, pool_scales=[2, 4, 8, 16], lags=[1, 2, 4, 8],
                crps_weight=1.0, multiscale_weight=1.0, variogram_weight=1.0, bias_weight=1.0,
                channel_weights=[1., 1., 1., 1., 1., 1.])
KEYS = set(DEFAULTS)


def settings_from(cfg):
    ft = cfg['train'].get('finetune') or {}
    rollout = ft.get('rollout')
    if not rollout:
        return None
    unknown = set(rollout)-KEYS
    if unknown:
        raise ValueError(f'Unknown finetune.rollout keys: {sorted(unknown)}')
    s = dict(DEFAULTS, **rollout)
    if s['members'] < 2 or s['patches'] < 1 or s['interval'] < 1 or s['ramp_epochs'] < 1:
        raise ValueError('rollout: members >= 2, patches >= 1, interval >= 1, ramp_epochs >= 1')
    if not 1 <= s['grad_steps'] <= s['steps']:
        raise ValueError('rollout: 1 <= grad_steps <= steps')
    if not 0 <= s['alpha'] <= 1 or s['weight'] < 0:
        raise ValueError('rollout: 0 <= alpha <= 1 and weight >= 0')
    if len(s['channel_weights']) != 6 or min(s['channel_weights']) < 0:
        raise ValueError('rollout.channel_weights: six non-negative values')
    return s


def strength(settings, epoch, batch_index):
    """Zero-based epoch of the fine-tune run; identical on every rank and resume."""
    if batch_index % settings['interval'] or epoch < settings['warmup_epochs']:
        return 0.
    return min(1., (epoch-settings['warmup_epochs']+1)/settings['ramp_epochs'])


def _average(field, area):
    """Area-weighted mean over the last two dims; field (..., B, C, H, W), area (B, H, W)."""
    a = area[:, None]
    return (field*a).sum((-2, -1))/a.sum((-2, -1))


def afcrps(members, truth, area, alpha):
    """Almost-fair CRPS per (patch, channel); members (M, B, C, H, W), truth (B, C, H, W)."""
    n = len(members)
    skill = (members-truth).abs().mean(0)
    ordered = members.sort(dim=0).values
    k = torch.arange(n, device=members.device, dtype=members.dtype)
    coefficients = (2*k-n+1).reshape(n, *([1]*(members.ndim-1)))
    pair_sum = (ordered*coefficients).sum(0)            # sum over i<j of |x_i - x_j|
    fair = pair_sum/(n*(n-1))                           # 0.5 * mean over i != j
    standard = pair_sum/(n*n)                           # 0.5 * mean over all i, j
    return _average(skill-alpha*fair-(1-alpha)*standard, area)


def multiscale_crps(members, truth, area, alpha, scales):
    values = []
    for s in scales:
        if s > min(truth.shape[-2:]):
            continue
        mass = F.avg_pool2d(area[:, None], s)            # (B, 1, h, w)

        def pool(x):                                     # (..., B, C, H, W)
            shape = x.shape
            flat = (x*area[:, None]).reshape(-1, *shape[-3:])
            pooled = F.avg_pool2d(flat, s)
            return pooled.reshape(*shape[:-2], *pooled.shape[-2:])/mass
        values.append(afcrps(pool(members), pool(truth), mass[:, 0], alpha))
    return torch.stack(values).mean(0) if values else truth.new_zeros(truth.shape[:2])


def variogram(members, truth, area, lags):
    """Relative fair edge error per (patch, channel)."""
    n = len(members)
    h, w = truth.shape[-2:]
    values = []
    for lag in lags:
        for dy, dx in ((0, lag), (lag, 0), (lag, lag), (lag, -lag)):
            if abs(dy) >= h or abs(dx) >= w:
                continue
            a = (..., slice(max(dy, 0), h+min(dy, 0)), slice(max(dx, 0), w+min(dx, 0)))
            b = (..., slice(max(-dy, 0), h-max(dy, 0)), slice(max(-dx, 0), w-max(dx, 0)))
            increments = (members[a]-members[b]).abs()
            observed = (truth[a]-truth[b]).abs()
            weights = (area[a]+area[b])/2
            fair = (increments.mean(0)-observed).square()-increments.var(0, correction=1)/n
            # Symmetric relative error: normalized by the mean of truth's and the
            # members' squared increments (detached, floored), so a patch scores at
            # most ~2. Dividing by truth alone blew up in dry/flat patches.
            scale = (.5*(_average(observed.square(), weights)+_average(increments.square().mean(0), weights))
                     ).detach().clamp_min(1e-4)
            values.append(_average(fair, weights)/scale)
    return torch.stack(values).mean(0) if values else truth.new_zeros(truth.shape[:2])


def bias(members, truth, area, rain_scale=1.):
    """Fair squared patch-mean error per (patch, channel); channel 1 (rain, sqrt1p z)
    also gets the physical rain amount in mm/h."""
    n = len(members)

    def fair(m, t):
        return (m.mean(0)-t).square()-m.var(0, correction=1)/n
    score = fair(_average(members, area), _average(truth, area))
    def rain(z):
        z = z.clamp_min(0)
        return rain_scale*z*(z+2)
    physical = fair(_average(rain(members[:, :, 1:2]), area), _average(rain(truth[:, 1:2]), area))
    return torch.cat([score[:, :1], score[:, 1:2]+physical, score[:, 2:]], dim=1)


def sample_scores(members, truth, area, settings, rain_scale=1.):
    """dict of (B, C) scores; members (M, B, C, H, W) and truth (B, C, H, W) in flow space."""
    members, truth, area = members.float(), truth.float(), area.float()
    return dict(crps=afcrps(members, truth, area, settings['alpha']),
                multiscale_crps=multiscale_crps(members, truth, area, settings['alpha'], settings['pool_scales']),
                variogram=variogram(members, truth, area, settings['lags']),
                bias=bias(members, truth, area, rain_scale))


def sample_members(network, batch, members, steps, grad_steps, generator=None):
    """Free members (M, B, C, H, W); only the last ``grad_steps`` Heun steps carry gradients."""
    target = batch['target']
    b = len(target)
    def repeat(value):
        return value.repeat(members, *([1]*(value.ndim-1)))
    condition, context, mean = repeat(batch['condition']), repeat(batch['context']), repeat(batch['mean'])
    x = torch.randn((members*b, *target.shape[1:]), device=target.device, generator=generator)
    dt = 1/steps
    def heun(x, i):
        t = torch.full((len(x),), i*dt, device=x.device)
        k1 = network(x, t, condition, context, mean).float()
        k2 = network(x+dt*k1, t+dt, condition, context, mean).float()
        return x+dt*(k1+k2)/2
    split = steps-grad_steps
    with torch.no_grad():
        for i in range(split):
            x = heun(x, i)
    x = x.detach()
    for i in range(split, steps):
        x = heun(x, i)
    return x.reshape(members, b, *target.shape[1:])


def rollout_loss(network, batch, settings, generator=None, rain_scale=1.):
    count = min(settings['patches'], len(batch['target']))
    selected = {k: v[:count] for k, v in batch.items() if torch.is_tensor(v)}
    members = sample_members(network, selected, settings['members'], settings['steps'],
                             settings['grad_steps'], generator)
    scores = sample_scores(members, selected['target'], selected['area_full'], settings, rain_scale)
    channels = torch.as_tensor(settings['channel_weights'], device=members.device, dtype=torch.float32)
    importance = selected['importance'].float()
    per_score = torch.stack([((scores[k]*channels).sum(1)/channels.sum()*importance).mean() for k in SCORES])
    weights = per_score.new_tensor([settings['crps_weight'], settings['multiscale_weight'], settings['variogram_weight'],
                                    settings['bias_weight']])
    return (per_score*weights).sum(), per_score.detach()


class RolloutObjective(nn.Module):
    """Flow loss on every patch plus, when ``strength`` > 0, sample scores on a subset.
    Returns (total loss, flow loss, per-score values)."""

    def __init__(self, network, settings, loss_options, rain_scale=1.):
        super().__init__()
        self.network, self.settings, self.loss_options = network, settings, loss_options
        self.rain_scale = float(rain_scale)

    @property
    def channel_weights(self):
        return self.network.channel_weights

    def forward(self, batch, strength=0.):
        flow = objective(self.network, batch, **self.loss_options)
        scores = flow.new_zeros(len(SCORES))
        loss = flow
        if strength:
            value, scores = rollout_loss(self.network, batch, self.settings, rain_scale=self.rain_scale)
            loss = flow+strength*self.settings['weight']*value
        return loss, flow.detach(), scores
