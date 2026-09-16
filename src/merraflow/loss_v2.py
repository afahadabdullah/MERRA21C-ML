"""Quadratic flow objectives preserve the conditional-mean velocity optimum.

No GAN/perceptual loss and no target-dependent rain-intensity weighting. Proposal
importance weights undo detail oversampling; validation uses uniform proposals.
The optional regression-only physical rainfall term also supervises rain amounts.
"""
import torch
from torch.nn import functional as F
from .model_v2 import regression_v2
from .rain_prior_v2 import training_noise_v2


def multiscale_v2(error, area, importance, weights, scales=(2, 4, 8, 16)):
    """Area-weighted pooled errors; a linear spatial constraint on velocity."""
    losses = []
    for scale in scales:
        if scale > min(error.shape[-2:]):
            continue
        mass = F.avg_pool2d(area[:, None], scale)
        pooled = F.avg_pool2d(error*area[:, None], scale)/mass
        losses.append(quadratic_v2(pooled, mass[:, 0], importance, weights)[0])
    return sum(losses)/len(losses) if losses else error.sum()*0


def rain_physical_v2(prediction, batch, halo):
    """Supervise actual rain amount and regional means, without a rain cap."""
    z = (prediction[:, 1]+batch['rain_baseline']).clamp_min(0)
    rain = batch['rain_scale'][:, None, None]*z*(z+2)
    # 10 mm/h defines the units of this auxiliary loss, not an output limit.
    error = core_v2(((rain-batch['rain_truth'])/10)[:, None], halo)
    area, importance = batch['area'], batch['importance']
    pixel = quadratic_v2(error, area, importance, [1.])[0]
    pooled = multiscale_v2(error, area, importance, [1.], (4, 16, 64))
    domain_error = (error[:, 0]*area).sum((-2, -1))/area.sum((-2, -1))
    amount = (domain_error.square()*importance).mean()
    return (pixel+pooled+amount)/3


def core_v2(x, halo):
    return x[:, :, halo:-halo, halo:-halo] if halo else x


def quadratic_v2(error, area, importance, channel_weights):
    per_channel = (error.square()*area[:, None]).sum((-2, -1))/area.sum((-2, -1))[:, None]
    per_channel = (per_channel*importance[:, None]).mean(0)
    weights = torch.as_tensor(channel_weights, device=error.device, dtype=torch.float32)
    return (per_channel*weights).sum()/weights.sum(), per_channel


def gradient_v2(error, area, importance, weights):
    dx = error[..., 1:]-error[..., :-1]
    dy = error[..., 1:, :]-error[..., :-1, :]
    return (quadratic_v2(dx, (area[..., 1:]+area[..., :-1])/2, importance, weights)[0]+
            quadratic_v2(dy, (area[..., 1:, :]+area[..., :-1, :])/2, importance, weights)[0])/2


def loss_v2(model, batch, cfg, stage, mean_model=None, flow_scale=None, generator=None):
    target = batch['target']
    weights, halo = cfg['train']['channel_weights'], cfg['patch']['halo']
    if stage == 'regression':
        prediction = regression_v2(model, batch)
        error = core_v2(prediction-target, halo)
        endpoint_error = error
    else:
        with torch.no_grad():
            mean = regression_v2(mean_model, batch)
        x1 = (target-mean)/flow_scale
        x0 = training_noise_v2(x1.shape, x1.device, generator, cfg)
        t = torch.rand(x1.shape[0], device=x1.device, generator=generator)
        time = t[:, None, None, None]
        xt = (1-time)*x0+time*x1
        velocity = model(xt, t, batch['condition'], batch['context'], mean).float()
        full_patch = cfg['loss'].get('flow_full_patch', False)
        error = core_v2(velocity-(x1-x0), 0 if full_patch else halo)
        # x_hat_1 = x_t + (1-t)*v; compute its error without dividing by 1-t.
        endpoint_error = (1-time)*error
    area = batch['area_full'] if stage == 'flow' and cfg['loss'].get('flow_full_patch', False) else batch['area']
    importance = batch['importance']
    value, channels = quadratic_v2(error, area, importance, weights)
    grad = gradient_v2(endpoint_error, area, importance, weights)
    multiscale = value*0
    if stage == 'regression':
        for scale in (2, 4):
            pooled = F.avg_pool2d(error, scale)
            pooled_area = F.avg_pool2d(area[:, None], scale)[:, 0]
            multiscale = multiscale+quadratic_v2(pooled, pooled_area, importance, weights)[0]/2
    elif cfg['loss'].get('flow_multiscale', 0):
        multiscale = multiscale_v2(endpoint_error, area, importance, weights)
    total = value+cfg['loss'][f'{stage}_gradient']*grad
    if stage == 'regression':
        total = total+cfg['loss']['regression_multiscale']*multiscale
        if cfg['loss'].get('regression_rain_physical', 0):
            total = total+cfg['loss']['regression_rain_physical']*rain_physical_v2(prediction, batch, halo)
    else:
        total = total+cfg['loss'].get('flow_multiscale', 0)*multiscale
    metrics = torch.cat([total[None], value[None], grad[None], multiscale[None], channels])
    return total, metrics.detach()
