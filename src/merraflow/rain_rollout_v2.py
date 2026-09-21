"""Optional physical-rain supervision through a differentiable Heun rollout.

Flow matching remains the primary objective. Ensemble scores act on free samples,
never on target-interpolated states. Finite-ensemble corrections avoid rewarding
collapsed ensembles merely because only a few members fit in training memory.
"""
import torch
from torch import nn
from torch.nn import functional as F

from .loss_v2 import core_v2, loss_v2
from .model_v2 import regression_v2, integrate_differentiable_v2
from .rain_prior_v2 import training_noise_v2


ROLLOUT_METRICS = ('weighted_loss', 'crps', 'variogram', 'coverage', 'mean_mse')


def fair_squared_error_v2(samples, target):
    """Unbiased estimate of squared error of an ensemble expectation.

    Can be negative for a finite draw; do not clamp away the variance correction.
    Members must be independent draws conditional on the same inputs.
    """
    return (samples.mean(0)-target).square()-samples.var(0, correction=1)/len(samples)


def rain_sample_scores_v2(members, target, area, settings):
    """Four per-patch losses; members have shape (M, B, H, W), in mm/h."""
    n = len(members)
    if n < 2:
        raise ValueError('Rain rollout requires at least two independent members')
    scale = settings['rate_scale_mm_h']
    rain, truth = members.float()/scale, target.float()/scale

    def average(field, weights=area):
        return (field*weights).sum((-2, -1))/weights.sum((-2, -1))

    ordered = rain.sort(dim=0).values
    coefficients = (2*torch.arange(n, device=rain.device)+1-n)[:, None, None, None]
    crps = average((rain-truth).abs().mean(0)-(ordered*coefficients).sum(0)/(n*(n-1)))
    spatial = []
    for lag in settings['lags']:
        for dy, dx in ((0, lag), (lag, 0), (lag, lag), (lag, -lag)):
            h, w = target.shape[-2:]
            if abs(dy) >= h or abs(dx) >= w:
                continue
            a = (..., slice(max(dy, 0), h+min(dy, 0)), slice(max(dx, 0), w+min(dx, 0)))
            b = (..., slice(max(-dy, 0), h-max(dy, 0)), slice(max(-dx, 0), w-max(dx, 0)))
            increments = (rain[a]-rain[b]).abs()
            observed = (truth[a]-truth[b]).abs()
            spatial.append(average(fair_squared_error_v2(increments, observed), (area[a]+area[b])/2))
    variogram = torch.stack(spatial).mean(0) if spatial else crps*0
    coverage, amounts = [], []
    for size in settings['pool_scales']:
        if size > min(target.shape[-2:]):
            continue
        # ceil_mode retains partial edge blocks, with matching area denominators.
        def pool(value):
            shape = value.shape
            pooled = F.avg_pool2d(value.reshape(-1, 1, *shape[-2:]), size,
                                  ceil_mode=True, count_include_pad=False)
            return pooled.reshape(*shape[:-2], *pooled.shape[-2:])
        mass = pool(area)
        pooled_truth = pool(truth*area)/mass
        pooled_mean = pool(rain.mean(0)*area)/mass
        # Deliberate finite-ensemble mean error term: the user's deterministic
        # objective, alongside (not replacing) distributional scores.
        # Sum pooling recovers exact area in partial edge blocks.
        block_area = F.avg_pool2d(area[:, None], size, ceil_mode=True,
                                  divisor_override=1)[:, 0]
        amounts.append(average((pooled_mean-pooled_truth).square(), block_area))
        for threshold in settings['thresholds_mm_h']:
            temperature = settings['coverage_temperature_mm_h']
            occurrence = torch.sigmoid((members.float()-threshold)/temperature)
            observed = torch.sigmoid((target.float()-threshold)/temperature)
            predicted_fraction = pool(occurrence*area)/mass
            observed_fraction = pool(observed*area)/mass
            coverage.append(average(fair_squared_error_v2(predicted_fraction, observed_fraction), block_area))
    return torch.stack((crps, variogram, torch.stack(coverage).mean(0), torch.stack(amounts).mean(0)), dim=-1)


def rollout_strength_v2(settings, epoch, batch_index):
    """Epoch is zero-based; cadence is identical on every DDP rank and resume."""
    if batch_index % settings['interval'] or epoch < settings['warmup_epochs']:
        return 0.
    return min(1., (epoch-settings['warmup_epochs']+1)/settings['ramp_epochs'])


def rain_rollout_loss_v2(model, batch, mean_model, flow_scale, cfg, generator=None):
    settings = cfg['train']['rain_rollout']
    count = min(settings['patches'], len(batch['target']))
    selected = {key: value[:count] for key, value in batch.items()}
    with torch.no_grad():
        mean = regression_v2(mean_model, selected)
    n = settings['members']
    # Member-major flattening: every member has the same conditioning, with
    # independent noise. The truth never enters the sampling trajectory.
    def repeat(value):
        return value.repeat(n, *([1]*(value.ndim-1)))
    mean = repeat(mean)
    noise = training_noise_v2(mean.shape, mean.device, generator, cfg)
    residual = integrate_differentiable_v2(model, noise, repeat(selected['condition']),
                                          repeat(selected['context']), mean, settings['steps'])
    z = (repeat(selected['rain_baseline'])+(mean+flow_scale*residual)[:, 1]).clamp_min(0)
    rain = repeat(selected['rain_scale'])[:, None, None]*z*(z+2)
    halo = cfg['patch']['halo']
    rain = core_v2(rain[:, None], halo)[:, 0].reshape(n, count, *selected['area'].shape[-2:])
    truth = core_v2(selected['rain_truth'][:, None], halo)[:, 0]
    scores = rain_sample_scores_v2(rain, truth, selected['area'], settings)
    scores = (scores*selected['importance'][:, None]).mean(0)
    weights = scores.new_tensor([settings[key] for key in ('crps_weight', 'variogram_weight',
                                                         'coverage_weight', 'mean_mse_weight')])
    return (scores*weights).sum(), scores.detach()


class RainFlowObjectiveV2(nn.Module):
    """One DDP forward encloses all velocity calls and both objectives.

    Wrapping each Heun evaluation independently with DDP/find_unused_parameters
    would make reduction bookkeeping depend on multiple outstanding forwards.
    The frozen mean is deliberately supplied by the caller, outside DDP state.
    """
    def __init__(self, network, cfg):
        super().__init__()
        self.network, self.cfg = network, cfg

    def forward(self, batch, mean_model, flow_scale, strength=0.):
        loss, metrics = loss_v2(self.network, batch, self.cfg, 'flow', mean_model, flow_scale)
        auxiliary = metrics.new_zeros(len(ROLLOUT_METRICS))
        if strength:
            value, components = rain_rollout_loss_v2(self.network, batch, mean_model, flow_scale, self.cfg)
            contribution = strength*self.cfg['train']['rain_rollout']['weight']*value
            loss = loss+contribution
            auxiliary = torch.cat((contribution.detach()[None], components))
        return loss, metrics, auxiliary
