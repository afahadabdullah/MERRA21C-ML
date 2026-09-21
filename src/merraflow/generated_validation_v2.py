"""Select flow checkpoints using generated validation rainfall in physical units."""
import torch
import torch.distributed as dist
from .loss_v2 import core_v2
from .model_v2 import regression_v2, integrate_v2
from .train import to_device, autocast
from .rain_prior_v2 import training_noise_v2


def rain_structure_score_v2(members, target, area):
    """Variogram score over horizontal/vertical pairs at 1, 4 and 16 pixels.

    Compare the ensemble's expected absolute spatial increments with the target's.
    Scoring individual increments penalizes grain hidden by the ensemble mean.
    """
    scores = []
    for lag in (1, 4, 16):
        for axis in (-1, -2):
            if target.shape[axis] <= lag:
                continue
            left, right = [slice(None)]*3, [slice(None)]*3
            left[axis], right[axis] = slice(lag, None), slice(None, -lag)
            left, right = tuple(left), tuple(right)
            observed = (target[left]-target[right]).abs()
            simulated = (members[(slice(None),)+left]-members[(slice(None),)+right]).abs().mean(0)
            weights = (area[left]+area[right])/2
            scores.append(((simulated-observed).square()*weights).sum((-2, -1))/weights.sum((-2, -1)))
    return torch.stack(scores).mean(0)


def rain_ensemble_scores_v2(members, target, area):
    """Per-patch CRPS and ensemble-mean squared error; no dry-pixel masking."""
    ordered = members.sort(dim=0).values
    n = members.shape[0]
    coefficients = (2*torch.arange(n, device=members.device)+1-n).reshape(n, 1, 1, 1)
    crps = (members-target).abs().mean(0)-(ordered*coefficients).sum(0)/(n*n)
    mse = (members.mean(0)-target).square()
    norm = area.sum((-2, -1))
    return (crps*area).sum((-2, -1))/norm, (mse*area).sum((-2, -1))/norm


@torch.no_grad()
def generated_validation_v2(flow, mean_model, flow_scale, loader, cfg, device, rank, world):
    settings = cfg['train']['generated_validation']
    rng = torch.Generator(device=device).manual_seed(cfg['train']['seed']+20000+rank)
    sums = torch.zeros(8, dtype=torch.float64, device=device)
    halo = cfg['patch']['halo']
    for index, batch in enumerate(loader):
        if index >= settings['batches']:
            break
        batch = to_device(batch, device)
        with autocast(device, cfg['train']['precision']):
            mean = regression_v2(mean_model, batch)
            rain_members = []
            for _ in range(settings['members']):
                noise = training_noise_v2(mean.shape, device, rng, cfg)
                residual = integrate_v2(flow, noise, batch['condition'], batch['context'], mean, settings['steps'])
                z = (batch['rain_baseline']+(mean+flow_scale*residual)[:, 1]).clamp_min(0)
                rain = batch['rain_scale'][:, None, None]*z*(z+2)
                rain_members.append(core_v2(rain[:, None], halo)[:, 0])
        target = core_v2(batch['rain_truth'][:, None], halo)[:, 0]
        members = torch.stack(rain_members)
        crps, mse = rain_ensemble_scores_v2(members, target, batch['area'])
        structure = rain_structure_score_v2(members, target, batch['area'])
        coarse = core_v2(batch['rain_coarse'][:, None], halo)[:, 0]
        coarse_mae, coarse_mse = rain_ensemble_scores_v2(coarse[None], target, batch['area'])
        coarse_structure = rain_structure_score_v2(coarse[None], target, batch['area'])
        zmean = (batch['rain_baseline']+mean[:, 1]).clamp_min(0)
        regression = core_v2((batch['rain_scale'][:, None, None]*zmean*(zmean+2))[:, None], halo)[:, 0]
        _, regression_mse = rain_ensemble_scores_v2(regression[None], target, batch['area'])
        sums[0] += crps.double().sum()
        sums[1] += mse.double().sum()
        sums[2] += structure.double().sum()
        sums[3] += len(target)
        sums[4] += coarse_mae.double().sum()
        sums[5] += coarse_mse.double().sum()
        sums[6] += coarse_structure.double().sum()
        sums[7] += regression_mse.double().sum()
    if world > 1:
        dist.all_reduce(sums)
    if sums[3] == 0 or not torch.isfinite(sums).all():
        raise FloatingPointError('Invalid generated validation rainfall')
    crps, rmse, structure = float(sums[0]/sums[3]), float((sums[1]/sums[3]).sqrt()), float((sums[2]/sums[3]).sqrt())
    coarse_mae, coarse_rmse, coarse_structure = float(sums[4]/sums[3]), float((sums[5]/sums[3]).sqrt()), float((sums[6]/sums[3]).sqrt())
    return dict(crps_mm_h=crps, ensemble_mean_rmse_mm_h=rmse,
                structure_rmse_mm_h=structure,
                coarse_mae_mm_h=coarse_mae, coarse_rmse_mm_h=coarse_rmse,
                coarse_structure_rmse_mm_h=coarse_structure,
                regression_rmse_mm_h=float((sums[7]/sums[3]).sqrt()),
                rmse_skill_vs_coarse=1-rmse/coarse_rmse if coarse_rmse > 0 else None,
                beats_coarse=(rmse < coarse_rmse and crps < coarse_mae and structure < coarse_structure),
                selection_score=crps+rmse+structure, patches=int(sums[3]),
                members=settings['members'], steps=settings['steps'],
                sampling='independent validation patches; full-domain inference is synchronized')
