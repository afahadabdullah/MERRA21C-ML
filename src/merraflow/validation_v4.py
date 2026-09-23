"""Uniform held-out ensemble evaluation in physical units for all six fields."""
import json
import math
import numpy as np
import torch
import torch.distributed as dist
from .v4 import TARGETS, UNITS, sample
from .train import to_device, autocast
from .train_v3_precip import reduce_totals
from .metrics import crps_ensemble


@torch.no_grad()
def validate(model, conditioner, loader, cfg, device, rain_scale=None):
    rank = dist.get_rank() if dist.is_initialized() else 0
    rng = torch.Generator(device=device).manual_seed(cfg['train']['seed']+7103+rank)
    halo, size = cfg['patch']['halo'], cfg['patch']['size']
    def core(value):
        return value[..., halo:halo+size, halo:halo+size]
    names = ('crps', 'coarse_mae', 'regression_mae', 'mse', 'bias', 'spread')
    sums = {f'{target}_{metric}': 0. for target in TARGETS for metric in names}
    sums.update(wet_fraction=0., truth_wet_fraction=0., false_wet_numerator=0., dry_area=0.,
                missed_wet_numerator=0., wet_area=0.)
    count, previews = 0, []
    for batch in loader:
        b = to_device(batch, device)
        with autocast(device, cfg['train']['precision']):
            mean = conditioner(b)
            draws = []
            for _ in range(cfg['train']['validation_members']):
                noise = torch.randn(b['target'].shape, device=device, generator=rng)
                value = sample(model, noise, b['condition'], b['context'], cfg['train']['validation_steps'], mean)
                draws.append(core(conditioner.physical(value.float(), mean, b['coarse'])).cpu().numpy())
            regression = core(conditioner.regression_physical(mean, b['coarse'])).float().cpu().numpy()
        ensemble = np.stack(draws)
        truth, coarse = [core(b[k]).cpu().numpy() for k in ('truth', 'coarse')]
        area = b['area'].cpu().numpy()
        area = area/area.sum((-2, -1), keepdims=True)
        values = dict(crps=crps_ensemble(ensemble, truth), coarse_mae=abs(coarse-truth),
                      regression_mae=abs(regression-truth), mse=(ensemble.mean(0)-truth)**2,
                      bias=ensemble.mean(0)-truth, spread=ensemble.std(0))
        for channel, name in enumerate(TARGETS):
            for key, value in values.items():
                sums[f'{name}_{key}'] += float((value[:, channel]*area).sum())
        wet = (ensemble[:, :, 1] >= .1).mean(0)
        observed = truth[:, 1] >= .1
        for key, value in dict(wet_fraction=wet, truth_wet_fraction=observed,
                               false_wet_numerator=wet*(~observed), dry_area=~observed,
                               missed_wet_numerator=(1-wet)*observed, wet_area=observed).items():
            sums[key] += float((value*area).sum())
        count += len(truth)
        if rank == 0:
            for j in range(min(len(truth), cfg['train']['validation_plot_samples']-len(previews))):
                previews.append(dict(truth=truth[j], coarse=coarse[j], regression=regression[j],
                                     ensemble=ensemble[:, j], area=area[j]))
    totals = reduce_totals([*sums.values(), count], device)
    if totals[-1] == 0 or not np.isfinite(totals).all():
        raise FloatingPointError('Invalid v4 validation')
    metrics = {k: float(v/totals[-1]) for k, v in zip(sums, totals[:-1])}
    for name in TARGETS:
        metrics[f'{name}_ensemble_mean_rmse'] = math.sqrt(metrics.pop(f'{name}_mse'))
    for num, den, key in [('false_wet_numerator', 'dry_area', 'false_wet_probability'),
                           ('missed_wet_numerator', 'wet_area', 'missed_wet_probability')]:
        numerator, denominator = metrics.pop(num), metrics.pop(den)
        metrics[key] = numerator/denominator if denominator else None
    # Explicit aliases retain the shared trainer's rain-based checkpoint selection.
    metrics['crps'] = metrics['precip_crps']
    return metrics, previews


def save_plots(previews, history, out):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    out.mkdir(parents=True, exist_ok=True)
    if not previews:
        raise ValueError('No validation previews')
    np.savez_compressed(out/'samples.npz', **{f'patch_{i}_{k}':v for i,b in enumerate(previews) for k,v in b.items()})
    for i, item in enumerate(previews):
        truth, coarse, ensemble = [item[k] for k in ('truth', 'coarse', 'ensemble')]
        fig, axes = plt.subplots(len(TARGETS), 7, figsize=(21, 17), constrained_layout=True)
        for c, (name, unit) in enumerate(zip(TARGETS, UNITS)):
            reference = np.stack([truth[c], coarse[c]])
            lo, hi = np.quantile(reference, [.005, .995])
            if c == 1:
                lo, hi = 0., max(1., float(hi))
            hi = max(float(hi), float(lo)+1e-4)
            fields = [('Truth', truth[c]), ('Coarse', coarse[c]), ('Coarse QV2M' if c == 5 else 'Frozen v2', item['regression'][c]),
                      ('Member 1', ensemble[0,c]), ('Member 2', ensemble[1,c]),
                      ('Mean', ensemble[:,c].mean(0)), ('Spread', ensemble[:,c].std(0))]
            for j, (label, field) in enumerate(fields):
                im = axes[c,j].imshow(field, origin='lower', cmap='Blues' if c == 1 or j == 6 else 'viridis',
                                      vmin=0 if j == 6 else lo, vmax=(hi-lo) if j == 6 else hi)
                axes[c,j].set(title=f'{name}: {label}', xticks=[], yticks=[])
                if j == 5:
                    fig.colorbar(im, ax=axes[c,:6].tolist(), label=unit, shrink=.8, extend='both')
                elif j == 6:
                    fig.colorbar(im, ax=axes[c,j], label=unit, shrink=.8, extend='max')
        fig.suptitle(f'V4 epoch {history[-1]["epoch"]}: fixed uniform validation patch {i+1}; rain hourly, states midpoint')
        fig.savefig(out/('fields.png' if i == 0 else f'fields_patch_{i+1}.png'), dpi=110)
        plt.close(fig)
    fig, axes = plt.subplots(2, 4, figsize=(20, 8), constrained_layout=True)
    axes.flat[0].plot([r['epoch'] for r in history], [r['training_loss'] for r in history])
    axes.flat[0].set(title='Weighted flow objective', xlabel='Epoch')
    rows = [r for r in history if 'crps' in r]
    for ax, name, unit in zip(list(axes.flat)[1:], TARGETS, UNITS):
        for metric in ('crps', 'coarse_mae', 'regression_mae'):
            ax.plot([r['epoch'] for r in rows], [r[f'{name}_{metric}'] for r in rows], label=metric)
        ax.set(title=name, ylabel=unit, xlabel='Epoch')
        ax.legend()
    axes.flat[-1].axis('off')
    fig.savefig(out/'history.png', dpi=120)
    plt.close(fig)
    (out/'metrics.json').write_text(json.dumps(history[-1], indent=2)+'\n')
