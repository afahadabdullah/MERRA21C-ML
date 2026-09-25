"""v4.1 validation: v4's metrics plus distribution diagnostics and fixed rainy previews.

Every validation (every ``validation_interval`` epochs) writes to
``validation_plots/epoch_NNNN/``:

* ``rain_patch_K.png``   rain maps for fixed rainy validation patches: coarse
                         input, frozen regression, truth, members, ensemble
                         mean, exceedance probability and spread
* ``states_patch_K.png`` t2m/ps/u10m/v10m/q2m for the same patches: coarse,
                         truth, ensemble mean, mean-truth and spread
* ``diagnostics.png``    over all validation patches: rain-intensity
                         distribution, rank histograms, radial power spectra
* ``history.png``        training/validation curves up to this epoch
* ``metrics.json`` and ``samples.npz`` (preview arrays)

Metrics use the uniform validation patches exactly as v4 does. The preview
patches are chosen once, from the validation hours with the most observed rain
among the input-rain-richest origins. They are stored in
``preview_patches.json`` and generated with fixed noise, so figures from
different epochs are directly comparable.
"""
from pathlib import Path
import json
import math
import numpy as np
import torch
from .v4 import TARGETS, UNITS, sample
from .train import to_device, autocast
from .train_v3_precip import reduce_totals
from .metrics import crps_ensemble

RAIN_EDGES = np.array([0, .1, .25, .5, 1, 2, 4, 8, 16, 32, 64, np.inf])
RAIN_LEVELS = [.1, .25, .5, 1, 2, 4, 8, 16, 32, 64]
RAIN_COLORS = ['#e1f5fe', '#9ad0f5', '#4ea3e6', '#1f6fd1', '#1a9850', '#91cf60',
               '#fee08b', '#fc8d59', '#d73027', '#8e0152']
PREVIEWS = 'preview_patches.json'


def _core(value, halo, size):
    return value[..., halo:halo+size, halo:halo+size]


def radial_spectrum(fields):
    """Mean radially averaged power spectrum of (..., n, n) fields, mean removed."""
    n = fields.shape[-1]
    f = np.fft.fft2(fields-fields.mean((-2, -1), keepdims=True))
    power = (np.abs(f)**2).reshape(-1, n*n).mean(0)
    k = np.fft.fftfreq(n)*n
    radius = np.rint(np.hypot(*np.meshgrid(k, k, indexing='ij'))).astype(int).ravel()
    sums = np.bincount(radius, power, minlength=n)[1:n//2+1]
    counts = np.bincount(radius, minlength=n)[1:n//2+1]
    return sums/np.maximum(counts, 1)


def rank_counts(ensemble, truth, rng):
    """Counts of the truth's rank among members, ties broken at random."""
    below = (ensemble < truth).sum(0)
    ties = (ensemble == truth).sum(0)
    rank = below+np.floor(rng.random(below.shape)*(ties+1)).astype(int)
    return np.bincount(rank.ravel(), minlength=len(ensemble)+1)


@torch.no_grad()
def validate(model, conditioner, loader, cfg, device, group=None):
    """v4's validation metrics over uniform patches, plus summed diagnostics."""
    import torch.distributed as dist
    rank = dist.get_rank() if dist.is_initialized() else 0
    rng = torch.Generator(device=device).manual_seed(cfg['train']['seed']+7103+rank)
    tie_rng = np.random.default_rng(cfg['train']['seed']+rank)
    halo, size = cfg['patch']['halo'], cfg['patch']['size']
    members = cfg['train']['validation_members']
    names = ('crps', 'coarse_mae', 'regression_mae', 'mse', 'bias', 'spread')
    sums = {f'{target}_{metric}': 0. for target in TARGETS for metric in names}
    sums.update(wet_fraction=0., truth_wet_fraction=0., false_wet_numerator=0., dry_area=0.,
                missed_wet_numerator=0., wet_area=0.)
    bins = len(RAIN_EDGES)-1
    diag = dict(rain_truth=np.zeros(bins), rain_ensemble=np.zeros(bins), rain_coarse=np.zeros(bins),
                rain_regression=np.zeros(bins), ranks=np.zeros((len(TARGETS), members+1)),
                spectra=np.zeros((4, len(TARGETS), size//2)), spectra_count=np.zeros(1))
    count = 0
    for batch in loader:
        b = to_device(batch, device)
        with autocast(device, cfg['train']['precision']):
            mean = conditioner(b)
            draws = []
            for _ in range(members):
                noise = torch.randn(b['target'].shape, device=device, generator=rng)
                value = sample(model, noise, b['condition'], b['context'], cfg['train']['validation_steps'], mean)
                draws.append(_core(conditioner.physical(value.float(), mean, b['coarse']), halo, size).cpu().numpy())
            regression = _core(conditioner.regression_physical(mean, b['coarse']), halo, size).float().cpu().numpy()
        ensemble = np.stack(draws)
        truth, coarse = [_core(b[k], halo, size).cpu().numpy() for k in ('truth', 'coarse')]
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
        # Area-weighted rain-intensity histograms (fraction of patch area).
        for key, field in (('rain_truth', truth[:, 1]), ('rain_coarse', coarse[:, 1]),
                           ('rain_regression', regression[:, 1])):
            index = np.digitize(field, RAIN_EDGES[1:-1])
            diag[key] += np.bincount(index.ravel(), (np.broadcast_to(area, field.shape)).ravel(), minlength=bins)
        index = np.digitize(ensemble[:, :, 1], RAIN_EDGES[1:-1])
        diag['rain_ensemble'] += np.bincount(index.ravel(), np.broadcast_to(area, index.shape).ravel(),
                                             minlength=bins)/members
        for channel in range(len(TARGETS)):
            diag['ranks'][channel] += rank_counts(ensemble[:, :, channel], truth[:, channel], tie_rng)
            for row, field in enumerate((truth[:, channel], ensemble[:, :, channel], coarse[:, channel],
                                         regression[:, channel])):
                diag['spectra'][row, channel] += radial_spectrum(field)*len(truth)
        diag['spectra_count'] += len(truth)
        count += len(truth)
    keys = list(diag)
    flat = np.concatenate([diag[k].ravel() for k in keys])
    totals = reduce_totals([*sums.values(), count, *flat.tolist()], device)
    head = totals[:len(sums)+1]
    if head[-1] == 0 or not np.isfinite(totals).all():
        raise FloatingPointError('Invalid v4.1 validation')
    metrics = {k: float(v/head[-1]) for k, v in zip(sums, head[:-1])}
    for name in TARGETS:
        metrics[f'{name}_ensemble_mean_rmse'] = math.sqrt(metrics.pop(f'{name}_mse'))
    for num, den, key in [('false_wet_numerator', 'dry_area', 'false_wet_probability'),
                           ('missed_wet_numerator', 'wet_area', 'missed_wet_probability')]:
        numerator, denominator = metrics.pop(num), metrics.pop(den)
        metrics[key] = numerator/denominator if denominator else None
    metrics['crps'] = metrics['precip_crps']  # checkpoint selection, as v4
    at, diagnostics = len(sums)+1, {}
    for key in keys:
        n = diag[key].size
        diagnostics[key] = np.asarray(totals[at:at+n]).reshape(diag[key].shape)
        at += n
    diagnostics['spectra'] /= max(diagnostics.pop('spectra_count')[0], 1)
    return metrics, diagnostics


# ----------------------------------------------------------------------------
# Fixed rainy preview patches
# ----------------------------------------------------------------------------

def select_previews(packed, cfg, count, out=None):
    """Choose (once) rainy validation (hour, origin) pairs, spread across time.

    Candidates are the input-rain-richest origin of each validation hour; the
    final choice ranks them by observed area-mean rain in the loss core. The
    choice is cached in ``out/preview_patches.json`` so it never changes.
    """
    from datetime import datetime
    from .v4_1 import DatasetV41
    path = Path(out)/PREVIEWS if out is not None else None
    if path is not None and path.exists():
        chosen = json.loads(path.read_text())
        if len(chosen) >= count and all(c['id'] in packed.by_id for c in chosen):
            return chosen[:count]
    data = DatasetV41(cfg, 'val', 1, 0, packed=packed)
    halo, size = cfg['patch']['halo'], cfg['patch']['size']
    best = []
    for entry in data.entries:
        score = packed.score(entry)
        j = int(np.argmax(score))
        best.append((float(score[j]), entry['id'], int(data.yy[j]), int(data.xx[j])))
    best.sort(key=lambda item: (-item[0], item[1]))
    scored = []
    for _, entry_id, y, x in best[:max(8*count, 32)]:
        entry = packed.by_id[entry_id]
        b = packed.sample(entry, y, x)
        rain = _core(b['truth'][1].numpy(), halo, size)
        area = b['area'].numpy()/b['area'].numpy().sum()
        scored.append(dict(id=entry_id, time=entry['time'], y=y, x=x,
                           truth_rain_mean=float((rain*area).sum()),
                           truth_wet_fraction=float(((rain >= .1)*area).sum())))
    scored.sort(key=lambda c: (-c['truth_rain_mean'], c['id']))
    chosen = []
    for candidate in scored:  # at least 12 h apart: distinct weather
        t = datetime.fromisoformat(candidate['time'])
        if all(abs((t-datetime.fromisoformat(c['time'])).total_seconds()) >= 12*3600 for c in chosen):
            chosen.append(candidate)
        if len(chosen) == count:
            break
    for candidate in scored:  # short validation sets: fill without spacing
        if len(chosen) == count:
            break
        if candidate not in chosen:
            chosen.append(candidate)
    if path is not None:
        path.write_text(json.dumps(chosen, indent=2)+'\n')
    return chosen


@torch.no_grad()
def previews(model, conditioner, packed, chosen, cfg, device):
    """Ensembles for the fixed preview patches, with fixed noise every epoch."""
    halo, size = cfg['patch']['halo'], cfg['patch']['size']
    members = cfg['train']['validation_members']
    samples = [packed.sample(packed.by_id[c['id']], c['y'], c['x']) for c in chosen]
    b = to_device({k: torch.stack([s[k] for s in samples]) for k in samples[0]}, device)
    rng = torch.Generator(device=device).manual_seed(cfg['train']['seed']+424242)
    with autocast(device, cfg['train']['precision']):
        mean = conditioner(b)
        draws = []
        for _ in range(members):
            noise = torch.randn(b['target'].shape, device=device, generator=rng)
            value = sample(model, noise, b['condition'], b['context'], cfg['train']['validation_steps'], mean)
            draws.append(_core(conditioner.physical(value.float(), mean, b['coarse']), halo, size).cpu().numpy())
        regression = _core(conditioner.regression_physical(mean, b['coarse']), halo, size).float().cpu().numpy()
    ensemble = np.stack(draws)
    truth, coarse = [_core(b[k], halo, size).cpu().numpy() for k in ('truth', 'coarse')]
    area = b['area'].cpu().numpy()
    return [dict(truth=truth[j], coarse=coarse[j], regression=regression[j], ensemble=ensemble[:, j],
                 area=area[j]/area[j].sum(), **chosen[j]) for j in range(len(chosen))]


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------

def _style(plt):
    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 10, 'axes.titleweight': 'normal',
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.facecolor': 'white', 'savefig.facecolor': 'white',
                         'axes.grid': False, 'legend.frameon': False})


def _rain_norm():
    from matplotlib import colors
    cmap = colors.ListedColormap(RAIN_COLORS[:len(RAIN_LEVELS)-1])
    cmap.set_under('white')
    cmap.set_over(RAIN_COLORS[-1])
    return cmap, colors.BoundaryNorm(RAIN_LEVELS, cmap.N)


def _panel(ax, field, title, **kwargs):
    im = ax.imshow(field, origin='lower', interpolation='nearest', **kwargs)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('#999999')
        spine.set_linewidth(.6)
    return im


def _stamp(ax, text):
    ax.text(.02, .02, text, transform=ax.transAxes, fontsize=8, va='bottom', ha='left',
            bbox=dict(boxstyle='round,pad=.25', fc='white', ec='none', alpha=.8))


def _heading(item, epoch):
    return (f'v4.1 · epoch {epoch} · validation {item["time"].replace("T", " ")[:16]} UTC · '
            f'patch origin (y={item["y"]}, x={item["x"]})')


def plot_rain(item, epoch, path, plt):
    cmap, norm = _rain_norm()
    area = item['area']
    ensemble = item['ensemble'][:, 1]
    fields = [('Coarse input (GEOS-FP)', item['coarse'][1]), ('Frozen v2 regression', item['regression'][1]),
              ('Truth (HWT hourly)', item['truth'][1]), ('Ensemble mean', ensemble.mean(0))]
    members = [(f'Member {m+1}', ensemble[m]) for m in range(min(4, len(ensemble)))]
    fig, axes = plt.subplots(2, 5, figsize=(19, 8.2), constrained_layout=True)
    for ax, (label, field) in zip(list(axes[0, :4])+list(axes[1, :4]), fields+members):
        im = _panel(ax, field, label, cmap=cmap, norm=norm)
        _stamp(ax, f'mean {float((field*area).sum()):.2f} · max {float(field.max()):.1f} mm/h · '
                   f'wet {float(((field >= .1)*area).sum()):.0%}')
    for ax in axes[1, len(members):4]:
        ax.axis('off')
    fig.colorbar(im, ax=axes[:, :4].ravel().tolist(), label='Rain rate (mm h$^{-1}$)', shrink=.85,
                 extend='both', ticks=RAIN_LEVELS, pad=.01)
    probability = (ensemble >= 1).mean(0)
    im = _panel(axes[0, 4], probability, 'P(rain ≥ 1 mm/h)', cmap='PuBuGn', vmin=0, vmax=1)
    axes[0, 4].contour(item['truth'][1] >= 1, levels=[.5], colors='k', linewidths=.7)
    _stamp(axes[0, 4], 'black contour: observed ≥ 1 mm/h')
    fig.colorbar(im, ax=axes[0, 4], shrink=.85, label='Probability')
    spread = ensemble.std(0)
    im = _panel(axes[1, 4], spread, 'Ensemble spread (std)', cmap='magma_r', vmin=0,
                vmax=max(float(np.quantile(spread, .995)), 1e-3))
    fig.colorbar(im, ax=axes[1, 4], shrink=.85, label='mm h$^{-1}$', extend='max')
    fig.suptitle(f'{_heading(item, epoch)}\nObserved area-mean {item["truth_rain_mean"]:.2f} mm/h, '
                 f'wet area {item["truth_wet_fraction"]:.0%} (≥ 0.1 mm/h) · loss core 128×128 · '
                 f'{len(item["ensemble"])} members', fontsize=12)
    fig.savefig(path, dpi=110)
    plt.close(fig)


STATE_STYLE = {'t2m': ('RdYlBu_r', False), 'ps': ('cividis', False), 'u10m': ('RdBu_r', True),
               'v10m': ('RdBu_r', True), 'q2m': ('YlGnBu', False)}


def plot_states(item, epoch, path, plt):
    names = [n for n in TARGETS if n != 'precip']
    fig, axes = plt.subplots(len(names), 5, figsize=(18, 3.3*len(names)), constrained_layout=True)
    for r, name in enumerate(names):
        c = TARGETS.index(name)
        unit = UNITS[c]
        cmap, symmetric = STATE_STYLE[name]
        truth, coarse, ensemble = item['truth'][c], item['coarse'][c], item['ensemble'][:, c]
        mean = ensemble.mean(0)
        lo, hi = np.quantile(np.stack([truth, coarse, mean]), [.005, .995])
        if symmetric:
            hi = max(abs(lo), abs(hi))
            lo = -hi
        hi = max(float(hi), float(lo)+1e-6)
        for j, (label, field) in enumerate((('Coarse input', coarse), ('Truth', truth), ('Ensemble mean', mean))):
            im = _panel(axes[r, j], field, f'{name} · {label}', cmap=cmap, vmin=lo, vmax=hi)
        fig.colorbar(im, ax=axes[r, :3].tolist(), label=unit, shrink=.9, pad=.01)
        error = mean-truth
        bound = max(float(np.quantile(abs(error), .995)), 1e-6)
        im = _panel(axes[r, 3], error, f'{name} · mean − truth', cmap='RdBu_r', vmin=-bound, vmax=bound)
        rmse = float(np.sqrt(((error**2)*item['area']).sum()))
        coarse_rmse = float(np.sqrt((((coarse-truth)**2)*item['area']).sum()))
        _stamp(axes[r, 3], f'RMSE {rmse:.3g} (coarse {coarse_rmse:.3g}) {unit}')
        fig.colorbar(im, ax=axes[r, 3], shrink=.9, label=unit)
        spread = ensemble.std(0)
        im = _panel(axes[r, 4], spread, f'{name} · spread', cmap='magma_r', vmin=0,
                    vmax=max(float(np.quantile(spread, .995)), 1e-6))
        fig.colorbar(im, ax=axes[r, 4], shrink=.9, label=unit, extend='max')
    fig.suptitle(f'{_heading(item, epoch)} · states (midpoint snapshot)', fontsize=12)
    fig.savefig(path, dpi=100)
    plt.close(fig)


def plot_diagnostics(diagnostics, epoch, path, plt):
    fig = plt.figure(figsize=(20, 12), constrained_layout=True)
    grid = fig.add_gridspec(3, 6)
    ax = fig.add_subplot(grid[0, :3])
    labels = [f'{a:g}–{b:g}' if np.isfinite(b) else f'≥{a:g}' for a, b in zip(RAIN_EDGES[:-1], RAIN_EDGES[1:])]
    labels[0] = '<0.1'
    x = np.arange(len(labels))
    total = max(diagnostics['rain_truth'].sum(), 1e-12)
    for key, label, style in (('rain_truth', 'Truth', dict(color='k', lw=2.4)),
                              ('rain_ensemble', 'Flow ensemble', dict(color='#d73027', lw=2)),
                              ('rain_regression', 'Frozen regression', dict(color='#4575b4', lw=1.5, ls='--')),
                              ('rain_coarse', 'Coarse input', dict(color='#999999', lw=1.5, ls=':'))):
        share = diagnostics[key]/total
        ax.step(x, np.where(share > 0, share, np.nan), where='mid', label=label, **style)  # gaps = empty bins
    ax.set(yscale='log', xticks=x, xticklabels=labels, xlabel='Rain rate (mm h$^{-1}$)',
           ylabel='Fraction of validation area', title='Rain-intensity distribution (all validation patches)')
    ax.legend()
    ax = fig.add_subplot(grid[0, 3:])
    members = diagnostics['ranks'].shape[1]-1
    for c, name in enumerate(TARGETS):
        counts = diagnostics['ranks'][c]
        ax.plot(np.arange(members+1), counts/max(counts.sum(), 1)*(members+1), marker='o', ms=3, label=name)
    ax.axhline(1, color='k', lw=.8, ls='--')
    ax.set(xlabel='Rank of truth among members', ylabel='Relative frequency (flat = 1)',
           title='Rank histograms (U: under-dispersed, ∩: over-dispersed, slope: bias)')
    ax.legend(ncol=3)
    k = np.arange(1, diagnostics['spectra'].shape[-1]+1)
    for c, (name, unit) in enumerate(zip(TARGETS, UNITS)):
        ax = fig.add_subplot(grid[1+c//3, (c % 3)*2:(c % 3)*2+2])
        for row, label, style in ((0, 'Truth', dict(color='k', lw=2.2)), (1, 'Flow member', dict(color='#d73027', lw=1.8)),
                                  (3, 'Frozen regression', dict(color='#4575b4', lw=1.3, ls='--')),
                                  (2, 'Coarse input', dict(color='#999999', lw=1.3, ls=':'))):
            ax.loglog(k, np.maximum(diagnostics['spectra'][row, c], 1e-20), label=label, **style)
        ax.set(title=f'{name}: radial power spectrum', xlabel='Wavenumber (cycles per 128 px)',
               ylabel=f'Power ({unit})²')
        if c == 0:
            ax.legend()
    fig.suptitle(f'v4.1 · epoch {epoch} · validation diagnostics', fontsize=13)
    fig.savefig(path, dpi=100)
    plt.close(fig)


def plot_history(history, path, plt):
    rows = [r for r in history if 'crps' in r]
    epochs = [r['epoch'] for r in history]
    fig, axes = plt.subplots(3, 4, figsize=(20, 12), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(epochs, [r['training_loss'] for r in history], color='k')
    ax.set(title='Training flow objective', xlabel='Epoch', yscale='log')
    twin = ax.twinx()
    twin.plot(epochs, [r['learning_rate'] for r in history], color='#d73027', lw=1, alpha=.7)
    twin.set_ylabel('Learning rate', color='#d73027')
    ax = axes[0, 1]
    for name in TARGETS:
        ax.plot([r['epoch'] for r in rows], [r[f'{name}_crps']/max(r[f'{name}_coarse_mae'], 1e-12) for r in rows],
                marker='o', ms=3, label=name)
    ax.axhline(1, color='k', lw=.8, ls='--')
    ax.set(title='CRPS / coarse MAE (below 1 beats the coarse input)', xlabel='Epoch')
    ax.legend(ncol=2)
    ax = axes[0, 2]
    for key, label in (('wet_fraction', 'Flow wet fraction'), ('truth_wet_fraction', 'Observed wet fraction'),
                       ('false_wet_probability', 'P(false wet)'), ('missed_wet_probability', 'P(missed wet)')):
        ax.plot([r['epoch'] for r in rows], [r[key] if r[key] is not None else np.nan for r in rows],
                marker='o', ms=3, label=label)
    ax.set(title='Rain occurrence (≥ 0.1 mm/h)', xlabel='Epoch', ylim=(0, None))
    ax.legend()
    ax = axes[0, 3]
    ax.plot(epochs, [r.get('samples_per_s', np.nan) for r in history], color='#1a9850')
    ax.set(title='Training throughput', xlabel='Epoch', ylabel='samples/s (all GPUs)', ylim=(0, None))
    for ax, (name, unit) in zip(list(axes[1:].flat), zip(TARGETS, UNITS)):
        x = [r['epoch'] for r in rows]
        for metric, label, style in (('crps', 'Flow CRPS', dict(color='#d73027', lw=2)),
                                     ('ensemble_mean_rmse', 'Ens-mean RMSE', dict(color='#fc8d59')),
                                     ('spread', 'Spread', dict(color='#fc8d59', ls=':')),
                                     ('regression_mae', 'Regression MAE', dict(color='#4575b4', ls='--')),
                                     ('coarse_mae', 'Coarse MAE', dict(color='#999999', ls=':'))):
            ax.plot(x, [r[f'{name}_{metric}'] for r in rows], marker='o', ms=2.5, label=label, **style)
        ax.set(title=name, ylabel=unit, xlabel='Epoch')
    axes[1, 0].legend()
    ax = axes[2, 2]
    for name in TARGETS:
        ax.plot([r['epoch'] for r in rows],
                [r[f'{name}_spread']/max(r[f'{name}_ensemble_mean_rmse'], 1e-12) for r in rows],
                marker='o', ms=3, label=name)
    ax.axhline(1, color='k', lw=.8, ls='--')
    ax.set(title='Spread / ensemble-mean RMSE (≈1 calibrated)', xlabel='Epoch')
    ax.legend(ncol=2)
    ax = axes[2, 3]
    ax.axis('off')
    if rows:
        last = rows[-1]
        text = [f'Epoch {last["epoch"]}', '']
        text += [f'{n:6s} CRPS {last[f"{n}_crps"]:.4g}  coarse {last[f"{n}_coarse_mae"]:.4g}  '
                 f'regr {last[f"{n}_regression_mae"]:.4g}' for n in TARGETS]
        ax.text(0, 1, '\n'.join(text), va='top', family='monospace', fontsize=9)
    fig.suptitle('v4.1 training and validation history', fontsize=13)
    fig.savefig(path, dpi=100)
    plt.close(fig)


def save_plots(items, diagnostics, history, out):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    _style(plt)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    epoch = history[-1]['epoch']
    np.savez_compressed(out/'samples.npz', **{f'patch_{i}_{k}': np.asarray(v) for i, item in enumerate(items)
                                                for k, v in item.items()},
                        **{f'diagnostics_{k}': v for k, v in (diagnostics or {}).items()})
    for i, item in enumerate(items):
        plot_rain(item, epoch, out/f'rain_patch_{i+1}.png', plt)
        plot_states(item, epoch, out/f'states_patch_{i+1}.png', plt)
    if diagnostics:
        plot_diagnostics(diagnostics, epoch, out/'diagnostics.png', plt)
    plot_history(history, out/'history.png', plt)
    (out/'metrics.json').write_text(json.dumps(history[-1], indent=2)+'\n')


EXPECTED = ('rain_patch_1.png', 'states_patch_1.png', 'diagnostics.png', 'history.png', 'metrics.json')
