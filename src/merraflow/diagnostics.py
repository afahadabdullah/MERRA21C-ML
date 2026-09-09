"""Headless plots on original LCC pixel axes; no geographic projection is invented."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from matplotlib.ticker import NullFormatter
from . import TARGETS, UNITS
from .dataset import Archive
from .evaluate import load_members
from .metrics import radial_psd, rank_histogram


def diagnostics(cfg, predictions=None, output=None, timestamp=None):
    a = Archive(cfg['data']['prepared'])
    root = Path(predictions or cfg['inference']['output'])
    out = Path(output or root/'diagnostics')
    out.mkdir(parents=True, exist_ok=True)
    entries = [e for e in a.index['entries'] if sorted(root.glob(f'{e["id"]}_m*.nc')) and (timestamp is None or timestamp in (e['id'], e['time']))]
    if not entries:
        raise ValueError('No matching predictions to plot')
    e = entries[0]
    ensemble, audits = load_members(sorted(root.glob(f'{e["id"]}_m*.nc')), a, e)
    truth, baseline = a.array(e, 'truth'), a.array(e, 'baseline')
    mean, spread = ensemble.mean(0), ensemble.std(0)
    fig, axes = plt.subplots(4, 5, figsize=(19, 13), constrained_layout=True)
    for i, (name, unit) in enumerate(zip(TARGETS, UNITS)):
        vmin, vmax = np.quantile(np.concatenate([truth[i].ravel(), mean[i].ravel(), baseline[i].ravel()]), [.01, .99])
        if vmax <= vmin:
            vmax = vmin+1
        values = [baseline[i], truth[i], ensemble[0, i], mean[i], mean[i]-truth[i]]
        for j, (value, title) in enumerate(zip(values, ['Coarse baseline', 'Original HR', 'Generated member 0', 'Ensemble mean', 'Mean − original HR'])):
            options = dict(origin='lower', interpolation='nearest', aspect='auto')
            if j == 4:
                lim = max(float(np.quantile(abs(value), .99)), 1e-5)
                options.update(cmap='RdBu_r', vmin=-lim, vmax=lim)
            elif i == 1:
                options.update(cmap='Blues', norm=SymLogNorm(linthresh=.1, vmin=0, vmax=max(vmax, .1)))
            else:
                options.update(cmap='viridis', vmin=vmin, vmax=vmax)
            im = axes[i, j].imshow(value, **options)
            axes[i, j].set_title(f'{name} · {title}', fontsize=10)
            axes[i, j].set_xlabel('LCC X pixel')
            axes[i, j].set_ylabel('LCC Y pixel')
            fig.colorbar(im, ax=axes[i, j], shrink=.75, label=unit)
    label = ' | Synthetic software test' if a.index['data_config'].get('synthetic', False) else ''
    fig.suptitle(f'{e["time"]} UTC{label}', fontsize=15)
    fig.savefig(out/'fields.png', dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(2, 4, figsize=(17, 8), constrained_layout=True)
    for i, name in enumerate(TARGETS):
        im = axes[0, i].imshow(spread[i], origin='lower', aspect='auto', cmap='magma')
        axes[0, i].set_title(f'{name}: ensemble standard deviation')
        fig.colorbar(im, ax=axes[0, i], label=UNITS[i], shrink=.7)
        freq, obs = radial_psd(truth[i])
        _, base = radial_psd(baseline[i])
        member_psd = np.stack([radial_psd(m[i])[1] for m in ensemble])
        axes[1, i].loglog(freq, np.maximum(obs, 1e-12), label='Original HR')
        axes[1, i].loglog(freq, np.maximum(base, 1e-12), label='Baseline')
        axes[1, i].loglog(freq, np.maximum(member_psd.mean(0), 1e-12), label='Member PSD mean')
        axes[1, i].set(xlabel='Cycles per LCC grid pixel', ylabel=f'Power ({UNITS[i]})²', title=f'{name}: radial spectrum')
        axes[1, i].xaxis.set_minor_formatter(NullFormatter())
        axes[1, i].legend(fontsize=8)
    fig.savefig(out/'spread_spectra.png', dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    for i, name in enumerate(TARGETS):
        ax = axes.ravel()[i]
        hist = rank_histogram(ensemble[:, i], truth[i], a.static['area'])
        ax.bar(np.arange(len(hist)), hist, color='#347c98')
        ax.axhline(1/len(hist), color='black', linestyle='--')
        ax.set(title=f'{name}: area-weighted ranks (ties randomized)', xlabel='Observation rank', ylabel='Fraction')
    ax = axes[1, 1]
    positive = [x[x > 0] for x in [truth[1], ensemble[:, 1], baseline[1]]]
    upper = max([float(x.max()) for x in positive if x.size]+[1.])
    bins = np.geomspace(.001, max(upper, .01), 50)
    for value, label in zip(positive, ['Original HR', 'Members pooled', 'Baseline']):
        if value.size:
            ax.hist(value, bins=bins, density=True, histtype='step', label=label)
    ax.set(xscale='log', yscale='log', xlabel='Wet precipitation (mm/hour)', ylabel='Conditional wet density', title='Precipitation positive tail')
    ax.legend()
    ax = axes[1, 2]
    ax.bar(np.arange(len(audits)), [v['projection_mae_mm_h'] for v in audits])
    ax.set(xlabel='Member', ylabel='Mean absolute change (mm/hour)', title='Effect of conservation projection')
    fig.savefig(out/'ranks_precip_mass.png', dpi=140)
    plt.close(fig)
    history = Path(cfg['train']['output'])/'history.jsonl'
    if history.exists():
        rows = [json.loads(line) for line in history.read_text().splitlines() if line.strip()]
        fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
        ax.plot([r['epoch'] for r in rows], [r['train_loss'] for r in rows], label='Train')
        ax.plot([r['epoch'] for r in rows], [r['val_loss'] for r in rows], label='Validation (EMA)')
        ax.set(xlabel='Epoch', ylabel='Area-weighted flow loss', title='Training history')
        ax.legend()
        fig.savefig(out/'training.png', dpi=140)
        plt.close(fig)
    metrics_path = root/'evaluation'/'per_hour.json'
    if metrics_path.exists():
        rows = json.loads(metrics_path.read_text())
        row = next((r for r in rows if r['id'] == e['id']), None)
        if row:
            fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
            for threshold in ('0.1', '1', '5', '10', '25'):
                scores = row['precipitation'][threshold]
                fss = scores['member_fss_mean']
                axes[0].plot([int(s)*3 for s in fss], [np.nan if v is None else v for v in fss.values()], marker='o', label=f'{threshold} mm/h')
                bins = [b for b in scores['reliability'] if b['count']]
                axes[1].plot([b['forecast_probability'] for b in bins], [b['observed_frequency'] for b in bins], marker='o', label=f'{threshold} mm/h')
            axes[0].set(xlabel='Nominal neighborhood width (km; 3 km/pixel)', ylabel='Member FSS mean', ylim=(0, 1), title='Precipitation spatial skill')
            axes[1].plot([0, 1], [0, 1], '--', color='black')
            axes[1].set(xlabel='Forecast probability', ylabel='Observed frequency', title='Precipitation reliability')
            for ax in axes:
                ax.legend(fontsize=8)
            fig.savefig(out/'precip_skill.png', dpi=140)
            plt.close(fig)
    return out
