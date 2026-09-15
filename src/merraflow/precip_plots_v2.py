"""Headless figures for saved-field rainfall audits."""
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter


COLORS = {'HWT': 'black', 'coarse': '#8497a6', 'regression': '#29915c', 'flow_mean': '#d37720'}


def finish(fig, path, title):
    fig.suptitle(title)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_case(out, report, ensemble, regression, baseline, truth):
    stamp = report['id']
    title = f'Precipitation audit · {report["split"]} · {report["time"]} UTC · {len(ensemble)} members'
    fields = {'HWT': truth, 'coarse': baseline, 'regression': regression, 'flow_mean': ensemble.mean(0)}
    fields.update({f'member_{i:03d}': member for i, member in enumerate(ensemble)})
    vmax = max(1e-6, max(float(v.max()) for v in fields.values()))
    fig, axes = plt.subplots(math.ceil(len(fields)/3), 3, figsize=(15, 3.4*math.ceil(len(fields)/3)),
                             squeeze=False, constrained_layout=True)
    for ax, (name, values) in zip(axes.flat, fields.items()):
        im = ax.imshow(values, origin='lower', interpolation='nearest', cmap='YlGnBu', vmin=0, vmax=vmax)
        profile = report['profiles'][name]
        ax.set(title=f'{name} · mean {profile["mean_mm_h"]:.3g} · max {profile["max_mm_h"]:.3g}', xticks=[], yticks=[])
        fig.colorbar(im, ax=ax, shrink=.7, label='mm h⁻¹ (linear, full range)')
    for ax in list(axes.flat)[len(fields):]:
        ax.axis('off')
    finish(fig, out/f'{stamp}_members_v2.png', title)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    errors = [regression-truth, fields['flow_mean']-truth]
    bound = max(1e-6, max(float(abs(x).max()) for x in errors))
    for ax, value, label in zip(axes, [*errors, ensemble.std(0)], ['Regression − HWT', 'Flow mean − HWT', 'Member standard deviation']):
        spread = label.startswith('Member')
        im = ax.imshow(value, origin='lower', interpolation='nearest', cmap='magma' if spread else 'RdBu_r',
                       vmin=0 if spread else -bound, vmax=max(float(value.max()), 1e-6) if spread else bound)
        ax.set(title=label, xticks=[], yticks=[])
        fig.colorbar(im, ax=ax, shrink=.7, label='mm h⁻¹')
    finish(fig, out/f'{stamp}_errors_v2.png', title)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    for name, color in COLORS.items():
        profile = report['profiles'][name]
        exc = profile['exceedance']
        axes[0, 0].loglog(exc['threshold_mm_h'], np.maximum(exc['area_fraction'], 1e-7), label=name, color=color)
        q = profile['area_quantiles_mm_h']
        axes[0, 1].plot(['p95', 'p99', 'p999', 'max'], [q['p95'], q['p99'], q['p999'], profile['max_mm_h']],
                        marker='o', color=color, label=name)
        if name != 'HWT':
            factors = list(report['coarsened'])
            axes[1, 0].plot([int(f) for f in factors], [report['coarsened'][f][name]['rmse'] for f in factors],
                            marker='o', color=color, label=name)
        axes[1, 1].loglog(report['spectra']['cycles_per_pixel'], np.maximum(report['spectra'][name], 1e-15), color=color, label=name)
    for name, profile in report['profiles'].items():
        if name.startswith('member_'):
            exc = profile['exceedance']
            axes[0, 0].loglog(exc['threshold_mm_h'], np.maximum(exc['area_fraction'], 1e-7), color='#b562ba', alpha=.3, linewidth=.7)
    names = list(COLORS)
    axes[0, 2].bar(names, [report['profiles'][n]['mean_mm_h'] for n in names], color=list(COLORS.values()))
    axes[0, 2].set(title='Domain mean precipitation rate', ylabel='mm h⁻¹')
    axes[0, 2].tick_params(axis='x', rotation=20)
    axes[0, 0].set(title='Exceedance (thin purple: members)', xlabel='Precipitation threshold (mm h⁻¹)', ylabel='Area fraction ≥ threshold', ylim=(1e-6, 1))
    axes[0, 1].set(title='Area-weighted upper quantiles and maximum', ylabel='mm h⁻¹')
    axes[1, 0].set(title='RMSE after area-weighted block averaging', xlabel='Block width (grid pixels)', ylabel='mm h⁻¹')
    axes[1, 0].set_xscale('log', base=2)
    axes[1, 0].set_xticks([int(f) for f in report['coarsened']], list(report['coarsened']))
    axes[1, 1].loglog(report['spectra']['cycles_per_pixel'], np.maximum(report['spectra']['member_average'], 1e-15), label='Mean member PSD', color='#b562ba')
    axes[1, 1].set(title='Hann-tapered rainfall power spectra', xlabel='cycles / grid pixel', ylabel='PSD')
    axes[1, 1].xaxis.set_major_locator(LogLocator(base=10, numticks=4))
    axes[1, 1].xaxis.set_minor_formatter(NullFormatter())
    ranks = report['rank_histogram']
    axes[1, 2].bar(np.arange(len(ranks)), ranks)
    axes[1, 2].axhline(1/len(ranks), color='black', linestyle='--')
    axes[1, 2].set(title='Area-weighted rank histogram (randomized ties)', xlabel='HWT rank among members', ylabel='Area fraction')
    axes[1, 2].set_xticks(np.arange(len(ranks)))
    for ax in (axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]):
        ax.legend(fontsize=8)
    finish(fig, out/f'{stamp}_distribution_structure_v2.png', title)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    for ax, threshold in zip(axes[0, :2], ('1', '5')):
        for label, color in [('coarse', COLORS['coarse']), ('regression', COLORS['regression']), ('flow', COLORS['flow_mean'])]:
            record = report['precipitation_skill'][label][threshold]
            fss = record['ensemble_mean_fss']
            ax.plot([int(s) for s in fss], [np.nan if v is None else v for v in fss.values()], marker='o', color=color, label=label)
        member = report['precipitation_skill']['flow'][threshold]['member_fss_mean']
        ax.plot([int(s) for s in member], [np.nan if v is None else v for v in member.values()], linestyle='--', label='Mean member FSS')
        ax.set(title=f'Fractions skill score ≥ {threshold} mm h⁻¹', xlabel='Neighborhood width (pixels)', ylim=(0, 1))
        ax.legend(fontsize=8)
    axes[0, 2].plot([0, 1], [0, 1], '--', color='black')
    for threshold in ('0.1', '1', '5'):
        bins = report['precipitation_skill']['flow'][threshold]['reliability']
        bins = [b for b in bins if b['area_m2'] > 0]
        axes[0, 2].plot([b['forecast_probability'] for b in bins], [b['observed_frequency'] for b in bins], marker='o', label=f'≥ {threshold}')
    axes[0, 2].set(title='Reliability (per-case, occupied bins only)', xlabel='Predicted probability', ylabel='Observed area frequency', xlim=(0, 1), ylim=(0, 1))
    axes[0, 2].legend(fontsize=8)
    levels = ['0.1', '1.0', '5.0', '10.0', '25.0']
    for name, color in COLORS.items():
        axes[1, 0].plot(levels, [report['profiles'][name]['wet_area_fraction'][x] for x in levels], marker='o', label=name, color=color)
    axes[1, 0].set(title='Wet-area fractions', xlabel='Threshold (mm h⁻¹)', ylabel='Domain area fraction')
    axes[1, 0].legend(fontsize=8)
    amplitude = report['postdecode_log_residual_shrinkage']
    for score in ('rmse', 'crps'):
        axes[1, 1].plot([float(a) for a in amplitude], [r['scores'][score] for r in amplitude.values()], marker='o', label=score)
    axes[1, 1].set(title='Output-only log residual shrinkage', xlabel='α (0: regression, 1: saved flow)', ylabel='mm h⁻¹')
    axes[1, 1].legend()
    text = [f'Ensemble CRPS: {report["probabilistic"]["crps"]:.3g} mm h⁻¹',
            f'Ensemble spread: {report["probabilistic"]["spread"]:.3g} mm h⁻¹',
            f'5–95% sample interval coverage: {report["probabilistic"]["coverage_90"]:.1%}',
            '', 'Offset tests and tile-region scores are in JSON.',
            'Offset/amplitude sweeps are diagnostic only.', 'Use validation cases for model changes.']
    axes[1, 2].text(.02, .9, '\n'.join(text), va='top', fontsize=10)
    axes[1, 2].axis('off')
    finish(fig, out/f'{stamp}_skill_calibration_v2.png', title)
