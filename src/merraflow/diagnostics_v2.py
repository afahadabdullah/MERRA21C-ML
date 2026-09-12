"""Plot actual generated members, the learned baseline, spectra and losses."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .dataset_v2 import ArchiveV2
from .evaluate_v2 import load_members_v2, evaluate_v2
from .physics_v2 import TARGETS_V2, UNITS_V2
from .metrics import continuous


def diagnostics_v2(cfg, split='val', timestamp=None):
    archive = ArchiveV2(cfg['data']['prepared'])
    root = Path(cfg['inference']['output'])
    out = root/'diagnostics_v2'
    out.mkdir(parents=True, exist_ok=True)
    entries = [e for e in archive.index['entries'] if e['split'] == split and
               (timestamp is None or timestamp in (e['id'], e['time'])) and list(root.glob(f'{e["id"]}_m*_v2.nc'))]
    if not entries:
        raise ValueError('No matching v2 prediction')
    entry = entries[0]
    ensemble, regression, _, identity = load_members_v2(sorted(root.glob(f'{entry["id"]}_m*_v2.nc')), archive, entry)
    truth, baseline = [np.asarray(archive.array(entry, k)) for k in ('truth', 'baseline')]
    area = archive.static['area']
    fig, axes = plt.subplots(5, 6, figsize=(23, 15), constrained_layout=True)
    for i, name in enumerate(TARGETS_V2):
        values = [baseline[i], truth[i], regression[i], ensemble[0, i], ensemble[:, i].mean(0), ensemble[:, i].mean(0)-truth[i]]
        lo, hi = np.quantile(np.stack(values[:5]), [.01, .99])
        if hi <= lo:
            hi = lo+1
        for j, (label, value) in enumerate(zip(('Coarse', 'Original HWT', 'Regression', 'Member 0', 'Ensemble mean', 'Mean − HWT'), values)):
            ax = axes[i, j]
            if j == 5:
                bound = max(float(np.quantile(abs(value), .99)), 1e-4)
                im = ax.imshow(value, origin='lower', cmap='RdBu_r', vmin=-bound, vmax=bound)
            else:
                im = ax.imshow(value, origin='lower', cmap='YlGnBu' if i == 1 else 'viridis', vmin=lo, vmax=hi)
            land = archive.static['land_fraction']
            if land.min() < .5 < land.max():
                ax.contour(land, levels=[.5], colors='black', linewidths=.25)
            ax.set_title(f'{name} · {label}')
            ax.set_xticks([])
            ax.set_yticks([])
            if j in (0, 2, 3, 4):
                score = continuous(value[None], truth[i], area)['rmse']
                ax.text(.02, .03, f'RMSE {score:.3g} {UNITS_V2[i]}', transform=ax.transAxes,
                        fontsize=8, bbox=dict(facecolor='white', alpha=.8, edgecolor='none'))
            fig.colorbar(im, ax=ax, shrink=.65, label=UNITS_V2[i])
    fig.suptitle(f'V2 · {entry["time"]} · {len(ensemble)} members · checkpoint {identity[0][:12]}')
    fig.savefig(out/f'fields_{entry["id"]}_v2.png', dpi=130)
    plt.close(fig)
    report_dir = evaluate_v2(cfg, split)
    reports = json.loads((report_dir/'per_hour_v2.json').read_text())
    record = next(r for r in reports if r['id'] == entry['id'])
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for ax, (name, spectra) in zip(axes.ravel(), record['spectra'].items()):
        for key in ('truth', 'regression', 'member_mean', 'ensemble_mean'):
            ax.loglog(spectra['cycles_per_pixel'], np.maximum(spectra[key], 1e-15), label=key)
        ax.set(title=name, xlabel='cycles / LCC grid pixel', ylabel='PSD')
        ax.legend(fontsize=8)
    fig.savefig(out/f'spectra_{entry["id"]}_v2.png', dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for ax, stage in zip(axes, ('regression', 'flow')):
        path = Path(cfg['train']['output'])/f'{stage}_v2'/'history_v2.jsonl'
        if path.exists():
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            for split_name in ('train', 'val'):
                ax.plot([r['epoch'] for r in rows], [r[split_name]['total'] for r in rows], label=split_name)
            ax.legend()
        ax.set(title=f'{stage} objective', xlabel='Completed epoch', ylabel='Loss')
    fig.savefig(out/'training_v2.png', dpi=140)
    plt.close(fig)
    return out
