"""One-member, full-domain validation comparisons during flow training."""
from pathlib import Path
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .inference_v2 import sample_frame_v2
from .physics_v2 import TARGETS_V2, UNITS_V2


def plot_flow_progress_v2(cfg, archive, mean_model, flow_model, flow_scale,
                          device, completed_epoch, run_dir):
    entry = next((e for e in archive.index['entries'] if e['split'] == 'val'), None)
    if entry is None:
        raise ValueError('A validation timestamp is required for flow progress plots')
    # Reuse the production tile/blend/ODE path, with a fixed one-member seed so
    # epochs are visually comparable. No prediction files are written.
    seed = int(np.random.SeedSequence([
        cfg['inference']['seed'], int(entry['id'].replace('_', '')), 0
    ]).generate_state(1)[0])
    prediction, regression, _ = sample_frame_v2(
        mean_model, flow_model, flow_scale, archive, entry, cfg, device, seed)
    baseline = np.asarray(archive.array(entry, 'baseline'))
    truth = np.asarray(archive.array(entry, 'truth'))
    area = archive.static['area']
    fields = (baseline, regression, prediction, truth)
    labels = ('Low-resolution input', 'Regression', 'Flow member 0', 'HWT reference', 'Flow − HWT')
    fig, axes = plt.subplots(5, 5, figsize=(20, 16), constrained_layout=True)
    try:
        for i, name in enumerate(TARGETS_V2):
            lo, hi = np.quantile(np.stack([f[i] for f in fields]), [.01, .99])
            if hi <= lo:
                hi = lo+1
            difference = prediction[i]-truth[i]
            bound = max(float(np.quantile(np.abs(difference), .99)), 1e-4)
            for j, field in enumerate((*[f[i] for f in fields], difference)):
                ax = axes[i, j]
                im = ax.imshow(field, origin='lower', cmap='RdBu_r' if j == 4 else
                               ('YlGnBu' if i == 1 else 'viridis'),
                               vmin=-bound if j == 4 else lo, vmax=bound if j == 4 else hi)
                ax.set(title=f'{name} · {labels[j]}', xticks=[], yticks=[])
                if j in (0, 1, 2):
                    rmse = np.sqrt(np.sum((field-truth[i])**2*area)/np.sum(area))
                    ax.text(.02, .03, f'RMSE {rmse:.3g} {UNITS_V2[i]}', transform=ax.transAxes,
                            fontsize=8, bbox=dict(facecolor='white', alpha=.8, edgecolor='none'))
                fig.colorbar(im, ax=ax, shrink=.62, label=UNITS_V2[i])
        fig.suptitle(f'V2 flow · epoch {completed_epoch} · {entry["time"]} · one full-domain member')
        dest_dir = Path(run_dir)/'plots_v2'
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir/f'epoch_{completed_epoch:04d}_{entry["id"]}_v2.png'
        tmp = dest.with_name(dest.name+'.tmp')
        fig.savefig(tmp, format='png', dpi=110)
        os.replace(tmp, dest)
    finally:
        plt.close(fig)
    return dest
