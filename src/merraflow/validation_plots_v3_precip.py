"""Rank-zero, fixed-patch generated rainfall diagnostics during training."""
import json
import numpy as np
from .metrics import radial_psd, rank_histogram


def plot_validation(previews, history, output, stage, epoch, cfg):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    if not previews:
        raise ValueError('Validation plotting needs at least one rank-zero patch')
    output = output/f'epoch_{epoch:04d}'
    output.mkdir(parents=True, exist_ok=True)
    labels = ['HWT truth', 'Coarse', 'Regression', 'Member 1', 'Ensemble mean', 'Ensemble spread']
    fig, axes = plt.subplots(len(previews), 6, figsize=(20, 3.5*len(previews)), squeeze=False, constrained_layout=True)
    arrays = {}
    for i, item in enumerate(previews):
        truth, coarse, regression, ensemble = [item[k] for k in ('truth', 'coarse', 'regression', 'ensemble')]
        # Fixed truth/coarse scale across epochs exposes both missing extremes
        # and overprediction, with saturation explicitly indicated on the bar.
        vmax = max(float(np.quantile(truth, .995)), float(np.quantile(coarse, .995)), 1.)
        for ax, label, value in zip(axes[i], labels, [truth, coarse, regression, ensemble[0], ensemble.mean(0), ensemble.std(0)]):
            artist = ax.imshow(value, origin='lower', cmap='Blues', vmin=0, vmax=vmax)
            ax.set_title(f'Patch {i+1}: {label}')
            ax.set_xticks([])
            ax.set_yticks([])
        fig.colorbar(artist, ax=axes[i].tolist(), label='mm/h (fixed truth/coarse scale)', extend='max')
        arrays.update({f'patch_{i}_{key}': value for key, value in item.items()})
    fig.suptitle(f'{stage} EMA — epoch {epoch} — fixed validation patches; {cfg["data"]["target_kind"]}')
    fig.savefig(output/'fields.png', dpi=140)
    plt.close(fig)
    np.savez_compressed(output/'samples.npz', **arrays)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4), constrained_layout=True)
    styles = [('truth', 'Truth'), ('coarse', 'Coarse'), ('regression', 'Regression'), ('ensemble', 'Generated members')]
    for key, label in styles:
        fields = [member for item in previews for member in (item[key] if key == 'ensemble' else item[key][None])]
        spectra = [radial_psd(field) for field in fields]
        axes[0].loglog(spectra[0][0], np.maximum(np.mean([x[1] for x in spectra], axis=0), 1e-12), label=label)
        thresholds = np.geomspace(.1, 100, 80)
        rates = np.concatenate([field.ravel() for field in fields])
        axes[1].loglog(thresholds, [np.mean(rates >= t) for t in thresholds], label=label)
    axes[0].set(title='Member spatial spectra', xlabel='Cycles / grid pixel', ylabel='PSD')
    axes[1].set(title='Rain-rate exceedances', xlabel='Threshold (mm/h)', ylabel='Pixel fraction')
    axes[0].legend(fontsize=8)
    ranks = np.mean([rank_histogram(x['ensemble'], x['truth'], x['area']) for x in previews], axis=0)
    axes[2].bar(np.arange(len(ranks)), ranks)
    axes[2].axhline(1/len(ranks), color='black', linestyle='--')
    axes[2].set(title='Ranks (ties randomized)', xlabel='Rank', ylabel='Area fraction')
    fig.suptitle(f'{stage}, epoch {epoch}: diagnostics of {len(previews)} preview patches only')
    fig.savefig(output/'diagnostics.png', dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    axes[0].plot([x['epoch'] for x in history], [x['training_loss'] for x in history])
    validated = [x for x in history if 'crps' in x]  # Generated validation may be interval-gated.
    epochs = [x['epoch'] for x in validated]
    axes[0].set(title='Training objective', xlabel='Epoch', ylabel='Loss')
    for key, label in [('crps', 'Generated'), ('coarse_crps', 'Coarse'), ('regression_crps', 'Regression')]:
        axes[1].plot(epochs, [x[key] for x in validated], label=label, marker='.')
    axes[1].set(title='Global validation CRPS', xlabel='Epoch', ylabel='mm/h')
    axes[1].legend()
    for key in ('bias', 'spread'):
        axes[2].plot(epochs, [x[key] for x in validated], label=key, marker='.')
    axes[2].set(title='Global validation bias and spread', xlabel='Epoch', ylabel='mm/h')
    axes[2].legend()
    fig.savefig(output/'history.png', dpi=140)
    plt.close(fig)
    (output/'metadata.json').write_text(json.dumps(dict(stage=stage, epoch=epoch,
        target_kind=cfg['data']['target_kind'], patches=len(previews), members=len(previews[0]['ensemble']),
        validation_noise_seed=cfg['train']['seed']+7103,
        sampling_steps=cfg['train']['validation_steps'], global_validation=history[-1],
        scope='Fields and spatial diagnostics use fixed rank-zero validation patches; history metrics combine all ranks.'), indent=2)+'\n')
