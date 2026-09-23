"""Make the wet-evaluation panels for one whole-domain direct-rainfall case."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from .dataset_v2 import ArchiveV2, crop_v2
from .evaluate_wet_precip_direct_v2 import patch_metrics
from .inference import starts, blend_window
from .inference_precip_direct_v2 import sample_frame
from .precip_direct_v2 import (FrozenRegression, check_checkpoint, decode_rain,
                               load_config, make_model)
from .train import autocast, device_for
from .train_v2 import file_hash_v2


DEFAULT_TIMESTAMP = '20260223_0530'


def select_entry(archive, timestamp, split):
    entries = [entry for entry in archive.index['entries'] if entry['split'] == split
               and timestamp in (entry['id'], entry['time'])]
    if len(entries) != 1:
        raise ValueError(f'Expected one {split} entry for {timestamp!r}, found {len(entries)}')
    return entries[0]


@torch.no_grad()
def frozen_regression_frame(conditioner, archive, entry, cfg, device):
    """Blend the encoded frozen-v2 condition over all full-domain tile cores."""
    if conditioner is None:
        return None
    p = cfg['patch']
    h, w = archive.shape
    size, halo = p['size'], p['halo']
    window = blend_window(size)
    total = np.zeros((h, w), dtype='float32')
    weight = np.zeros((h, w), dtype='float32')
    baseline = archive.array(entry, 'baseline')[1:2]
    for y in starts(h, size, p['stride']):
        for x in starts(w, size, p['stride']):
            condition, context = archive.inputs(entry, y, x, p)
            coarse = crop_v2(baseline, y, x, size, halo).copy()
            batch = dict(condition=condition[None].to(device),
                         context=context[None].to(device),
                         coarse=torch.from_numpy(coarse[None]).to(device))
            with autocast(device, cfg['train']['precision']):
                encoded = conditioner(batch)
            core = encoded[0, 0, halo:halo+size, halo:halo+size].float().cpu().numpy()
            total[y:y+size, x:x+size] += core*window
            weight[y:y+size, x:x+size] += window
    if np.any(weight <= 0):
        raise ValueError('Frozen regression tiling left uncovered pixels')
    return decode_rain(total/weight, archive.stats['precip_log_scale'])


def plot_full(path, entry, truth, coarse, regression, ensemble, metrics, epoch):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    mean, spread = ensemble.mean(0), ensemble.std(0)
    fields = [('HWT truth', truth), ('Coarse input', coarse)]
    if regression is not None:
        fields.append(('Frozen v2 input', regression))
    fields += [('Member 1', ensemble[0]), ('Member 2', ensemble[1]),
               ('Ensemble mean', mean), ('Ensemble spread', spread)]
    vmax = max(1., float(np.quantile(np.stack([truth, coarse, mean]), .995)))
    fig, axes = plt.subplots(2, 4, figsize=(24, 10), constrained_layout=True)
    try:
        for ax, (label, field) in zip(axes.flat, fields):
            image = ax.imshow(field, origin='lower', cmap='Blues', vmin=0, vmax=vmax,
                              interpolation='nearest', rasterized=True)
            ax.set(title=label, xticks=[], yticks=[])
        for ax in list(axes.flat)[len(fields):]:
            ax.axis('off')
        fig.colorbar(image, ax=axes.ravel().tolist(), label='mm/h; common scale',
                     extend='max', shrink=.8)
        fig.suptitle(f'Full CONUS direct precipitation · {entry["time"]} UTC · epoch {epoch} · '
                     f'wet truth {metrics["truth_wet_fraction"]:.1%} · '
                     f'generated {metrics["generated_wet_fraction"]:.1%} · '
                     f'CRPS {metrics["crps_mm_h"]:.3f} vs coarse MAE '
                     f'{metrics["coarse_mae_mm_h"]:.3f} mm/h')
        fig.savefig(path, dpi=140)
    finally:
        plt.close(fig)


def save_npy(path, value):
    temporary = Path(str(path)+'.tmp')
    with temporary.open('wb') as stream:
        np.save(stream, value)
    os.replace(temporary, path)


def load_saved_field(path, shape):
    value = np.load(path, mmap_mode='r')
    if value.shape != shape or not np.isfinite(value).all() or np.any(value < 0):
        raise ValueError(f'Invalid saved whole-domain field: {path}')
    return np.asarray(value, dtype='float32')


@torch.no_grad()
def evaluate(cfg, checkpoint, output, timestamp=DEFAULT_TIMESTAMP, split='test',
             members=2, steps=None):
    if members < 2 or steps is not None and steps < 1:
        raise ValueError('Need at least two members and positive steps')
    archive = ArchiveV2(cfg['data']['prepared'])
    entry = select_entry(archive, timestamp, split)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    check_checkpoint(saved, cfg, archive)
    local_cfg = dict(cfg, inference=dict(cfg['inference']))
    if steps is not None:
        local_cfg['inference']['steps'] = steps
    digest = file_hash_v2(checkpoint)
    seeds = [int(np.random.SeedSequence([cfg['inference']['seed'],
             int(entry['id'].replace('_', '')), member]).generate_state(1)[0])
             for member in range(members)]
    manifest = dict(id=entry['id'], time=entry['time'], split=split,
                    checkpoint_sha256=digest, checkpoint_epoch=saved['epoch']+1,
                    archive_fingerprint=archive.index['fingerprint'], members=members,
                    steps=local_cfg['inference']['steps'], seeds=seeds)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out/'manifest.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError('Existing whole-domain output has different checkpoint or settings')
    elif any(out.iterdir()):
        raise FileExistsError(f'Output is nonempty without a matching manifest: {out}')
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    device = device_for(cfg['train']['device'])
    if device.type != 'cuda':
        raise RuntimeError('Whole-domain direct-flow sampling requires a visible GPU')
    model = make_model(archive.index['condition_channels'], cfg).to(device).eval()
    model.load_state_dict(saved['ema'])
    bundle = saved.get('regression_condition')
    conditioner = FrozenRegression(bundle, archive.index['condition_channels'],
                                   archive.stats).to(device) if bundle else None
    print(f'Full CONUS {entry["time"]} UTC; epoch {saved["epoch"]+1}; '
          f'{members} members; {manifest["steps"]} steps; device={device}', flush=True)
    regression_path = out/'frozen_v2_input.npy'
    regression = None
    if conditioner is not None:
        if not regression_path.exists():
            save_npy(regression_path, frozen_regression_frame(
                conditioner, archive, entry, local_cfg, device))
        regression = load_saved_field(regression_path, archive.shape)
    generated = []
    for member, seed in enumerate(seeds):
        path = out/f'member_{member+1:02d}.npy'
        if not path.exists():
            save_npy(path, sample_frame(model, conditioner, archive, entry,
                                        local_cfg, device, seed))
        generated.append(load_saved_field(path, archive.shape))
        print(f'Member {member+1}/{members} ready: {path}', flush=True)
    ensemble = np.stack(generated)
    truth = np.asarray(archive.array(entry, 'truth')[1], dtype='float32')
    coarse = np.asarray(archive.array(entry, 'baseline')[1], dtype='float32')
    area = archive.static['area']
    metrics = patch_metrics(ensemble, truth, coarse, regression, area)
    report = dict(**manifest, checkpoint=str(Path(checkpoint).resolve()),
                  metrics=metrics, target_kind='HWT midpoint PRECTOT',
                  caveat='One test hour is a case study, not population test skill. '
                         'The direct model uses full rainfall as its target.')
    (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    image = out/f'{entry["id"]}_full_conus_direct_v2.png'
    plot_full(image, entry, truth, coarse, regression, ensemble,
              metrics, saved['epoch']+1)
    print(f'Wrote {image}', flush=True)
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/discover_precip_direct_v2.yaml')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--timestamp', default=DEFAULT_TIMESTAMP)
    parser.add_argument('--split', choices=('val', 'test'), default='test')
    parser.add_argument('--members', type=int, default=2)
    parser.add_argument('--steps', type=int)
    args = parser.parse_args()
    print(evaluate(load_config(args.config), args.checkpoint, args.output,
                   args.timestamp, args.split, args.members, args.steps))


if __name__ == '__main__':
    main()
