"""Select rainy validation patches and compare direct-flow ensembles in mm/h."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .dataset_v2 import ArchiveV2, box_means_v2, candidates_v2, crop_v2
from .metrics import crps_ensemble
from .precip_direct_v2 import (FrozenRegression, check_checkpoint, decode_rain,
                               load_config, make_model, sample)
from .train import autocast, device_for


def select_wet_cases(archive, cfg, split, scan_hours, cases, min_wet_fraction, threshold=.1):
    """HWT truth selects evaluation cases only; it is never a model input."""
    entries = [e for e in archive.index['entries'] if e['split'] == split]
    if not entries:
        raise ValueError(f'No {split} entries')
    size, stride = cfg['patch']['size'], cfg['patch']['sampling_stride']
    yy, xx = candidates_v2(archive.shape, size, stride)
    indices = np.unique(np.linspace(0, len(entries)-1, min(scan_hours, len(entries)), dtype=int))
    candidates = []
    for index in indices:
        entry = entries[int(index)]
        rain = np.asarray(archive.array(entry, 'truth')[1], dtype='float32')
        wet = box_means_v2(rain >= threshold, yy, xx, size)
        amount = box_means_v2(rain, yy, xx, size)
        # One top patch per hour preserves temporal variety among selected cases.
        eligible = np.flatnonzero(wet >= min_wet_fraction)
        if not len(eligible):
            continue
        best = int(eligible[np.argmax(amount[eligible])])
        candidates.append(dict(entry=entry, y=int(yy[best]), x=int(xx[best]),
                               selection_wet_fraction=float(wet[best]),
                               selection_mean_mm_h=float(amount[best])))
    candidates.sort(key=lambda item: item['selection_mean_mm_h'], reverse=True)
    if len(candidates) < cases:
        raise ValueError(f'Only {len(candidates)} qualifying wet hours in {len(indices)} scanned; '
                         'lower --min-wet-fraction or increase --scan-hours')
    return candidates[:cases], dict(split=split, scanned_hours=len(indices),
        qualifying_hours=len(candidates), cases=cases, selection='highest HWT mean rainfall '
        'among patches with required wet fraction; one case per selected hour',
        wet_threshold_mm_h=threshold, min_wet_fraction=min_wet_fraction)


def patch_metrics(ensemble, truth, coarse, regression, area, threshold=.1):
    weights = area/area.sum()
    def avg(value):
        return float(np.sum(value*weights))
    result = dict(crps_mm_h=avg(crps_ensemble(ensemble, truth)),
                  coarse_mae_mm_h=avg(abs(coarse-truth)),
                  ensemble_mean_rmse_mm_h=avg((ensemble.mean(0)-truth)**2)**.5,
                  mean_bias_mm_h=avg(ensemble.mean(0)-truth),
                  truth_wet_fraction=avg(truth >= threshold),
                  generated_wet_fraction=avg((ensemble >= threshold).mean(0)),
                  coarse_wet_fraction=avg(coarse >= threshold),
                  member_mean_mm_h=avg(ensemble.mean(0)), truth_mean_mm_h=avg(truth))
    if regression is not None:
        result['regression_mae_mm_h'] = avg(abs(regression-truth))
        result['regression_wet_fraction'] = avg(regression >= threshold)
    probability = (ensemble >= threshold).mean(0)
    observed = truth >= threshold
    result['wet_brier'] = avg((probability-observed)**2)
    return result


def plot_case(out, case, truth, coarse, regression, ensemble, metrics):
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
    fig, axes = plt.subplots(1, len(fields), figsize=(3.1*len(fields), 3.6), constrained_layout=True)
    for ax, (label, field) in zip(axes, fields):
        image = ax.imshow(field, origin='lower', cmap='Blues', vmin=0, vmax=vmax)
        ax.set(title=label, xticks=[], yticks=[])
    fig.colorbar(image, ax=axes.tolist(), label='mm/h; common scale', extend='max')
    fig.suptitle(f'{case["entry"]["time"]} · wet truth {metrics["truth_wet_fraction"]:.1%} '
                 f'· generated {metrics["generated_wet_fraction"]:.1%} '
                 f'· CRPS {metrics["crps_mm_h"]:.3f} vs coarse MAE {metrics["coarse_mae_mm_h"]:.3f}')
    fig.savefig(out, dpi=140)
    plt.close(fig)


@torch.no_grad()
def evaluate(cfg, checkpoint, output, split='val', scan_hours=64, cases=6,
             min_wet_fraction=.1, members=4, steps=24):
    archive = ArchiveV2(cfg['data']['prepared'])
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    check_checkpoint(saved, cfg, archive)
    selected, selection = select_wet_cases(archive, cfg, split, scan_hours, cases, min_wet_fraction)
    device = device_for(cfg['train']['device'])
    model = make_model(archive.index['condition_channels'], cfg).to(device).eval()
    model.load_state_dict(saved['ema'])
    bundle = saved.get('regression_condition')
    conditioner = FrozenRegression(bundle, archive.index['condition_channels'], archive.stats).to(device) if bundle else None
    out = Path(output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Evaluation output already exists: {out}')
    out.mkdir(parents=True, exist_ok=True)
    size, halo = cfg['patch']['size'], cfg['patch']['halo']
    values = []
    for i, case in enumerate(selected):
        entry, y, x = case['entry'], case['y'], case['x']
        condition, context = archive.inputs(entry, y, x, cfg['patch'])
        coarse_full = crop_v2(archive.array(entry, 'baseline')[1:2], y, x, size, halo).copy()
        truth = crop_v2(archive.array(entry, 'truth')[1:2], y, x, size)[0].copy()
        coarse = crop_v2(archive.array(entry, 'baseline')[1:2], y, x, size)[0].copy()
        area = crop_v2(archive.static['area'], y, x, size).copy()
        condition, context = condition[None].to(device), context[None].to(device)
        with autocast(device, cfg['train']['precision']):
            mean = conditioner(dict(condition=condition, context=context,
                coarse=torch.from_numpy(coarse_full[None]).to(device))) if conditioner else None
            rain = None if mean is None else decode_rain(
                mean[0, 0, halo:halo+size, halo:halo+size].float().cpu().numpy(), archive.stats['precip_log_scale'])
            generated = []
            for member in range(members):
                seed = int(np.random.SeedSequence([cfg['inference']['seed'],
                    int(entry['id'].replace('_', '')), y, x, member]).generate_state(1)[0])
                rng = torch.Generator(device=device).manual_seed(seed)
                noise = torch.randn((1, 1, size+2*halo, size+2*halo), device=device, generator=rng)
                encoded = sample(model, noise, condition, context, steps, mean)
                generated.append(decode_rain(encoded[0, 0, halo:halo+size,
                    halo:halo+size].float().cpu().numpy(), archive.stats['precip_log_scale']))
        ensemble = np.stack(generated)
        metrics = patch_metrics(ensemble, truth, coarse, rain, area)
        record = dict(id=entry['id'], time=entry['time'], y=y, x=x, **metrics)
        values.append(record)
        plot_case(out/f'case_{i+1:02d}_{entry["id"]}.png', case, truth, coarse, rain, ensemble, metrics)
        np.savez_compressed(out/f'case_{i+1:02d}_{entry["id"]}.npz', truth=truth, coarse=coarse,
                            regression=rain if rain is not None else np.empty(0),
                            ensemble=ensemble, area=area)
        print(json.dumps(record), flush=True)
    aggregate = {key: float(np.mean([v[key] for v in values])) for key in metrics}
    report = dict(checkpoint=str(Path(checkpoint).resolve()), epoch=saved['epoch']+1,
                  target_kind='HWT midpoint PRECTOT; wet validation patches selected from truth',
                  selection=selection, members=members, steps=steps, mean_of_case_metrics=aggregate,
                  cases=values, caveat='Selected wet cases measure conditional skill only; '
                  'they do not estimate uniform-population skill or justify tuning on test data.')
    (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    return out


def main():
    parser = argparse.ArgumentParser(description='Audit direct precipitation flow on observed wet patches')
    parser.add_argument('--config', default='configs/discover_precip_direct_v2.yaml')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--scan-hours', type=int, default=64)
    parser.add_argument('--cases', type=int, default=6)
    parser.add_argument('--min-wet-fraction', type=float, default=.1)
    parser.add_argument('--members', type=int, default=4)
    parser.add_argument('--steps', type=int, default=24)
    args = parser.parse_args()
    if min(args.scan_hours, args.cases, args.members, args.steps) < 1 or args.members < 2:
        parser.error('Use positive counts and at least two members')
    if not 0 < args.min_wet_fraction <= 1:
        parser.error('--min-wet-fraction must be in (0,1]')
    print(evaluate(load_config(args.config), args.checkpoint, args.output, args.split,
                   args.scan_hours, args.cases, args.min_wet_fraction, args.members, args.steps))


if __name__ == '__main__':
    main()
