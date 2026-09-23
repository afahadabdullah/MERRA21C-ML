"""Train only full-precipitation flow, optionally conditioned on frozen v2 regression."""
from contextlib import nullcontext
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
import json
import math
import os
import time
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset
from .precip_direct_v2 import (VERSION, DirectPrecipDataset, make_model, objective, sample,
                              decode_rain, validate_config, initialize_from_v2, check_checkpoint,
                              FrozenRegression, regression_bundle)
from .train import device_for, autocast, to_device, atomic_save
from .train_v3_precip import lr_lambda, reduce_totals, reduce_max
from .metrics import crps_ensemble


@torch.no_grad()
def validate(model, conditioner, loader, cfg, device, rain_scale):
    rank = dist.get_rank() if dist.is_initialized() else 0
    rng = torch.Generator(device=device).manual_seed(cfg['train']['seed']+7103+rank)
    halo, size = cfg['patch']['halo'], cfg['patch']['size']
    def core(value):
        return value[..., halo:halo+size, halo:halo+size]
    sums = dict(crps=0., coarse_crps=0., mse=0., bias=0., spread=0., wet_fraction=0., truth_wet_fraction=0.)
    if conditioner is not None:
        sums['regression_crps'] = 0.
    count, previews = 0, []
    for batch in loader:
        b = to_device(batch, device)
        with autocast(device, cfg['train']['precision']):
            mean = conditioner(b) if conditioner is not None else None
            draws = []
            for _ in range(cfg['train']['validation_members']):
                noise = torch.randn(b['target'].shape, device=device, generator=rng)
                full = sample(model, noise, b['condition'], b['context'], cfg['train']['validation_steps'], mean)
                draws.append(decode_rain(core(full).float().cpu().numpy()[:, 0], rain_scale))
        ensemble = np.stack(draws)
        truth, coarse = [core(b[k]).cpu().numpy()[:, 0] for k in ('truth', 'coarse')]
        area = b['area'].cpu().numpy()
        values = dict(crps=crps_ensemble(ensemble, truth), coarse_crps=abs(coarse-truth),
                      mse=(ensemble.mean(0)-truth)**2, bias=ensemble.mean(0)-truth,
                      spread=ensemble.std(0), wet_fraction=(ensemble >= .1).mean(0),
                      truth_wet_fraction=(truth >= .1).astype(float))
        regression = None if mean is None else decode_rain(core(mean).float().cpu().numpy()[:, 0], rain_scale)
        if regression is not None:
            values['regression_crps'] = abs(regression-truth)
        for key, value in values.items():
            sums[key] += float(((value*area).sum((-2, -1))/area.sum((-2, -1))).sum())
        count += len(truth)
        if rank == 0:
            for j in range(min(len(truth), cfg['train']['validation_plot_samples']-len(previews))):
                item = dict(truth=truth[j], coarse=coarse[j], ensemble=ensemble[:, j], area=area[j])
                if regression is not None:
                    item['regression'] = regression[j]
                previews.append(item)
    totals = reduce_totals([*sums.values(), count], device)
    if totals[-1] == 0 or not np.isfinite(totals).all():
        raise FloatingPointError('Invalid direct precipitation validation')
    metrics = {k: v/totals[-1] for k, v in zip(sums, totals[:-1])}
    metrics['ensemble_mean_rmse'] = math.sqrt(metrics.pop('mse'))
    return metrics, previews


def save_plots(previews, history, out):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    out.mkdir(parents=True, exist_ok=True)
    if not previews:
        raise ValueError('No validation previews')
    arrays = {f'patch_{i}_{k}': v for i, item in enumerate(previews) for k, v in item.items()}
    np.savez_compressed(out/'samples.npz', **arrays)
    has_regression = 'regression' in previews[0]
    columns = 7 if has_regression else 6
    fig, axes = plt.subplots(len(previews), columns, figsize=(3*columns, 3.4*len(previews)),
                             squeeze=False, constrained_layout=True)
    for i, item in enumerate(previews):
        truth, coarse, ensemble = [item[k] for k in ('truth', 'coarse', 'ensemble')]
        fields = [('HWT truth', truth), ('Coarse', coarse)]
        if has_regression:
            fields.append(('Frozen v2 input', item['regression']))
        fields += [('Member 1', ensemble[0]), ('Member 2', ensemble[1]),
                   ('Ensemble mean', ensemble.mean(0)), ('Spread', ensemble.std(0))]
        vmax = max(1., float(np.quantile(truth, .995)), float(np.quantile(coarse, .995)))
        for ax, (name, value) in zip(axes[i], fields):
            im = ax.imshow(value, origin='lower', cmap='Blues', vmin=0, vmax=vmax)
            ax.set(title=f'Patch {i+1}: {name}', xticks=[], yticks=[])
        fig.colorbar(im, ax=axes[i].tolist(), label='mm/h (fixed truth/coarse scale)', extend='max')
    fig.suptitle(f'Full precipitation flow — epoch {history[-1]["epoch"]} — fixed validation patches')
    fig.savefig(out/'fields.png', dpi=130)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    axes[0].plot([r['epoch'] for r in history], [r['training_loss'] for r in history])
    axes[0].set(title='Flow objective', xlabel='Epoch')
    rows = [r for r in history if 'crps' in r]
    for key in ('crps', 'coarse_crps', 'regression_crps'):
        if key in rows[-1]:
            axes[1].plot([r['epoch'] for r in rows], [r[key] for r in rows], label=key)
    axes[1].set(title='Validation CRPS', xlabel='Epoch', ylabel='mm/h')
    axes[1].legend()
    for key in ('bias', 'spread', 'ensemble_mean_rmse'):
        axes[2].plot([r['epoch'] for r in rows], [r[key] for r in rows], label=key)
    axes[2].set(title='Rainfall metrics', xlabel='Epoch', ylabel='mm/h')
    axes[2].legend()
    fig.savefig(out/'history.png', dpi=130)
    plt.close(fig)
    (out/'metrics.json').write_text(json.dumps(history[-1], indent=2)+'\n')


def rank_zero_action(action, rank, group):
    """CPU coordination for saving/plotting without a pending NCCL collective."""
    result = [None]
    if rank == 0:
        try:
            action()
        except Exception as exc:
            result[0] = f'{type(exc).__name__}: {exc}'
    if group is not None:
        dist.broadcast_object_list(result, src=0, group=group)
    if result[0] is not None:
        raise RuntimeError(result[0])


def train(cfg, resume=None, initialize=None):
    validate_config(cfg)
    if resume and initialize:
        raise ValueError('Resume and weight initialization are mutually exclusive')
    world, rank, local = [int(os.getenv(k, default)) for k, default in
                          [('WORLD_SIZE', '1'), ('RANK', '0'), ('LOCAL_RANK', '0')]]
    device = device_for(cfg['train']['device'])
    if world > 1:
        if device.type == 'cuda':
            torch.cuda.set_device(local)
            device = torch.device('cuda', local)
        elif device.type != 'cpu':
            raise ValueError('DDP requires CUDA or CPU')
        dist.init_process_group('nccl' if device.type == 'cuda' else 'gloo')
    group = dist.new_group(backend='gloo', timeout=timedelta(hours=12)) if world > 1 else None
    try:
        return _train(cfg, resume, initialize, device, rank, world, local, group)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _train(cfg, resume, initialize, device, rank, world, local, group):
    started = time.monotonic()
    tr, p = cfg['train'], cfg['patch']
    if p['samples_per_epoch'] % world:
        raise ValueError('samples_per_epoch must divide world size')
    torch.manual_seed(tr['seed']+rank)
    data = DirectPrecipDataset(cfg, 'train', p['samples_per_epoch'], tr['seed'])
    val = DirectPrecipDataset(cfg, 'val', tr['validation_patches'], tr['seed']+991)
    archive = data.archive
    sampler = DistributedSampler(data, world, rank, seed=tr['seed']) if world > 1 else None
    kwargs = dict(batch_size=tr['batch_size'], num_workers=tr['workers'], pin_memory=device.type == 'cuda')
    loader = DataLoader(data, sampler=sampler, **kwargs)
    vloader = DataLoader(Subset(val, range(rank, len(val), world)), **kwargs)
    out = Path(tr['output'])
    if resume is None and out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Use a fresh output or --resume: {out}')
    # Multiple workers may start together; only rank zero creates artifacts.
    rank_zero_action(lambda: out.mkdir(parents=True, exist_ok=True), rank, group)
    saved = torch.load(resume, map_location='cpu', weights_only=True) if resume else None
    if saved:
        check_checkpoint(saved, cfg, archive, world)
        if Path(resume).resolve().parent != out.resolve():
            raise ValueError('Resume in the original output directory to retain best checkpoint and plots')
    bundle = saved.get('regression_condition') if saved else (
        regression_bundle(cfg['conditioning']['checkpoint'], archive) if cfg.get('conditioning') else None)
    conditioner = FrozenRegression(bundle, archive.index['condition_channels'], archive.stats).to(device) if bundle else None
    model = make_model(archive.index['condition_channels'], cfg).to(device)
    initialization = saved.get('initialization') if saved else (
        initialize_from_v2(model, initialize, cfg, archive) if initialize else None)
    if saved:
        model.load_state_dict(saved['model'])
    training_model = DDP(model, device_ids=[local] if device.type == 'cuda' else None,
                         find_unused_parameters=True) if world > 1 else model
    ema = deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tr['learning_rate'], weight_decay=tr['weight_decay'])
    scheduler_cfg = dict(train=dict(tr, lr_schedule='cosine'))
    total_steps = math.ceil(len(loader)/tr['accumulate'])*tr['epochs']
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda(scheduler_cfg, total_steps))
    start, best, history = 0, float('inf'), []
    if saved:
        ema.load_state_dict(saved['ema'])
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        start, best, history = saved['epoch']+1, saved['best'], saved['history']
        torch.set_rng_state(saved['rng'][rank]['cpu'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state(saved['rng'][rank]['cuda'], device)
    if rank == 0:
        print(f'Direct precipitation flow: world={world}; one target; '
              f'frozen regression condition={conditioner is not None}; '
              f'effective batch={tr["batch_size"]*tr["accumulate"]*world}', flush=True)
    # Recover a validation artifact if a job ended after its durable checkpoint.
    if saved and 'crps' in history[-1]:
        destination = out/'validation_plots'/f'epoch_{start:04d}'
        missing = not all((destination/f).exists() for f in ('fields.png', 'history.png', 'samples.npz', 'metrics.json'))
        if group is not None:
            message = [missing if rank == 0 else None]
            dist.broadcast_object_list(message, src=0, group=group)
            missing = message[0]
        if missing:
            with torch.random.fork_rng(devices=[device.index] if device.type == 'cuda' else []):
                _, previews = validate(ema, conditioner, vloader, cfg, device, data.rain_scale)
            rank_zero_action(lambda: save_plots(previews, history, destination), rank, group)
    longest = 0.
    for epoch in range(start, tr['epochs']):
        epoch_started = time.monotonic()
        data.epoch = epoch
        if sampler:
            sampler.set_epoch(epoch)
        training_model.train()
        optimizer.zero_grad(set_to_none=True)
        total, count = 0., 0
        for i, batch in enumerate(loader):
            b = to_device(batch, device)
            window_start = i//tr['accumulate']*tr['accumulate']
            window_samples = min(tr['accumulate']*tr['batch_size'], len(data)//world-window_start*tr['batch_size'])
            do_step = (i+1) % tr['accumulate'] == 0 or i+1 == len(loader)
            with training_model.no_sync() if world > 1 and not do_step else nullcontext():
                with autocast(device, tr['precision']):
                    if conditioner is not None:
                        b['mean'] = conditioner(b)
                    loss = objective(training_model, b)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite direct flow loss')
                (loss*len(b['target'])/window_samples).backward()
            if do_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), tr['grad_clip'], error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    for averaged, current in zip(ema.parameters(), model.parameters()):
                        averaged.lerp_(current, 1-tr['ema_decay'])
            total += loss.item()*len(b['target'])
            count += len(b['target'])
        total, count = reduce_totals([total, count], device)
        due = (epoch+1) % tr['validation_interval'] == 0 or epoch+1 == tr['epochs']
        metrics, previews = validate(ema, conditioner, vloader, cfg, device, data.rain_scale) if due else ({}, [])
        row = dict(epoch=epoch+1, training_loss=total/count, learning_rate=scheduler.get_last_lr()[0], **metrics)
        if device.type == 'cuda':
            row['peak_gpu_gb'] = reduce_max(torch.cuda.max_memory_allocated(device)/1024**3, device)
        history.append(row)
        improved = due and metrics['crps'] < best
        if improved:
            best = metrics['crps']
        rng = dict(cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(device) if device.type == 'cuda' else None)
        states = [None]*world
        if world > 1:
            dist.all_gather_object(states, rng)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
        else:
            states[0] = rng
        payload = dict(version=VERSION, targets=['precip'], epoch=epoch, config=cfg, world_size=world,
                       fingerprint=archive.index['fingerprint'], stats=archive.stats,
                       model=model.state_dict(), ema=ema.state_dict(), regression_condition=bundle,
                       optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                       rng=states, best=best, history=history, initialization=initialization)
        def save():
            atomic_save(out/'last_direct_v2.pt', payload)
            if improved:
                atomic_save(out/'best_direct_v2.pt', payload)
            (out/'history.json').write_text(json.dumps(history, indent=2)+'\n')
            print(json.dumps(row), flush=True)
            if due:
                save_plots(previews, history, out/'validation_plots'/f'epoch_{epoch+1:04d}')
        rank_zero_action(save, rank, group)
        longest = max(longest, reduce_max(time.monotonic()-epoch_started, device))
        used = reduce_max(time.monotonic()-started, device)
        if tr.get('time_limit_hours') and used+1.5*longest+600 >= tr['time_limit_hours']*3600:
            if rank == 0:
                print('Saved completed epoch; resume last_direct_v2.pt in the next allocation.', flush=True)
            break
    return out/'last_direct_v2.pt'
