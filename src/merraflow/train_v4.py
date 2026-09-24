"""Four-GPU hybrid v4 flow training with rank-safe ensemble validation."""
from contextlib import nullcontext
from copy import deepcopy
from datetime import timedelta
from itertools import islice
from pathlib import Path
import json
import math
import os
import time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset
from .v4 import (VERSION, TARGETS, ArchiveV4, DatasetV4, make_model, objective, validate_config,
                 check_checkpoint, FrozenRegression, regression_bundle)
from .validation_v4 import validate, save_plots
from .train_precip_direct_v2 import rank_zero_action
from .train import device_for, autocast, to_device, atomic_save
from .train_v3_precip import lr_lambda, reduce_totals, reduce_max


@torch.no_grad()
def calibrate(conditioner, loader, cfg, device, group=None):
    rank = dist.get_rank() if dist.is_initialized() else 0
    started = time.monotonic()
    batches = len(loader)
    if batches > cfg['train']['calibration_batches']:
        raise ValueError('Calibration requires a loader bounded to calibration_batches')
    print(f'[rank {rank}] Calibration starting: {batches} batches', flush=True)
    sums = torch.zeros(len(TARGETS)+1, dtype=torch.float64, device=device)
    halo, size = cfg['patch']['halo'], cfg['patch']['size']
    for i, batch in enumerate(loader):
        b = to_device(batch, device)
        with autocast(device, cfg['train']['precision']):
            error = (b['target']-conditioner(b))[:, :, halo:halo+size, halo:halo+size]
        weighted = (error.double().square()*b['area'][:, None]).sum((-2,-1))/b['area'].sum((-2,-1))[:,None]
        sums[:-1] += weighted.sum(0)
        sums[-1] += len(error)
        if i == 0 or (i+1) % 16 == 0 or i+1 == batches:
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            print(f'[rank {rank}] Calibration {i+1}/{batches} '
                  f'({time.monotonic()-started:.1f}s)', flush=True)
    if dist.is_initialized():
        # Archive I/O can leave ranks far apart. Use the existing long-lived
        # CPU control group, not a pending NCCL operation, while they catch up.
        if group is not None:
            sums = sums.cpu()
        print(f'[rank {rank}] Calibration local work complete; reducing totals', flush=True)
        dist.all_reduce(sums, group=group)
    if not sums[-1] or not torch.isfinite(sums).all():
        raise ValueError('Invalid training-only calibration')
    scale = torch.sqrt(sums[:-1]/sums[-1]).clamp_min(.05).float()[None,:,None,None]
    scale[:,1] = 1.  # Rain is full sqrt rainfall, never calibrated as a residual.
    conditioner.flow_scale.copy_(scale)
    print(f'[rank {rank}] Calibration complete ({time.monotonic()-started:.1f}s)', flush=True)
    return scale.cpu()


def train(cfg, resume=None, initialize=None):
    validate_config(cfg)
    if initialize:
        raise ValueError('v4 flow starts fresh; only original frozen regression weights are reused')
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
    try:
        group = dist.new_group(backend='gloo', timeout=timedelta(hours=12)) if world > 1 else None
        print(f'[rank {rank}] V4 startup: device={device}; world={world}; '
              f'workers={cfg["train"]["workers"]}', flush=True)
        if world > 1:
            # Check communication before archive reads or worker startup so a
            # transport failure is distinguishable from slow calibration.
            print(f'[rank {rank}] Checking distributed communication', flush=True)
            probe = torch.ones(1, device=device)
            dist.all_reduce(probe)
            if probe.item() != world:
                raise RuntimeError('Distributed startup check returned an invalid rank count')
            print(f'[rank {rank}] Distributed communication ready', flush=True)
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
    print(f'[rank {rank}] Checking v4 archive manifests', flush=True)
    archive = ArchiveV4(cfg, verify_files=False)
    print(f'[rank {rank}] V4 archive manifests valid; preparing train and val datasets', flush=True)
    data = DatasetV4(cfg, 'train', p['samples_per_epoch'], tr['seed'], archive=archive)
    val = DatasetV4(cfg, 'val', tr['validation_patches'], tr['seed']+991, archive=archive)
    sampler = DistributedSampler(data, world, rank, seed=tr['seed']) if world > 1 else None
    kwargs = dict(batch_size=tr['batch_size'], num_workers=tr['workers'], pin_memory=device.type == 'cuda')
    if tr['workers'] > 0:
        # Workers start on iteration, after CUDA/NCCL setup. Forking then can
        # deadlock; spawn starts a clean interpreter on Linux as well as macOS.
        # Keep workers nonpersistent so data.epoch reaches each new iterator.
        kwargs['multiprocessing_context'] = 'spawn'
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
    conditioner = FrozenRegression(bundle, archive.index['condition_channels'], archive.stats, archive.scale, cfg['data']['humidity_scale_kg_kg']).to(device) if bundle else None
    model = make_model(archive.channels, cfg).to(device)
    initialization = None
    if saved:
        conditioner.flow_scale.copy_(saved['flow_scale'].to(device))
    else:
        # Bound indices before workers start. Fully exhaust this loader so no
        # prefetched training batches remain during calibration worker shutdown.
        calibration_batches = list(islice(loader.batch_sampler, tr['calibration_batches']))
        calibration_loader = DataLoader(data, batch_sampler=calibration_batches,
                                       **{k: v for k, v in kwargs.items() if k != 'batch_size'})
        calibrate(conditioner, calibration_loader, cfg, device, group)
    if saved:
        model.load_state_dict(saved['model'])
    print(f'[rank {rank}] Preparing training model', flush=True)
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
        print(f'V4 flow: {sum(p.numel() for p in model.parameters()):,} parameters; world={world}; six targets; '
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
        if rank == 0:
            print(f'Epoch {epoch+1}/{tr["epochs"]}: starting {len(loader)} training batches per rank', flush=True)
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
                        conditioner.prepare(b)
                    loss = objective(training_model, b)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite v4 flow loss')
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
            if rank == 0 and (i == 0 or (i+1) % 32 == 0 or i+1 == len(loader)):
                print(f'Epoch {epoch+1}/{tr["epochs"]}: batch {i+1}/{len(loader)}; '
                      f'loss={total/count:.6g}; elapsed={(time.monotonic()-epoch_started)/60:.1f} min', flush=True)
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
        payload = dict(version=VERSION, targets=TARGETS, epoch=epoch, config=cfg, world_size=world,
                       fingerprint=archive.index['fingerprint'], stats=archive.stats,
                       hourly_fingerprint=archive.hourly_fingerprint, humidity_fingerprint=archive.humidity_fingerprint,
                       flow_scale=conditioner.flow_scale.detach().cpu(),
                       model=model.state_dict(), ema=ema.state_dict(), regression_condition=bundle,
                       optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                       rng=states, best=best, history=history, initialization=initialization)
        def save():
            atomic_save(out/'last_v4.pt', payload)
            if improved:
                atomic_save(out/'best_v4.pt', payload)
            (out/'history.json').write_text(json.dumps(history, indent=2)+'\n')
            print(json.dumps(row), flush=True)
            if due:
                save_plots(previews, history, out/'validation_plots'/f'epoch_{epoch+1:04d}')
        rank_zero_action(save, rank, group)
        longest = max(longest, reduce_max(time.monotonic()-epoch_started, device))
        used = reduce_max(time.monotonic()-started, device)
        if tr.get('time_limit_hours') and used+1.5*longest+600 >= tr['time_limit_hours']*3600:
            if rank == 0:
                print('Saved completed epoch; resume last_v4.pt in the next allocation.', flush=True)
            break
    return out/'last_v4.pt'
