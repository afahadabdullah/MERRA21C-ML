"""Four-GPU v4.1 training: v4's flow objective and validation on the packed loader."""
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
from torch.utils.data import DataLoader, Subset
from .v4 import TARGETS, ArchiveV4, make_model, objective, FrozenRegression, regression_bundle
from .v4_1 import (VERSION, validate_config, base_config, PackedArchive, DatasetV41, EpochSampler,
                   check_checkpoint, validation_due)
from .validation_v4_1 import validate, select_previews, previews, save_plots, EXPECTED
from .train_v4 import calibrate
from .train_precip_direct_v2 import rank_zero_action
from .train import device_for, autocast, to_device, atomic_save
from .train_v3_precip import lr_lambda, reduce_totals, reduce_max

LAST, BEST = 'last_v4_1.pt', 'best_v4_1.pt'


def loader_options(tr, device, workers=None, persistent=True):
    workers = tr['workers'] if workers is None else workers
    kwargs = dict(num_workers=workers, pin_memory=device.type == 'cuda')
    if workers > 0:
        # Workers start after CUDA/NCCL setup; spawn avoids fork deadlocks.
        kwargs.update(multiprocessing_context='spawn', prefetch_factor=tr.get('prefetch_factor', 4),
                      persistent_workers=persistent and tr.get('persistent_workers', True))
    return kwargs


class FirstBatches:
    """The first ``count`` batches of a loader's current epoch, then stop."""

    def __init__(self, loader, count):
        self.loader, self.count = loader, min(count, len(loader))

    def __len__(self):
        return self.count

    def __iter__(self):
        return islice(iter(self.loader), self.count)


def train(cfg, resume=None):
    validate_config(cfg)
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
    if device.type == 'cuda':
        # Fixed patch shapes: let cuDNN pick the fastest kernels once.
        torch.backends.cudnn.benchmark = True
    try:
        group = dist.new_group(backend='gloo', timeout=timedelta(hours=12)) if world > 1 else None
        print(f'[rank {rank}] V4.1 startup: device={device}; world={world}; '
              f'workers={cfg["train"]["workers"]}', flush=True)
        if world > 1:
            probe = torch.ones(1, device=device)
            dist.all_reduce(probe)
            if probe.item() != world:
                raise RuntimeError('Distributed startup check returned an invalid rank count')
            print(f'[rank {rank}] Distributed communication ready', flush=True)
        return _train(cfg, resume, device, rank, world, local, group)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _train(cfg, resume, device, rank, world, local, group):
    started = time.monotonic()
    tr, p = cfg['train'], cfg['patch']
    if p['samples_per_epoch'] % (world*tr['batch_size']):
        raise ValueError('samples_per_epoch must be divisible by world_size*batch_size')
    torch.manual_seed(tr['seed']+rank)
    base = base_config(cfg)
    archive = ArchiveV4(base, verify_files=False)
    packed = PackedArchive(cfg, archive)
    print(f'[rank {rank}] Packed archive ready: {len(packed.manifest["entries"])} hours; '
          f'fingerprint {packed.fingerprint[:12]}', flush=True)
    data = DatasetV41(cfg, 'train', p['samples_per_epoch'], tr['seed'], packed=packed)
    val = DatasetV41(cfg, 'val', tr['validation_patches'], tr['seed']+991, packed=packed)
    sampler = EpochSampler(len(data), rank, world)
    kwargs = loader_options(tr, device)
    loader = DataLoader(data, sampler=sampler, batch_size=tr['batch_size'], **kwargs)
    vloader = DataLoader(Subset(val, range(rank, len(val), world)), batch_size=tr['batch_size'],
                         **loader_options(tr, device, min(tr['workers'], tr.get('val_workers', 4))))
    out = Path(tr['output'])
    if resume is None and out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Use a fresh output or --resume: {out}')
    rank_zero_action(lambda: out.mkdir(parents=True, exist_ok=True), rank, group)
    if rank == 0:
        print(f'Data loading: packed {packed.root}; workers={tr["workers"]}/rank (persistent); '
              f'prefetch={kwargs.get("prefetch_factor", 0)} batches/worker; '
              f'{len(data.yy)} candidate origins; train hours={len(data.entries)}', flush=True)
    saved = torch.load(resume, map_location='cpu', weights_only=True) if resume else None
    if saved:
        check_checkpoint(saved, cfg, archive, packed.fingerprint, world)
        if Path(resume).resolve().parent != out.resolve():
            raise ValueError('Resume in the original output directory to retain best checkpoint and plots')
    bundle = saved['regression_condition'] if saved else regression_bundle(cfg['conditioning']['checkpoint'], archive)
    conditioner = FrozenRegression(bundle, archive.index['condition_channels'], archive.stats, archive.scale,
                                   cfg['data']['humidity_scale_kg_kg']).to(device)
    model = make_model(archive.channels, cfg).to(device)
    if saved:
        if rank == 0:
            print(f'Loaded checkpoint after epoch {saved["epoch"]+1}; using saved calibration.', flush=True)
        conditioner.flow_scale.copy_(saved['flow_scale'].to(device))
        model.load_state_dict(saved['model'])
    else:
        # The first training batches of epoch 0, as v4 calibrated on, drawn
        # through the training loader itself: its persistent workers start
        # once and go straight on into epoch 1 (no second worker pool).
        sampler.set_epoch(0)
        calibrate(conditioner, FirstBatches(loader, tr['calibration_batches']), cfg, device, group)
    training_model = DDP(model, device_ids=[local] if device.type == 'cuda' else None,
                         find_unused_parameters=True) if world > 1 else model
    ema = deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tr['learning_rate'], weight_decay=tr['weight_decay'])
    total_steps = math.ceil(len(loader)/tr['accumulate'])*tr['epochs']
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda(dict(train=dict(tr, lr_schedule='cosine')), total_steps))
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
        print(f'V4.1 flow: {sum(q.numel() for q in model.parameters()):,} parameters; world={world}; '
              f'effective batch={tr["batch_size"]*tr["accumulate"]*world}; '
              f'{len(loader)} batches/rank/epoch; activation_checkpointing={model.checkpointing}', flush=True)
    def plots(diagnostics, rows, destination):
        # Rank 0 only: fixed rainy previews (chosen once, cached in the output) + figures.
        chosen = select_previews(packed, cfg, tr['validation_plot_samples'], out)
        items = previews(ema, conditioner, packed, chosen, cfg, device)
        save_plots(items, diagnostics, rows, destination)

    if saved and 'crps' in history[-1]:
        # Recover a validation artifact if a job ended after its durable checkpoint.
        destination = out/'validation_plots'/f'epoch_{start:04d}'
        missing = not all((destination/f).exists() for f in EXPECTED)
        if group is not None:
            message = [missing if rank == 0 else None]
            dist.broadcast_object_list(message, src=0, group=group)
            missing = message[0]
        if missing:
            with torch.random.fork_rng(devices=[device.index] if device.type == 'cuda' else []):
                _, diagnostics = validate(ema, conditioner, vloader, cfg, device, group=group)
                rank_zero_action(lambda: plots(diagnostics, history, destination), rank, group)
    longest = 0.
    per_rank = len(data)//world
    for epoch in range(start, tr['epochs']):
        epoch_started = time.monotonic()
        if rank == 0:
            print(f'Epoch {epoch+1}/{tr["epochs"]}: starting {len(loader)} training batches per rank', flush=True)
        sampler.set_epoch(epoch)
        training_model.train()
        optimizer.zero_grad(set_to_none=True)
        total, count = 0., 0
        data_wait_s, step_s = 0., 0.
        waiting_since = time.monotonic()
        for i, batch in enumerate(loader):
            batch_ready = time.monotonic()
            data_wait_s += batch_ready-waiting_since
            b = to_device(batch, device)
            window_start = i//tr['accumulate']*tr['accumulate']
            window_samples = min(tr['accumulate']*tr['batch_size'], per_rank-window_start*tr['batch_size'])
            do_step = (i+1) % tr['accumulate'] == 0 or i+1 == len(loader)
            with training_model.no_sync() if world > 1 and not do_step else nullcontext():
                with autocast(device, tr['precision']):
                    conditioner.prepare(b)
                    loss = objective(training_model, b)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite v4.1 flow loss')
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
            # Includes transfer, compute and any DDP wait on slower ranks.
            step_s += time.monotonic()-batch_ready
            if rank == 0 and (i == 0 or (i+1) % 64 == 0 or i+1 == len(loader)):
                elapsed = time.monotonic()-epoch_started
                print(f'Epoch {epoch+1}/{tr["epochs"]}: batch {i+1}/{len(loader)}; '
                      f'loss={total/count:.6g}; elapsed={elapsed/60:.1f} min; '
                      f'data_wait={data_wait_s/(i+1):.3f}s/batch; step={step_s/(i+1):.3f}s/batch; '
                      f'{(i+1)*tr["batch_size"]*world/elapsed:.1f} samples/s', flush=True)
            waiting_since = time.monotonic()
        total, count, wait_total, step_total = reduce_totals([total, count, data_wait_s, step_s], device)
        train_wall = reduce_max(time.monotonic()-epoch_started, device)
        due = validation_due(epoch+1, tr)
        validation_started = time.monotonic()
        if due and rank == 0:
            print(f'Epoch {epoch+1}: validating {tr["validation_patches"]} patches, '
                  f'{tr["validation_members"]} members, {tr["validation_steps"]} steps', flush=True)
        metrics, diagnostics = validate(ema, conditioner, vloader, cfg, device, group=group) if due else ({}, None)
        row = dict(epoch=epoch+1, training_loss=total/count, learning_rate=scheduler.get_last_lr()[0],
                   data_wait_s_per_rank=wait_total/world, step_s_per_rank=step_total/world,
                   train_wall_s=train_wall, samples_per_s=len(data)/train_wall, **metrics)
        if due:
            row['validation_wall_s'] = reduce_max(time.monotonic()-validation_started, device)
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
                       packed_fingerprint=packed.fingerprint,
                       flow_scale=conditioner.flow_scale.detach().cpu(),
                       model=model.state_dict(), ema=ema.state_dict(), regression_condition=bundle,
                       optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                       rng=states, best=best, history=history, initialization=None)
        def save():
            atomic_save(out/LAST, payload)
            if improved:
                atomic_save(out/BEST, payload)
            # Kept checkpoints land on validation epochs, tagged with precip CRPS.
            if due:
                (out/'checkpoints').mkdir(exist_ok=True)
                atomic_save(out/'checkpoints'/f'epoch_{epoch+1:04d}_crps{metrics["crps"]:.4f}_v4_1.pt', payload)
            (out/'history.json').write_text(json.dumps(history, indent=2)+'\n')
            print(json.dumps(row), flush=True)
            if due:
                plots(diagnostics, history, out/'validation_plots'/f'epoch_{epoch+1:04d}')
                print(f'Epoch {epoch+1}: validation plots in {out}/validation_plots/epoch_{epoch+1:04d}', flush=True)
        rank_zero_action(save, rank, group)
        longest = max(longest, reduce_max(time.monotonic()-epoch_started, device))
        used = reduce_max(time.monotonic()-started, device)
        if tr.get('time_limit_hours') and used+1.5*longest+600 >= tr['time_limit_hours']*3600:
            if rank == 0:
                print(f'Saved completed epoch; resume {LAST} in the next allocation.', flush=True)
            break
    return out/LAST

