"""DDP training with EMA, warmup+cosine LR, rank-specific resume and generated validation plots.

Long diffusion training is split into several Slurm jobs: ``time_limit_hours``
stops cleanly after the last epoch that fits, and the next job resumes from
``last_v3_precip.pt`` (see scripts/submit_v3_precip.sh).
"""
from contextlib import nullcontext
import math
import time
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import os
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset
from .config_v3_precip import validate_config
from .dataset_v3_precip import PrecipDataset, decode_rain, rain_codec
from .model_v3_precip import make_regression, PrecipEDM, objective, sample_edm, core, weighted_loss
from .model_v2 import regression_v2
from .metrics import crps_ensemble
from .train import device_for, autocast, to_device, atomic_save


def reduce_totals(values, device):
    values = torch.tensor(values, dtype=torch.float64, device=device)
    if dist.is_initialized():
        dist.all_reduce(values)
    return values.cpu().tolist()


def lr_lambda(cfg, total_steps):
    """Linear warmup, then cosine decay to min_lr_ratio*lr (constant if lr_schedule is 'constant')."""
    tr = cfg['train']
    warmup, floor, kind = tr.get('warmup_steps', 0), tr.get('min_lr_ratio', 1.), tr.get('lr_schedule', 'constant')
    def factor(step):
        if warmup and step < warmup:
            return (step+1)/warmup
        if kind == 'constant':
            return 1.
        progress = min((step-warmup)/max(total_steps-warmup, 1), 1.)
        return floor+(1-floor)*.5*(1+math.cos(math.pi*progress))
    return factor


def reduce_max(value, device):
    value = torch.tensor([value], dtype=torch.float64, device=device)
    if dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return float(value.item())


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def check_checkpoint(ckpt, cfg, archive, resume=False):
    if ckpt.get('version') != 'v3_precip' or ckpt.get('targets') != ['precip']:
        raise ValueError('Require a one-target v3_precip checkpoint')
    if ckpt['fingerprint'] != archive.index['fingerprint']:
        raise ValueError('Checkpoint archive fingerprint mismatch')
    if ckpt.get('hourly_fingerprint') != archive.hourly_fingerprint:
        raise ValueError('Checkpoint hourly target fingerprint mismatch; start a fresh run')
    for key in ('data', 'patch', 'model', 'diffusion'):
        old, new = deepcopy(ckpt['config'][key]), deepcopy(cfg[key])
        if key == 'data':
            old.pop('prepared', None)
            new.pop('prepared', None)
        if old != new:
            raise ValueError(f'Checkpoint {key} mismatch')
    if resume:
        # time_limit_hours only controls job segmentation, never the optimization.
        ignored = ('output', 'workers', 'device', 'time_limit_hours')
        if ({k: v for k, v in ckpt['config']['train'].items() if k not in ignored}
                != {k: v for k, v in cfg['train'].items() if k not in ignored}):
            raise ValueError('Exact resume requires the same training configuration')


@torch.no_grad()
def calibrate_scale(regression, loader, cfg, device):
    total, count = 0., 0
    for i, batch in enumerate(loader):
        if i >= cfg['train']['calibration_batches']:
            break
        batch = to_device(batch, device)
        with autocast(device, cfg['train']['precision']):
            error = (batch['target']-regression_v2(regression, batch)).square()
        total += float(weighted_loss(error, batch, cfg['patch'], full=cfg['patch'].get('loss_on_halo', False)))*len(error)
        count += len(error)
    total, count = reduce_totals([total, count], device)
    return max((total/count)**.5, .05)


@torch.no_grad()
def validate_generated(model, regression, scale, loader, cfg, device, previews=None):
    """Fixed uniform validation patches/seeds; score samples in mm/h, not noise loss."""
    rank = dist.get_rank() if dist.is_initialized() else 0
    generator = torch.Generator(device=device).manual_seed(cfg['train']['seed']+7103+rank)
    p = cfg['patch']
    codec = rain_codec(cfg)
    decode = lambda value: decode_rain(value, codec['scale'], codec['wet_threshold'])
    sums = dict(wet_fraction=0., truth_wet_fraction=0., crps=0., coarse_crps=0., regression_crps=0., bias=0., spread=0.)
    count = 0
    for batch in loader:
        batch = to_device(batch, device)
        with autocast(device, cfg['train']['precision']):
            mean = regression_v2(regression if regression is not None else model, batch)
            draws = []
            for _ in range(cfg['train']['validation_members'] if regression is not None else 1):
                value = mean
                if regression is not None:
                    noise = torch.randn(mean.shape, device=device, generator=generator)
                    value = mean+scale*sample_edm(model, noise, batch['condition'], batch['context'], mean,
                                                  cfg, cfg['train']['validation_steps'])
                draws.append(decode(core(batch['baseline']+value, p['halo'], p['size']).float().cpu().numpy()))
        ensemble = np.stack(draws)[:, :, 0]
        truth = core(batch['truth'], p['halo'], p['size']).cpu().numpy()[:, 0]
        coarse = core(batch['coarse'], p['halo'], p['size']).cpu().numpy()[:, 0]
        deterministic = decode(core(batch['baseline']+mean, p['halo'], p['size']).float().cpu().numpy())[:, 0]
        area = batch['area'].cpu().numpy()
        if previews is not None:
            for j in range(min(len(truth), cfg['train'].get('validation_plot_samples', 2)-len(previews))):
                previews.append(dict(truth=truth[j], coarse=coarse[j], regression=deterministic[j],
                                     ensemble=ensemble[:, j], area=area[j]))
        values = dict(crps=crps_ensemble(ensemble, truth), coarse_crps=abs(coarse-truth),
                      regression_crps=abs(deterministic-truth), bias=ensemble.mean(0)-truth,
                      spread=ensemble.std(0), wet_fraction=(ensemble >= .1).mean(0),
                      truth_wet_fraction=(truth >= .1).astype('float64'))
        for key, value in values.items():
            sums[key] += float(((value*area).sum((1, 2))/area.sum((1, 2))).sum())
        count += len(truth)
    totals = reduce_totals([*sums.values(), count], device)
    return {key: value/totals[-1] for key, value in zip(sums, totals[:-1])}


def train(cfg, stage, regression_checkpoint=None, resume=None, time_limit_hours=None):
    validate_config(cfg)
    if stage not in ('regression', 'diffusion'):
        raise ValueError('stage must be regression or diffusion')
    world, rank, local = [int(os.getenv(k, default)) for k, default in [('WORLD_SIZE', '1'), ('RANK', '0'), ('LOCAL_RANK', '0')]]
    device = device_for(cfg['train']['device'])
    if world > 1:
        if device.type == 'cuda':
            torch.cuda.set_device(local)
            device = torch.device('cuda', local)
        elif device.type != 'cpu':
            raise ValueError('DDP supports CUDA production or CPU verification')
        dist.init_process_group('nccl' if device.type == 'cuda' else 'gloo')
    try:
        limit = time_limit_hours if time_limit_hours is not None else cfg['train'].get('time_limit_hours')
        return _train(cfg, stage, regression_checkpoint, resume, device, rank, world, local, limit)
    finally:
        if world > 1 and dist.is_initialized():
            dist.destroy_process_group()


def _train(cfg, stage, regression_checkpoint, resume, device, rank, world, local, time_limit_hours=None):
    started = time.monotonic()
    tr, p = cfg['train'], cfg['patch']
    if p['samples_per_epoch'] % world:
        raise ValueError('samples_per_epoch must divide world size without duplicated patches')
    torch.manual_seed(tr['seed']+rank)
    data = PrecipDataset(cfg, 'train', p['samples_per_epoch'], tr['seed'])
    val = PrecipDataset(cfg, 'val', tr['val_batches']*tr['batch_size'], tr['seed']+991)
    kwargs = dict(batch_size=tr['batch_size'], num_workers=tr['workers'], pin_memory=device.type == 'cuda')
    sampler = DistributedSampler(data, num_replicas=world, rank=rank, seed=tr['seed']) if world > 1 else None
    loader = DataLoader(data, sampler=sampler, **kwargs)
    # Strided validation subsets have no duplicated/padded samples. Some ranks
    # may be empty for tiny fixtures; only the final totals use collectives.
    val_loader = DataLoader(Subset(val, range(rank, len(val), world)), **kwargs)
    archive = data.archive
    out = Path(tr['output'])/(stage+'_v3_precip')
    if out.exists() and any(out.iterdir()) and resume is None:
        raise FileExistsError(f'Use a fresh run or --resume: {out}')
    out.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(resume, map_location='cpu', weights_only=True) if resume else None
    if ckpt:
        check_checkpoint(ckpt, cfg, archive, resume=True)
        if ckpt['stage'] != stage:
            raise ValueError('Resume stage mismatch')
        if ckpt.get('world_size', 1) != world:
            raise ValueError('Exact resume requires the same DDP world size')
    regression, scale, regression_hash = None, 1., None
    if stage == 'diffusion':
        regression = make_regression(archive.channels, cfg).to(device).eval().requires_grad_(False)
        if ckpt:
            regression.load_state_dict(ckpt['regression_ema'])
            scale, regression_hash = ckpt['residual_scale'], ckpt['regression_sha256']
        else:
            if not regression_checkpoint:
                raise ValueError('Diffusion needs --regression-checkpoint')
            saved = torch.load(regression_checkpoint, map_location='cpu', weights_only=True)
            check_checkpoint(saved, cfg, archive)
            if saved['stage'] != 'regression':
                raise ValueError('Expected a regression checkpoint')
            last = Path(regression_checkpoint).parent/'last_v3_precip.pt'
            if last.exists():
                finished = torch.load(last, map_location='cpu', weights_only=True)
                if finished['epoch']+1 < finished['config']['train']['regression_epochs']:
                    raise ValueError(f'Regression run is incomplete ({finished["epoch"]+1} epochs): '
                                     'resume it before training diffusion')
            regression.load_state_dict(saved['ema'])
            regression_hash = file_hash(regression_checkpoint)
            scale = calibrate_scale(regression, loader, cfg, device)
        model = PrecipEDM(archive.channels, cfg).to(device)
    else:
        model = make_regression(archive.channels, cfg).to(device)
    # Self-attention inherits two unused context_norm parameters from UNetV2.
    training_model = DDP(model, device_ids=[local] if device.type == 'cuda' else None,
                         find_unused_parameters=True) if world > 1 else model
    ema = deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tr['learning_rate'], weight_decay=tr['weight_decay'])
    updates_per_epoch = math.ceil(len(loader)/tr['accumulate'])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda(cfg, updates_per_epoch*tr[stage+'_epochs']))
    start, best, history = 0, float('inf'), []
    if ckpt:
        model.load_state_dict(ckpt['model'])
        ema.load_state_dict(ckpt['ema'])
        optimizer.load_state_dict(ckpt['optimizer'])
        if ckpt.get('scheduler') is None:
            raise ValueError('Checkpoint predates the LR schedule; start a fresh run')
        scheduler.load_state_dict(ckpt['scheduler'])
        start, best, history = ckpt['epoch']+1, ckpt['best'], ckpt['history']
        if 'rng_by_rank' in ckpt:
            rng = ckpt['rng_by_rank'][rank]
            torch.set_rng_state(rng['cpu'])
            if device.type == 'cuda':
                torch.cuda.set_rng_state(rng['cuda'], device)
        else:  # Original single-device checkpoints.
            torch.set_rng_state(ckpt['rng_cpu'])
            if device.type == 'cuda' and ckpt['rng_cuda'] is not None:
                torch.cuda.set_rng_state_all(ckpt['rng_cuda'])
        # A resumed run in a new directory must retain the earlier best model.
        if rank == 0 and not (out/'best_v3_precip.pt').exists():
            previous_best = Path(resume).parent/'best_v3_precip.pt'
            if not previous_best.exists() and math.isfinite(best):
                raise FileNotFoundError('Resume requires the companion best_v3_precip.pt')
            if previous_best.exists():
                best_ckpt = torch.load(previous_best, map_location='cpu', weights_only=True)
                check_checkpoint(best_ckpt, cfg, archive, resume=True)
                atomic_save(out/'best_v3_precip.pt', best_ckpt)
    for epoch in range(start, tr[stage+'_epochs']):
        data.epoch = epoch  # Nonpersistent workers pick up this new epoch.
        if sampler is not None:
            sampler.set_epoch(epoch)
        training_model.train()
        optimizer.zero_grad(set_to_none=True)
        total, count = 0., 0
        for step, batch in enumerate(loader):
            batch = to_device(batch, device)
            window_start = (step//tr['accumulate'])*tr['accumulate']
            window_samples = min(tr['accumulate']*tr['batch_size'], len(data)//world-window_start*tr['batch_size'])
            do_step = (step+1) % tr['accumulate'] == 0 or step+1 == len(loader)
            n = len(batch['target'])
            sync = training_model.no_sync() if world > 1 and not do_step else nullcontext()
            with sync:
                with autocast(device, tr['precision']):
                    loss = objective(training_model, batch, cfg, regression, scale)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite training loss')
                (loss*n/window_samples).backward()
            if do_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), tr['grad_clip'], error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    for averaged, value in zip(ema.parameters(), model.parameters()):
                        averaged.lerp_(value, 1-tr['ema_decay'])
            total += loss.item()*n
            count += n
        total, count = reduce_totals([total, count], device)
        # Regression validation is one cheap deterministic pass; generated
        # validation (members x Heun steps) is gated by validation_interval.
        interval = 1 if stage == 'regression' else tr.get('validation_interval', 1)
        final = epoch+1 == tr[stage+'_epochs']
        validate_due = (epoch+1) % interval == 0 or final
        plot_due = validate_due and ((epoch+1) % tr.get('validation_plot_interval', 5) == 0 or final)
        previews = [] if rank == 0 and plot_due else None
        metrics = validate_generated(ema, regression, scale, val_loader, cfg, device, previews) if validate_due else {}
        row = dict(epoch=epoch+1, training_loss=total/count, learning_rate=scheduler.get_last_lr()[0], **metrics)
        if device.type == 'cuda':  # Measure real A100 headroom for the larger [fix 2] model.
            row['peak_gpu_gb'] = torch.cuda.max_memory_allocated(device)/1024**3
        history.append(row)
        improved = validate_due and metrics['crps'] < best
        if validate_due:
            best = min(best, metrics['crps'])
        if rank == 0 and plot_due:
            from .validation_plots_v3_precip import plot_validation
            plot_validation(previews, history, out/'validation_plots', stage, epoch+1, cfg)
        rng = dict(cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(device) if device.type == 'cuda' else None)
        rng_by_rank = [None]*world
        if world > 1:
            dist.all_gather_object(rng_by_rank, rng)
        else:
            rng_by_rank[0] = rng
        saved = dict(version='v3_precip', targets=['precip'], stage=stage, config=cfg,
                     fingerprint=archive.index['fingerprint'], epoch=epoch, best=best, history=history,
                     hourly_fingerprint=archive.hourly_fingerprint,
                     model=model.state_dict(), ema=ema.state_dict(), optimizer=optimizer.state_dict(),
                     scheduler=scheduler.state_dict(),
                     regression_ema=regression.state_dict() if regression is not None else None,
                     regression_sha256=regression_hash, residual_scale=scale,
                     world_size=world, rng_by_rank=rng_by_rank,
                     rng_cpu=rng['cpu'], rng_cuda=[rng['cuda']] if device.type == 'cuda' else None)
        if rank == 0:
            atomic_save(out/'last_v3_precip.pt', saved)
            if improved:
                atomic_save(out/'best_v3_precip.pt', saved)
            (out/'history_v3_precip.json').write_text(json.dumps(history, indent=2, allow_nan=False)+'\n')
            print(json.dumps(dict(stage=stage, world_size=world, **row)), flush=True)
        if world > 1:
            dist.barrier()
        if time_limit_hours and epoch+1 < tr[stage+'_epochs']:
            elapsed = time.monotonic()-started
            per_epoch = elapsed/(epoch+1-start)
            # Every rank must take the same decision; use the slowest rank's clock.
            stop = reduce_max(elapsed+1.5*per_epoch, device) > time_limit_hours*3600
            if stop:
                if rank == 0:
                    print(json.dumps(dict(stage=stage, stopped_for_time_limit=True, completed_epochs=epoch+1,
                                          resume=str(out/'last_v3_precip.pt'))), flush=True)
                break
    return out/('best_v3_precip.pt' if math.isfinite(best) else 'last_v3_precip.pt')
