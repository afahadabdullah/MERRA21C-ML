"""Two-stage v2 training, DDP, EMA, exact epoch-boundary resume."""
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import math
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from .config import write_json
from .config_v2 import validate_config_v2
from .dataset_v2 import PatchDatasetV2
from .model_v2 import UNetV2, regression_v2
from .loss_v2 import loss_v2, core_v2
from .train import device_for, autocast, to_device, atomic_save


def file_hash_v2(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def check_checkpoint_v2(ckpt, archive, cfg, stage=None):
    if ckpt.get('version') != 'v2' or (stage and ckpt['stage'] != stage):
        raise ValueError('Wrong checkpoint version or stage')
    if ckpt['fingerprint'] != archive.index['fingerprint'] or ckpt['stats'] != archive.stats:
        raise ValueError('V2 checkpoint and archive/statistics mismatch')
    if ckpt['config']['model'] != cfg['model'] or ckpt['config']['patch'] != cfg['patch']:
        raise ValueError('V2 checkpoint model/patch mismatch')


@torch.no_grad()
def calibrate_v2(mean_model, loader, cfg, device, world):
    # RMS (not centered std): any remaining regression bias stays in the residual.
    sums = torch.zeros(6, dtype=torch.float64, device=device)
    for i, batch in enumerate(loader):
        if i >= cfg['train']['calibration_batches']:
            break
        b = to_device(batch, device)
        with autocast(device, cfg['train']['precision']):
            error = core_v2(b['target']-regression_v2(mean_model, b), cfg['patch']['halo'])
        weighted = (error.double().square()*b['area'][:, None]).sum((-2, -1))/b['area'].sum((-2, -1))[:, None]
        sums[:5] += (weighted*b['importance'][:, None]).sum(0)
        sums[5] += len(error)
    if world > 1:
        dist.all_reduce(sums)
    if sums[5] == 0 or not torch.isfinite(sums).all():
        raise ValueError('Invalid training-only residual calibration')
    return torch.sqrt(sums[:5]/sums[5]).clamp_min(.05).float()[None, :, None, None]


def train_v2(cfg, stage, resume=None, regression_checkpoint=None):
    validate_config_v2(cfg)
    if stage not in ('regression', 'flow'):
        raise ValueError('Choose regression or flow stage')
    tr, p = cfg['train'], cfg['patch']
    rank, world, local = [int(os.getenv(k, default)) for k, default in [('RANK', 0), ('WORLD_SIZE', 1), ('LOCAL_RANK', 0)]]
    device = device_for(tr['device'])
    if world > 1:
        if device.type != 'cuda':
            raise ValueError('Production DDP requires CUDA')
        torch.cuda.set_device(local)
        device = torch.device('cuda', local)
        dist.init_process_group('nccl')
    torch.manual_seed(tr['seed']+rank)
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
    data = PatchDatasetV2(cfg['data']['prepared'], 'train', p, p['samples_per_epoch'], tr['seed'])
    val = PatchDatasetV2(cfg['data']['prepared'], 'val', p, tr['val_batches']*tr['batch_size']*world, tr['seed']+991)
    if world > 1 and len(data) % world:
        raise ValueError('samples_per_epoch must divide world size; avoid duplicated DDP patches')
    sampler = DistributedSampler(data, world, rank, seed=tr['seed']) if world > 1 else None
    vsampler = DistributedSampler(val, world, rank, shuffle=False) if world > 1 else None
    kwargs = dict(batch_size=tr['batch_size'], num_workers=tr['workers'], pin_memory=device.type == 'cuda')
    loader = DataLoader(data, sampler=sampler, shuffle=False, **kwargs)
    vloader = DataLoader(val, sampler=vsampler, shuffle=False, **kwargs)
    nc = data.archive.index['condition_channels']
    base = UNetV2(nc, **cfg['model'], mean_condition=stage == 'flow').to(device)
    model = DDP(base, device_ids=[local]) if world > 1 else base
    ema = deepcopy(base).eval().requires_grad_(False)
    out = Path(tr['output'])/f'{stage}_v2'
    out.mkdir(parents=True, exist_ok=True)
    if not resume and any(out.iterdir()):
        raise FileExistsError(f'{out} is not empty; resume explicitly or use a new v2 run')
    ckpt = torch.load(resume, map_location='cpu', weights_only=True) if resume else None
    if ckpt:
        check_checkpoint_v2(ckpt, data.archive, cfg, stage)
        def settings(value):
            return {k: v for k, v in value.items() if k not in ('output', 'device', 'workers')}
        if (settings(ckpt['config']['train']) != settings(tr) or ckpt['config']['loss'] != cfg['loss']
                or len(ckpt['rng']) != world):
            raise ValueError('Exact resume requires same training/loss settings and world size')
    mean_model, flow_scale, mean_hash = None, None, None
    if stage == 'flow':
        mean_model = UNetV2(nc, **cfg['model']).to(device).eval().requires_grad_(False)
        if ckpt:
            mean_model.load_state_dict(ckpt['regression_ema'])
            mean_hash = ckpt['regression_sha256']
            flow_scale = ckpt['flow_scale'].to(device)
        else:
            if not regression_checkpoint:
                raise ValueError('Flow training requires --regression-checkpoint')
            mean_ckpt = torch.load(regression_checkpoint, map_location='cpu', weights_only=True)
            check_checkpoint_v2(mean_ckpt, data.archive, cfg, 'regression')
            mean_model.load_state_dict(mean_ckpt['ema'])
            mean_hash = file_hash_v2(regression_checkpoint)
            flow_scale = calibrate_v2(mean_model, loader, cfg, device, world)
    opt = torch.optim.AdamW(base.parameters(), lr=tr['learning_rate'], weight_decay=tr['weight_decay'])
    epochs = tr[f'{stage}_epochs']
    total_steps = math.ceil(len(loader)/tr['accumulate'])*epochs
    def lr_scale(step):
        if step < tr['warmup_steps']:
            return (step+1)/max(1, tr['warmup_steps'])
        progress = min(1, (step-tr['warmup_steps'])/max(1, total_steps-tr['warmup_steps']))
        return tr['min_lr_ratio']+(1-tr['min_lr_ratio'])*(1+math.cos(math.pi*progress))/2
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_scale)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda' and tr['precision'] == 'fp16')
    start, step, best = 0, 0, float('inf')
    if ckpt:
        base.load_state_dict(ckpt['model'])
        ema.load_state_dict(ckpt['ema'])
        opt.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        scaler.load_state_dict(ckpt['scaler'])
        start, step, best = ckpt['epoch']+1, ckpt['step'], ckpt['best']
        torch.set_rng_state(ckpt['rng'][rank]['cpu'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state(ckpt['rng'][rank]['cuda'], device)
    if rank == 0:
        write_json(out/'config_v2.json', cfg)
        print(f'V2 {stage}: {sum(x.numel() for x in base.parameters()):,} parameters; world={world}; '
              f'effective_batch={tr["batch_size"]*tr["accumulate"]*world}', flush=True)
    names = ('total', 'value', 'gradient', 'multiscale', 't2m', 'precip', 'ps', 'u10m', 'v10m')
    for epoch in range(start, epochs):
        data.epoch = epoch
        if sampler:
            sampler.set_epoch(epoch)
        model.train()
        sums = torch.zeros(10, device=device, dtype=torch.float64)
        opt.zero_grad(set_to_none=True)
        for i, batch in enumerate(loader):
            b = to_device(batch, device)
            begin = (i//tr['accumulate'])*tr['accumulate']
            end = min(begin+tr['accumulate'], len(loader))
            count = min(end*tr['batch_size'], len(loader.sampler))-begin*tr['batch_size']
            do_step = i+1 == end
            sync = model.no_sync() if world > 1 and not do_step else nullcontext()
            with sync:
                with autocast(device, tr['precision']):
                    loss, metrics = loss_v2(model, b, cfg, stage, mean_model, flow_scale)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Nonfinite v2 {stage} loss in epoch {epoch}')
                scaler.scale(loss*len(b['target'])/count).backward()
            sums[:9] += metrics*len(b['target'])
            sums[9] += len(b['target'])
            if do_step:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(base.parameters(), tr['grad_clip'], error_if_nonfinite=not scaler.is_enabled())
                old_scale = scaler.get_scale()
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                if scaler.get_scale() >= old_scale:
                    scheduler.step()
                    step += 1
                    with torch.no_grad():
                        for ep, bp in zip(ema.parameters(), base.parameters()):
                            ep.lerp_(bp, 1-tr['ema_decay'])
        vsums = torch.zeros_like(sums)
        generator = torch.Generator(device=device).manual_seed(tr['seed']+10000+rank)
        with torch.no_grad():
            for batch in vloader:
                b = to_device(batch, device)
                with autocast(device, tr['precision']):
                    _, metrics = loss_v2(ema, b, cfg, stage, mean_model, flow_scale, generator)
                vsums[:9] += metrics*len(b['target'])
                vsums[9] += len(b['target'])
        if world > 1:
            dist.all_reduce(sums)
            dist.all_reduce(vsums)
        train_metrics, val_metrics = (sums[:9]/sums[9]).tolist(), (vsums[:9]/vsums[9]).tolist()
        if not all(math.isfinite(x) for x in train_metrics+val_metrics):
            raise FloatingPointError('Nonfinite v2 epoch metrics')
        improved = val_metrics[0] < best
        best = min(best, val_metrics[0])
        rng = {'cpu': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state(device) if device.type == 'cuda' else None}
        rng_all = [None]*world
        if world > 1:
            dist.all_gather_object(rng_all, rng)
        else:
            rng_all[0] = rng
        if rank == 0:
            row = dict(stage=stage, epoch=epoch+1, step=step, train=dict(zip(names, train_metrics)),
                       val=dict(zip(names, val_metrics)), learning_rate=opt.param_groups[0]['lr'])
            with open(out/'history_v2.jsonl', 'a') as f:
                f.write(json.dumps(row)+'\n')
            payload = dict(version='v2', stage=stage, model=base.state_dict(), ema=ema.state_dict(),
                           optimizer=opt.state_dict(), scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
                           epoch=epoch, step=step, best=best, rng=rng_all, config=cfg,
                           stats=data.archive.stats, fingerprint=data.archive.index['fingerprint'],
                           regression_ema=mean_model.state_dict() if mean_model is not None else None,
                           regression_sha256=mean_hash, flow_scale=flow_scale)
            atomic_save(out/'last_v2.pt', payload)
            if improved:
                atomic_save(out/'best_v2.pt', payload)
            if (epoch+1) % tr['checkpoint_interval'] == 0:
                atomic_save(out/f'epoch_{epoch+1:04d}_v2.pt', payload)
            print(json.dumps(row), flush=True)
    if world > 1:
        dist.destroy_process_group()
    # A resumed run may have an earlier best in the original directory.
    best_path = out/'best_v2.pt'
    if best_path.exists():
        return best_path
    last_path = out/'last_v2.pt'
    return last_path if last_path.exists() else Path(resume)
