from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
import json
import math
import os
import random
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from .config import write_json
from .dataset import PatchDataset
from .model import VelocityUNet, flow_loss


def device_for(request):
    if request == 'auto':
        request = 'cuda' if torch.cuda.is_available() else 'cpu'
    return torch.device(request)


def autocast(device, precision):
    if device.type != 'cuda' or precision == 'fp32':
        return nullcontext()
    if precision == 'bf16' and not torch.cuda.is_bf16_supported():
        raise ValueError('BF16 unsupported on this GPU; configure fp16 or fp32')
    return torch.autocast('cuda', dtype=torch.bfloat16 if precision == 'bf16' else torch.float16)


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def atomic_save(path, value):
    tmp = str(path)+'.tmp'
    torch.save(value, tmp)
    os.replace(tmp, path)


def train(cfg, resume=None):
    tr, p = cfg['train'], cfg['patch']
    rank, world, local = int(os.getenv('RANK', 0)), int(os.getenv('WORLD_SIZE', 1)), int(os.getenv('LOCAL_RANK', 0))
    device = device_for(tr['device'])
    if world > 1:
        if device.type != 'cuda':
            raise ValueError('Distributed production training requires CUDA')
        torch.cuda.set_device(local)
        device = torch.device('cuda', local)
        dist.init_process_group('nccl')
    seed = tr['seed']+rank
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
    data = PatchDataset(cfg['data']['prepared'], 'train', p['size'], p['halo'], p['samples_per_epoch'], tr['seed'])
    val = PatchDataset(cfg['data']['prepared'], 'val', p['size'], p['halo'], tr['val_batches']*tr['batch_size']*world, tr['seed']+991)
    sampler = DistributedSampler(data, num_replicas=world, rank=rank, seed=tr['seed']) if world > 1 else None
    val_sampler = DistributedSampler(val, num_replicas=world, rank=rank, shuffle=False) if world > 1 else None
    # Workers restart each epoch so deterministic epoch-dependent patch seeds propagate.
    kwargs = dict(batch_size=tr['batch_size'], num_workers=tr['workers'], pin_memory=device.type == 'cuda')
    loader = DataLoader(data, sampler=sampler, shuffle=False, **kwargs)
    val_loader = DataLoader(val, sampler=val_sampler, **kwargs)
    base = VelocityUNet(data.archive.index['condition_channels'], **cfg['model']).to(device)
    model = DDP(base, device_ids=[local]) if world > 1 else base
    # DDP broadcasts rank-zero initialization; copy EMA only after that broadcast.
    ema = deepcopy(base).eval().requires_grad_(False)
    opt = torch.optim.AdamW(base.parameters(), lr=tr['learning_rate'], weight_decay=tr['weight_decay'])
    steps_per_epoch = math.ceil(len(loader)/tr['accumulate'])
    total_steps = steps_per_epoch*tr['epochs']
    def lr_scale(step):
        if step < tr['warmup_steps']:
            return (step+1)/max(1, tr['warmup_steps'])
        fraction = min(1, (step-tr['warmup_steps'])/max(1, total_steps-tr['warmup_steps']))
        return tr['min_lr_ratio']+(1-tr['min_lr_ratio'])*.5*(1+math.cos(math.pi*fraction))
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_scale)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda' and tr['precision'] == 'fp16')
    out = Path(tr['output'])
    out.mkdir(parents=True, exist_ok=True)
    start, step, best = 0, 0, float('inf')
    if resume:
        ckpt = torch.load(resume, map_location='cpu', weights_only=True)
        if ckpt['fingerprint'] != data.archive.index['fingerprint'] or ckpt['config']['model'] != cfg['model'] or ckpt['config']['patch'] != cfg['patch']:
            raise ValueError('Checkpoint dataset/model/patch mismatch')
        previous_train = {k: v for k, v in ckpt['config']['train'].items() if k not in ('output', 'device', 'workers')}
        current_train = {k: v for k, v in tr.items() if k not in ('output', 'device', 'workers')}
        if previous_train != current_train or len(ckpt['rng']) != world:
            raise ValueError('Exact epoch-boundary resume requires same training settings and world size')
        base.load_state_dict(ckpt['model'])
        ema.load_state_dict(ckpt['ema'])
        opt.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        scaler.load_state_dict(ckpt['scaler'])
        start, step, best = ckpt['epoch']+1, ckpt['step'], ckpt['best']
        torch.set_rng_state(ckpt['rng'][rank]['cpu'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state(ckpt['rng'][rank]['cuda'], device)
    elif (out/'last.pt').exists():
        raise FileExistsError('Run already exists; use --resume or a new train.output')
    if rank == 0:
        write_json(out/'config.json', cfg)
        write_json(out/'stats.json', data.archive.stats)
        print(f'{sum(v.numel() for v in base.parameters()):,} parameters; {device}; world={world}; effective batch={tr["batch_size"]*tr["accumulate"]*world}', flush=True)
    weights = tr['channel_weights']
    if len(weights) != 4 or min(weights) <= 0:
        raise ValueError('Need four positive channel weights')
    for epoch in range(start, tr['epochs']):
        data.epoch = epoch
        if sampler:
            sampler.set_epoch(epoch)
        model.train()
        sums = torch.zeros(2, device=device, dtype=torch.float64)
        opt.zero_grad(set_to_none=True)
        for i, batch in enumerate(loader):
            batch = to_device(batch, device)
            group_start = (i//tr['accumulate'])*tr['accumulate']
            group_end = min(group_start+tr['accumulate'], len(loader))
            # Weight a short final batch by its actual sample count.
            group_samples = min(group_end*tr['batch_size'], len(loader.sampler))-group_start*tr['batch_size']
            do_step = i+1 == group_end
            sync = model.no_sync() if world > 1 and not do_step else nullcontext()
            with sync:
                with autocast(device, tr['precision']):
                    loss = flow_loss(model, batch, p['halo'], weights)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Nonfinite training loss in epoch {epoch}')
                scaler.scale(loss*batch['target'].shape[0]/group_samples).backward()
            sums[0] += loss.detach()*batch['target'].shape[0]
            sums[1] += batch['target'].shape[0]
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
        ema.eval()
        vsums = torch.zeros(2, device=device, dtype=torch.float64)
        gen = torch.Generator(device=device).manual_seed(tr['seed']+10000+rank)
        with torch.no_grad():
            for batch in val_loader:
                batch = to_device(batch, device)
                with autocast(device, tr['precision']):
                    loss = flow_loss(ema, batch, p['halo'], weights, generator=gen)
                vsums[0] += loss*batch['target'].shape[0]
                vsums[1] += batch['target'].shape[0]
        if world > 1:
            dist.all_reduce(sums)
            dist.all_reduce(vsums)
        train_loss, val_loss = (sums[0]/sums[1]).item(), (vsums[0]/vsums[1]).item()
        if not math.isfinite(val_loss):
            raise FloatingPointError('Nonfinite validation loss')
        improved = val_loss < best
        best = min(best, val_loss)
        rng = {'cpu': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state(device) if device.type == 'cuda' else None}
        rng_all = [None]*world
        if world > 1:
            dist.all_gather_object(rng_all, rng)
        else:
            rng_all[0] = rng
        if rank == 0:
            row = dict(epoch=epoch, step=step, train_loss=train_loss, val_loss=val_loss, learning_rate=opt.param_groups[0]['lr'],
                       peak_vram_gb=torch.cuda.max_memory_allocated(device)/1e9 if device.type == 'cuda' else 0.)
            with open(out/'history.jsonl', 'a') as f:
                f.write(json.dumps(row)+'\n')
            payload = dict(model=base.state_dict(), ema=ema.state_dict(), optimizer=opt.state_dict(), scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
                           epoch=epoch, step=step, best=best, rng=rng_all, config=cfg, stats=data.archive.stats, fingerprint=data.archive.index['fingerprint'])
            atomic_save(out/'last.pt', payload)
            if improved:
                atomic_save(out/'best.pt', payload)
            print(json.dumps(row), flush=True)
    if world > 1:
        dist.destroy_process_group()
    return out/'best.pt'
