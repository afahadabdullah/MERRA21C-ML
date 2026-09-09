#!/usr/bin/env python3
"""Measure single-GPU training memory with synthetic patches, including EMA/AdamW.

Run inside an A100 allocation after installing the package. This measures memory
and compute, not data-loader I/O or model skill. DDP adds communication overhead.
"""
import argparse
from copy import deepcopy
from time import perf_counter
import torch
from merraflow.config import load_config, write_json
from merraflow.model import VelocityUNet, flow_loss
from merraflow.train import autocast


def one(cfg, batch_size, steps):
    p = cfg['patch']
    device = torch.device('cuda')
    torch.cuda.reset_peak_memory_stats()
    model = VelocityUNet(len(cfg['data']['predictors'])+16, **cfg['model']).to(device).train()
    ema = deepcopy(model).eval().requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['train']['learning_rate'])
    scaler = torch.amp.GradScaler('cuda', enabled=cfg['train']['precision'] == 'fp16')
    side = p['size']+2*p['halo']
    batch = {'target': torch.randn(batch_size, 4, side, side, device=device),
             'condition': torch.randn(batch_size, len(cfg['data']['predictors'])+16, side, side, device=device),
             'area': torch.ones(batch_size, p['size'], p['size'], device=device)}
    times = []
    for step in range(steps+1):
        torch.cuda.synchronize()
        start = perf_counter()
        opt.zero_grad(set_to_none=True)
        with autocast(device, cfg['train']['precision']):
            loss = flow_loss(model, batch, p['halo'], cfg['train']['channel_weights'])
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['train']['grad_clip'])
        scaler.step(opt)
        scaler.update()
        with torch.no_grad():
            for a, b in zip(ema.parameters(), model.parameters()):
                a.lerp_(b, 1-cfg['train']['ema_decay'])
        torch.cuda.synchronize()
        if step:
            times.append(perf_counter()-start)
    total = torch.cuda.get_device_properties(0).total_memory
    peak = torch.cuda.max_memory_allocated()
    return {'batch_size': batch_size, 'status': 'ok', 'seconds_per_microbatch_with_optimizer': sum(times)/len(times),
            'peak_allocated_gb': peak/1e9, 'peak_reserved_gb': torch.cuda.max_memory_reserved()/1e9,
            'device_total_gb': total/1e9, 'allocated_fraction': peak/total}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/discover.yaml')
    p.add_argument('--batches', default='1,2,4')
    p.add_argument('--steps', type=int, default=3)
    p.add_argument('--output', default='runs/a100_benchmark.json')
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit('Run this benchmark in a CUDA GPU allocation')
    if args.steps < 1:
        raise ValueError('steps must be positive')
    cfg = load_config(args.config)
    rows = []
    for batch_size in map(int, args.batches.split(',')):
        if batch_size < 1:
            raise ValueError('batch sizes must be positive')
        torch.cuda.empty_cache()
        try:
            row = one(cfg, batch_size, args.steps)
        except torch.cuda.OutOfMemoryError:
            row = {'batch_size': batch_size, 'status': 'out_of_memory'}
        rows.append(row)
        print(row, flush=True)
    write_json(args.output, {'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__, 'config': cfg,
                            'scope': 'single-GPU synthetic compute/memory only; allow headroom for production', 'results': rows})


if __name__ == '__main__':
    main()
