#!/usr/bin/env python3
"""Measure both v2 stages without writing checkpoints or altering training runs."""
import argparse
import json
import time
import torch
from merraflow.config_v2 import load_config_v2
from merraflow.dataset_v2 import PatchDatasetV2
from merraflow.model_v2 import UNetV2
from merraflow.loss_v2 import loss_v2
from merraflow.train import autocast, device_for, to_device
from merraflow.physics_v2 import precipitation_representation_v2
from merraflow.rain_rollout_v2 import RainFlowObjectiveV2
from merraflow.inference_v2 import load_models_v2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/discover_v2.yaml')
    parser.add_argument('--steps', type=int, default=5)
    parser.add_argument('--stage', choices=('both', 'regression', 'flow'), default='both')
    parser.add_argument('--initialize-flow')
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('--steps must be positive')
    cfg = load_config_v2(args.config)
    if args.initialize_flow and args.stage != 'flow':
        parser.error('--initialize-flow requires --stage flow')
    device = device_for(cfg['train']['device'])
    data = PatchDatasetV2(cfg['data']['prepared'], 'train', cfg['patch'], cfg['train']['batch_size'],
                          precipitation_representation=precipitation_representation_v2(cfg))
    batch = to_device(next(iter(torch.utils.data.DataLoader(data, batch_size=len(data)))), device)
    nc = data.archive.index['condition_channels']
    mean = UNetV2(nc, **cfg['model']).to(device)
    for stage in (('regression', 'flow') if args.stage == 'both' else (args.stage,)):
        scale = torch.ones(1, 5, 1, 1, device=device)
        if stage == 'flow':
            mean.eval().requires_grad_(False)
            model = UNetV2(nc, **cfg['model'], mean_condition=True).to(device)
            if args.initialize_flow:
                mean, model, scale, _ = load_models_v2(cfg, args.initialize_flow, data.archive, device,
                                                     allow_sampling_change=True)
                mean.requires_grad_(False)
                if model is None:
                    raise ValueError('Benchmark initialization must be a flow checkpoint')
                model.requires_grad_(True)
            model.train()
        else:
            model = mean
        objective = RainFlowObjectiveV2(model, cfg) if stage == 'flow' and cfg['train'].get('rain_rollout') else None
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['train']['learning_rate'])
        scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda' and cfg['train']['precision'] == 'fp16')
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        elapsed = []
        for step in range(args.steps+1):
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            with autocast(device, cfg['train']['precision']):
                if objective:
                    loss, _, _ = objective(batch, mean, scale, 1.)
                else:
                    loss, _ = loss_v2(model, batch, cfg, stage, mean if stage == 'flow' else None, scale)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite benchmark objective')
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            if step:
                elapsed.append(time.perf_counter()-start)
        print(json.dumps(dict(stage=stage, device=str(device), batch=len(data),
                              parameters=sum(p.numel() for p in model.parameters()),
                              seconds_per_step=sum(elapsed)/len(elapsed),
                              peak_allocated_gb=torch.cuda.max_memory_allocated(device)/1e9 if device.type == 'cuda' else None,
                              generated_rain_training=objective is not None,
                              note='Disposable model copies; includes full-strength rollout when configured. No checkpoint is modified.')))


if __name__ == '__main__':
    main()
