"""v4.1: pack the training archive once, then train v4's flow on the fast loader."""
import argparse
from contextlib import contextmanager
from pathlib import Path
import json
import time
import numpy as np
import torch
from .v4 import ArchiveV4
from .v4_1 import load_config, base_config, PackedArchive, DatasetV41, check_checkpoint


def reference_sample(archive, entry, y, x, cfg):
    """v4's DatasetV4.__getitem__ body for a fixed hour and origin (slow path)."""
    from .dataset_v2 import crop_v2
    from .dataset_v3_precip import encode_rain
    p, a = cfg['patch'], archive
    b = a.inputs_with_original(entry, y, x, p)
    truth = crop_v2(a.array(entry, 'truth'), y, x, p['size'], p['halo']).copy()
    truth[1:2] = a.rain(entry, 'truth', y, x, p['size'], p['halo'])
    coarse = crop_v2(a.array(entry, 'baseline'), y, x, p['size'], p['halo']).copy()
    humidity = crop_v2(a.humidity(entry), y, x, p['size'], p['halo'])
    qcoarse = crop_v2(a.array(entry, 'condition')[a.q_index:a.q_index+1], y, x, p['size'], p['halo'])
    truth = np.concatenate([truth, humidity])
    coarse = np.concatenate([coarse, qcoarse])
    target = (truth-coarse-a.target_rm)/a.target_rs
    target[1:2] = encode_rain(truth[1:2], a.scale)
    area = crop_v2(a.static['area'], y, x, p['size']).copy()
    full = crop_v2(a.static['area'], y, x, p['size'], p['halo']).copy()
    b.update(target=torch.from_numpy(target), truth=torch.from_numpy(truth), coarse=torch.from_numpy(coarse),
             area=torch.from_numpy(area/area.mean()), area_full=torch.from_numpy(full/area.mean()),
             importance=torch.tensor(1.))
    return b


def compare(packed_sample, reference, context_atol=0.):
    """Largest absolute difference per key; raises if any key differs."""
    report = {}
    for key, value in reference.items():
        other = packed_sample[key]
        if other.shape != value.shape or other.dtype != value.dtype:
            raise ValueError(f'{key}: packed {tuple(other.shape)}/{other.dtype} vs v4 {tuple(value.shape)}/{value.dtype}')
        difference = float((other.double()-value.double()).abs().max()) if value.numel() else 0.
        report[key] = difference
        tolerance = context_atol if 'context' in key else 0.
        if not difference <= tolerance:
            raise ValueError(f'{key}: packed sample differs from v4 by {difference:g}')
    return report


def plan_packed(cfg):
    from .packed_v4_1 import plan, geometry
    archive = ArchiveV4(base_config(cfg), verify_files=False)
    entries = plan(cfg, archive)
    geo = geometry(cfg, archive.shape, len(archive.cm))
    months = sorted({e['time'][:7] for e in entries})
    summary = dict(hours=len(entries), supervised=sum(e['targets'] for e in entries), months=months,
                   bytes_per_hour=geo['file_bytes'], terabytes=round(geo['file_bytes']*len(entries)/1e12, 3),
                   candidates=geo['candidates'], pooled_grid=geo['pooled'], pooling_factor=geo['factor'])
    return summary


def preflight(cfg, resume=None, full=False, samples=2, data_only=False):
    """Run once on a CPU node after packing (the prepare job's finalize stage
    does this); training jobs never run it, so GPU time starts on batch one."""
    from .precip_direct_v2 import original_checkpoint
    print(f'Preflight v4.1: {"full packed-file audit" if full else "quick manifest check"}...', flush=True)
    archive = ArchiveV4(base_config(cfg), verify_files=False)
    packed = PackedArchive(cfg, archive, verify_files=full)
    for split in ('train', 'val'):
        print(f'{split}: {len(packed.entries(split))} packed hours', flush=True)
    if data_only:
        pass
    elif resume:
        check_checkpoint(torch.load(resume, map_location='cpu', weights_only=True), cfg, archive, packed.fingerprint)
    else:
        original_checkpoint(cfg['conditioning']['checkpoint'], archive)
        out = Path(cfg['train']['output'])
        if out.exists() and any(out.iterdir()):
            raise FileExistsError(f'Existing v4.1 output: {out}; use --resume {out}/last_v4_1.pt')
    # Real-data equality against the v4 slow path, including one edge origin.
    for split in ('train', 'val'):
        data = DatasetV41(cfg, split, max(samples, 1), cfg['train']['seed'], packed=packed)
        locations = [data.locate(i) for i in range(samples)]
        entry = locations[0][0]
        locations.append((entry, int(data.yy.max()), int(data.xx.max())))
        for entry, y, x in locations:
            report = compare(packed.sample(entry, y, x), reference_sample(archive, entry, y, x, cfg), 1e-5)
            print(f'Preflight {split} {entry["id"]} origin=({y},{x}) matches v4 '
                  f'(max context diff {max(v for k, v in report.items() if "context" in k):.2g})', flush=True)
    print('V4.1 packed data validated.', flush=True)


def benchmark(cfg, samples=64, workers=None, batches=40, compare_v4=0):
    """Loader throughput on this node; says whether the GPUs can be kept busy."""
    from torch.utils.data import DataLoader
    from .v4_1 import EpochSampler
    from .train_v4_1 import loader_options
    tr = cfg['train']
    archive = ArchiveV4(base_config(cfg), verify_files=False)
    packed = PackedArchive(cfg, archive)
    size = max(cfg['patch']['samples_per_epoch'], 4*tr['batch_size']*(batches+1), 4*(samples+compare_v4))
    data = DatasetV41(cfg, 'train', size, tr['seed']+12345, packed=packed)
    started = time.monotonic()
    for i in range(samples):
        data[i]
    single = (time.monotonic()-started)/samples
    print(f'Single process: {single*1e3:.1f} ms/sample ({1/single:.1f} samples/s)', flush=True)
    if compare_v4:
        started = time.monotonic()
        for i in range(compare_v4):
            reference_sample(archive, *data.locate(samples+i), cfg)
        slow = (time.monotonic()-started)/compare_v4
        print(f'v4 slow path: {slow*1e3:.1f} ms/sample; packed is {slow/single:.1f}x faster per sample', flush=True)
    workers = tr['workers'] if workers is None else workers
    options = loader_options(tr, torch.device('cpu'), workers)
    sampler = EpochSampler(len(data), 0, 4)
    loader = iter(DataLoader(data, sampler=sampler, batch_size=tr['batch_size'], **options))
    next(loader)
    started = time.monotonic()
    for _ in range(batches):
        next(loader)
    rate = batches*tr['batch_size']/(time.monotonic()-started)
    epoch_min = cfg['patch']['samples_per_epoch']/(4*rate)/60
    result = dict(ms_per_sample_single=round(single*1e3, 2), workers=workers,
                  loader_samples_per_s_per_rank=round(rate, 1),
                  data_bound_epoch_minutes_4gpu=round(epoch_min, 2))
    print(json.dumps(result), flush=True)
    return result


@contextmanager
def v4_inference_for(saved_cfg):
    """Run v4's full-domain inference with a (checked) v4.1 checkpoint."""
    from unittest import mock
    from . import inference_v4
    with mock.patch.object(inference_v4, 'check_checkpoint', lambda *args, **kwargs: None):
        yield inference_v4


def predict(cfg, checkpoint, split='test', limit=1, timestamp=None):
    base = base_config(cfg)
    archive = ArchiveV4(base)
    check_checkpoint(torch.load(checkpoint, map_location='cpu', weights_only=True), cfg, archive)
    with v4_inference_for(cfg) as inference:
        return inference.predict(base, checkpoint, split, limit, timestamp)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['plan-packed', 'prepare-packed', 'finalize-packed', 'preflight',
                                            'benchmark', 'train', 'predict'])
    parser.add_argument('--config', default='configs/discover_v4_1.yaml')
    parser.add_argument('--month', help='YYYY-MM to pack (prepare-packed)')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--resume')
    parser.add_argument('--checkpoint')
    parser.add_argument('--full', action='store_true', help='Audit every packed hour file during preflight')
    parser.add_argument('--data-only', action='store_true', help='Preflight: skip checkpoint/output checks')
    parser.add_argument('--samples', type=int, default=64)
    parser.add_argument('--workers', type=int)
    parser.add_argument('--batches', type=int, default=40)
    parser.add_argument('--compare-v4', type=int, default=0)
    parser.add_argument('--split', choices=['val', 'test'], default='test')
    parser.add_argument('--time')
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.command == 'plan-packed':
        print(json.dumps(plan_packed(cfg), indent=2))
    elif args.command == 'prepare-packed':
        from .packed_v4_1 import prepare
        started = time.monotonic()
        result = prepare(cfg, ArchiveV4(base_config(cfg), verify_files=False), args.month, args.limit)
        result['seconds'] = round(time.monotonic()-started, 1)
        print(json.dumps(result), flush=True)
    elif args.command == 'finalize-packed':
        from .packed_v4_1 import finalize
        print(json.dumps(finalize(cfg, ArchiveV4(base_config(cfg), verify_files=False))), flush=True)
    elif args.command == 'preflight':
        preflight(cfg, args.resume, full=args.full, data_only=args.data_only)
    elif args.command == 'benchmark':
        benchmark(cfg, args.samples, args.workers, args.batches, args.compare_v4)
    elif args.command == 'train':
        from .train_v4_1 import train
        print(train(cfg, args.resume))
    else:
        if not args.checkpoint or (args.limit or 1) < 1:
            parser.error('predict requires --checkpoint')
        print(predict(cfg, args.checkpoint, args.split, args.limit or 1, args.time))


if __name__ == '__main__':
    main()
