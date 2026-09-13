"""Time the v2 patch loader and say where a sample's milliseconds go.

Training is GPU bound only if the loader can keep up. This reads real patches
from a prepared archive and reports a per-stage breakdown plus the implied epoch
time at a given worker count. Read-only; it submits nothing and writes nothing.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))

from merraflow.config_v2 import load_config_v2
from merraflow.dataset_v2 import PatchDatasetV2, crop_v2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--samples', type=int, default=64)
    parser.add_argument('--split', default='train')
    parser.add_argument('--workers', type=int, help='Worker count to project epoch time for; defaults to train.workers')
    args = parser.parse_args()
    torch.set_num_threads(1)
    cfg = load_config_v2(args.config)
    patch = cfg['patch']
    workers = args.workers or cfg['train']['workers']
    data = PatchDatasetV2(cfg['data']['prepared'], args.split, patch, args.samples, cfg['train']['seed'])
    archive = data.archive
    print(f"{len(data.entries)} {args.split} hours, cache "
          f"{'loaded' if data.cached_scores is not None else 'ABSENT (scores computed per sample)'}, "
          f"detail_fraction {data.detail}", flush=True)
    size, halo = patch['size'], patch['halo']
    width = size+2*halo
    broad = width*patch['context_scale']
    offset = (broad-size)//2
    totals = dict(proposal=0., local=0., context=0., downsample=0., target=0.)
    rng = np.random.default_rng(0)
    started = time.perf_counter()
    for _ in range(args.samples):
        entry = data.entries[rng.integers(len(data.entries))]
        mark = time.perf_counter()
        q = data.proposal(entry)
        totals['proposal'] += time.perf_counter()-mark
        i = rng.choice(len(q), p=q)
        y, x = data.yy[i], data.xx[i]
        mark = time.perf_counter()
        archive.condition(entry, y, x, size, halo)
        totals['local'] += time.perf_counter()-mark
        mark = time.perf_counter()
        context = archive.condition(entry, y-offset, x-offset, broad)
        totals['context'] += time.perf_counter()-mark
        mark = time.perf_counter()
        F.interpolate(torch.from_numpy(context)[None], size=(patch['context_size'],)*2, mode='area')
        totals['downsample'] += time.perf_counter()-mark
        mark = time.perf_counter()
        crop_v2(archive.array(entry, 'residual'), y, x, size, halo)
        totals['target'] += time.perf_counter()-mark
    wall = time.perf_counter()-started
    per = wall/args.samples
    print(f'\n{args.samples} samples in {wall:.1f}s = {per*1000:.0f} ms/sample single threaded')
    for name, value in sorted(totals.items(), key=lambda kv: -kv[1]):
        print(f'  {name:<11}{value/args.samples*1000:7.1f} ms  {value/wall*100:5.1f}%')
    context_mb = broad*broad*archive.index['condition_channels']*4/1e6
    print(f'\ncontext view builds {context_mb:.0f} MB per sample before downsampling to '
          f"{patch['context_size']}x{patch['context_size']}")
    epoch = patch['samples_per_epoch']*per/max(workers, 1)/60
    print(f"at {workers} workers, {patch['samples_per_epoch']} samples/epoch: ~{epoch:.0f} min/epoch "
          f'(perfect scaling; GPFS contention makes this optimistic)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
