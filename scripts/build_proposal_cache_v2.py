"""Precompute the rain part of the v2 patch proposal for every prepared hour.

Sampling needs box means of log1p(rain) over the fixed candidate grid. Computing
them inside the data loader costs a full-field read and a float64 double cumsum
per sample, which dominates training time. They depend only on the archive and on
(patch size, sampling stride), so they are computed once here and memory-mapped
during training. Rerun after preparing more hours; nothing else is modified.
"""
import argparse
import json
import os
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))

from merraflow.config_v2 import load_config_v2
from merraflow.dataset_v2 import ArchiveV2, box_means_v2, candidates_v2, proposal_cache_paths, rain_edge_scores_v2

_STATE = {}


def _init(root, yy, xx, size, kind='rain'):
    _STATE.update(root=Path(root), yy=yy, xx=xx, size=size, kind=kind)


def _score(entry_id):
    rain = np.asarray(np.load(_STATE['root']/entry_id/'truth_v2.npy', mmap_mode='r')[1])
    if _STATE['kind'] == 'structure':
        return rain_edge_scores_v2(rain, _STATE['yy'], _STATE['xx'], _STATE['size']).astype('float32')
    return box_means_v2(np.log1p(rain), _STATE['yy'], _STATE['xx'], _STATE['size']).astype('float32')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--kind', choices=('rain', 'structure'), default='rain')
    parser.add_argument('--force', action='store_true', help='Rebuild even if a matching cache exists')
    args = parser.parse_args()
    cfg = load_config_v2(args.config)
    archive = ArchiveV2(cfg['data']['prepared'])
    patch = cfg['patch']
    yy, xx = candidates_v2(archive.shape, patch['size'], patch['sampling_stride'])
    ids = [entry['id'] for entry in archive.index['entries'] if args.kind != 'structure' or entry['split'] == 'train']
    scores_path, meta_path = proposal_cache_paths(archive.root, patch, args.kind)
    if scores_path.exists() and meta_path.exists() and not args.force:
        meta = json.loads(meta_path.read_text())
        if meta.get('ids') == ids and meta.get('candidates') == len(yy):
            print(f'{scores_path} already covers {len(ids)} hours; use --force to rebuild')
            return 0
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    print(f'{len(ids)} hours, {len(yy)} candidates, patch size {patch["size"]}, '
          f'stride {patch["sampling_stride"]}, {args.workers} workers', flush=True)
    scores = np.empty((len(ids), len(yy)), dtype='float32')
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init,
                             initargs=(str(archive.root), yy, xx, patch['size'], args.kind)) as pool:
        for position, row in enumerate(pool.map(_score, ids, chunksize=8)):
            scores[position] = row
            if position % 500 == 0:
                elapsed = time.time()-started
                print(f'{position+1}/{len(ids)} in {elapsed:.0f}s', flush=True)
    if not np.isfinite(scores).all():
        raise ValueError('Nonfinite proposal scores; check the prepared truth shards')
    handle, temporary = tempfile.mkstemp(dir=scores_path.parent, suffix='.npy')
    os.close(handle)
    np.save(temporary, scores)  # the suffix is already .npy, so no extension is appended
    os.replace(temporary, scores_path)
    meta_path.write_text(json.dumps(
        {'size': patch['size'], 'sampling_stride': patch['sampling_stride'],
         'candidates': int(len(yy)), 'shape': list(archive.shape), 'ids': ids, 'kind': args.kind}, indent=2))
    print(f'{scores_path} written in {time.time()-started:.0f}s '
          f'({scores.nbytes/1e6:.1f} MB); {meta_path} lists {len(ids)} hours')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
