"""V2 patches with local HR detail, broad coarse context, and unbiased proposals."""
from pathlib import Path
import json
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from .dataset import crop
from .prepare import time_features
from .physics_v2 import transform_v2


def crop_v2(array, y, x, size, halo=0):
    """Identical to crop, but one contiguous read when the window is inside the grid.

    Fancy indexing gathers element by element, which dominates the cost of the
    576-pixel context view; interior windows are the overwhelming majority.
    """
    h, w = array.shape[-2:]
    y0, x0, y1, x1 = y-halo, x-halo, y+size+halo, x+size+halo
    if y0 >= 0 and x0 >= 0 and y1 <= h and x1 <= w:
        return np.asarray(array[..., y0:y1, x0:x1], dtype='float32')
    return crop(array, y, x, size, halo)


def proposal_cache_paths(root, patch):
    """Scores depend on the archive and on the candidate grid, so key them by both."""
    stem = Path(root)/'_proposals_v2'/f"size{patch['size']}_stride{patch['sampling_stride']}"
    return stem.with_suffix('.npy'), stem.with_suffix('.json')


def load_proposal_cache(root, patch, candidates, shape):
    """Return (scores, {entry id: row}) or (None, {}) when no usable cache exists."""
    scores_path, meta_path = proposal_cache_paths(root, patch)
    if not (scores_path.exists() and meta_path.exists()):
        return None, {}
    meta = json.loads(meta_path.read_text())
    if (meta.get('size') != patch['size'] or meta.get('sampling_stride') != patch['sampling_stride']
            or meta.get('candidates') != candidates or list(meta.get('shape', ())) != list(shape)):
        return None, {}
    scores = np.load(scores_path, mmap_mode='r')
    if scores.shape != (len(meta['ids']), candidates):
        return None, {}
    return scores, {entry_id: row for row, entry_id in enumerate(meta['ids'])}


class ArchiveV2:
    def __init__(self, root):
        self.root = Path(root)
        self.index = json.loads((self.root/'index_v2.json').read_text())
        if self.index.get('format') != 'v2':
            raise ValueError('Require a v2 prepared archive')
        self.stats = json.loads((self.root/'stats_v2.json').read_text())
        with np.load(self.root/'static_v2.npz') as f:
            self.static = {k: f[k] for k in f.files}
        self.shape = self.static['area'].shape
        self.cm, self.cs = [np.asarray(self.stats['condition'][k], dtype='float32')[:, None, None] for k in ('mean', 'std')]
        self.rm, self.rs = [np.asarray(self.stats['residual'][k], dtype='float32')[:, None, None] for k in ('mean', 'std')]

    def array(self, entry, name):
        return np.load(self.root/entry['id']/f'{name}_v2.npy', mmap_mode='r')

    def condition(self, entry, y, x, size, halo=0):
        dyn = (crop_v2(self.array(entry, 'condition'), y, x, size, halo)-self.cm)/self.cs
        fixed = crop_v2(self.static['features'], y, x, size, halo)
        temporal = time_features(entry['time'], crop_v2(self.static['lon'], y, x, size, halo))
        base = transform_v2(crop_v2(self.array(entry, 'baseline'), y, x, size, halo), self.stats['precip_log_scale'])
        base[0] = (base[0]-280)/20
        base[2] = (base[2]-90000)/15000
        base[3:] /= 10
        return np.concatenate([dyn, fixed, temporal, base]).astype('float32')

    def inputs(self, entry, y, x, patch):
        size, halo = patch['size'], patch['halo']
        local = self.condition(entry, y, x, size, halo)
        width = size+2*halo
        broad = width*patch['context_scale']
        offset = (broad-size)//2
        context = self.condition(entry, y-offset, x-offset, broad)
        # Area downsampling gives a broad view; local input remains native resolution.
        context = F.interpolate(torch.from_numpy(context)[None], size=(patch['context_size'],)*2, mode='area')[0]
        return torch.from_numpy(local), context


def candidates_v2(shape, size, stride):
    axes = [np.unique(np.r_[np.arange(0, n-size+1, stride), n-size]) for n in shape]
    if min(shape) < size:
        raise ValueError('Patch exceeds archive grid')
    yy, xx = np.meshgrid(*axes, indexing='ij')
    return yy.ravel().astype(int), xx.ravel().astype(int)


def box_means_v2(field, yy, xx, size):
    integral = np.pad(np.asarray(field, dtype='float64'), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    return (integral[yy+size, xx+size]-integral[yy, xx+size]-integral[yy+size, xx]+integral[yy, xx])/(size*size)


class PatchDatasetV2(Dataset):
    def __init__(self, root, split, patch, samples, seed=0):
        self.archive = ArchiveV2(root)
        self.entries = [e for e in self.archive.index['entries'] if e['split'] == split]
        if not self.entries:
            raise ValueError(f'Empty {split} split')
        self.patch, self.samples, self.seed, self.epoch = patch, samples, seed, 0
        self.detail = patch['detail_fraction'] if split == 'train' else 0.
        self.yy, self.xx = candidates_v2(self.archive.shape, patch['size'], patch['sampling_stride'])
        land = self.archive.static['land_fraction']
        coast = np.abs(np.gradient(land, axis=0))+np.abs(np.gradient(land, axis=1))
        self.coast = box_means_v2(coast, self.yy, self.xx, patch['size'])
        self.proposals = {}
        # Precomputed rain scores turn a full-field read plus a double cumsum per
        # sample into one 4-byte-per-candidate row. Missing or mismatched caches
        # fall back to computing the same value on demand.
        self.cached_scores, self.cached_rows = load_proposal_cache(
            self.archive.root, patch, len(self.yy), self.archive.shape)

    def __len__(self):
        return self.samples

    def rain_score(self, entry):
        row = self.cached_rows.get(entry['id'])
        if row is not None:
            return np.asarray(self.cached_scores[row], dtype='float64')
        rain = np.asarray(self.archive.array(entry, 'truth')[1])
        return box_means_v2(np.log1p(rain), self.yy, self.xx, self.patch['size'])

    def proposal(self, entry):
        n = len(self.yy)
        if not self.detail:
            return np.full(n, 1/n)
        if entry['id'] not in self.proposals:
            score = self.rain_score(entry)+5*self.coast+.01
            self.proposals[entry['id']] = (1-self.detail)/n+self.detail*score/score.sum()
        return self.proposals[entry['id']]

    def __getitem__(self, index):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, int(index)]))
        entry = self.entries[rng.integers(len(self.entries))]
        q = self.proposal(entry)
        i = rng.choice(len(q), p=q)
        y, x = self.yy[i], self.xx[i]
        local, context = self.archive.inputs(entry, y, x, self.patch)
        a, p = self.archive, self.patch
        target = (crop_v2(a.array(entry, 'residual'), y, x, p['size'], p['halo'])-a.rm)/a.rs
        area = crop_v2(a.static['area'], y, x, p['size'])
        return dict(condition=local, context=context, target=torch.from_numpy(target),
                    area=torch.from_numpy(area/area.mean()),
                    importance=torch.tensor(1/(len(q)*q[i]), dtype=torch.float32))
