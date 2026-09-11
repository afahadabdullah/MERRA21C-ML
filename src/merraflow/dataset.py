from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from .prepare import time_features, PREPARATION_FORMAT
from .physics import transform_target


def read_json(path):
    with open(path) as f:
        return json.load(f)


def crop(array, y, x, size, halo=0):
    """Read only needed pixels; replicate outside-domain boundaries."""
    h, w = array.shape[-2:]
    iy = np.clip(np.arange(y-halo, y+size+halo), 0, h-1)
    ix = np.clip(np.arange(x-halo, x+size+halo), 0, w-1)
    return np.asarray(array[..., iy[:, None], ix[None, :]], dtype='float32')


class Archive:
    def __init__(self, root):
        self.root = Path(root)
        self.index = read_json(self.root/'index.json')
        if self.index.get('format') != PREPARATION_FORMAT:
            raise ValueError('Legacy prepared archive: rebuild with matched HWT PRECTOT in a new '
                             'directory, recompute statistics and retrain; APCP targets cannot be reused')
        self.stats = read_json(self.root/'stats.json')
        with np.load(self.root/'static.npz') as f:
            self.static = {k: f[k] for k in f.files}
        self.shape = self.static['area'].shape
        self.cm = np.array(self.stats['condition']['mean'], dtype='float32')[:, None, None]
        self.cs = np.array(self.stats['condition']['std'], dtype='float32')[:, None, None]
        self.rm = np.array(self.stats['residual']['mean'], dtype='float32')[:, None, None]
        self.rs = np.array(self.stats['residual']['std'], dtype='float32')[:, None, None]

    def array(self, entry, name):
        return np.load(self.root/entry['id']/f'{name}.npy', mmap_mode='r')

    def condition(self, entry, y, x, size, halo):
        dyn = (crop(self.array(entry, 'condition'), y, x, size, halo)-self.cm)/self.cs
        fixed = crop(self.static['features'], y, x, size, halo)
        temporal = time_features(entry['time'], crop(self.static['lon'], y, x, size, halo))
        base = crop(self.array(entry, 'baseline'), y, x, size, halo)
        base = transform_target(base, self.stats['precip_log_scale'], self.stats['wind_log_scale'])
        base[0] = (base[0]-280)/20
        base[2] = (base[2]-90000)/15000
        return np.concatenate([dyn, fixed, temporal, base]).astype('float32')


class PatchDataset(Dataset):
    def __init__(self, root, split, size, halo, samples, seed=0):
        self.archive = Archive(root)
        self.entries = [e for e in self.archive.index['entries'] if e['split'] == split]
        if not self.entries or min(self.archive.shape) < size:
            raise ValueError('Empty split or patch larger than domain')
        self.size, self.halo, self.samples, self.seed, self.epoch = size, halo, samples, seed, 0

    def __len__(self):
        return self.samples

    def __getitem__(self, index):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, int(index)]))
        e = self.entries[rng.integers(len(self.entries))]
        y = rng.integers(self.archive.shape[0]-self.size+1)
        x = rng.integers(self.archive.shape[1]-self.size+1)
        c = self.archive.condition(e, y, x, self.size, self.halo)
        a = self.archive
        target = (crop(a.array(e, 'residual'), y, x, self.size, self.halo)-a.rm)/a.rs
        area = crop(a.static['area'], y, x, self.size)
        return {'condition': torch.from_numpy(c), 'target': torch.from_numpy(target),
                'area': torch.from_numpy(area/area.mean())}
