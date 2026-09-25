"""v4.1: the v4 six-field flow with a packed, I/O-light training data pipeline.

Model, targets, frozen regression, calibration, loss and validation are v4's.
What changes:

* samples come from the packed archive (``packed_v4_1``): a few contiguous
  reads per sample instead of ~23 MB of strided memory-map faults;
* the broad context is read pre-pooled, never rebuilt from 576x576 crops;
* candidate origins sit on multiples of the pooling factor (sampling_stride 24
  in production), which is what makes the pre-pooled context exact;
* loader workers persist across epochs (the epoch is encoded in the index).

Every sample equals v4's ``DatasetV4`` sample for the same hour and origin, so
v4.1 checkpoints run through the unchanged v4 full-domain inference.
"""
from copy import deepcopy
from functools import partial
from pathlib import Path
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset, Sampler
from .dataset_v2 import box_means_v2, crop_v2, time_features
from .dataset_v3_precip import encode_rain
from .physics_v2 import encode_fields_v2
from .packed_v4_1 import SCORES, BASELINE, TRUTH, HourFiles, candidates_v4_1, load_manifest
from .v4 import TARGETS, validate_config as validate_v4, ArchiveV4

VERSION = 'v4.1'
TRAIN_ONLY_KEYS = ('val_workers', 'fd_cache_size', 'persistent_workers', 'checkpoint_interval',
                   'validation_plot_samples')


def base_config(cfg):
    """The equivalent v4 configuration (model, data, targets, inference)."""
    base = deepcopy(cfg)
    base['version'] = 'v4'
    base.pop('packed', None)
    return base


def load_config(path):
    return validate_config(yaml.safe_load(Path(path).read_text()))


def validate_config(cfg):
    if cfg.get('version') != VERSION:
        raise ValueError('Require version: v4.1')
    validate_v4(base_config(cfg))
    pk, p, tr = cfg.get('packed') or {}, cfg['patch'], cfg['train']
    if not pk.get('root'):
        raise ValueError('Missing packed.root')
    splits = pk.get('splits')
    if not isinstance(splits, list) or not {'train', 'val'} <= set(splits) or len(set(splits)) != len(splits):
        raise ValueError('packed.splits must list train and val once each')
    if not set(splits) <= {'train', 'val', 'test'}:
        raise ValueError('Unknown packed split')
    for key in ('tile', 'context_tile'):
        if type(pk.get(key)) is not int or pk[key] < 4:
            raise ValueError(f'packed.{key} must be an integer >= 4')
    broad = (p['size']+2*p['halo'])*p['context_scale']
    if broad % p['context_size']:
        raise ValueError('(size+2*halo)*context_scale must be a multiple of context_size')
    factor = broad//p['context_size']
    if p['sampling_stride'] % factor:
        raise ValueError(f'patch.sampling_stride must be a multiple of the context pooling factor ({factor}) '
                         'so pre-pooled context is exact')
    for key, default, minimum in (('val_workers', 4, 0), ('fd_cache_size', 64, 1)):
        if type(tr.get(key, default)) is not int or tr.get(key, default) < minimum:
            raise ValueError(f'train.{key} must be an integer >= {minimum}')
    if type(tr.get('checkpoint_interval', tr['validation_interval'])) is not int or tr.get('checkpoint_interval', 1) < 1:
        raise ValueError('train.checkpoint_interval must be a positive integer')
    if tr['validation_plot_samples'] < 1:
        raise ValueError('train.validation_plot_samples must be >= 1')
    if type(tr.get('persistent_workers', True)) is not bool:
        raise ValueError('train.persistent_workers must be boolean')
    if type(tr['workers']) is not int:
        raise ValueError('train.workers must be an integer')
    return cfg


class PackedArchive:
    """Everything a v4.1 sample needs: small in-memory statics + packed hours."""

    def __init__(self, cfg, archive=None, verify_files=False):
        archive = archive if archive is not None else ArchiveV4(base_config(cfg), verify_files=False)
        self.manifest = load_manifest(cfg, archive, verify_files=verify_files)
        self.root = Path(cfg['packed']['root'])
        self.geo = self.manifest['contract']['geometry']
        self.fingerprint = self.manifest['fingerprint']
        self.by_id = {e['id']: e for e in self.manifest['entries']}
        # Normalization and codecs, identical values to ArchiveV4's. Plain
        # arrays/functions only: this object is pickled into every worker.
        self.cm, self.cs = archive.cm, archive.cs
        self.target_rm, self.target_rs = archive.target_rm, archive.target_rs
        self.scale, self.q_index = archive.scale, archive.q_index
        self.encode = partial(encode_fields_v2, scale=archive.stats['precip_log_scale'],
                              representation=archive.precipitation_representation)
        self.condition_channels = archive.index['condition_channels']
        self.channels = archive.channels
        self.predictors = len(archive.cm)
        self.shape = archive.shape
        statics = self.manifest['statics']
        self.static_root = self.root/statics['folder']
        self.daily_index = {key: i for i, key in enumerate(statics['daily_keys'])}
        self.files = HourFiles(self.root, self.geo, cfg['train'].get('fd_cache_size', 64))
        self._scores = None
        self._static = None

    def __getstate__(self):
        # Workers reopen their own descriptors/maps; never pickle memmap contents.
        return dict(self.__dict__, _scores=None, _static=None)

    @property
    def static(self):
        if self._static is None:
            names = self.manifest['statics']['shapes']
            self._static = {name: np.load(self.static_root/f'{name}.npy', mmap_mode='r') for name in names}
        return self._static

    def entries(self, split):
        result = [e for e in self.manifest['entries'] if e['targets'] and e['split'] == split]
        if not result:
            raise ValueError(f'No packed {split} hours')
        return result

    def score(self, entry):
        if self._scores is None:
            self._scores = np.load(self.root/SCORES, mmap_mode='r')
        return np.asarray(self._scores[entry['row']], dtype='float64')

    def temporal(self, entry):
        """Pooled time features: channels 0-3 are spatially constant, 4-5 were
        pooled per time of day during packing."""
        # Pool a 2x2-block constant patch: a (1,1) output would take PyTorch's
        # mean() shortcut, which rounds differently from v4's block kernel.
        f = self.geo['factor']
        block = time_features(entry['time'], self.static['lon_padded'][:2*f, :2*f])[:4]
        constant = torch.nn.functional.interpolate(torch.from_numpy(block)[None], size=(2, 2), mode='area')[0, :, :1, :1].numpy()
        return constant, self.static['pooled_daily'][self.daily_index[entry['time'][11:16]]]

    def condition(self, entry, raw, baseline, history, y, x):
        """Local native-resolution inputs, exactly ArchiveV2/PrecipArchive.condition."""
        size, halo = self.geo['size'], self.geo['halo']
        dyn = (raw-self.cm)/self.cs
        fixed = crop_v2(self.static['features'], y, x, size, halo)
        temporal = time_features(entry['time'], crop_v2(self.static['lon'], y, x, size, halo))
        base = self.encode(baseline)
        base[0] = (base[0]-280)/20
        base[2] = (base[2]-90000)/15000
        base[3:] /= 10
        current = np.concatenate([dyn, fixed, temporal, base]).astype('float32')
        return np.concatenate([current, *[(h-self.cm)/self.cs for h in history]]).astype('float32')

    def context(self, entry, y, x):
        f, n, p = self.geo['factor'], self.geo['context_size'], self.predictors
        cy, cx = y//f, x//f
        if y % f or x % f:
            raise ValueError('Packed context requires origins on multiples of the pooling factor')
        current = self.files.block(entry, 'ctx', cy, cy+n, cx, cx+n)
        constant, daily = self.temporal(entry)
        temporal = np.concatenate([np.broadcast_to(constant, (4, n, n)), daily[:, cy:cy+n, cx:cx+n]])
        history = [self.files.block(self.by_id[h], 'ctx', cy, cy+n, cx, cx+n)[:p] for h in entry['history']]
        value = np.concatenate([current[:p], self.static['pooled_features'][:, cy:cy+n, cx:cx+n], temporal, current[p:], *history])
        return torch.from_numpy(np.ascontiguousarray(value, dtype='float32'))

    def sample(self, entry, y, x):
        """The v4 DatasetV4 sample for this hour and origin."""
        size, halo, n = self.geo['size'], self.geo['halo'], self.geo['window']
        raw = self.files.crop(entry, 'cond', y-halo, x-halo, n)
        rest = self.files.crop(entry, 'rest', y-halo, x-halo, n)
        history = [self.files.crop(self.by_id[h], 'cond', y-halo, x-halo, n) for h in entry['history']]
        baseline = np.ascontiguousarray(rest[BASELINE])
        truth = np.ascontiguousarray(rest[TRUTH])
        condition = torch.from_numpy(self.condition(entry, raw, baseline, history, y, x))
        context = self.context(entry, y, x)
        coarse = np.concatenate([baseline, raw[self.q_index:self.q_index+1]])
        target = (truth-coarse-self.target_rm)/self.target_rs
        target[1:2] = encode_rain(truth[1:2], self.scale)  # Direct rain, no subtraction.
        area = crop_v2(self.static['area'], y, x, size).copy()
        full = crop_v2(self.static['area'], y, x, size, halo).copy()
        channels = self.condition_channels
        return dict(condition=condition, context=context,
                    original_condition=condition[:channels], original_context=context[:channels],
                    target=torch.from_numpy(target), truth=torch.from_numpy(truth),
                    coarse=torch.from_numpy(coarse), area=torch.from_numpy(area/area.mean()),
                    area_full=torch.from_numpy(full/area.mean()), importance=torch.tensor(1.))


class DatasetV41(Dataset):
    """Index ``epoch*samples + i`` -> the i-th random patch of that epoch.

    Encoding the epoch in the index lets workers persist across epochs while
    keeping v4's per-(seed, epoch, i) sampling. Indices below ``samples`` are
    epoch 0, so Subset/validation use is unchanged.
    """

    def __init__(self, cfg, split, samples, seed=0, packed=None):
        self.packed = packed if packed is not None else PackedArchive(cfg)
        self.entries = self.packed.entries(split)
        self.samples, self.seed = samples, seed
        g = self.packed.geo
        self.yy, self.xx = candidates_v4_1(self.packed.shape, g['size'], g['sampling_stride'], g['factor'])
        self.detail = cfg['patch']['detail_fraction'] if split == 'train' else 0.
        land = np.asarray(self.packed.static['land_fraction'])
        coast = np.abs(np.gradient(land, axis=0))+np.abs(np.gradient(land, axis=1))
        self.coast = box_means_v2(coast, self.yy, self.xx, g['size'])
        self.rain_scale = self.packed.scale

    def __len__(self):
        return self.samples

    def proposal(self, entry):
        n = len(self.yy)
        if not self.detail:
            return np.full(n, 1/n)
        score = self.packed.score(entry)+5*self.coast+.01
        q = np.full(n, (1-self.detail-0.)/n)
        q += self.detail*score/score.sum()
        return q

    def locate(self, index):
        epoch, i = divmod(int(index), self.samples)
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, epoch, i]))
        entry = self.entries[rng.integers(len(self.entries))]
        q = self.proposal(entry)
        location = rng.choice(len(q), p=q)
        return entry, int(self.yy[location]), int(self.xx[location])

    def __getitem__(self, index):
        return self.packed.sample(*self.locate(index))


class EpochSampler(Sampler):
    """Rank-strided indices of one epoch, offset by epoch*samples."""

    def __init__(self, samples, rank=0, world=1):
        if samples % world:
            raise ValueError('samples_per_epoch must divide world size')
        self.samples, self.rank, self.world, self.epoch = samples, rank, world, 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.samples//self.world

    def __iter__(self):
        start = self.epoch*self.samples
        return iter(range(start+self.rank, start+self.samples, self.world))


def check_checkpoint(saved, cfg, archive, packed_fingerprint=None, world=None):
    if saved.get('version') != VERSION or saved.get('targets') != TARGETS:
        raise ValueError('Require a v4.1 six-variable checkpoint')
    if (saved['fingerprint'] != archive.index['fingerprint'] or saved['stats'] != archive.stats
            or saved['hourly_fingerprint'] != archive.hourly_fingerprint
            or saved['humidity_fingerprint'] != archive.humidity_fingerprint):
        raise ValueError('v4.1 archive/hourly/humidity fingerprint mismatch')
    if packed_fingerprint is not None and saved['packed_fingerprint'] != packed_fingerprint:
        raise ValueError('v4.1 packed archive fingerprint mismatch')
    for key in ('data', 'patch', 'conditioning'):
        if saved['config'][key] != cfg[key]:
            raise ValueError(f'Checkpoint {key} mismatch')
    content = lambda value: {k: v for k, v in value.items() if k != 'root'}
    if content(saved['config']['packed']) != content(cfg['packed']):
        raise ValueError('Checkpoint packed layout mismatch')
    if ({k: v for k, v in saved['config']['model'].items() if k != 'activation_checkpointing'} !=
            {k: v for k, v in cfg['model'].items() if k != 'activation_checkpointing'}):
        raise ValueError('Checkpoint model mismatch')
    if world is not None:
        ignore = {'output', 'workers', 'device', 'time_limit_hours', 'prefetch_factor', *TRAIN_ONLY_KEYS}
        if (saved['world_size'] != world or
                {k: v for k, v in cfg['train'].items() if k not in ignore} !=
                {k: v for k, v in saved['config']['train'].items() if k not in ignore}):
            raise ValueError('Exact resume requires matching training settings and world size')
