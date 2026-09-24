"""Hybrid six-field flow: direct sqrt rainfall and residual states including humidity."""
from copy import deepcopy
from collections import OrderedDict
from pathlib import Path
import numpy as np
import torch
import yaml
from .dataset_v2 import PatchDatasetV2, crop_v2
from .dataset_v3_precip import PrecipArchive, PrecipDataset, encode_rain, HOURLY_FILE
from .model_v2 import UNetV2, integrate_v2
from .precip_direct_v2 import regression_bundle as original_bundle, validate_config as validate_direct
from .loss_v2 import quadratic_v2
from .physics_v2 import TARGETS_V2, UNITS_V2
from .prepare_v4 import load_index
from .loading_v4 import ProposalCache

VERSION = 'v4'
TARGETS = list(TARGETS_V2)+['q2m']
UNITS = list(UNITS_V2)+['kg kg-1']


def regression_bundle(path, archive):
    bundle = original_bundle(path, archive)
    bundle['role'] = ('Frozen conditional mean for t2m/ps/u10m/v10m; '
                      'rainfall input only; q2m uses coarse baseline')
    return bundle


def load_config(path):
    return validate_config(yaml.safe_load(Path(path).read_text()))


def validate_config(cfg):
    if cfg.get('version') != VERSION or not (cfg.get('conditioning') or {}).get('checkpoint'):
        raise ValueError('v4 requires an original frozen v2 regression checkpoint')
    # Reuse common optimizer, shape, ensemble and model checks without changing cfg.
    common = deepcopy(cfg)
    common['version'] = 'v2_precip_direct'
    common['data'].update(target_kind='midpoint_rate', representation='log1p')
    validate_direct(common)
    d, p, tr = cfg['data'], cfg['patch'], cfg['train']
    if d['target_kind'] != 'hourly_mean_trapezoid' or d['representation'] != 'sqrt1p':
        raise ValueError('v4 requires processed hourly targets and direct sqrt1p rainfall')
    lags = d['history_hours']
    if (not lags or any(type(l) is not int for l in lags) or lags[-1] != 0
            or lags != sorted(set(lags)) or max(lags) != 0 or -1 not in lags):
        raise ValueError('History must include previous/current hours, be sorted, unique and causal')
    for key in ('rain_scale_mm_h', 'humidity_scale_kg_kg'):
        if not np.isfinite(d[key]) or d[key] <= 0:
            raise ValueError(f'Invalid {key}')
    if any(d.get(k, 0) for k in ('dry_offset', 'dry_threshold_mm_h', 'wet_threshold_z')):
        raise ValueError('v4 uses continuous sqrt1p, without dry margins or thresholding')
    for key in ('hourly_targets', 'humidity_targets', 'humidity_target_variable'):
        if not d.get(key):
            raise ValueError(f'Missing data.{key}')
    if p.get('proposal') != 'coarse' or not p.get('loss_on_halo'):
        raise ValueError('v4 requires input-only coarse proposals and halo supervision')
    model = cfg['model']
    widths = model['channel_mult']
    if (not widths or any(type(w) is not int or w < 1 for w in widths)
            or model['base_channels'] < 1 or model['attention_heads'] < 1
            or model['base_channels']*widths[-1] % model['attention_heads']
            or (p['size']+2*p['halo']) % (2**(len(widths)-1))):
        raise ValueError('Invalid U-Net widths, attention heads, or patch divisibility')
    weights = tr['channel_weights']
    if len(weights) != len(TARGETS) or not np.isfinite(weights).all() or min(weights) <= 0:
        raise ValueError('Provide six positive channel loss weights')
    if type(tr['calibration_batches']) is not int or tr['calibration_batches'] < 1:
        raise ValueError('calibration_batches must be positive')
    for key, default, minimum in [('prefetch_factor', 4, 1), ('array_cache_size', 16, 0)]:
        value = tr.get(key, default)
        if type(value) is not int or value < minimum:
            raise ValueError(f'{key} must be an integer >= {minimum}')
    if type(tr.get('proposal_cache', True)) is not bool:
        raise ValueError('proposal_cache must be boolean')
    return cfg


class ArchiveV4(PrecipArchive):
    def __init__(self, cfg, verify_files=True):
        self.array_cache_size = cfg['train'].get('array_cache_size', 16)
        self._array_cache = OrderedDict()
        super().__init__(cfg, verify_files=verify_files)
        if 'QV2M' not in self.stats['predictors']:
            raise ValueError('v4 requires already prepared QV2M input')
        self.q_index = self.stats['predictors'].index('QV2M')
        self.humidity_root = Path(cfg['data']['humidity_targets'])
        self.humidity_fingerprint = load_index(cfg, self, verify_files=verify_files)['fingerprint']
        self.target_rm = np.concatenate([self.rm, np.zeros((1,1,1), dtype='float32')])
        self.target_rs = np.concatenate([self.rs, np.full((1,1,1), cfg['data']['humidity_scale_kg_kg'], dtype='float32')])

    def __getstate__(self):
        # NumPy can serialize a memmap's contents. Workers must reopen their own
        # bounded mappings instead of copying full-domain arrays through spawn.
        return dict(self.__dict__, _array_cache=OrderedDict())

    def mapped_array(self, path):
        if path in self._array_cache:
            self._array_cache.move_to_end(path)
            return self._array_cache[path]
        value = np.load(path, mmap_mode='r', allow_pickle=False)
        if self.array_cache_size:
            self._array_cache[path] = value
            while len(self._array_cache) > self.array_cache_size:
                # Do not explicitly close mappings: a returned crop may still
                # hold a view. Normal reference counting handles those views.
                self._array_cache.popitem(last=False)
        return value

    def array(self, entry, name):
        return self.mapped_array(self.root/entry['id']/f'{name}_v2.npy')

    def truth_field(self, entry):
        return self.mapped_array(self.hourly_root/entry['id']/HOURLY_FILE)

    def humidity(self, entry):
        return self.mapped_array(self.humidity_root/entry['id']/'q2m_v4.npy')

    def coarse(self, entry):
        return np.concatenate([self.array(entry, 'baseline'), self.array(entry, 'condition')[self.q_index:self.q_index+1]])

    def physical_truth(self, entry):
        truth = np.array(self.array(entry, 'truth'), copy=True)
        truth[1:2] = self.truth_field(entry)
        return np.concatenate([truth, self.humidity(entry)])

    def inputs_with_original(self, entry, y, x, patch):
        condition, context = self.inputs(entry, y, x, patch)
        # History channels are appended after the unchanged v2 channels.
        # Area pooling is channel-independent, so their context is identical
        # too. Reuse both instead of rereading and rebuilding each broad crop.
        channels = self.index['condition_channels']
        return dict(condition=condition, context=context,
                    original_condition=condition[:channels], original_context=context[:channels])


class DatasetV4(PrecipDataset):
    def __init__(self, cfg, split, samples, seed=0, archive=None):
        # Parent sets candidate grids and input-only rain scoring, never uses old
        # truth proposal caches for coarse proposals.
        PatchDatasetV2.__init__(self, cfg['data']['prepared'], split, cfg['patch'], samples, seed)
        self.proposal_kind = 'coarse'
        # Parent caches are truth-based and unused by v4. Do not serialize
        # potentially large memmaps into every spawned coarse-proposal worker.
        self.cached_scores = self.cached_edges = None
        self.cached_rows, self.cached_edge_rows = {}, {}
        self.archive = archive if archive is not None else ArchiveV4(cfg)
        self.entries = self.archive.eligible(split)
        self.rain_scale = self.archive.scale
        self.disk_proposals = None

    def enable_proposal_cache(self, root):
        self.disk_proposals = ProposalCache(root, self.archive.index['fingerprint'],
                                            self.patch, self.archive.shape, len(self.yy))

    def proposal(self, entry):
        if self.disk_proposals is None or not self.detail or self.structure:
            return super().proposal(entry)
        if entry['id'] not in self.proposals:
            cached = self.disk_proposals.load(entry['id'])
            if cached is not None:
                self.proposals[entry['id']] = cached
            else:
                q = super().proposal(entry)
                self.disk_proposals.save(entry['id'], q)
        return self.proposals[entry['id']]

    def __getitem__(self, index):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, int(index)]))
        entry = self.entries[rng.integers(len(self.entries))]
        q = self.proposal(entry)
        i = rng.choice(len(q), p=q)
        y, x = self.yy[i], self.xx[i]
        a, p = self.archive, self.patch
        b = a.inputs_with_original(entry, y, x, p)
        # Crop first: do not copy five full CONUS target arrays per sample.
        truth = crop_v2(a.array(entry, 'truth'), y, x, p['size'], p['halo']).copy()
        truth[1:2] = a.rain(entry, 'truth', y, x, p['size'], p['halo'])
        coarse = crop_v2(a.array(entry, 'baseline'), y, x, p['size'], p['halo']).copy()
        humidity = crop_v2(a.humidity(entry), y, x, p['size'], p['halo'])
        qcoarse = crop_v2(a.array(entry, 'condition')[a.q_index:a.q_index+1], y, x, p['size'], p['halo'])
        truth = np.concatenate([truth, humidity])
        coarse = np.concatenate([coarse, qcoarse])
        target = (truth-coarse-a.target_rm)/a.target_rs
        target[1:2] = encode_rain(truth[1:2], a.scale)  # No rain subtraction.
        area = crop_v2(a.static['area'], y, x, p['size']).copy()
        full = crop_v2(a.static['area'], y, x, p['size'], p['halo']).copy()
        b.update(target=torch.from_numpy(target), truth=torch.from_numpy(truth),
                 coarse=torch.from_numpy(coarse), area=torch.from_numpy(area/area.mean()),
                 area_full=torch.from_numpy(full/area.mean()), importance=torch.tensor(1.))
        return b


class FrozenRegression(torch.nn.Module):
    def __init__(self, bundle, channels, stats, rain_scale=1., humidity_scale=.001):
        super().__init__()
        self.network = UNetV2(channels, **bundle['model_config'])
        self.network.load_state_dict(bundle['weights'])
        self.requires_grad_(False).eval()
        for name, values in [('rm', stats['residual']['mean']+[0.]), ('rs', stats['residual']['std']+[humidity_scale])]:
            self.register_buffer(name, torch.tensor(values, dtype=torch.float32)[None, :, None, None])
        self.register_buffer('flow_scale', torch.ones(1, len(TARGETS), 1, 1))
        self.old_rain_scale = float(stats['precip_log_scale'])
        self.rain_scale = rain_scale

    @torch.no_grad()
    def forward(self, b):
        zeros = torch.zeros_like(b['coarse'][:, :5])
        residual = self.network(zeros, zeros.new_zeros(len(zeros)),
                                b['original_condition'], b['original_context']).float()
        z = (torch.log1p(b['coarse'][:, 1:2]/self.old_rain_scale)
             +residual[:, 1:2]*self.rs[:, 1:2]+self.rm[:, 1:2]).clamp_min(0)
        rain = torch.expm1(z)*self.old_rain_scale
        value = rain/self.rain_scale
        residual[:, 1:2] = value/(torch.sqrt(1+value)+1)
        return torch.cat([residual, torch.zeros_like(residual[:, :1])], dim=1)

    def prepare(self, b):
        b['mean'] = self(b)
        target = (b['target']-b['mean'])/self.flow_scale
        target[:, 1:2] = b['target'][:, 1:2]  # Direct rainfall, no regression subtraction.
        b['target'] = target
        return b

    def physical(self, generated, mean, coarse):
        result = (generated*self.flow_scale+mean)*self.rs+self.rm+coarse
        z = generated[:, 1:2].clamp_min(0)
        result[:, 1:2] = self.rain_scale*z*(z+2)  # No rain add-back.
        result[:, 5:6] = result[:, 5:6].clamp(0, 1)
        if not torch.isfinite(result).all():
            raise FloatingPointError('Nonfinite v4 physical fields')
        return result

    def regression_physical(self, mean, coarse):
        result = mean*self.rs+self.rm+coarse
        z = mean[:, 1:2].clamp_min(0)
        result[:, 1:2] = self.rain_scale*z*(z+2)
        return result


def make_model(channels, cfg):
    model = UNetV2(channels, **cfg['model'], target_channels=len(TARGETS), mean_condition=True)
    model.channel_weights = cfg['train']['channel_weights']
    return model


def objective(model, batch, generator=None):
    x1 = batch['target']
    x0 = torch.randn(x1.shape, device=x1.device, generator=generator)
    t = torch.rand(len(x1), device=x1.device, generator=generator)
    time = t[:, None, None, None]
    velocity = model((1-time)*x0+time*x1, t, batch['condition'], batch['context'], batch['mean']).float()
    weights = (model.module if hasattr(model, 'module') else model).channel_weights
    return quadratic_v2(velocity-(x1-x0), batch['area_full'], batch['importance'], weights)[0]


@torch.no_grad()
def sample(model, noise, condition, context, steps, mean):
    return integrate_v2(model, noise, condition, context, mean, steps)


def check_checkpoint(saved, cfg, archive, world=None):
    if saved.get('version') != VERSION or saved.get('targets') != TARGETS:
        raise ValueError('Require v4 six-variable checkpoint; start a fresh v4 flow')
    if (saved['fingerprint'] != archive.index['fingerprint'] or saved['stats'] != archive.stats
            or saved['hourly_fingerprint'] != archive.hourly_fingerprint
            or saved['humidity_fingerprint'] != archive.humidity_fingerprint):
        raise ValueError('v4 archive/hourly/humidity fingerprint mismatch')
    for key in ('data', 'patch', 'conditioning'):
        if saved['config'][key] != cfg[key]:
            raise ValueError(f'Checkpoint {key} mismatch')
    # Recomputation changes memory/compute use, not architecture or parameters.
    if ({k: v for k, v in saved['config']['model'].items() if k != 'activation_checkpointing'} !=
            {k: v for k, v in cfg['model'].items() if k != 'activation_checkpointing'}):
        raise ValueError('Checkpoint model mismatch')
    if world is not None:
        ignore = {'output', 'workers', 'device', 'time_limit_hours',
                  'prefetch_factor', 'array_cache_size', 'proposal_cache'}
        if (saved['world_size'] != world or
                {k:v for k,v in cfg['train'].items() if k not in ignore} !=
                {k:v for k,v in saved['config']['train'].items() if k not in ignore}):
            raise ValueError('Exact resume requires matching training settings and world size')
