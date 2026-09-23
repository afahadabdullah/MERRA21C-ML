"""One-channel conditional flow matching of full log1p rainfall.

No target subtraction, residual normalization, or output add-back. Optional
frozen v2 regression is an input only. Original midpoint truth stays read-only.
"""
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch
import yaml
from .dataset_v2 import PatchDatasetV2, crop_v2
from .model_v2 import UNetV2, integrate_v2
from .loss_v2 import quadratic_v2
from .train_v2 import architecture_v2, file_hash_v2

VERSION = 'v2_precip_direct'


def load_config(path):
    cfg = yaml.safe_load(Path(path).read_text())
    return validate_config(cfg)


def validate_config(cfg):
    if cfg.get('version') != VERSION:
        raise ValueError('Require version: v2_precip_direct')
    if cfg.get('conditioning') is not None and not cfg['conditioning'].get('checkpoint'):
        raise ValueError('Specify an original v2 regression checkpoint or conditioning: null')
    if cfg['data'].get('target_kind') != 'midpoint_rate' or cfg['data'].get('representation') != 'log1p':
        raise ValueError('Direct v2 uses original midpoint rainfall with log1p encoding')
    for key in ('steps', 'members'):
        if type(cfg['inference'][key]) is not int or cfg['inference'][key] < 1:
            raise ValueError(f'Invalid inference.{key}')
    p, tr = cfg['patch'], cfg['train']
    for key in ('size', 'stride', 'samples_per_epoch', 'context_scale', 'context_size', 'sampling_stride'):
        if type(p[key]) is not int or p[key] < 1:
            raise ValueError(f'Invalid patch.{key}')
    if type(p['halo']) is not int or p['halo'] < 0 or p['stride'] > p['size']:
        raise ValueError('Invalid halo/stride')
    if not 0 <= p['detail_fraction'] < 1 or p.get('structure_fraction', 0):
        raise ValueError('Use the original v2 proposal with a positive uniform component')
    for key in ('epochs', 'batch_size', 'accumulate', 'validation_interval', 'validation_patches',
                'validation_members', 'validation_steps', 'validation_plot_samples'):
        if type(tr[key]) is not int or tr[key] < 1:
            raise ValueError(f'Invalid train.{key}')
    if tr['validation_members'] < 2 or tr['precision'] not in ('fp32', 'bf16'):
        raise ValueError('Use >=2 members and fp32 or bf16')
    if not 0 <= tr['ema_decay'] < 1 or not 0 <= tr['min_lr_ratio'] <= 1:
        raise ValueError('Invalid EMA/LR settings')
    for key in ('learning_rate', 'grad_clip'):
        if not np.isfinite(tr[key]) or tr[key] <= 0:
            raise ValueError(f'Invalid train.{key}')
    if tr['workers'] < 0 or tr['warmup_steps'] < 0 or tr['weight_decay'] < 0:
        raise ValueError('Invalid optimizer/worker settings')
    if tr.get('time_limit_hours') is not None and tr['time_limit_hours'] <= 0:
        raise ValueError('time_limit_hours must be positive or null')
    if any(k in cfg['model'] for k in ('mean_condition', 'target_channels')):
        raise ValueError('Direct precipitation fixes one target and no mean conditioning')
    return cfg


def encode_rain(rain, scale):
    if not np.isfinite(rain).all() or np.any(rain < 0) or not np.isfinite(scale) or scale <= 0:
        raise ValueError('Invalid rainfall or scale')
    return np.log1p(rain/scale).astype('float32')


def decode_rain(value, scale):
    if not np.isfinite(value).all() or np.max(value) > 30:
        raise FloatingPointError('Nonfinite or overflowing direct rainfall')
    return (np.expm1(np.maximum(value, 0))*scale).astype('float32')


class DirectPrecipDataset(PatchDatasetV2):
    def __init__(self, cfg, split, samples, seed):
        super().__init__(cfg['data']['prepared'], split, cfg['patch'], samples, seed)
        if self.archive.stats.get('target_channels') != ['t2m', 'precip', 'ps', 'u10m', 'v10m']:
            raise ValueError('Require original five-channel v2 source archive')
        data = self.archive.index['data_config']
        if data.get('precip_source') != 'hwt_30mn_slv_LCC.PRECTOT' or data.get('state_alignment') != 'midpoint_snapshot':
            raise ValueError('Require HWT midpoint PRECTOT truth')
        self.rain_scale = self.archive.stats['precip_log_scale']

    def __getitem__(self, index):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, int(index)]))
        entry = self.entries[rng.integers(len(self.entries))]
        q = self.proposal(entry)
        i = rng.choice(len(q), p=q)
        y, x = self.yy[i], self.xx[i]
        a, p = self.archive, self.patch
        condition, context = a.inputs(entry, y, x, p)
        # Slice the precipitation channel before cropping. Never read residual_v2.npy.
        truth = crop_v2(a.array(entry, 'truth')[1:2], y, x, p['size'], p['halo']).copy()
        coarse = crop_v2(a.array(entry, 'baseline')[1:2], y, x, p['size'], p['halo']).copy()
        area = crop_v2(a.static['area'], y, x, p['size']).copy()
        full = crop_v2(a.static['area'], y, x, p['size'], p['halo']).copy()
        return dict(condition=condition, context=context,
                    target=torch.from_numpy(encode_rain(truth, self.rain_scale)),
                    truth=torch.from_numpy(truth), coarse=torch.from_numpy(coarse),
                    area=torch.from_numpy(area/area.mean()), area_full=torch.from_numpy(full/area.mean()),
                    importance=torch.tensor(1/(len(q)*q[i]), dtype=torch.float32))


def make_model(channels, cfg):
    return UNetV2(channels, **cfg['model'], target_channels=1, mean_condition=bool(cfg.get('conditioning')))


def objective(model, batch, generator=None):
    x1 = batch['target']  # FULL encoded rainfall, not a prediction error.
    x0 = torch.randn(x1.shape, device=x1.device, generator=generator)
    t = torch.rand(x1.shape[0], device=x1.device, generator=generator)
    time = t[:, None, None, None]
    velocity = model((1-time)*x0+time*x1, t, batch['condition'], batch['context'], batch.get('mean')).float()
    # Full halo supervision supports synchronized domain inference.
    return quadratic_v2(velocity-(x1-x0), batch['area_full'], batch['importance'], [1.])[0]


@torch.no_grad()
def sample(model, noise, condition, context, steps, mean=None):
    return integrate_v2(model, noise, condition, context, mean, steps)


def original_checkpoint(path, archive):
    source = torch.load(path, map_location='cpu', weights_only=True)
    if source.get('version') != 'v2' or source.get('stage') not in ('flow', 'regression'):
        raise ValueError('Require an original v2 regression or flow checkpoint')
    if source.get('initialization') or source['config']['train'].get('rain_rollout'):
        raise ValueError('Use the original v2 checkpoint, not a fine-tuned checkpoint')
    if source['fingerprint'] != archive.index['fingerprint'] or source['stats'] != archive.stats:
        raise ValueError('Source checkpoint archive/statistics mismatch')
    if source['config'].get('representation', {}).get('precip', 'log1p') != 'log1p':
        raise ValueError('Require the original log1p v2 checkpoint')
    return source


class FrozenRegression(torch.nn.Module):
    """Full encoded rain predicted by old v2; used ONLY as an extra condition."""
    def __init__(self, bundle, channels, stats):
        super().__init__()
        self.network = UNetV2(channels, **bundle['model_config'])
        self.network.load_state_dict(bundle['weights'])
        self.requires_grad_(False).eval()
        self.offset = float(stats['residual']['mean'][1])
        self.scale = float(stats['residual']['std'][1])
        self.rain_scale = float(stats['precip_log_scale'])

    @torch.no_grad()
    def forward(self, batch):
        coarse = batch['coarse']
        zeros = coarse.new_zeros((len(coarse), 5, *coarse.shape[-2:]))
        residual = self.network(zeros, zeros.new_zeros(len(coarse)), batch['condition'], batch['context']).float()
        return (torch.log1p(coarse/self.rain_scale)+residual[:, 1:2]*self.scale+self.offset).clamp_min(0)


def regression_bundle(path, archive):
    source = original_checkpoint(path, archive)
    return dict(weights=source['ema'] if source['stage'] == 'regression' else source['regression_ema'],
                model_config=source['config']['model'], sha256=file_hash_v2(path),
                checkpoint=str(Path(path).resolve()), role='Frozen condition only, never subtracted from target')


def initialize_from_v2(model, checkpoint, cfg, archive):
    """Transfer only internal EMA layers. Target-dependent layers stay fresh."""
    source = original_checkpoint(checkpoint, archive)
    if architecture_v2(source['config']['model']) != architecture_v2(cfg['model']):
        raise ValueError('Source and direct model architectures must match')
    source_weights, weights = source['ema'], model.state_dict()
    reset = ('input.', 'conditions.', 'output.')
    copied = []
    for name, value in weights.items():
        if name.startswith(reset):
            continue
        if name not in source_weights or source_weights[name].shape != value.shape:
            raise ValueError(f'Incompatible source layer {name}')
        weights[name] = source_weights[name]
        copied.append(name)
    model.load_state_dict(weights)
    return dict(checkpoint=str(Path(checkpoint).resolve()), sha256=file_hash_v2(checkpoint),
                source_epoch=source['epoch']+1, source_stage=source['stage'], copied_layers=copied,
                reset_layers=[k for k in weights if k.startswith(reset)],
                mode='Internal EMA weights only; new input/conditioning/output and optimizer')


def check_checkpoint(saved, cfg, archive, world=None):
    if saved.get('version') != VERSION or saved.get('targets') != ['precip']:
        raise ValueError('Require a direct precipitation checkpoint; residual checkpoints cannot resume')
    if bool(saved.get('regression_condition')) != bool(saved['config'].get('conditioning')):
        raise ValueError('Checkpoint frozen regression is missing or inconsistent')
    if saved['fingerprint'] != archive.index['fingerprint'] or saved['stats'] != archive.stats:
        raise ValueError('Archive/statistics mismatch')
    if bool(saved['config'].get('conditioning')) != bool(cfg.get('conditioning')):
        raise ValueError('Checkpoint conditioning mode mismatch')
    for key in ('data', 'patch', 'model'):
        old, new = deepcopy(saved['config'][key]), deepcopy(cfg[key])
        if key == 'data':
            old.pop('prepared', None)
            new.pop('prepared', None)
        if old != new:
            raise ValueError(f'Checkpoint {key} mismatch')
    if world is not None:
        ignored = {'output', 'workers', 'device', 'time_limit_hours'}
        if saved['world_size'] != world or (
                {k: v for k, v in saved['config']['train'].items() if k not in ignored} !=
                {k: v for k, v in cfg['train'].items() if k not in ignored}):
            raise ValueError('Exact resume requires matching training settings and world size')
