"""Configuration for the isolated, single-target precipitation experiment."""
from pathlib import Path
import math
import yaml


def load_config(path):
    with open(path) as stream:
        return validate_config(yaml.safe_load(stream))


def validate_config(cfg):
    if cfg.get('version') != 'v3_precip':
        raise ValueError('Require version: v3_precip')
    d, p, m, tr, ed, inf = [cfg[k] for k in ('data', 'patch', 'model', 'train', 'diffusion', 'inference')]
    if d['target_kind'] not in ('midpoint_rate', 'hourly_mean_trapezoid'):
        raise ValueError('target_kind must be midpoint_rate or hourly_mean_trapezoid (APCP accumulations are not supported)')
    for key in ('dry_threshold_mm_h', 'dry_offset', 'wet_threshold_z'):
        value = d.get(key, 0.)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f'{key} must be a finite nonnegative number')
    if (d.get('dry_offset', 0.) > 0) != (d.get('dry_threshold_mm_h', 0.) > 0):
        raise ValueError('dry_offset and dry_threshold_mm_h must be set together')
    lags = d['history_hours']
    if not lags or lags[-1] != 0 or sorted(set(lags)) != lags or any(type(x) is not int or x > 0 for x in lags):
        raise ValueError('history_hours must be sorted unique nonpositive integers ending in zero')
    if not math.isfinite(d['rain_scale_mm_h']) or d['rain_scale_mm_h'] <= 0:
        raise ValueError('rain_scale_mm_h must be positive')
    output, predictions, archive = [Path(x).resolve() for x in (tr['output'], inf['output'], d['prepared'])]
    for path in (output, predictions):
        if 'v3_precip' not in path.name or path == archive or path in archive.parents or archive in path.parents:
            raise ValueError('Outputs require separate v3_precip directories outside the archive')
    if output == predictions or predictions in output.parents:
        raise ValueError('Training and prediction output directories must differ')
    if d['target_kind'] == 'hourly_mean_trapezoid':
        if not d.get('hourly_targets'):
            raise ValueError('hourly_mean_trapezoid requires data.hourly_targets')
        hourly = Path(d['hourly_targets']).resolve()
        if hourly == archive or archive in hourly.parents or hourly in (output, predictions):
            raise ValueError('hourly_targets must be a separate directory outside the read-only v2 archive')
    for section, keys in [(p, ('size', 'stride', 'context_scale', 'context_size', 'sampling_stride', 'samples_per_epoch')),
                          (tr, ('batch_size', 'accumulate', 'val_batches', 'regression_epochs', 'diffusion_epochs', 'calibration_batches', 'validation_members', 'validation_steps')),
                          (inf, ('members', 'steps'))]:
        for key in keys:
            if type(section[key]) is not int or section[key] < 1:
                raise ValueError(f'{key} must be a positive integer')
    if p['size'] < 8 or p['halo'] < 0 or p['stride'] > p['size'] or p['context_size'] < 8:
        raise ValueError('Invalid patch geometry')
    if not 0 <= p['detail_fraction'] < 1 or p.get('structure_fraction', 0) != 0:
        raise ValueError('Use a rain/uniform proposal mixture with positive uniform support')
    if p.get('proposal', 'coarse') not in ('coarse', 'truth') or not isinstance(p.get('loss_on_halo', False), bool):
        raise ValueError('patch.proposal must be coarse or truth; loss_on_halo must be boolean')
    if (not m['channel_mult'] or min(m['channel_mult']) < 1 or m['base_channels'] < 1
            or m['time_dim'] < 4 or m['time_dim'] % 2 or m['blocks_per_level'] < 1
            or m['attention_heads'] < 1 or m['base_channels']*m['channel_mult'][-1] % m['attention_heads']
            or (p['size']+2*p['halo']) % 2**(len(m['channel_mult'])-1)):
        raise ValueError('Invalid U-Net geometry')
    if 'target_channels' in m or 'mean_condition' in m:
        raise ValueError('v3 fixes target_channels=1 and controls mean conditioning internally')
    if tr['precision'] not in ('fp32', 'bf16') or tr['workers'] < 0 or not 0 <= tr['ema_decay'] < 1:
        raise ValueError('Use fp32 or bf16 and valid workers/EMA')
    if tr.get('lr_schedule', 'constant') not in ('constant', 'cosine'):
        raise ValueError('lr_schedule must be constant or cosine')
    if not 0 < tr.get('min_lr_ratio', 1.) <= 1:
        raise ValueError('min_lr_ratio must be in (0, 1]')
    if type(tr.get('warmup_steps', 0)) is not int or tr.get('warmup_steps', 0) < 0:
        raise ValueError('warmup_steps must be a nonnegative integer')
    limit = tr.get('time_limit_hours')
    if limit is not None and (not isinstance(limit, (int, float)) or not limit > 0):
        raise ValueError('time_limit_hours must be positive when set')
    for key, default in [('validation_plot_interval', 5), ('validation_plot_samples', 2), ('validation_interval', 1)]:
        if type(tr.get(key, default)) is not int or tr.get(key, default) < 1:
            raise ValueError(f'{key} must be a positive integer')
    for key in ('learning_rate', 'grad_clip'):
        if not math.isfinite(tr[key]) or tr[key] <= 0:
            raise ValueError(f'Invalid {key}')
    if not math.isfinite(tr['weight_decay']) or tr['weight_decay'] < 0 or tr['validation_members'] < 2 or inf['members'] < 2:
        raise ValueError('Invalid weight decay or ensemble size (need >=2)')
    if any(not math.isfinite(ed[k]) or ed[k] <= 0 for k in ('sigma_min', 'sigma_max', 'sigma_data', 'rho', 'p_std')):
        raise ValueError('Invalid diffusion scales')
    if ed['sigma_min'] >= ed['sigma_max'] or not math.isfinite(ed['p_mean']) or min(inf['steps'], tr['validation_steps']) < 2:
        raise ValueError('Invalid diffusion noise distribution')
    if inf['blend'] not in ('synchronized', 'owner', 'weighted'):
        raise ValueError('blend must be synchronized, owner or weighted')
    if inf['blend'] == 'synchronized' and not p.get('loss_on_halo', False):
        raise ValueError('Synchronized sampling blends halo outputs: train with patch.loss_on_halo: true')
    if type(inf.get('tile_batch', 8)) is not int or inf.get('tile_batch', 8) < 1:
        raise ValueError('tile_batch must be a positive integer')
    return cfg
