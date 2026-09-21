"""Strict v2 configuration and namespace isolation."""
from pathlib import Path
import math
import yaml
from .noise_v2 import noise_padding_v2
from .physics_v2 import precipitation_representation_v2
from .rain_prior_v2 import rain_noise_sigma_v2


def v2_path(path):
    p = Path(path)
    if 'v2' not in p.name.lower() or 'v2' not in p.resolve().name.lower():
        raise ValueError(f'V2 output directory must have v2 in its basename: {p}')
    return p


def load_config_v2(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    validate_config_v2(cfg)
    return cfg


def validate_config_v2(cfg):
    noise_padding_v2(cfg)
    if cfg.get('version') != 'v2':
        raise ValueError('Require version: v2')
    if precipitation_representation_v2(cfg) not in ('log1p', 'sqrt1p'):
        raise ValueError('Unsupported precipitation representation')
    sigma = rain_noise_sigma_v2(cfg)
    if not math.isfinite(sigma) or not 0 <= sigma <= 16:
        raise ValueError('rain_noise_sigma_pixels must be in [0, 16]')
    if sigma and noise_padding_v2(cfg) != 'independent_halo':
        raise ValueError('Correlated rainfall prior requires independent halo support')
    if cfg['inference'].get('sampler', 'independent') not in ('independent', 'synchronized'):
        raise ValueError('Unsupported v2 sampler')
    if cfg['inference'].get('sampler') == 'synchronized' and not cfg['loss'].get('flow_full_patch', False):
        raise ValueError('Synchronized sampling requires full-patch flow supervision')
    if not isinstance(cfg['loss'].get('flow_full_patch', False), bool):
        raise ValueError('flow_full_patch must be boolean')
    for key in ('flow_multiscale', 'regression_rain_physical'):
        if cfg['loss'].get(key, 0) < 0:
            raise ValueError(f'{key} must be nonnegative')
    if cfg['loss'].get('regression_rain_physical', 0) and precipitation_representation_v2(cfg) != 'sqrt1p':
        raise ValueError('Physical rainfall loss requires sqrt1p representation')
    if cfg['inference'].get('tile_cache_mb', 512) < 0:
        raise ValueError('tile_cache_mb must be nonnegative')
    if cfg['inference'].get('sampler') == 'synchronized' and cfg['inference'].get('blend', 'weighted') != 'weighted':
        raise ValueError('Synchronized sampler uses weighted velocities')
    generated = cfg['train'].get('generated_validation', {})
    if generated:
        if precipitation_representation_v2(cfg) != 'sqrt1p':
            raise ValueError('Generated rainfall validation requires sqrt1p')
        if any(not isinstance(generated.get(k), int) or generated[k] < 1 for k in ('batches', 'interval', 'steps')) or generated.get('members', 0) < 2:
            raise ValueError('Invalid generated validation counts')
    rollout = cfg['train'].get('rain_rollout', {})
    if rollout:
        if precipitation_representation_v2(cfg) != 'sqrt1p' or not generated:
            raise ValueError('Rain rollout requires sqrt1p and generated validation')
        for key in ('steps', 'members', 'patches', 'interval', 'ramp_epochs'):
            if type(rollout.get(key)) is not int or rollout[key] < (2 if key == 'members' else 1):
                raise ValueError(f'Invalid rain_rollout.{key}')
        if type(rollout.get('warmup_epochs')) is not int or rollout['warmup_epochs'] < 0:
            raise ValueError('Invalid rain_rollout.warmup_epochs')
        for key in ('weight', 'crps_weight', 'variogram_weight', 'coverage_weight', 'mean_mse_weight',
                    'rate_scale_mm_h', 'coverage_temperature_mm_h'):
            value = rollout.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'Invalid rain_rollout.{key}')
        for key in ('lags', 'pool_scales'):
            values = rollout.get(key)
            if not isinstance(values, list) or not values or any(type(v) is not int or v < 1 for v in values):
                raise ValueError(f'Invalid rain_rollout.{key}')
            if min(values) > cfg['patch']['size'] or (key == 'lags' and min(values) >= cfg['patch']['size']):
                raise ValueError(f'No usable rain_rollout.{key} for this patch')
        thresholds = rollout.get('thresholds_mm_h')
        if not isinstance(thresholds, list) or not thresholds or any(
                not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in thresholds):
            raise ValueError('Invalid rain_rollout.thresholds_mm_h')
    d, p, m, tr = (cfg[k] for k in ('data', 'patch', 'model', 'train'))
    if 'reference_world_size' in tr and (type(tr['reference_world_size']) is not int or tr['reference_world_size'] < 1):
        raise ValueError('reference_world_size must be a positive integer')
    structure_fraction = p.get('structure_fraction', 0.)
    if not isinstance(structure_fraction, (int, float)) or not math.isfinite(structure_fraction) or not 0 <= structure_fraction < 1-p['detail_fraction']:
        raise ValueError('structure_fraction + detail_fraction must be < 1, leaving uniform coverage')
    if 'context_tokens' in m and (type(m['context_tokens']) is not int or not 1 <= m['context_tokens'] <= math.ceil(p['context_size']/4)):
        raise ValueError('context_tokens must fit the encoded context grid')
    if d['conserve_training_precip'] is not False or cfg['inference']['conserve_precip'] is not False:
        raise ValueError('V2 never projects precipitation')
    if d['precip_source'] != 'hwt_30mn_slv_LCC.PRECTOT' or d['state_alignment'] != 'midpoint_snapshot':
        raise ValueError('V2 requires matched HWT PRECTOT midpoint snapshots')
    if any(k in d for k in ('accumulation_timestamp', 'accumulation_hours')):
        raise ValueError('Legacy APCP configuration is incompatible with v2')
    paths = [v2_path(d['prepared']), v2_path(tr['output']), v2_path(cfg['inference']['output'])]
    resolved = [x.resolve() for x in paths]
    if len(set(resolved)) != 3 or any(resolved[0] in x.parents or x in resolved[0].parents for x in resolved[1:]):
        raise ValueError('Prepared data and run/output directories must be separate')
    factor = 2**(len(m['channel_mult'])-1)
    if (not m['channel_mult'] or min(m['channel_mult']) < 1 or m['base_channels'] < 1
            or m['time_dim'] < 4 or m['time_dim'] % 2
            or m['blocks_per_level'] < 1 or m['attention_heads'] < 1
            or m['base_channels']*m['channel_mult'][-1] % m['attention_heads']):
        raise ValueError('Invalid v2 architecture widths/attention')
    if (p['size'] < 8 or p['halo'] < 0 or (p['size']+2*p['halo']) % factor
            or not 0 < p['stride'] <= p['size'] or p['context_scale'] < 1
            or p['context_size'] < 8 or p['sampling_stride'] < 1
            or not isinstance(p['context_scale'], int)
            or not 0 <= p['detail_fraction'] < 1):
        raise ValueError('Invalid patch/context/sampling settings')
    if min(p['samples_per_epoch'], tr['batch_size'], tr['accumulate'], tr['val_batches'],
           tr['regression_epochs'], tr['flow_epochs'], tr['checkpoint_interval'], tr['calibration_batches']) < 1:
        raise ValueError('Training counts must be positive')
    if (tr['workers'] < 0 or tr['precision'] not in ('fp32', 'bf16', 'fp16')
            or not 0 <= tr['ema_decay'] < 1 or tr['learning_rate'] <= 0
            or tr['weight_decay'] < 0 or tr['warmup_steps'] < 0
            or not 0 <= tr['min_lr_ratio'] <= 1 or tr['grad_clip'] <= 0):
        raise ValueError('Invalid training optimizer/precision settings')
    if len(tr['channel_weights']) != 5 or min(tr['channel_weights']) <= 0:
        raise ValueError('Require five positive v2 channel weights')
    if d['stats_stride'] < 1 or d['precip_log_scale'] <= 0:
        raise ValueError('Invalid statistics/precipitation transform')
    if not d.get('static') or not d['static'].get('path'):
        raise ValueError('Provide data.static.path with genuine HR land and lake fractions')
    if len(set(d['predictors'])) != len(d['predictors']):
        raise ValueError('Duplicate dynamic predictors')
    if cfg['inference']['members'] < 1 or cfg['inference']['steps'] < 1:
        raise ValueError('Positive member and integration counts required')
    if cfg['inference'].get('blend', 'weighted') not in ('weighted', 'owner'):
        raise ValueError('Invalid v2 overlap blend')
    for name in ('regression_gradient', 'regression_multiscale', 'flow_gradient'):
        if cfg['loss'][name] < 0:
            raise ValueError('Loss coefficients must be nonnegative')
    return cfg
