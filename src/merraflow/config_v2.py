"""Strict v2 configuration and namespace isolation."""
from pathlib import Path
import yaml


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
    if cfg.get('version') != 'v2':
        raise ValueError('Require version: v2')
    d, p, m, tr = (cfg[k] for k in ('data', 'patch', 'model', 'train'))
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
