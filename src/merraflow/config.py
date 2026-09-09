from pathlib import Path
import json
import os
import tempfile
import yaml


def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    p, m, tr = cfg['patch'], cfg['model'], cfg['train']
    if not m['channel_mult'] or min(m['channel_mult']) < 1 or m['base_channels'] < 1:
        raise ValueError('Model channel widths must be positive')
    factor = 2 ** (len(m['channel_mult']) - 1)
    if p['size'] <= 0 or p['halo'] < 0 or (p['size'] + 2*p['halo']) % factor:
        raise ValueError('Patch plus halo must be positive and divisible by U-Net downsampling factor')
    if not 0 < p['stride'] <= p['size']:
        raise ValueError('Require 0 < stride <= patch size for complete coverage')
    if min(tr['batch_size'], tr['accumulate'], tr['epochs'], p['samples_per_epoch'], tr['val_batches']) <= 0:
        raise ValueError('Batch, accumulation, epochs, samples and validation count must be positive')
    if tr['workers'] < 0 or not 0 <= tr['ema_decay'] < 1 or tr['learning_rate'] <= 0 or tr['grad_clip'] <= 0:
        raise ValueError('Invalid worker count, EMA decay, learning rate or gradient clip')
    if tr['warmup_steps'] < 0 or not 0 <= tr['min_lr_ratio'] <= 1 or tr['weight_decay'] < 0:
        raise ValueError('Invalid learning rate schedule or weight decay')
    inf = cfg['inference']
    if inf['members'] < 1 or inf['steps'] < 1 or inf['dry_threshold'] < 0:
        raise ValueError('Invalid ensemble size, ODE step count or dry threshold')
    if tr['precision'] not in ('fp32', 'bf16', 'fp16'):
        raise ValueError('Unknown precision')
    if cfg['data']['state_alignment'] != 'midpoint_snapshot':
        raise ValueError('Only explicitly declared midpoint_snapshot state alignment is implemented')
    if cfg['data']['accumulation_timestamp'] != 'end' or cfg['data']['accumulation_hours'] != 1:
        raise ValueError('This pipeline pairs hourly end-labeled APCP with midpoint-labeled GEOS-FP')
    return cfg


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(obj, f, indent=2, allow_nan=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
