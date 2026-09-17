"""Validation for controlled changes to v2 training process count."""
import math
import os


def resume_world_change_v2(ckpt, cfg, world):
    """Validate an opt-in change of DDP width at an epoch boundary."""
    old_world = len(ckpt['rng'])
    if old_world == world:
        return False
    if os.getenv('ALLOW_WORLD_SIZE_CHANGE') != '1':
        raise ValueError('Exact resume requires the same world size; set ALLOW_WORLD_SIZE_CHANGE=1 for an epoch-boundary migration')
    old = ckpt['config']['train']
    new = cfg['train']
    if old['batch_size']*old['accumulate']*old_world != new['batch_size']*new['accumulate']*world:
        raise ValueError('World-size migration must preserve the effective global batch size')
    samples = cfg['patch']['samples_per_epoch']
    if samples % old_world or samples % world:
        raise ValueError('World-size migration requires samples_per_epoch to divide both world sizes')
    def steps_per_epoch(train, ranks):
        per_rank = math.ceil(samples/ranks)
        return math.ceil(math.ceil(per_rank/train['batch_size'])/train['accumulate'])
    if steps_per_epoch(old, old_world) != steps_per_epoch(new, world):
        raise ValueError('World-size migration must preserve optimizer steps per epoch and the LR schedule')
    if old['val_batches']*old['batch_size']*old_world != new['val_batches']*new['batch_size']*world:
        raise ValueError('World-size migration must preserve the validation patch count')
    return True
