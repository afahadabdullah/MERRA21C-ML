"""Activation checkpointing is a memory strategy, not part of the architecture.

Turning recomputation off changes activation memory and speed, never parameter
shapes or gradients, so a checkpoint written with it on must still resume.
"""
import unittest
from types import SimpleNamespace

from merraflow.train_v2 import architecture_v2, check_checkpoint_v2


MODEL = {'base_channels': 32, 'channel_mult': [1, 2, 4, 4], 'time_dim': 128,
         'blocks_per_level': 2, 'attention_heads': 4, 'activation_checkpointing': True}


def archive():
    return SimpleNamespace(index={'fingerprint': 'abc'}, stats={'t2m': 1.},
                           precipitation_representation='sqrt')


def config(**model):
    merged = dict(MODEL, **model)
    return {'model': merged, 'patch': {'size': 128, 'halo': 32},
            'representation': {'precip': 'sqrt', 'rain_noise_sigma_pixels': 1.5},
            'loss': {'flow_full_patch': False}}


def checkpoint(cfg):
    return {'version': 'v2', 'stage': 'flow', 'fingerprint': 'abc', 'stats': {'t2m': 1.},
            'config': cfg}


class CheckpointCompatibilityTests(unittest.TestCase):
    def test_architecture_ignores_recomputation_only(self):
        self.assertEqual(architecture_v2(config()['model']),
                         architecture_v2(config(activation_checkpointing=False)['model']))
        self.assertNotEqual(architecture_v2(config()['model']),
                            architecture_v2(config(base_channels=48)['model']))

    def test_resume_allows_disabling_activation_checkpointing(self):
        check_checkpoint_v2(checkpoint(config()), archive(), config(activation_checkpointing=False), 'flow')

    def test_resume_still_rejects_a_real_architecture_change(self):
        with self.assertRaisesRegex(ValueError, 'model/patch mismatch'):
            check_checkpoint_v2(checkpoint(config()), archive(), config(base_channels=48), 'flow')
