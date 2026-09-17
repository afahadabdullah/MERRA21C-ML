import unittest
from unittest.mock import patch

from merraflow.resume_v2 import resume_world_change_v2


def setup():
    old = {'batch_size': 8, 'accumulate': 2, 'val_batches': 32}
    new = {'batch_size': 2, 'accumulate': 2, 'val_batches': 32}
    checkpoint = {'rng': [None], 'config': {'train': old}}
    config = {'train': new, 'patch': {'samples_per_epoch': 8192}}
    return checkpoint, config


class WorldMigrationTests(unittest.TestCase):
    def test_world_change_requires_explicit_opt_in(self):
        checkpoint, config = setup()
        with patch.dict('os.environ', {'ALLOW_WORLD_SIZE_CHANGE': ''}):
            with self.assertRaisesRegex(ValueError, 'same world size'):
                resume_world_change_v2(checkpoint, config, 4)

    def test_world_change_preserves_batch_and_scheduler(self):
        checkpoint, config = setup()
        with patch.dict('os.environ', {'ALLOW_WORLD_SIZE_CHANGE': '1'}):
            self.assertTrue(resume_world_change_v2(checkpoint, config, 4))
            config['train']['batch_size'] = 8
            with self.assertRaisesRegex(ValueError, 'effective global batch size'):
                resume_world_change_v2(checkpoint, config, 4)
            config['train']['batch_size'] = 2
            config['train']['val_batches'] = 16
            with self.assertRaisesRegex(ValueError, 'validation patch count'):
                resume_world_change_v2(checkpoint, config, 4)
