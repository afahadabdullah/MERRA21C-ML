"""CPU-only checks for the saved wet-case spatial audit."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from merraflow.analyze_wet_precip_direct_v2 import analyze, spatial_scores, uniform_epoch


class SpatialAuditTests(unittest.TestCase):
    def test_neighborhood_skill_distinguishes_shift_from_texture(self):
        truth = np.zeros((64, 64))
        truth[20:30, 18:28] = 8
        shifted = np.zeros_like(truth)
        shifted[20:30, 26:36] = 8
        area = np.ones_like(truth)
        scores = spatial_scores(np.stack([truth, truth]), truth, shifted, shifted,
                                area, thresholds=(5.,), scales=(1, 17))['5.0']
        self.assertAlmostEqual(scores['1']['member_mean'], 1.)
        self.assertAlmostEqual(scores['1']['ensemble_mean'], 1.)
        self.assertLess(scores['1']['coarse'], scores['17']['coarse'])
        self.assertEqual(scores['1']['coarse'], scores['1']['regression'])

    def test_existing_arrays_and_matching_uniform_epoch(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            results = root/'results'
            results.mkdir()
            truth = np.zeros((64, 64))
            truth[10:25, 12:30] = 12
            coarse = np.roll(truth, 8, axis=1)
            np.savez_compressed(results/'case_01_20251203_0030.npz',
                                ensemble=np.stack([truth, coarse]), truth=truth,
                                coarse=coarse, regression=coarse, area=np.ones_like(truth))
            (results/'report.json').write_text(json.dumps(dict(epoch=40, cases=[dict(
                id='20251203_0030', time='2025-12-03T00:30:00')])))
            history = root/'history.json'
            history.write_text(json.dumps([dict(epoch=35, crps=.3), dict(
                epoch=40, crps=.2, coarse_crps=.4, regression_crps=.5,
                wet_fraction=.2, truth_wet_fraction=.1)]))
            output = analyze(results, history)
            audit = json.loads(output.read_text())
            self.assertEqual(audit['epoch'], 40)
            self.assertEqual(audit['uniform_validation']['crps_improvement_vs_coarse_percent'], 50.)
            scores = audit['mean_wet_case_fss']['5.0']['1']
            self.assertAlmostEqual(scores['member_mean'], (1+scores['coarse'])/2)
            self.assertTrue((results/'spatial_skill.png').is_file())
            with self.assertRaisesRegex(ValueError, 'epoch 41'):
                uniform_epoch(history, 41)


if __name__ == '__main__':
    unittest.main()
