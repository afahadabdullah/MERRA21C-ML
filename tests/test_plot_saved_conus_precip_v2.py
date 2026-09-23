"""Saved whole-domain plots must preserve model identity and timestamp."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import xarray as xr

from merraflow.plot_saved_conus_precip_v2 import (archive_case, available_member_groups,
                                                   member_paths, plot_saved)


class SavedConusTests(unittest.TestCase):
    def make_archive(self, root):
        archive = root/'archive'
        case = archive/'20260223_0530'
        case.mkdir(parents=True)
        truth = np.zeros((40, 48), dtype='float32')
        truth[8:25, 10:30] = 8
        np.save(case/'truth_v2.npy', np.stack([truth, truth, truth, truth, truth]))
        np.save(case/'baseline_v2.npy', np.stack([truth, np.roll(truth, 5, axis=1),
                                                  truth, truth, truth]))
        np.savez(archive/'static_v2.npz', area=np.ones_like(truth))
        (archive/'index_v2.json').write_text(json.dumps(dict(fingerprint='fixture', entries=[
            dict(id='20260223_0530', time='2026-02-23T05:30:00', split='test')])))
        return archive, truth

    def save_member(self, directory, truth, suffix, version, member=0):
        path = directory/f'20260223_0530_m{member:03d}{suffix}.nc'
        attributes = dict(version=version, split='test', ensemble_member=member,
                          checkpoint_sha256='same-checkpoint', checkpoint_epoch=40,
                          target='full precipitation', dataset_fingerprint='fixture')
        xr.Dataset({'precip': (('time', 'Ydim', 'Xdim'), truth[None])},
                   coords={'time': [np.datetime64('2026-02-23T05:30:00', 'ns')]},
                   attrs=attributes).to_netcdf(path, engine='h5netcdf')
        return path

    def test_detect_and_plot_saved_direct_members(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive, truth = self.make_archive(root)
            direct = root/'runs'/'direct'/'predictions'
            residual = root/'runs'/'original'/'predictions'
            direct.mkdir(parents=True)
            residual.mkdir(parents=True)
            self.save_member(direct, truth, '_direct_v2', 'v2_precip_direct')
            self.save_member(direct, np.roll(truth, 2, axis=1), '_direct_v2',
                             'v2_precip_direct', member=1)
            self.save_member(residual, truth, '_v2', 'v2')
            self.assertEqual(len(available_member_groups(root/'runs', '20260223_0530')), 2)
            kind, paths = member_paths(root/'runs', '20260223_0530')
            self.assertEqual(kind, 'direct')
            self.assertEqual(len(paths), 2)
            output = root/'figure'
            image = plot_saved(archive, root/'runs', output)
            report = json.loads((output/'report.json').read_text())
            self.assertTrue(image.is_file())
            self.assertEqual(report['model_kind'], 'direct')
            self.assertEqual(report['members'], 2)
            self.assertGreater(report['metrics']['truth_mean_mm_h'], 0)
            original_output = root/'original_figure'
            original = plot_saved(archive, residual, original_output, kind='residual-v2')
            self.assertTrue(original.is_file())
            self.assertEqual(json.loads((original_output/'report.json').read_text())['model_kind'],
                             'residual-v2')
            with self.assertRaisesRegex(FileExistsError, 'fresh output'):
                plot_saved(archive, root/'runs', output)
            self.assertEqual(archive_case(archive, '20260223_0530', 'test')[0]['split'], 'test')

    def test_multiple_direct_runs_require_exact_directory(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, truth = self.make_archive(root)
            for name in ('first', 'second'):
                folder = root/name
                folder.mkdir()
                self.save_member(folder, truth, '_direct_v2', 'v2_precip_direct')
            with self.assertRaisesRegex(ValueError, 'exact directory'):
                member_paths(root, '20260223_0530')


if __name__ == '__main__':
    unittest.main()
