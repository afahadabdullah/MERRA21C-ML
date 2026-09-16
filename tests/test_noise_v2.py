from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
import xarray as xr

from merraflow.dataset import crop
from merraflow.noise_v2 import padded_noise_v2, saved_noise_padding_v2
from merraflow.inference_v2 import sample_frame_v2
from merraflow.precip_audit_v2 import member_identity
from test_flow_diagnostic_v2 import prepared, ZeroMean
from test_best_model_v2 import DIAGNOSTIC


def test_original_draws_and_legacy_crops_preserved_all_edges_and_zero_halo():
    shape, halo, size = (5, 40, 52), 8, 16
    original = np.random.default_rng(317).standard_normal(shape, dtype=np.float32)
    fixed = padded_noise_v2(317, shape, halo)
    old = padded_noise_v2(317, shape, halo, 'replicate')
    np.testing.assert_array_equal(fixed[:, halo:-halo, halo:-halo], original)
    np.testing.assert_array_equal(padded_noise_v2(317, shape, 0), original)
    for y, x in [(0, 0), (0, 36), (24, 0), (24, 36), (12, 20)]:
        reference = crop(original, y, x, size, halo)
        actual = crop(old, y+halo, x+halo, size, halo)
        np.testing.assert_array_equal(reference, actual)
        assert reference.strides == actual.strides
    assert np.unique(old[1, :halo+1, :halo+1]).size == 1
    assert np.unique(fixed[1, :halo+1, :halo+1]).size == (halo+1)**2
    np.testing.assert_array_equal(fixed, padded_noise_v2(317, shape, halo))


def test_halo_is_shared_between_tiles_and_has_independent_gaussian_statistics():
    halo = 32
    fixed = padded_noise_v2(18, (5, 256, 256), halo)
    left = crop(fixed, halo, halo, 128, halo)
    right = crop(fixed, halo, halo+96, 128, halo)
    np.testing.assert_array_equal(left[:, :, 96:], right[:, :, :96])
    top = fixed[:, :halo, :].ravel()
    assert abs(float(top.mean())) < .03
    assert .97 < float(top.std()) < 1.03
    assert abs(float(np.corrcoef(top[:-1], top[1:])[0, 1])) < .03


def test_boundary_correction_changes_contextual_flow_only_in_affected_cores(prepared):
    class ContextFlow(torch.nn.Module):
        def forward(self, x, time, condition, context, mean):
            return x.mean((-2, -1), keepdim=True).expand_as(x)*.2
    cfg, archive, _ = prepared
    cfg = deepcopy(cfg)
    entry = archive.index['entries'][-1]
    values, means = [], []
    for mode in ('replicate', 'independent_halo'):
        cfg['inference']['noise_padding'] = mode
        pred, mean, _ = sample_frame_v2(ZeroMean(), ContextFlow(), torch.full((1,5,1,1), .2),
                                       archive, entry, cfg, torch.device('cpu'), 317)
        values.append(pred); means.append(mean)
    affected = DIAGNOSTIC.boundary_affected_mask(archive.shape, cfg['patch'])
    assert affected.any() and (~affected).any()
    np.testing.assert_array_equal(means[0], means[1])
    np.testing.assert_array_equal(values[0][:, ~affected], values[1][:, ~affected])
    assert np.max(np.abs(values[0][:, affected]-values[1][:, affected])) > 0


def test_padding_provenance_distinguishes_old_files_from_fixed():
    assert saved_noise_padding_v2({}) == 'replicate'
    assert saved_noise_padding_v2({'noise_padding': 'independent_halo'}) == 'independent_halo'
    entry = {'id': '20260223_0530', 'split': 'test', 'time': '2026-02-23T05:30:00'}
    attrs = dict(version='v2', stage='flow', dataset_fingerprint='x', split='test', ensemble_member=0,
                 seed=317, checkpoint_sha256='c', regression_sha256='r', checkpoint_epoch=87,
                 ode_steps=24, blend='weighted', target_alignment='midpoint', conservation='none')
    ds = xr.Dataset(coords={'time': [np.datetime64(entry['time'])]}, attrs=attrs)
    old = member_identity(ds, 'old.nc', entry, 'x', 0)
    ds.attrs['noise_padding'] = 'independent_halo'
    new = member_identity(ds, 'new.nc', entry, 'x', 0)
    assert old != new and old['noise_padding'] == 'replicate'


def test_comparison_cli_runs_both_modes_without_changing_checkpoint(prepared, tmp_path):
    cfg, archive, checkpoint = prepared
    before = checkpoint.read_bytes()
    config = Path(cfg['data']['prepared']).parent/'config_v2.yaml'
    out = tmp_path/'padding_compare_v2'
    entry = archive.index['entries'][-1]
    subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1]/'scripts/test_best_model_v2.py'),
                    '--config', str(config), '--checkpoint', str(checkpoint), '--output', str(out),
                    '--timestamps', entry['id'], '--members', '2', '--steps', '1', '--compare-noise-padding'], check=True)
    report = json.loads((out/'padding_comparison_v2.json').read_text())
    for mode, run in report.items():
        assert run['noise_padding'] == mode and run['ode_steps'] == 1
        assert run['samples'][0]['padding_check']['max_member_mm_h'] >= 0
        with xr.open_dataset(out/f'{mode}_v2'/'predictions_v2'/f'{entry["id"]}_m000_v2.nc') as ds:
            assert ds.attrs['noise_padding'] == mode
    assert checkpoint.read_bytes() == before
    assert (out/'padding_comparison_v2.md').exists()


def test_batch_wrapper_forwards_controlled_comparison_and_clears_memory_conflict():
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PROJECT_DIR=str(root), SLURM_JOB_ID='42',
               SLURM_MEM_PER_CPU='2', SLURM_MEM_PER_NODE='3', SLURM_MEM_PER_GPU='48G',
               CHECKPOINT='runs/flow_v2/best_v2-Copy1.pt', OUTPUT='runs/padding_compare_v2',
               STEPS='24', MEMBERS='5', COMPARE_NOISE_PADDING='1', NOISE_PADDING='', INCLUDE_DATE='',
               TIMESTAMPS='20260223_0530 20260209_1530')
    result = subprocess.run(['bash', '-c', r'''
source() { :; }
conda() { :; }
srun() {
  test -z "${SLURM_MEM_PER_CPU:-}" && test -z "${SLURM_MEM_PER_NODE:-}"
  test "$SLURM_MEM_PER_GPU" = 48G
  printf '%s\n' "$@"
}
. scripts/slurm_test_best_model_v2.sh
'''], cwd=root, env=env, text=True, capture_output=True, check=True)
    args = result.stdout.splitlines()
    assert args[args.index('--checkpoint')+1].endswith('best_v2-Copy1.pt')
    assert args[args.index('--steps')+1] == '24'
    assert args[args.index('--members')+1] == '5'
    assert '--compare-noise-padding' in args
    assert '--noise-padding' not in args
    assert args[-3:] == ['--timestamps', '20260223_0530', '20260209_1530']
