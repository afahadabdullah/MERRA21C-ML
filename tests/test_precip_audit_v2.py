"""Numerical and saved-NetCDF integration tests; no GPU or Torch required."""
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest
import xarray as xr

from merraflow.precip_audit_v2 import (
    area_coarsen, rainfall_profile, lag_scan, audit_case, load_members, run_audit,
)


def test_coarsening_preserves_amount_with_partial_blocks_and_unequal_area():
    area = np.arange(1, 36, dtype=float).reshape(5, 7)
    field = np.arange(35, dtype=float).reshape(5, 7)
    coarse, weights = area_coarsen(field, area, 3)
    assert coarse.shape == (2, 3)
    assert weights.sum() == area.sum()
    np.testing.assert_allclose((coarse*weights).sum(), (field*area).sum())
    members, member_weights = area_coarsen(np.stack([field, 2*field]), area, 3)
    np.testing.assert_allclose(members[1], 2*coarse)
    np.testing.assert_array_equal(member_weights, weights)


def test_profile_uses_area_for_quantiles_and_volume_rate():
    field = np.array([[0., 10.]])
    area = np.array([[99., 1.]])
    profile = rainfall_profile(field, area)
    assert profile['area_quantiles_mm_h']['p95'] == 0
    assert profile['area_quantiles_mm_h']['p999'] == 10
    assert profile['wet_area_fraction']['1.0'] == .01
    assert profile['mean_mm_h'] == .1
    assert profile['volume_rate_m3_s'] == pytest.approx(10/3.6e6)


def test_lag_scan_detects_known_displacement_without_wrapping():
    truth = np.zeros((25, 25))
    truth[8:12, 10:14] = 8
    prediction = np.zeros_like(truth)
    prediction[10:14, 9:13] = 8
    result = lag_scan(prediction, truth, np.ones_like(truth))
    assert result['unshifted_rmse'] > 0
    assert result['best']['rmse'] == 0
    assert result['best']['prediction_offset_y'] == 2
    assert result['best']['prediction_offset_x'] == -1
    dry = lag_scan(truth*0, truth*0, np.ones_like(truth))
    assert dry['best']['prediction_offset_y'] == dry['best']['prediction_offset_x'] == 0


def test_dry_case_is_json_safe_and_shrinkage_zero_matches_regression():
    truth = np.zeros((16, 24), dtype='float32')
    regression = np.ones_like(truth)
    ensemble = np.stack([regression, 3*regression])
    report = audit_case(ensemble, regression, truth, truth, np.ones_like(truth), {'size': 8, 'stride': 6}, 1.)
    assert report['profiles']['flow_mean']['mean_ratio_to_HWT'] is None
    assert report['postdecode_log_residual_shrinkage']['0.0']['scores']['rmse'] == pytest.approx(1)
    assert report['postdecode_log_residual_shrinkage']['1.0']['scores']['rmse'] == pytest.approx(2)
    assert report['jensen_gap']['mean_mm_h'] > 0
    json.dumps(report, allow_nan=False)
    with pytest.raises(ValueError, match='nonnegative'):
        audit_case(-ensemble, regression, truth, truth, np.ones_like(truth), {'size': 8, 'stride': 6}, 1.)


@pytest.fixture
def saved(tmp_path):
    archive = tmp_path/'archive_v2'
    predictions = tmp_path/'diagnostic_v2'/'predictions_v2'
    archive.mkdir()
    predictions.mkdir(parents=True)
    h, w = 24, 32
    yy, xx = np.mgrid[:h, :w]
    static = {'area': np.full((h, w), 4e6), 'lat': 30+yy*.02, 'lon': -100+xx*.02}
    np.savez(archive/'static_v2.npz', **static)
    entry = {'id': '20260223_0530', 'time': '2026-02-23T05:30:00', 'split': 'test'}
    (archive/'index_v2.json').write_text(json.dumps({'format': 'v2', 'fingerprint': 'fixture', 'entries': [entry]}))
    (archive/'stats_v2.json').write_text(json.dumps({'precip_log_scale': 1.}))
    (archive/entry['id']).mkdir()
    truth = np.exp(-((yy-12)**2+(xx-18)**2)/20).astype('float32')*12
    values = np.zeros((5, h, w), dtype='float32')
    values[1] = truth
    np.save(archive/entry['id']/'truth_v2.npy', values)
    values[1] = truth*.8
    np.save(archive/entry['id']/'baseline_v2.npy', values)
    paths = []
    for i, factor in enumerate((.9, 1.2)):
        ds = xr.Dataset({
            'precip': (('time', 'Ydim', 'Xdim'), (truth*factor)[None], {'units': 'mm h-1'}),
            'regression_precip': (('time', 'Ydim', 'Xdim'), (truth*.95)[None], {'units': 'mm h-1'}),
            **{k: (('Ydim', 'Xdim'), static[k]) for k in ('lat', 'lon')},
        }, coords={'time': [np.datetime64(entry['time'])]})
        ds.attrs.update(version='v2', stage='flow', dataset_fingerprint='fixture', split='test',
                        ensemble_member=i, seed=317+i, checkpoint_sha256='checkpoint', regression_sha256='regression',
                        checkpoint_epoch=100, ode_steps=24, blend='weighted',
                        target_alignment='HR snapshot vs coarse hourly mean', conservation='none')
        path = predictions/f'{entry["id"]}_m{i:03d}_v2.nc'
        ds.to_netcdf(path, engine='h5netcdf')
        paths.append(path)
    cfg = {'data': {'prepared': str(archive)}, 'patch': {'size': 16, 'stride': 12}}
    return cfg, predictions, paths, entry, static


def test_saved_prediction_audit_end_to_end_and_protects_inputs(saved, tmp_path):
    cfg, predictions, paths, entry, _ = saved
    before = [path.read_bytes() for path in paths]
    out = run_audit(cfg, predictions, tmp_path/'audit_v2', members=2)
    summary = json.loads((out/'summary_v2.json').read_text())
    assert summary['hours_audited'] == 1 and summary['members'] == 2
    assert summary['identity']['checkpoint_epoch'] == 100
    assert len(list(out.glob('*.png'))) == 4
    assert (out/'rainfall_scores_v2.csv').is_file()
    assert (out/'report_v2.md').is_file()
    report = json.loads((out/f'{entry["id"]}_audit_v2.json').read_text())
    assert report['profiles']['flow_mean']['mean_ratio_to_HWT'] == pytest.approx(1.05)
    assert abs(report['scores']['flow_mean']['bias']) < abs(report['scores']['coarse']['bias'])
    assert [path.read_bytes() for path in paths] == before
    with pytest.raises(FileExistsError):
        run_audit(cfg, predictions, out, members=2)
    with pytest.raises(ValueError, match='No unique val'):
        run_audit(cfg, predictions, tmp_path/'bad_split_v2', split='val', timestamps=[entry['id']])


@pytest.mark.parametrize('problem', ['units', 'time', 'checkpoint', 'regression', 'grid', 'seed'])
def test_loader_rejects_mixed_or_corrupt_members(saved, problem):
    _, _, paths, entry, static = saved
    with xr.open_dataset(paths[1]) as source:
        ds = source.load()
    if problem == 'units':
        ds.precip.attrs['units'] = 'kg m-2 s-1'
    elif problem == 'time':
        ds = ds.assign_coords(time=[np.datetime64('2026-02-23T06:30:00')])
    elif problem == 'checkpoint':
        ds.attrs['checkpoint_sha256'] = 'other'
    elif problem == 'regression':
        ds.regression_precip.values[:] += 1
    elif problem == 'grid':
        ds.lon.values[:] += .1
    else:
        ds.attrs['seed'] = 317
    ds.to_netcdf(paths[1], engine='h5netcdf', mode='w')
    with pytest.raises(ValueError):
        load_members(paths, entry, 'fixture', static, expected_members=2)


def test_incomplete_member_count_is_not_silently_accepted(saved, tmp_path):
    cfg, predictions, paths, entry, static = saved
    with pytest.raises(ValueError, match='incomplete ensemble'):
        load_members(paths, entry, 'fixture', static, expected_members=5)
    with pytest.raises(ValueError, match='incomplete ensemble'):
        run_audit(cfg, predictions, tmp_path/'incomplete_v2', members=5, plots=False)


def test_cpu_batch_wrapper_clears_inherited_gpu_memory_and_forwards_args():
    project = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PROJECT_DIR=str(project), SLURM_JOB_ID='123',
               SLURM_MEM_PER_CPU='4000', SLURM_MEM_PER_GPU='48000', SLURM_MEM_PER_NODE='32768',
               SLURM_CPUS_PER_GPU='4', PREDICTIONS='runs/member data_v2', OUTPUT='runs/audit output_v2',
               MEMBERS='5', SPLIT='test', TIMESTAMPS='20260223_0530 20260224_0030')
    # Intercept conda activation and srun, then execute the real wrapper. No job
    # is submitted and no dependency installation is performed by this test.
    script = r'''
source() { :; }
conda() { :; }
srun() {
  test -z "${SLURM_MEM_PER_CPU:-}" && test -z "${SLURM_MEM_PER_GPU:-}"
  test -z "${SLURM_CPUS_PER_GPU:-}" && test "$SLURM_MEM_PER_NODE" = 32768
  printf '%s\n' "$@"
}
. scripts/slurm_audit_precip_v2.sh
'''
    result = subprocess.run(['bash', '-c', script], cwd=project, env=env, text=True, capture_output=True, check=True)
    args = result.stdout.splitlines()
    assert args[:2] == ['python', 'scripts/audit_precip_v2.py']
    assert args[args.index('--predictions')+1] == 'runs/member data_v2'
    assert args[args.index('--output')+1] == 'runs/audit output_v2'
    assert args[-3:] == ['--timestamps', '20260223_0530', '20260224_0030']
