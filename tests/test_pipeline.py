from copy import deepcopy
from pathlib import Path
import json
import hashlib
import numpy as np
import pytest
import torch
import xarray as xr
from merraflow.config import load_config
from merraflow.synthetic import make_synthetic
from merraflow.prepare import prepare, prepare_month, finalize_prepare, manifest, field, grid_mismatches
from merraflow.dataset import Archive, PatchDataset, crop
from merraflow.model import VelocityUNet, flow_loss, integrate
from merraflow.inference import starts, blend_window, sample_frame, predict
from merraflow.train import train
from merraflow.evaluate import evaluate


@pytest.fixture(scope='module')
def prepared(tmp_path_factory):
    torch.set_num_threads(1)
    root = tmp_path_factory.mktemp('archive')
    template = load_config(Path(__file__).resolve().parents[1]/'configs/discover.yaml')
    cfg = load_config(make_synthetic(root, template))
    prepare(cfg)
    return cfg


def test_pairing_crosses_month_and_split_validation(prepared):
    entries, missing = manifest(prepared)
    assert not missing and len(entries) == 6
    e = entries[1]
    assert all('20250831_2330' in e[key] for key in ('hr', 'lr', 'native'))
    assert all('acc' not in entry for entry in entries)
    assert '20250901_0030' in entries[2]['hr']
    cfg = deepcopy(prepared)
    cfg['data']['splits']['val'][0] = '2025-08-01'
    with pytest.raises(ValueError, match='overlap'):
        manifest(cfg)


def test_stats_train_only_and_patches(prepared):
    a = Archive(prepared['data']['prepared'])
    entries = [e for e in a.index['entries'] if e['split'] == 'train']
    exact = np.concatenate([a.array(e, 'condition')[:, ::2, ::2].reshape(4, -1) for e in entries], axis=1)
    np.testing.assert_allclose(a.cm.ravel(), exact.mean(1), rtol=1e-5, atol=1e-5)
    ds = PatchDataset(a.root, 'train', 16, 4, 8, seed=12)
    assert ds[0]['target'].shape == (4, 24, 24)
    assert ds[0]['condition'].shape == (20, 24, 24)
    np.testing.assert_array_equal(ds[0]['target'], ds[0]['target'])
    first = ds[0]['condition'].clone()
    ds.epoch = 1
    assert not torch.equal(first, ds[0]['condition'])
    assert crop(np.ones((4, 5)), 0, 0, 2, 3).shape == (8, 8)


def test_hr_precip_matches_diagnostic_source_and_conversion(prepared):
    from merraflow.prepare import build_arrays
    archive = Archive(prepared['data']['prepared'])
    cfg = deepcopy(prepared)
    cfg['data']['conserve_training_precip'] = False
    for entry in archive.index['entries']:
        with xr.open_dataset(entry['hr']) as hr:
            expected = field(hr, 'PRECTOT')*3600.0
        np.testing.assert_array_equal(archive.array(entry, 'truth')[1], expected)
        # With conservation off, training uses exactly the diagnostic's HR field.
        arrays = build_arrays(cfg, entry, archive.static)
        np.testing.assert_array_equal(arrays['target'][1], expected)
        np.testing.assert_array_equal(arrays['truth'][1], expected)


def test_prepare_needs_no_accumulation_files(tmp_path):
    template = load_config(Path(__file__).resolve().parents[1]/'configs/discover.yaml')
    cfg = load_config(make_synthetic(tmp_path/'no_acc', template))
    for path in Path(cfg['data']['highres_root']).glob('hwt_01hr_acc_LCC/*/*.nc4'):
        path.unlink()
    assert not manifest(cfg)[1]
    prepare(cfg)
    assert len(Archive(cfg['data']['prepared']).index['entries']) == 6


@pytest.mark.parametrize('problem', ['missing', 'units', 'time'])
def test_invalid_hr_precip_fails_without_apcp_fallback(prepared, tmp_path, problem):
    from merraflow.prepare import build_arrays
    archive = Archive(prepared['data']['prepared'])
    entry = dict(archive.index['entries'][0])
    with xr.open_dataset(entry['hr']) as source:
        hr = source.load()
    hr['APCP'] = hr.PRECTOT*3600
    if problem == 'missing':
        hr = hr.drop_vars('PRECTOT')
        message = 'missing PRECTOT'
    elif problem == 'units':
        hr.PRECTOT.attrs['units'] = 'mm'
        message = 'unsupported rate units'
    else:
        hr = hr.assign_coords(time=hr.time + np.timedelta64(30, 'm'))
        message = 'data time'
    path = tmp_path/'invalid_hr.nc'
    hr.to_netcdf(path, engine='h5netcdf')
    entry['hr'] = str(path)
    with pytest.raises(ValueError, match=message):
        build_arrays(prepared, entry, archive.static)


def test_legacy_archives_and_partial_shards_are_rejected(prepared, tmp_path):
    from merraflow.prepare import ensure_work_signature
    from merraflow.config import write_json
    cfg = deepcopy(prepared)
    cfg['data']['prepared'] = str(tmp_path)
    old_index = {'data_config': cfg['data'], 'format': 2}
    write_json(tmp_path/'index.json', old_index)
    with pytest.raises(ValueError, match='Legacy prepared archive'):
        Archive(tmp_path)
    with pytest.raises(ValueError, match='different data configuration'):
        prepare_month(cfg, '2025-08')
    with pytest.raises(ValueError, match='different data configuration'):
        finalize_prepare(cfg)
    (tmp_path/'index.json').unlink()
    write_json(tmp_path/'_preparation.json', {'format': 2, 'data_config': cfg['data']})
    with pytest.raises(ValueError, match='different preparation configuration'):
        ensure_work_signature(tmp_path, cfg)
    (tmp_path/'_preparation.json').unlink()
    shard = tmp_path/'20250831_2230'
    shard.mkdir()
    np.save(shard/'truth.npy', np.zeros((4, 2, 2), dtype='float32'))
    with pytest.raises(ValueError, match='unversioned shards'):
        ensure_work_signature(tmp_path, cfg)


def test_missing_files_are_reported(prepared):
    cfg = deepcopy(prepared)
    cfg['data']['end'] = '2025-09-01T04:30:00'
    cfg['data']['splits']['test'][1] = '2025-09-01T06:00:00'
    _, missing = manifest(cfg)
    assert len(missing) == 1 and len(missing[0]['missing']) == 3


def test_monthly_prepare_resumes_and_finalizes(tmp_path):
    template = load_config(Path(__file__).resolve().parents[1]/'configs/discover.yaml')
    cfg = load_config(make_synthetic(tmp_path/'monthly', template))
    august = prepare_month(cfg, '2025-08')
    first = json.loads(august.read_text())
    assert first['written'] == 2 and first['skipped'] == 0
    second = json.loads(prepare_month(cfg, '2025-08').read_text())
    assert second['written'] == 0 and second['skipped'] == 2
    (Path(cfg['data']['prepared'])/'20250831_2230'/'residual.npy').unlink()
    repaired = json.loads(prepare_month(cfg, '2025-08').read_text())
    assert repaired['written'] == 1 and repaired['skipped'] == 1
    prepare_month(cfg, '2025-09')
    root = finalize_prepare(cfg)
    archive = Archive(root)
    assert len(archive.index['entries']) == 6
    assert archive.stats['condition']['count_per_channel'] == 792
    assert (root/'20250831_2230'/'condition.npy').exists()


def test_month_filter_requires_zero_padding(prepared):
    with pytest.raises(ValueError, match='zero-padded'):
        manifest(prepared, month='2025-8')


def test_grid_comparison_tolerates_encoding_roundoff(prepared):
    static = Archive(prepared['data']['prepared']).static
    rounded = {name: value.copy() for name, value in static.items()}
    rounded['lat'] += 5e-6
    rounded['area'] *= 1+5e-7
    assert not grid_mismatches(static, rounded)
    rounded['lon'] += 1e-2
    assert any(item.startswith('lon ') for item in grid_mismatches(static, rounded))


def test_predictor_gaps_report_file_and_all_missing_variables(tmp_path):
    from merraflow.prepare import predictor_gaps
    path = tmp_path/'incomplete.nc'
    xr.Dataset({'T2M': (('y', 'x'), np.ones((2, 2))) }).to_netcdf(path)
    gaps = predictor_gaps([{'id': '20250107_0130', 'lr': str(path)}],
                          ['T2M', 'PRECTOT', 'TQV'])
    assert gaps == [{'id': '20250107_0130', 'path': str(path),
                     'missing': ['PRECTOT', 'PS', 'TQV', 'U10M', 'V10M']}]


def test_network_backward_and_heun(prepared):
    ds = PatchDataset(prepared['data']['prepared'], 'train', 16, 4, 2)
    batch = {k: v[None] for k, v in ds[0].items()}
    model = VelocityUNet(20, **prepared['model'])
    loss = flow_loss(model, batch, 4, [1]*4)
    loss.backward()
    assert torch.isfinite(loss) and model.output[-1].weight.grad.abs().sum() > 0
    class Constant(torch.nn.Module):
        def forward(self, x, t, c):
            return torch.ones_like(x)*2
    zero = torch.zeros((1, 4, 8, 8))
    np.testing.assert_allclose(integrate(Constant(), zero, zero, 3).numpy(), 2, atol=1e-6)


def test_stitching_covers_irregular_domain():
    h, w, size, stride = 35, 47, 16, 12
    denom = np.zeros((h, w))
    numer = np.zeros((h, w))
    win = blend_window(size)
    for y in starts(h, size, stride):
        for x in starts(w, size, stride):
            denom[y:y+size, x:x+size] += win
            numer[y:y+size, x:x+size] += 7*win
    assert denom.min() > 0
    np.testing.assert_allclose(numer/denom, 7, rtol=1e-6)


def test_train_resume_predict_evaluate(prepared, tmp_path, monkeypatch):
    import importlib
    module = importlib.import_module('merraflow.train')
    original_save = module.atomic_save
    epoch0 = tmp_path/'epoch0.pt'
    def capture(path, value):
        original_save(path, value)
        if str(path).endswith('last.pt') and value['epoch'] == 0:
            original_save(epoch0, value)
    monkeypatch.setattr(module, 'atomic_save', capture)
    ckpt_path = train(prepared)
    ckpt = torch.load(ckpt_path, weights_only=True)
    assert ckpt['step'] > 0 and np.isfinite(ckpt['best'])
    uninterrupted = torch.load(Path(prepared['train']['output'])/'last.pt', weights_only=True)
    cfg = deepcopy(prepared)
    cfg['train']['output'] = str(tmp_path/'resumed')
    train(cfg, resume=epoch0)
    resumed = torch.load(Path(cfg['train']['output'])/'last.pt', weights_only=True)
    for key in uninterrupted['model']:
        torch.testing.assert_close(uninterrupted['model'][key], resumed['model'][key], rtol=0, atol=0)
    predict(prepared, ckpt_path, limit=1)
    dest = evaluate(prepared)
    report = json.loads((dest/'summary.json').read_text())
    assert report['hours_evaluated'] == 1 and len(report['missing_hours']) == 1
    for path in Path(prepared['inference']['output']).glob('*.nc'):
        with xr.open_dataset(path) as ds:
            assert ds.precip.min() >= 0 and ds.wind10m.min() >= 0
            audit = json.loads(ds.attrs['conservation_audit'])
            assert audit['after']['max_relative_wet'] < 1e-6
            assert (ds.lr_time_bounds.values[0, 1]-ds.lr_time_bounds.values[0, 0])/np.timedelta64(1, 'h') == 1
            assert 'bounds' not in ds.time.attrs
            assert ds.precip.attrs['cell_methods'] == 'time: point'
            assert ds.attrs['precip_source'] == 'hwt_30mn_slv_LCC.PRECTOT'
            assert ds.attrs['checkpoint_sha256'] == hashlib.sha256(Path(ckpt_path).read_bytes()).hexdigest()


def test_unlabeled_inference_uses_frozen_statistics(prepared, tmp_path):
    from merraflow.prepare import prepare_predict
    cfg = deepcopy(prepared)
    cfg['data']['prepared'] = str(tmp_path/'unlabeled')
    # Deliberately inaccessible HR paths: production inference must not read labels.
    cfg['data']['highres_root'] = '/does-not-exist'
    cfg['data']['start'] = cfg['data']['end'] = '2025-09-01T02:30:00'
    prepare_predict(cfg, prepared['data']['prepared'])
    a = Archive(cfg['data']['prepared'])
    original = Archive(prepared['data']['prepared'])
    assert a.stats == original.stats and a.index['fingerprint'] == original.index['fingerprint']
    e = a.index['entries'][0]
    assert e['split'] == 'predict'
    assert not (a.root/e['id']/'truth.npy').exists()
    assert a.condition(e, 0, 0, 16, 4).shape == (20, 24, 24)


def test_precip_rate_units_and_time_mismatch_fail():
    from merraflow.prepare import units, assert_time
    ds = xr.Dataset({'PRECTOT': (('Ydim', 'Xdim'), np.ones((2, 2)), {'units': 'mm'})}, coords={'time': [np.datetime64('2025-09-01T01:00')]})
    with pytest.raises(ValueError, match='unsupported'):
        units(ds, 'PRECTOT', 'rate')
    from datetime import datetime
    with pytest.raises(ValueError, match='time'):
        assert_time(ds, datetime(2025, 9, 1, 2), 'fixture')


def test_hwt_plus_exponent_units_are_recognized():
    from merraflow.prepare import unit_text, units
    ds = xr.Dataset({
        'AREA': (('Ydim', 'Xdim'), np.ones((2, 2)), {'units': 'm+2'}),
        'HGT_SFC': (('Ydim', 'Xdim'), np.ones((2, 2)), {'units': 'm+2 s-2'}),
    })
    units(ds, 'AREA', 'area')
    assert unit_text(ds.HGT_SFC) == 'm2s-2'
