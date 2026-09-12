from copy import deepcopy
from pathlib import Path
import json
import numpy as np
import pytest
import torch
import xarray as xr
from merraflow.config_v2 import load_config_v2, validate_config_v2
from merraflow.synthetic_v2 import make_synthetic_v2
from merraflow.prepare_v2 import prepare, prepare_month, finalize_prepare, prepare_predict
from merraflow.dataset_v2 import ArchiveV2, PatchDatasetV2
from merraflow.static_v2 import add_surface_features_v2
from merraflow.physics_v2 import transform_v2, inverse_v2
from merraflow.model_v2 import UNetV2, regression_v2
from merraflow.loss_v2 import loss_v2, quadratic_v2, gradient_v2
from merraflow.train_v2 import train_v2
from merraflow.inference_v2 import predict_v2, sample_frame_v2
from merraflow.evaluate_v2 import evaluate_v2


@pytest.fixture(scope='module')
def prepared_v2(tmp_path_factory):
    torch.set_num_threads(1)
    root = tmp_path_factory.mktemp('pipeline_v2')
    cfg = load_config_v2(make_synthetic_v2(root, load_config_v2('configs/discover_v2.yaml')))
    prepare(cfg)
    return cfg


def test_original_targets_and_signed_wind_v2(prepared_v2):
    a = ArchiveV2(prepared_v2['data']['prepared'])
    from merraflow.prepare import field
    for e in a.index['entries']:
        with xr.open_dataset(e['hr']) as ds:
            expected = np.stack([field(ds, 'TMP_2M'), field(ds, 'PRECTOT')*3600, field(ds, 'PRES_SFC'),
                                 field(ds, 'UGRD_10M'), field(ds, 'VGRD_10M')])
        np.testing.assert_array_equal(a.array(e, 'target'), expected)
        np.testing.assert_array_equal(a.array(e, 'truth'), expected)
    x = np.array([280, 4, 98000, -7, -3], dtype='float32')[:, None, None]
    np.testing.assert_allclose(inverse_v2(transform_v2(x, 1), 1), x, rtol=1e-6)
    assert not (a.root/'index.json').exists()
    audit = json.loads((a.root/'precip_audit_v2.json').read_text())
    assert all(x['precip_adjustment_mae_mm_h'] == 0 for x in audit)


def test_static_grid_and_fraction_validation_v2(prepared_v2, tmp_path):
    a = ArchiveV2(prepared_v2['data']['prepared'])
    spec = dict(prepared_v2['data']['static'])
    with xr.open_dataset(spec['path']) as ds:
        original = ds.load()
    wrong = original.copy(deep=True)
    wrong['lons'] += .1
    spec['path'] = str(tmp_path/'wrong_v2.nc')
    wrong.to_netcdf(spec['path'], engine='h5netcdf')
    with pytest.raises(ValueError, match='does not match'):
        add_surface_features_v2(dict(a.static), spec)
    wrong = original.copy(deep=True)
    wrong['land_fraction'][:] = 1
    wrong.to_netcdf(spec['path'], engine='h5netcdf')
    with pytest.raises(ValueError, match='disjoint'):
        add_surface_features_v2(dict(a.static), spec)


def test_frocean_with_optional_lakes_v2(prepared_v2, tmp_path):
    a = ArchiveV2(prepared_v2['data']['prepared'])
    path = tmp_path/'frocean_v2.nc'
    ds = xr.Dataset({k: (('Ydim', 'Xdim'), value) for k, value in
                     [('FROCEAN', a.static['ocean_fraction']), ('lats', a.static['lat']), ('lons', a.static['lon'])]})
    ds.to_netcdf(path, engine='h5netcdf')
    spec = dict(path=str(path), ocean='FROCEAN', lake='FRLAKE', lat='lats', lon='lons')
    original = dict(a.static)
    original['features'] = original['features'][:6]
    with pytest.warns(UserWarning, match='unresolved inland lakes'):
        result = add_surface_features_v2(dict(original), spec)
    assert not result['lake_fraction_known'].any()
    np.testing.assert_allclose(result['land_fraction'], 1-result['ocean_fraction'])
    with pytest.raises(ValueError, match='missing'):
        add_surface_features_v2(dict(original), {**spec, 'require_lake': True})
    ds['FRLAKE'] = (('Ydim', 'Xdim'), a.static['lake_fraction'])
    ds.to_netcdf(path, engine='h5netcdf')
    result = add_surface_features_v2(dict(original), spec)
    assert result['lake_fraction_known'].all()
    np.testing.assert_allclose(result['land_fraction'], a.static['land_fraction'])


def test_preflight_rejects_incomplete_manifest_v2(prepared_v2, monkeypatch):
    from merraflow import audit_v2
    from merraflow.prepare_v2 import manifest
    entries, missing = manifest(prepared_v2)
    assert not missing
    monkeypatch.setattr(audit_v2, 'manifest', lambda cfg: (entries, [{'missing': ['absent.nc4']}]))
    with pytest.raises(FileNotFoundError, match='incomplete hourly pairs'):
        audit_v2.audit_v2(prepared_v2)
    monkeypatch.setattr(audit_v2, 'manifest', lambda cfg: ([e for e in entries if e['split'] != 'test'], []))
    with pytest.raises(ValueError, match='each train/val/test split'):
        audit_v2.audit_v2(prepared_v2)


def test_preflight_reports_surface_fractions_v2(prepared_v2):
    from merraflow.audit_v2 import audit_v2
    result = audit_v2(prepared_v2)
    assert result['lake_fraction_known']
    assert result['surface_file'] == prepared_v2['data']['static']['path']
    fractions = result['surface_fractions']
    assert set(fractions) == {'land_fraction', 'lake_fraction', 'ocean_fraction'}
    for values in fractions.values():
        assert 0 <= values['min'] <= values['mean'] <= values['max'] <= 1
    assert sum(values['mean'] for values in fractions.values()) == pytest.approx(1)


def test_patch_proposals_and_context_v2(prepared_v2):
    cfg = prepared_v2
    data = PatchDatasetV2(cfg['data']['prepared'], 'train', cfg['patch'], 10, 17)
    q = data.proposal(data.entries[0])
    weights = 1/(len(q)*q)
    np.testing.assert_allclose(np.sum(q*weights), 1)
    signal = np.arange(len(q))**2
    np.testing.assert_allclose(np.sum(q*weights*signal), signal.mean())
    assert not np.allclose(q, 1/len(q))
    b = data[0]
    assert b['target'].shape == (5, 24, 24)
    assert b['condition'].shape == (31, 24, 24)
    assert b['context'].shape == (31, 16, 16)
    assert torch.equal(b['target'], data[0]['target'])
    data.epoch = 1
    assert not torch.equal(b['condition'], data[0]['condition'])
    val = PatchDatasetV2(cfg['data']['prepared'], 'val', cfg['patch'], 2)
    np.testing.assert_allclose(val.proposal(val.entries[0]), 1/len(val.yy))
    assert val[0]['importance'] == 1


def test_train_only_statistics_v2(prepared_v2):
    a = ArchiveV2(prepared_v2['data']['prepared'])
    values = np.concatenate([a.array(e, 'residual')[:, ::2, ::2].reshape(5, -1)
                             for e in a.index['entries'] if e['split'] == 'train'], axis=1)
    np.testing.assert_allclose(a.rm.ravel(), values.mean(1), rtol=1e-5, atol=1e-5)
    coverage = a.stats['training_coverage']
    assert coverage['hours'] == 2
    assert coverage['hours_by_month'] == {'2025-08': 2}


def test_monthly_statistics_merge_twelve_training_months_v2(prepared_v2, tmp_path, monkeypatch):
    from merraflow import prepare_v2 as preparation
    cfg = deepcopy(prepared_v2)
    cfg['data']['prepared'] = str(tmp_path/'annual_v2')
    annual = load_config_v2('configs/discover_annual_v2.yaml')
    for name in ('start', 'end', 'splits'):
        cfg['data'][name] = annual['data'][name]
    archive = ArchiveV2(prepared_v2['data']['prepared'])
    static = archive.static
    months = preparation.preparation_months(cfg)
    entries = [dict(archive.index['entries'][0], time=f'{month}-15T00:30:00',
                    id=f'{month.replace("-", "")}15_0030',
                    split='train' if i < 12 else 'val' if i < 14 else 'test')
               for i, month in enumerate(months)]
    monkeypatch.setattr(preparation, 'manifest', lambda cfg, month=None:
                        ([e for e in entries if month is None or e['time'].startswith(month)], []))
    monkeypatch.setattr(preparation, 'static_for', lambda entry, cfg: static)

    def arrays(cfg, entry, static):
        i = entries.index(entry)
        value = i+1 if entry['split'] == 'train' else 1e6
        return {name: np.full((len(cfg['data']['predictors']) if name == 'condition' else
                              1 if name == 'native_reference' else 5, *static['area'].shape),
                             value, dtype='float32') for name in preparation.SHARD_NAMES}

    monkeypatch.setattr(preparation, 'build_arrays', arrays)
    for month in months:
        prepare_month(cfg, month)
    finalize_prepare(cfg)
    stats = json.loads((Path(cfg['data']['prepared'])/'stats_v2.json').read_text())
    expected = np.arange(1, 13)
    for name in ('condition', 'residual'):
        np.testing.assert_allclose(stats[name]['mean'], expected.mean())
        np.testing.assert_allclose(stats[name]['std'], expected.std())
    assert stats['training_coverage']['hours_by_month'] == dict.fromkeys(months[:12], 1)
    assert stats['training_coverage']['hours'] == 12


def test_coverage_identifies_missing_source_v2(tmp_path):
    from merraflow.audit_v2 import coverage_v2
    from merraflow.prepare_v2 import manifest
    cfg = load_config_v2(make_synthetic_v2(tmp_path/'coverage_v2', load_config_v2('configs/discover_v2.yaml')))
    entries, _ = manifest(cfg)
    Path(entries[0]['lr']).unlink()
    result = coverage_v2(cfg)
    assert result['expected_hours'] == 6 and result['paired_hours'] == 5
    assert result['incomplete_hours'] == 1 and not result['all_requested_pairs_present']
    assert result['months'][0]['missing_by_source'] == {'lr': 1, 'hr': 0, 'native': 0}
    assert result['example_missing_paths'] == {'lr': entries[0]['lr']}
    spot = result['predictor_spot_check']
    assert spot['months_sampled'] == sum(1 for month in result['months'] if month['paired_hours'])
    assert spot['gaps'] == [] and spot['all_sampled_months_have_predictors']


def test_coverage_spot_check_reports_absent_predictor_v2(tmp_path):
    from merraflow.audit_v2 import coverage_v2
    cfg = load_config_v2(make_synthetic_v2(tmp_path/'spot_v2', load_config_v2('configs/discover_v2.yaml')))
    cfg['data']['predictors'] = cfg['data']['predictors']+['OMEGA700']
    result = coverage_v2(cfg)
    assert result['all_requested_pairs_present']
    spot = result['predictor_spot_check']
    assert not spot['all_sampled_months_have_predictors']
    assert spot['gaps'][0]['missing'] == ['OMEGA700']


def test_gradient_loss_distinguishes_detail_v2():
    area, importance = torch.ones(1, 16, 16), torch.ones(1)
    low = torch.ones(1, 5, 16, 16)
    high = low.clone()
    high[..., ::2] = -1
    torch.testing.assert_close(quadratic_v2(low, area, importance, [1]*5)[0],
                               quadratic_v2(high, area, importance, [1]*5)[0])
    assert gradient_v2(low, area, importance, [1]*5) == 0
    assert gradient_v2(high, area, importance, [1]*5) > 0


def test_both_objectives_backward_v2(prepared_v2):
    cfg = prepared_v2
    data = PatchDatasetV2(cfg['data']['prepared'], 'train', cfg['patch'], 2)
    batch = {k: v[None] for k, v in data[0].items()}
    mean = UNetV2(31, **cfg['model'])
    loss, _ = loss_v2(mean, batch, cfg, 'regression')
    loss.backward()
    assert torch.isfinite(loss) and mean.output[-1].weight.grad.abs().sum() > 0
    mean.zero_grad(set_to_none=True)
    mean.eval().requires_grad_(False)
    flow = UNetV2(31, **cfg['model'], mean_condition=True)
    loss, metrics = loss_v2(flow, batch, cfg, 'flow', mean, torch.ones(1, 5, 1, 1))
    loss.backward()
    assert metrics.shape == (9,) and flow.output[-1].weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in mean.parameters())


def test_no_inference_conservation_v2(prepared_v2):
    class Rain(torch.nn.Module):
        def forward(self, x, time, condition, context):
            out = torch.zeros_like(x)
            out[:, 1] = 5
            return out
    a = ArchiveV2(prepared_v2['data']['prepared'])
    entry = a.index['entries'][0]
    result, _, audit = sample_frame_v2(Rain(), None, None, a, entry, prepared_v2, torch.device('cpu'), 9)
    dry = a.array(entry, 'native_reference')[0] == 0
    assert dry.any() and (result[1, dry] > 0).any()
    assert audit['max_relative_wet'] > 0
    cfg = deepcopy(prepared_v2)
    cfg['inference']['blend'] = 'owner'
    owned, _, _ = sample_frame_v2(Rain(), None, None, a, entry, cfg, torch.device('cpu'), 9)
    np.testing.assert_allclose(result, owned, rtol=1e-5, atol=1e-4)


def test_two_stages_resume_predict_evaluate_v2(prepared_v2, tmp_path):
    cfg = deepcopy(prepared_v2)
    cfg['train']['output'] = str(tmp_path/'run_v2')
    cfg['inference']['output'] = str(tmp_path/'predictions_v2')
    mean = train_v2(cfg, 'regression')
    flow = train_v2(cfg, 'flow', regression_checkpoint=mean)
    for stage in ('regression', 'flow'):
        last = torch.load(Path(cfg['train']['output'])/f'{stage}_v2'/'last_v2.pt', weights_only=True)
        resumed_cfg = deepcopy(cfg)
        resumed_cfg['train']['output'] = str(tmp_path/f'resumed_{stage}_v2')
        train_v2(resumed_cfg, stage, resume=Path(cfg['train']['output'])/f'{stage}_v2'/'epoch_0001_v2.pt')
        resumed = torch.load(Path(resumed_cfg['train']['output'])/f'{stage}_v2'/'last_v2.pt', weights_only=True)
        for key in last['model']:
            torch.testing.assert_close(last['model'][key], resumed['model'][key], rtol=0, atol=0)
        if stage == 'flow':
            assert (last['flow_scale'] >= .05).all()
            torch.testing.assert_close(last['flow_scale'], resumed['flow_scale'])
        changed = deepcopy(cfg)
        changed['loss']['flow_gradient'] += .1
        with pytest.raises(ValueError, match='Exact resume'):
            train_v2(changed, stage, resume=Path(cfg['train']['output'])/f'{stage}_v2'/'last_v2.pt')
    predict_v2(cfg, flow, limit=1)
    with pytest.raises(FileExistsError):
        predict_v2(cfg, flow, limit=1)
    report = evaluate_v2(cfg)
    summary = json.loads((report/'summary_v2.json').read_text())
    assert summary['hours_evaluated'] == 1 and len(summary['missing_hours']) == 1
    assert len(summary['metrics']['ensemble']) == 6
    output = next(Path(cfg['inference']['output']).glob('*_v2.nc'))
    with xr.open_dataset(output) as ds:
        assert ds.attrs['conservation'] == 'none; audit only'
        assert ds.precip.min() >= 0
        np.testing.assert_allclose(ds.wind10m, np.hypot(ds.u10m, ds.v10m))


def test_monthly_and_unlabeled_v2(tmp_path):
    cfg = load_config_v2(make_synthetic_v2(tmp_path/'monthly_v2', load_config_v2('configs/discover_v2.yaml')))
    prepare_month(cfg, '2025-08')
    result = json.loads(prepare_month(cfg, '2025-08').read_text())
    assert result['written'] == 0 and result['skipped'] == 2
    prepare_month(cfg, '2025-09')
    root = finalize_prepare(cfg)
    inference = deepcopy(cfg)
    inference['data'].update(prepared=str(tmp_path/'unlabeled_v2'), highres_root='/inaccessible',
                             start='2025-09-01T02:30:00', end='2025-09-01T02:30:00')
    prepare_predict(inference, root)
    a, reference = ArchiveV2(inference['data']['prepared']), ArchiveV2(root)
    assert a.stats == reference.stats and a.index['fingerprint'] == reference.index['fingerprint']
    assert not (a.root/a.index['entries'][0]['id']/'truth_v2.npy').exists()
    assert a.inputs(a.index['entries'][0], 0, 0, cfg['patch'])[0].shape[0] == 30


def test_namespace_and_conservation_guards_v2(prepared_v2):
    cfg = deepcopy(prepared_v2)
    cfg['data']['prepared'] = 'data/paired_hourly_prectot'
    with pytest.raises(ValueError, match='v2 in its basename'):
        validate_config_v2(cfg)
    cfg = deepcopy(prepared_v2)
    cfg['inference']['conserve_precip'] = True
    with pytest.raises(ValueError, match='never projects'):
        validate_config_v2(cfg)


def test_first_hr_frocean_and_static_change_guard_v2(tmp_path):
    from merraflow.prepare_v2 import manifest, static_for
    cfg = load_config_v2(make_synthetic_v2(tmp_path/'sources_v2', load_config_v2('configs/discover_v2.yaml')))
    entries, _ = manifest(cfg)
    with xr.open_dataset(entries[0]['hr']) as source:
        hr = source.load()
    with xr.open_dataset(cfg['data']['static']['path']) as surface:
        hr['FROCEAN'] = 1-surface.land_fraction-surface.lake_fraction
        hr['FRLAKE'] = surface.lake_fraction.load()
    hr.to_netcdf(entries[0]['hr'], engine='h5netcdf')
    ocean_cfg = deepcopy(cfg)
    ocean_cfg['data']['static'] = dict(path='first_hr', ocean='FROCEAN', lake='FRLAKE', lat='lats', lon='lons')
    assert static_for(entries[0], ocean_cfg)['lake_fraction_known'].all()
    prepare_month(cfg, '2025-08')
    path = cfg['data']['static']['path']
    with xr.open_dataset(path) as source:
        changed = source.load()
    changed.attrs['revision'] = 'changed_v2'
    changed.to_netcdf(path, engine='h5netcdf')
    with pytest.raises(ValueError, match='different preparation configuration'):
        prepare_month(cfg, '2025-08')
