"""v4.1 evaluation: checkpoint resolution, sampler equivalence with v4 inference, end-to-end outputs."""
from copy import deepcopy
import json
import numpy as np
import pytest
import torch
from merraflow.v4_1 import base_config
from merraflow.train_v4_1 import train
from merraflow.inference_v4 import sample_frame
from merraflow.evaluate_v4_1 import (resolve_checkpoint, load_model, DomainSampler, member_seed, select_cases,
                                     evaluate, event_window, random_windows)
from test_v4 import setup  # noqa: F401
from test_v4_1 import packed_cfg  # noqa: F401


@pytest.fixture(scope='module')
def trained(packed_cfg, tmp_path_factory):
    cfg = deepcopy(packed_cfg)
    cfg['train'].update(output=str(tmp_path_factory.mktemp('eval')/'run'), epochs=1, time_limit_hours=None)
    train(cfg)
    return cfg


def test_resolve_checkpoint_variants(trained):
    out = trained['train']['output']
    assert resolve_checkpoint(trained).name == 'best_v4_1.pt'
    assert resolve_checkpoint(trained, 'latest').name == 'last_v4_1.pt'
    epoch = resolve_checkpoint(trained, '1')
    assert epoch.name.startswith('epoch_0001_crps') and epoch.parent.name == 'checkpoints'
    assert resolve_checkpoint(trained, str(epoch)) == epoch
    with pytest.raises(FileNotFoundError):
        resolve_checkpoint(trained, '7')
    with pytest.raises(FileNotFoundError):
        resolve_checkpoint(trained, f'{out}/missing.pt')


def test_batched_sampler_matches_v4_inference(trained):
    device = torch.device('cpu')
    archive, model, conditioner, _ = load_model(trained, resolve_checkpoint(trained), device)
    entry = archive.eligible('test')[0]
    base = base_config(trained)
    base['inference']['steps'] = 2
    seed = member_seed(trained, entry, 0)
    fast = DomainSampler(model, conditioner, archive, entry, base, device, batch=3, threads=2).sample(seed, 2)
    reference = sample_frame(model, conditioner, archive, entry, base, device, seed)
    np.testing.assert_allclose(fast, reference, rtol=1e-5, atol=1e-5)


def test_case_selection_and_windows(trained):
    archive, *_ = load_model(trained, resolve_checkpoint(trained), torch.device('cpu'))
    cases = select_cases(archive, 'test', samples=3, wettest=1, log=lambda *a: None)
    assert cases[0]['reason'].startswith('wettest') and len({c['entry']['id'] for c in cases}) == len(cases)
    explicit = select_cases(archive, 'test', timestamps=[cases[0]['entry']['id']])
    assert explicit[0]['entry']['id'] == cases[0]['entry']['id']
    rain = np.zeros(archive.shape)
    rain[5:9, 7:11] = 10
    ys, xs = event_window(rain, 8)
    assert ys.stop-ys.start == 8 and ys.start <= 5 and ys.stop >= 9 and xs.start <= 7 and xs.stop >= 11
    for ys, xs in random_windows(archive.shape, 8, 3, 1):
        assert 0 <= ys.start and ys.stop <= archive.shape[0] and 0 <= xs.start and xs.stop <= archive.shape[1]


@pytest.mark.parametrize('cartopy', [True, False])
def test_evaluate_writes_maps_diagnostics_and_report(trained, tmp_path, cartopy):
    out = evaluate(trained, 'best', tmp_path/'eval', split='test', samples=1, wettest=1, members=2, steps=2,
                   zooms=1, zoom_size=12, batch=4, threads=2, use_cartopy=cartopy, map_features=False,
                   save=cartopy, log=lambda *a: None)
    metrics = json.loads((out/'metrics_v4_1.json').read_text())
    assert metrics['version'] == 'v4.1' and metrics['members'] == 2 and metrics['cases']
    case = out/'cases'/metrics['cases'][0]['id']
    for name in ('conus_precip.png', 'conus_states.png', 'zoom_event.png', 'zoom_random_1.png'):
        assert (case/name).stat().st_size > 10_000, name
    for name in ('summary_scores.png', 'summary_precip.png', 'summary_categorical.png', 'summary_spectra.png',
                 'summary_calibration.png', 'summary_maps.png', 'report.md'):
        assert (out/name).exists(), name
    scores = metrics['cases'][0]['metrics']
    assert set(scores) >= {'t2m', 'precip', 'q2m', 'wind_speed', 'fss', 'precip_categorical', 'precip_probabilistic'}
    assert scores['precip']['ensemble']['crps'] >= 0
    assert (case/'fields.nc').exists() == cartopy
    with pytest.raises(FileExistsError):
        evaluate(trained, 'best', out, split='test', samples=1, wettest=0, members=2, steps=1, log=lambda *a: None)


def test_native_geosfp_fields_are_original_cells(trained):
    from merraflow.evaluate_v4_1 import native_fields, Canvas
    archive, *_ = load_model(trained, resolve_checkpoint(trained), torch.device('cpu'))
    entry = archive.eligible('test')[0]
    fields, missing = native_fields(entry, archive.static['lat'], archive.static['lon'], log=lambda *a: None)
    # Synthetic slv files carry T2M/PS/U10M/V10M but no QV2M: q2m falls back, the rest are native.
    assert missing == ['q2m']
    assert {'precip', 't2m', 'ps', 'u10m', 'v10m', 'wind_speed'} <= set(fields)
    precip = fields['precip']
    assert precip['values'].shape == (len(precip['lat']), len(precip['lon']))  # its own lat/lon grid
    assert precip['values'].shape[0] < archive.shape[0] and (precip['values'] >= 0).all()
    np.testing.assert_allclose(fields['wind_speed']['values'], np.hypot(fields['u10m']['values'], fields['v10m']['values']))
    no_native = dict(entry, native=str(entry['native']).replace('flx_Nx', 'missing_Nx'))
    assert native_fields(no_native, archive.static['lat'], archive.static['lon'], log=lambda *a: None)[0] == {}


def test_sharpness_window_and_time_stepping():
    from merraflow.evaluate_v4_1 import make_window, make_time_steps
    # Hann window
    hann = make_window(32, 'hann')
    assert hann.shape == (32, 32)
    assert hann.max() <= 1.0 and hann.min() >= 1e-4

    # Tukey window has flat center
    tukey = make_window(32, 'tukey', tukey_alpha=0.3)
    assert tukey.shape == (32, 32)
    assert np.isclose(tukey[16, 16], 1.0)
    assert (tukey == 1.0).sum() > 0

    # Uniform time steps (gamma=1.0)
    t_uni = make_time_steps(10, gamma=1.0)
    assert len(t_uni) == 11 and t_uni[0] == 0.0 and t_uni[-1] == 1.0
    np.testing.assert_allclose(np.diff(t_uni), 0.1)

    # Warped time steps (gamma=1.5): earlier steps larger, later steps smaller
    t_warp = make_time_steps(10, gamma=1.5)
    assert len(t_warp) == 11 and t_warp[0] == 0.0 and np.isclose(t_warp[-1], 1.0)
    dt = np.diff(t_warp)
    assert dt[0] > dt[-1]  # concentrated near t=1


def test_sampler_sharpness_variants(trained):
    from merraflow.evaluate_v4_1 import churn_time
    archive, model, conditioner, _ = load_model(trained, resolve_checkpoint(trained), torch.device('cpu'))
    entry = archive.eligible('test')[0]
    base = base_config(trained)
    seed = member_seed(trained, entry, 0)
    sampler = DomainSampler(model, conditioner, archive, entry, base, torch.device('cpu'), batch=2, threads=2)
    out_base = sampler.sample(seed, steps=2)
    core = sampler.integrate(seed, 2)
    np.testing.assert_array_equal(sampler.decode(core), out_base)

    out_warp = sampler.sample(seed, steps=2, time_warp_gamma=1.5)
    assert out_warp.shape == out_base.shape and np.isfinite(out_warp).all()

    # Residual scaling amplifies departures from the regression: a sample equal to
    # the regression (rain z and state residuals) is left unchanged.
    at_mean = np.zeros_like(core)
    at_mean[1] = sampler.mean[1]
    np.testing.assert_allclose(sampler.decode(at_mean, residual_scale=1.3), sampler.decode(at_mean), rtol=1e-6, atol=1e-6)
    scaled = sampler.decode(core, residual_scale=1.1)
    z = np.maximum(sampler.mean[1]+1.1*(core[1]-sampler.mean[1]), 0)
    np.testing.assert_allclose(scaled[1], archive.scale*z*(z+2), rtol=1e-5, atol=1e-6)
    assert not np.array_equal(scaled[0], out_base[0])

    out_cutoff = sampler.decode(core, dry_cutoff=0.1)
    assert (out_cutoff[1][out_cutoff[1] > 0] >= 0.1).all()

    # Churn: exact re-noising level, deterministic per seed, off by default.
    t, gamma = .4, .25
    t_hat = churn_time(t, gamma)
    assert t_hat < t and np.isclose((1-t_hat)/t_hat, (1+gamma)*(1-t)/t)
    churned = sampler.sample(seed, steps=3, churn=.2, churn_range=(0., 1.))
    assert np.isfinite(churned).all() and not np.allclose(churned, sampler.sample(seed, steps=3))
    np.testing.assert_array_equal(churned, sampler.sample(seed, steps=3, churn=.2, churn_range=(0., 1.)))

    # Tukey window switches in place and back.
    out_tukey = sampler.sample(seed, steps=2, window_type='tukey', tukey_alpha=0.3)
    assert out_tukey.shape == out_base.shape and sampler.window_key == ('tukey', 0.3)
    np.testing.assert_array_equal(sampler.sample(seed, steps=2, window_type='hann'), out_base)
    with pytest.raises(ValueError):
        sampler.integrate(seed, 2, guide_weight=1.5)


def test_autoguidance(trained):
    from merraflow.evaluate_v4_1 import load_guide, find_guide_checkpoint
    archive, model, conditioner, saved = load_model(trained, resolve_checkpoint(trained), torch.device('cpu'))
    entry = archive.eligible('test')[0]
    base = base_config(trained)
    seed = member_seed(trained, entry, 0)
    plain = DomainSampler(model, conditioner, archive, entry, base, torch.device('cpu'), batch=2, threads=2)
    reference = plain.sample(seed, steps=2)
    # A guide identical to the model leaves the sample unchanged for any weight.
    same = DomainSampler(model, conditioner, archive, entry, base, torch.device('cpu'), batch=2, threads=2, guide=model)
    np.testing.assert_allclose(same.sample(seed, steps=2, guide_weight=2.), reference, rtol=1e-5, atol=1e-5)
    weak = deepcopy(model)
    with torch.no_grad():
        weak.output[-1].weight.zero_()
        weak.output[-1].bias.zero_()  # v_weak = 0, so guidance scales the velocity by w
    guided = DomainSampler(model, conditioner, archive, entry, base, torch.device('cpu'), batch=2, threads=2, guide=weak)
    np.testing.assert_allclose(guided.sample(seed, steps=2, guide_weight=1.), reference, rtol=1e-5, atol=1e-5)
    moved = guided.integrate(seed, 1, guide_weight=1.5)-plain.integrate(seed, 1)
    assert np.abs(moved).max() > 0 and np.isfinite(moved).all()
    # Only earlier kept checkpoints qualify as the weak model.
    assert find_guide_checkpoint(trained, saved['epoch']+1) is None
    assert find_guide_checkpoint(trained, saved['epoch']+5) is not None
    assert load_guide(trained, 'auto', archive, saved, torch.device('cpu'), log=lambda *a: None) is None
    later = dict(saved, epoch=saved['epoch']+4)
    guide = load_guide(trained, 'auto', archive, later, torch.device('cpu'), log=lambda *a: None)
    assert guide is not None and guide['epoch'] == saved['epoch']+1


def test_compare_sharpness_end_to_end(trained, tmp_path):
    from merraflow.compare_sharpness_v4_1 import compare_sharpness, build_methods, METHOD_IDS
    specs = build_methods(METHOD_IDS, ('churn', 'autoguide', 'residual_scale'), 4, 1.5, .1, 1.5, 1.1, .1, .3)
    combined = specs[-1]
    assert combined['id'] == 'combined' and combined['churn'] == .1 and combined['guide_weight'] == 1.5
    assert combined['residual_scale'] == 1.1 and combined['window_type'] == 'hann'
    with pytest.raises(ValueError):
        build_methods(('nope',), (), 4, 1.5, .1, 1.5, 1.1, .1, .3)
    out = compare_sharpness(trained, 'best', tmp_path/'sharp', split='test', samples=1, wettest=1, steps=2,
                            members=2, batch=4, threads=2, use_cartopy=False, map_features=False,
                            log=lambda *a: None)
    summary = json.loads((out/'summary_metrics.json').read_text())
    methods = [m['id'] for m in summary['settings']['methods']]
    # No earlier kept checkpoint in a one-epoch run: autoguidance is dropped, not faked.
    assert 'autoguide' not in methods and summary['settings']['guide'] is None
    assert methods[0] == 'baseline' and methods[-1] == 'combined'
    assert summary['summary']['cases'] >= 1 and summary['summary']['members'] == 2
    for m in methods:
        s = summary['summary']['methods'][m]
        assert s['crps'] >= 0 and s['large_scale_rmse'] >= 0
    assert set(summary['verdicts']) == set(methods)-{'baseline'}
    for folder in (out/'cases').iterdir():
        for name in ('sharpness_compare_precip.png', 'sharpness_compare_zoom.png', 'sharpness_spectra.png', 'metrics.json'):
            assert (folder/name).stat().st_size > 1000, name
    assert (out/'summary_tradeoff.png').exists() and 'Verdicts' in (out/'report.md').read_text()
    with pytest.raises(FileExistsError):
        compare_sharpness(trained, 'best', out, split='test', samples=0, wettest=1, steps=1, members=1,
                          log=lambda *a: None)
