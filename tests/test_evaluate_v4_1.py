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


@pytest.mark.parametrize('maps', ['index', 'cartopy', 'mesh'])
def test_compare_sharpness_end_to_end(trained, tmp_path, monkeypatch, maps):
    if maps != 'index':
        pytest.importorskip('cartopy')
    if maps == 'mesh':  # LCC grid not recoverable: lon/lat pcolormesh on GeoAxes (Discover's mode)
        import merraflow.evaluate_v4_1 as ev
        monkeypatch.setattr(ev, '_lcc', lambda *a, **k: [])
    from merraflow.compare_sharpness_v4_1 import compare_sharpness, build_methods, METHOD_IDS
    specs = build_methods(METHOD_IDS, ('churn', 'autoguide', 'residual_scale'), 4, 1.5, .1, 1.5, 1.1, .1, .3)
    combined = specs[-1]
    assert combined['id'] == 'combined' and combined['churn'] == .1 and combined['guide_weight'] == 1.5
    assert combined['residual_scale'] == 1.1 and combined['window_type'] == 'hann'
    with pytest.raises(ValueError):
        build_methods(('nope',), (), 4, 1.5, .1, 1.5, 1.1, .1, .3)
    out = compare_sharpness(trained, 'best', tmp_path/'sharp', split='test', samples=1, wettest=1, steps=2,
                            members=2, batch=4, threads=2, zoom_size=12, dpi=60, use_cartopy=maps != 'index', map_features=False,
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
        for name in ('sharpness_spectra.png', 'metrics.json', 'sharpness_all_fields_zoom.png',
                     'sharpness_spectra_all_fields.png', 'sharpness_spectra.pdf'):
            assert (folder/name).stat().st_size > 1000, name
        for var in ('precip', 't2m', 'ps', 'u10m', 'v10m', 'q2m', 'wind_speed'):
            for kind in ('conus', 'zoom'):
                for suffix in ('png', 'pdf'):
                    assert (folder/'maps'/f'{var}_{kind}.{suffix}').stat().st_size > 1000, (var, kind, suffix)
    assert all((out/f'summary_{n}.{x}').exists() for n in ('tradeoff', 'scorecard') for x in ('png', 'pdf'))
    report = (out/'report.md').read_text()
    assert 'Verdicts' in report and all(f'(`{n}`)' in report for n in ('t2m', 'ps', 'u10m', 'v10m', 'q2m', 'wind_speed'))
    states = summary['summary']['states']['baseline']
    assert set(states) == {'t2m', 'ps', 'u10m', 'v10m', 'q2m', 'wind_speed'} and states['t2m']['crps'] >= 0
    with pytest.raises(FileExistsError):
        compare_sharpness(trained, 'best', out, split='test', samples=0, wettest=1, steps=1, members=1,
                          log=lambda *a: None)


def test_front_diagnostic(trained, tmp_path):
    from merraflow.diag_front_v4_1 import run, transition_width, profile
    s = np.arange(-20., 21.)
    step = np.where(s < 0, 0., 1.)
    ramp = np.clip((s+10)/20, 0, 1)
    assert transition_width(s, step, 2.) <= 2*2. and transition_width(s, ramp, 2.) > 25
    from merraflow.diag_front_v4_1 import slope_width
    ramp_then_step = np.where(s < 0, .3*(s+20)/20, 1.)  # gradual pre-frontal ramp, then a jump
    assert slope_width(s, ramp_then_step, 2.) < transition_width(s, ramp_then_step, 2.)
    assert abs(slope_width(s, ramp, 2.)-40) < 3
    field = np.tile(np.where(np.arange(40) < 20, 0., 1.), (40, 1))
    offsets, values = profile(field, (20, 20), (0., 1.), 10, 4)
    assert values[0] == 0 and values[-1] == 1
    out = run(trained, 'best', split='test', members=2, output=tmp_path/'diag', batch=4, threads=2, dpi=50,
              profile_length=6, profile_band=2, log=lambda *a: None)
    metrics = json.loads((out/'metrics.json').read_text())
    assert set(metrics['fields']) == {'q2m', 't2m', 'u10m', 'v10m', 'ps', 'precip'} and metrics['verdict']
    for v in ('tiled@32', 'tiled@64', 'single@64'):
        assert len(metrics['fields']['q2m'][v]['member_grad_tail_ratio']) == 3
    for stem in ('profiles', 'gradient_tails', 'fields_q2m', 'fields_precip'):
        assert (out/f'{stem}.png').exists() and (out/f'{stem}.pdf').exists()
    assert 'Verdict' in (out/'report.md').read_text() and 'Max-slope' in (out/'report.md').read_text()
    assert 'slope_width_km' in metrics['fields']['q2m']['truth']
    lat, lon = (float(np.asarray(v).mean()) for v in (load_model(trained, resolve_checkpoint(trained), torch.device('cpu'))[0].static[k]
                                                   for k in ('lat', 'lon')))
    out2 = run(trained, 'best', split='test', members=1, output=tmp_path/'diag2', batch=4, threads=2, dpi=40, pdf=False,
               profile_length=6, profile_band=2, center_latlon=(lat, lon), log=lambda *a: None)
    assert (out2/'report.md').exists()


def test_gradient_finetune_starts_from_checkpoint(trained, tmp_path):
    from merraflow.v4_1 import objective, validate_config
    from merraflow.v4 import objective as objective_v4
    source = resolve_checkpoint(trained, 'latest')
    cfg = deepcopy(trained)
    cfg['train'].update(output=str(tmp_path/'ft'), epochs=1, time_limit_hours=None,
                        finetune=dict(init=str(source), gradient_weight=1., late_time_fraction=.5, late_time_shift=3.))
    validate_config(cfg)
    with pytest.raises(ValueError):
        validate_config(dict(deepcopy(cfg), train=dict(cfg['train'], finetune=dict(init=str(source), bogus=1))))
    # Defaults reproduce the v4 objective exactly.
    archive, model, conditioner, _ = load_model(trained, source, torch.device('cpu'))
    from merraflow.v4_1 import DatasetV41
    b = torch.utils.data.default_collate([DatasetV41(trained, 'train', 2, 0)[i] for i in range(2)])
    conditioner.prepare(b)
    a = objective(model, b, torch.Generator().manual_seed(3))
    ref = objective_v4(model, b, torch.Generator().manual_seed(3))
    assert torch.allclose(a, ref)
    assert objective(model, b, torch.Generator().manual_seed(3), gradient_weight=1.) > a
    train(cfg)
    saved = torch.load(tmp_path/'ft'/'last_v4_1.pt', map_location='cpu', weights_only=True)
    original = torch.load(source, map_location='cpu', weights_only=True)
    assert saved['initialization']['path'] == str(source.resolve())
    assert torch.equal(saved['flow_scale'], original['flow_scale'])
    assert saved['epoch'] == 0 and saved['history'][-1]['training_loss'] > 0
    bad = deepcopy(cfg)
    bad['train']['output'] = str(source.parent)
    with pytest.raises((ValueError, FileExistsError)):
        train(bad)


def test_sample_scores_are_proper_and_reward_edges():
    from merraflow.rollout_v4_1 import afcrps, variogram, multiscale_crps
    from merraflow.metrics import crps_ensemble
    g = torch.Generator().manual_seed(0)
    members = torch.randn(4, 2, 6, 16, 16, generator=g)
    truth = torch.randn(2, 6, 16, 16, generator=g)
    area = torch.ones(2, 16, 16)
    reference = crps_ensemble(members.numpy(), truth.numpy()).mean((-2, -1))
    np.testing.assert_allclose(afcrps(members, truth, area, 0.).numpy(), reference, rtol=1e-5)  # alpha=0: standard CRPS
    # A sharp front: truth-like members score ~0 on edges; smoothed members are penalized.
    x = torch.linspace(-1, 1, 32)
    front = (x[None, :] > 0).float().expand(32, 32)
    truth = front.expand(1, 6, 32, 32).clone()
    area = torch.ones(1, 32, 32)
    sharp = truth.expand(2, 1, 6, 32, 32).clone()
    smooth = torch.sigmoid(x[None, :]/.2).expand(32, 32).expand(2, 1, 6, 32, 32).clone()
    assert variogram(sharp, truth, area, [1, 2, 4]).abs().max() < 1e-6
    assert (variogram(smooth, truth, area, [1, 2, 4]) > .1).all()
    dry = torch.zeros(1, 6, 32, 32)  # flat truth + noisy members: bounded, not exploding
    noisy = 1e-3*torch.randn(2, 1, 6, 32, 32, generator=g)
    assert variogram(noisy, dry, area, [1, 2, 4]).max() < 2.5
    assert (afcrps(smooth, truth, area, .95) > afcrps(sharp, truth, area, .95)).all()
    assert multiscale_crps(smooth, truth, area, .95, [2, 4]).shape == (1, 6)
    from merraflow.rollout_v4_1 import bias
    assert bias(sharp, truth, area).abs().max() < 1e-6
    shifted = sharp+.5  # a biased ensemble is penalized in every field, rain twice (z and mm/h)
    assert (bias(shifted, truth, area) > .2).all() and bias(shifted, truth, area)[0, 1] > bias(shifted, truth, area)[0, 0]


def test_rollout_loss_backpropagates_through_last_steps(trained):
    from merraflow.rollout_v4_1 import rollout_loss, DEFAULTS, RolloutObjective
    from merraflow.v4_1 import DatasetV41
    archive, model, conditioner, _ = load_model(trained, resolve_checkpoint(trained), torch.device('cpu'))
    model.train()
    b = torch.utils.data.default_collate([DatasetV41(trained, 'train', 3, 0)[i] for i in range(3)])
    conditioner.prepare(b)
    settings = dict(DEFAULTS, patches=2, members=2, steps=3, grad_steps=1)
    value, parts = rollout_loss(model, b, settings, torch.Generator().manual_seed(1))
    assert torch.isfinite(value) and parts.shape == (4,)
    value.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    wrapper = RolloutObjective(model, settings, {})
    loss, flow, scores = wrapper(b, 0.)
    assert torch.equal(scores, torch.zeros(4)) and torch.isclose(loss, flow)


def test_rollout_finetune_trains_from_checkpoint(trained, tmp_path):
    source = resolve_checkpoint(trained, 'latest')
    cfg = deepcopy(trained)
    cfg['model']['activation_checkpointing'] = True
    cfg['patch']['samples_per_epoch'] = trained['patch']['samples_per_epoch']//2
    cfg['train'].update(output=str(tmp_path/'ro'), epochs=1, time_limit_hours=None,
                        finetune=dict(init=str(source), rollout=dict(patches=1, members=2, steps=3, grad_steps=1)))
    train(cfg)
    saved = torch.load(tmp_path/'ro'/'last_v4_1.pt', map_location='cpu', weights_only=True)
    row = saved['history'][-1]
    assert saved['initialization']['path'] == str(source.resolve())
    assert set(row['sample_scores']) == {'crps', 'multiscale_crps', 'variogram', 'bias'} and row['flow_loss'] > 0
    assert row['training_loss'] != row['flow_loss']
