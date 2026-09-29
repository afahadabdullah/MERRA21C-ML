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
    for v in ('tiled@32', 'tiled@64', 'single@64', 'guided@64'):
        assert len(metrics['fields']['q2m'][v]['member_grad_tail_ratio']) == 3
        assert metrics['fields']['t2m'][v]['crps'] >= 0
    assert set(metrics['jumps']) == {'q2m', 't2m', 'u10m', 'v10m', 'ps', 'precip'}
    report_text = (out/'report.md').read_text()
    assert 'Wind-guided sharpening' in report_text and 'Jump check' in report_text
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
    assert torch.isfinite(value) and parts.shape == (5,)
    value.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    wrapper = RolloutObjective(model, settings, {})
    loss, flow, scores = wrapper(b, 0.)
    assert torch.equal(scores, torch.zeros(5)) and torch.isclose(loss, flow)


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
    assert set(row['sample_scores']) == {'crps', 'multiscale_crps', 'variogram', 'bias', 'increment_crps'} and row['flow_loss'] > 0
    assert row['training_loss'] != row['flow_loss']


def test_increment_crps_scales_and_front_selection():
    from merraflow.rollout_v4_1 import increment_crps, increment_scales, front_order
    x = torch.linspace(-1, 1, 32)
    truth = (x[None, :] > 0).float().expand(32, 32).expand(1, 6, 32, 32).clone()
    area = torch.ones(1, 32, 32)
    scales = increment_scales(truth, area, [1, 2, 4])
    assert scales.shape == (6, 3) and (scales > 0).all()
    sharp = truth.expand(4, 1, 6, 32, 32).clone()
    smooth = torch.sigmoid(x[None, :]/.2).expand(32, 32).expand(4, 1, 6, 32, 32).clone()
    flipped = 1-sharp  # right magnitude, wrong sign: the variogram cannot see this, signed increments do
    assert increment_crps(sharp, truth, area, [1, 2, 4], scales).abs().max() < 1e-6
    assert (increment_crps(smooth, truth, area, [1, 2, 4], scales) > .1).all()
    assert (increment_crps(flipped, truth, area, [1, 2, 4], scales) > increment_crps(smooth, truth, area, [1, 2, 4], scales)).all()
    coarse = torch.zeros(3, 6, 16, 16)
    coarse[2, 0, :, 8:] = 5.  # a t2m front in patch 2 only
    coarse[1, 1, :, 8:] = 50.  # a rain edge in patch 1 does not count
    assert front_order(coarse)[0].item() == 2


def test_rollout_round2_finetune_resumes_with_fixed_scales(trained, tmp_path):
    source = resolve_checkpoint(trained, 'latest')
    cfg = deepcopy(trained)
    cfg['model']['activation_checkpointing'] = True
    # Two epochs, but the time limit ends the first job after epoch 1 (as a 12 h job would).
    cfg['train'].update(output=str(tmp_path/'ro2'), epochs=2, time_limit_hours=.01,
                        finetune=dict(init=str(source), rollout=dict(
                            patches=2, front_patches=1, members=3, steps=4, grad_steps=2, alpha=1.,
                            variogram_weight=0., increment_weight=1., scale_batches=2)))
    train(cfg)
    first = torch.load(tmp_path/'ro2'/'last_v4_1.pt', map_location='cpu', weights_only=True)
    scales = first['rollout_scales']
    assert scales.shape == (6, 4) and (scales > 0).all() and not torch.allclose(scales, torch.ones_like(scales))
    assert first['history'][-1]['sample_scores']['increment_crps'] > 0
    assert first['epoch'] == 0
    cfg['train']['time_limit_hours'] = None
    train(cfg, resume=str(tmp_path/'ro2'/'last_v4_1.pt'))
    second = torch.load(tmp_path/'ro2'/'last_v4_1.pt', map_location='cpu', weights_only=True)
    assert second['epoch'] == 1 and torch.equal(second['rollout_scales'], scales)


def test_raw_and_ema_weights(trained, tmp_path):
    from merraflow.diag_front_v4_1 import run
    path = resolve_checkpoint(trained, 'latest')
    _, ema, _, saved = load_model(trained, path, torch.device('cpu'))
    _, raw, _, _ = load_model(trained, path, torch.device('cpu'), weights='raw')
    for name, value in raw.state_dict().items():
        assert torch.equal(value, saved['model'][name])
    assert any(not torch.equal(a, b) for a, b in zip(ema.state_dict().values(), raw.state_dict().values()))
    with pytest.raises(ValueError):
        load_model(trained, path, torch.device('cpu'), weights='swa')
    out = run(trained, 'latest', split='test', members=1, output=tmp_path/'raw', batch=4, threads=2, dpi=40,
              pdf=False, profile_length=6, profile_band=2, weights='raw', log=lambda *a: None)
    assert json.loads((out/'metrics.json').read_text())['weights'] == 'raw'


def test_guided_sharpening_transfers_wind_edges_only_where_correlated():
    from merraflow.diag_front_v4_1 import guided_sharpen, slope_width, profile, jump_check
    n = 96
    yy, xx = np.mgrid[:n, :n].astype(float)
    d = xx-n/2+.3*(yy-n/2)                                   # a slanted front
    wind_v = np.where(d > 0, 8., -4.)                        # sharp wind shift
    wind_u = 2.+.1*np.sin(yy/7)
    t2m_smooth = 290+2*np.tanh(d/12)                         # same front, smeared over ~12 px
    out = guided_sharpen(t2m_smooth, (wind_u, wind_v), radius=8, eps=1e-2, sigma=2.)
    normal = (.3/np.hypot(1, .3), 1/np.hypot(1, .3))
    s, before = profile(t2m_smooth, (n//2, n//2), normal, 30, 6)
    _, after = profile(out, (n//2, n//2), normal, 30, 6)
    assert slope_width(s, after, 1.) < .8*slope_width(s, before, 1.)   # clearly sharper edge
    assert abs(float(out.mean())-float(t2m_smooth.mean())) < .05       # mean kept
    # A field that does not co-vary with the wind (terrain-like smooth pattern) is left alone.
    unrelated = 285+np.sin(yy/9)*np.cos(xx/13)
    kept = guided_sharpen(unrelated, (wind_u, wind_v), radius=8, eps=1e-2, sigma=2.)
    far = np.abs(d) > 20                                     # away from the wind shift
    assert np.abs(kept-unrelated)[far].max() < .05
    # Jump check: the baseline carries part of the jump; the rest is the model's share.
    truth = np.zeros((6, n, n)); truth[0] = np.where(d > 0, 294., 290.)
    base = np.zeros((6, n, n)); base[0] = 290+2*(1+np.tanh(d/30))
    j = jump_check(truth, base, truth.copy(), (n//2, n//2), normal, 40, 6)['t2m']
    assert abs(j['truth_jump']-4) < .1 and 0 < j['residual_share'] < 1


def test_hann_power_windows():
    from merraflow.evaluate_v4_1 import make_window
    hann, h2, h3 = (make_window(32, k) for k in ('hann', 'hann2', 'hann3'))
    assert h2.min() >= 1e-4 and np.isclose(h2.max(), hann.max()**2, rtol=1e-6)
    # Centre-weighted: relative weight at a quarter of the tile falls with the power.
    assert h3[8, 16]/h3[16, 16] < h2[8, 16]/h2[16, 16] < hann[8, 16]/hann[16, 16]


def test_region_engine_matches_domain_sampler_and_hard_blending(trained):
    from merraflow.explore_inference_v4_1 import RegionEngine, DEFAULTS
    device = torch.device('cpu')
    archive, model, conditioner, _ = load_model(trained, resolve_checkpoint(trained), device)
    entry = archive.eligible('test')[0]
    h, w = archive.shape
    seed = member_seed(trained, entry, 0)
    engine = RegionEngine(model, conditioner, archive, entry, trained, device, (0, h, 0, w), margin=10**4, batch=3,
                          threads=2)
    spec = dict(DEFAULTS, id='baseline', steps=3)
    reference = DomainSampler(model, conditioner, archive, entry, trained, device, batch=3, threads=2).sample(seed, 3)
    np.testing.assert_allclose(engine.sample(seed, spec), reference, rtol=1e-5, atol=1e-5)
    # Hard blending: each covered pixel gets exactly one tile's velocity.
    for name in ('A', 'B', 'C', 'D'):
        tiles, norm = engine.weights(name, 'hard')
        assert torch.all((norm == 0) | (norm == 1)) and float(norm.max()) == 1
    for extra in (dict(blend='hard', grids='AB'), dict(blend='hard', grids='ACBD'), dict(blend='hann3'),
                  dict(langevin=.3), dict(restart=1), dict(sde=1., sde_range=(0., 1.)),
                  dict(restart=1, restart_t=.5, restart_blend='hard', restart_grids='ACBD'),
                  dict(vguide=1.), dict(hf=.3, hf_start=0.), dict(churn=.2), dict(temp=1.1)):
        out = engine.sample(seed, dict(spec, id='x', **extra), dict(vguide_fields=('t2m', 'q2m', 'ps'),
                                                                    vguide_radius=2, vguide_eps=1e-2, vguide_sigma=1.))
        assert out.shape == reference.shape and np.isfinite(out).all()
        assert np.abs(out-reference).max() > 0, extra
    # A small region samples only the tiles around it.
    small = RegionEngine(model, conditioner, archive, entry, trained, device, (0, h//3, 0, w//3), margin=0, batch=3,
                         threads=2)
    assert small.tile_count['A'] < engine.tile_count['A']
    assert small.sample(seed, spec).shape == (6, h//3, w//3)


def test_torch_guided_filter_and_spectral_fix():
    from merraflow.explore_inference_v4_1 import guided_torch, spectral_fix, climatology  # noqa: F401
    from merraflow.diag_front_v4_1 import slope_width, profile
    from merraflow.metrics import radial_psd
    n = 96
    yy, xx = np.mgrid[:n, :n].astype(float)
    d = xx-n/2+.3*(yy-n/2)
    v10m = torch.from_numpy(np.where(d > 0, 8., -4.))
    t2m = 290+2*np.tanh(d/12)
    out = guided_torch(torch.from_numpy(t2m), v10m, 8, 1e-2, 2.).numpy()
    normal = (.3/np.hypot(1, .3), 1/np.hypot(1, .3))
    s, before = profile(t2m, (n//2, n//2), normal, 30, 6)
    _, after = profile(out, (n//2, n//2), normal, 30, 6)
    assert slope_width(s, after, 1.) < .8*slope_width(s, before, 1.) and abs(out.mean()-t2m.mean()) < .05
    rng = np.random.default_rng(0)
    member = np.stack([gaussian_smooth(rng.standard_normal((n, n)), 1) for _ in range(6)]).astype('float32')
    member[1] = np.abs(member[1])
    member[5] = np.clip(member[5]*.01+.01, 0, 1)
    target = [radial_psd(np.sqrt(member[c]) if c == 1 else member[c])[1]*4 for c in range(6)]
    clim = dict(psd=np.array(target))
    fixed = spectral_fix(member, clim, 2., max_gain=1.5, cutoff_km=20.)
    for c in (0, 2, 3):
        assert abs(fixed[c].mean()-member[c].mean()) < 1e-4
        f0, p0 = radial_psd(member[c])
        _, p1 = radial_psd(fixed[c])
        fine = 2./np.maximum(f0, 1e-12) < 8
        assert (p1[fine] >= p0[fine]*.99).all() and p1[fine].sum() > 1.5*p0[fine].sum()


def gaussian_smooth(x, sigma):
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(x, sigma)


def test_inference_exploration_end_to_end(trained, tmp_path):
    from merraflow.explore_inference_v4_1 import run
    out = run(trained, 'best', split='test', methods=('baseline', 'hann2', 'shift_hard', 'shift4_hard', 'langevin',
                                                      'sde', 'restart', 'restart_shift', 'autoguide_hf', 'fk_steer',
                                                      'vguide', 'select_clim', 'select_sharp', 'prescreen',
                                                      'spectral', 'vpost'),
              sde_strengths=(1.,), fk_particles=3, fk_times=(.3, .6),
              members=2, pool=3, steps=3, region=20, margin=4, clim_count=2, vguide_strengths=(.5, 1.),
              vguide_radius=2, prescreen_steps=1, profile_length=6, profile_band=2, output=tmp_path/'explore',
              batch=4, threads=2, dpi=40, pdf=True, save_members=True, log=lambda *a: None)
    metrics = json.loads((out/'metrics.json').read_text())
    expected = {'baseline', 'hann2', 'shift_hard', 'shift4_hard', 'langevin', 'sde_1', 'restart', 'restart_shift',
                'fk_steer', 'vguide_0.5', 'vguide_1', 'select_clim', 'select_sharp', 'prescreen', 'spectral',
                'vpost_0.5', 'vpost_1'}
    assert expected <= set(metrics['results'])
    assert metrics['results']['baseline']['feasible'] and metrics['best'] in metrics['results']
    assert len(metrics['selected']['select_clim']) == 2 and 'prescreen_rank_correlation' in metrics['selection']
    assert len(metrics['selection']['fk_mean_ess']) == 2   # FK steering resampled twice
    for name in ('scorecard', 'spectra', 'profiles', 'selection', 'members_t2m', 'members_precip'):
        assert (out/f'{name}.png').exists() and (out/f'{name}.pdf').exists()
    text = (out/'report.md').read_text()
    assert 'Best feasible recipe' in text and 'Ranking' in text
    assert (out/'members.npz').exists()


def test_hf_autoguidance_is_fine_scale_and_orthogonal():
    from merraflow.explore_inference_v4_1 import RegionEngine
    g = torch.Generator().manual_seed(0)
    value = torch.randn(2, 6, 32, 32, generator=g)
    weak = value+torch.randn(2, 6, 32, 32, generator=g)+3.   # includes a large-scale offset
    extra = RegionEngine._hf_guidance(value, weak, 4.)
    assert torch.allclose((extra*value).sum((-2, -1)), torch.zeros(2, 6), atol=1e-3)   # APG: no parallel part
    assert float(extra.mean((-2, -1)).abs().max()) < .2                            # offset removed by high-pass


def test_combined_recipes(trained, tmp_path):
    from merraflow.explore_inference_v4_1 import combine_methods, OPTION_DEFAULTS, run
    o = dict(OPTION_DEFAULTS, fk_lambda=4.)
    specs = combine_methods(('autoguide_hf+fk_steer+spectral', 'shift4_hard+sde_1'), o)
    a = specs['autoguide_hf+fk_steer+spectral']
    assert a['kind'] == 'fk' and a['ag_hf'] and a['guide_weight'] == o['guide_weight'] and a['langevin'] > 0
    assert [t['post'] for t in a['then']] == ['spectral']
    b = specs['shift4_hard+sde_1']
    assert b['blend'] == 'hard' and b['grids'] == 'ACBD' and b['sde'] == 1. and not b['then']
    with pytest.raises(ValueError):
        combine_methods(('select_clim+spectral',), o)
    lines = []
    out = run(trained, 'best', split='test', methods=('baseline', 'spectral', 'fk_edge@4'),
              combine=('sde_1+spectral', 'fk_steer+spectral', 'autoguide_hf+spectral', 'fk_edge@4+spectral'),
              phase2=False,
              members=2, pool=2, steps=3, region=20, margin=4, clim_count=2, fk_particles=2, fk_times=(.5,),
              profile_length=6, profile_band=2, output=tmp_path/'combo', batch=4, threads=2, dpi=40, pdf=False,
              log=lines.append)
    metrics = json.loads((out/'metrics.json').read_text())
    assert {'baseline', 'spectral', 'sde_1+spectral', 'fk_steer+spectral', 'fk_edge@4',
            'fk_edge@4+spectral'} <= set(metrics['results'])
    assert metrics['methods']['fk_edge@4']['fk_lambda'] == 4. and metrics['methods']['fk_edge@4']['fk_reward'] == 'edge'
    assert any('fk_edge@4+spectral: reuses the members of fk_edge@4' in str(x) for x in lines)
    assert (out/'edge_zoom.png').exists() and (out/'edge_profiles.png').exists()
    assert 'autoguide_hf+spectral' not in metrics['results']   # no earlier checkpoint in the test run
    assert metrics['phase2_winner'] is None


def test_edge_step_scores_the_sharp_line_not_the_ramp():
    from merraflow.explore_inference_v4_1 import truth_edges, edge_scores
    n = 120
    yy, xx = np.mgrid[:n, :n].astype(float)
    d = xx-70+.2*(yy-n/2)
    ramp = .3*np.clip((d+60)/60, 0, 1)                     # broad ramp west of the edge
    truth = ramp+np.where(d > 0, 1., 0.)                    # plus a sharp step (the dark line)
    edges = truth_edges(truth, count=10)
    assert len(edges) >= 5
    assert all(abs(x-(70-.2*(y-n/2))) <= 2 for (y, x), _ in edges)   # on the step, not in the ramp
    same, _ = edge_scores(np.stack([truth, truth]), truth, edges)
    shifted, _ = edge_scores(np.stack([np.roll(truth, 5, axis=1)]), truth, edges)
    smooth, prof = edge_scores(np.stack([ramp+.5*(1+np.tanh(d/8))]), truth, edges)
    assert abs(same-1) < 1e-6 and shifted > .95 and smooth < .5
    assert len(prof['truth']) == len(prof['members'])


def test_fk_weights_are_scale_free():
    from merraflow.explore_inference_v4_1 import RegionEngine
    tiny = np.array([-.3001, -.3004, -.2998, -.3010])          # region-mean rewards differ by ~1e-3
    for lam, low in ((1., .1), (3., .001)):
        w = np.exp(RegionEngine._fk_logw(tiny-(-.3), lam))
        w /= w.sum()
        assert w.argmax() == 2 and w.min() < 1-low*0 and 1/np.sum(w**2) < 3.9   # real selection, not uniform
    assert np.allclose(RegionEngine._fk_logw(np.zeros(4), 3.), 0)             # identical particles: uniform
    big = np.exp(RegionEngine._fk_logw(100*(tiny+.3), 2.))
    small = np.exp(RegionEngine._fk_logw(tiny+.3, 2.))
    assert np.allclose(big/big.sum(), small/small.sum())                       # independent of reward scale
