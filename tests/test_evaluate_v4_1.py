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
