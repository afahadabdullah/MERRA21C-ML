from copy import deepcopy
from pathlib import Path
import json
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader
import xarray as xr

from merraflow.config_v2 import load_config_v2, validate_config_v2
from merraflow.physics_v2 import encode_fields_v2, decode_fields_v2
from merraflow.dataset_v2 import ArchiveV2, PatchDatasetV2
from merraflow.rain_prior_v2 import training_noise_v2, inference_rain_noise_v2, rain_noise_kernel_v2
from merraflow.loss_v2 import loss_v2, rain_physical_v2, multiscale_v2
from merraflow.model_v2 import UNetV2
from merraflow.synchronized_v2 import integrate_global_v2, synchronized_frame_v2
from merraflow.generated_validation_v2 import rain_ensemble_scores_v2, rain_structure_score_v2
from merraflow.train_v2 import train_v2, check_checkpoint_v2
from merraflow.inference_v2 import predict_v2
from merraflow.evaluate_v2 import evaluate_v2
from test_pipeline_v2 import prepared_v2


def rain_config(cfg):
    cfg = deepcopy(cfg)
    cfg['representation'] = dict(precip='sqrt1p', rain_noise_sigma_pixels=2.)
    cfg['loss'].update(regression_rain_physical=1., flow_full_patch=True, flow_multiscale=.1)
    cfg['inference'].update(sampler='synchronized', noise_padding='independent_halo', tile_cache_mb=2)
    cfg['train']['generated_validation'] = dict(batches=1, members=2, steps=2, interval=1)
    return cfg


def test_square_root_roundtrip_and_polynomial_tail():
    rain = np.array([0, 1e-8, .01, 1, 100, 10000], dtype='float32')
    fields = np.stack([rain+280, rain, rain+90000, -rain, rain])[:, None]
    transformed = encode_fields_v2(fields, 1, 'sqrt1p')
    np.testing.assert_allclose(decode_fields_v2(transformed, 1, 'sqrt1p'), fields, rtol=2e-6)
    transformed[1] = 10
    assert np.all(decode_fields_v2(transformed, 1, 'sqrt1p')[1] == 120)
    # No rain cap: larger transformed values still produce larger rain amounts.
    transformed[1] = 100
    assert np.all(decode_fields_v2(transformed, 1, 'sqrt1p')[1] == 10200)


def test_square_root_archive_reuses_physical_fields_without_mutating_stats(prepared_v2):
    root = Path(prepared_v2['data']['prepared'])
    before = (root/'stats_v2.json').read_bytes()
    original = ArchiveV2(root)
    changed = ArchiveV2(root, 'sqrt1p')
    assert original.stats == changed.stats
    assert changed.rm[1] == 0 and changed.rs[1] == 1
    data = PatchDatasetV2(root, 'train', prepared_v2['patch'], 1, precipitation_representation='sqrt1p')
    batch = data[0]
    z = batch['rain_baseline']+batch['target'][1]
    torch.testing.assert_close(z*(z+2)*batch['rain_scale'], batch['rain_truth'], rtol=2e-5, atol=1e-5)
    assert batch['area_full'].shape == batch['target'].shape[-2:]
    assert (root/'stats_v2.json').read_bytes() == before


def test_training_and_inference_prior_have_matching_stationary_covariance():
    cfg = {'representation': {'rain_noise_sigma_pixels': 2.}}
    rng = torch.Generator().manual_seed(81)
    train = training_noise_v2((1, 5, 512, 512), torch.device('cpu'), rng, cfg)[0].numpy()
    original = np.random.default_rng(81).standard_normal((5, 512, 512), dtype=np.float32)
    infer = inference_rain_noise_v2(original.copy(), 82, cfg)
    np.testing.assert_array_equal(infer[[0, 2, 3, 4]], original[[0, 2, 3, 4]])
    kernel = rain_noise_kernel_v2(2.)
    expected_corr = np.sum(kernel[1:]*kernel[:-1])
    for array in (train, infer):
        assert abs(float(array[1].mean())) < .04
        assert .94 < float(array[1].std()) < 1.06
        correlation = np.corrcoef(array[1, :, 1:].ravel(), array[1, :, :-1].ravel())[0, 1]
        assert abs(correlation-expected_corr) < .02
        assert np.mean(np.diff(array[1], axis=1)**2) < .2
    # White-noise legacy path keeps the exact original draws.
    rng = torch.Generator().manual_seed(81)
    torch.testing.assert_close(training_noise_v2((1, 5, 4, 4), torch.device('cpu'), rng, {}),
                               torch.randn(1, 5, 4, 4, generator=torch.Generator().manual_seed(81)))


def test_spatial_score_detects_grain_with_identical_histogram():
    target = torch.zeros(1, 32, 32)
    target[:, 12:20] = 2
    coherent = target[None].repeat(4, 1, 1, 1)
    shuffled = target.flatten()[torch.randperm(target.numel(), generator=torch.Generator().manual_seed(9))].reshape_as(target)
    grain = shuffled[None].repeat(4, 1, 1, 1)
    assert torch.equal(coherent.flatten().sort().values, grain.flatten().sort().values)
    area = torch.ones_like(target)
    assert rain_structure_score_v2(coherent, target, area).item() == 0
    assert rain_structure_score_v2(grain, target, area).item() > .1


def test_crps_matches_pairwise_reference():
    members = torch.tensor([0., 1., 3.])[:, None, None, None].expand(3, 2, 4, 4)
    target = torch.ones(2, 4, 4)*2
    crps, mse = rain_ensemble_scores_v2(members, target, torch.ones_like(target))
    reference = (members-target).abs().mean(0)-.5*(members[:, None]-members[None]).abs().mean((0, 1))
    torch.testing.assert_close(crps, reference.mean((-2, -1)))
    torch.testing.assert_close(mse, torch.full_like(mse, (4/3-2)**2))


def test_global_heun_uses_updated_predictor_state():
    calls = []
    def rhs(x, t):
        calls.append((t, x.copy()))
        return -x
    initial = np.ones((5, 8, 8), dtype='float32')
    result = integrate_global_v2(initial, rhs, 2)
    np.testing.assert_array_equal(calls[1][1], initial*.5)
    np.testing.assert_allclose(result, .625**2)
    np.testing.assert_array_equal(initial, 1)


def test_synchronized_tiles_equal_global_pointwise_flow_with_halo(prepared_v2):
    class Mean(torch.nn.Module):
        def forward(self, x, time, condition, context):
            return torch.zeros_like(x)
    class Flow(torch.nn.Module):
        def forward(self, x, time, condition, context, mean):
            return -x
    cfg = rain_config(prepared_v2)
    archive = ArchiveV2(cfg['data']['prepared'], 'sqrt1p')
    halo = cfg['patch']['halo']
    h, w = archive.shape
    noise = np.random.default_rng(6).standard_normal((5, h+2*halo, w+2*halo), dtype=np.float32)
    entry = archive.index['entries'][0]
    actual, mean = synchronized_frame_v2(Mean(), Flow(), torch.ones(1, 5, 1, 1), archive,
                                         entry, cfg, torch.device('cpu'), noise)
    expected = integrate_global_v2(noise, lambda x, t: -x, cfg['inference']['steps'])
    np.testing.assert_allclose(actual, expected[:, halo:halo+h, halo:halo+w], atol=1e-6, rtol=2e-6)
    np.testing.assert_array_equal(mean, 0)


def test_full_patch_loss_supervises_halo_and_physical_loss_has_gradient(prepared_v2):
    cfg = rain_config(prepared_v2)
    batch = next(iter(DataLoader(PatchDatasetV2(cfg['data']['prepared'], 'train', cfg['patch'], 2,
                                                precipitation_representation='sqrt1p'), batch_size=2)))
    class Mean(torch.nn.Module):
        def forward(self, x, time, condition, context):
            return torch.zeros_like(x)
    class Flow(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.field = torch.nn.Parameter(torch.zeros_like(batch['target']))
        def forward(self, *args):
            return self.field
    flow = Flow()
    value, _ = loss_v2(flow, batch, cfg, 'flow', Mean(), torch.ones(1, 5, 1, 1))
    value.backward()
    assert flow.field.grad[..., 0, :].abs().sum() > 0
    prediction = (batch['target']+.5).detach().requires_grad_()
    physical = rain_physical_v2(prediction, batch, cfg['patch']['halo'])
    physical.backward()
    assert physical > 0 and prediction.grad[:, 1].abs().sum() > 0
    assert torch.isfinite(prediction.grad).all()
    zero = rain_physical_v2(batch['target'], batch, cfg['patch']['halo'])
    assert zero < 1e-10


def test_checkpoint_rejects_old_representation_and_prior(prepared_v2):
    cfg = rain_config(prepared_v2)
    archive = ArchiveV2(cfg['data']['prepared'], 'sqrt1p')
    payload = dict(version='v2', stage='flow', fingerprint=archive.index['fingerprint'], stats=archive.stats,
                   config=deepcopy(prepared_v2))
    with pytest.raises(ValueError, match='representation mismatch'):
        check_checkpoint_v2(payload, archive, cfg)
    payload['config'] = deepcopy(cfg)
    payload['config']['representation']['rain_noise_sigma_pixels'] = 0
    with pytest.raises(ValueError, match='noise prior mismatch'):
        check_checkpoint_v2(payload, archive, cfg)


def test_new_config_valid_and_invalid_settings_rejected():
    cfg = load_config_v2('configs/discover_rain_structure_v2.yaml')
    bad = deepcopy(cfg)
    bad['loss']['flow_full_patch'] = False
    with pytest.raises(ValueError, match='full-patch'):
        validate_config_v2(bad)
    bad = deepcopy(cfg)
    bad['inference']['noise_padding'] = 'replicate'
    with pytest.raises(ValueError, match='Correlated'):
        validate_config_v2(bad)


def test_retrain_resume_and_synchronized_prediction_end_to_end(prepared_v2, tmp_path, monkeypatch):
    cfg = rain_config(prepared_v2)
    cfg['train']['output'] = str(tmp_path/'rain_run_v2')
    cfg['inference']['output'] = str(tmp_path/'rain_predictions_v2')
    monkeypatch.setenv('FLOW_PLOT_INTERVAL', '1000')
    monkeypatch.setattr('merraflow.train_v2.flow_time_left_seconds', lambda: None)
    mean = train_v2(cfg, 'regression')
    flow = train_v2(cfg, 'flow', regression_checkpoint=mean)
    payload = torch.load(flow, weights_only=True)
    assert payload['selection_metric'] == 'rain_crps_plus_mean_rmse_plus_structure'
    history = [json.loads(row) for row in (flow.parent/'history_v2.jsonl').read_text().splitlines()]
    best = min(row['generated_validation']['selection_score'] for row in history)
    assert payload['best'] == best
    resumed = deepcopy(cfg)
    resumed['train']['output'] = str(tmp_path/'rain_resumed_v2')
    train_v2(resumed, 'flow', resume=flow.parent/'epoch_0001_v2.pt')
    uninterrupted = torch.load(flow.parent/'last_v2.pt', weights_only=True)
    continued = torch.load(Path(resumed['train']['output'])/'flow_v2/last_v2.pt', weights_only=True)
    for key in uninterrupted['model']:
        torch.testing.assert_close(uninterrupted['model'][key], continued['model'][key], rtol=0, atol=0)
    predict_v2(cfg, flow, limit=1)
    report = evaluate_v2(cfg)
    summary = json.loads((report/'summary_v2.json').read_text())
    assert summary['precipitation_representation'] == 'sqrt1p'
    assert summary['sampler'] == 'synchronized'
    assert summary['rain_noise_sigma_pixels'] == 2
    with xr.open_dataset(next(Path(cfg['inference']['output']).glob('*_v2.nc'))) as ds:
        assert np.isfinite(ds.precip).all() and float(ds.precip.min()) >= 0
