from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from contextlib import nullcontext
import json
import numpy as np

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from merraflow.config_v2 import load_config_v2, validate_config_v2
from merraflow.dataset_v2 import PatchDatasetV2, rain_edge_scores_v2, proposal_cache_paths
from merraflow.model_v2 import UNetV2, integrate_v2, integrate_differentiable_v2
from merraflow.rain_rollout_v2 import (RainFlowObjectiveV2, fair_squared_error_v2,
    rain_sample_scores_v2, rollout_strength_v2, rain_rollout_loss_v2)
from merraflow.train_v2 import train_v2, check_checkpoint_v2
from merraflow.inference_v2 import predict_v2
from test_pipeline_v2 import prepared_v2
from test_rain_structure_v2 import rain_config


def rollout_config(cfg):
    cfg = rain_config(cfg)
    cfg['model']['activation_checkpointing'] = True
    settings = load_config_v2('configs/discover_rain_rollout_v2.yaml')['train']['rain_rollout']
    settings.update(members=3, steps=2, interval=1, warmup_epochs=0, ramp_epochs=1,
                    lags=[1, 4], pool_scales=[1, 4])
    cfg['train']['rain_rollout'] = settings
    return cfg


def test_differentiable_heun_matches_inference_and_analytic_gradient():
    class Linear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = torch.nn.Parameter(torch.tensor(.3, dtype=torch.float64))
        def forward(self, x, *args):
            return self.a*x
    model = Linear()
    x = torch.ones(1, 5, 4, 4)
    y = integrate_differentiable_v2(model, x, None, None, None, 2)
    expected = (1+.3/2+.3**2/8)**2
    assert float(y.mean().detach()) == pytest.approx(expected)
    y.mean().backward()
    assert float(model.a.grad) == pytest.approx(2*(1+.3/2+.3**2/8)*(.5+.3/4), rel=1e-6)
    torch.testing.assert_close(y.detach(), integrate_v2(model, x, None, None, None, 2))


def test_finite_ensemble_corrections_and_spatial_sensitivity():
    # Enumerate iid two-member draws from {0, 2}; fair score expectation is
    # (E[X]-1)^2=0, whereas the raw squared sample-mean error is 0.5.
    draws = torch.tensor([[0., 0., 2., 2.], [0., 2., 0., 2.]])
    torch.testing.assert_close(fair_squared_error_v2(draws, torch.ones(4)).mean(), torch.tensor(0.))
    settings = rollout_config(load_config_v2('configs/discover_v2.yaml'))['train']['rain_rollout']
    target = torch.zeros(1, 32, 32)
    target[:, 10:20, 3:29] = 5
    exact = target[None].repeat(3, 1, 1, 1)
    shuffled = target.flatten()[torch.randperm(target.numel(), generator=torch.Generator().manual_seed(19))].reshape_as(target)
    noise = shuffled[None].repeat(3, 1, 1, 1)
    area = torch.ones_like(target)
    assert rain_sample_scores_v2(exact, target, area, settings).abs().max() < 1e-6
    scores = rain_sample_scores_v2(noise, target, area, settings)
    assert scores[0, 1] > .01  # Same histogram, wrong organization.
    assert scores[0, 2] > .01
    drizzle = (exact+.2).requires_grad_()
    coverage = rain_sample_scores_v2(drizzle, target, area, settings)[0, 2]
    coverage.backward()
    assert coverage > 0 and drizzle.grad[:, :, :5].abs().sum() > 0


def test_crps_fair_pairwise_formula_and_partial_block_area():
    settings = rollout_config(load_config_v2('configs/discover_v2.yaml'))['train']['rain_rollout']
    settings['pool_scales'] = [4]
    members = torch.tensor([0., 1., 3.])[:, None, None, None].expand(3, 1, 5, 7)
    target = torch.ones(1, 5, 7)*2
    area = torch.arange(1., 36.).reshape_as(target)
    scores = rain_sample_scores_v2(members, target, area, settings)
    pairwise = (members[:, None]-members[None]).abs().sum((0, 1))/(2*3*2)
    expected = ((members-target).abs().mean(0)-pairwise)/settings['rate_scale_mm_h']
    torch.testing.assert_close(scores[:, 0], (expected*area).sum((-2, -1))/area.sum((-2, -1)))
    # Vary only the partial edge block: compare the independently computed
    # area-weighted block errors, including all 35 cells.
    members = members.clone()
    members[..., 4, :] += 4
    expected_mse = 0.
    for y in (0, 4):
        for x in (0, 4):
            weight = area[:, y:y+4, x:x+4]
            error = (members.mean(0)-target)[:, y:y+4, x:x+4]/10
            expected_mse += (error*weight).sum().square()/weight.sum()
    expected_mse /= area.sum()
    scores = rain_sample_scores_v2(members, target, area, settings)
    torch.testing.assert_close(scores[0, 3], expected_mse)


def test_rollout_backpropagates_without_training_regression(prepared_v2):
    cfg = rollout_config(prepared_v2)
    data = PatchDatasetV2(cfg['data']['prepared'], 'train', cfg['patch'], 2, precipitation_representation='sqrt1p')
    batch = next(iter(DataLoader(data, batch_size=2)))
    nc = data.archive.index['condition_channels']
    mean = UNetV2(nc, **cfg['model']).eval().requires_grad_(False)
    flow = UNetV2(nc, **cfg['model'], mean_condition=True).train()
    loss, components = rain_rollout_loss_v2(flow, batch, mean, torch.ones(1, 5, 1, 1), cfg)
    loss.backward()
    assert torch.isfinite(components).all()
    assert flow.output[-1].weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in mean.parameters())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in flow.parameters())


def test_rollout_schedule_configs_and_invalid_settings():
    for name in ('rollout', 'control', 'edges', 'context'):
        cfg = load_config_v2(f'configs/discover_rain_{name}_v2.yaml')
        assert cfg['train']['batch_size']*cfg['train']['accumulate']*2 == 16
        assert cfg['train']['val_batches']*cfg['train']['batch_size']*2 == 256
        assert cfg['train']['generated_validation']['batches']*cfg['train']['batch_size']*2 == 64
    settings = cfg['train']['rain_rollout']
    assert rollout_strength_v2(settings, 0, 0) == 0
    assert rollout_strength_v2(settings, 5, 0) == .2
    assert rollout_strength_v2(settings, 10, 1) == 0
    for field, value in [('members', 1), ('interval', 0), ('weight', float('nan')),
                         ('pool_scales', []), ('thresholds_mm_h', [-1])]:
        bad = deepcopy(cfg)
        bad['train']['rain_rollout'][field] = value
        with pytest.raises(ValueError, match='rain_rollout'):
            validate_config_v2(bad)


def test_initialize_train_resume_and_predict(prepared_v2, tmp_path, monkeypatch):
    monkeypatch.setenv('FLOW_PLOT_INTERVAL', '1000')
    monkeypatch.setattr('merraflow.train_v2.flow_time_left_seconds', lambda: None)
    old = rain_config(prepared_v2)
    old['train']['output'] = str(tmp_path/'old_v2')
    regression = train_v2(old, 'regression')
    initial = train_v2(old, 'flow', regression_checkpoint=regression)
    before = initial.read_bytes()
    cfg = rollout_config(old)
    cfg['patch']['structure_fraction'] = .25
    cfg['train']['output'] = str(tmp_path/'rollout_v2')
    cfg['inference']['output'] = str(tmp_path/'predictions_v2')
    trained = train_v2(cfg, 'flow', initialize_flow=initial)
    payload = torch.load(trained.parent/'last_v2.pt', weights_only=True)
    source = torch.load(initial, weights_only=True)
    assert payload['initialization']['epoch'] == source['epoch']+1
    assert payload['step'] == 4  # New two-epoch schedule, not old+new steps.
    torch.testing.assert_close(payload['flow_scale'], source['flow_scale'])
    for key in source['regression_ema']:
        torch.testing.assert_close(payload['regression_ema'][key], source['regression_ema'][key], rtol=0, atol=0)
    rows = [json.loads(line) for line in (trained.parent/'history_v2.jsonl').read_text().splitlines()]
    assert all(row['rain_rollout_train']['patches'] > 0 for row in rows)
    assert all('coarse_rmse_mm_h' in row['generated_validation'] for row in rows)
    assert (trained.parent/'best_skill_v2.pt').exists() == any(row['generated_validation']['beats_coarse'] for row in rows)
    resumed = deepcopy(cfg)
    resumed['train']['output'] = str(tmp_path/'resume_v2')
    train_v2(resumed, 'flow', resume=trained.parent/'epoch_0001_v2.pt')
    continued = torch.load(Path(resumed['train']['output'])/'flow_v2/last_v2.pt', weights_only=True)
    for key in payload['model']:
        torch.testing.assert_close(payload['model'][key], continued['model'][key], rtol=0, atol=0)
    assert initial.read_bytes() == before
    predict_v2(cfg, trained, limit=1)
    assert list(Path(cfg['inference']['output']).glob('*_v2.nc'))
    with pytest.raises(ValueError, match='cannot be combined'):
        train_v2(cfg, 'flow', resume=trained, initialize_flow=initial)


def test_edge_proposals_cache_and_unbiased_weights(prepared_v2):
    cfg = deepcopy(prepared_v2)
    cfg['patch']['structure_fraction'] = .3
    data = PatchDatasetV2(cfg['data']['prepared'], 'train', cfg['patch'], 2)
    entry = data.entries[0]
    q = data.proposal(entry)
    uniform_part = 1-cfg['patch']['detail_fraction']-.3
    assert np.all(q >= uniform_part/len(q))
    signal = np.arange(len(q), dtype=float)**2
    np.testing.assert_allclose(np.sum(q*signal/(len(q)*q)), signal.mean())
    assert not np.allclose(q, np.full_like(q, 1/len(q)))
    validation = PatchDatasetV2(cfg['data']['prepared'], 'val', cfg['patch'], 1)
    np.testing.assert_allclose(validation.proposal(validation.entries[0]), 1/len(q))
    scores, meta = proposal_cache_paths(cfg['data']['prepared'], cfg['patch'], 'structure')
    scores.parent.mkdir(exist_ok=True)
    np.save(scores, rain_edge_scores_v2(np.asarray(data.archive.array(entry, 'truth')[1]), data.yy, data.xx, cfg['patch']['size'])[None].astype('float32'))
    meta.write_text(json.dumps(dict(size=cfg['patch']['size'], sampling_stride=cfg['patch']['sampling_stride'],
                                   candidates=len(q), shape=list(data.archive.shape), ids=[entry['id']])))
    try:
        cached = PatchDatasetV2(cfg['data']['prepared'], 'train', cfg['patch'], 2)
        assert cached.cached_edges is not None
        np.testing.assert_allclose(cached.proposal(entry), q, rtol=1e-6)
    finally:
        scores.unlink()
        meta.unlink()
    # A wet/dry boundary gets more proposal mass than flat dry or wet interiors.
    rain = np.zeros((24, 24), dtype='float32')
    rain[:, 12:] = 5
    values = rain_edge_scores_v2(rain, np.array([0, 0, 0]), np.array([0, 8, 16]), 8)
    assert values[1] > max(values[0], values[2])
    bad = deepcopy(cfg)
    bad['patch']['structure_fraction'] = 1-cfg['patch']['detail_fraction']
    with pytest.raises(ValueError, match='uniform coverage'):
        validate_config_v2(bad)


def test_sampling_changes_allowed_only_for_explicit_initialization(prepared_v2):
    from merraflow.dataset_v2 import ArchiveV2
    cfg = rain_config(prepared_v2)
    archive = ArchiveV2(cfg['data']['prepared'], 'sqrt1p')
    ckpt = dict(version='v2', stage='flow', config=deepcopy(cfg),
                fingerprint=archive.index['fingerprint'], stats=archive.stats)
    cfg['patch']['structure_fraction'] = .3
    check_checkpoint_v2(ckpt, archive, cfg, 'flow', allow_sampling_change=True)
    with pytest.raises(ValueError, match='model/patch mismatch'):
        check_checkpoint_v2(ckpt, archive, cfg, 'flow')
    cfg['patch']['size'] *= 2
    with pytest.raises(ValueError, match='model/patch mismatch'):
        check_checkpoint_v2(ckpt, archive, cfg, 'flow', allow_sampling_change=True)


def test_generated_validation_compares_same_patches_to_coarse(prepared_v2, monkeypatch):
    from merraflow.generated_validation_v2 import generated_validation_v2
    cfg = rollout_config(prepared_v2)
    data = PatchDatasetV2(cfg['data']['prepared'], 'val', cfg['patch'], 2, precipitation_representation='sqrt1p')
    batch = next(iter(DataLoader(data, batch_size=2)))
    class Zero(torch.nn.Module):
        def forward(self, x, *args):
            return torch.zeros_like(x)
    monkeypatch.setattr('merraflow.generated_validation_v2.integrate_v2', lambda *args: batch['target'])
    result = generated_validation_v2(Zero(), Zero(), torch.ones(1, 5, 1, 1), [batch], cfg, torch.device('cpu'), 0, 1)
    assert result['beats_coarse'] and result['rmse_skill_vs_coarse'] > .99
    assert result['ensemble_mean_rmse_mm_h'] < 1e-6
    monkeypatch.setattr('merraflow.generated_validation_v2.integrate_v2', lambda *args: torch.zeros_like(batch['target']))
    result = generated_validation_v2(Zero(), Zero(), torch.ones(1, 5, 1, 1), [batch], cfg, torch.device('cpu'), 0, 1)
    assert result['ensemble_mean_rmse_mm_h'] == pytest.approx(result['coarse_rmse_mm_h'], rel=1e-5)
    assert abs(result['rmse_skill_vs_coarse']) < 1e-5


def ddp_rollout_worker(rank, store, cfg, batch_path, destination):
    torch.set_num_threads(1)
    batch = torch.load(batch_path, weights_only=True)
    dist.init_process_group('gloo', init_method=f'file://{store}', rank=rank,
                            world_size=2, timeout=timedelta(seconds=45))
    try:
        torch.manual_seed(31)
        nc = batch['condition'].shape[1]
        mean = UNetV2(nc, **cfg['model']).eval().requires_grad_(False)
        flow = UNetV2(nc, **cfg['model'], mean_condition=True)
        objective = DDP(RainFlowObjectiveV2(flow, cfg), find_unused_parameters=True)
        optimizer = torch.optim.AdamW(flow.parameters(), lr=1e-3)
        torch.manual_seed(71+rank)
        for i in range(4):
            with (objective.no_sync() if i % 2 == 0 else nullcontext()):
                loss, _, _ = objective(batch, mean, torch.ones(1, 5, 1, 1), float(i % 2 == 0))
                (loss/2).backward()
            if i % 2:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        torch.save(flow.state_dict(), Path(destination)/f'rank{rank}.pt')
    finally:
        dist.destroy_process_group()


def test_two_rank_rollout_with_accumulation_and_unused_parameters(prepared_v2, tmp_path):
    cfg = rollout_config(prepared_v2)
    data = PatchDatasetV2(cfg['data']['prepared'], 'train', cfg['patch'], 1, precipitation_representation='sqrt1p')
    batch = next(iter(DataLoader(data, batch_size=1)))
    torch.save(batch, tmp_path/'batch.pt')
    mp.spawn(ddp_rollout_worker, args=(str(tmp_path/'store'), cfg, str(tmp_path/'batch.pt'), str(tmp_path)), nprocs=2, join=True)
    left, right = [torch.load(tmp_path/f'rank{i}.pt', weights_only=True) for i in (0, 1)]
    for key in left:
        torch.testing.assert_close(left[key], right[key], atol=0, rtol=0)
