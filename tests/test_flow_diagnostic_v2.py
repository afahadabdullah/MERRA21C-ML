from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest
import torch
import xarray as xr

from merraflow.config_v2 import load_config_v2
from merraflow.dataset_v2 import ArchiveV2
from merraflow.flow_diagnostic_v2 import SamplerTrace, run_flow_diagnostic, structure_scores
from merraflow.inference_v2 import sample_frame_v2
from merraflow.model_v2 import UNetV2, integrate_v2
from merraflow.prepare_v2 import prepare
from merraflow.synthetic_v2 import make_synthetic_v2


class ZeroMean(torch.nn.Module):
    def forward(self, x, time, condition, context):
        return torch.zeros_like(x)


class DecayFlow(torch.nn.Module):
    def forward(self, x, time, condition, context, mean):
        return -x


@pytest.fixture(scope='module')
def prepared(tmp_path_factory):
    torch.set_num_threads(1)
    root = tmp_path_factory.mktemp('flow_diagnostic')
    cfg = load_config_v2(make_synthetic_v2(root, load_config_v2('configs/discover_v2.yaml')))
    prepare(cfg)
    archive = ArchiveV2(cfg['data']['prepared'])
    nc = archive.index['condition_channels']
    checkpoint = root/'best_v2-Copy1.pt'
    torch.save({'version': 'v2', 'stage': 'flow', 'epoch': 7, 'config': cfg,
                'fingerprint': archive.index['fingerprint'], 'stats': archive.stats,
                'ema': UNetV2(nc, **cfg['model'], mean_condition=True).state_dict(),
                'regression_ema': UNetV2(nc, **cfg['model']).state_dict(),
                'flow_scale': torch.full((1, 5, 1, 1), .2), 'regression_sha256': 'synthetic-regression'}, checkpoint)
    return cfg, archive, checkpoint


def test_observer_preserves_heun_and_records_convergent_trajectory():
    noise = torch.ones(1, 5, 4, 4)
    errors = []
    for steps in (2, 4, 8):
        states = []
        plain = integrate_v2(DecayFlow(), noise, None, None, None, steps)
        traced = integrate_v2(DecayFlow(), noise, None, None, None, steps,
                              observer=lambda step, time, x: states.append((step, time, x.clone())))
        torch.testing.assert_close(traced, plain, rtol=0, atol=0)
        assert states[0][0:2] == (0, 0.) and states[-1][0:2] == (steps, 1.)
        errors.append(abs(float(traced.mean())-np.exp(-1)))
    assert errors[2] < errors[1] < errors[0]


@pytest.mark.parametrize('blend', ['weighted', 'owner'])
def test_trace_preserves_production_samples_and_decode(prepared, blend):
    cfg, archive, _ = prepared
    cfg = deepcopy(cfg)
    cfg['inference']['blend'] = blend
    entry = archive.index['entries'][-1]
    scale = torch.full((1, 5, 1, 1), .3)
    plain = sample_frame_v2(ZeroMean(), DecayFlow(), scale, archive, entry, cfg, torch.device('cpu'), 317)
    trace = SamplerTrace(archive, entry, cfg, scale)
    actual = sample_frame_v2(ZeroMean(), DecayFlow(), scale, archive, entry, cfg, torch.device('cpu'), 317, diagnostics=trace)
    for a, b in zip(plain[:2], actual[:2]):
        np.testing.assert_array_equal(a, b)
    log_rain = trace.fields['flow_log_precip_before_clipping']
    np.testing.assert_allclose(np.expm1(np.maximum(log_rain, 0))*archive.stats['precip_log_scale'], actual[0][1], rtol=2e-6, atol=1e-6)
    assert (trace.count > 1).any() and trace.trajectory and len(trace.tiles) > len(trace.traced)
    np.testing.assert_allclose(trace.fields['flow_log_correction'],
                               trace.fields['flow_normalized_correction']*archive.rs[1, 0, 0])


def test_trace_retains_predecode_fields_on_overflow(prepared):
    class Extreme(torch.nn.Module):
        def forward(self, x, time, condition, context, mean):
            return torch.ones_like(x)*1000
    cfg, archive, _ = prepared
    entry = archive.index['entries'][-1]
    scale = torch.ones(1, 5, 1, 1)
    trace = SamplerTrace(archive, entry, cfg, scale)
    with pytest.raises(FloatingPointError):
        sample_frame_v2(ZeroMean(), Extreme(), scale, archive, entry, cfg, torch.device('cpu'), 0, diagnostics=trace)
    assert trace.fields['flow_log_precip_before_clipping'].max() > 30


def test_structure_distinguishes_isolated_pixels_from_connected_band():
    isolated, band = np.zeros((16, 16)), np.zeros((16, 16))
    isolated[::4, ::4] = 2
    band[8, :] = 2
    area, patch = np.ones((16, 16)), {'size': 8, 'stride': 6}
    a, b = [structure_scores(v, area, patch)['wet_objects']['1.0'] for v in (isolated, band)]
    assert a['wet_area_fraction'] == b['wet_area_fraction']
    assert a['objects'] == 16 and b['objects'] == 1
    assert a['wet_area_in_objects_le_4_pixels'] == 1 and b['wet_area_in_objects_le_4_pixels'] == 0


def test_end_to_end_uses_fixed_checkpoint_seeds_and_saves_raw_fields(prepared, tmp_path):
    cfg, archive, checkpoint = prepared
    entry = archive.index['entries'][-1]
    before = checkpoint.read_bytes()
    out = run_flow_diagnostic(cfg, checkpoint, tmp_path/'diagnostic_v2', [entry['id']], steps=(1, 2), members=2)
    manifest = json.loads((out/'manifest_v2.json').read_text())
    assert manifest['checkpoint_epoch'] == 8 and manifest['checkpoint'].endswith('best_v2-Copy1.pt')
    case = out/entry['id']
    summary = json.loads((case/'summary_v2.json').read_text())
    assert summary['normalization_checks']['stored_residual_matches_transform']
    # The synthetic zero-velocity network leaves noise unchanged at both step counts.
    assert all(r['rain_change_rmse_mm_h'] == 0 for r in summary['same_seed_convergence']['1_to_2'])
    for steps in (1, 2):
        folder = case/f'steps_{steps:03d}_v2'
        for m, seed in enumerate(summary['seeds']):
            with xr.open_dataset(folder/'predictions_v2'/f'{entry["id"]}_m{m:03d}_v2.nc') as ds:
                assert ds.attrs['seed'] == seed and ds.attrs['checkpoint_sha256'] == manifest['checkpoint_sha256']
                assert ds.attrs['ode_steps'] == steps and ds.precip.attrs['units'] == 'mm h-1'
                assert 'flow_endpoint_reconstructed_after_blending' in ds and 'tile_prediction_log_std' in ds
                np.testing.assert_allclose(np.expm1(np.maximum(ds.flow_log_precip_before_clipping, 0)), ds.precip, rtol=2e-6, atol=1e-6)
        assert len(list(folder.glob('*_internal_v2.png'))) == 2
        assert (folder/'audit_v2.json').exists()
    assert checkpoint.read_bytes() == before
    with pytest.raises(FileExistsError):
        run_flow_diagnostic(cfg, checkpoint, out, [entry['id']], steps=(1, 2), members=2, plots=False)


def test_array_wrapper_pins_checkpoint_and_forwards_all_step_counts():
    env = dict(os.environ, PROJECT_DIR=str(Path(__file__).resolve().parents[1]), SLURM_JOB_ID='42',
               SLURM_ARRAY_TASK_ID='4', SLURM_MEM_PER_CPU='2', SLURM_MEM_PER_NODE='3', SLURM_MEM_PER_GPU='48G',
               SLURM_CPUS_PER_GPU='4', CHECKPOINT='runs/flow_v2/best_v2-Copy1.pt', OUTPUT='runs/diag_v2')
    result = subprocess.run(['bash', '-c', r'''
source() { :; }
conda() { :; }
srun() {
  test -z "${SLURM_MEM_PER_CPU:-}" && test -z "${SLURM_MEM_PER_NODE:-}" && test -z "${SLURM_CPUS_PER_GPU:-}"
  test "$SLURM_MEM_PER_GPU" = 48G
  printf '%s\n' "$@"
}
. scripts/slurm_diagnose_flow_v2.sh
'''], cwd=env['PROJECT_DIR'], env=env, text=True, capture_output=True, check=True)
    args = result.stdout.splitlines()
    assert args[args.index('--checkpoint')+1].endswith('best_v2-Copy1.pt')
    assert args[args.index('--steps')+1:args.index('--case-index')] == ['24', '48', '96']
    assert args[-2:] == ['--case-index', '4']
