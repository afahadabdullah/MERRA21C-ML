from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import socket
import numpy as np
import pytest
import torch
from merraflow.config_v2 import load_config_v2
from merraflow.config_v3_precip import load_config, validate_config
from merraflow.synthetic_v2 import make_synthetic_v2
from merraflow.prepare_v2 import prepare
from merraflow.dataset_v3_precip import PrecipArchive, PrecipDataset, encode_rain, decode_rain
from merraflow.model_v3_precip import make_regression, PrecipEDM, objective, sample_edm, weighted_loss, heun_edm, sigma_schedule
from merraflow.model_v2 import regression_v2
from merraflow.train_v3_precip import train
from merraflow.inference_v3_precip import predict
from merraflow.evaluate_v3_precip import evaluate


@pytest.fixture(scope='module')
def config(tmp_path_factory):
    torch.set_num_threads(1)
    root = tmp_path_factory.mktemp('precip')
    source = load_config_v2(make_synthetic_v2(root, load_config_v2('configs/discover_v2.yaml')))
    prepare(source)
    cfg = load_config('configs/discover_v3_precip.yaml')
    cfg['data'].update(prepared=source['data']['prepared'], history_hours=[-1, 0], target_kind='midpoint_rate')
    cfg['data'].pop('hourly_targets')
    cfg['patch'] = dict(source['patch'], proposal='coarse', loss_on_halo=True)
    cfg['model'] = source['model']
    cfg['train'].update(device='cpu', workers=0, batch_size=2, accumulate=3, precision='fp32',
                        regression_epochs=2, diffusion_epochs=1, val_batches=1, calibration_batches=1,
                        validation_members=2, validation_steps=2, ema_decay=.5, warmup_steps=2,
                        validation_interval=1, validation_plot_interval=5, time_limit_hours=None,
                        output=str(root/'run_v3_precip'))
    cfg['inference'].update(members=2, steps=2, output=str(root/'predictions_v3_precip'), tile_batch=3)
    return validate_config(cfg)


def test_single_target_and_causal_history(config):
    data = PrecipDataset(config, 'train', 2)
    batch = data[0]
    assert batch['target'].shape[0] == 1
    assert batch['condition'].shape[0] == data.archive.channels
    a = data.archive
    for split in ('train', 'val', 'test'):
        entries = a.eligible(split)
        assert len(entries) == 1  # First hour of each split cannot borrow preceding split.
        history = a.history(entries[0])
        assert all(e['split'] == split for e in history)
        assert np.datetime64(history[1]['time'])-np.datetime64(history[0]['time']) == np.timedelta64(1, 'h')
    # Supervision is reconstructed exclusively from physical precipitation.
    expected = a.encode_rain(batch['truth'].numpy())-batch['baseline'].numpy()
    np.testing.assert_array_equal(batch['target'], expected)
    assert float(batch['importance']) == 1.  # Deliberate rain-emphasized training objective.
    assert batch['area_full'].shape[-1] == batch['target'].shape[-1]
    a.by_time.pop(next(iter(a.by_time)))
    with pytest.raises(ValueError, match='No train hours'):
        a.eligible('train')


def test_transform_and_zero_rain():
    rain = np.array([0., 1e-8, .1, 1., 30., 300.], dtype='float32')
    np.testing.assert_allclose(decode_rain(encode_rain(rain, 2.), 2.), rain, rtol=2e-6)
    assert decode_rain(np.array([-10., 0.]), 1.).tolist() == [0., 0.]
    with pytest.raises(FloatingPointError):
        decode_rain(np.array([np.nan]), 1.)


def test_dry_margin_encoding_blocks_drizzle_leak():
    rain = np.array([0., .01, .02, .5, 20.], dtype='float32')
    z = encode_rain(rain, 1., dry_threshold=.02, dry_offset=.25)
    assert z[:2].tolist() == [-.25, -.25] and np.all(z[2:] > 0)
    np.testing.assert_allclose(decode_rain(z, 1.), [0, 0, .02, .5, 20.], rtol=2e-6)
    # Positive noise smaller than the margin keeps a dry pixel dry.
    assert decode_rain(z[:2]+.2, 1.).tolist() == [0., 0.]
    # A decode threshold removes weak positive values.
    assert decode_rain(np.array([.01, .5]), 1., wet_threshold=.05)[0] == 0.


def test_one_channel_models_and_finite_gradients(config):
    from torch.utils.data import DataLoader
    data = PrecipDataset(config, 'train', 2)
    batch = next(iter(DataLoader(data, batch_size=2)))
    mean = make_regression(data.archive.channels, config)
    assert regression_v2(mean, batch).shape == batch['target'].shape
    objective(mean, batch, config).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in mean.parameters())
    mean.eval().requires_grad_(False)
    diffusion = PrecipEDM(data.archive.channels, config)
    loss = objective(diffusion, batch, config, mean)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in diffusion.parameters() if p.grad is not None)
    assert diffusion.net.output[-1].out_channels == 1
    assert diffusion.net.input.in_channels == data.archive.channels+2


def test_importance_weights_preserve_uniform_objective():
    # Proposal [0.25, 0.75] with corrected weights; expectation must be 5.
    losses = []
    for error, importance in [(1., 2.), (9., 2/3)]:
        batch = dict(area=torch.ones(1, 2, 2), importance=torch.tensor([importance]))
        losses.append(weighted_loss(torch.full((1, 1, 2, 2), error), batch, dict(halo=0, size=2)))
    assert float(.25*losses[0]+.75*losses[1]) == pytest.approx(5.)


def test_truth_proposal_keeps_exact_weights(config):
    cfg = deepcopy(config)
    cfg['patch']['proposal'] = 'truth'
    data = PrecipDataset(cfg, 'train', 16)
    weights = [float(data[i]['importance']) for i in range(16)]
    assert any(abs(w-1) > 1e-6 for w in weights)


def test_lr_warmup_then_cosine(config):
    from merraflow.train_v3_precip import lr_lambda
    cfg = deepcopy(config)
    cfg['train'].update(warmup_steps=10, lr_schedule='cosine', min_lr_ratio=.1)
    f = lr_lambda(cfg, 110)
    assert f(0) == pytest.approx(.1) and f(9) == pytest.approx(1.)
    assert f(60) == pytest.approx(.55) and f(110) == pytest.approx(.1) and f(500) == pytest.approx(.1)


def test_synchronized_tiling_matches_global_trajectory(config):
    # With a pointwise exact Gaussian denoiser, blending overlapping tiles at every
    # step must reproduce the single global trajectory exactly (no seams).
    from merraflow.inference_v3_precip import synchronized_residual
    class Gaussian(torch.nn.Module):
        def forward(self, x, sigma, *args):
            return x/(1+sigma[:, None, None, None]**2)
    class Tiles:
        size, halo, device, precision = 8, 2, torch.device('cpu'), 'fp32'
        width = 12
        tiles = [(y, x) for y in (0, 6, 12) for x in (0, 6, 10)]
        def chunks(self):
            yield range(0, 4), None, None, None
            yield range(4, len(self.tiles)), None, None, None
    noise = np.random.default_rng(3).standard_normal((1, 24, 22)).astype('float32')
    tiled = synchronized_residual(Gaussian(), Tiles(), noise, config, steps=6)
    reference = heun_edm(lambda x, s: x/(1+s**2), torch.from_numpy(noise), sigma_schedule(config, 6, torch.device('cpu')))
    np.testing.assert_allclose(tiled, reference[0].numpy(), rtol=1e-5, atol=1e-6)


def test_hourly_trapezoid_targets(config, tmp_path, monkeypatch):
    import xarray as xr
    from datetime import datetime, timedelta
    import merraflow.prepare_v3_precip as prep
    from merraflow.dataset_v2 import ArchiveV2
    archive = ArchiveV2(config['data']['prepared'])
    # The fixture only has :30 files; synthesize the bounding :00 snapshots.
    for entry in archive.index['entries']:
        middle = datetime.fromisoformat(entry['time'])
        for t, factor in ((middle-timedelta(minutes=30), 2.), (middle+timedelta(minutes=30), 3.)):
            path = prep.snapshot_path(entry, t)
            if path.exists():
                continue
            with xr.open_dataset(entry['hr']) as ds:
                copy = ds.load()
            copy['PRECTOT'] = copy['PRECTOT']*factor
            copy = copy.assign_coords(time=[np.datetime64(t)])
            path.parent.mkdir(parents=True, exist_ok=True)
            copy.to_netcdf(path, engine='h5netcdf')
    cfg = deepcopy(config)
    cfg['data'].update(target_kind='hourly_mean_trapezoid', hourly_targets=str(tmp_path/'hourly_v3_precip'))
    validate_config(cfg)
    with pytest.raises(FileNotFoundError, match='prepare-hourly'):
        PrecipArchive(cfg)
    n = len(archive.index['entries'])
    assert prep.prepare_hourly(cfg)['written'] == n
    assert prep.prepare_hourly(cfg)['skipped'] == n  # Restart-safe.
    assert prep.finalize_hourly(cfg)['missing'] == 0
    hourly = PrecipArchive(cfg)
    entry = hourly.eligible('train')[0]
    t = datetime.fromisoformat(entry['time'])
    early, late = [prep.read_rate(prep.snapshot_path(entry, t+timedelta(minutes=m)), t+timedelta(minutes=m)) for m in (-30, 30)]
    mid = np.asarray(archive.array(entry, 'truth')[1])
    np.testing.assert_allclose(hourly.truth_field(entry)[0], .25*early+.5*mid+.25*late, rtol=1e-5, atol=1e-6)
    assert not np.allclose(hourly.truth_field(entry)[0], mid)
    data = PrecipDataset(cfg, 'train', 2)
    assert np.isfinite(data[0]['target'].numpy()).all()
    assert hourly.hourly_fingerprint
    # A changed archive/method cannot relabel old targets under a new fingerprint.
    changed_archive = deepcopy(archive)
    changed_archive.index['fingerprint'] = 'changed-archive'
    with monkeypatch.context() as patch:
        patch.setattr(prep, 'ArchiveV2', lambda _: changed_archive)
        for operation in (prep.prepare_hourly, prep.finalize_hourly):
            with pytest.raises(ValueError, match='provenance mismatch'):
                operation(cfg)
    with monkeypatch.context() as patch:
        patch.setattr(prep, 'WEIGHTS', (.2, .6, .2))
        with pytest.raises(ValueError, match='provenance mismatch'):
            prep.prepare_hourly(cfg)
    source_path = prep.snapshot_path(entry, t)
    original_stat = source_path.stat()
    try:
        os.utime(source_path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns+1000000))
        for operation in (prep.prepare_hourly, prep.finalize_hourly, PrecipArchive):
            with pytest.raises(ValueError, match='provenance mismatch'):
                operation(cfg)
    finally:
        os.utime(source_path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    # A target change invalidates both finalization and already-finalized datasets.
    target_path = hourly.hourly_root/entry['id']/prep.HOURLY_FILE
    target_stat = target_path.stat()
    try:
        os.utime(target_path, ns=(target_stat.st_atime_ns, target_stat.st_mtime_ns+1000000))
        with pytest.raises(ValueError, match='provenance mismatch'):
            PrecipArchive(cfg)
    finally:
        os.utime(target_path, ns=(target_stat.st_atime_ns, target_stat.st_mtime_ns))
    # Checksums also catch target corruption when timestamps and shape survive.
    original_bytes = target_path.read_bytes()
    try:
        altered = np.load(target_path).copy()+1
        np.save(target_path, altered)
        os.utime(target_path, ns=(target_stat.st_atime_ns, target_stat.st_mtime_ns))
        with pytest.raises(ValueError, match='checksum mismatch'):
            prep.finalize_hourly(cfg)
    finally:
        target_path.write_bytes(original_bytes)
        os.utime(target_path, ns=(target_stat.st_atime_ns, target_stat.st_mtime_ns))
    assert PrecipArchive(cfg).hourly_fingerprint == hourly.hourly_fingerprint
    # A :30 snapshot that disagrees with the archived truth (mis-pairing) is rejected.
    original = prep.read_rate
    monkeypatch.setattr(prep, 'read_rate', lambda path, time: original(path, time)+(1e-2 if time.minute == 30 else 0))
    tampered = deepcopy(cfg)
    tampered['data']['hourly_targets'] = str(tmp_path/'tampered_v3_precip')
    with pytest.raises(ValueError, match='pairing'):
        prep.prepare_hourly(tampered)


def test_time_limit_stops_and_resumes(config, tmp_path):
    cfg = deepcopy(config)
    cfg['train'].update(output=str(tmp_path/'segment_v3_precip'), regression_epochs=3)
    best = train(cfg, 'regression', time_limit_hours=1e-9)
    first = torch.load(best.parent/'last_v3_precip.pt', weights_only=True)
    assert first['epoch'] == 0 and first['scheduler']['last_epoch'] > 0
    train(cfg, 'regression', resume=best.parent/'last_v3_precip.pt')
    done = torch.load(best.parent/'last_v3_precip.pt', weights_only=True)
    assert done['epoch'] == 2 and [r['epoch'] for r in done['history']] == [1, 2, 3]
    incomplete = deepcopy(cfg)
    incomplete['train'].update(output=str(tmp_path/'incomplete_v3_precip'))
    partial = train(incomplete, 'regression', time_limit_hours=1e-9)
    with pytest.raises(ValueError, match='incomplete'):
        train(incomplete, 'diffusion', regression_checkpoint=partial)


def test_diffusion_resume_before_first_validation(config, tmp_path):
    cfg = deepcopy(config)
    cfg['train'].update(output=str(tmp_path/'early_v3_precip'), regression_epochs=1,
                        diffusion_epochs=5, validation_interval=5)
    mean = train(cfg, 'regression')
    checkpoint = train(cfg, 'diffusion', regression_checkpoint=mean, time_limit_hours=1e-9)
    assert checkpoint.name == 'last_v3_precip.pt'
    assert checkpoint.exists() and not (checkpoint.parent/'best_v3_precip.pt').exists()
    saved = torch.load(checkpoint, weights_only=True)
    assert saved['epoch'] == 0 and saved['best'] == float('inf')
    assert 'crps' not in saved['history'][0]
    best = train(cfg, 'diffusion', resume=checkpoint)
    assert best.name == 'best_v3_precip.pt' and best.exists()
    finished = torch.load(best, weights_only=True)
    assert finished['epoch'] == 4 and np.isfinite(finished['best'])


def test_tile_cache_disables_autograd(config):
    from merraflow.inference_v3_precip import TileField
    archive = PrecipArchive(config)
    mean = make_regression(archive.channels, config).eval()
    assert torch.is_grad_enabled() and any(p.requires_grad for p in mean.parameters())
    field = TileField(mean, archive, archive.eligible('val')[0], config, torch.device('cpu'))
    assert field.means and all(not t.requires_grad and t.grad_fn is None for t in field.means)
    assert torch.is_grad_enabled()  # Do not leak no_grad into training.


def test_legacy_hourly_directory_rejected(config, tmp_path):
    from merraflow.prepare_v3_precip import prepare_hourly, HOURLY_FILE
    cfg = deepcopy(config)
    cfg['data'].update(target_kind='hourly_mean_trapezoid', hourly_targets=str(tmp_path/'legacy_v3_precip'))
    dest = Path(cfg['data']['hourly_targets'])/'old_hour'/HOURLY_FILE
    dest.parent.mkdir(parents=True)
    np.save(dest, np.zeros((1, 2, 2), dtype='float32'))
    with pytest.raises(ValueError, match='Unverified legacy'):
        prepare_hourly(cfg)


def test_checkpoint_rejects_changed_hourly_targets(config):
    from merraflow.train_v3_precip import check_checkpoint
    archive = PrecipArchive(config)
    checkpoint = dict(version='v3_precip', targets=['precip'], fingerprint=archive.index['fingerprint'],
                      hourly_fingerprint='old-hourly-targets', config=config)
    with pytest.raises(ValueError, match='hourly target fingerprint'):
        check_checkpoint(checkpoint, config, archive)


def test_heun_gaussian_reference(config):
    # Exact posterior denoiser for N(0, 1): probability-flow ODE maps the
    # high-sigma Gaussian back to unit variance (within numerical tolerance).
    class Gaussian(torch.nn.Module):
        def forward(self, x, sigma, *args):
            return x/(1+sigma[:, None, None, None]**2)
    noise = torch.randn((2, 1, 8, 8), generator=torch.Generator().manual_seed(71))
    sampled = sample_edm(Gaussian(), noise, None, None, None, config, steps=128)
    np.testing.assert_allclose(sampled, noise, rtol=.003, atol=.001)


def test_config_rejects_accumulation_and_future_leakage(config):
    for update in (dict(target_kind='hourly_accumulation'), dict(history_hours=[0, 1]), dict(history_hours=[0, 0]),
                   dict(target_kind='hourly_mean_trapezoid'), dict(dry_offset=.25, dry_threshold_mm_h=0)):
        cfg = deepcopy(config)
        cfg['data'].update(update)
        with pytest.raises(ValueError):
            validate_config(cfg)
    cfg = deepcopy(config)
    cfg['patch']['loss_on_halo'] = False
    with pytest.raises(ValueError, match='loss_on_halo'):
        validate_config(cfg)
    cfg['inference']['blend'] = 'owner'
    validate_config(cfg)


def test_training_resume_and_end_to_end(config, tmp_path, monkeypatch):
    import merraflow.train_v3_precip as module
    cfg = deepcopy(config)
    cfg['train']['output'] = str(tmp_path/'train_v3_precip')
    cfg['inference']['output'] = str(tmp_path/'predictions_v3_precip')
    cfg['train']['regression_epochs'] = 5
    cfg['train']['diffusion_epochs'] = 5
    original_save = module.atomic_save
    recovery = {}
    def save(path, value):
        if value['stage'] == 'regression' and value['epoch'] == 0:
            recovery[path.name] = deepcopy(value)
        original_save(path, value)
    monkeypatch.setattr(module, 'atomic_save', save)
    mean = train(cfg, 'regression')
    monkeypatch.setattr(module, 'atomic_save', original_save)
    first = tmp_path/'recovery'
    first.mkdir()
    for name, value in recovery.items():
        torch.save(value, first/name)
    resumed = deepcopy(cfg)
    resumed['train']['output'] = str(tmp_path/'resumed_v3_precip')
    train(resumed, 'regression', resume=first/'last_v3_precip.pt')
    end = torch.load(mean.parent/'last_v3_precip.pt', weights_only=True)
    resumed_end = torch.load(tmp_path/'resumed_v3_precip/regression_v3_precip/last_v3_precip.pt', weights_only=True)
    for key, tensor in end['model'].items():
        torch.testing.assert_close(tensor, resumed_end['model'][key], rtol=0, atol=0)
    assert end['history'] == resumed_end['history']
    diffusion = train(cfg, 'diffusion', regression_checkpoint=mean)
    for checkpoint in (mean, diffusion):
        plots = checkpoint.parent/'validation_plots'
        assert [x.name for x in plots.iterdir()] == ['epoch_0005']
        for name in ('fields.png', 'diagnostics.png', 'history.png', 'samples.npz', 'metadata.json'):
            assert (plots/'epoch_0005'/name).exists()
    predict(cfg, diffusion, limit=1)
    report = evaluate(cfg)
    summary = json.loads(report.read_text())
    assert summary['synthetic'] is True
    assert summary['target_kind'] == 'midpoint_rate'
    assert summary['metrics']['diffusion']['crps'] >= 0
    assert summary['evaluated_hours'] == 1
    assert (report.parent/'precip_comparison.png').exists()
    assert np.isfinite(summary['metrics']['diffusion']['crps'])
    with pytest.raises(FileExistsError):
        predict(cfg, diffusion, limit=1)
    changed = deepcopy(cfg)
    changed['inference']['steps'] += 1
    with pytest.raises(ValueError, match='Mixed checkpoints'):
        predict(changed, diffusion, limit=1)
    # Evaluation must reject an incomplete ensemble, not score its survivors.
    member = next(report.parent.parent.glob('*_m001_v3_precip.nc'))
    member.rename(member.with_suffix('.hidden'))
    with pytest.raises(ValueError, match='Incomplete ensemble'):
        evaluate(cfg)


def test_four_process_training_and_plots(config, tmp_path):
    """Exercise real DDP collectives, unused parameters and both objectives on CPU."""
    import yaml
    cfg = deepcopy(config)
    cfg['patch']['samples_per_epoch'] = 24
    cfg['train'].update(output=str(tmp_path/'ddp_v3_precip'), regression_epochs=5,
                        diffusion_epochs=5, accumulate=2)
    path = tmp_path/'config.yaml'
    path.write_text(yaml.safe_dump(cfg))
    env = dict(os.environ, OMP_NUM_THREADS='1', MPLCONFIGDIR=str(tmp_path/'mpl'),
               GLOO_SOCKET_IFNAME='lo0' if sys.platform == 'darwin' else 'lo')
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    command = [sys.executable, '-m', 'torch.distributed.run', '--master-addr=127.0.0.1',
               f'--master-port={port}', '--nnodes=1',
               '--nproc-per-node=4', '-m', 'merraflow.cli_v3_precip', 'train', '--config', str(path)]
    mean = Path(cfg['train']['output'])/'regression_v3_precip/best_v3_precip.pt'
    for stage in ('regression', 'diffusion'):
        args = command+['--stage', stage]
        if stage == 'diffusion':
            args += ['--regression-checkpoint', str(mean)]
        result = subprocess.run(args, env=env, capture_output=True, text=True, timeout=180)
        assert result.returncode == 0, result.stdout+result.stderr
        directory = Path(cfg['train']['output'])/(stage+'_v3_precip')
        saved = torch.load(directory/'last_v3_precip.pt', weights_only=True)
        assert saved['world_size'] == 4
        assert len(saved['rng_by_rank']) == 4
        assert not torch.equal(saved['rng_by_rank'][0]['cpu'], saved['rng_by_rank'][1]['cpu'])
        assert saved['epoch'] == 4 and len(saved['history']) == 5
        assert np.isfinite(saved['residual_scale']) and saved['residual_scale'] > 0
        assert (directory/'validation_plots/epoch_0005/fields.png').exists()
        # Completed-epoch resume restores all four rank states without rewriting history.
        result = subprocess.run(command+['--stage', stage, '--resume', str(directory/'last_v3_precip.pt')],
                                env=env, capture_output=True, text=True, timeout=90)
        assert result.returncode == 0, result.stdout+result.stderr


def test_submission_pipeline_uses_dependency(config, tmp_path):
    import yaml
    cfg = deepcopy(config)
    cfg['train']['output'] = str(tmp_path/'submission_v3_precip')
    path = tmp_path/'config.yaml'
    path.write_text(yaml.safe_dump(cfg))
    bin_path = tmp_path/'bin'
    bin_path.mkdir()
    log = tmp_path/'submitted'
    sbatch = bin_path/'sbatch'
    sbatch.write_text('#!/bin/bash\nprintf "%s %s\\n" "$STAGE" "$*" >> "$SUBMISSION_LOG"\n'
                      'if [[ "$STAGE" == regression ]]; then echo 123; else echo 124; fi\n')
    sbatch.chmod(0o755)
    env = dict(os.environ, CONFIG=str(path), SUBMISSION_LOG=str(log), ENV_DIR=str(Path(sys.executable).parent.parent),
               PATH=f'{bin_path}:{Path(sys.executable).parent}:'+os.environ['PATH'])
    result = subprocess.run(['bash', 'scripts/submit_v3_precip.sh'], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout+result.stderr
    calls = log.read_text().splitlines()
    assert len(calls) == 2+8  # Default: 2 regression + 8 diffusion segments.
    assert calls[0].startswith('regression ') and '--dependency' not in calls[0]
    assert calls[1].startswith('regression ') and '--dependency=afterok:123' in calls[1]
    assert calls[2].startswith('diffusion ') and '--dependency=afterok:123' in calls[2]
    assert '--dependency=afterok:124' in calls[3]
    assert all('--kill-on-invalid-dep=yes' in c for c in calls[1:])
    script = Path('scripts/slurm_train_v3_precip.sh').read_text()
    assert ('#SBATCH --gres=gpu:4' in script and '#SBATCH --cpus-per-gpu=4' in script
            and '--nproc-per-node=4' in script)


def test_prepare_and_train_submission_chain(config, tmp_path):
    import yaml
    cfg = deepcopy(config)
    cfg['data'].update(target_kind='hourly_mean_trapezoid', hourly_targets=str(tmp_path/'hourly_v3_precip'))
    cfg['train']['output'] = str(tmp_path/'submission_v3_precip')
    path = tmp_path/'config.yaml'
    path.write_text(yaml.safe_dump(cfg))
    bin_path = tmp_path/'bin'
    bin_path.mkdir()
    log = tmp_path/'submitted'
    counter = tmp_path/'counter'
    sbatch = bin_path/'sbatch'
    sbatch.write_text('#!/bin/bash\nn=0\n[[ ! -f "$COUNTER" ]] || n=$(cat "$COUNTER")\n'
                      'n=$((n+1))\necho "$n" > "$COUNTER"\n'
                      'printf "%s finalize=%s %s\\n" "${STAGE:-prepare}" "${FINALIZE:-0}" "$*" >> "$SUBMISSION_LOG"\n'
                      'echo "$n"\n')
    sbatch.chmod(0o755)
    env = dict(os.environ, CONFIG=str(path), SUBMISSION_LOG=str(log), COUNTER=str(counter), PREPARE_FIRST='1',
               ENV_DIR=str(Path(sys.executable).parent.parent),
               REGRESSION_SEGMENTS='2', DIFFUSION_SEGMENTS='8', PREPARE_ARRAY_CONCURRENCY='3',
               PATH=f'{bin_path}:{Path(sys.executable).parent}:'+os.environ['PATH'])
    result = subprocess.run(['bash', 'scripts/submit_v3_precip.sh'], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout+result.stderr
    calls = log.read_text().splitlines()
    assert len(calls) == 12
    assert '--array=' in calls[0] and '%3' in calls[0] and 'finalize=0' in calls[0]
    assert 'finalize=1' in calls[1] and '--dependency=afterok:1' in calls[1]
    assert calls[2].startswith('regression ') and '--dependency=afterok:2' in calls[2]
    assert calls[4].startswith('diffusion ') and '--dependency=afterok:4' in calls[4]
    assert '--dependency=afterok:11' in calls[-1]
    # Reject bad job counts before any sbatch invocation.
    log.unlink()
    env['DIFFUSION_SEGMENTS'] = '0'
    failed = subprocess.run(['bash', 'scripts/submit_v3_precip.sh'], env=env, capture_output=True, text=True)
    assert failed.returncode != 0 and not log.exists()
    env.pop('PREPARE_FIRST')
    env['PREPARE_ARRAY_CONCURRENCY'] = 'zero'
    failed = subprocess.run(['bash', 'scripts/submit_prepare_hourly_v3_precip.sh'], env=env,
                            capture_output=True, text=True)
    assert failed.returncode != 0
    production = load_config('configs/discover_v3_precip.yaml')
    assert production['train']['validation_plot_interval'] == 5
    assert production['train']['workers'] == 4


def test_training_can_depend_on_existing_hourly_finalizer(config, tmp_path):
    import yaml
    cfg = deepcopy(config)
    cfg['train']['output'] = str(tmp_path/'dependency_v3_precip')
    path = tmp_path/'config.yaml'
    path.write_text(yaml.safe_dump(cfg))
    bin_path = tmp_path/'bin'
    bin_path.mkdir()
    log = tmp_path/'submitted'
    sbatch = bin_path/'sbatch'
    sbatch.write_text('#!/bin/bash\nprintf "%s %s\\n" "$STAGE" "$*" >> "$SUBMISSION_LOG"\n'
                      'if [[ "$STAGE" == regression ]]; then echo 123; else echo 124; fi\n')
    sbatch.chmod(0o755)
    env = dict(os.environ, CONFIG=str(path), SUBMISSION_LOG=str(log), AFTEROK_JOB='58501027',
               ENV_DIR=str(Path(sys.executable).parent.parent),
               PATH=f'{bin_path}:{Path(sys.executable).parent}:'+os.environ['PATH'])
    result = subprocess.run(['bash', 'scripts/submit_v3_precip.sh'], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout+result.stderr
    calls = log.read_text().splitlines()
    assert len(calls) == 10
    assert calls[0].startswith('regression ') and '--dependency=afterok:58501027' in calls[0]
    assert calls[1].startswith('regression ') and '--dependency=afterok:123' in calls[1]
    assert calls[2].startswith('diffusion ') and '--dependency=afterok:123' in calls[2]
    for key, value in [('AFTEROK_JOB', 'not-a-job'), ('PREPARE_FIRST', '1')]:
        bad = dict(env)
        bad[key] = value
        failed = subprocess.run(['bash', 'scripts/submit_v3_precip.sh'], env=bad, capture_output=True, text=True)
        assert failed.returncode != 0
