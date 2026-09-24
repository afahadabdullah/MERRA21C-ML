"""Synthetic contract tests; no claims of meteorological skill or GPU throughput."""
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
import json
import os
import pickle
import subprocess
import socket
import sys
import numpy as np
import pytest
import torch
import xarray as xr
import yaml
from torch.utils.data import DataLoader
from merraflow.config_v2 import load_config_v2
from merraflow.synthetic_v2 import make_synthetic_v2
from merraflow.prepare_v2 import prepare
from merraflow.prepare_v3_precip import prepare_hourly, finalize_hourly, snapshot_path
from merraflow.prepare_v4 import prepare_humidity, finalize_humidity, read_q2m, load_index
from merraflow.dataset_v2 import ArchiveV2
from merraflow.v4 import (load_config, validate_config, ArchiveV4, DatasetV4, TARGETS,
    FrozenRegression, regression_bundle, make_model, objective, check_checkpoint)
from merraflow.model_v2 import UNetV2
from merraflow.train_v4 import train, calibrate
from merraflow.cli_v4 import preflight
from merraflow.inference_v4 import predict, sample_frame


@pytest.fixture(scope='module')
def setup(tmp_path_factory):
    torch.set_num_threads(1)
    root = tmp_path_factory.mktemp('v4')
    original = load_config_v2(make_synthetic_v2(root, load_config_v2('configs/discover_v2.yaml')))
    for path in (root/'hr'/'hwt_30mn_slv_LCC').glob('*/*.nc4'):
        with xr.open_dataset(path) as ds:
            copy = ds.load()
        copy['SPFH_2M'] = copy['TMP_2M']*0+.009
        copy['SPFH_2M'].attrs['units'] = 'kg kg-1'
        copy.to_netcdf(path, engine='h5netcdf')
    prepare(original)
    a = ArchiveV2(original['data']['prepared'])
    for entry in a.index['entries']:
        middle = datetime.fromisoformat(entry['time'])
        for time, factor in ((middle-timedelta(minutes=30), .5), (middle+timedelta(minutes=30), 1.5)):
            path = snapshot_path(entry, time)
            if path.exists():
                continue
            with xr.open_dataset(entry['hr']) as ds:
                copy = ds.load()
            copy['PRECTOT'].values *= factor
            copy = copy.assign_coords(time=[np.datetime64(time)])
            path.parent.mkdir(parents=True, exist_ok=True)
            copy.to_netcdf(path, engine='h5netcdf')
    cfg = load_config('configs/discover_v4.yaml')
    cfg['data'].update(prepared=original['data']['prepared'], hourly_targets=str(root/'hourly'),
                        humidity_targets=str(root/'humidity'))
    cfg['patch'] = dict(original['patch'], proposal='coarse', loss_on_halo=True)
    cfg['model'] = deepcopy(original['model'])
    cfg['train'].update(device='cpu', precision='fp32', workers=0, epochs=5, batch_size=2,
        accumulate=2, validation_patches=4, validation_members=2, validation_steps=2,
        validation_plot_samples=1, calibration_batches=2, time_limit_hours=None, output=str(root/'run_v4'))
    cfg['inference'].update(steps=2, members=2, output=str(root/'predictions_v4'))
    prepare_hourly(cfg)
    finalize_hourly(cfg)
    prepare_humidity(cfg)
    finalize_humidity(cfg)
    regression = UNetV2(a.index['condition_channels'], **original['model'])
    source = root/'original_regression.pt'
    torch.save(dict(version='v2', stage='regression', epoch=29, config=original, stats=a.stats,
                    fingerprint=a.index['fingerprint'], ema=regression.state_dict(), initialization=None), source)
    cfg['conditioning']['checkpoint'] = str(source)
    return cfg


def conditioner_for(cfg, archive):
    return FrozenRegression(regression_bundle(cfg['conditioning']['checkpoint'], archive),
        archive.index['condition_channels'], archive.stats, archive.scale, cfg['data']['humidity_scale_kg_kg'])


def test_contract_six_targets_hourly_history_and_q2m(setup):
    d = DatasetV4(setup, 'train', 2, 3)
    assert d.cached_scores is None and d.cached_edges is None
    b = d[0]
    a = d.archive
    assert len(TARGETS) == 6 and TARGETS[-1] == 'q2m'
    assert b['target'].shape[0] == 6
    assert a.channels == a.index['condition_channels']+len(a.cm)
    assert b['condition'].shape[0] == a.channels
    assert b['original_condition'].shape[0] == a.index['condition_channels']
    for split in ('train','val','test'):
        assert len(a.eligible(split)) == 1
        history = a.history(a.eligible(split)[0])
        assert history[0]['split'] == history[1]['split'] == split
        assert np.datetime64(history[1]['time'])-np.datetime64(history[0]['time']) == np.timedelta64(1,'h')
    np.testing.assert_allclose(b['target'][1], np.sqrt(1+b['truth'][1].numpy()/a.scale)-1, atol=1e-6)
    np.testing.assert_allclose(b['truth'][5], .009)
    np.testing.assert_allclose(b['coarse'][5], .008)
    np.testing.assert_allclose(b['target'][5], 1., rtol=1e-5)
    assert float(b['importance']) == 1.
    # Hourly target differs from the original midpoint fixture.
    entry = a.eligible('train')[0]
    assert not np.allclose(a.physical_truth(entry)[1], a.array(entry, 'truth')[1])
    # Reused regression inputs must equal the original v2 computation exactly,
    # including broad crops that cross the domain boundary.
    original = ArchiveV2(setup['data']['prepared'])
    h, w = a.shape
    size = setup['patch']['size']
    for y, x in ((0, 0), ((h-size)//2, (w-size)//2), (h-size, w-size)):
        inputs = a.inputs_with_original(entry, y, x, setup['patch'])
        condition, context = original.inputs(entry, y, x, setup['patch'])
        torch.testing.assert_close(inputs['original_condition'], condition, rtol=0, atol=0)
        torch.testing.assert_close(inputs['original_context'], context, rtol=0, atol=0)


def test_rain_never_residual_and_state_roundtrip(setup):
    d = DatasetV4(setup, 'train', 2, 4)
    b = next(iter(DataLoader(d, batch_size=2)))
    c = conditioner_for(setup, d.archive)
    c.flow_scale.fill_(.4)
    c.flow_scale[:,1] = 1.
    original = b['target'].clone()
    c.prepare(b)
    torch.testing.assert_close(b['target'][:,1], original[:,1])
    torch.testing.assert_close(b['target'][:,[0,2,3,4,5]], ((original-b['mean'])/.4)[:,[0,2,3,4,5]])
    torch.testing.assert_close(c.physical(b['target'], b['mean'], b['coarse']), b['truth'], rtol=2e-5, atol=1e-4)
    changed_mean = b['mean']+100
    changed_coarse = b['coarse']+10
    torch.testing.assert_close(c.physical(b['target'],changed_mean,changed_coarse)[:,1],b['truth'][:,1],rtol=1e-5,atol=1e-6)
    assert not b['mean'].requires_grad
    model = make_model(d.archive.channels, setup)
    loss = objective(model, b)
    loss.backward()
    assert torch.isfinite(loss) and any(p.grad is not None for p in model.parameters())
    assert all(p.grad is None for p in c.parameters())
    assert model.output[-1].out_channels == 6


def test_checkpointing_toggle_preserves_loss_and_gradients(setup):
    d = DatasetV4(setup, 'train', 2, 4)
    batch = next(iter(DataLoader(d, batch_size=2)))
    conditioner_for(setup, d.archive).prepare(batch)
    checkpointed = make_model(d.archive.channels, setup).train()
    checkpointed.checkpointing = True
    # A nonzero output layer exercises gradients throughout the U-Net.
    with torch.no_grad():
        checkpointed.output[-1].weight.normal_(0, .02)
    direct = deepcopy(checkpointed)
    direct.checkpointing = False
    losses = []
    for model in (checkpointed, direct):
        loss = objective(model, batch, generator=torch.Generator().manual_seed(42))
        loss.backward()
        losses.append(loss.detach())
    torch.testing.assert_close(*losses)
    assert checkpointed.input.weight.grad.abs().sum() > 0
    for (name, before), (other_name, after) in zip(checkpointed.named_parameters(), direct.named_parameters()):
        assert name == other_name
        if before.grad is None:
            assert after.grad is None
        else:
            torch.testing.assert_close(before.grad, after.grad)


def test_sampling_does_not_use_truth_scores(setup):
    d = DatasetV4(setup, 'train', 3, 1)
    original = d.archive.array
    def array(entry, name):
        assert name not in ('truth','residual')
        return original(entry,name)
    d.archive.array = array
    assert np.isclose(d.proposal(d.entries[0]).sum(),1)
    val = DatasetV4(setup, 'val', 3, 1)
    np.testing.assert_allclose(val.proposal(val.entries[0]), 1/len(val.yy))


def test_humidity_provenance_and_no_fake_target(setup):
    a = ArchiveV2(setup['data']['prepared'])
    assert prepare_humidity(setup)['skipped'] == len(a.index['entries'])
    assert load_index(setup,a)['fingerprint']
    bad = deepcopy(setup)
    bad['data']['humidity_target_variable'] = 'DOES_NOT_EXIST'
    with pytest.raises(KeyError, match='missing'):
        read_q2m(bad,a,a.index['entries'][0])
    with pytest.raises(ValueError, match='provenance'):
        load_index(bad,a)


def test_config_rejects_leakage_wrong_codec_and_loss(setup):
    for patch in ({'history_hours':[0,1]}, {'history_hours':[-1,-1,0]}, {'representation':'log1p'}, {'dry_offset':.25}):
        cfg = deepcopy(setup)
        cfg['data'].update(patch)
        with pytest.raises(ValueError):
            validate_config(cfg)
    cfg = deepcopy(setup)
    cfg['train']['channel_weights'] = [1]*5
    with pytest.raises(ValueError, match='six'):
        validate_config(cfg)


def test_training_resume_validation_and_full_inference(setup, tmp_path, capsys):
    cfg = deepcopy(setup)
    cfg['model']['activation_checkpointing'] = True
    for key in ('prefetch_factor', 'array_cache_size', 'proposal_cache'):
        cfg['train'].pop(key, None)  # Emulate a checkpoint from before loader tuning.
    cfg['train']['output'] = str(tmp_path/'training')
    # Epoch-boundary stop exercises a real optimizer/scheduler resume.
    cfg['train']['time_limit_hours'] = .00001
    path = train(cfg)
    first = torch.load(path, weights_only=True)
    assert first['epoch'] == 0 and first['flow_scale'].shape == (1,6,1,1)
    assert first['history'][0]['data_wait_s_per_rank'] >= 0
    assert first['history'][0]['step_s_per_rank'] > 0
    assert first['flow_scale'][0,1,0,0] == 1
    cfg['train']['time_limit_hours'] = None
    cfg['train'].update(prefetch_factor=2, array_cache_size=4, proposal_cache=True)
    cfg['model']['activation_checkpointing'] = False
    capsys.readouterr()
    train(cfg,resume=path)
    resume_log = capsys.readouterr().out
    assert 'using saved calibration (no recalibration)' in resume_log
    assert 'Calibration starting' not in resume_log
    assert 'activation_checkpointing=False' in resume_log
    saved = torch.load(path,weights_only=True)
    assert saved['epoch'] == 4 and saved['targets'] == TARGETS
    assert len(saved['history']) == 5
    assert 'crps' not in saved['history'][3]
    assert saved['history'][4]['validation_wall_s'] > 0
    assert all(f'{name}_crps' in saved['history'][4] for name in TARGETS)
    folder = path.parent/'validation_plots'/'epoch_0005'
    for name in ('fields.png','history.png','metrics.json','samples.npz'):
        assert (folder/name).exists()
    assert np.load(folder/'samples.npz')['patch_0_ensemble'].shape[2] == 16
    assert (path.parent/'best_v4.pt').exists()
    a = ArchiveV4(cfg)
    incompatible = deepcopy(cfg)
    incompatible['model']['base_channels'] += 1
    with pytest.raises(ValueError, match='Checkpoint model mismatch'):
        check_checkpoint(saved, incompatible, a)
    bad = dict(saved,hourly_fingerprint='wrong')
    with pytest.raises(ValueError,match='fingerprint'):
        check_checkpoint(bad,cfg,a)
    cfg['inference']['output'] = str(tmp_path/'predictions')
    result = predict(cfg,path,'test',1)
    files = list(result.glob('*.nc'))
    assert len(files) == 2
    with xr.open_dataset(files[0]) as ds:
        assert set(TARGETS).issubset(ds.data_vars)
        assert ds.precip.attrs['cell_methods'] == 'time: mean'
        assert ds.q2m.attrs['cell_methods'] == 'time: point'
        assert ds.q2m.min() >= 0 and ds.q2m.max() <= 1
        assert np.isfinite(ds.precip).all()
    # Idempotent finished resume, then plot recovery after interrupted saving.
    (folder/'fields.png').unlink()
    train(cfg,resume=path)
    assert (folder/'fields.png').exists()


def test_full_domain_rain_is_direct(setup):
    a = ArchiveV4(setup)
    c = conditioner_for(setup,a)
    class Zero(torch.nn.Module):
        def forward(self, x, *args):
            return torch.zeros_like(x)
    actual = sample_frame(Zero(),c,a,a.eligible('test')[0],setup,torch.device('cpu'),42)
    h,w = a.shape
    halo = setup['patch']['halo']
    noise = np.random.default_rng(42).standard_normal((6,h+2*halo,w+2*halo)).astype('float32')
    z = np.maximum(noise[1,halo:halo+h,halo:halo+w],0)
    np.testing.assert_allclose(actual[1],a.scale*z*(z+2),rtol=1e-6)


def test_calibration_control_group_and_batch_limit(monkeypatch):
    group = object()
    monkeypatch.setattr(torch.distributed, 'is_initialized', lambda: True)
    monkeypatch.setattr(torch.distributed, 'get_rank', lambda: 0)
    calls = []

    def all_reduce(totals, group=None):
        calls.append(group)
        assert totals.device.type == 'cpu'
        assert totals.dtype == torch.float64
        torch.testing.assert_close(totals, torch.tensor([8.]*6+[2.], dtype=torch.float64))
        # A remote rank contributes one sample with error 3 in each channel.
        totals.add_(torch.tensor([9.]*6+[1.], dtype=torch.float64))

    monkeypatch.setattr(torch.distributed, 'all_reduce', all_reduce)

    class Conditioner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer('flow_scale', torch.ones(1, 6, 1, 1))

        def forward(self, batch):
            return torch.zeros_like(batch['target'])

    class Loader:
        exhausted = False

        def __len__(self):
            return 1

        def __iter__(self):
            yield dict(target=torch.full((2, 6, 1, 1), 2.), area=torch.ones(2, 1, 1))
            self.exhausted = True

    cfg = dict(patch=dict(halo=0, size=1), train=dict(calibration_batches=1, precision='fp32'))
    conditioner = Conditioner()
    loader = Loader()
    scale = calibrate(conditioner, loader, cfg, torch.device('cpu'), group)
    expected = torch.full((1, 6, 1, 1), (17/3)**.5)
    expected[:, 1] = 1.
    assert calls == [group]
    assert loader.exhausted
    torch.testing.assert_close(scale, expected)
    torch.testing.assert_close(conditioner.flow_scale, expected)


def test_quick_preflight_skips_all_hours_file_scan(setup, tmp_path, monkeypatch):
    cfg = deepcopy(setup)
    cfg['train']['output'] = str(tmp_path/'quick_preflight')

    def forbid_file_scan(*args, **kwargs):
        raise RuntimeError('full scan invoked')

    monkeypatch.setattr('merraflow.prepare_v3_precip.verify_target', forbid_file_scan)
    monkeypatch.setattr('merraflow.prepare_v4.verify', forbid_file_scan)
    preflight(cfg)
    with pytest.raises(RuntimeError, match='full scan invoked'):
        preflight(cfg, full=True)


def test_four_process_training_with_plots(setup,tmp_path):
    cfg = deepcopy(setup)
    cfg['model']['activation_checkpointing'] = False
    # Exercise worker startup after process-group setup, including the rank
    # with no validation samples. Production uses the same spawn context.
    cfg['train'].update(epochs=1, workers=1, output=str(tmp_path/'ddp'),validation_interval=1,validation_patches=3)
    # Calibration must end its own loader normally before the longer training
    # loader starts; this reproduces the production partial-epoch boundary.
    cfg['patch']['samples_per_epoch'] = 16
    cfg['train']['calibration_batches'] = 1
    path = tmp_path/'config.yaml'
    path.write_text(yaml.safe_dump(cfg))
    env = dict(os.environ,PYTHONPATH=str(Path('src').resolve()),OMP_NUM_THREADS='1',MPLBACKEND='Agg',
               MPLCONFIGDIR=str(tmp_path/'mpl'), GLOO_SOCKET_IFNAME='lo0' if sys.platform == 'darwin' else 'lo')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0))
        port = sock.getsockname()[1]
    result = subprocess.run([sys.executable,'-m','torch.distributed.run','--master-addr=127.0.0.1',
        f'--master-port={port}','--nnodes=1','--nproc-per-node=4',
        '-m','merraflow.cli_v4','train','--config',str(path)],env=env,capture_output=True,text=True,timeout=300)
    assert result.returncode == 0, result.stdout+result.stderr
    assert 'Calibration 1/1' in result.stdout
    assert 'batch 2/2' in result.stdout
    assert 'activation_checkpointing=False' in result.stdout
    assert 'Exception ignored in:' not in result.stderr
    assert 'terminate called' not in result.stderr
    saved = torch.load(Path(cfg['train']['output'])/'last_v4.pt',weights_only=True)
    assert saved['world_size'] == 4 and len(saved['rng']) == 4
    assert (Path(cfg['train']['output'])/'validation_plots'/'epoch_0001'/'fields.png').exists()


def test_humidity_units_and_missing_fields(setup,tmp_path):
    a = ArchiveV2(setup['data']['prepared'])
    entry = dict(a.index['entries'][0])
    with xr.open_dataset(entry['hr']) as source:
        copy = source.load()
    path = tmp_path/'q2m.nc'
    copy['SPFH_2M'].values *= 1000
    copy['SPFH_2M'].attrs['units'] = 'g kg-1'
    copy.to_netcdf(path,engine='h5netcdf')
    entry['hr'] = str(path)
    np.testing.assert_allclose(read_q2m(setup,a,entry),.009)
    copy['SPFH_2M'].attrs['units'] = '%'
    copy.to_netcdf(path,engine='h5netcdf')
    with pytest.raises(ValueError,match='units'):
        read_q2m(setup,a,entry)


def test_submission_chain_defers_data_checks_to_job(setup,tmp_path):
    cfg = deepcopy(setup)
    cfg['train']['output'] = str(tmp_path/'scheduled')
    path = tmp_path/'config.yaml'
    path.write_text(yaml.safe_dump(cfg))
    binary = tmp_path/'bin'
    binary.mkdir()
    capture = tmp_path/'calls'
    sbatch = binary/'sbatch'
    sbatch.write_text('#!/bin/bash\nprintf "%s\\n" "$*" >> "$CAPTURE"\nwc -l < "$CAPTURE" | tr -d " "\n')
    sbatch.chmod(0o755)
    env = dict(os.environ,CONFIG=str(path),PYTHON_BIN=sys.executable,CAPTURE=str(capture),
        PATH=f'{binary}:'+os.environ['PATH'],RESUME='',DEPENDENCY='',REGRESSION_CHECKPOINT='',
        HOURLY_TARGETS='',SUBMIT_TRAIN='1',MAX_PARALLEL='2')
    result = subprocess.run(['bash','scripts/submit_prepare_v4.sh'],env=env,capture_output=True,text=True,timeout=60)
    assert result.returncode == 0, result.stdout+result.stderr
    calls = capture.read_text().splitlines()
    assert len(calls) == 3
    assert '--array=0-1%2' in calls[0]
    assert '--dependency=afterok:1' in calls[1]
    assert '--dependency=afterok:2' in calls[2] and 'slurm_train_v4.sh' in calls[2]
    # Submission is immediate; the scheduled job performs quick preflight.
    result = subprocess.run(['bash','scripts/submit_v4.sh'],env=env,capture_output=True,text=True,timeout=60)
    assert result.returncode == 0, result.stdout+result.stderr
    assert len(capture.read_text().splitlines()) == 4
    cfg['data']['humidity_targets'] = str(tmp_path/'missing_humidity')
    path.write_text(yaml.safe_dump(cfg))
    result = subprocess.run(['bash','scripts/submit_v4.sh'],env=env,capture_output=True,text=True,timeout=60)
    assert result.returncode == 0, result.stdout+result.stderr
    assert len(capture.read_text().splitlines()) == 5
    with pytest.raises(FileNotFoundError):
        preflight(cfg)


def test_array_mapping_cache_bound_and_spawn_serialization(setup, monkeypatch):
    cfg = deepcopy(setup)
    cfg['train']['array_cache_size'] = 2
    a = ArchiveV4(cfg)
    entry = a.eligible('train')[0]
    original_load = np.load
    opened = []

    def load(path, *args, **kwargs):
        opened.append(path)
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(np, 'load', load)
    first = a.array(entry, 'condition')
    assert a.array(entry, 'condition') is first and len(opened) == 1
    view = first[:, :2, :2]
    expected = view.copy()
    a.array(entry, 'baseline')
    a.humidity(entry)
    assert len(a._array_cache) == 2
    assert a.root/entry['id']/'condition_v2.npy' not in a._array_cache
    np.testing.assert_array_equal(view, expected)  # Eviction must not close live views.
    clone = pickle.loads(pickle.dumps(a))
    assert not clone._array_cache
    np.testing.assert_array_equal(clone.array(entry, 'condition'), first)


def test_proposal_disk_cache_exact_reuse_and_regeneration(setup, tmp_path, monkeypatch):
    reference = DatasetV4(setup, 'train', 4, 91)
    entry = reference.entries[0]
    expected = reference.proposal(entry).copy()
    root = tmp_path/'proposals'

    def dataset(cfg=setup):
        result = DatasetV4(cfg, 'train', 4, 91, archive=reference.archive)
        result.enable_proposal_cache(root)
        return result

    first = dataset()
    np.testing.assert_array_equal(first.proposal(entry), expected)
    cached = dataset()

    def no_rescan(*args):
        raise AssertionError('A cached proposal rescanned the full coarse rain field')

    monkeypatch.setattr(cached, 'rain_score', no_rescan)
    np.testing.assert_array_equal(cached.proposal(entry), expected)
    for epoch in (0, 1):
        reference.epoch = cached.epoch = epoch
        actual = cached[0]
        for key, value in reference[0].items():
            torch.testing.assert_close(actual[key], value, rtol=0, atol=0)
    cache_file = first.disk_proposals.root/f'{entry["id"]}.npy'
    cache_file.write_bytes(b'incomplete cache')
    repaired = dataset()
    np.testing.assert_array_equal(repaired.proposal(entry), expected)
    np.testing.assert_array_equal(np.load(cache_file), expected)
    changed = deepcopy(setup)
    changed['patch']['detail_fraction'] /= 2
    assert dataset(changed).disk_proposals.root != first.disk_proposals.root
    disabled = dataset()
    disabled.disk_proposals.enabled = False
    np.testing.assert_array_equal(disabled.proposal(entry), expected)


def test_prefetch_preserves_samples_across_epoch_worker_restarts(setup, tmp_path):
    reference = DatasetV4(setup, 'train', 4, 73)
    cached = DatasetV4(setup, 'train', 4, 73, archive=reference.archive)
    cached.enable_proposal_cache(tmp_path/'proposals')
    for epoch in (0, 1):
        reference.epoch = cached.epoch = epoch
        loader = DataLoader(cached, batch_size=2, num_workers=1,
                            multiprocessing_context='spawn', prefetch_factor=4)
        actual = list(loader)
        expected = list(DataLoader(reference, batch_size=2, num_workers=0))
        for left, right in zip(actual, expected):
            for key in left:
                torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
