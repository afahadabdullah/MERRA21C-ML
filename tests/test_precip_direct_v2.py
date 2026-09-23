from copy import deepcopy
from pathlib import Path
import json
import os
import socket
import subprocess
import sys
import numpy as np
import pytest
import torch
import yaml
from torch.utils.data import DataLoader
from merraflow.config_v2 import load_config_v2
from merraflow.synthetic_v2 import make_synthetic_v2
from merraflow.prepare_v2 import prepare
from merraflow.model_v2 import UNetV2
from merraflow.precip_direct_v2 import (DirectPrecipDataset, load_config, make_model, objective,
    encode_rain, decode_rain, FrozenRegression, regression_bundle, initialize_from_v2)
from merraflow.train_precip_direct_v2 import train
from merraflow.inference_precip_direct_v2 import sample_frame, predict


@pytest.fixture(scope='module')
def setup(tmp_path_factory):
    torch.set_num_threads(1)
    root = tmp_path_factory.mktemp('direct')
    original = load_config_v2(make_synthetic_v2(root, load_config_v2('configs/discover_v2.yaml')))
    prepare(original)
    cfg = load_config('configs/discover_precip_direct_v2.yaml')
    cfg['data']['prepared'] = original['data']['prepared']
    cfg['patch'] = deepcopy(original['patch'])
    cfg['model'] = deepcopy(original['model'])
    cfg['train'].update(device='cpu', workers=0, precision='fp32', batch_size=2, accumulate=2,
        epochs=5, validation_patches=4, validation_members=2, validation_steps=2,
        validation_plot_samples=1, time_limit_hours=None, output=str(root/'direct_v2'))
    cfg['inference'].update(steps=2, members=2, output=str(root/'predictions'))
    archive = DirectPrecipDataset(cfg, 'train', 1, 1).archive
    regression = UNetV2(archive.index['condition_channels'], **original['model'])
    source = root/'original_regression.pt'
    torch.save(dict(version='v2', stage='regression', epoch=29, config=original,
                    stats=archive.stats, fingerprint=archive.index['fingerprint'],
                    ema=regression.state_dict(), initialization=None), source)
    cfg['conditioning']['checkpoint'] = str(source)
    return cfg


def test_full_target_and_frozen_condition(setup):
    data = DirectPrecipDataset(setup, 'train', 2, 3)
    # Prove the dataset never reads the archived residual target.
    original_array = data.archive.array
    def array(entry, name):
        assert name != 'residual'
        return original_array(entry, name)
    data.archive.array = array
    b = next(iter(DataLoader(data, batch_size=2)))
    np.testing.assert_allclose(b['target'], encode_rain(b['truth'].numpy(), data.rain_scale))
    np.testing.assert_allclose(decode_rain(b['target'].numpy(), data.rain_scale), b['truth'], rtol=1e-6)
    conditioner = FrozenRegression(regression_bundle(setup['conditioning']['checkpoint'], data.archive),
                                    data.archive.index['condition_channels'], data.archive.stats)
    b['mean'] = conditioner(b)
    assert b['target'].shape[1] == b['mean'].shape[1] == 1
    assert not b['mean'].requires_grad
    frozen = deepcopy(conditioner.state_dict())
    model = make_model(data.archive.index['condition_channels'], setup)
    objective(model, b).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert all(p.grad is None for p in conditioner.parameters())
    for key, value in frozen.items():
        torch.testing.assert_close(value, conditioner.state_dict()[key], rtol=0, atol=0)


def test_objective_does_not_subtract_condition(setup):
    b = next(iter(DataLoader(DirectPrecipDataset(setup, 'val', 2, 3), batch_size=2)))
    class Oracle(torch.nn.Module):
        def forward(self, xt, time, condition, context, mean=None):
            return (b['target']-xt)/(1-time[:, None, None, None])
    b['mean'] = torch.ones_like(b['target'])*50  # Deliberately unrelated regression input.
    loss = objective(Oracle(), b, torch.Generator().manual_seed(4))
    assert loss < 1e-9


def test_frame_has_no_coarse_or_regression_addback(setup):
    a = DirectPrecipDataset(setup, 'val', 1, 3).archive
    class ZeroVelocity(torch.nn.Module):
        def forward(self, x, *args):
            return torch.zeros_like(x)
    # Zero velocity leaves the initial noise unchanged, even with huge mean input.
    class LargeCondition(torch.nn.Module):
        def forward(self, batch):
            return torch.ones_like(batch['coarse'])*100
    cfg = deepcopy(setup)
    cfg['inference']['steps'] = 1
    entry = next(e for e in a.index['entries'] if e['split'] == 'val')
    actual = sample_frame(ZeroVelocity(), LargeCondition(), a, entry, cfg, torch.device('cpu'), 19)
    halo = cfg['patch']['halo']
    h, w = a.shape
    noise = np.random.default_rng(19).standard_normal((1, h+2*halo, w+2*halo)).astype('float32')
    expected = decode_rain(noise[0, halo:halo+h, halo:halo+w], a.stats['precip_log_scale'])
    np.testing.assert_array_equal(actual, expected)


def test_weight_transfer_and_reject_finetuned(setup, tmp_path):
    a = DirectPrecipDataset(setup, 'train', 1, 3).archive
    model = make_model(a.index['condition_channels'], setup)
    fresh_input = model.input.weight.detach().clone()
    report = initialize_from_v2(model, setup['conditioning']['checkpoint'], setup, a)
    assert report['copied_layers'] and report['reset_layers']
    torch.testing.assert_close(model.input.weight, fresh_input, rtol=0, atol=0)
    source = torch.load(setup['conditioning']['checkpoint'], weights_only=True)
    source['initialization'] = {'checkpoint': 'older'}
    path = tmp_path/'finetuned.pt'
    torch.save(source, path)
    with pytest.raises(ValueError, match='fine-tuned'):
        regression_bundle(path, a)


def test_training_plots_resume_and_predict(setup, tmp_path):
    cfg = deepcopy(setup)
    cfg['train']['output'] = str(tmp_path/'trained_direct_v2')
    cfg['inference']['output'] = str(tmp_path/'prediction')
    # End before first generated validation; resume must not require best yet.
    cfg['train']['time_limit_hours'] = 1e-6
    checkpoint = train(cfg)
    assert torch.load(checkpoint, weights_only=True)['epoch'] == 0
    assert not (checkpoint.parent/'best_direct_v2.pt').exists()
    cfg['train']['time_limit_hours'] = None
    train(cfg, resume=checkpoint)
    saved = torch.load(checkpoint, weights_only=True)
    assert saved['epoch'] == 4 and len(saved['history']) == 5
    assert 'crps' in saved['history'][-1]
    plot = checkpoint.parent/'validation_plots/epoch_0005/fields.png'
    assert plot.exists()
    assert len(list((checkpoint.parent/'validation_plots').iterdir())) == 1
    plot.unlink()  # Simulate interruption after checkpointing but during plotting.
    train(cfg, resume=checkpoint)
    assert plot.exists()
    assert len(torch.load(checkpoint, weights_only=True)['history']) == 5
    prediction = predict(cfg, checkpoint, limit=1)
    assert len(list(prediction.glob('*.nc'))) == 2
    with pytest.raises(FileExistsError):
        predict(cfg, checkpoint, limit=1)


def test_four_process_direct_training(setup, tmp_path):
    cfg = deepcopy(setup)
    cfg['train']['output'] = str(tmp_path/'ddp_direct_v2')
    cfg['patch']['samples_per_epoch'] = 16
    cfg['train']['validation_patches'] = 3  # One rank empty: no duplicate validation samples.
    path = tmp_path/'config.yaml'
    path.write_text(yaml.safe_dump(cfg))
    env = dict(os.environ, OMP_NUM_THREADS='1', MPLCONFIGDIR=str(tmp_path/'mpl'),
               GLOO_SOCKET_IFNAME='lo0' if sys.platform == 'darwin' else 'lo')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    command = [sys.executable, '-m', 'torch.distributed.run', '--master-addr=127.0.0.1',
               f'--master-port={port}', '--nnodes=1', '--nproc-per-node=4',
               '-m', 'merraflow.cli_precip_direct_v2', 'train', '--config', str(path)]
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout+result.stderr
    out = Path(cfg['train']['output'])
    saved = torch.load(out/'last_direct_v2.pt', weights_only=True)
    assert saved['world_size'] == 4 and len(saved['rng']) == 4
    assert saved['epoch'] == 4
    assert not torch.equal(saved['rng'][0]['cpu'], saved['rng'][1]['cpu'])
    assert (out/'validation_plots/epoch_0005/fields.png').exists()
    source = torch.load(cfg['conditioning']['checkpoint'], weights_only=True)
    for key, value in source['ema'].items():
        torch.testing.assert_close(value, saved['regression_condition']['weights'][key], rtol=0, atol=0)
    result = subprocess.run(command+['--resume', str(out/'last_direct_v2.pt')],
                            env=env, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout+result.stderr


def test_submit_only_flow(setup, tmp_path):
    cfg = deepcopy(setup)
    cfg['train']['output'] = str(tmp_path/'submission')
    path = tmp_path/'config.yaml'
    path.write_text(yaml.safe_dump(cfg))
    binary = tmp_path/'bin'
    binary.mkdir()
    capture = tmp_path/'calls'
    sbatch = binary/'sbatch'
    sbatch.write_text('#!/bin/bash\nprintf "%s\\n" "$*" >> "$CAPTURE"\necho 42\n')
    sbatch.chmod(0o755)
    env = dict(os.environ, CONFIG=str(path), PYTHON_BIN=sys.executable, CAPTURE=str(capture),
               PATH=f'{binary}:'+os.environ['PATH'], RESUME='', INITIALIZE_V2='')
    result = subprocess.run(['bash', 'scripts/submit_precip_direct_v2.sh'], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout+result.stderr
    calls = capture.read_text().splitlines()
    assert len(calls) == 1 and calls[0].endswith('scripts/slurm_train_precip_direct_v2.sh')
    script = Path('scripts/slurm_train_precip_direct_v2.sh').read_text()
    assert '#SBATCH --gres=gpu:4' in script and '--nproc-per-node=4' in script


def test_rainy_validation_case_report(setup, tmp_path):
    from merraflow.evaluate_wet_precip_direct_v2 import evaluate, select_wet_cases
    cfg = deepcopy(setup)
    archive = DirectPrecipDataset(cfg, 'val', 1, 3).archive
    selected, selection = select_wet_cases(archive, cfg, 'val', scan_hours=2,
                                           cases=1, min_wet_fraction=.001)
    assert selected[0]['entry']['split'] == 'val'
    assert selected[0]['selection_wet_fraction'] >= .001
    assert selection['qualifying_hours'] >= 1
    checkpoint = tmp_path/'snapshot.pt'
    torch.save(dict(version='v2_precip_direct', targets=['precip'], config=cfg,
                    fingerprint=archive.index['fingerprint'], stats=archive.stats,
                    epoch=0, ema=make_model(archive.index['condition_channels'], cfg).state_dict(),
                    regression_condition=regression_bundle(cfg['conditioning']['checkpoint'], archive)), checkpoint)
    output = tmp_path/'wet_cases'
    evaluate(cfg, checkpoint, output, scan_hours=2, cases=1,
             min_wet_fraction=.001, members=2, steps=1)
    report = json.loads((output/'report.json').read_text())
    assert report['selection']['split'] == 'val' and report['epoch'] == 1
    assert report['cases'][0]['truth_wet_fraction'] >= .001
    assert 'crps_mm_h' in report['cases'][0] and 'regression_mae_mm_h' in report['cases'][0]
    assert len(list(output.glob('case_*.png'))) == len(list(output.glob('case_*.npz'))) == 1


def test_wet_evaluation_submission_snapshots_current_checkpoint(setup, tmp_path):
    source = tmp_path/'last_direct_v2.pt'
    torch.save(dict(version='v2_precip_direct', targets=['precip'], epoch=9,
                    config=setup), source)
    binary = tmp_path/'bin'
    binary.mkdir()
    capture = tmp_path/'submission.json'
    sbatch = binary/'sbatch'
    sbatch.write_text(f'#!{sys.executable}\nimport json, os, sys\n'
        'with open(os.environ["CAPTURE"], "w") as f:\n'
        '    json.dump({"args": sys.argv[1:], "checkpoint": os.environ["CHECKPOINT"], '
        '"config": os.environ["CONFIG"], "output": os.environ["OUTPUT"]}, f)\n'
        'print("24680")\n')
    sbatch.chmod(0o755)
    env = dict(os.environ, CHECKPOINT=str(source), EVALUATION_ROOT=str(tmp_path/'evals'),
               PYTHON_BIN=sys.executable, PATH=f'{binary}:'+os.environ['PATH'], CAPTURE=str(capture))
    result = subprocess.run(['bash', 'scripts/submit_wet_eval_precip_direct_v2.sh'],
                            env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout+result.stderr
    submitted = json.loads(capture.read_text())
    assert submitted['args'][-1] == 'scripts/slurm_wet_eval_precip_direct_v2.sh'
    snapshot = Path(submitted['checkpoint'])
    assert snapshot.exists() and snapshot != source
    assert torch.load(snapshot, weights_only=True)['epoch'] == 9
    assert Path(submitted['config']).exists()
    assert submitted['output'].endswith('/results')


@pytest.mark.parametrize('completed', [False, True])
def test_job_continues_only_unfinished_flow(setup, tmp_path, completed):
    cfg = deepcopy(setup)
    out = tmp_path/'run'
    out.mkdir()
    cfg['train']['output'] = str(out)
    config = tmp_path/'cfg.yaml'
    config.write_text(yaml.safe_dump(cfg))
    torch.save(dict(epoch=cfg['train']['epochs']-1 if completed else 0), out/'last_direct_v2.pt')
    binaries = tmp_path/'bin'
    binaries.mkdir()
    python = binaries/'python'
    python.write_text(f'#!{sys.executable}\nimport os, sys\n'
        'if sys.argv[1:2] == ["-c"] and "cuda.device_count" in sys.argv[2]:\n'
        '    print(4)\nelse:\n'
        f'    os.execv({sys.executable!r}, [{sys.executable!r}]+sys.argv[1:])\n')
    python.chmod(0o755)
    submit = binaries/'sbatch'
    submit.write_text(f'#!{sys.executable}\nimport os, sys, json\n'
        'assert not any(os.environ.get(k) for k in ("SLURM_MEM_PER_CPU", "SLURM_MEM_PER_NODE", "SLURM_MEM_PER_GPU"))\n'
        'with open(os.environ["CAPTURE"], "w") as f:\n'
        '    json.dump({"args": sys.argv[1:], "resume": os.environ["RESUME"], "init": os.environ["INITIALIZE_V2"]}, f)\n'
        'print(42)\n')
    submit.chmod(0o755)
    capture, srun_capture = tmp_path/'submit.json', tmp_path/'srun'
    env = dict(os.environ, CONFIG=str(config), PROJECT_DIR=str(Path.cwd()), SLURM_JOB_ID='123',
               PATH=f'{binaries}:'+os.environ['PATH'], CAPTURE=str(capture), SRUN_CAPTURE=str(srun_capture),
               RESUME='', INITIALIZE_V2='', REGRESSION_CHECKPOINT='',
               SLURM_MEM_PER_CPU='1', SLURM_MEM_PER_NODE='2', SLURM_MEM_PER_GPU='32G')
    result = subprocess.run(['bash', '-c', '''
source() { :; }
conda() { :; }
srun() { printf '%s\n' "$*" >> "$SRUN_CAPTURE"; }
. scripts/slurm_train_precip_direct_v2.sh
'''], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout+result.stderr
    assert '--nproc-per-node=4' in srun_capture.read_text()
    assert capture.exists() != completed
    if not completed:
        record = json.loads(capture.read_text())
        assert record['resume'] == str(out/'last_direct_v2.pt') and record['init'] == ''
        assert '--dependency=afterok:123' in record['args']
