"""Exercise the real submission wrapper with a fake scheduler, without jobs."""
from pathlib import Path
import json
import os
import subprocess
import sys

import pytest
import torch
import yaml


@pytest.mark.parametrize('edges', [False, True])
@pytest.mark.parametrize('gpus', [2, 4])
def test_submission_freezes_source_and_orders_cache_before_training(tmp_path, edges, gpus):
    root = Path(__file__).resolve().parents[1]
    name = 'edges' if edges else 'rollout'
    cfg = yaml.safe_load((root/f'configs/discover_rain_{name}_v2.yaml').read_text())
    cfg['train']['output'] = str(tmp_path/'run_v2')
    config = tmp_path/'config_v2.yaml'
    config.write_text(yaml.safe_dump(cfg))
    source = tmp_path/'source_v2.pt'
    old_cfg = yaml.safe_load((root/'configs/discover_rain_structure_v2.yaml').read_text())
    torch.save(dict(version='v2', stage='flow', epoch=24, config=old_cfg), source)
    original = source.read_bytes()
    binaries = tmp_path/'bin'
    binaries.mkdir()
    scheduler = binaries/'sbatch'
    scheduler.write_text(f'#!{sys.executable}\nimport os, sys, json\n'
                        'assert not any(os.environ.get(k) for k in ("SLURM_MEM_PER_CPU", "SLURM_MEM_PER_NODE", "SLURM_MEM_PER_GPU"))\n'
                        'with open(os.environ["SUBMISSION_CAPTURE"], "a") as f:\n'
                        '    f.write(json.dumps({"args": sys.argv[1:], "env": dict(os.environ)})+"\\n")\n'
                        'print("888" if sys.argv[-1].endswith("slurm_cache_rain_edges_v2.sh") else "999")\n')
    scheduler.chmod(0o755)
    capture = tmp_path/'submissions.jsonl'
    env = dict(os.environ, PATH=f'{binaries}:{os.environ["PATH"]}', PYTHON_BIN=sys.executable,
               CONFIG=str(config), SOURCE_CHECKPOINT=str(source), SUBMISSION_CAPTURE=str(capture),
               GPUS=str(gpus),
               CACHE_JOB_ID='', SLURM_MEM_PER_CPU='1', SLURM_MEM_PER_NODE='2', SLURM_MEM_PER_GPU='3',
               TRAIN_BATCH_SIZE_OVERRIDE='100', TRAIN_WORKERS_OVERRIDE='100')
    subprocess.run(['bash', 'scripts/submit_rain_rollout_v2.sh'], cwd=root, env=env,
                   text=True, capture_output=True, check=True)
    calls = [json.loads(line) for line in capture.read_text().splitlines()]
    assert len(calls) == (2 if edges else 1)
    training = calls[-1]
    assert f'--gres=gpu:{gpus}' in training['args']
    frozen_config = yaml.safe_load(Path(training['env']['CONFIG']).read_text())
    tr = frozen_config['train']
    assert tr['reference_world_size'] == gpus
    assert tr['batch_size']*tr['accumulate']*gpus == 16
    assert tr['batch_size']*tr['val_batches']*gpus == 256
    assert tr['batch_size']*tr['generated_validation']['batches']*gpus == 64
    assert ('--dependency=afterok:888' in training['args']) == edges
    assert training['env']['STAGE'] == 'flow' and training['env']['RESUME'] == ''
    assert 'TRAIN_BATCH_SIZE_OVERRIDE' not in training['env']
    assert 'TRAIN_WORKERS_OVERRIDE' not in training['env']
    frozen = Path(training['env']['INITIALIZE_FLOW'])
    assert frozen.read_bytes() == original and source.read_bytes() == original
    assert training['env']['TRAIN_PREFLIGHT'] == training['env']['PREFER_SKILL_CHECKPOINT'] == '1'
    retry = subprocess.run(['bash', 'scripts/submit_rain_rollout_v2.sh'], cwd=root, env=env,
                           text=True, capture_output=True)
    assert retry.returncode != 0 and 'do not submit a duplicate' in retry.stderr
    assert len(capture.read_text().splitlines()) == len(calls)
