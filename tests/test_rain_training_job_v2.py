from pathlib import Path
import json
import os
import subprocess
import sys
import torch
import pytest
import yaml


@pytest.mark.parametrize('stage,epoch,expected_stage,expected_script', [
    ('regression', 0, 'regression', 'slurm_train_flow_v2.sh'),
    ('regression', 1, 'flow', 'slurm_train_flow_v2.sh'),
    ('flow', 0, 'flow', 'slurm_train_flow_v2.sh'),
    ('flow', 1, 'flow', 'slurm_test_best_model_v2.sh'),
])
def test_training_job_continuation_and_final_test(tmp_path, stage, epoch, expected_stage, expected_script):
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root/'configs/discover_rain_structure_v2.yaml').read_text())
    cfg['train'].update(output=str(tmp_path/'run_v2'), regression_epochs=2, flow_epochs=2)
    cfg['inference']['output'] = str(tmp_path/'predictions_v2')
    config = tmp_path/'config_v2.yaml'
    config.write_text(yaml.safe_dump(cfg))
    stage_dir = Path(cfg['train']['output'])/f'{stage}_v2'
    stage_dir.mkdir(parents=True)
    torch.save(dict(version='v2', stage=stage, epoch=epoch), stage_dir/'last_v2.pt')
    torch.save(dict(version='v2', stage=stage, epoch=epoch), stage_dir/'best_v2.pt')
    binaries = tmp_path/'bin'
    binaries.mkdir()
    python = binaries/'python'
    python.write_text(f'#!{sys.executable}\nimport os, sys\n'
                      'if sys.argv[1:2] == ["-c"] and "cuda.device_count" in sys.argv[2]:\n'
                      '    print(1)\n'
                      'else:\n'
                      f'    os.execv({sys.executable!r}, [{sys.executable!r}]+sys.argv[1:])\n')
    python.chmod(0o755)
    submit = binaries/'sbatch'
    submit.write_text(f'#!{sys.executable}\nimport os, sys, json\n'
                      'assert not any(os.environ.get(k) for k in ("SLURM_MEM_PER_CPU", "SLURM_MEM_PER_NODE", "SLURM_MEM_PER_GPU"))\n'
                      'with open(os.environ["SUBMISSION_CAPTURE"], "w") as f:\n'
                      '    json.dump({"args": sys.argv[1:], "env": dict(os.environ)}, f)\n'
                      'print("999")\n')
    submit.chmod(0o755)
    capture = tmp_path/'submission.json'
    env = dict(os.environ, PATH=f'{binaries}:{os.environ["PATH"]}', PROJECT_DIR=str(root),
               CONFIG=str(config), STAGE=stage, RESUME='', REGRESSION_CHECKPOINT='',
               TRAIN_FLOW_AFTER_REGRESSION='1', TEST_AFTER_TRAINING='1', SLURM_JOB_ID='123',
               SLURM_MEM_PER_CPU='2', SLURM_MEM_PER_NODE='3', SLURM_MEM_PER_GPU='32G',
               SUBMISSION_CAPTURE=str(capture))
    subprocess.run(['bash', '-c', '''
source() { :; }
conda() { :; }
srun() { printf '%s\n' "$*" >> "$SRUN_CAPTURE"; }
. scripts/slurm_train_flow_v2.sh
'''], cwd=root, env={**env, 'SRUN_CAPTURE': str(tmp_path/'srun.txt')}, text=True, capture_output=True, check=True)
    srun_calls = (tmp_path/'srun.txt').read_text().splitlines()
    assert srun_calls and all(call.startswith('--cpu-bind=none ') for call in srun_calls)
    submitted = json.loads(capture.read_text())
    assert submitted['args'][-1].endswith(expected_script)
    assert submitted['env']['STAGE'] == expected_stage
    if epoch == 0:
        assert submitted['env']['RESUME'] == str(stage_dir/'last_v2.pt')
        assert '--gres=gpu:1' in submitted['args']
    elif stage == 'regression':
        assert submitted['env']['REGRESSION_CHECKPOINT'] == str(stage_dir/'best_v2.pt')
        assert submitted['env']['RESUME'] == ''
    else:
        assert submitted['env']['CHECKPOINT'] == str(stage_dir/'best_v2.pt')
        assert submitted['env']['TIMESTAMPS'].split() == ['20260223_0530', '20260209_1530', '20260209_2030', '20260305_1230', '20260306_1830']
        assert submitted['env']['MEMBERS'] == '5'
        assert submitted['env']['COMPARE_NOISE_PADDING'] == '0'
