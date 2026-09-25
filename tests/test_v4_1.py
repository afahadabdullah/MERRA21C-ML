"""v4.1 packed pipeline: sample-exact equivalence with v4, restart safety, training."""
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
from merraflow.v4 import ArchiveV4, DatasetV4
from merraflow.v4_1 import (load_config as load_v41, validate_config, base_config, PackedArchive,
                            DatasetV41, EpochSampler, check_checkpoint)
from merraflow.packed_v4_1 import (prepare, finalize, plan, geometry, hour_path, load_manifest,
                                   candidates_v4_1, INDEX)
from merraflow.train_v4_1 import train
from merraflow.cli_v4_1 import reference_sample, compare, preflight, predict, benchmark
from test_v4 import setup  # noqa: F401  (module fixture: synthetic v4 archive)


@pytest.fixture(scope='module')
def packed_cfg(setup, tmp_path_factory):
    root = tmp_path_factory.mktemp('v41')
    cfg = load_v41('configs/discover_v4_1.yaml')
    for key in ('conditioning', 'data', 'model', 'inference'):
        cfg[key] = deepcopy(setup[key])
    cfg['patch'] = deepcopy(setup['patch'])
    # Synthetic: broad (16+2*4)*2 = 48 pooled to 16 -> factor 3.
    cfg['patch']['sampling_stride'] = 6
    # Small tiles so every crop spans several tiles and the grid edge.
    cfg['packed'].update(root=str(root/'packed'), tile=8, context_tile=4)
    cfg['train'] = dict(setup['train'], workers=0, val_workers=0, fd_cache_size=3, persistent_workers=True,
                        batch_size=2, accumulate=1, calibration_batches=2, output=str(root/'run_v41'))
    cfg['inference']['output'] = str(root/'predictions_v41')
    validate_config(cfg)
    archive = ArchiveV4(base_config(cfg), verify_files=False)
    months = sorted({e['time'][:7] for e in plan(cfg, archive)})
    for month in months:
        prepare(cfg, archive, month)
    finalize(cfg, archive)
    return cfg


def test_geometry_alignment_and_packed_size(packed_cfg):
    archive = ArchiveV4(base_config(packed_cfg), verify_files=False)
    geo = geometry(packed_cfg, archive.shape, len(archive.cm))
    assert geo['factor'] == 3
    yy, xx = candidates_v4_1(archive.shape, geo['size'], geo['sampling_stride'], geo['factor'])
    assert not (yy % 3).any() and not (xx % 3).any()
    h, w = archive.shape
    assert yy.max() > h-geo['size']-3 and xx.max() > w-geo['size']-3
    for entry in load_manifest(packed_cfg, archive)['entries']:
        assert hour_path(packed_cfg['packed']['root'], entry).stat().st_size == geo['file_bytes']
    # Production geometry: the numbers quoted in docs/v4_1.md.
    prod = load_v41('configs/discover_v4_1.yaml')
    g = geometry(prod, (1059, 1799), 6)
    assert g['factor'] == 6 and g['offset'] == 224 and g['pooled'] == [251, 374]
    assert g['candidates'] == 40*71
    assert 135e6 < g['file_bytes'] < 142e6


def test_samples_equal_v4_datasetv4(packed_cfg):
    """Same rng draws, same candidates -> identical tensors to v4's DatasetV4."""
    base = base_config(packed_cfg)
    packed = PackedArchive(packed_cfg)
    for split, samples in (('train', 40), ('val', 12)):
        fast = DatasetV41(packed_cfg, split, samples, 7, packed=packed)
        slow = DatasetV4(base, split, samples, 7)
        slow.yy, slow.xx, slow.coast = fast.yy, fast.xx, fast.coast
        slow.proposals = {}
        assert [e['id'] for e in slow.entries] == [e['id'] for e in fast.entries]
        for epoch in (0, 3):
            slow.epoch = epoch
            for i in range(samples):
                a, b = fast[epoch*samples+i], slow[i]
                assert set(a) == set(b)
                for key in b:
                    torch.testing.assert_close(a[key], b[key], rtol=0, atol=0, msg=key)


def test_edge_and_interior_origins_match_reference(packed_cfg):
    base = base_config(packed_cfg)
    archive = ArchiveV4(base)
    packed = PackedArchive(packed_cfg, archive)
    data = DatasetV41(packed_cfg, 'train', 1, 0, packed=packed)
    for entry in packed.entries('train')+packed.entries('val'):
        for y in (0, int(np.median(data.yy)), int(data.yy.max())):
            for x in (0, int(data.xx.max())):
                compare(packed.sample(entry, y, x), reference_sample(archive, entry, y, x, base))


def test_reference_sample_is_datasetv4(packed_cfg):
    base = base_config(packed_cfg)
    slow = DatasetV4(base, 'train', 4, 11)
    archive = slow.archive
    for i in range(4):
        rng = np.random.default_rng(np.random.SeedSequence([11, 0, i]))
        entry = slow.entries[rng.integers(len(slow.entries))]
        q = slow.proposal(entry)
        j = rng.choice(len(q), p=q)
        compare(reference_sample(archive, entry, int(slow.yy[j]), int(slow.xx[j]), base), slow[i])


def test_prepare_is_restart_safe_and_manifest_detects_changes(packed_cfg, tmp_path):
    archive = ArchiveV4(base_config(packed_cfg), verify_files=False)
    again = prepare(packed_cfg, archive)
    assert again['written'] == 0 and again['skipped'] == len(plan(packed_cfg, archive))
    load_manifest(packed_cfg, archive, verify_files=True)
    # A changed geometry is refused, not silently read.
    other = deepcopy(packed_cfg)
    other['packed']['tile'] = 16
    with pytest.raises(ValueError, match='different data, geometry'):
        load_manifest(other, archive)
    # A truncated hour is found by the full audit and repacked by prepare.
    entry = load_manifest(packed_cfg, archive)['entries'][0]
    path = hour_path(packed_cfg['packed']['root'], entry)
    original = path.read_bytes()
    path.write_bytes(original[:-8])
    with pytest.raises(ValueError, match='missing or truncated'):
        load_manifest(packed_cfg, archive, verify_files=True)
    assert prepare(packed_cfg, archive)['written'] == 1
    assert path.read_bytes() == original


def test_history_only_hours_are_never_supervised(packed_cfg):
    packed = PackedArchive(packed_cfg)
    entries = packed.manifest['entries']
    supervised = {e['id'] for e in entries if e['targets']}
    for e in entries:
        for h in e['history']:
            assert h in packed.by_id
    for split in ('train', 'val'):
        assert all(e['id'] in supervised for e in packed.entries(split))
    history_only = [e for e in entries if not e['targets']]
    if history_only:
        rest = packed.files.block(history_only[0], 'rest', 0, 4, 0, 4)
        assert np.isnan(rest[5:]).all() and np.isfinite(rest[:5]).all()


def test_sampler_worker_count_and_persistence_do_not_change_batches(packed_cfg):
    packed = PackedArchive(packed_cfg)
    data = DatasetV41(packed_cfg, 'train', 8, 3, packed=packed)
    def epoch_batches(workers, epoch, persistent):
        sampler = EpochSampler(8, 1, 2)
        sampler.set_epoch(epoch)
        kwargs = dict(num_workers=workers)
        if workers:
            kwargs.update(multiprocessing_context='spawn', persistent_workers=persistent)
        return [b['target'] for b in DataLoader(data, sampler=sampler, batch_size=2, **kwargs)]
    for epoch in (0, 2):
        a, b = epoch_batches(0, epoch, False), epoch_batches(2, epoch, True)
        assert len(a) == 2
        for x, y in zip(a, b):
            torch.testing.assert_close(x, y, rtol=0, atol=0)
    assert not torch.equal(epoch_batches(0, 0, False)[0], epoch_batches(0, 1, False)[0])
    # Ranks see disjoint indices.
    ranks = [set(EpochSampler(8, r, 2)) for r in range(2)]
    assert not ranks[0] & ranks[1] and len(ranks[0] | ranks[1]) == 8


def test_training_resume_checkpoint_and_inference(packed_cfg, tmp_path, capsys):
    cfg = deepcopy(packed_cfg)
    cfg['train'].update(output=str(tmp_path/'run'), epochs=5, time_limit_hours=.00001)
    path = train(cfg)
    first = torch.load(path, weights_only=True)
    assert first['version'] == 'v4.1' and first['epoch'] == 0 and first['flow_scale'][0, 1, 0, 0] == 1
    assert first['history'][0]['samples_per_s'] > 0
    cfg['train']['time_limit_hours'] = None
    cfg['train'].update(workers=1, val_workers=1)  # persistent spawn workers on resume
    capsys.readouterr()
    train(cfg, resume=path)
    log = capsys.readouterr().out
    assert 'using saved calibration' in log and 'Calibration starting' not in log
    saved = torch.load(path, weights_only=True)
    assert saved['epoch'] == 4 and len(saved['history']) == 5
    folder = path.parent/'validation_plots'/'epoch_0005'
    for name in ('rain_patch_1.png', 'states_patch_1.png', 'diagnostics.png', 'history.png', 'metrics.json'):
        assert (folder/name).exists(), name
    assert (path.parent/'best_v4_1.pt').exists()
    kept = list((path.parent/'checkpoints').glob('epoch_0005_crps*_v4_1.pt'))
    assert len(kept) == 1 and f'{saved["history"][4]["crps"]:.4f}' in kept[0].name
    assert not list((path.parent/'checkpoints').glob('epoch_0004_*.pt'))
    chosen = json.loads((path.parent/'preview_patches.json').read_text())
    assert chosen and chosen[0]['truth_rain_mean'] >= max(c['truth_rain_mean'] for c in chosen)
    metrics = json.loads((folder/'metrics.json').read_text())
    assert all(f'{n}_crps' in metrics for n in ('t2m', 'precip', 'q2m'))
    (folder/'rain_patch_1.png').unlink()  # recovered on an idempotent resume
    train(cfg, resume=path)
    assert (folder/'rain_patch_1.png').exists()
    archive = ArchiveV4(base_config(cfg))
    packed = PackedArchive(cfg, archive)
    check_checkpoint(saved, cfg, archive, packed.fingerprint, world=1)
    with pytest.raises(ValueError, match='packed archive fingerprint'):
        check_checkpoint(saved, cfg, archive, 'other', world=1)
    changed = deepcopy(cfg)
    changed['train']['batch_size'] = 4
    with pytest.raises(ValueError, match='Exact resume'):
        check_checkpoint(saved, changed, archive, packed.fingerprint, world=1)
    cfg['inference'].update(output=str(tmp_path/'predictions'), members=1)
    files = list(predict(cfg, path, 'test', 1).glob('*.nc'))
    assert len(files) == 1


def test_preflight_and_benchmark(packed_cfg, tmp_path, capsys):
    cfg = deepcopy(packed_cfg)
    cfg['train']['output'] = str(tmp_path/'fresh')
    preflight(cfg, full=True)
    assert 'matches v4' in capsys.readouterr().out
    result = benchmark(cfg, samples=4, workers=0, batches=2, compare_v4=2)
    assert result['loader_samples_per_s_per_rank'] > 0


def test_validation_schedule_and_resume_safety(packed_cfg):
    from merraflow.v4_1 import validation_due, check_checkpoint
    from merraflow.v4 import TARGETS
    tr = dict(validation_interval=5, validation_interval_late=2, schedule_switch_epoch=20, epochs=250)
    due = [e for e in range(1, 41) if validation_due(e, tr)]
    assert due == [5, 10, 15, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40]
    assert validation_due(250, tr)  # always on the final epoch
    plain = dict(validation_interval=5, epochs=250)
    assert [e for e in range(1, 21) if validation_due(e, plain)] == [5, 10, 15, 20]
    # Changing cadence (and adding the schedule keys) must not break exact resume.
    base = deepcopy(packed_cfg)
    archive = ArchiveV4(base_config(base))
    packed = PackedArchive(base, archive)
    saved = dict(version='v4.1', targets=list(TARGETS),
                 fingerprint=archive.index['fingerprint'], stats=archive.stats,
                 hourly_fingerprint=archive.hourly_fingerprint, humidity_fingerprint=archive.humidity_fingerprint,
                 packed_fingerprint=packed.fingerprint, world_size=1, config=deepcopy(base))
    changed = deepcopy(base)
    changed['train'].update(validation_interval=5, validation_interval_late=2, schedule_switch_epoch=20)
    check_checkpoint(saved, changed, archive, packed.fingerprint, world=1)  # no raise


def test_config_rejects_misaligned_stride(packed_cfg):
    bad = deepcopy(packed_cfg)
    bad['train'] = dict(bad['train'], schedule_switch_epoch=20)
    bad['train'].pop('validation_interval_late', None)
    with pytest.raises(ValueError, match='together, or neither'):
        validate_config(bad)
    bad = deepcopy(packed_cfg)
    bad['patch']['sampling_stride'] = 8
    with pytest.raises(ValueError, match='pooling factor'):
        validate_config(bad)
    bad = deepcopy(packed_cfg)
    bad['packed']['splits'] = ['train']
    with pytest.raises(ValueError, match='train and val'):
        validate_config(bad)


def test_two_process_training(packed_cfg, tmp_path):
    cfg = deepcopy(packed_cfg)
    cfg['train'].update(epochs=1, workers=1, val_workers=1, output=str(tmp_path/'ddp'),
                        validation_interval=1, validation_patches=3, calibration_batches=1)
    cfg['patch']['samples_per_epoch'] = 8
    path = tmp_path/'config.yaml'
    path.write_text(yaml.safe_dump(cfg))
    env = dict(os.environ, PYTHONPATH=str(Path('src').resolve()), OMP_NUM_THREADS='1', MPLBACKEND='Agg',
               MPLCONFIGDIR=str(tmp_path/'mpl'), GLOO_SOCKET_IFNAME='lo0' if sys.platform == 'darwin' else 'lo')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    result = subprocess.run([sys.executable, '-m', 'torch.distributed.run', '--master-addr=127.0.0.1',
                             f'--master-port={port}', '--nnodes=1', '--nproc-per-node=2',
                             '-m', 'merraflow.cli_v4_1', 'train', '--config', str(path)],
                            env=env, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stdout+result.stderr
    assert 'batch 2/2' in result.stdout
    saved = torch.load(Path(cfg['train']['output'])/'last_v4_1.pt', weights_only=True)
    assert saved['world_size'] == 2 and len(saved['rng']) == 2


def test_submission_packs_months_then_finalizes_then_trains(packed_cfg, tmp_path):
    cfg = deepcopy(packed_cfg)
    path = tmp_path/'config.yaml'
    path.write_text(yaml.safe_dump(cfg))
    binary = tmp_path/'bin'
    binary.mkdir()
    capture = tmp_path/'calls'
    sbatch = binary/'sbatch'
    sbatch.write_text('#!/bin/bash\nprintf "%s\\n" "$*" >> "$CAPTURE"\nwc -l < "$CAPTURE" | tr -d " "\n')
    sbatch.chmod(0o755)
    env = dict(os.environ, CONFIG=str(path), PYTHON_BIN=sys.executable, CAPTURE=str(capture),
               PATH=f'{binary}:'+os.environ['PATH'], PYTHONPATH=str(Path('src').resolve()), MAX_PARALLEL='3')
    result = subprocess.run(['bash', 'scripts/submit_prepare_v4_1.sh'], env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout+result.stderr
    calls = capture.read_text().splitlines()
    assert len(calls) == 3
    assert '--array=0-' in calls[0] and '%3' in calls[0]
    assert '--dependency=afterok:1' in calls[1] and '--dependency=afterok:2' in calls[2]
    assert calls[2].endswith('scripts/slurm_train_v4_1.sh')
    train_script = Path('scripts/slurm_train_v4_1.sh').read_text()
    assert 'cli_v4_1 preflight' not in train_script
    assert '--cpus-per-gpu=12' in train_script
