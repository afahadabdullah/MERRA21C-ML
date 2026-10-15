"""Progress must count matching completed hours, not files or stale finalizers."""
import hashlib
import json
from pathlib import Path
import runpy
import subprocess

import pytest

CHECKER = runpy.run_path(str(Path(__file__).parents[1] / 'scripts/check_evaluation_v4_1.py'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def run(tmp_path, saved=False):
    entries = [dict(id=f'20260203_{hour:02d}30', time=f'2026-02-03T{hour:02d}:30:00') for hour in range(3)]
    manifest = dict(version=1, entries=entries, settings=dict(post='both', members=2, steps=64,
                    saved_cases=[entries[0]['id']] if saved else []), excluded=[{}])
    manifest['run_id'] = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    write(tmp_path / 'manifest.json', manifest)
    return manifest


def case(tmp_path, manifest, index):
    entry = manifest['entries'][index]
    record = dict(**entry, run_id=manifest['run_id'], sources={name: dict(members=1 if name in ('coarse', 'regression') else 2,
        fields={f: {} for f in CHECKER['FIELDS']}) for name in ('coarse', 'regression', 'raw', 'spectral')})
    if entry['id'] in manifest['settings']['saved_cases']:
        record['saved_fields'] = {}
        for method in ('raw', 'spectral'):
            relative = f'fields/{entry["id"]}_{method}.npz'
            path = tmp_path / relative; path.parent.mkdir(exist_ok=True); path.write_bytes(b'original')
            record['saved_fields'][method] = dict(path=relative, sha256=CHECKER['digest'](path))
    write(tmp_path / 'cases' / f'{entry["id"]}.json', record)
    return record


def test_partial_counts_and_shards_ignore_locks_and_cached_completion(tmp_path):
    m = run(tmp_path); case(tmp_path, m, 0)
    write(tmp_path / 'completion.json', dict(complete=True, expected=3, completed=3))
    lock = tmp_path / 'cases' / f'{m["entries"][1]["id"]}.lock'
    write(lock / 'owner.json', dict(slurm_job='123'))
    report = CHECKER['inspect'](tmp_path, shards=2)
    assert report['completed'] == 1 and report['remaining'] == 2
    assert report['percent'] == pytest.approx(100/3)
    assert len(report['locks']) == 1 and len(report['missing']) == 2
    assert report['status'] == 'partial' and report['aggregation'] == 'not_finished'
    assert report['shards'] == [dict(shard=0, completed=1, expected=2), dict(shard=1, completed=0, expected=1)]


def test_foreign_incomplete_corrupt_and_extra_records_are_not_completed(tmp_path):
    m = run(tmp_path); record = case(tmp_path, m, 0)
    record['run_id'] = 'foreign'; write(tmp_path/'cases'/f'{record["id"]}.json', record)
    record = case(tmp_path, m, 1); del record['sources']['raw']['fields']['q2m']
    write(tmp_path/'cases'/f'{record["id"]}.json', record)
    path = tmp_path/'cases'/f'{m["entries"][2]["id"]}.json';path.write_text('{broken')
    write(tmp_path/'cases'/'unexpected.json', {})
    report = CHECKER['inspect'](tmp_path)
    assert report['completed'] == 0 and len(report['invalid']) == 3
    assert report['unexpected'] == ['unexpected'] and report['status'] == 'needs_attention'
    assert CHECKER['main'](['--output', str(tmp_path), '--json']) == 1


def test_saved_fields_presence_and_optional_hashes(tmp_path):
    m = run(tmp_path, saved=True); record = case(tmp_path, m, 0)
    path = tmp_path / record['saved_fields']['raw']['path']
    path.write_bytes(b'changed')
    assert CHECKER['inspect'](tmp_path)['completed'] == 1
    assert CHECKER['inspect'](tmp_path, verify_files=True)['completed'] == 0
    path.unlink()
    assert CHECKER['inspect'](tmp_path)['completed'] == 0


@pytest.mark.parametrize('value', [[], None])
def test_non_object_case_json_is_reported_as_invalid(tmp_path, value):
    m = run(tmp_path)
    write(tmp_path/'cases'/f'{m["entries"][0]["id"]}.json', value)
    report = CHECKER['inspect'](tmp_path)
    assert report['completed'] == 0 and len(report['invalid']) == 1


def test_summary_requires_matching_run_and_current_case_records(tmp_path):
    m = run(tmp_path)
    for i in range(3): case(tmp_path, m, i)
    assert CHECKER['inspect'](tmp_path)['status'] == 'hours_saved'
    write(tmp_path/'summary.json', dict(run_id=m['run_id'], complete=True, hours=3))
    assert CHECKER['inspect'](tmp_path)['status'] == 'aggregated'
    (tmp_path/'cases'/f'{m["entries"][0]["id"]}.json').unlink()
    report = CHECKER['inspect'](tmp_path)
    assert report['status'] == 'partial'
    assert report['aggregation'] == 'saved_summary_conflicts_with_current_cases'


def test_not_initialized_and_manifest_tampering(tmp_path):
    report = CHECKER['inspect'](tmp_path)
    assert report['expected'] is None and report['percent'] is None
    m = run(tmp_path); m['settings']['members'] = 99;write(tmp_path/'manifest.json', m)
    with pytest.raises(ValueError, match='hash mismatch'): CHECKER['inspect'](tmp_path)


def test_scheduler_is_read_only_and_reports_states(monkeypatch):
    calls = []
    def fake(command, **kwargs):
        calls.append(command)
        assert kwargs['timeout'] == 15
        if command[0] == 'squeue':return subprocess.CompletedProcess(command, 0, '123_0|RUNNING|1:00|node\n', '')
        return subprocess.CompletedProcess(command, 0, '123_0|RUNNING|1:00\n123_1|TIMEOUT|12:00:00\n', '')
    monkeypatch.setattr(subprocess, 'run', fake)
    status = CHECKER['slurm_status']('123')
    assert [c[0] for c in calls] == ['squeue', 'sacct']
    assert status['queue_states'] == {'RUNNING': 1}
    assert status['accounting_states']['TIMEOUT'] == 1


def test_timing_separates_sampling_from_other_work_and_ignores_missing_values(tmp_path):
    m = run(tmp_path)
    for i in range(3):
        record = case(tmp_path, m, i)
        if i < 2:
            record.update(sampling_seconds=600, total_seconds=1800)
            write(tmp_path/'cases'/f'{record["id"]}.json', record)
    report = CHECKER['inspect'](tmp_path)
    assert report['completed'] == 3
    assert report['timing'] == dict(hours=2, mean_total_seconds=1800, mean_sampling_seconds=600,
                                     mean_other_seconds=1200, sampling_fraction=1/3)
