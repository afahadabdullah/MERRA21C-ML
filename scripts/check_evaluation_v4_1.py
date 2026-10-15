#!/usr/bin/env python3
"""Read-only MERRA-Flow full-test progress checker; Python standard library only.

Kept outside src/merraflow so copying this checker into a running checkout does
not change the evaluator's frozen source hash. No model, inference, or metrics
are loaded. Optional --verify-files hashes saved example arrays (potentially GB).
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time

FIELDS = {'t2m', 'precip', 'ps', 'u10m', 'v10m', 'q2m', 'wind_speed'}


def reject_constant(value):
    raise ValueError(f'Nonfinite JSON value: {value}')


def read_json(path):
    return json.loads(Path(path).read_text(), parse_constant=reject_constant)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def validate_case(path, manifest, entry, verify_files=False):
    record = read_json(path)
    if not isinstance(record, dict):
        raise ValueError('Case JSON must be an object')
    if any(record.get(k) != v for k, v in dict(run_id=manifest['run_id'], **entry).items()):
        raise ValueError('Case identity/run mismatch')
    expected = {'coarse', 'regression', 'raw'}
    if manifest['settings']['post'] == 'both':
        expected.add('spectral')
    if set(record['sources']) != expected:
        raise ValueError('Missing or unexpected evaluation methods')
    for name, source in record['sources'].items():
        members = 1 if name in ('coarse', 'regression') else manifest['settings']['members']
        if set(source['fields']) != FIELDS or source['members'] != members:
            raise ValueError(f'Incomplete variables/member count: {name}')
    if entry['id'] in manifest['settings'].get('saved_cases', []):
        methods = {'raw'} | ({'spectral'} if 'spectral' in expected else set())
        if set(record.get('saved_fields', {})) != methods:
            raise ValueError('Missing saved-field records')
        for method, item in record['saved_fields'].items():
            relative = f'fields/{entry["id"]}_{method}.npz'
            target = path.parent.parent / relative
            if item['path'] != relative or not target.is_file() or target.stat().st_size == 0:
                raise ValueError(f'Missing saved field: {relative}')
            if verify_files and digest(target) != item['sha256']:
                raise ValueError(f'Saved field hash mismatch: {relative}')
    return record


def slurm_status(job_id):
    """Read scheduler state only; never submit, cancel, resume, or clear locks."""
    result = {}
    for name, command in (
        ('queue', ['squeue', '-h', '-r', '-j', job_id, '-o', '%i|%T|%M|%R']),
        ('accounting', ['sacct', '-n', '-X', '-j', job_id, '--format=JobIDRaw,State,Elapsed', '-P']),
    ):
        try:
            run = subprocess.run(command, capture_output=True, text=True, timeout=15, check=False)
            result[name] = run.stdout.strip().splitlines()
            result[name + '_states'] = dict(Counter(line.split('|')[1].split()[0].rstrip('+')
                for line in result[name] if len(line.split('|')) > 1 and line.split('|')[1].strip()))
            if run.returncode:
                result[name + '_error'] = run.stderr.strip() or f'Exit {run.returncode}'
        except (OSError, subprocess.TimeoutExpired) as exc:
            result[name + '_error'] = str(exc)
    return result


def inspect(output, shards=None, verify_files=False, job_id=None, now=None):
    root = Path(output)
    now = time.time() if now is None else now
    report = dict(output=str(root), checked_utc=datetime.fromtimestamp(now, timezone.utc).isoformat(),
                  status='not_initialized', expected=None, completed=0, percent=None,
                  validation='case JSON and saved-field existence' + (' and SHA256' if verify_files else '; field hashes not checked'))
    if not (root / 'manifest.json').is_file():
        report['note'] = 'No manifest.json: initialization has not finished, or this is the wrong output directory.'
        if job_id:
            report['slurm'] = slurm_status(job_id)
        return report
    manifest = read_json(root / 'manifest.json')
    if not isinstance(manifest, dict):
        raise ValueError('Manifest JSON must be an object')
    body = {k: v for k, v in manifest.items() if k != 'run_id'}
    actual_hash = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    if manifest['version'] != 1 or actual_hash != manifest['run_id']:
        raise ValueError('Manifest version/hash mismatch')
    entries = manifest['entries']
    ids = [entry['id'] for entry in entries]
    if not ids or len(set(ids)) != len(ids) or any(not re.fullmatch(r'[A-Za-z0-9_-]+', key) for key in ids):
        raise ValueError('Manifest has empty, duplicate, or invalid hour IDs')
    expected = set(ids)
    paths = {p.stem: p for p in (root / 'cases').glob('*.json')}
    good, invalid, mtimes = set(), {}, []
    for entry in entries:
        path = paths.get(entry['id'])
        if path is None:
            continue
        try:
            validate_case(path, manifest, entry, verify_files)
            mtimes.append(path.stat().st_mtime)
            good.add(entry['id'])
        except (OSError, ValueError, KeyError, TypeError) as exc:
            invalid[entry['id']] = str(exc)
    missing = expected - paths.keys()
    locks = []
    for lock in sorted((root / 'cases').glob('*.lock')):
        try:
            stamp = lock.stat().st_mtime
        except FileNotFoundError:  # A live worker may have just removed its lock.
            continue
        try:
            owner = read_json(lock / 'owner.json')
        except (OSError, ValueError):
            owner = {}
        locks.append(dict(id=lock.stem, expected=lock.stem in expected, completed=lock.stem in good, owner=owner,
                          age_seconds=max(0, now - stamp)))
    extra = sorted(paths.keys() - expected)
    ready = len(good) == len(expected) and not extra
    aggregation = 'not_finished'
    try:
        summary = read_json(root / 'summary.json')
        if summary.get('run_id') == manifest['run_id'] and summary.get('complete') and summary.get('hours') == len(expected):
            aggregation = 'complete' if ready else 'saved_summary_conflicts_with_current_cases'
        else:
            aggregation = 'summary_does_not_match_run'
    except FileNotFoundError:
        pass
    except (OSError, ValueError, AttributeError):
        aggregation = 'unreadable_summary'
    report.update(run_id=manifest['run_id'], split=manifest['settings'].get('split'),
                  status='needs_attention' if invalid or extra else ('aggregated' if ready and aggregation == 'complete' else
                         'hours_saved' if ready else 'partial'),
                  expected=len(expected), completed=len(good), percent=100 * len(good) / len(expected),
                  remaining=len(expected) - len(good), missing=sorted(missing), invalid=invalid, unexpected=extra,
                  excluded=len(manifest.get('excluded', [])), locks=locks, aggregation=aggregation,
                  newest_case_age_seconds=max(0, now - max(mtimes)) if mtimes else None,
                  completed_last_hour=sum(now - 3600 <= t <= now for t in mtimes),
                  members=manifest['settings']['members'], steps=manifest['settings'].get('steps'),
                  post=manifest['settings']['post'])
    if shards:
        report['shards'] = [dict(shard=i, completed=sum(e['id'] in good for e in entries[i::shards]),
                                 expected=len(entries[i::shards])) for i in range(shards)]
    if verify_files and manifest.get('saved_grid'):
        item = manifest['saved_grid']
        if item['path'] != 'fields/grid.npz' or digest(root / 'fields/grid.npz') != item['sha256']:
            report.update(status='needs_attention', grid_error='Saved grid hash mismatch')
    if job_id:
        report['slurm'] = slurm_status(job_id)
    return report


def display(report, details=False):
    print(f"MERRA-Flow full-test progress | {report['checked_utc']}")
    print(f"Output: {report['output']}")
    if report['expected'] is None:
        print(report['note'])
    else:
        print(f"Saved hours: {report['completed']}/{report['expected']} ({report['percent']:.1f}%) | remaining: {report['remaining']}")
        print(f"Missing: {len(report['missing'])} | invalid: {len(report['invalid'])} | unexpected: {len(report['unexpected'])} | excluded: {report['excluded']}")
        print(f"Lock directories: {len(report['locks'])} (not proof of live workers; may be stale)")
        print(f"Completed in last hour: {report['completed_last_hour']} | aggregation: {report['aggregation']} | status: {report['status']}")
        age = report['newest_case_age_seconds']
        if age is not None:
            print(f'Newest completed hour saved {age / 60:.1f} minutes ago')
        print(f"Checks: {report['validation']}")
        if report.get('grid_error'):
            print(report['grid_error'])
        if details:
            for key in ('missing', 'invalid', 'unexpected', 'locks', 'shards'):
                if report.get(key):
                    print(f'{key}: {json.dumps(report[key], sort_keys=True)}')
        elif report['invalid']:
            print('Invalid examples: ' + json.dumps(dict(list(report['invalid'].items())[:3])))
    if 'slurm' in report:
        for name in ('queue', 'accounting'):
            if name + '_states' in report['slurm']:
                print(f"Slurm {name}: {json.dumps(report['slurm'][name + '_states'], sort_keys=True)}")
            if name + '_error' in report['slurm']:
                print(f"Slurm {name}: {report['slurm'][name + '_error']}")
        if details:
            print('Slurm records: ' + json.dumps(report['slurm'], sort_keys=True))
    print(flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, help='The frozen full-test evaluation directory')
    parser.add_argument('--shards', type=int, help='Current array size, for optional per-shard progress')
    parser.add_argument('--job-id', help='Optional numeric array job ID for read-only squeue/sacct output')
    parser.add_argument('--verify-files', action='store_true', help='Also hash saved example arrays; may read several GB')
    parser.add_argument('--details', action='store_true', help='List missing IDs, invalid records, locks, and shards')
    parser.add_argument('--json', action='store_true', help='Print each snapshot as JSON')
    parser.add_argument('--watch', type=float, default=0, metavar='SECONDS', help='Repeat until Ctrl-C (minimum 5 seconds)')
    args = parser.parse_args(argv)
    if args.shards is not None and args.shards < 1:
        parser.error('--shards must be positive')
    if args.watch and args.watch < 5:
        parser.error('--watch must be at least 5 seconds')
    if args.job_id and not re.fullmatch(r'\d+', args.job_id):
        parser.error('--job-id must be a numeric array/job ID')
    try:
        while True:
            report = inspect(args.output, args.shards, args.verify_files, args.job_id)
            if args.json:
                print(json.dumps(report, sort_keys=True, allow_nan=False), flush=True)
            else:
                display(report, args.details)
            if not args.watch:
                return 1 if report['status'] == 'needs_attention' else 0
            time.sleep(args.watch)
    except KeyboardInterrupt:
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f'Cannot check evaluation: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
