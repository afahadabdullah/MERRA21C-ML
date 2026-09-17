#!/usr/bin/env python3
"""Run independent v2 diagnostic cases across the GPUs in one allocation."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys

from merraflow.config import write_json

from test_best_model_v2 import plot_summary


def visible_devices():
    raw = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    devices = [device.strip() for device in raw.split(',') if device.strip()]
    if not devices:
        raise RuntimeError('CUDA_VISIBLE_DEVICES must list the allocated GPUs')
    return devices


def merge_cases(output, case_dirs, checkpoint, members, steps):
    reports = []
    metadata = None
    for case_dir in case_dirs:
        metrics = json.loads((case_dir/'metrics_v2.json').read_text())
        expected = (metrics['checkpoint_sha256'], metrics['members'], metrics['ode_steps'],
                    metrics['inference_seed'], metrics['noise_padding'], metrics['sampler'],
                    metrics['precipitation_representation'])
        if metadata is None:
            metadata = expected
        elif expected != metadata:
            raise ValueError(f'Mixed checkpoint or sampler settings in {case_dir}')
        if (metrics['split'] != 'test' or len(metrics['samples']) != 1
                or metrics['samples'][0]['id'] != case_dir.name):
            raise ValueError(f'Unexpected case in {case_dir}')
        reports.extend(metrics['samples'])
    if metadata[1] != members or (steps is not None and metadata[2] != steps):
        raise ValueError('Member count or ODE steps differ from the requested evaluation')
    combined = dict(metrics)
    combined['checkpoint'] = str(checkpoint.resolve())
    combined['selection'] = {'strategy': 'explicit timestamps'}
    combined['samples'] = reports
    combined['case_directories'] = [str(path) for path in case_dirs]
    write_json(output/'metrics_v2.json', combined)
    plot_summary(output, reports, members, 'test')
    return combined


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--members', type=int, default=10)
    parser.add_argument('--expected-gpus', type=int, default=4)
    parser.add_argument('--steps', type=int)
    parser.add_argument('--timestamps', nargs='+', required=True)
    args = parser.parse_args()
    if len(set(args.timestamps)) != len(args.timestamps):
        parser.error('Timestamps must be unique')
    if args.members < 2:
        parser.error('--members must be >= 2')
    if not args.checkpoint.is_file():
        parser.error(f'Checkpoint does not exist: {args.checkpoint}')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error(f'Output directory is not empty: {args.output}')
    args.output.mkdir(parents=True, exist_ok=True)
    devices = visible_devices()
    if len(devices) != args.expected_gpus:
        raise RuntimeError(f'Expected {args.expected_gpus} allocated GPUs; found {len(devices)}: {devices}')
    print(f'Running {len(args.timestamps)} cases with {args.members} members across '
          f'{len(devices)} GPUs: {devices}', flush=True)

    def run_case(index, stamp):
        case_dir = args.output/'cases_v2'/stamp
        command = [sys.executable, 'scripts/test_best_model_v2.py',
                   '--config', args.config, '--checkpoint', str(args.checkpoint),
                   '--output', str(case_dir), '--split', 'test', '--members', str(args.members),
                   '--timestamps', stamp]
        if args.steps is not None:
            command += ['--steps', str(args.steps)]
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = devices[index]
        env['MPLCONFIGDIR'] = str(args.output/f'matplotlib_{index}_v2')
        with (args.output/f'{stamp}_v2.log').open('w') as log:
            subprocess.run(command, check=True, env=env, stdout=log, stderr=subprocess.STDOUT)
        return case_dir

    def run_device(index, stamps):
        for stamp in stamps:
            run_case(index, stamp)
            print(f'Completed {stamp}', flush=True)

    case_dirs = [args.output/'cases_v2'/stamp for stamp in args.timestamps]
    device_count = min(len(devices), len(case_dirs))
    with ThreadPoolExecutor(max_workers=device_count) as pool:
        futures = {pool.submit(run_device, index, args.timestamps[index::device_count]): index
                   for index in range(device_count)}
        failures = []
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                failures.append(exc)
                print(f'GPU worker {futures[future]} failed: {exc}; see case logs in {args.output}', flush=True)
    if failures:
        raise RuntimeError(f'{len(failures)} diagnostic case(s) failed')
    combined = merge_cases(args.output, case_dirs, args.checkpoint, args.members, args.steps)
    print(f'Combined {len(combined["samples"])} cases in {args.output}', flush=True)


if __name__ == '__main__':
    main()
