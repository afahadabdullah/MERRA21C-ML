"""V4 data audit, humidity preparation, six-variable training and inference."""
import argparse
from pathlib import Path
import json
import torch
from .precip_direct_v2 import original_checkpoint
from .v4 import load_config, ArchiveV4, DatasetV4, check_checkpoint


def inspect_data(cfg):
    from .dataset_v2 import ArchiveV2
    from .dataset_v3_precip import PrecipArchive, HOURLY_INDEX
    from .prepare_v4 import read_q2m
    archive = ArchiveV2(cfg['data']['prepared'])
    configured = Path(cfg['data']['hourly_targets'])
    found = sorted(set([p.parent.resolve() for p in Path('data').glob('*/'+HOURLY_INDEX)]
                       +([configured.resolve()] if (configured/HOURLY_INDEX).exists() else [])))
    print('Hourly target directories:', [str(p) for p in found], flush=True)
    print('Configured hourly targets:', str(configured.resolve()), flush=True)
    if not (configured/HOURLY_INDEX).exists():
        raise FileNotFoundError('Configured hourly index missing; choose a detected path with --hourly-targets')
    hourly = PrecipArchive(cfg)
    summary = dict(prepared=str(archive.root.resolve()), hourly_targets=str(configured.resolve()),
        hourly_fingerprint=hourly.hourly_fingerprint, predictors=archive.stats['predictors'],
        humidity_targets=str(Path(cfg['data']['humidity_targets']).resolve()),
        eligible_hours={split:len(hourly.eligible(split)) for split in ('train','val','test')})
    if 'QV2M' not in archive.stats['predictors']:
        raise ValueError('QV2M is not in this prepared archive')
    for split in ('train','val','test'):
        entry = hourly.eligible(split)[0]
        q = read_q2m(cfg, archive, entry)
        print(f'{split} humidity source checked: {entry["hr"]}; range {q.min():g}..{q.max():g} kg/kg', flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def preflight(cfg, resume=None):
    archive = ArchiveV4(cfg)
    for split in ('train','val','test'):
        entries = archive.eligible(split)
        print(f'{split}: {len(entries)} hours with complete history and hourly rainfall', flush=True)
    if resume:
        check_checkpoint(torch.load(resume, map_location='cpu', weights_only=True), cfg, archive)
    else:
        original_checkpoint(cfg['conditioning']['checkpoint'], archive)
        out = Path(cfg['train']['output'])
        if out.exists() and any(out.iterdir()):
            raise FileExistsError(f'Existing v4 output: {out}; use --resume {out}/last_v4.pt')
    # Exercise real crops, humidity sidecars, previous-hour inputs and targets.
    for split in ('train', 'val'):
        sample = DatasetV4(cfg, split, 1, cfg['train']['seed'])[0]
        if any(not torch.isfinite(value).all() for value in sample.values()):
            raise ValueError('Nonfinite preflight sample')
    print(f'V4 validated: six targets, {archive.channels} flow inputs, hourly rainfall, no CAPE.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['inspect-data','prepare-humidity','finalize-humidity','preflight','train','predict'])
    parser.add_argument('--config', default='configs/discover_v4.yaml')
    parser.add_argument('--month')
    parser.add_argument('--hourly-targets')
    parser.add_argument('--regression-checkpoint')
    parser.add_argument('--resume')
    parser.add_argument('--checkpoint')
    parser.add_argument('--split', choices=['val','test'], default='test')
    parser.add_argument('--limit', type=int, default=1)
    parser.add_argument('--time')
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.hourly_targets:
        cfg['data']['hourly_targets'] = args.hourly_targets
    if args.regression_checkpoint:
        cfg['conditioning']['checkpoint'] = args.regression_checkpoint
    if args.command == 'inspect-data':
        inspect_data(cfg)
    elif args.command == 'prepare-humidity':
        from .prepare_v4 import prepare_humidity
        print(prepare_humidity(cfg, args.month))
    elif args.command == 'finalize-humidity':
        from .prepare_v4 import finalize_humidity
        print(finalize_humidity(cfg))
    elif args.command == 'preflight':
        preflight(cfg, args.resume)
    elif args.command == 'train':
        from .train_v4 import train
        print(train(cfg, args.resume))
    else:
        if not args.checkpoint or args.limit < 1:
            parser.error('predict requires --checkpoint and positive --limit')
        from .inference_v4 import predict
        print(predict(cfg, args.checkpoint, args.split, args.limit, args.time))


if __name__ == '__main__':
    main()
