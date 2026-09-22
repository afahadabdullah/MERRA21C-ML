"""python -m merraflow.cli_v3_precip --help"""
import argparse
from copy import deepcopy
from pathlib import Path
import json
import yaml
from .config_v3_precip import load_config, validate_config


def smoke(workdir):
    import torch
    from .config_v2 import load_config_v2
    from .synthetic_v2 import make_synthetic_v2
    from .prepare_v2 import prepare
    from .train_v3_precip import train
    from .inference_v3_precip import predict
    from .evaluate_v3_precip import evaluate
    torch.set_num_threads(1)
    root = Path(workdir).resolve()
    source = load_config_v2(make_synthetic_v2(root, load_config_v2('configs/discover_v2.yaml')))
    prepare(source)
    cfg = deepcopy(load_config('configs/discover_v3_precip.yaml'))
    # The synthetic fixture only writes :30 HWT files, so smoke uses snapshot targets.
    cfg['data'].update(prepared=source['data']['prepared'], history_hours=[-1, 0], target_kind='midpoint_rate')
    cfg['data'].pop('hourly_targets', None)
    cfg['patch'] = dict(source['patch'], proposal='coarse', loss_on_halo=True)
    cfg['model'] = source['model']
    cfg['train'].update(device='cpu', workers=0, batch_size=2, accumulate=2, precision='fp32',
                        regression_epochs=1, diffusion_epochs=1, val_batches=1, calibration_batches=1,
                        validation_members=2, validation_steps=2, ema_decay=.5, warmup_steps=1,
                        validation_interval=1, output=str(root/'run_v3_precip'))
    cfg['inference'].update(members=2, steps=2, output=str(root/'predictions_v3_precip'))
    validate_config(cfg)
    (root/'config_v3_precip.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    mean = train(cfg, 'regression')
    diffusion = train(cfg, 'diffusion', regression_checkpoint=mean)
    predict(cfg, diffusion, limit=1)
    return evaluate(cfg)


def main():
    parser = argparse.ArgumentParser(description='v3_precip: one-target regression + conditional EDM')
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('audit', 'train', 'predict', 'evaluate', 'prepare-hourly', 'finalize-hourly', 'months'):
        command = sub.add_parser(name)
        command.add_argument('--config', required=True)
        if name == 'train':
            command.add_argument('--stage', choices=['regression', 'diffusion'], required=True)
            command.add_argument('--regression-checkpoint')
            command.add_argument('--resume')
            command.add_argument('--time-limit-hours', type=float,
                                 help='Stop cleanly after the last epoch that fits; resume in the next job')
        elif name == 'predict':
            command.add_argument('--checkpoint', required=True)
            command.add_argument('--split', choices=['val', 'test'], default='val')
            command.add_argument('--limit', type=int)
        elif name == 'prepare-hourly':
            command.add_argument('--month', help='YYYY-MM (one Slurm array task); omit for all hours')
        elif name == 'evaluate':
            command.add_argument('--split', choices=['val', 'test'], default='val')
    command = sub.add_parser('smoke')
    command.add_argument('--workdir', required=True)
    args = parser.parse_args()
    if args.command == 'smoke':
        print(smoke(args.workdir))
        return
    cfg = load_config(args.config)
    if args.command == 'audit':
        from .dataset_v3_precip import PrecipArchive
        archive = PrecipArchive(cfg)
        counts = {split: len(archive.eligible(split)) for split in ('train', 'val', 'test')}
        print(json.dumps(dict(targets=['precip'], target_kind=archive.target_kind, channels=archive.channels,
                              eligible_hours=counts, history_hours=archive.lags,
                              training_coverage=archive.stats['training_coverage'],
                              caveat='HWT simulation rates (not observations); independent hourly members'), indent=2))
    elif args.command == 'train':
        from .train_v3_precip import train
        print(train(cfg, args.stage, args.regression_checkpoint, args.resume, args.time_limit_hours))
    elif args.command == 'prepare-hourly':
        from .prepare_v3_precip import prepare_hourly
        print(json.dumps(prepare_hourly(cfg, args.month)))
    elif args.command == 'finalize-hourly':
        from .prepare_v3_precip import finalize_hourly
        print(json.dumps(finalize_hourly(cfg), indent=2))
    elif args.command == 'months':
        from .prepare_v3_precip import months
        print('\n'.join(months(cfg)))
    elif args.command == 'predict':
        from .inference_v3_precip import predict
        print(predict(cfg, args.checkpoint, args.split, args.limit))
    else:
        from .evaluate_v3_precip import evaluate
        print(evaluate(cfg, args.split))


if __name__ == '__main__':
    main()
