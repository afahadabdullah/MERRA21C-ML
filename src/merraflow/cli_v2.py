"""Separate v2 CLI; the existing merraflow command remains v1."""
import argparse
from .config_v2 import load_config_v2


def main():
    parser = argparse.ArgumentParser(description='MERRAflow v2: regression + residual flow')
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('audit', 'prepare', 'prepare-finalize', 'prepare-predict', 'train', 'predict', 'evaluate', 'plot'):
        p = sub.add_parser(name)
        p.add_argument('--config', required=True)
        if name == 'prepare':
            p.add_argument('--month')
        elif name == 'prepare-predict':
            p.add_argument('--reference-archive', required=True)
        elif name == 'train':
            p.add_argument('--stage', choices=('regression', 'flow'), required=True)
            p.add_argument('--resume')
            p.add_argument('--regression-checkpoint')
        elif name == 'predict':
            p.add_argument('--checkpoint', required=True)
            p.add_argument('--split', choices=('train', 'val', 'test', 'predict'), default='val')
            p.add_argument('--limit', type=int)
            p.add_argument('--timestamp')
        elif name in ('evaluate', 'plot'):
            p.add_argument('--split', choices=('val', 'test'), default='val')
            if name == 'plot':
                p.add_argument('--timestamp')
    p = sub.add_parser('smoke')
    p.add_argument('--workdir', required=True)
    p.add_argument('--template', default='configs/discover_v2.yaml')
    args = parser.parse_args()
    if args.command == 'smoke':
        import torch
        from .synthetic_v2 import make_synthetic_v2
        from .prepare_v2 import prepare
        from .train_v2 import train_v2
        from .inference_v2 import predict_v2
        from .diagnostics_v2 import diagnostics_v2
        torch.set_num_threads(1)
        cfg = load_config_v2(make_synthetic_v2(args.workdir, load_config_v2(args.template)))
        prepare(cfg)
        mean = train_v2(cfg, 'regression')
        flow = train_v2(cfg, 'flow', regression_checkpoint=mean)
        predict_v2(cfg, flow, limit=1)
        print(diagnostics_v2(cfg))
        return
    cfg = load_config_v2(args.config)
    if args.command == 'audit':
        from .audit_v2 import audit_v2
        audit_v2(cfg)
        return
    elif args.command == 'prepare':
        from .prepare_v2 import prepare, prepare_month
        result = prepare_month(cfg, args.month) if args.month else prepare(cfg)
    elif args.command == 'prepare-finalize':
        from .prepare_v2 import finalize_prepare
        result = finalize_prepare(cfg)
    elif args.command == 'prepare-predict':
        from .prepare_v2 import prepare_predict
        result = prepare_predict(cfg, args.reference_archive)
    elif args.command == 'train':
        from .train_v2 import train_v2
        result = train_v2(cfg, args.stage, args.resume, args.regression_checkpoint)
    elif args.command == 'predict':
        from .inference_v2 import predict_v2
        result = predict_v2(cfg, args.checkpoint, args.split, args.limit, args.timestamp)
    elif args.command == 'evaluate':
        from .evaluate_v2 import evaluate_v2
        result = evaluate_v2(cfg, args.split)
    else:
        from .diagnostics_v2 import diagnostics_v2
        result = diagnostics_v2(cfg, args.split, args.timestamp)
    print(result)


if __name__ == '__main__':
    main()
