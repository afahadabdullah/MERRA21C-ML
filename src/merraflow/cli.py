import argparse
from .config import load_config


def main():
    parser = argparse.ArgumentParser(description='GEOS-FP → HWT LCC patch conditional flow matching')
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('prepare', 'prepare-predict', 'train', 'predict', 'evaluate', 'plot'):
        p = sub.add_parser(command)
        p.add_argument('--config', required=True)
        if command == 'prepare-predict':
            p.add_argument('--reference-archive', required=True)
        elif command == 'train':
            p.add_argument('--resume')
        elif command == 'predict':
            p.add_argument('--checkpoint', required=True)
            p.add_argument('--split', choices=['train', 'val', 'test', 'predict'], default='test')
            p.add_argument('--limit', type=int)
            p.add_argument('--timestamp')
        elif command in ('evaluate', 'plot'):
            p.add_argument('--predictions')
            p.add_argument('--output')
            if command == 'evaluate':
                p.add_argument('--split', choices=['val', 'test'], default='test')
            else:
                p.add_argument('--timestamp')
    p = sub.add_parser('smoke', help='Generate tiny fixtures, train, sample, score and plot on CPU')
    p.add_argument('--workdir', required=True)
    p.add_argument('--template', default='configs/discover.yaml')
    args = parser.parse_args()
    if args.command == 'smoke':
        import torch
        from .synthetic import make_synthetic
        from .prepare import prepare
        from .train import train
        from .inference import predict
        from .evaluate import evaluate
        from .diagnostics import diagnostics
        torch.set_num_threads(1)
        config_path = make_synthetic(args.workdir, load_config(args.template))
        cfg = load_config(config_path)
        prepare(cfg)
        checkpoint = train(cfg)
        predict(cfg, checkpoint, limit=1)
        evaluate(cfg)
        out = diagnostics(cfg)
        print(f'Synthetic smoke pipeline complete: {out}. This is software validation, not model skill.')
        return
    cfg = load_config(args.config)
    if args.command == 'prepare':
        from .prepare import prepare
        result = prepare(cfg)
    elif args.command == 'prepare-predict':
        from .prepare import prepare_predict
        result = prepare_predict(cfg, args.reference_archive)
    elif args.command == 'train':
        from .train import train
        result = train(cfg, args.resume)
    elif args.command == 'predict':
        from .inference import predict
        result = predict(cfg, args.checkpoint, args.split, args.limit, args.timestamp)
    elif args.command == 'evaluate':
        from .evaluate import evaluate
        result = evaluate(cfg, args.predictions, args.split, args.output)
    else:
        from .diagnostics import diagnostics
        result = diagnostics(cfg, args.predictions, args.output, args.timestamp)
    print(result)


if __name__ == '__main__':
    main()
