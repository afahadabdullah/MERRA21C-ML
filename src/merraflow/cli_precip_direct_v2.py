"""Entry point for direct precipitation flow training on the original v2 archive."""
import argparse
from pathlib import Path
import torch
from .precip_direct_v2 import load_config, DirectPrecipDataset, original_checkpoint, check_checkpoint


def preflight(cfg, resume=None, initialize=None):
    archive = DirectPrecipDataset(cfg, 'train', 1, cfg['train']['seed']).archive
    DirectPrecipDataset(cfg, 'val', 1, cfg['train']['seed'])
    if resume:
        saved = torch.load(resume, map_location='cpu', weights_only=True)
        check_checkpoint(saved, cfg, archive)
    else:
        if cfg.get('conditioning'):
            original_checkpoint(cfg['conditioning']['checkpoint'], archive)
        out = Path(cfg['train']['output'])
        if out.exists() and any(out.iterdir()):
            raise FileExistsError(f'Existing direct run: {out}; set RESUME to last_direct_v2.pt')
    if initialize:
        from .precip_direct_v2 import make_model, initialize_from_v2
        initialize_from_v2(make_model(archive.index['condition_channels'], cfg), initialize, cfg, archive)
    print(f'Validated direct precipitation archive {archive.root}; frozen regression input={bool(cfg.get("conditioning"))}', flush=True)


def main():
    parser = argparse.ArgumentParser(description='Full precipitation flow; no residual target or output add-back')
    parser.add_argument('command', choices=['preflight', 'train', 'predict'])
    parser.add_argument('--config', required=True)
    parser.add_argument('--resume')
    parser.add_argument('--initialize-v2')
    parser.add_argument('--regression-checkpoint', help='Override frozen original v2 predictor; use none to disable')
    parser.add_argument('--checkpoint')
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--limit', type=int, default=1)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.regression_checkpoint:
        cfg['conditioning'] = None if args.regression_checkpoint == 'none' else dict(checkpoint=args.regression_checkpoint)
    if args.resume and args.initialize_v2:
        parser.error('--resume and --initialize-v2 are mutually exclusive')
    if args.command == 'preflight':
        preflight(cfg, args.resume, args.initialize_v2)
    elif args.command == 'train':
        from .train_precip_direct_v2 import train
        print(train(cfg, args.resume, args.initialize_v2))
    else:
        if not args.checkpoint or args.limit < 1:
            parser.error('predict requires --checkpoint and a positive --limit')
        from .inference_precip_direct_v2 import predict
        print(predict(cfg, args.checkpoint, args.split, args.limit))


if __name__ == '__main__':
    main()
