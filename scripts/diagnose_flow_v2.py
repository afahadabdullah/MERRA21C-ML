#!/usr/bin/env python3
"""Trace v2 flow normalization, spatial structure and same-seed ODE convergence."""
import argparse
from pathlib import Path

from merraflow.config_v2 import load_config_v2
from merraflow.flow_diagnostic_v2 import run_flow_diagnostic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/discover_annual_v2.yaml')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True, help='Fresh output root; array tasks use case_NN_v2 subdirectories')
    parser.add_argument('--timestamps', nargs='+', required=True)
    parser.add_argument('--steps', nargs='+', type=int, default=[24, 48, 96])
    parser.add_argument('--members', type=int, default=5)
    parser.add_argument('--split', choices=['val', 'test'], default='test')
    parser.add_argument('--case-index', type=int, help='Zero-based timestamp index for a Slurm array task')
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args()
    if args.case_index is not None:
        if not 0 <= args.case_index < len(args.timestamps):
            parser.error('--case-index is outside --timestamps')
        args.timestamps = [args.timestamps[args.case_index]]
        args.output = Path(args.output)/f'case_{args.case_index:02d}_v2'
    result = run_flow_diagnostic(load_config_v2(args.config), args.checkpoint, args.output, args.timestamps,
                                steps=args.steps, members=args.members, split=args.split, plots=not args.no_plots)
    print(f'Flow diagnostic written to {result}', flush=True)


if __name__ == '__main__':
    main()
