#!/usr/bin/env python3
"""Audit existing v2 rainfall members on CPU; no prediction or training."""
import argparse
from merraflow.config_v2 import load_config_v2
from merraflow.precip_audit_v2 import run_audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/discover_annual_v2.yaml')
    parser.add_argument('--predictions', required=True, help='Directory containing *_mNNN_v2.nc')
    parser.add_argument('--output', help='Fresh directory; default: sibling precip_audit_v2')
    parser.add_argument('--split', choices=('val', 'test'), default='test')
    parser.add_argument('--timestamps', nargs='+', help='Optional exact archive IDs or ISO times')
    parser.add_argument('--members', type=int, help='Expected member count; default: infer and require same count across hours')
    parser.add_argument('--no-plots', action='store_true')
    parser.add_argument('--group-by-identity', action='store_true',
                        help='Audit different checkpoint/sampler identities in separate groups; never mix their scores')
    args = parser.parse_args()
    if args.members is not None and args.members < 2:
        parser.error('--members must be at least two')
    result = run_audit(load_config_v2(args.config), args.predictions, args.output, args.split,
                       args.timestamps, args.members, not args.no_plots, group_by_identity=args.group_by_identity)
    print(f'Rainfall audit written to {result}', flush=True)


if __name__ == '__main__':
    main()
