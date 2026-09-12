"""Find regridded hours that predate a v2 predictor and stage them for rebuilding.

The regridder's own resume check accepts a file that holds the v1 required
variables, so an output written before QV2M, SLP or OMEGA500 were emitted is
skipped rather than repaired. This reports such files for a v2 config and, with
--move-aside, renames them so a regridding re-run recreates them. Nothing is
deleted, and no job is submitted.
"""
import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))

from merraflow.config_v2 import load_config_v2
from merraflow.prepare_v2 import manifest, predictor_gaps

SUFFIX = '.incomplete_v2'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--month', help='Restrict to one YYYY-MM month')
    parser.add_argument('--move-aside', action='store_true',
                        help=f'Rename incomplete files with the {SUFFIX} suffix so regridding rebuilds them')
    parser.add_argument('--report', help='Write the full gap list to this JSON path')
    args = parser.parse_args()
    cfg = load_config_v2(args.config)
    entries, missing = manifest(cfg, month=args.month)
    if not entries:
        print('No paired hours to check; run coverage first.')
        return 1
    months = sorted({entry['time'][:7] for entry in entries})
    gaps, moved = [], 0
    for month in months:
        selected = [entry for entry in entries if entry['time'].startswith(month)]
        month_gaps = predictor_gaps(selected, cfg['data']['predictors'])
        print(f'{month}: checked {len(selected)} paired hours, {len(month_gaps)} incomplete', flush=True)
        gaps.extend(month_gaps)
    for gap in gaps:
        if args.move_aside and os.path.exists(gap['path']):
            os.replace(gap['path'], gap['path']+SUFFIX)
            moved += 1
    counts = Counter(name for gap in gaps for name in gap.get('missing', ['unreadable']))
    summary = {'config': args.config, 'paired_hours_checked': len(entries),
               'unpaired_hours': len(missing), 'incomplete_files': len(gaps),
               'missing_variable_counts': dict(sorted(counts.items())),
               'moved_aside': moved, 'suffix': SUFFIX if moved else None,
               'examples': gaps[:5]}
    print(json.dumps(summary, indent=2))
    if args.report:
        Path(args.report).write_text(json.dumps(gaps, indent=2))
        print(f'Full list: {args.report}')
    if gaps and not args.move_aside:
        print('Re-run with --move-aside, then resubmit regridding for these months.')
    return 1 if gaps and not args.move_aside else 0


if __name__ == '__main__':
    raise SystemExit(main())
