"""Read-only preflight of v2 predictor availability and surface metadata."""
import json
from .prepare_v2 import manifest, predictor_gaps, static_for, build_arrays, audit_arrays


def audit_v2(cfg):
    entries, missing = manifest(cfg)
    if not entries:
        raise FileNotFoundError('No complete paired hours for v2')
    # One representative of each split; full prepare audits every timestamp.
    selected = [next(e for e in entries if e['split'] == split) for split in ('train', 'val', 'test')
                if any(e['split'] == split for e in entries)]
    gaps = predictor_gaps(selected, cfg['data']['predictors'])
    if gaps:
        raise ValueError('V2 predictors missing from sample hours: '+json.dumps(gaps))
    static = static_for(entries[0], cfg)
    arrays = build_arrays(cfg, entries[0], static)
    result = dict(version='v2', paired_hours=len(entries), incomplete_hours=len(missing),
                  counts={s: sum(e['split'] == s for e in entries) for s in ('train', 'val', 'test')},
                  condition_channels=len(cfg['data']['predictors'])+static['features'].shape[0]+11,
                  grid_shape=list(static['area'].shape), lake_fraction_known=bool(static['lake_fraction_known'].all()),
                  sample_audit=audit_arrays(entries[0]['id'], arrays['truth'], arrays['target'], arrays['baseline'], static, arrays['native_reference']),
                  note='Read-only sample preflight; preparation validates every paired hour. Midpoint versus hourly-mean mismatch remains.')
    print(json.dumps(result, indent=2))
    return result
