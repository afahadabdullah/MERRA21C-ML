"""Read-only preflight of v2 predictor availability and surface metadata."""
import json
from collections import Counter
from .prepare_v2 import manifest, requested_hours, validate_manifest, predictor_gaps, static_for, build_arrays, audit_arrays


def coverage_v2(cfg):
    """Report file presence by month/source, plus one regridded predictor check per month."""
    entries, missing = manifest(cfg)
    months = {}
    expected_splits = Counter()
    for timestamp, split in requested_hours(cfg):
        month = timestamp.strftime('%Y-%m')
        row = months.setdefault(month, dict(month=month, expected_hours=0, paired_hours=0,
                                           missing_by_source=dict(lr=0, hr=0, native=0)))
        row['expected_hours'] += 1
        expected_splits[split] += 1
    for entry in entries:
        months[entry['time'][:7]]['paired_hours'] += 1
    examples = {}
    for entry in missing:
        for source, path in zip(entry['sources'], entry['missing']):
            months[entry['time'][:7]]['missing_by_source'][source] += 1
            examples.setdefault(source, path)
    # A regridded month can exist yet predate an added predictor, because the
    # regridder writes optional state variables only when the source has them.
    # One file per month is opened here so preparation does not discover it later.
    sampled, seen = [], set()
    for entry in entries:
        month = entry['time'][:7]
        if month not in seen:
            seen.add(month)
            sampled.append(entry)
    gaps = predictor_gaps(sampled, cfg['data']['predictors'])
    result = dict(version='v2', expected_hours=len(entries)+len(missing), paired_hours=len(entries),
                  incomplete_hours=len(missing), all_requested_pairs_present=bool(entries) and not missing,
                  requested_split_hours=dict(expected_splits),
                  first_paired_hour=entries[0]['time'] if entries else None,
                  last_paired_hour=entries[-1]['time'] if entries else None,
                  months=list(months.values()), example_missing_paths=examples,
                  predictor_spot_check=dict(months_sampled=len(sampled), gaps=gaps,
                                            all_sampled_months_have_predictors=bool(sampled) and not gaps),
                  note='File presence by month, plus the configured predictors in one regridded file per '
                       'month: lr=regridded predictors, hr=HWT labels, native=GEOS precipitation. '
                       'Audit checks sample contents and static alignment; preparation checks every paired hour.')
    print(json.dumps(result, indent=2))
    return result


def audit_v2(cfg):
    entries, missing = manifest(cfg)
    if not entries:
        raise FileNotFoundError('No complete paired hours for v2')
    # A successful preflight must not queue preparation that is already known
    # to fail on missing hours or an empty training/validation/test split.
    validate_manifest(cfg, entries, missing, require_splits=True)
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
                  surface_file=cfg['data']['static']['path'],
                  surface_fractions={name: dict(min=float(static[name].min()), max=float(static[name].max()),
                                               mean=float(static[name].mean()))
                                     for name in ('land_fraction', 'lake_fraction', 'ocean_fraction')},
                  sample_audit=audit_arrays(entries[0]['id'], arrays['truth'], arrays['target'], arrays['baseline'], static, arrays['native_reference']),
                  note='Read-only sample preflight; preparation validates every paired hour. Midpoint versus hourly-mean mismatch remains.')
    print(json.dumps(result, indent=2))
    return result
