#!/usr/bin/env python3
"""All-hours per-pixel error maps for the frozen MERRA-Flow v4.1 evaluation.

The full-test evaluator stores domain-mean scores per hour, not maps. This separate
GPU pass regenerates a deterministic subsample of the same test hours with the same
frozen checkpoint, code, sampler, and member seeds (members 1..M of the evaluation),
and accumulates per-pixel sums of

  crps        empirical CRPS of the M-member ensemble
  mae_coarse  |interpolated GEOS-FP - reference|
  mae_reg     |frozen regression - reference|
  se_mean     (ensemble mean - reference)^2
  se_coarse   (interpolated GEOS-FP - reference)^2
  se_reg      (regression - reference)^2
  var         ensemble variance (population)
  bs1, bs1_coarse   Brier score of P(rain >= 1 mm/h) for the ensemble and GEOS-FP (precip only)

for t2m, precip, ps, u10m, v10m, q2m and member wind speed. Nothing is written into
the evaluation directory, and the script lives outside src/merraflow so the running
evaluator's source hash is unchanged.

  run         one shard of hours on a GPU; resumable (per-shard state, one atomic write per hour)
  combine     sum the hours finished so far (partial is fine) into maps.npz for the paper plotter
  from-saved  no GPU: maps from the example hours the evaluation already archived (fields/*.npz)

The paper builder accepts the maps directory itself (--maps EVAL_maps) and combines whatever
hours are finished at that moment, so the figure can be drawn while the pass is still running.

Example (Discover):
  python scripts/evaluate_maps_v4_1.py run --evaluation EVAL --output EVAL_maps --every 3 \
      --members 8 --shard-index 0 --shards 8 --device cuda
  python scripts/evaluate_maps_v4_1.py combine --evaluation EVAL --output EVAL_maps
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
import uuid

import numpy as np

FIELDS = ('t2m', 'precip', 'ps', 'u10m', 'v10m', 'q2m', 'wind_speed')
METRICS = ('crps', 'mae_coarse', 'mae_reg', 'se_mean', 'se_coarse', 'se_reg', 'var')
RAIN_METRICS = ('bs1', 'bs1_coarse')


def keys():
    return [f'{f}_{m}' for f in FIELDS for m in METRICS]+[f'precip_{m}' for m in RAIN_METRICS]


def atomic_npz(path, **arrays):
    path = Path(path); tmp = path.with_name(path.name+f'.{uuid.uuid4().hex}.tmp')
    try:
        with tmp.open('wb') as f:
            np.savez(f, **arrays)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_json(path, value):
    path = Path(path); tmp = path.with_name(path.name+f'.{uuid.uuid4().hex}.tmp')
    try:
        tmp.write_text(json.dumps(value, indent=2)+'\n'); os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def plan(manifest, every, members):
    if every < 1 or not 2 <= members <= manifest['settings']['members']:
        raise ValueError('Need every >= 1 and 2 <= members <= evaluation members')
    ids = [e['id'] for e in manifest['entries']]
    chosen = ids[::every]
    return dict(version=1, run_id=manifest['run_id'], checkpoint_sha256=manifest['checkpoint_sha256'],
                every=every, members=members, steps=manifest['settings']['steps'],
                weights=manifest['settings']['weights'], hours=chosen, expected=len(ids))


def accumulate(sums, truth, coarse, regression, ensemble):
    from merraflow.evaluate_all_v4_1 import field
    from merraflow.metrics import crps_ensemble
    for c, name in enumerate(FIELDS):
        t = field(truth, c).astype('float64'); e = field(ensemble, c); co = field(coarse, c); rg = field(regression, c)
        mean = e.mean(0, dtype='float64')
        sums[f'{name}_crps'] += crps_ensemble(e, t)
        sums[f'{name}_mae_coarse'] += np.abs(co-t)
        sums[f'{name}_mae_reg'] += np.abs(rg-t)
        sums[f'{name}_se_mean'] += (mean-t)**2
        sums[f'{name}_se_coarse'] += (co-t)**2
        sums[f'{name}_se_reg'] += (rg-t)**2
        sums[f'{name}_var'] += e.var(0, dtype='float64')
        if name == 'precip':
            o = (t >= 1.).astype('float64')
            sums['precip_bs1'] += ((e >= 1.).mean(0)-o)**2
            sums['precip_bs1_coarse'] += ((co >= 1.).astype('float64')-o)**2


def separate_output(evaluation, output):
    ev, out = Path(evaluation).resolve(), Path(output).resolve()
    if ev == out or ev in out.parents:
        raise ValueError('Maps output must be separate from the frozen evaluation')
    return ev, out


def initialize_plan(out, planned):
    import fcntl
    out = Path(out)
    with (out/'.plan.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (out/'plan.json').exists():
            if json.loads((out/'plan.json').read_text()) != planned:
                raise ValueError('Existing maps plan differs; use a new output directory')
        else:
            atomic_json(out/'plan.json', planned)


def run(evaluation, output, **kwargs):
    """One writer per shard; the OS releases the lock after job exit or failure."""
    import fcntl
    _, out = separate_output(evaluation, output)
    out.mkdir(parents=True, exist_ok=True)
    with (out/f'.shard_{kwargs.get("shard_index", 0):03d}.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('Another worker is writing this maps shard') from None
        return _run(evaluation, output, **kwargs)


def _run(evaluation, output, every=3, members=8, shard_index=0, shards=1, batch=8, threads=4, device=None,
        check_source=True):
    import torch
    from merraflow.evaluate_all_v4_1 import read_manifest, digest, source_digest, plan_entries, fingerprints
    from merraflow.evaluate_v4_1 import load_model, DomainSampler, member_seed
    from merraflow.train import device_for
    ev, out = separate_output(evaluation, output)
    manifest = read_manifest(ev)
    if members < 2 or members > manifest['settings']['members'] or every < 1:
        raise ValueError('Need 2 <= members <= evaluation members and every >= 1')
    if not 0 <= shard_index < shards or batch < 1 or threads < 1:
        raise ValueError('Invalid shard')
    if digest(ev/'checkpoint.pt') != manifest['checkpoint_sha256']:
        raise ValueError('Evaluation checkpoint changed')
    if check_source and source_digest() != manifest['source_sha256']:
        raise ValueError('src/merraflow differs from the frozen evaluation code; run from the frozen checkout '
                         '(or pass --no-source-check to accept a deliberate difference)')
    out.mkdir(parents=True, exist_ok=True); (out/'shards').mkdir(exist_ok=True)
    p = dict(plan(manifest, every, members), evaluation=str(ev), shards=shards,
             maps_source_sha256=digest(__file__), source_sha256=source_digest())
    initialize_plan(out, p)
    cfg, opt = manifest['config'], manifest['settings']
    dev = device_for(device or cfg['train']['device'])
    torch.set_num_threads(threads)
    archive, model, conditioner, saved = load_model(cfg, ev/'checkpoint.pt', dev, opt['weights'])
    if fingerprints(archive) != manifest['fingerprints']:
        raise ValueError('Archive fingerprint changed')
    entries = {e['id']: e for e in plan_entries(archive, opt['split'])[0]}
    mine = p['hours'][shard_index::shards]
    state = out/'shards'/f'shard_{shard_index:03d}.npz'
    if state.exists():
        with np.load(state, allow_pickle=False) as f:
            sums = {k: f[k].astype('float64') for k in keys()}
            done = json.loads(str(f['hours']))
        if len(done) != len(set(done)) or not set(done) <= set(mine):
            raise ValueError('Saved shard membership differs from its frozen plan')
    else:
        h, w = archive.shape
        sums = {k: np.zeros((h, w), dtype='float64') for k in keys()}
        done = []
    for k, hid in enumerate(mine):
        if hid in done:
            continue
        started = time.monotonic(); entry = entries[hid]
        sampler = DomainSampler(model, conditioner, archive, entry, cfg, dev, batch=batch, threads=threads)
        truth = np.asarray(archive.physical_truth(entry), dtype='float32')
        ensemble = np.empty((members, *truth.shape), dtype='float32')
        for m in range(members):
            ensemble[m] = sampler.sample(member_seed(cfg, entry, m), opt['steps'])
        accumulate(sums, truth, sampler.coarse, sampler.regression, ensemble)
        done.append(hid)
        atomic_npz(state, hours=np.asarray(json.dumps(done)), **{k: v.astype('float32') for k, v in sums.items()})
        print(f'shard {shard_index}: {hid} ({len(done)}/{len(mine)}) {time.monotonic()-started:.0f}s', flush=True)
        del sampler, ensemble, truth
        if dev.type == 'cuda':
            torch.cuda.empty_cache()
    return len(done)


def write_maps(out, meta, sums, n, grid):
    extra = {}
    if Path(grid).exists():
        with np.load(grid, allow_pickle=False) as g:
            extra = {k: g[k] for k in ('lat', 'lon', 'area') if k in g.files}
    meta = dict(meta, n_hours=n, created_utc=datetime.now(timezone.utc).isoformat())
    atomic_npz(Path(out)/'maps.npz', meta=np.asarray(json.dumps(meta)), **extra,
               **{k: (v/n).astype('float32') for k, v in sums.items()})
    return Path(out)/'maps.npz'


def from_saved(evaluation, output):
    """Maps from the evaluation's archived example hours (all their members); no new inference."""
    from merraflow.evaluate_all_v4_1 import read_manifest, read_case, digest
    ev, out = separate_output(evaluation, output)
    manifest = read_manifest(ev); out.mkdir(parents=True, exist_ok=True)
    sums, hours, members = None, [], None
    for hid in manifest['settings'].get('saved_cases', []):
        case = ev/'cases'/f'{hid}.json'
        if not case.exists():
            continue                     # not finished yet
        entry = next(e for e in manifest['entries'] if e['id'] == hid)
        record = read_case(case, manifest, entry)
        item = record['saved_fields']['raw']
        if digest(ev/item['path']) != item['sha256']:
            raise ValueError(f'Saved fields changed: {item["path"]}')
        with np.load(ev/item['path'], allow_pickle=False) as f:
            if str(f['run_id']) != manifest['run_id'] or f['seeds'].tolist() != record['seeds']:
                raise ValueError('Saved fields belong to a different run or member seeds')
            truth, coarse, reg, ens = f['truth'], f['coarse'], f['regression'], f['ensemble']
        if sums is None:
            sums = {k: np.zeros(truth.shape[-2:], dtype='float64') for k in keys()}
        accumulate(sums, truth, coarse, reg, ens); hours.append(hid); members = len(ens)
        del truth, coarse, reg, ens
    if not hours:
        raise ValueError('No archived example hours are complete yet')
    meta = dict(version=1, source='saved_fields', run_id=manifest['run_id'], every=None, members=members,
                hours=manifest['settings']['saved_cases'], completed=hours, complete=False)
    path = write_maps(out, meta, sums, len(hours), ev/'fields'/'grid.npz')
    print(f'{path}: {len(hours)} archived example hour(s), {members} members (no new inference)', flush=True)
    return path


def ensure_maps(path, evaluation=None):
    """maps.npz path for the paper: a file is used as is; a maps directory is combined now (partial OK)."""
    path = Path(path)
    if path.is_file():
        return path
    if (path/'shards').is_dir() and any((path/'shards').glob('shard_*.npz')):
        return combine(evaluation or json.loads((path/'plan.json').read_text()).get('evaluation'), path)
    if (path/'maps.npz').is_file():
        return path/'maps.npz'
    raise FileNotFoundError(f'No maps.npz or finished shards in {path}')


def combine(evaluation, output, allow_partial=True):
    from merraflow.evaluate_all_v4_1 import read_manifest
    ev, out = separate_output(evaluation, output)
    manifest = read_manifest(ev); p = json.loads((out/'plan.json').read_text())
    p.pop('evaluation', None)
    if p['run_id'] != manifest['run_id']:
        raise ValueError('Maps plan belongs to a different evaluation')
    total, hours = None, []
    for path in sorted((out/'shards').glob('shard_*.npz')):
        with np.load(path, allow_pickle=False) as f:
            part = json.loads(str(f['hours']))
            if len(part) != len(set(part)) or set(part) & set(hours):
                raise ValueError('Overlapping shard hours')
            if not set(part) <= set(p['hours']):
                raise ValueError('Shard contains hours outside the frozen maps plan')
            hours += part
            total = {k: f[k].astype('float64') for k in keys()} if total is None else \
                {k: total[k]+f[k] for k in keys()}
    if not hours:
        raise ValueError('No completed map hours')
    if not allow_partial and len(hours) != len(p['hours']):
        raise ValueError(f'Only {len(hours)}/{len(p["hours"])} planned hours are complete')
    n = len(hours)
    meta = dict(p, source='maps_pass', completed=sorted(hours), complete=n == len(p['hours']))
    path = write_maps(out, meta, total, n, ev/'fields'/'grid.npz')
    print(f'{path}: {n}/{len(p["hours"])} planned hours so far (every {p["every"]}th test hour, '
          f'{p["members"]} members)', flush=True)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='command', required=True)
    r = sub.add_parser('run'); r.add_argument('--evaluation', required=True); r.add_argument('--output', required=True)
    r.add_argument('--every', type=int, default=3); r.add_argument('--members', type=int, default=8)
    r.add_argument('--shard-index', type=int, default=0); r.add_argument('--shards', type=int, default=1)
    r.add_argument('--batch', type=int, default=8); r.add_argument('--threads', type=int, default=4)
    r.add_argument('--device', default=None)
    r.add_argument('--no-source-check', dest='check_source', action='store_false')
    c = sub.add_parser('combine'); c.add_argument('--evaluation', required=True); c.add_argument('--output', required=True)
    c.add_argument('--require-complete', dest='allow_partial', action='store_false')
    fs = sub.add_parser('from-saved'); fs.add_argument('--evaluation', required=True); fs.add_argument('--output', required=True)
    a = vars(ap.parse_args()); cmd = a.pop('command')
    dict(run=run, combine=combine, **{'from-saved': from_saved})[cmd](**a)


if __name__ == '__main__':
    main()
