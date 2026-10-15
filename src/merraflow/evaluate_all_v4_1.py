"""Resumable, sharded full-test evaluation. See docs/evaluation_all_v4_1.md.

No case selection, maps, or test-based recipe tuning. Raw and spectral outputs
use exactly the same members. One atomic JSON per hour, a frozen checkpoint,
training-only spectral reference, and strict complete-manifest aggregation.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
from scipy.ndimage import uniform_filter
from .metrics import continuous, crps_ensemble, radial_psd, rank_histogram, weighted_mean

VERSION = 1
FIELDS = ('t2m', 'precip', 'ps', 'u10m', 'v10m', 'q2m', 'wind_speed')
UNITS = ('K', 'mm h-1', 'Pa', 'm s-1', 'm s-1', 'kg kg-1', 'm s-1')
THRESHOLDS = (.1, 1., 5., 10., 25.)
SCALES = (1, 5, 17, 33, 65)
QUANTILES = (.5, .95, .99, .999)


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def source_digest():
    h = hashlib.sha256()
    for p in sorted(Path(__file__).parent.glob('*.py')):
        h.update(p.name.encode()); h.update(p.read_bytes())
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def serial(value):
    if isinstance(value, dict):
        return {k: serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serial(v) for v in value]
    if isinstance(value, np.ndarray):
        return serial(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name+f'.{os.getpid()}.tmp')
    try:
        tmp.write_text(json.dumps(serial(value), indent=2, allow_nan=False)+'\n')
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_npz(path, **arrays):
    """Lossless meteorological fields, committed before their case JSON."""
    path = Path(path)
    tmp = path.with_name(path.name+f'.{os.getpid()}.tmp')
    try:
        with tmp.open('wb') as f:
            np.savez_compressed(f, **arrays)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def saved_case_ids(entries, selection='auto'):
    """Fix map examples before inference; never select on test scores or sharpness."""
    ids = [e['id'] for e in entries]
    if not ids:
        raise ValueError('No eligible evaluation hours')
    if selection == 'auto':
        return [ids[i] for i in sorted(set(np.linspace(0, len(ids)-1, min(3, len(ids)), dtype=int)))]
    if selection == 'all':
        return ids
    if selection == 'none':
        return []
    selected = selection.split(',')
    if len(set(selected)) != len(selected) or not set(selected) <= set(ids):
        raise ValueError('Saved map cases must be distinct eligible timestamp IDs')
    return selected


@contextmanager
def lock(path):
    path = Path(path)
    try:
        path.mkdir()
    except FileExistsError as exc:
        raise RuntimeError(f'Locked: {path}. Confirm no live worker before removing a stale lock.') from exc
    try:
        (path/'owner.json').write_text(json.dumps(dict(pid=os.getpid(), host=os.uname().nodename,
            time=datetime.now(timezone.utc).isoformat(), slurm_job=os.environ.get('SLURM_JOB_ID'))))
        yield
    finally:
        shutil.rmtree(path)


def field(arr, c):
    return np.hypot(arr[..., 3, :, :], arr[..., 4, :, :]) if c == 6 else arr[..., c, :, :]


def field_scores(ensemble, truth, area, seed=0):
    """Physical-unit scores; no pairwise M x M allocation; all members used."""
    m = len(ensemble)
    score = continuous(ensemble, truth, area)
    first = float(np.mean([weighted_mean(abs(x-truth), area) for x in ensemble]))
    pair = first-score['crps']  # sum_{i,j}|xi-xj| / (2 M^2)
    score['fair_crps'] = first-m/(m-1)*pair if m > 1 else None
    score['mse'] = score['rmse']**2
    score['member_mae'] = first
    score['member_mse'] = float(np.mean([weighted_mean((x.astype('float64')-truth)**2, area) for x in ensemble]))
    score['variance_population'] = score['spread']**2
    score['variance_unbiased'] = score['spread']**2*m/(m-1) if m > 1 else None
    score['mean'] = weighted_mean(ensemble.mean(0), area)
    score['reference_mean'] = weighted_mean(truth, area)
    # Pixel quantiles are explicitly index-weighted, not area-weighted.
    score['pixel_quantiles'] = np.quantile(ensemble, QUANTILES).tolist()
    score['reference_pixel_quantiles'] = np.quantile(truth, QUANTILES).tolist()
    score['rank_histogram'] = rank_histogram(ensemble, truth, area, seed) if m > 1 else None
    freq, target = radial_psd(truth)
    member = np.mean([radial_psd(x)[1] for x in ensemble], axis=0)
    score['spectrum'] = dict(frequency_cycles_pixel=freq.tolist(), reference=target.tolist(),
                             members=member.tolist(), ensemble_mean=radial_psd(ensemble.mean(0))[1].tolist())
    def gradient(x):
        dy, dx = np.gradient(np.asarray(x, dtype='float64'))
        return float(np.quantile(np.hypot(dy, dx), .99))
    score['gradient_p99_reference'] = gradient(truth)
    score['gradient_p99_members_mean'] = float(np.mean([gradient(x) for x in ensemble]))
    return score


def fss_parts(pred, truth, threshold, scale, area):
    if scale > min(truth.shape):
        return None
    p = uniform_filter((pred >= threshold).astype('float64'), size=scale, mode='constant')
    o = uniform_filter((truth >= threshold).astype('float64'), size=scale, mode='constant')
    r = scale//2
    region = (slice(r, -r), slice(r, -r)) if r else (slice(None), slice(None))
    p, o, a = p[region], o[region], area[region]
    return dict(numerator=float(np.sum(a*(p-o)**2)), denominator=float(np.sum(a*(p*p+o*o))))


def rain_scores(ensemble, truth, area):
    m = len(ensemble)
    avg = ensemble.mean(0)
    out = {}
    for threshold in THRESHOLDS:
        observed = truth >= threshold
        probability = (ensemble >= threshold).mean(0)
        predicted = avg >= threshold
        # Store additive area counts, so reliability and contingency ratios pool exactly.
        hit = float(area[predicted & observed].sum())
        miss = float(area[~predicted & observed].sum())
        false = float(area[predicted & ~observed].sum())
        bins = np.rint(probability*m).astype(int)
        out[str(threshold)] = dict(
            brier=weighted_mean((probability-observed)**2, area),
            hit_area=hit, miss_area=miss, false_alarm_area=false,
            observed_area=float(area[observed].sum()), observed_pixels=int(observed.sum()),
            reliability_area=np.bincount(bins.ravel(), weights=area.ravel(), minlength=m+1).tolist(),
            reliability_observed_area=np.bincount(bins.ravel(), weights=(area*observed).ravel(), minlength=m+1).tolist(),
            fss={})
        if threshold in (1., 5., 10.):
            for scale in SCALES:
                part = fss_parts(avg, truth, threshold, scale, area)
                if part is not None:
                    members = [fss_parts(x, truth, threshold, scale, area) for x in ensemble]
                    out[str(threshold)]['fss'][str(scale)] = dict(ensemble_mean=part,
                        members={k: float(np.mean([p[k] for p in members])) for k in part})
    return out


def score_source(ensemble, truth, area, seed=0):
    if (ensemble.ndim != 4 or ensemble.shape[1:] != truth.shape or truth.shape[0] != 6
            or area.shape != truth.shape[-2:] or min(area.shape) < 2
            or not np.isfinite(ensemble).all() or not np.isfinite(truth).all()
            or not np.isfinite(area).all() or np.any(area <= 0)):
        raise ValueError('Expected finite [member,6,y,x] fields and positive cell areas')
    return dict(members=len(ensemble),
        fields={name: field_scores(field(ensemble, c), field(truth, c), area, seed+c)
                for c, name in enumerate(FIELDS)}, precipitation=rain_scores(ensemble[:, 1], truth[1], area))


def plan_entries(archive, split):
    entries = sorted(archive.eligible(split), key=lambda e: e['time'])
    if len({e['id'] for e in entries}) != len(entries):
        raise ValueError('Duplicate eligible hour IDs')
    eligible = {e['id'] for e in entries}
    excluded = []
    for e in archive.index['entries']:
        if e['split'] == split and e['id'] not in eligible:
            reason = 'missing hourly precipitation target' if not archive.has_target(e) else 'incomplete within-split history'
            excluded.append(dict(id=e['id'], time=e['time'], reason=reason))
    return entries, excluded


def fingerprints(archive):
    return dict(prepared=archive.index['fingerprint'], hourly=archive.hourly_fingerprint,
                humidity=archive.humidity_fingerprint)


def initialize(cfg, output, checkpoint='best', split='test', members=16, steps=64,
               post='both', weights='ema', spectral_count=12, spectral_gain=1.3, spectral_cutoff=40.,
               save_cases='auto'):
    from .evaluate_v4_1 import resolve_checkpoint
    from .v4 import ArchiveV4
    from .v4_1 import base_config
    if members < 2 or steps < 1 or post not in ('none', 'both') or weights not in ('ema', 'raw'):
        raise ValueError('Require members >= 2, steps >= 1, valid post and weights')
    if spectral_count < 1 or not np.isfinite([spectral_gain, spectral_cutoff]).all() or spectral_gain < 1 or spectral_cutoff <= 0:
        raise ValueError('Invalid spectral settings')
    out = Path(output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Use a new run directory, or resume with run: {out}')
    path = resolve_checkpoint(cfg, checkpoint)
    archive = ArchiveV4(base_config(cfg), verify_files=False)
    entries, excluded = plan_entries(archive, split)
    selected = saved_case_ids(entries, save_cases)
    out.mkdir(parents=True, exist_ok=True)
    (out/'cases').mkdir()
    (out/'fields').mkdir()
    # Snapshot once: a training job updating best/latest cannot change an array task.
    before = digest(path)
    shutil.copyfile(path, out/'checkpoint.pt')
    if digest(out/'checkpoint.pt') != before or digest(path) != before:
        raise RuntimeError('Checkpoint changed during snapshot; initialize a new run from a stable checkpoint')
    settings = dict(split=split, members=members, steps=steps, post=post, weights=weights,
                    spectral_count=spectral_count, spectral_gain=spectral_gain, spectral_cutoff=spectral_cutoff,
                    saved_cases=selected)
    manifest = dict(version=VERSION, config=cfg, settings=settings, source_sha256=source_digest(),
                    checkpoint_sha256=before, original_checkpoint=str(path.resolve()),
                    fingerprints=fingerprints(archive), shape=list(archive.shape),
                    entries=[dict(id=e['id'], time=e['time']) for e in entries], excluded=excluded,
                    created_utc=datetime.now(timezone.utc).isoformat())
    if selected:
        atomic_npz(out/'fields'/'grid.npz', **{k: np.asarray(archive.static[k]) for k in ('lat', 'lon', 'area')})
        manifest['saved_grid'] = dict(path='fields/grid.npz', sha256=digest(out/'fields'/'grid.npz'))
    if post == 'both':
        from .explore_inference_v4_1 import climatology
        training = {e['time'] for e in archive.index['entries'] if e['split'] == 'train'}
        if not training:
            raise ValueError('Spectral processing requires training-only reference hours; no fallback permitted')
        h, w = archive.shape
        clim = climatology(archive, entries[0], (0, h, 0, w), count=spectral_count)
        if len(clim['times']) != spectral_count or not set(clim['times']) <= training:
            raise ValueError('Could not obtain the requested number of training-only spectral references')
        np.savez(out/'spectral_reference.npz', psd=clim['psd'], freq=clim['freq'])
        manifest['spectral_reference'] = dict(times=clim['times'], sha256=digest(out/'spectral_reference.npz'),
                                             anchor_time=entries[0]['time'])
    manifest['run_id'] = hashlib.sha256(canonical(manifest).encode()).hexdigest()
    atomic_json(out/'manifest.json', manifest)  # Last: incomplete initialization is not runnable.
    print(f'{len(entries)} eligible {split} hours; {len(excluded)} excluded. Frozen run: {out}', flush=True)
    return out


def read_manifest(output):
    m = json.loads((Path(output)/'manifest.json').read_text())
    body = {k: v for k, v in m.items() if k != 'run_id'}
    if m['version'] != VERSION or hashlib.sha256(canonical(body).encode()).hexdigest() != m['run_id']:
        raise ValueError('Manifest version/hash mismatch')
    return m


def read_case(path, manifest, entry):
    record = json.loads(Path(path).read_text())
    if (record.get('run_id') != manifest['run_id'] or record.get('id') != entry['id']
            or record.get('time') != entry['time']):
        raise ValueError(f'Case identity mismatch: {path}')
    expected = {'coarse', 'regression', 'raw'} | ({'spectral'} if manifest['settings']['post'] == 'both' else set())
    if set(record['sources']) != expected:
        raise ValueError(f'Incomplete sources: {path}')
    for name, source in record['sources'].items():
        if set(source['fields']) != set(FIELDS) or source['members'] != (1 if name in ('coarse', 'regression') else manifest['settings']['members']):
            raise ValueError(f'Incomplete fields/member count: {path}')
    canonical(record)  # Reject NaN/infinite JSON from damaged or foreign writers.
    if entry['id'] in manifest['settings'].get('saved_cases', []):
        expected_files = {'raw'} | ({'spectral'} if 'spectral' in expected else set())
        if set(record.get('saved_fields', {})) != expected_files:
            raise ValueError(f'Missing saved map fields: {path}')
        root = Path(path).parent.parent
        for name, item in record['saved_fields'].items():
            relative = f'fields/{entry["id"]}_{name}.npz'
            if item['path'] != relative or digest(root/relative) != item['sha256']:
                raise ValueError(f'Saved map fields changed: {relative}')
    return record


def run(output, shard_index=0, shards=1, batch=8, threads=4, device=None):
    import torch
    from .evaluate_v4_1 import load_model, DomainSampler, member_seed
    from .train import device_for
    from .explore_inference_v4_1 import spectral_fix
    out = Path(output); manifest = read_manifest(out)
    if shards < 1 or not 0 <= shard_index < shards or batch < 1 or threads < 1:
        raise ValueError('Invalid shard, batch, or thread settings')
    if source_digest() != manifest['source_sha256'] or digest(out/'checkpoint.pt') != manifest['checkpoint_sha256']:
        raise ValueError('Code/checkpoint changed; use the frozen code and checkpoint or initialize a new run')
    cfg, opt = manifest['config'], manifest['settings']
    dev = device_for(device or cfg['train']['device'])
    torch.set_num_threads(threads)
    archive, model, conditioner, saved = load_model(cfg, out/'checkpoint.pt', dev, opt['weights'])
    if fingerprints(archive) != manifest['fingerprints']:
        raise ValueError('Archive fingerprint changed')
    entries, _ = plan_entries(archive, opt['split'])
    if [dict(id=e['id'], time=e['time']) for e in entries] != manifest['entries']:
        raise ValueError('Eligible hours changed')
    area = np.asarray(archive.static['area'], dtype='float64')
    dx = float(np.sqrt(np.median(area))/1000)
    clim = None
    if opt['post'] == 'both':
        p = out/'spectral_reference.npz'
        if digest(p) != manifest['spectral_reference']['sha256']:
            raise ValueError('Spectral reference changed')
        with np.load(p, allow_pickle=False) as f:
            clim = {k: f[k].copy() for k in f.files}
    completed = 0
    for entry in entries[shard_index::shards]:
        path = out/'cases'/f'{entry["id"]}.json'
        if path.exists():
            read_case(path, manifest, entry); completed += 1; continue
        with lock(path.with_suffix('.lock')):
            started = time.monotonic()
            sampler = DomainSampler(model, conditioner, archive, entry, cfg, dev, batch=batch, threads=threads)
            truth = np.asarray(archive.physical_truth(entry), dtype='float32')
            sources = {}
            for name, values in [('coarse', sampler.coarse), ('regression', sampler.regression)]:
                sources[name] = score_source(values[None], truth, area)
            seeds = [member_seed(cfg, entry, m) for m in range(opt['members'])]
            ensemble = np.empty((opt['members'], *truth.shape), dtype='float32')
            sample_start = time.monotonic()
            for m, seed in enumerate(seeds):
                ensemble[m] = sampler.sample(seed, opt['steps'])
                print(f'{entry["id"]}: member {m+1}/{opt["members"]}', flush=True)
            sample_seconds = time.monotonic()-sample_start
            sources['raw'] = score_source(ensemble, truth, area, seeds[0])
            saved_fields = {}
            save_maps = entry['id'] in opt.get('saved_cases', [])
            if save_maps:
                from .evaluate_v4_1 import native_fields
                native, native_missing = native_fields(entry, archive.static['lat'], archive.static['lon'])
                native_arrays = {f'native_{name}_{key}': item[key] for name,item in native.items()
                                 for key in ('values','lat','lon')}
                relative = f'fields/{entry["id"]}_raw.npz'
                atomic_npz(out/relative, truth=truth, coarse=sampler.coarse,
                           regression=sampler.regression, ensemble=ensemble,
                           run_id=np.asarray(manifest['run_id']), seeds=np.asarray(seeds), **native_arrays)
                saved_fields['raw'] = dict(path=relative, sha256=digest(out/relative),native_missing=native_missing)
            if clim is not None:
                for m in range(len(ensemble)):
                    ensemble[m] = spectral_fix(ensemble[m], clim, dx, opt['spectral_gain'], opt['spectral_cutoff'])
                sources['spectral'] = score_source(ensemble, truth, area, seeds[0])
                if save_maps:
                    relative = f'fields/{entry["id"]}_spectral.npz'
                    atomic_npz(out/relative, ensemble=ensemble, run_id=np.asarray(manifest['run_id']),
                               seeds=np.asarray(seeds))
                    saved_fields['spectral'] = dict(path=relative, sha256=digest(out/relative))
            record = dict(run_id=manifest['run_id'], id=entry['id'], time=entry['time'],
                          epoch=saved['epoch']+1, seeds=seeds, area_m2=float(area.sum()),
                          dx_km_approx=dx, source_units=dict(zip(FIELDS, UNITS)), sources=sources,
                          sampling_seconds=sample_seconds, total_seconds=time.monotonic()-started,
                          device=str(dev), host=os.uname().nodename, batch=batch, threads=threads,
                          saved_fields=saved_fields)
            atomic_json(path, record)
            del sampler, ensemble, truth
            if dev.type == 'cuda':
                torch.cuda.empty_cache()
        completed += 1
        print(f'Shard {shard_index}/{shards}: {completed}/{len(entries[shard_index::shards])} hours complete', flush=True)
    return completed


def ratio(a, b):
    return a/b if b > 0 else None


def aggregate_sources(records):
    """Pooled area-time RMSE and counts, equal-hour linear scores; grid is fixed."""
    if not records or len({r['area_m2'] for r in records}) != 1:
        raise ValueError('Expected nonempty records on one fixed-area grid')
    result = {}
    for source in records[0]['sources']:
        result[source] = dict(fields={}, precipitation={})
        for name in FIELDS:
            items = [r['sources'][source]['fields'][name] for r in records]
            linear = ('mae', 'bias', 'crps', 'fair_crps', 'mse', 'member_mae', 'member_mse',
                      'variance_population', 'variance_unbiased', 'coverage_90', 'mean', 'reference_mean',
                      'gradient_p99_reference', 'gradient_p99_members_mean')
            summary = {k: float(np.mean([i[k] for i in items])) if items[0][k] is not None else None for k in linear}
            summary['rmse_pooled'] = float(np.sqrt(summary['mse']))
            summary['rmse_hour_mean'] = float(np.mean([i['rmse'] for i in items]))
            m = records[0]['sources'][source]['members']
            summary['spread_rms_population'] = float(np.sqrt(summary['variance_population']))
            summary['spread_skill_corrected'] = (ratio(np.sqrt(summary['variance_population']*(m+1)/(m-1)),
                                                       summary['rmse_pooled']) if m > 1 else None)
            for key in ('pixel_quantiles', 'reference_pixel_quantiles', 'rank_histogram'):
                summary[key+'_hour_mean'] = np.mean([i[key] for i in items], axis=0).tolist() if items[0][key] is not None else None
            summary['spectrum'] = dict(frequency_cycles_pixel=items[0]['spectrum']['frequency_cycles_pixel'])
            for k in ('reference', 'members', 'ensemble_mean'):
                summary['spectrum'][k] = np.mean([i['spectrum'][k] for i in items], axis=0).tolist()
            result[source]['fields'][name] = summary
        for threshold in records[0]['sources'][source]['precipitation']:
            items = [r['sources'][source]['precipitation'][threshold] for r in records]
            counts = {k: sum(i[k] for i in items) for k in ('hit_area','miss_area','false_alarm_area','observed_area','observed_pixels')}
            h, miss, false = [counts[k] for k in ('hit_area','miss_area','false_alarm_area')]
            total = dict(counts, csi=ratio(h, h+miss+false), pod=ratio(h,h+miss), far=ratio(false,h+false),
                         frequency_bias=ratio(h+false,h+miss), brier=float(np.mean([i['brier'] for i in items])))
            a = np.sum([i['reliability_area'] for i in items], axis=0)
            o = np.sum([i['reliability_observed_area'] for i in items], axis=0)
            total.update(reliability_area=a.tolist(), reliability_observed_area=o.tolist(),
                         reliability_frequency=[ratio(x,y) for x,y in zip(o,a)],
                         probability_bins=np.linspace(0,1,len(a)).tolist(), fss={})
            for scale in items[0]['fss']:
                total['fss'][scale] = {}
                for kind in ('ensemble_mean','members'):
                    num = sum(i['fss'][scale][kind]['numerator'] for i in items)
                    den = sum(i['fss'][scale][kind]['denominator'] for i in items)
                    total['fss'][scale][kind] = 1-num/den if den > 0 else None
            result[source]['precipitation'][threshold] = total
    for source, report in result.items():
        for name, f in report['fields'].items():
            coarse = result['coarse']['fields'][name]['crps']
            regression = result['regression']['fields'][name]['crps']
            f['crpss_vs_coarse'] = 1-f['crps']/coarse if coarse > 0 else None
            f['crpss_vs_regression'] = 1-f['crps']/regression if regression > 0 else None
    return result


def paired_intervals(records, draws=1000, block_hours=72, seed=2026):
    """Paired nonoverlapping temporal-block bootstrap; blocks retain spatial dependence.

    Blocks are anchored to elapsed UTC hours, not row numbers, so missing hours do
    not silently shorten them. Report a sensitivity to block length separately.
    """
    if draws < 1 or block_hours < 1:
        raise ValueError('Positive bootstrap draws/block hours required')
    times = np.array([np.datetime64(r['time'], 'h').astype(int) for r in records])
    blocks = (times-times.min())//block_hours
    labels = np.unique(blocks)
    output = dict(block_hours=block_hours, blocks=len(labels), draws=draws, seed=seed, comparisons={})
    if len(labels) < 5:
        output['status'] = 'insufficient_blocks'; return output
    rng = np.random.default_rng(seed)
    # Same draws for every variable/method preserve paired comparisons.
    chosen = rng.integers(len(labels), size=(draws,len(labels)))
    counts = np.array([(blocks==b).sum() for b in labels])
    for source in ('raw', 'spectral'):
        if source not in records[0]['sources']:
            continue
        refs = ('coarse','regression')+ (('raw',) if source=='spectral' else ())
        for ref in refs:
            key = source+'_minus_'+ref
            output['comparisons'][key] = {}
            for name in FIELDS:
                output['comparisons'][key][name] = {}
                for metric in ('crps', 'mse', 'bias'):
                    values = np.array([r['sources'][source]['fields'][name][metric]-r['sources'][ref]['fields'][name][metric] for r in records])
                    sums = np.array([values[blocks==b].sum() for b in labels])
                    samples = sums[chosen].sum(1)/counts[chosen].sum(1)
                    output['comparisons'][key][name][metric] = dict(difference=float(values.mean()),
                        interval_95=np.quantile(samples,[.025,.975]).tolist())
    output['status'] = 'complete'
    return output


def aggregate(output, draws=1000, block_hours=72):
    import csv
    out = Path(output); manifest = read_manifest(out)
    expected = {e['id'] for e in manifest['entries']}
    actual = {p.stem for p in (out/'cases').glob('*.json')}
    if actual != expected:
        missing, extra = sorted(expected-actual), sorted(actual-expected)
        atomic_json(out/'completion.json', dict(complete=False, expected=len(expected), completed=len(actual & expected),
                                               missing=missing, extra=extra))
        raise ValueError(f'Incomplete evaluation: missing {len(missing)}, unexpected {len(extra)} hours; see completion.json')
    records = [read_case(out/'cases'/f'{e["id"]}.json',manifest,e) for e in manifest['entries']]
    summary = dict(version=VERSION, run_id=manifest['run_id'], complete=True, hours=len(records),
                   dx_km_approx=float(np.median([r.get('dx_km_approx', 1.) for r in records])),
                   excluded=manifest['excluded'], settings=manifest['settings'],
                   units=dict(zip(FIELDS,UNITS)), pixel_quantile_probabilities=QUANTILES,
                   aggregation='Fixed-grid area weighting; equal hours; pooled MSE before RMSE; pooled contingency/FSS/reliability counts; mean hourly pixel quantiles are not pooled quantiles.',
                   sources=aggregate_sources(records), bootstrap=paired_intervals(records,draws,block_hours),
                   sampling_seconds=sum(r['sampling_seconds'] for r in records),
                   total_worker_seconds=sum(r['total_seconds'] for r in records))
    atomic_json(out/'summary.json',summary)
    rows=[]
    for source, report in summary['sources'].items():
        for field_name, values in report['fields'].items():
            rows.append(dict(source=source,field=field_name,unit=summary['units'][field_name],
                 **{k:values[k] for k in ('rmse_pooled','mae','bias','crps','fair_crps','crpss_vs_coarse',
                                       'crpss_vs_regression','coverage_90','spread_skill_corrected')}))
    with open(out/'scorecard.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['# Full CONUS evaluation', '',f'Complete: {len(records)} eligible hours; {len(manifest["excluded"])} excluded hours documented in the manifest.',
           '',f'Checkpoint SHA256: `{manifest["checkpoint_sha256"]}`. Epoch: {records[0]["epoch"]}.',
           '', 'All scores verify against the high-resolution GEOS reference, not independent observations.',
           'See scorecard.csv for all six channels plus derived wind speed; summary.json includes spectra, ranks, precipitation reliability, FSS, and paired block intervals.',
           '', 'Raw and spectral ensembles use identical noise draws. Spectral settings are fixed before evaluation; no test-truth tuning occurs.',
           'Direct-versus-residual precipitation superiority requires a separately trained, matched residual-rain ablation; this evaluator does not manufacture that comparison.',
           '', 'RMSE is computed from pooled MSE. Member FSS pools member numerator/denominator terms, not scores. Quantiles are means of hourly index-weighted quantiles.',
           'Bootstrap intervals cover paired CRPS, MSE and bias differences, use temporal blocks, and are not multiplicity-adjusted. Test block-length sensitivity.',
           'The all-domain aggregate includes ocean cells. Geographic/land-only and independent observational validation remain separate analyses.',
           'No inference-only speedup is inferred from total worker time. Sampling time excludes metric computation, baseline preparation, and spectral post-processing.']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    atomic_json(out/'completion.json',dict(complete=True,expected=len(expected),completed=len(records)))
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    init=sub.add_parser('init');init.add_argument('--config',default='configs/discover_v4_1.yaml')
    init.add_argument('--output',required=True);init.add_argument('--checkpoint',default='best')
    init.add_argument('--split',choices=('val','test'),default='test')
    init.add_argument('--members',type=int,default=16);init.add_argument('--steps',type=int,default=64)
    init.add_argument('--post',choices=('none','both'),default='both');init.add_argument('--weights',choices=('ema','raw'),default='ema')
    init.add_argument('--spectral-count',type=int,default=12);init.add_argument('--spectral-gain',type=float,default=1.3)
    init.add_argument('--spectral-cutoff',type=float,default=40.)
    init.add_argument('--save-cases',default='auto',help='auto: first/middle/last eligible hours; none, all, or comma-separated IDs')
    work=sub.add_parser('run');work.add_argument('--output',required=True)
    work.add_argument('--shard-index',type=int,default=0);work.add_argument('--shards',type=int,default=1)
    work.add_argument('--batch',type=int,default=8);work.add_argument('--threads',type=int,default=4)
    work.add_argument('--device',default=None)
    final=sub.add_parser('aggregate');final.add_argument('--output',required=True)
    final.add_argument('--draws',type=int,default=1000);final.add_argument('--block-hours',type=int,default=72)
    a=vars(p.parse_args());command=a.pop('command')
    if command=='init':
        from .v4_1 import load_config
        a['cfg']=load_config(a.pop('config'));initialize(**a)
    elif command=='run':run(**a)
    else:aggregate(**a)


if __name__=='__main__':
    main()
