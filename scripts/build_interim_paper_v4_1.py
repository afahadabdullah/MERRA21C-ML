#!/usr/bin/env python3
"""Freeze completed MERRA-Flow hours and build the paper from them.

Run from the frozen evaluation checkout. This script and its plotter
(scripts/paper_plots_v4_1.py) live outside src/ so they can be fetched without
invalidating the live evaluator, which hashes src/merraflow/*.py. It never runs
inference or writes to the parent evaluation. --snapshot redraws saved data.

Figures (scripts/paper_plots_v4_1.py) are map-first: native GEOS-FP cells versus
3-km MERRA-Flow members and the reference, zooms, ensemble probabilities, gain
maps, and aggregate skill. An optional --storm storm_fields.nc (written by
merraflow.storm_animation_v4_1) adds the storm case study. The manuscript is
updated only inside its "% BEGIN AUTO <name>" blocks.
"""
import argparse
import base64
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import zipfile

import numpy as np


def snapshot(evaluation, output, include_fields=True):
    from merraflow.evaluate_all_v4_1 import (read_manifest, read_case, digest, canonical,
        atomic_json, aggregate_sources, FIELDS, UNITS, QUANTILES)
    root, out = Path(evaluation).resolve(), Path(output).resolve()
    if root == out or root in out.parents:
        raise ValueError('Use a separate sibling output, outside the live evaluation')
    if out.exists():
        raise FileExistsError('Snapshot output already exists; use --snapshot to redraw it')
    manifest = read_manifest(root)
    expected = {e['id'] for e in manifest['entries']}
    if len(expected) != len(manifest['entries']) or any(not re.fullmatch(r'[A-Za-z0-9_-]+', key) for key in expected):
        raise ValueError('Invalid or duplicate manifest hour IDs')
    observed = {p.stem for p in (root/'cases').glob('*.json')}
    if observed-expected:
        raise ValueError('Unexpected case IDs; resolve them before making a snapshot')
    selected = [e for e in manifest['entries'] if e['id'] in observed]
    if not selected:
        raise ValueError('No completed hour records are available')
    # Freeze the ID list first. New records committed during this operation do
    # not change the subset. Every selected record must pass the real validator.
    records = [read_case(root/'cases'/f'{e["id"]}.json', manifest, e) for e in selected]
    out.mkdir(parents=True);(out/'cases').mkdir();(out/'fields').mkdir()
    shutil.copyfile(root/'manifest.json', out/'manifest.json')
    hashes = {}
    for entry, record in zip(selected, records):
        relative = f'cases/{entry["id"]}.json'
        source = root/relative
        data = source.read_bytes()
        if json.loads(data) != record:
            raise ValueError(f'Case changed during snapshot: {entry["id"]}')
        (out/relative).write_bytes(data);hashes[relative] = digest(out/relative)
    available_maps = [e['id'] for e in selected if e['id'] in manifest['settings'].get('saved_cases', [])]
    if include_fields and available_maps:
        grid = manifest['saved_grid']
        if grid['path'] != 'fields/grid.npz' or digest(root/grid['path']) != grid['sha256']:
            raise ValueError('Saved coordinate grid mismatch')
        shutil.copyfile(root/grid['path'], out/grid['path']);hashes[grid['path']] = grid['sha256']
        for record in records:
            if record['id'] not in available_maps:continue
            for item in record['saved_fields'].values():
                shutil.copyfile(root/item['path'], out/item['path'])
                if digest(out/item['path']) != item['sha256']:
                    raise ValueError('Saved example changed during copy')
                hashes[item['path']] = item['sha256']
    else:
        available_maps = []
    created = datetime.now(timezone.utc).isoformat()
    metadata = dict(scope='interim', complete=False, parent_run_id=manifest['run_id'],
        expected_hours=len(expected), hours=len(selected), entries=selected, case_sha256=hashes,
        available_map_cases=available_maps, created_utc=created,
        selection='All atomic case JSON present at snapshot start, in parent manifest order; completion-ordered, not random',
        parent_manifest_sha256=digest(out/'manifest.json'), builder_sha256=digest(__file__),
        warning='Descriptive interim subset; no full-test or significance claims. Parent run remains incomplete.')
    summary = dict(version=1, scope='interim', complete=False, run_id=manifest['run_id'],
        hours=len(records), expected_hours=len(expected),
        created_utc=created, settings=manifest['settings'], excluded=manifest['excluded'],
        units=dict(zip(FIELDS, UNITS)), pixel_quantile_probabilities=QUANTILES,
        dx_km_approx=float(np.median([r.get('dx_km_approx', 1) for r in records])),
        sources=aggregate_sources(records),
        bootstrap=dict(status='disabled_interim_completion_ordered_subset', comparisons={}, draws=0, block_hours=72, blocks=0),
        sampling_seconds=sum(r['sampling_seconds'] for r in records),
        total_worker_seconds=sum(r['total_seconds'] for r in records))
    atomic_json(out/'summary.json', summary)
    metadata['summary_sha256'] = digest(out/'summary.json')
    metadata['snapshot_id'] = hashlib.sha256(canonical(metadata).encode()).hexdigest()
    atomic_json(out/'completion.json', dict(complete=False, expected=len(expected), completed=len(records), scope='interim'))
    atomic_json(out/'snapshot.json', metadata)
    return out


def load_snapshot(root):
    from merraflow.evaluate_all_v4_1 import digest, read_manifest, canonical
    root = Path(root); metadata = json.loads((root/'snapshot.json').read_text())
    body = {k:v for k,v in metadata.items() if k != 'snapshot_id'}
    if hashlib.sha256(canonical(body).encode()).hexdigest() != metadata['snapshot_id']:
        raise ValueError('Snapshot metadata changed')
    manifest = read_manifest(root)
    if digest(root/'manifest.json') != metadata['parent_manifest_sha256']:
        raise ValueError('Parent manifest changed')
    for relative, expected in metadata['case_sha256'].items():
        if digest(root/relative) != expected:raise ValueError(f'Snapshot data changed: {relative}')
    summary = json.loads((root/'summary.json').read_text())
    if digest(root/'summary.json') != metadata['summary_sha256']:
        raise ValueError('Snapshot summary changed')
    if summary['complete'] or summary['scope'] != 'interim':
        raise ValueError('Not a matching interim summary')
    # Summary is regenerated only on initial snapshot, not on plot redraw.
    if summary['hours'] != len(metadata['entries']) or summary['run_id'] != manifest['run_id']:
        raise ValueError('Summary does not match snapshot')
    return manifest, metadata, summary


def _plots():
    """Import the publication plotter that sits next to this script (also when staged elsewhere)."""
    import importlib.util
    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location('paper_plots_v4_1', here/'paper_plots_v4_1.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def _finite(value):
    """JSON-safe figure metadata: NaN/inf become None, NumPy scalars become Python."""
    if isinstance(value, dict):
        return {str(k): _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def render(root, dpi=300, storm=None, maps=None):
    """Redraw every paper figure from the frozen snapshot (no analysis, no inference)."""
    from merraflow.evaluate_all_v4_1 import atomic_json, digest
    plots = _plots()
    manifest, meta, summary = load_snapshot(root)
    root = Path(root); out = root/'paper_figures'; out.mkdir(exist_ok=True)
    maps_manifest = deepcopy(manifest)
    maps_manifest['settings']['saved_cases'] = meta['available_map_cases']
    ids = [e['id'] for e in meta['entries']]
    subject = (f'MERRA-Flow v4.1 | {summary["hours"]}/{summary["expected_hours"]} held-out hours'
               + ('' if summary['hours'] == summary['expected_hours'] else ' (interim, completion-ordered)'))
    if maps:
        import importlib.util
        here = Path(__file__).resolve().parent
        spec = importlib.util.spec_from_file_location('evaluate_maps_v4_1', here/'evaluate_maps_v4_1.py')
        mapper = importlib.util.module_from_spec(spec); spec.loader.exec_module(mapper)
        live_maps = mapper.ensure_maps(maps)     # combine completed shards once, then freeze their map data
        frozen = root/'error_maps.npz'
        temporary = root/'error_maps.npz.tmp'
        shutil.copyfile(live_maps, temporary)
        with np.load(temporary, allow_pickle=False) as f:
            if json.loads(str(f['meta']))['run_id'] != manifest['run_id']:
                temporary.unlink()
                raise ValueError('Error maps come from a different evaluation run')
        if frozen.exists() and digest(frozen) != digest(temporary):
            temporary.unlink()
            raise ValueError('This snapshot already froze different error maps; create a new snapshot for new hours')
        temporary.replace(frozen)
        atomic_json(root/'error_maps.json', dict(sha256=digest(frozen), run_id=manifest['run_id']))
        maps = str(frozen)
    elif (root/'error_maps.npz').is_file():
        maps = str(root/'error_maps.npz')
        if digest(maps) != json.loads((root/'error_maps.json').read_text())['sha256']:
            raise ValueError('Frozen error maps changed')
    for old in out.glob('*'):
        if old.suffix in ('.pdf', '.png', '.svg'):
            old.unlink()
    figs = plots.render_all(root, maps_manifest, summary, ids, out, dpi, storm_path=storm, subject=subject, maps_path=maps)
    if storm:
        figs['storm']['sha256'] = digest(storm)
    if maps:
        figs['skill_maps']['sha256'] = digest(maps)
        with np.load(maps, allow_pickle=False) as f:
            if json.loads(str(f['meta']))['run_id'] != manifest['run_id']:
                raise ValueError('Error maps come from a different evaluation run')
    figs = _finite(figs)
    atomic_json(out/'figures.json', figs)
    atomic_json(out/'figure_data.json', summary)
    import csv
    with (out/'table01_all_variables.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['hours', 'expected_hours', 'method', 'field', 'unit', 'rmse', 'mae', 'bias', 'crps', 'fair_crps',
                         'crps_skill_vs_geosfp', 'coverage_90', 'spread_skill_corrected'])
        for method in plots.methods(summary):
            for i, field in enumerate(plots.FIELDS):
                v = summary['sources'][method]['fields'][field]
                writer.writerow([summary['hours'], summary['expected_hours'], method, field, plots.UNITS[i],
                                 *[None if v[k] is None else v[k]*plots.FACTORS[i] for k in
                                   ('rmse_pooled', 'mae', 'bias', 'crps', 'fair_crps')],
                                 v['crpss_vs_coarse'], v['coverage_90'], v['spread_skill_corrected']])
    with (out/'score_matrix.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['field', 'method', 'crps_skill_percent', 'rmse_reduction_percent',
                         'coverage_90_percent', 'hours', 'expected_hours'])
        for field in plots.FIELDS:
            base = summary['sources']['coarse']['fields'][field]['rmse_pooled']
            for method in plots.methods(summary):
                v = summary['sources'][method]['fields'][field]
                writer.writerow([field, method,
                    None if v['crpss_vs_coarse'] is None else 100*v['crpss_vs_coarse'],
                    100*(1-v['rmse_pooled']/base) if base > 0 else None,
                    None if v['coverage_90'] is None else 100*v['coverage_90'],
                    summary['hours'], summary['expected_hours']])
    atomic_json(out/'plot_manifest.json', dict(scope='interim' if summary['hours'] < summary['expected_hours'] else 'all-hours',
        snapshot_id=meta['snapshot_id'], run_id=manifest['run_id'], hours=summary['hours'],
        expected_hours=summary['expected_hours'], dpi=dpi, storm=figs.get('storm') and figs['storm'].get('sha256'),
        error_maps=figs.get('skill_maps') and figs['skill_maps'].get('sha256'),
        summary_sha256=digest(root/'summary.json'), builder_sha256=digest(__file__),
        plotter_sha256=digest(Path(__file__).resolve().parent/'paper_plots_v4_1.py'),
        files={p.name: digest(p) for p in sorted(out.iterdir()) if p.suffix in ('.pdf', '.png')}))
    return figs


# ----------------------------------------------------------------------------- manuscript text

NAMES = ('temperature', 'precipitation', 'surface pressure', 'zonal wind', 'meridional wind', 'specific humidity',
         'wind speed')
TITLES = ('Temperature', 'Precipitation', 'Surface pressure', 'Zonal wind', 'Meridional wind', 'Specific humidity',
          'Wind speed')
FACTORS = (1, 1, .01, 1, 1, 1000, 1)
UNITS = (r'K', r'mm\,h$^{-1}$', r'hPa', r'm\,s$^{-1}$', r'm\,s$^{-1}$', r'g\,kg$^{-1}$', r'm\,s$^{-1}$')
FIELD_KEYS = ('t2m', 'precip', 'ps', 'u10m', 'v10m', 'q2m', 'wind_speed')


def number(value, factor=1, digits=3):
    return '--' if value is None or not np.isfinite(value) else f'{value*factor:.{digits}g}'


def pct(value, digits=0):
    """Percentage text for a fraction, e.g. 0.384 -> '38'."""
    if value is None or not np.isfinite(value):
        return '--'
    v = 100*value
    return f'{v:.{1 if abs(v) < 10 and digits == 0 else digits}f}'


def join(items):
    items = list(items)
    if not items:
        return '--'
    return items[0] if len(items) == 1 else ', '.join(items[:-1])+', and '+items[-1]


def apply_block(source, key, text):
    start, end = f'% BEGIN AUTO {key}', f'% END AUTO {key}'
    if start not in source:
        raise ValueError(f'Manuscript lacks the "{start}" marker; copy the current local manuscript '
                         'with AUTO blocks to Discover, or use --plots-only')
    first = source.index(start); last = source.index(end, first)+len(end)
    return source[:first]+start+'\n'+text.rstrip()+'\n'+end+source[last:]


def table(caption, label, columns, headings, rows, star=True, size='\\small', sep=None):
    env = 'table*' if star else 'table'
    size = size+(f'\\setlength{{\\tabcolsep}}{{{sep}}}' if sep else '')
    return (f'\\begin{{{env}}}[t]\n\\caption{{{caption}}}\\label{{{label}}}\n\\centering{size}\n'
            + '\\begin{tabular}{'+columns+'}\\toprule\n'+' & '.join(headings)+r' \\ \midrule'+'\n'
            + '\n'.join(' & '.join(row)+r' \\' for row in rows)+'\n\\bottomrule\\end{tabular}\n'+f'\\end{{{env}}}\n')


def figure(stem, caption, label, placement='t', width='\\textwidth'):
    name = stem+'.pdf'
    preview = stem+'_preview.pdf'
    return (f'\\begin{{figure*}}[{placement}]\\centering\n'
            + '\\IfFileExists{figures/auto/'+name+'}{\\includegraphics[width='+width+']{figures/auto/'+name+'}}{%\n'
            + '\\IfFileExists{manuscript/figures/auto/'+name+'}{\\includegraphics[width='+width+']{manuscript/figures/auto/'+name+'}}'
            + '{\\includegraphics[width='+width+']{'+preview+'}}}\n\\caption{'+caption+'}\\label{'+label+'}\n\\end{figure*}\n')


def inline_pdf(path, width=518.4, max_pixels=1800):
    """ASCII PDF fallback for the editor's single-source compiler (PNG preview, RGB)."""
    from PIL import Image
    import zlib
    with Image.open(Path(path).with_suffix('.png')) as image:
        rgb = image.convert('RGB')
        if rgb.size[0] > max_pixels:
            rgb = rgb.resize((max_pixels, round(rgb.size[1]*max_pixels/rgb.size[0])), Image.LANCZOS)
        pixels = rgb.tobytes(); w, h = rgb.size
    stream = base64.a85encode(zlib.compress(pixels, 9), wrapcol=100)+b'~>\n'; height = width*h/w
    objects = [b'<< /Type /Catalog /Pages 2 0 R >>', b'<< /Type /Pages /Kids [3 0 R] /Count 1 >>',
               f'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width} {height}] /Resources << /XObject << /Im 4 0 R >> >> /Contents 5 0 R >>'.encode(),
               f'<< /Type /XObject /Subtype /Image /Width {w} /Height {h} /ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter [/ASCII85Decode /FlateDecode] /Length {len(stream)} >>\nstream\n'.encode()+stream+b'endstream']
    content = f'q {width} 0 0 {height} 0 0 cm /Im Do Q\n'.encode()
    objects.append(f'<< /Length {len(content)} >>\nstream\n'.encode()+content+b'endstream')
    result = b'%PDF-1.4\n'; offsets = [0]
    for i, obj in enumerate(objects, 1):
        offsets.append(len(result)); result += f'{i} 0 obj\n'.encode()+obj+b'\nendobj\n'
    xref = len(result); result += f'xref\n0 {len(objects)+1}\n0000000000 65535 f \n'.encode()
    result += b''.join(f'{p:010d} 00000 n \n'.encode() for p in offsets[1:])
    result += f'trailer\n<< /Size {len(objects)+1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n'.encode()
    return result.decode('ascii')


def when(time):
    """'2026-03-07T19:30:00' -> '19:30 UTC on 7 March 2026'."""
    t = datetime.fromisoformat(str(time)[:19])
    return f'{t:%H:%M} UTC on {t.day} {t:%B %Y}'


class Context:
    def __init__(self, manifest, meta, summary, figs):
        self.m, self.meta, self.s, self.f = manifest, meta, summary, figs
        self.n, self.N = summary['hours'], summary['expected_hours']
        self.complete = self.n == self.N
        self.src = summary['sources']
        self.methods = [m for m in ('coarse', 'regression', 'raw', 'spectral') if m in self.src]
        self.M = summary['settings']['members']
        if self.complete:
            self.scope = f'all {self.N} eligible held-out hours'
            self.scope_cap = f'All {self.N} eligible held-out hours.'
        else:
            self.scope = (f'{self.n} of {self.N} eligible held-out hours ({100*self.n/self.N:.0f}\\%; '
                          'an interim, completion-ordered subset)')
            self.scope_cap = (f'Interim snapshot: {self.n} of {self.N} eligible held-out hours '
                              f'({100*self.n/self.N:.0f}\\%, completion-ordered).')

    def v(self, method, field, key):
        return self.src[method]['fields'][field][key]

    def skill(self, method='raw'):
        return [self.v(method, f, 'crpss_vs_coarse') for f in FIELD_KEYS]

    def rmse_red(self, method='raw'):
        out = []
        for f in FIELD_KEYS:
            c = self.v('coarse', f, 'rmse_pooled')
            out.append(1-self.v(method, f, 'rmse_pooled')/c if c > 0 else None)
        return out


def settings_table(c):
    cfg = c.m['config']; patch = cfg['patch']; model = cfg.get('model', {})
    rows = [['Evaluated hours', c.scope],
            ['Members / Heun steps', f"{c.M} / {c.s['settings']['steps']}"],
            ['Weights', f"{c.s['settings']['weights'].upper()} (frozen checkpoint, hash in manifest)"],
            ['Core / halo / context', f"{patch['size']} / {patch['halo']} px; {(patch['size']+2*patch['halo'])*patch.get('context_scale', 3)} px context pooled to {patch['context_size']}"],
            ['Network', f"U-Net, base width {model.get('base_channels', 64)}, multipliers {tuple(model.get('channel_mult', [1, 2, 3, 4]))}"],
            ['Tile stride', f"{patch['stride']} px with Hann blending"],
            ['Spectral variant', 'paired with identical members' if c.s['settings']['post'] == 'both' else 'not evaluated'],
            ['Saved example hours', ', '.join(x.replace('_', r'\_') for x in c.m['settings'].get('saved_cases', [])) or 'none']]
    rows = [[a, b] for a, b in rows]
    return table('Frozen evaluation and model settings, read from the evaluation manifest.', 'tab:settings',
                 'lp{0.62\\textwidth}', ['Setting', 'Value'], rows, star=True)


def scores_table(c):
    has_sp = 'spectral' in c.src
    head = ['Variable', 'Unit', 'GEOS-FP MAE', 'Regr. MAE', 'Flow CRPS'] + (['+spec. CRPS'] if has_sp else []) + \
           ['CRPSS (\\%)', 'GEOS-FP RMSE', 'Flow RMSE', '90\\% cov.']
    rows = []
    for i, f in enumerate(FIELD_KEYS):
        k = FACTORS[i]
        row = [TITLES[i], UNITS[i], number(c.v('coarse', f, 'mae'), k), number(c.v('regression', f, 'mae'), k),
               number(c.v('raw', f, 'crps'), k)]
        if has_sp:
            row.append(number(c.v('spectral', f, 'crps'), k))
        row += [pct(c.v('raw', f, 'crpss_vs_coarse')), number(c.v('coarse', f, 'rmse_pooled'), k),
                number(c.v('raw', f, 'rmse_pooled'), k), pct(c.v('raw', f, 'coverage_90'))]
        rows.append(row)
    caption = (c.scope_cap+' Deterministic predictions are scored by MAE, which equals their CRPS. '
               f'Flow scores use {c.M} members; CRPSS is the raw-flow CRPS skill relative to GEOS-FP MAE. '
               'RMSE is for the ensemble mean, pooled over hours. Coverage is the fraction of reference values inside the '
               'central 90\\% ensemble interval.' + ('' if c.complete else ' No significance intervals are given for an interim subset.'))
    return table(caption, 'tab:scores', 'll'+'r'*(len(head)-2), head, rows, size='\\footnotesize', sep='3pt')


def processing_table(c):
    if 'spectral' not in c.src:
        return ''
    rows = []
    for i, f in enumerate(FIELD_KEYS):
        r, s = c.src['raw']['fields'][f], c.src['spectral']['fields'][f]
        g = lambda v: number(v['gradient_p99_members_mean']/v['gradient_p99_reference']) if v['gradient_p99_reference'] > 0 else '--'
        rows.append([TITLES[i], number(s['crps']-r['crps'], FACTORS[i]), number(r['bias'], FACTORS[i]), number(s['bias'], FACTORS[i]),
                     g(r), g(s)])
    return table('Paired effect of the spectral adjustment (identical members). $\\Delta$CRPS is adjusted minus raw in '
                 'physical units (negative favors the adjustment). Gradient ratios compare the members\' 99th-percentile '
                 'gradient magnitude with the reference.', 'tab:spectral', 'lrrrrr',
                 ['Variable', '$\\Delta$CRPS', 'Raw bias', 'Adjusted bias', 'Raw gradient', 'Adjusted gradient'], rows)


CASE_NOTES = {
    # Human interpretation of specific archived hours (used only when that hour is the main case).
    '20260203_0130': dict(
        overview='an evening hour during a cold-air outbreak over the northern Plains and Great Lakes',
        zoomB=('Over northern Ontario downwind of Lake Huron (box B), the coarse fields show a diffuse humidity and '
               'snowfall pattern, whereas the member generates sharp lake-shore moisture gradients and cellular, banded '
               'light precipitation similar in character to the lake-effect bands of the reference.'),
        gain=', and under the cold air over the Great Lakes',
        zoomB_short=(', and lake-shore moisture gradients and banded light snowfall downwind of Lake Huron, similar in '
                     'character to the lake-effect bands of the reference (box B)')),
}


def reliability_error(c, method, threshold):
    """Area-weighted mean |observed frequency - forecast probability| over issued probability bins."""
    r = c.src[method]['precipitation'].get(threshold)
    if not r:
        return None
    pairs = [(a, abs(o-p)) for a, o, p in zip(r['reliability_area'], r['reliability_frequency'], r['probability_bins'])
             if o is not None and a > 0 and p > 0]   # issued (non-zero) probabilities only
    total = sum(a for a, _ in pairs)
    return sum(a*e for a, e in pairs)/total if total else None


def _ratio(a, b):
    return a/b if (a is not None and b not in (None, 0)) else None


def brier_skill(c, method, threshold):
    p = c.src[method]['precipitation'].get(threshold)
    base = c.src['coarse']['precipitation'].get(threshold)
    if not p or not base or not base['brier']:
        return None
    return 1-p['brier']/base['brier']


def fss_value(c, method, threshold, scale, kind='ensemble_mean'):
    try:
        v = c.src[method]['precipitation'][threshold]['fss'][str(scale)][kind]
    except KeyError:
        return None
    return v


def grid_km(c):
    return float(c.f.get('grid_km') or 3.)


def headline(c):
    """Numbers that the abstract, key points, results and conclusions share."""
    sk, rsk = c.skill('raw'), c.skill('regression')
    i = {k: n for n, k in enumerate(FIELD_KEYS)}
    core = [k for k in ('t2m', 'u10m', 'v10m', 'q2m', 'wind_speed') if sk[i[k]] is not None]
    lo = min(core, key=lambda k: sk[i[k]]); hi = max(core, key=lambda k: sk[i[k]])
    mult = [_ratio(sk[i[k]], rsk[i[k]]) for k in ('t2m', 'u10m', 'v10m', 'wind_speed', 'precip')
            if rsk[i[k]] is not None and rsk[i[k]] > .01]
    t = c.f['temporal']
    sp = c.f['spectra']['ratio']
    return dict(sk=sk, rsk=rsk, i=i, lo=lo, hi=hi, mult=[m for m in mult if m],
                pos_hours=min(v for v in t['positive_fraction'].values() if v is not None),
                p10=t.get('p10', {}), sp=sp)


def results(c):
    """Compact, findings-first results section (numbers checked against the frozen scores)."""
    f = c.f; s = []
    h = headline(c); sk, rsk, i = h['sk'], h['rsk'], h['i']
    red = c.rmse_red('raw')
    dx = grid_km(c)
    cases = f.get('cases') or []
    main = next((x for x in cases if x.get('main')), None)
    t = f['temporal']
    s.append('\\section{Results}\\label{sec:results}\n')
    s.append(f'Aggregate scores use {c.scope}, each with {c.M} members and {c.s["settings"]["steps"]} Heun steps; '
             'maps come from hours archived before inference.\n\n')

    # 1. Skill
    s.append('\\subsection{Large, consistent gains over GEOS-FP}\\label{sec:skill}\n')
    s.append('Relative to interpolated GEOS-FP, \\model{} reduces CRPS by '
             + join([f'{pct(sk[i[k]])}\\% for {NAMES[i[k]]}' for k in ('t2m', 'q2m', 'wind_speed')])
             + f", {pct(sk[i['u10m']])}--{pct(sk[i['v10m']])}\\% for the wind components, {pct(sk[i['precip']])}\\% for "
             f"precipitation, and {pct(sk[i['ps']])}\\% for terrain-sensitive surface pressure (Table~\\ref{{tab:scores}}, "
             'Fig.~\\ref{fig:scorecard}). ')
    if h['mult']:
        s.append(f"These gains are {min(h['mult']):.1f}--{max(h['mult']):.1f} times those of the frozen regression that "
                 'conditions the generator, and for humidity, where the regression has no output, the generator alone '
                 f"supplies a {pct(sk[i['q2m']])}\\% improvement. ")
    better_rmse = sum(1 for v in red if v is not None and v > 0)
    s.append(f"The ensemble mean also lowers RMSE for {better_rmse} of {len(FIELD_KEYS)} fields "
             f"({pct(red[i['t2m']])}\\% for temperature, {pct(red[i['q2m']])}\\% for humidity). "
             f"Skill is positive in {h['pos_hours']:.0f}\\% of the {t['hours']} hours for every field")
    p10 = h['p10']
    if p10:
        worst = min((k for k in FIELD_KEYS if p10.get(k) is not None), key=lambda k: p10[k])
        s.append(f", with a 10th percentile of {p10[worst]:.0f}\\% for the weakest field ({NAMES[i[worst]]})")
    di = (t.get('diurnal') or {}).get('t2m')
    if di:
        hi_h = max(di, key=lambda k: di[k]); lo_h = min(di, key=lambda k: di[k])
        s.append(f"; temperature skill peaks at {int(hi_h):02d}~UTC ({di[hi_h]:.0f}\\%) and is lowest at "
                 f"{int(lo_h):02d}~UTC ({di[lo_h]:.0f}\\%)"
                 + (', consistent with the largest sub-grid contrasts under shallow nocturnal boundary layers'
                    if 0 <= int(hi_h) <= 12 and 13 <= int(lo_h) <= 20 else ''))
    s.append(' (Appendix Fig.~\\ref{fig:temporal}).\n\n')
    s.append(scores_table(c))
    s.append(figure(f['scorecard']['stem'],
        c.scope_cap+' (a) CRPS skill (left) and ensemble-mean RMSE reduction (right) relative to interpolated GEOS-FP for '
        'the regression, \\model{} (Flow), and the spectrally adjusted ensemble; blue is better than GEOS-FP. (b) Central '
        '90\\% interval coverage; dashed line, nominal.', 'fig:scorecard'))

    sm = f.get('skill_maps')
    if sm:
        nm = sm['numbers']; names = dict(zip(FIELD_KEYS, NAMES))
        state = [k for k in ('t2m', 'wind_speed', 'q2m') if k in nm]
        s.append('\\subsection{Where downscaling helps}\\label{sec:skillmaps}\n')
        span = (f"{sm['hours']} test hours (every {sm['every']}th hour, {sm['members']} members)"
                if sm.get('source') != 'saved_fields' else
                f"the {sm['hours']} archived example hour{'s' if sm['hours'] != 1 else ''} ({sm['members']} members)")
        s.append(f"Figure~\\ref{{fig:skillmaps}} maps the time-mean errors over {span}. GEOS-FP errors concentrate over complex terrain, "
                 'along coasts and lake shores, and in the main storm tracks; \\model{} removes much of that error, and '
                 'CRPS skill is positive over ' + join([f"{pct(nm[k]['positive_area'])}\\% of the domain for {names[k]}"
                                                        for k in state]) + '. ')
        gain_hi = [k for k in state if nm[k]['skill_top_decile'] is not None and nm[k]['skill_rest'] is not None
                   and nm[k]['skill_top_decile'] > nm[k]['skill_rest']]
        if gain_hi:
            rest_scores = join([pct(nm[k]['skill_rest'])+r'\%' for k in gain_hi])
            s.append('Skill is highest where GEOS-FP errs most: in the 10\\% of the domain with the largest GEOS-FP error it '
                     'reaches ' + join([f"{pct(nm[k]['skill_top_decile'])}\\% for {names[k]}" for k in gain_hi])
                     + f" (versus {rest_scores} elsewhere). ")
        if 'precip' in nm:
            s.append(f"For precipitation, skill is positive over {pct(nm['precip']['positive_area'])}\\% of the area "
                     'where rain errors are non-negligible. ')
        s.append('\n\n')
        s.append(figure(sm['stem'],
            ("Where downscaling helps: time means over " + span
             + ('' if sm.get('complete', True) or sm.get('source') == 'saved_fields'
                else f"; {sm['hours']} of {sm['planned']} planned hours finished") + '. ')
            + 'Columns: mean absolute error of interpolated GEOS-FP; mean \\model{} CRPS (same scale); CRPS skill relative '
            'to GEOS-FP (labels: domain mean); ensemble-mean RMSE reduction. Fields are averaged over $4\\times4$ blocks; '
            'grey marks blocks where GEOS-FP rain error is below 0.01~mm\\,h$^{-1}$, where skill is undefined.',
            'fig:skillmaps'))

    # 2. Maps
    if main:
        note = CASE_NOTES.get(main['id'], {})
        native = main['overview'].get('native', True)
        cell_note = ('' if native else ' Native GEOS-FP files were not archived for this hour; the GEOS-FP column shows '
                     'the interpolated input averaged onto the $0.25^\\circ\\times0.3125^\\circ$ cells.')
        s.append('\\subsection{What the generator adds}\\label{sec:maps}\n')
        s.append(f"Figure~\\ref{{fig:overview}} shows {when(main['time'])}"
                 + (', '+note['overview'] if note.get('overview') else '') + '. GEOS-FP renders mountains, lakes, and '
                 'coastlines as staircases of $\\sim$25-km cells. A single \\model{} member reproduces what the coarse '
                 'field cannot carry---cold valleys and warm slopes across the West, sharp contrasts along coasts and '
                 'lake shores, terrain-channelled winds---with the placement and amplitude of the reference. The zooms '
                 '(Fig.~\\ref{fig:zoom}) make the resolution ladder explicit: interpolation smooths the cells, the '
                 'regression recovers the main ridges, and the member adds the valley network and wind channels of the '
                 'reference over the Colorado Rockies' + (note.get('zoomB_short', '') or '') + '.\n\n')
        s.append(figure(main['overview']['stem'],
            f"Downscaling GEOS-FP for {when(main['time'])}: 2-m temperature, 10-m wind speed, precipitation rate, and 2-m "
            'specific humidity for GEOS-FP on $0.25^\\circ\\times0.3125^\\circ$ cells, \\model{} member 1, and the '
            'GEOS-CAM reference (rows share a color scale; $2\\times2$ block means). Boxes mark the zooms of '
            'Fig.~\\ref{fig:zoom}.' + cell_note, 'fig:overview'))
        s.append(figure(main['zoom']['stem'],
            'Resolution ladder in box A (fixed Colorado Rockies window) and box B (centered on the heaviest GEOS-FP '
            'precipitation, chosen from the input only): GEOS-FP cells (outlined), bilinear GEOS-FP, regression, a '
            '\\model{} member' + (', the same member with spectral adjustment' if 'spectral' in c.src else '')
            + ', and the reference, at full 3-km resolution.', 'fig:zoom'))

    # 3. Scales
    sp = h['sp']
    band = [f"{pct(sp[k]['raw'])}\\% for {NAMES[i[k]]}" for k in ('t2m', 'wind_speed', 'q2m', 'precip')
            if sp.get(k, {}).get('raw') is not None]
    coarse_band = [sp.get(k, {}).get('coarse') for k in ('t2m', 'wind_speed', 'q2m')]
    s.append('\\subsection{Restored kilometer-scale variability}\\label{sec:scales}\n')
    s.append('The spectra (Fig.~\\ref{fig:spectra}) explain where the skill comes from. All predictions follow the '
             'reference at synoptic scales; below the GEOS-FP grid scale interpolation loses essentially all variance '
             'and the regression retains only part of it, while \\model{} members keep the reference spectral slope '
             'nearly to the grid scale. ')
    if band and all(v is not None for v in coarse_band):
        s.append(f"Between {2*dx:.0f} and 54~km, member power is " + join(band) + ' of the reference, versus '
                 f"{pct(min(coarse_band), 1)}--{pct(max(coarse_band), 1)}\\% for interpolated GEOS-FP and "
                 f"{pct(sp['t2m'].get('regression'))}\\% (temperature) for the regression. ")
    g = {k: _ratio(c.v('raw', k, 'gradient_p99_members_mean'), c.v('raw', k, 'gradient_p99_reference')) for k in ('t2m', 'q2m')}
    gc = {k: _ratio(c.v('coarse', k, 'gradient_p99_members_mean'), c.v('coarse', k, 'gradient_p99_reference')) for k in ('t2m', 'q2m')}
    if None not in list(g.values())+list(gc.values()):
        s.append(f"Extreme gradients follow: the 99th-percentile gradient reaches {g['t2m']:.2f} (temperature) and "
                 f"{g['q2m']:.2f} (humidity) of the reference, against {gc['t2m']:.2f} and {gc['q2m']:.2f} for GEOS-FP. "
                 'Realistic structure and better proper scores therefore come together.\n\n')
    s.append(figure(f['spectra']['stem'],
        c.scope_cap+' Hour-averaged radial power spectra (top) and their ratio to the reference (bottom) versus '
        f"wavelength ({dx:.1f}-km grid). Shading marks scales below twice the GEOS-FP grid spacing.", 'fig:spectra'))

    # 4. Precipitation
    s.append('\\subsection{Reliable rain probabilities and recovered extremes}\\label{sec:precip}\n')
    bss = {th: (brier_skill(c, 'raw', th), brier_skill(c, 'regression', th)) for th in ('0.1', '1.0', '5.0', '10.0')}
    bss = {k: v for k, v in bss.items() if v[0] is not None}
    rel = reliability_error(c, 'raw', '1.0')
    if bss:
        s.append('Precipitation separates generative from deterministic downscaling most clearly. Member exceedance '
                 'fractions are reliable probabilities'
                 + (f" (mean departure from the diagonal {rel:.2f} at 1~mm\\,h$^{{-1}}$)" if rel is not None else '')
                 + ' and improve the Brier score of GEOS-FP by '
                 + join([f"{pct(v[0])}\\% at {float(k):g}" for k, v in bss.items()]) + '~mm\\,h$^{-1}$, against at most '
                 + f"{pct(max(v[1] for v in bss.values() if v[1] is not None))}\\% for the regression. ")
    q = f['precip']['tails']; probs = f['precip']['quantiles']; top = len(probs)-1
    if q.get('reference') and q.get('raw'):
        s.append(f"The {100*probs[top]:g}th-percentile hourly rate is {number(q['raw'][top], 1, 2)}~mm\\,h$^{{-1}}$ in "
                 f"\\model{{}} members versus {number(q['reference'][top], 1, 2)} in the reference, "
                 f"{number(q['coarse'][top], 1, 2)} in GEOS-FP, and {number(q['regression'][top], 1, 2)} in the regression, "
                 'which washes out heavy rain as any conditional-mean estimator does. ')
    w = lambda km: min((1, 5, 17, 33, 65), key=lambda p: abs(p*dx-km))
    reg5, raw5 = fss_value(c, 'regression', '5.0', w(50)), fss_value(c, 'raw', '5.0', w(50))
    mem10, co10 = fss_value(c, 'raw', '10.0', w(100), 'members'), fss_value(c, 'coarse', '10.0', w(100))
    if None not in (reg5, raw5, mem10, co10):
        s.append(f"Neighbourhood skill shows the same contrast (Fig.~\\ref{{fig:precip}}): at 5~mm\\,h$^{{-1}}$ and "
                 f"{w(50)*dx:.0f}~km the regression's FSS collapses to {number(reg5, 1, 2)} while the ensemble mean keeps "
                 f"{number(raw5, 1, 2)}, and at 10~mm\\,h$^{{-1}}$ individual members, which keep intense cores, reach "
                 f"{number(mem10, 1, 2)} at {w(100)*dx:.0f}~km versus {number(co10, 1, 2)} for GEOS-FP. ")
    if main and main.get('ensemble'):
        th = main['ensemble'].get('threshold', 1.)
        s.append('For a single hour (Fig.~\\ref{fig:ensemble}), one coarse input yields distinct, equally detailed '
                 f"realizations whose exceedance fraction at {th:g}~mm\\,h$^{{-1}}$ maps where rain is likely, to be read "
                 'against the reference area (red).')
    s.append('\n\n')
    s.append(figure(f['precip']['stem'],
        c.scope_cap+' (a--c) Fractions skill score versus neighbourhood width at 1, 5, and 10~mm\\,h$^{-1}$ (solid: '
        'ensemble mean or deterministic field; dotted: individual members). (d) Pixel percentiles of rain rate.',
        'fig:precip'))
    if main and main.get('ensemble'):
        th = main['ensemble'].get('threshold', 1.)
        s.append(figure(main['ensemble']['stem'],
            f"One input, many outcomes (box B, {when(main['time'])}): GEOS-FP cells, three members, the reference, the "
            f"{main['members']}-member mean and spread, and the fraction of members $\\geq${th:g}~mm\\,h$^{{-1}}$ with the "
            'reference contour in red. The threshold is set from the GEOS-FP input only.', 'fig:ensemble'))

    storm = f.get('storm')
    if storm:
        s.append(storm_section(c, storm))

    # 5. Calibration (one paragraph)
    cov = [c.v('raw', fk, 'coverage_90') for fk in FIELD_KEYS]
    ssr = [c.v('raw', fk, 'spread_skill_corrected') for fk in FIELD_KEYS]
    s.append(f"\\paragraph{{Calibration.}} The ensemble is moderately under-dispersive: central 90\\% intervals cover "
             f"{pct(min(cov))}--{pct(max(cov))}\\% of reference values and corrected spread--error ratios are "
             f"{number(min(ssr), 1, 2)}--{number(max(ssr), 1, 2)} (Appendix Fig.~\\ref{{fig:calibration}}). ")
    if 'spectral' in c.src:
        dc = max(abs(c.v('spectral', k, 'crps')/c.v('raw', k, 'crps')-1) for k in FIELD_KEYS if c.v('raw', k, 'crps'))
        pc = (c.v('raw', 'precip', 'coverage_90'), c.v('spectral', 'precip', 'coverage_90'))
        s.append(f"The optional spectral adjustment changes CRPS by at most {100*dc:.1f}\\% but widens the rain intervals "
                 f"(coverage {pct(pc[0])}\\% $\\rightarrow$ {pct(pc[1])}\\%).")
    s.append('\n\n\\FloatBarrier\n')
    return ''.join(s)


def storm_section(c, storm):
    s = []
    ev = storm['evolution']['hours']; pk = next((h for h in ev if h['time'].startswith(storm['peak'][:16])), ev[-1])
    se = storm['series']['maxima']
    s.append('\\subsection{Storm case study}\\label{sec:storm}\n')
    s.append(f"Figures~\\ref{{fig:storm}}--\\ref{{fig:stormseries}} follow a held-out storm through {storm['hours']} "
             f"consecutive hours ({when(storm['start'])} to {when(storm['end'])}) with {storm['members']} members. At the "
             f"peak hour the window maximum rain rate is {pk['geos_max']:.0f}~mm\\,h$^{{-1}}$ in GEOS-FP, "
             f"{pk['member_max']:.0f}~mm\\,h$^{{-1}}$ in member 1, and {pk['truth_max']:.0f}~mm\\,h$^{{-1}}$ in the reference. "
             f"Over the event the largest 99.9th-percentile rate is {se['truth'][1]:.0f}, {se['member'][1]:.0f}, and "
             f"{se['geos'][1]:.0f}~mm\\,h$^{{-1}}$ for the reference, the member mean, and GEOS-FP.\n\n")
    s.append(figure(storm['evolution']['stem'],
        'Evolution of the storm case: native GEOS-FP cells, two \\model{} members, and the reference; shading is '
        'precipitation rate, contours terrain-reduced sea-level pressure (4 hPa).', 'fig:storm', placement='p'))
    s.append(figure(storm['peak_fields']['stem'],
        f"Storm structure at the peak hour ({when(storm['peak'])}).", 'fig:stormpeak'))
    s.append(figure(storm['series']['stem'],
        'Window statistics through the storm: heavy-rain area, 99.9th-percentile rain rate, and 99th-percentile wind speed.',
        'fig:stormseries'))
    if storm.get('totals'):
        s.append(figure(storm['totals']['stem'], 'Event-total precipitation over the storm window.', 'fig:stormtotals'))
    return ''.join(s)


def appendix(c):
    f = c.f; s = ['\\section{Additional diagnostics}\\label{app:calibration}\n']
    s.append(figure(f['temporal']['stem'],
        c.scope_cap+' (a) Daily-mean CRPS skill relative to GEOS-FP; the strip shows eligible (light) and evaluated (dark) '
        'hours per day. (b) Mean skill by UTC hour. (c) Hourly skill distribution. (d) Hours with positive skill.',
        'fig:temporal'))
    s.append(figure(f['calibration']['stem'],
        c.scope_cap+' (a, b) Relative rank frequency of the reference among the members (1 = uniform) for the raw and '
        'spectrally adjusted ensembles. (c) Corrected spread--error ratio. (d, e) Rain reliability at 1 and '
        '5~mm\\,h$^{-1}$ (inset: issued-probability area). (f) Brier skill relative to GEOS-FP by threshold.',
        'fig:calibration'))
    for case in (f.get('cases') or []):
        if case.get('main'):
            continue
        s.append(figure(case['overview']['stem'],
            f"As Fig.~\\ref{{fig:overview}}, for the archived example hour {when(case['time'])}.",
            'fig:case_'+case['id'].replace('_', '')))
    return ''.join(s)


def abstract(c):
    h = headline(c); sk, i = h['sk'], h['i']
    text = (f"On {c.scope} from February--March 2026, the {c.M}-member ensemble reduces the continuous ranked probability "
            f"score of interpolated GEOS-FP by {pct(sk[i[h['lo']]])}--{pct(sk[i[h['hi']]])}\\% for temperature, humidity, and "
            f"winds, by {pct(sk[i['precip']])}\\% for precipitation, and by {pct(sk[i['ps']])}\\% for surface pressure")
    if h['mult']:
        text += (f"---{min(h['mult']):.1f} to {max(h['mult']):.1f} times the improvement of the deterministic regression it "
                 'builds on')
    text += f", with positive skill in {h['pos_hours']:.0f}\\% of evaluated hours for every field. "
    sp = h['sp']
    keys = ('t2m', 'wind_speed', 'q2m')
    if all(sp.get(k, {}).get('raw') is not None and sp.get(k, {}).get('coarse') is not None for k in keys):
        text += (f"Generated members retain {pct(min(sp[k]['raw'] for k in keys))}--{pct(max(sp[k]['raw'] for k in keys))}\\% "
                 'of the reference variance at 6--54~km wavelengths for these fields, where interpolated GEOS-FP retains '
                 f"at most {pct(max(sp[k]['coarse'] for k in keys), 1)}\\%. ")
    b1 = brier_skill(c, 'raw', '1.0')
    q = c.f['precip']['tails']
    if b1 is not None and q.get('reference') and q.get('raw'):
        text += (f"Precipitation exceedance probabilities are reliable and improve the Brier score by {pct(b1)}\\% at "
                 '1~mm\\,h$^{-1}$, and the 99.9th-percentile hourly rate recovers to '
                 f"{number(q['raw'][-1], 1, 2)}~mm\\,h$^{{-1}}$ (reference {number(q['reference'][-1], 1, 2)}; regression "
                 f"{number(q['regression'][-1], 1, 2)}). ")
    if not c.complete:
        text += 'These figures come from an interim subset of the held-out period; the complete evaluation is in progress.'
    return text


def keypoints(c):
    h = headline(c); sk, i = h['sk'], h['i']
    items = ['A six-variable conditional flow-matching model downscales hourly GEOS-FP from $\\sim$25-km cells to a 3-km '
             'CONUS grid and generates multi-member ensembles from a single coarse input.',
             f"CRPS improves on GEOS-FP by {pct(sk[i[h['lo']]])}--{pct(sk[i[h['hi']]])}\\% for temperature, humidity, and "
             f"winds and by {pct(sk[i['precip']])}\\% for precipitation, in {h['pos_hours']:.0f}\\% of hours"
             + (f", {min(h['mult']):.1f}--{max(h['mult']):.1f} times the regression baseline." if h['mult'] else '.'),
             'Members restore kilometer-scale variance, sharp gradients, and heavy-rain tails that interpolation and '
             'regression remove, with reliable precipitation probabilities.']
    return '\\begin{itemize}\n'+'\n'.join('\\item '+x for x in items)+'\n\\end{itemize}'


def conclusions(c):
    h = headline(c); sk, i = h['sk'], h['i']
    text = (f"On {c.scope}, three results stand out. First, the generated ensembles beat interpolated GEOS-FP under CRPS "
            f"for every field and in {h['pos_hours']:.0f}\\% of hours, by {pct(sk[i[h['lo']]])}--{pct(sk[i[h['hi']]])}\\% for "
            f"temperature, humidity, and winds and {pct(sk[i['precip']])}\\% for precipitation"
            + (f", {min(h['mult']):.1f}--{max(h['mult']):.1f} times the gain of the deterministic regression" if h['mult'] else '')
            + '. Second, the members restore the kilometer-scale variance and sharp gradients that the 25-km grid cannot '
            'carry, so improved scores and realistic structure come together rather than being traded against each other. '
            'Third, for precipitation the ensemble yields reliable exceedance probabilities and recovers much of the '
            'heavy-rain tail that a conditional-mean regression removes. The main remaining weakness is moderate '
            'under-dispersion.')
    if not c.complete:
        text += ' These conclusions rest on an interim subset and will be confirmed with the complete held-out evaluation.'
    return text


def storm_method(c):
    st = c.f.get('storm')
    if not st:
        return '% No storm case study was supplied to the builder (--storm).'
    window = 'a window that follows the surface low' if st.get('tracked') else 'a fixed regional window'
    return ('\\subsection{Storm case study}\n'
            f"To examine a high-impact event in time, a separate run with the same frozen checkpoint and sampler generates "
            f"{st['hours']} consecutive hours around the selected held-out hour ({when(st['peak'])}) on {window}, writing "
            f"native GEOS-FP cells, {st['members']} members, and the reference. "
            + ('Members are raw flow output. ' if st['post'] == 'none' else 'Members include the spectral adjustment. ')
            + 'Fixed noise seeds can retain member identity across hours but do not establish learned temporal coherence. '
            'Sea-level pressure is reduced from surface pressure, 2-m temperature, and elevation for display only.\n')


def update_paper(root, manuscript, figs=None, inline=True):
    manifest, meta, summary = load_snapshot(root)
    root = Path(root); path = Path(manuscript)
    if figs is None:
        figs = json.loads((root/'paper_figures'/'figures.json').read_text())
    c = Context(manifest, meta, summary, figs)
    source = path.read_text()
    blocks = dict(
        ABSTRACT=abstract(c), KEYPOINTS=keypoints(c),
        SCOPE='\\noindent{\\small\\color{muted}Evaluation scope: '+c.scope+'. '
              + ('' if c.complete else 'Numbers will be updated when the full held-out evaluation completes. ')
              + 'Author declarations remain to be completed.}',
        TABLE_SETTINGS=settings_table(c), TABLE_SCORES='% Table~\\ref{tab:scores} is placed with the results.',
        RESULTS=results(c), CONCLUSIONS=conclusions(c), APPENDIX=appendix(c), STORM_METHOD=storm_method(c))
    for key, text in blocks.items():
        source = apply_block(source, key, text)
    stems = sorted(set(re.findall(r'figures/auto/([A-Za-z0-9_\-]+)\.pdf', source)))
    assets = path.parent/'figures'/'auto'; assets.mkdir(parents=True, exist_ok=True)
    for old in assets.glob('fig_*'):
        old.unlink()
    previews = ''
    for stem in stems:
        shutil.copyfile(root/'paper_figures'/f'{stem}.pdf', assets/f'{stem}.pdf')
        shutil.copyfile(root/'paper_figures'/f'{stem}.png', assets/f'{stem}.png')
        if inline:
            previews += ('\\begin{filecontents*}[overwrite]{'+stem+'_preview.pdf}\n'
                         + inline_pdf(assets/f'{stem}.pdf')+'\\end{filecontents*}\n')
    source = apply_block(source, 'FIGURES', previews or '% (no inline previews)')
    backup = root/'manuscript_before_update.tex'
    if not backup.exists():
        shutil.copyfile(path, backup)
    temporary = path.with_name(path.name+'.auto.tmp')
    temporary.write_text(source); temporary.replace(path)
    (root/'manuscript_update.json').write_text(json.dumps(dict(
        manuscript=str(path), snapshot_id=meta['snapshot_id'], hours=c.n, expected_hours=c.N, figures=stems), indent=2)+'\n')
    return path


def compile_paper(root, manuscript):
    path=Path(manuscript).resolve();out=Path(root).resolve()/'pdf';out.mkdir(exist_ok=True)
    tectonic=shutil.which('tectonic')
    if tectonic:
        subprocess.run([tectonic,'--outdir',str(out),path.name],cwd=path.parent,check=True)
    elif shutil.which('pdflatex'):
        for _ in range(3):
            subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error','-output-directory',str(out),path.name],
                           cwd=path.parent,check=True,stdout=subprocess.DEVNULL)
    else:
        print('No tectonic/pdflatex available: source and figures are ready; compile the source in the desktop editor.',flush=True)
        return None
    return out/(path.stem+'.pdf')



def bundle(root, manuscript=None):
    root = Path(root); path = Path(manuscript) if manuscript else None
    target = root/('MERRAFlow_paper.zip' if path else 'MERRAFlow_plots_metrics.zip')
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as z:
        if path:
            z.write(path, 'manuscript/'+path.name)
            workflow = path.parent/'figures'/'MERRA21C_ML_workflow.pdf'
            if workflow.is_file():
                z.write(workflow, 'manuscript/figures/'+workflow.name)
            for p in sorted((path.parent/'figures'/'auto').glob('*.pdf')):
                z.write(p, 'manuscript/figures/auto/'+p.name)
        for name in ('snapshot.json', 'manifest.json', 'summary.json', 'manuscript_update.json'):
            if (root/name).is_file():z.write(root/name, 'analysis/'+name)
        for name in ('error_maps.json', 'error_maps.npz'):
            if (root/name).is_file():z.write(root/name, 'analysis/'+name)
        for p in (root/'cases').glob('*.json'):
            z.write(p, 'analysis/cases/'+p.name)
        for p in (root/'paper_figures').iterdir():
            if p.suffix in ('.pdf', '.png', '.csv', '.json', '.md'):
                z.write(p, 'analysis/paper_figures/'+p.name)
        for p in (root/'pdf').glob('*.pdf') if (root/'pdf').is_dir() else ():
            z.write(p, 'pdf/'+p.name)
        z.writestr('README.txt', 'MERRA-Flow results bundle. Numerical results cover the frozen hours listed in '
                   'analysis/snapshot.json. Example NPZ fields and storm NetCDF stay in the snapshot/run directories.\n'
                   'Compile manuscript/MERRA21C_ML_arxiv.tex with pdfLaTeX (twice) or Tectonic.\n')
    return target


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = p.add_mutually_exclusive_group(); mode.add_argument('--evaluation'); mode.add_argument('--snapshot')
    p.add_argument('--output', help='New immutable snapshot directory; defaults to a timestamped sibling of the evaluation')
    p.add_argument('--manuscript', default='manuscript/MERRA21C_ML_arxiv.tex', help='Manuscript .tex with AUTO blocks, updated in place')
    p.add_argument('--plots-only', action='store_true', help='Generate plots and metrics without reading or updating a manuscript')
    p.add_argument('--storm', help='storm_fields.nc from merraflow.storm_animation_v4_1 for the storm case study')
    p.add_argument('--maps', help='maps.npz or the maps directory of scripts/evaluate_maps_v4_1.py; a directory is '
                                  'combined with whatever hours are finished')
    p.add_argument('--skip-maps', action='store_true', help='Do not copy/render saved example fields')
    p.add_argument('--no-inline', action='store_true', help='Do not embed PNG previews in the .tex (smaller source)')
    p.add_argument('--dpi', type=int, default=300)
    p.add_argument('--compile', action='store_true', help='Export a PDF if tectonic or pdflatex is installed')
    a = p.parse_args()
    if not a.plots_only and not Path(a.manuscript).is_file():
        p.error('--manuscript must be an existing source file')
    if a.dpi < 72:
        p.error('--dpi must be at least 72')
    if a.storm and not Path(a.storm).is_file():
        p.error(f'--storm file not found: {a.storm}')
    if not a.snapshot:
        a.evaluation = a.evaluation or 'runs/merraflow_v4_1/evaluation/all_test_best_m16_s64'
        a.output = a.output or (a.evaluation+'_interim_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    root = snapshot(a.evaluation, a.output, not a.skip_maps) if a.evaluation else Path(a.snapshot)
    if a.maps and not Path(a.maps).exists():
        p.error(f'--maps not found: {a.maps}')
    figs = render(root, a.dpi, a.storm, a.maps)
    if not a.plots_only:
        update_paper(root, a.manuscript, figs, inline=not a.no_inline)
    if a.compile and not a.plots_only:
        compile_paper(root, a.manuscript)
    archive = bundle(root, None if a.plots_only else a.manuscript)
    print(f'Snapshot and figures: {root}\nResults bundle: {archive}', flush=True)
    if not a.plots_only:print(f'Updated manuscript: {a.manuscript}', flush=True)


if __name__ == '__main__':
    main()
