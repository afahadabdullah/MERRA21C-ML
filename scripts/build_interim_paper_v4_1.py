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


def render(root, dpi=300, storm=None):
    """Redraw every paper figure from the frozen snapshot (no analysis, no inference)."""
    from merraflow.evaluate_all_v4_1 import atomic_json, digest
    plots = _plots()
    manifest, meta, summary = load_snapshot(root)
    root = Path(root); out = root/'paper_figures'; out.mkdir(exist_ok=True)
    for old in out.glob('*'):
        if old.suffix in ('.pdf', '.png', '.svg'):
            old.unlink()
    maps_manifest = deepcopy(manifest)
    maps_manifest['settings']['saved_cases'] = meta['available_map_cases']
    ids = [e['id'] for e in meta['entries']]
    subject = (f'MERRA-Flow v4.1 | {summary["hours"]}/{summary["expected_hours"]} held-out hours'
               + ('' if summary['hours'] == summary['expected_hours'] else ' (interim, completion-ordered)'))
    figs = plots.render_all(root, maps_manifest, summary, ids, out, dpi, storm_path=storm, subject=subject)
    if storm:
        figs['storm']['sha256'] = digest(storm)
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


def results(c):
    f = c.f; s = []
    sk, red = c.skill('raw'), c.rmse_red('raw')
    rsk = c.skill('regression')
    positive = [i for i, v in enumerate(sk) if v is not None and v > 0]
    s.append('\\section{Results}\\label{sec:results}\n')
    s.append(f'All quantitative results use {c.scope}, generated with {c.M} members and '
             f"{c.s['settings']['steps']} Heun steps. Maps come from the archived example hours fixed before inference "
             '(Section~\\ref{sec:fulltest}); aggregate scores pool every evaluated hour.\n\n')
    cases = f.get('cases') or []
    main = next((x for x in cases if x.get('main')), None)
    if main:
        g = main['gain']['numbers']
        s.append('\\subsection{From 25-km cells to kilometer-scale structure}\\label{sec:maps}\n')
        s.append(f"Figure~\\ref{{fig:overview}} shows {when(main['time'])}, the archived hour with the largest GEOS-FP "
                 'wet area. At its native resolution GEOS-FP represents the Rockies, the Appalachians, and the coastlines '
                 'as staircases of $\\sim$25-km cells, and convective rain as broad, weak blobs. A single \\model{} member '
                 '(middle column) carries the valley--ridge temperature contrasts, terrain-channelled winds, sharp coastal '
                 'humidity gradients, and compact rain cores that appear in the kilometer-scale reference (right). '
                 'The member is a sample, not a forecast of the reference realization, so individual small features need not '
                 'coincide; the correspondence is in their location relative to terrain and in their statistics.\n\n')
        s.append(figure(main['overview']['stem'],
            f"Downscaling GEOS-FP for {when(main['time'])}. Rows show 2-m temperature, 10-m wind speed, precipitation rate, "
            'and 2-m specific humidity. Left: GEOS-FP on its native $0.25^\\circ\\times0.3125^\\circ$ cells (each 3-km pixel '
            'takes the value of the GEOS-FP cell containing it; no interpolation). Middle: \\model{} member 1. Right: '
            'GEOS-CAM reference. Each row shares one color scale. Panels are displayed at 6-km '
            '($2\\times2$ block mean) for legibility; boxes A and B mark the zooms in Fig.~\\ref{fig:zoom}.',
            'fig:overview'))
        s.append('Inside the zoom boxes (Fig.~\\ref{fig:zoom}) the native GEOS-FP cells are outlined. Bilinear '
                 'interpolation removes the staircase but adds no information, and the deterministic regression adds '
                 'the predictable, terrain-locked component while remaining smooth because it estimates a conditional mean. '
                 '\\model{} members add the remaining kilometer-scale variability: cold-air drainage and ridge-top winds in '
                 'box A, and cellular rain cores embedded in the stratiform shield in box B. ')
        if 'spectral' in c.src:
            s.append('The spectral adjustment (fifth column) sharpens the same member without changing its structure. ')
        s.append('\n\n')
        s.append(figure(main['zoom']['stem'],
            'Resolution ladder inside the zoom boxes of Fig.~\\ref{fig:overview}. Box A (rows 1--2) is a fixed Colorado '
            'Rockies window; box B (rows 3--4) is centered on the heaviest smoothed GEOS-FP precipitation, chosen without '
            'reference to the model output or the reference. Columns: native GEOS-FP cells (white outlines mark cell '
            'boundaries), bilinearly interpolated GEOS-FP, deterministic regression, a \\model{} member'
            + (', the same member after spectral adjustment' if 'spectral' in c.src else '')
            + ', and the reference. All panels show every 3-km pixel.', 'fig:zoom'))
        best = sorted([(v.get('skill') or -9, k) for k, v in g.items()], reverse=True)
        names = dict(zip(FIELD_KEYS, NAMES))
        s.append('\\subsection{Where downscaling helps}\\label{sec:gain}\n')
        s.append('Figure~\\ref{fig:gain} maps the pixelwise GEOS-FP absolute error, the \\model{} CRPS, and their difference '
                 'for the same hour. For this hour the domain-mean CRPS skill relative to the GEOS-FP error is '
                 + join([f"{pct(v['skill'])}\\% for {names[k]}" for k, v in g.items() if v.get('skill') is not None]) + '. '
                 'The left column marks where sub-grid structure matters---large GEOS-FP errors over complex terrain, along '
                 'coastlines, and around precipitating systems---and the right column shows how much of that error the '
                 'ensemble removes (blue) or adds (red).\n\n')
        s.append(figure(main['gain']['stem'],
            f"Where downscaling helps, {when(main['time'])}. Left: absolute error of interpolated GEOS-FP. Middle: "
            f"CRPS of the {main['members']}-member \\model{{}} ensemble (same scale). Right: GEOS-FP error minus CRPS; blue "
            'marks improvement. Fields are averaged over $4\\times4$ blocks (12~km) for display. The label gives the '
            'domain-mean CRPS skill for this hour.', 'fig:gain'))
    storm = f.get('storm')
    if storm:
        ev = storm['evolution']['hours']; pk = next((h for h in ev if h['time'].startswith(storm['peak'][:16])), ev[-1])
        se = storm['series']['maxima']
        s.append('\\subsection{Storm case study}\\label{sec:storm}\n')
        s.append(f"Figures~\\ref{{fig:storm}}--\\ref{{fig:stormseries}} follow the wettest held-out event through "
                 f"{storm['hours']} consecutive hours ({when(storm['start'])} to {when(storm['end'])}), with "
                 f"{storm['members']} members that keep their noise seed from hour to hour. GEOS-FP resolves the "
                 'pressure pattern and the location of the rain shield, but its 25-km cells spread the convective cores '
                 'into broad maxima. At the peak hour the maximum rain rate in the window is '
                 f"{pk['geos_max']:.0f}~mm\\,h$^{{-1}}$ in GEOS-FP, {pk['member_max']:.0f}~mm\\,h$^{{-1}}$ in member 1, and "
                 f"{pk['truth_max']:.0f}~mm\\,h$^{{-1}}$ in the reference. ")
        closer = [k for k, lab in enumerate(('heavy-rain area', '99.9th-percentile rain rate', '99th-percentile wind speed'))
                  if abs(se['member'][k]-se['truth'][k]) < abs(se['geos'][k]-se['truth'][k])]
        labels = ('heavy-rain area', '99.9th-percentile rain rate', '99th-percentile wind speed')
        s.append('Through the event (Fig.~\\ref{fig:stormseries}), the largest 99.9th-percentile rain rate is '
                 f"{se['truth'][1]:.0f}~mm\\,h$^{{-1}}$ in the reference, {se['member'][1]:.0f}~mm\\,h$^{{-1}}$ for the member "
                 f"mean, and {se['geos'][1]:.0f}~mm\\,h$^{{-1}}$ for GEOS-FP. "
                 + (f"The members are closer to the reference than GEOS-FP in {join([labels[k] for k in closer])}. " if closer
                    else 'GEOS-FP is closer to the reference than the members in all three statistics. '))
        if storm.get('totals'):
            t = storm['totals']['max']
            s.append(f"Event totals (Fig.~\\ref{{fig:stormtotals}}) reach {t['truth']:.0f}~mm in the reference, "
                     f"{t['member']:.0f}~mm in member 1, and {t['geos']:.0f}~mm in GEOS-FP. ")
        s.append('Members differ in the exact placement of individual cells, which is the uncertainty the ensemble is '
                 'designed to express; the organization of the rainbands and the wind maxima along them is common to all.\n\n')
        s.append(figure(storm['evolution']['stem'],
            'Evolution of the storm case. Rows are valid times through the event; columns are native GEOS-FP cells, two '
            '\\model{} members, and the reference. Shading is precipitation rate; contours are terrain-reduced sea-level '
            'pressure (4 hPa interval, smoothed, masked above 1000 m).' + (' Raw flow members.' if storm['post'] == 'none'
            else ' Members include the spectral adjustment.'), 'fig:storm', placement='p'))
        s.append(figure(storm['peak_fields']['stem'],
            f"Storm structure at the peak hour ({when(storm['peak'])}): 10-m wind speed, 2-m temperature, and 2-m specific "
            'humidity for native GEOS-FP, two members, and the reference.', 'fig:stormpeak'))
        s.append(figure(storm['series']['stem'],
            'Window statistics through the storm: (a) area with rain rate $\\geq$5~mm\\,h$^{-1}$, (b) 99.9th-percentile '
            'rain rate, and (c) 99th-percentile 10-m wind speed. The shaded band spans the members; the blue line is their mean.',
            'fig:stormseries'))
        if storm.get('totals'):
            s.append(figure(storm['totals']['stem'],
                'Event-total precipitation over the storm window for native GEOS-FP, member 1, the ensemble mean, and the '
                'reference.', 'fig:stormtotals'))
    if main and main.get('ensemble'):
        s.append('\\subsection{Ensemble realizations and probabilities}\\label{sec:ensemble}\n')
        s.append('Because the generator samples the conditional distribution, a single coarse input yields many distinct, '
                 'equally detailed fields (Fig.~\\ref{fig:ensemble}). Members agree on the rain shield and on the region '
                 'where heavy rain is possible but differ in the placement of individual cores. The ensemble mean is '
                 'therefore smooth, while the spread concentrates in and around the convective cores, and the member '
                 'fraction exceeding 1~mm\\,h$^{-1}$ gives a probability map that can be compared directly with the '
                 'reference rain area (red contour). This is information a deterministic downscaling cannot supply.\n\n')
        s.append(figure(main['ensemble']['stem'],
            f"Ensemble realizations in box B, {when(main['time'])}. Top: native GEOS-FP and three \\model{{}} members. "
            f"Bottom: reference, {main['members']}-member ensemble mean, ensemble standard deviation, and the fraction of "
            'members with rain rate $\\geq$1~mm\\,h$^{-1}$, with the reference 1~mm\\,h$^{-1}$ contour in red.',
            'fig:ensemble'))
    # Aggregate skill
    s.append('\\subsection{Skill over the held-out period}\\label{sec:skill}\n')
    nrs = sum(1 for v in red if v is not None and v > 0)
    s.append(f"Table~\\ref{{tab:scores}} and Fig.~\\ref{{fig:scorecard}} summarize {c.scope}. Relative to interpolated "
             'GEOS-FP, the raw \\model{} ensemble lowers CRPS by '
             + join([f'{pct(sk[i])}\\% for {NAMES[i]}' for i in range(len(FIELD_KEYS))]) + '. '
             f"CRPS skill is positive for {len(positive)} of {len(FIELD_KEYS)} fields, and ensemble-mean RMSE is lower "
             f"than GEOS-FP for {nrs} of {len(FIELD_KEYS)}. ")
    better_reg = [NAMES[i] for i, f in enumerate(FIELD_KEYS)
                  if f != 'q2m' and c.v('raw', f, 'crps') < c.v('regression', f, 'mae')]
    s.append(f"Against the deterministic regression, the ensemble has lower CRPS than regression MAE for "
             f"{len(better_reg)} of {len(FIELD_KEYS)-1} fields with a regression output"
             + (f" ({join(better_reg)})" if better_reg and len(better_reg) < len(FIELD_KEYS)-1 else '') + '. ')
    p_i = FIELD_KEYS.index('precip')
    if red[p_i] is not None and red[p_i] < 0:
        s.append('For precipitation, ensemble-mean RMSE is higher than for the smooth GEOS-FP field even though CRPS improves: '
                 'sharp members that misplace cells by a few kilometers are penalized twice by a point-wise squared error, '
                 'which is why neighbourhood and intensity diagnostics (Section~\\ref{sec:scales}) are more informative. ')
    s.append('\n\n')
    s.append(scores_table(c))
    s.append(figure(f['scorecard']['stem'],
        c.scope_cap+' (a) CRPS skill (left block) and ensemble-mean RMSE reduction (right block) relative to interpolated '
        'GEOS-FP, for the deterministic regression (Regr.), the raw \\model{} ensemble (Flow), and the spectrally '
        'adjusted ensemble (Flow+sp.); blue is better than GEOS-FP. (b) Central 90\\% interval coverage; the dashed line '
        'marks the nominal value.', 'fig:scorecard'))
    t = f['temporal']
    lowest = min(range(len(FIELD_KEYS)), key=lambda i: t['positive_fraction'][FIELD_KEYS[i]])
    low = t['positive_fraction'][FIELD_KEYS[lowest]]
    s.append((f"The gain is not driven by a few favorable hours (Fig.~\\ref{{fig:temporal}}). " if low >= 80 else
              f"Figure~\\ref{{fig:temporal}} resolves the skill hour by hour. ")
             + f"CRPS skill is positive in {low:.0f}\\% or more of the {t['hours']} evaluated hours for every field "
             f"(lowest: {NAMES[lowest]}). "
             + f"Daily-mean skill is positive on every evaluated day for {sum(1 for v in t['daily_positive'].values() if v == 1)} "
             f"of {len(FIELD_KEYS)} fields. "
             'The diurnal composite shows how the gain varies with the boundary-layer cycle; hourly precipitation skill is '
             'expected to vary most because many hours contain little rain.\n\n')
    s.append(figure(t['stem'],
        c.scope_cap+' (a) Daily-mean CRPS skill of the raw ensemble relative to GEOS-FP for every field; the bar strip '
        'above shows eligible (light) and evaluated (dark) hours per day. (b) Mean skill by UTC hour. (c) Distribution '
        'of hourly skill (violins, with medians). (d) Percentage of hours with positive skill.', 'fig:temporal'))
    # Scales
    sp = f['spectra']['ratio']; fs = f['precip']
    s.append('\\subsection{Scale-dependent variance and precipitation structure}\\label{sec:scales}\n')
    words = []
    for k in ('t2m', 'wind_speed', 'precip', 'q2m'):
        r = sp.get(k, {})
        if r.get('raw') is not None and r.get('coarse') is not None:
            words.append(f"{dict(zip(FIELD_KEYS, NAMES))[k]} ({pct(r['raw'])}\\% versus {pct(r['coarse'])}\\%)")
    retained = all((sp.get(k, {}).get('raw') or 0) > (sp.get(k, {}).get('coarse') or 0) for k in ('t2m', 'wind_speed', 'precip', 'q2m'))
    s.append('Power spectra (Fig.~\\ref{fig:spectra}) show where the added variance lives. At synoptic wavelengths all '
             'predictions follow the reference closely, as expected for a replay constrained by GEOS-FP. Below the GEOS-FP '
             'grid scale, interpolated GEOS-FP and the regression lose power rapidly'
             + (', whereas \\model{} members retain much of it. ' if retained else '. ')
             + 'Averaged over wavelengths from 6 to 54~km, member power relative to the reference is '
             + join(words) + ' for the raw ensemble versus interpolated GEOS-FP. '
             + ('The spectral adjustment raises this ratio further. ' if 'spectral' in c.src else '') + '\n\n')
    s.append(figure(f['spectra']['stem'],
        c.scope_cap+' Top: hour-averaged radial power spectra of the reference, interpolated GEOS-FP, regression, and '
        '\\model{} members, versus wavelength (3-km grid spacing; index-space spectra). Bottom: ratio to the reference. '
        'Shading marks wavelengths shorter than twice the GEOS-FP grid spacing.', 'fig:spectra'))
    def fss_at(th, m, w):
        try:
            row = fs['fss'][th][m]
            return row[w] if w in row else row[str(w)]
        except KeyError:
            return None
    w5 = 5
    q = fs['tails']; probs = fs['quantiles']
    top = len(probs)-1
    s.append('Neighbourhood verification (Fig.~\\ref{fig:precip}) shows how precipitation skill depends on scale. '
             f"At 1~mm\\,h$^{{-1}}$ and a 15-km neighbourhood, ensemble-mean FSS is {number(fss_at('1.0', 'raw', w5), 1, 2)} for "
             f"\\model{{}} and {number(fss_at('1.0', 'coarse', w5), 1, 2)} for GEOS-FP; at 5~mm\\,h$^{{-1}}$ the values are "
             f"{number(fss_at('5.0', 'raw', w5), 1, 2)} and {number(fss_at('5.0', 'coarse', w5), 1, 2)}. "
             'Individual members (dotted) are scored separately from the ensemble mean; any gap between them at small '
             'neighbourhoods is the price of sharp, stochastic placement, and it narrows as the neighbourhood grows. ')
    if q.get('reference') and q.get('raw') and q.get('coarse'):
        s.append(f"The intensity distribution is where downscaling matters most: the {100*probs[top]:g}th-percentile "
                 f"rain rate is {number(q['reference'][top], 1, 3)}~mm\\,h$^{{-1}}$ in the reference, "
                 f"{number(q['raw'][top], 1, 3)}~mm\\,h$^{{-1}}$ in \\model{{}} members, and "
                 f"{number(q['coarse'][top], 1, 3)}~mm\\,h$^{{-1}}$ in interpolated GEOS-FP. ")
    s.append('\n\n')
    s.append(figure(fs['stem'],
        c.scope_cap+' (a--c) Fractions skill score versus neighbourhood width for rain-rate thresholds of 1, 5, and '
        '10~mm\\,h$^{-1}$; solid lines use ensemble means (or the deterministic field), dotted lines the average over '
        'members. (d) Hour-averaged pixel percentiles of precipitation rate (symmetric-log axis).', 'fig:precip'))
    # Calibration
    cov = [c.v('raw', fk, 'coverage_90') for fk in FIELD_KEYS]
    ssr = [c.v('raw', fk, 'spread_skill_corrected') for fk in FIELD_KEYS]
    s.append('\\subsection{Calibration and the spectral adjustment}\\label{sec:calibration}\n')
    s.append(f"Central 90\\% interval coverage ranges from {pct(min(cov))}\\% to {pct(max(cov))}\\% across fields, and the "
             f"corrected spread--error ratio from {number(min(ssr), 1, 2)} to {number(max(ssr), 1, 2)} "
             '(Appendix~\\ref{app:calibration}). Coverage below the nominal value together with a ratio below one indicates '
             'an under-dispersive ensemble; rank histograms and reliability diagrams in Fig.~\\ref{fig:calibration} show '
             'which fields and which tails are affected. ')
    if 'spectral' in c.src:
        better = [NAMES[i] for i, fk in enumerate(FIELD_KEYS) if c.v('spectral', fk, 'crps') < c.v('raw', fk, 'crps')]
        s.append(f"The paired spectral adjustment lowers CRPS for {len(better)} of {len(FIELD_KEYS)} fields"
                 + (f" ({join(better)})" if 0 < len(better) < len(FIELD_KEYS) else '')
                 + ' (Table~\\ref{tab:spectral}); its main effect is on fine-scale amplitude rather than on placement.')
    s.append('\n\n')
    return ''.join(s)


def appendix(c):
    f = c.f; s = ['\\section{Additional diagnostics}\\label{app:calibration}\n']
    s.append(figure(f['calibration']['stem'],
        c.scope_cap+' (a, b) Relative frequency of the reference rank among the ensemble members (1 = uniform; '
        'randomized ties), for the raw and spectrally adjusted ensembles. U-shapes indicate under-dispersion; slopes '
        'indicate bias. (c) Corrected spread--error ratio. (d, e) Precipitation reliability at 1 and 5~mm\\,h$^{-1}$ '
        'with the issued-probability area distribution (inset). (f) Brier skill score relative to GEOS-FP by threshold.',
        'fig:calibration'))
    s.append(processing_table(c))
    for case in (f.get('cases') or []):
        if case.get('main'):
            continue
        s.append(figure(case['overview']['stem'],
            f"As Fig.~\\ref{{fig:overview}}, for the archived example hour {when(case['time'])}.",
            'fig:case_'+case['id'].replace('_', '')))
    return ''.join(s)


def abstract(c):
    sk = c.skill('raw'); p = FIELD_KEYS.index('precip')
    state = [i for i in range(len(FIELD_KEYS)) if i != p and sk[i] is not None]
    lo = min(state, key=lambda i: sk[i]); hi = max(state, key=lambda i: sk[i])
    t = c.f['temporal']
    text = (f"Evaluated on {c.scope}, the {c.M}-member ensemble has continuous ranked probability "
            f"score skill relative to interpolated GEOS-FP ranging from {pct(sk[lo])} to {pct(sk[hi])}\\% for the state variables "
            f"and {pct(sk[p])}\\% for precipitation, with positive skill in at least "
            f"{min(t['positive_fraction'].values()):.0f}\\% of hours for every field. ")
    sp = c.f['spectra']['ratio']
    if all((sp.get(k, {}).get('raw') or 0) > (sp.get(k, {}).get('coarse') or 0) for k in ('t2m', 'wind_speed', 'precip', 'q2m')):
        text += 'Member spectra contain more power than interpolated GEOS-FP in the evaluated fine-scale band; this does not establish spatial phase accuracy. '
    q = c.f['precip']['tails']
    if q.get('reference') and q.get('raw') and q.get('coarse') and \
            abs(q['raw'][-1]-q['reference'][-1]) < abs(q['coarse'][-1]-q['reference'][-1]):
        text += 'The upper tail of hourly rain rates is closer to the reference than in GEOS-FP. '
    st = c.f.get('storm')
    if st:
        se = st['series']['maxima']
        if abs(se['member'][1]-se['truth'][1]) < abs(se['geos'][1]-se['truth'][1]):
            text += ('In a storm case study, generated members reproduce peak rain intensities that GEOS-FP spreads across '
                     'its grid cells. ')
        else:
            text += 'A storm case study illustrates the structure and limits of the generated rain cores. '
    if not c.complete:
        text += 'These numbers describe an interim, completion-ordered subset; the full held-out evaluation is in progress.'
    return text


def keypoints(c):
    sk = c.skill('raw'); t = c.f['temporal']
    pos = sum(1 for v in sk if v is not None and v > 0)
    items = ['A six-variable conditional flow-matching model downscales hourly GEOS-FP from $\\sim$25-km cells to a '
             '3-km CONUS grid and generates multi-member ensembles from a single coarse input.',
             f"The ensemble improves the CRPS of interpolated GEOS-FP for {pos} of {len(FIELD_KEYS)} fields"
             + (', consistently from hour to hour' if min(t['positive_fraction'].values()) >= 80 else '')
             + '; spectra and calibration assess the associated fine-scale variability.']
    st = c.f.get('storm')
    if st and abs(st['series']['maxima']['member'][1]-st['series']['maxima']['truth'][1]) < \
            abs(st['series']['maxima']['geos'][1]-st['series']['maxima']['truth'][1]):
        items.append('In a storm case study, members reproduce convective rain cores and peak intensities that the native '
                     'GEOS-FP cells cannot resolve.')
    elif st:
        items.append('A storm case study follows convective rain cores and wind maxima at 3~km through the event.')
    else:
        items.append('Available archived examples compare generated and reference fields; ensemble spread and '
                     'exceedance probabilities describe conditional variability.')
    return '\\begin{itemize}\n'+'\n'.join('\\item '+x for x in items)+'\n\\end{itemize}'


def conclusions(c):
    sk = c.skill('raw')
    pos = sum(1 for v in sk if v is not None and v > 0)
    text = (f"On {c.scope}, the generated ensembles have positive CRPS skill relative to interpolated GEOS-FP for {pos} of "
            f"{len(FIELD_KEYS)} evaluated fields. The score matrix reports both probabilistic skill and ensemble-mean "
            'error, while spectra and neighbourhood scores assess structure at different scales. Ensemble spread, '
            'ranks, and exceedance reliability provide separate calibration diagnostics. These comparisons do not '
            'establish observational accuracy, causal benefits of individual design choices, or learned temporal coherence.')
    if not c.complete:
        text += ' These conclusions rest on an interim subset and will be revisited with the complete held-out evaluation.'
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
    figs = render(root, a.dpi, a.storm)
    if not a.plots_only:
        update_paper(root, a.manuscript, figs, inline=not a.no_inline)
    if a.compile and not a.plots_only:
        compile_paper(root, a.manuscript)
    archive = bundle(root, None if a.plots_only else a.manuscript)
    print(f'Snapshot and figures: {root}\nResults bundle: {archive}', flush=True)
    if not a.plots_only:print(f'Updated manuscript: {a.manuscript}', flush=True)


if __name__ == '__main__':
    main()
