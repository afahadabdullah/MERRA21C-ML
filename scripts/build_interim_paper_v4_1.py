#!/usr/bin/env python3
"""Freeze completed MERRA-Flow hours and build an explicitly interim paper.

Run from the frozen evaluation checkout. This script lives outside src so it
can be fetched without invalidating live workers. It never runs inference or
writes to the parent evaluation. --snapshot redraws saved data without analysis.
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


def render(root, dpi=300):
    from merraflow import paper_figures_v4_1 as plots
    from merraflow.evaluate_all_v4_1 import atomic_json, digest
    manifest, meta, summary = load_snapshot(root)
    out = Path(root)/'paper_figures';out.mkdir(exist_ok=True)
    plots.style();original = plots.export
    def interim_export(fig, output, stem, dpi=300):
        fig.text(.5, -.10, f'MERRA-Flow | INTERIM: {summary["hours"]}/{summary["expected_hours"]} eligible test hours | completion-ordered subset',
                 ha='center', va='top', fontsize=9, color='#a04a15')
        original(fig, output, stem, dpi)
    plots.export = interim_export
    try:
        plots.skill(summary,out,dpi);plots.calibration(summary,out,dpi)
        plots.structure(summary,out,dpi);plots.precipitation(summary,out,dpi);plots.filtering(summary,out,dpi)
        maps_manifest = deepcopy(manifest)
        maps_manifest['settings']['saved_cases'] = meta['available_map_cases']
        maps = plots.saved_maps(Path(root), maps_manifest, out, dpi)
        # Temporal support: all eligible hours versus the frozen completed subset.
        import matplotlib.pyplot as plt
        expected = np.array([e['time'] for e in manifest['entries']], dtype='datetime64[m]')
        included = np.array([e['time'] for e in meta['entries']], dtype='datetime64[m]')
        days, counts = np.unique(expected.astype('datetime64[D]'), return_counts=True)
        finished = {str(d):n for d,n in zip(*np.unique(included.astype('datetime64[D]'),return_counts=True))}
        fig, ax = plt.subplots(figsize=(7.2,2.8),layout='constrained')
        ax.bar(days.astype(object), counts, color='#d6dee4', label='All eligible hours')
        ax.bar(days.astype(object), [finished.get(str(d),0) for d in days], color='#0072b2',label='Completed snapshot')
        ax.set(ylabel='Hours per UTC day',xlabel='Test date (UTC)',title='Temporal coverage of the interim snapshot')
        ax.legend(frameon=False);fig.autofmt_xdate()
        plots.export(fig,out,'interim_test_coverage',dpi)
    finally:
        plots.export = original
    atomic_json(out/'figure_data.json', summary)
    import csv
    with (out/'table01_all_variables.csv').open('w',newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(['scope','hours','expected_hours','method','field','unit','rmse','mae','bias','crps','fair_crps','crps_skill','coverage_90'])
        for method in plots.methods(summary):
            for i,field in enumerate(plots.FIELDS):
                values=summary['sources'][method]['fields'][field]
                writer.writerow(['interim',summary['hours'],summary['expected_hours'],method,field,plots.UNITS[i],
                    *[None if values[key] is None else values[key]*plots.FACTORS[i] for key in
                      ('rmse_pooled','mae','bias','crps','fair_crps')],values['crpss_vs_coarse'],values['coverage_90']])
    (out/'captions.md').write_text(f'# MERRA-Flow interim snapshot\n\n{summary["hours"]}/{summary["expected_hours"]} eligible hours; completion-ordered subset.\n'
        +'No full-test or statistical-significance claim. All figures use the same saved subset; no confidence intervals are drawn.\n'
        +'See the populated paper captions and figure_data.json for metrics and conventions.\n')
    atomic_json(out/'plot_manifest.json', dict(scope='interim', complete=False, snapshot_id=meta['snapshot_id'],
        run_id=manifest['run_id'], hours=summary['hours'], expected_hours=summary['expected_hours'], dpi=dpi,
        summary_sha256=digest(Path(root)/'summary.json'), builder_sha256=digest(__file__),
        files={p.name:digest(p) for p in out.iterdir() if p.suffix in ('.pdf','.svg','.png')}))
    return maps


def number(value, factor=1, digits=3):
    return '--' if value is None or not np.isfinite(value) else f'{value*factor:.{digits}g}'


def table(caption, label, columns, headings, rows):
    return ('\\begin{table*}[t]\n\\caption{'+caption+'}\\label{'+label+'}\n\\centering\\small\n'
        +'\\begin{tabular}{'+columns+'}\\toprule\n'+' & '.join(headings)+r' \\ \midrule'+'\n'
        +'\n'.join(' & '.join(row)+r' \\' for row in rows)+'\n\\bottomrule\\end{tabular}\n\\end{table*}\n')


def apply_block(source, key, text, pattern=None):
    start, end = f'% BEGIN INTERIM {key}', f'% END INTERIM {key}'
    block = start+'\n'+text.rstrip()+'\n'+end+'\n'
    if start in source:
        first=source.index(start);last=source.index(end,first)+len(end)
        return source[:first]+block+source[last:].lstrip('\n')
    if pattern:
        source, count = re.subn(pattern, lambda _:block.rstrip(), source, count=1, flags=re.M)
        if count != 1:raise ValueError(f'Cannot locate manuscript block: {key}')
        return source
    raise ValueError(f'Missing manuscript marker: {key}')


def figure(stem, caption, label):
    name='MERRAFlow_interim_'+stem+'.pdf'
    return ('\\begin{figure*}[t]\\centering\n'
        +'\\IfFileExists{figures/interim/'+name+'}{\\includegraphics[width=\\textwidth]{figures/interim/'+name+'}}{%\n'
        +'\\IfFileExists{manuscript/figures/interim/'+name+'}{\\includegraphics[width=\\textwidth]{manuscript/figures/interim/'+name+'}}'
        +'{\\includegraphics[width=\\textwidth]{'+name+'}}}\n\\caption{'+caption+'}\\label{'+label+'}\n\\end{figure*}\n')


def inline_pdf(path, width=518.4):
    """ASCII PDF fallback for the editor's single-source compiler; lossless RGB."""
    from PIL import Image
    with Image.open(Path(path).with_suffix('.png')) as image:
        rgb=image.convert('RGB');pixels=rgb.tobytes();w,h=rgb.size
    import zlib
    stream=base64.a85encode(zlib.compress(pixels),wrapcol=100)+b'~>\n';height=width*h/w
    objects=[b'<< /Type /Catalog /Pages 2 0 R >>',b'<< /Type /Pages /Kids [3 0 R] /Count 1 >>',
        f'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width} {height}] /Resources << /XObject << /Im 4 0 R >> >> /Contents 5 0 R >>'.encode(),
        f'<< /Type /XObject /Subtype /Image /Width {w} /Height {h} /ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter [/ASCII85Decode /FlateDecode] /Length {len(stream)} >>\nstream\n'.encode()+stream+b'endstream']
    content=f'q {width} 0 0 {height} 0 0 cm /Im Do Q\n'.encode()
    objects.append(f'<< /Length {len(content)} >>\nstream\n'.encode()+content+b'endstream')
    result=b'%PDF-1.4\n';offsets=[0]
    for i,obj in enumerate(objects,1):offsets.append(len(result));result+=f'{i} 0 obj\n'.encode()+obj+b'\nendobj\n'
    xref=len(result);result+=f'xref\n0 {len(objects)+1}\n0000000000 65535 f \n'.encode()
    result+=b''.join(f'{p:010d} 00000 n \n'.encode() for p in offsets[1:])
    result+=f'trailer\n<< /Size {len(objects)+1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n'.encode()
    return result.decode('ascii')


def update_paper(root, manuscript):
    from merraflow.evaluate_all_v4_1 import FIELDS
    manifest, meta, s = load_snapshot(root)
    root=Path(root);path=Path(manuscript);source=path.read_text();n,N=s['hours'],s['expected_hours']
    names=('Temperature','Precipitation','Surface pressure','Zonal wind','Meridional wind','Specific humidity','Wind speed')
    factors=(1,1,.01,1,1,1000,1)
    units=(r'K',r'mm h$^{-1}$',r'hPa',r'm s$^{-1}$',r'm s$^{-1}$',r'g kg$^{-1}$',r'm s$^{-1}$')
    methods=[m for m in ('coarse','regression','raw','spectral') if m in s['sources']]
    labels=dict(coarse='Coarse',regression='Regression',raw='Raw flow',spectral='Spectral flow')
    raw=s['sources']['raw']['fields'];cov=[100*raw[f]['coverage_90'] for f in FIELDS]
    skill=[100*raw[f]['crpss_vs_coarse'] for f in FIELDS if raw[f]['crpss_vs_coarse'] is not None]
    count=sum(v>0 for v in skill);pct=f'{100*n/N:.1f}'
    scope=f'Interim snapshot: {n} of {N} eligible test hours ({pct}\\%).'
    source=re.sub(r'Interim snapshot: \d+ of \d+ eligible test hours \([\d.]+\\%\)\.',lambda _:scope,source)
    abstract=(scope+' Raw-flow empirical CRPS skill relative to coarse input ranges from '
        +number(min(skill))+' to '+number(max(skill))+r'\% across the evaluated fields, with positive skill for '
        +f'{count} of {len(skill)} fields. Empirical central 90'+r'\% interval coverage ranges from '
        +number(min(cov))+' to '+number(max(cov))+r'\%. Raw and spectrally processed ensembles are compared using identical noise draws. '
        +'These are descriptive estimates for a completion-ordered subset, not a random or complete test sample; full-test inference and aggregation remain in progress.')
    if '% BEGIN INTERIM ABSTRACT' not in source:
        a=source.index('In an exploratory evaluation of three selected held-out hours')
        b=source.index('\n\n\\vspace{0.8em}',a)
        source=source[:a]+'% BEGIN INTERIM ABSTRACT\n'+abstract+'\n% END INTERIM ABSTRACT'+source[b:]
    else:source=apply_block(source,'ABSTRACT',abstract)
    source=re.sub(r'Draft status: Figure 1[^\n]+',lambda _:scope+' All populated numerical results refer to this frozen interim subset. Full-test evaluation and author declarations remain in progress.}',source,count=1)
    # Update the scope wherever legacy case-study claims would otherwise persist.
    old='The available numerical evidence comprises three selected held-out hours, rather than a completed climatological evaluation. We report those results explicitly, identify their restrictions, and specify the additional comparisons needed to establish general skill.'
    source=source.replace(old,scope+' We report the frozen completion-ordered subset explicitly and retain the full-test manifest. Partial completion does not establish representative population skill.')
    source=source.replace('The preliminary diagnostic discussed below instead uses the latest checkpoint at epoch 243; it must not be relabeled as the validation-best model.',
        'The quantitative evaluation checkpoint is frozen by the parent run and identified in Table~\\ref{tab:settings}; workflow map illustrations retain their separate development provenance.')
    source=source.replace('The diagnostic averages these per-hour statistics; averaging RMSE across cases differs from taking the square root of a pooled mean-square error.',
        'The interim evaluator pools area-weighted MSE across hours before taking the square root. MAE and bias are averaged equally across hours.')
    source=source.replace('The saved central interval uses the interpolated empirical 5th and 95th percentiles of only eight members.',
        'The central interval uses interpolated empirical 5th and 95th percentiles of the frozen evaluation ensemble.')
    source=source.replace('Low coverage across several variables nevertheless identifies a practical limitation requiring larger-ensemble verification and calibration analysis.',
        'Coverage and corrected spread/error are reported for every field without assuming calibrated population quantiles.')
    source=source.replace('The saved evaluator uses Eq.~\\eqref{eq:crps}, not a finite-ensemble fair score.',
        'The evaluator saves both empirical CRPS in Eq.~\\eqref{eq:crps} and finite-ensemble fair CRPS; the main score table uses the empirical score.')
    source=source.replace('Fair scores cannot be reconstructed from the aggregate scorecard alone without its underlying pairwise statistics or ensemble fields.',
        'Both empirical and fair scores are retained in the saved per-hour records and pooled summary.')
    source=source.replace('The current three cases do not support stable block-bootstrap confidence intervals or seasonal significance tests; none are reported.',
        'The current completion-ordered subset does not support representative population intervals or seasonal significance tests; none are reported.')
    source=source.replace('The default production configuration requests 32 steps and 16 members, whereas the presentation evidence uses 64 steps and eight members.',
        'The frozen run settings in Table~\\ref{tab:settings} supersede default inference settings for the interim quantitative evaluation.')
    a=source.index('\\caption{GEOS-based generative downscaling framework.') if '\\caption{GEOS-based generative downscaling framework.' in source else source.index('\\caption{MERRA-Flow generative downscaling framework.')
    b=source.index('\\label{fig:workflow}',a)
    source=source[:a]+(r'\caption{MERRA-Flow generative downscaling framework. \textbf{A:} Representative stacked native GEOS-FP maps and the complete weather, coarse-baseline, static, and temporal input inventory. SLP and PS denote sea-level and surface pressure, respectively. \textbf{B:} Matched 2-m temperature maps on the same extent and scale, with a threefold tighter crop and unchanged source cells. \textbf{C:} Regression-conditioned flow matching, the conditional U-Net, weighted core/halo velocity loss, synchronized tiled Heun sampling, and a compact six-field output stack. The architecture illustration reuses the 7 March 2026, 19:30 UTC development diagnostic (epoch-243 EMA, eight members, 64 steps, spectral processing). Its map checkpoint is distinct from the frozen quantitative interim run in Table~\ref{tab:settings}. Input/output stacks summarize variables without a shared scalar color scale; panel B retains its temperature scale.}'+'\n')+source[b:]
    a=source.index('\\subsection{Spectral processing in the exploratory diagnostics}') if '\\subsection{Spectral processing in the exploratory diagnostics}' in source else source.index('\\subsection{Spectral processing}')
    b=source.index('\\section{Evaluation design}',a)
    dates=manifest.get('spectral_reference',{}).get('times',[])
    spectral=('\\subsection{Spectral processing}\nThe frozen evaluation records '+str(len(dates))+' training-reference hours for a fixed spectral adjustment. '
        +'Raw and processed predictions use the same member seeds. The transform is applied after generation; it does not retrain the model or use evaluation truth. '
        +'Reference dates and fixed gain/cutoff settings are retained in the parent manifest. Sharper gradients and spectra are assessed together with CRPS, bias, and calibration; amplitude adjustment does not establish correct feature location.\n\n')
    source=source[:a]+spectral+source[b:]
    a=source.index('The saved diagnostic contains 7 March 2026') if 'The saved diagnostic contains 7 March 2026' in source else source.index('% BEGIN INTERIM EVALUATION')
    b=source.index('\\subsection{Deterministic and probabilistic scores}',a)
    evaluation=(scope+' The snapshot includes every valid atomic hour record present when its ID list was frozen. '
        +'Completion depends on shard scheduling and runtime, so this subset is neither random nor a contiguous independent holdout. '
        +'Figure~\\ref{fig:data} shows its temporal support against all eligible hours. The original full-test completion checks remain unchanged.\n\n'
        +'\\subsection{Frozen full-test protocol and interim extraction}\\label{sec:fulltest}\n'
        +f"The parent run fixes {s['settings']['members']} members and {s['settings']['steps']} Heun steps with {s['settings']['weights']} weights. "
        +'Completed per-hour metrics are pooled on the fixed grid, with equal hour weighting and area weighting within each hour. '
        +'RMSE is the square root of pooled MSE; precipitation contingency, reliability, and FSS statistics pool their numerator/denominator terms. '
        +'The snapshot preserves parent run identity, selected IDs, case hashes, and any completed predeclared example fields. No new inference is performed for this interim report. '
        +'Temporal-block confidence intervals are withheld because completion-order sampling does not justify population inference.\n\n')
    source=source[:a]+'% BEGIN INTERIM EVALUATION\n'+evaluation+source[b:]
    # Four genuine tables, with unavailable controlled ablations explicitly unclaimed.
    rows=[[names[i]+' ('+units[i]+')',coarse,ref,support] for i,(coarse,ref,support) in enumerate([
        ('T2M','TMP\\_2M','Midpoint'),('PRECTOT','PRECTOT','Hourly trapezoid'),('PS','PRES\\_SFC','Midpoint'),
        ('U10M','UGRD\\_10M','Midpoint'),('V10M','VGRD\\_10M','Midpoint'),('QV2M','SPFH\\_2M','Midpoint')])]
    data_table=table('Generated variables, units, and learning reference. All reference fields are high-resolution GEOS replay output.',
        'tab:data','llll',['Variable','Coarse baseline','GEOS reference','Temporal support'],rows)
    cfg=manifest['config'];patch=cfg['patch'];model=cfg.get('model',{});record=json.loads((root/'cases'/f'{meta["entries"][0]["id"]}.json').read_text())
    rows=[['Snapshot scope',f'{n}/{N} eligible hours; descriptive interim subset'],['Evaluation checkpoint',f"Epoch {record['epoch']}; {s['settings']['weights']} weights"],
        ['Members / Heun steps',f"{s['settings']['members']} / {s['settings']['steps']}"],['Core / halo',f"{patch['size']} / {patch['halo']} pixels"],
        ['Context',f"Scale {patch['context_scale']}; pooled size {patch['context_size']} pixels"],
        ['Network',f"Base width {model.get('base_channels','not recorded')}; multipliers {model.get('channel_mult',[])}"],
        ['Post-processing',str(s['settings']['post'])],['Score confidence intervals','Withheld for completion-ordered interim subset']]
    settings_table=table('Frozen evaluation and configured model settings. The checkpoint epoch is read from saved cases.','tab:settings','lp{0.65\\textwidth}',['Setting','Value'],rows)
    rows=[]
    for i,f in enumerate(FIELDS):
        for m in methods:
            v=s['sources'][m]['fields'][f]
            rows.append([names[i],labels[m],number(v['rmse_pooled'],factors[i]),number(v['mae'],factors[i]),
                number(v['crps'],factors[i]),number(v['crpss_vs_coarse'],100),number(v['coverage_90'],100)])
    scores_table=table(scope+' Scores use the physical units in Table~\\ref{tab:data}; wind speed uses m s$^{-1}$. CRPS skill is relative to coarse input. Coverage is empirical central 90\\% coverage. No significance intervals are claimed.',
        'tab:scores','llrrrrr',['Variable','Method','RMSE','MAE','CRPS','Skill (\\%)','Coverage (\\%)'],rows)
    rows=[]
    for i,f in enumerate(FIELDS):
        r=raw[f];v=s['sources'].get('spectral',{}).get('fields',{}).get(f)
        rows.append([names[i],number(v['crps']-r['crps'],factors[i]) if v else '--',
            number(r['bias'],factors[i]),number(v['bias'],factors[i]) if v else '--',
            number(r['gradient_p99_members_mean']/r['gradient_p99_reference']) if r['gradient_p99_reference']>0 else '--',
            number(v['gradient_p99_members_mean']/v['gradient_p99_reference']) if v and v['gradient_p99_reference']>0 else '--'])
    processing_table=table('Paired processing effects on the interim subset. CRPS difference is spectral minus raw in physical units; negative favors processing. Gradient ratios compare member gradient p99 to the reference. Controlled representation/history/context ablations are not available and are not inferred from this table.',
        'tab:ablation','lrrrrr',['Variable','$\\Delta$CRPS','Raw bias','Spectral bias','Raw gradient','Spectral gradient'],rows)
    for key,text in [('data',data_table),('settings',settings_table),('scores',scores_table)]:
        source=apply_block(source,'TABLE_'+key,text,r'^\\tableslot\{[^\n]+\}\{tab:'+key+r'\}$')
    source=apply_block(source,'FIG_data',figure('interim_test_coverage',scope+' Expected and completed hours per UTC day. Completion-order coverage need not represent the full test period.','fig:data'),r'^\\figslot\{[^\n]+\}\{fig:data\}$')
    # Replace the old three-case results section in its entirety.
    results='\\section{Interim results}\\label{sec:results}\n'+scope+' All results below use this frozen subset; the remaining '+str(N-n)+' hours were not included.\n\n'
    results+='\\subsection{Deterministic and probabilistic comparison}\nTable~\\ref{tab:scores} and Fig.~\\ref{fig:skill} compare coarse input, frozen regression, raw flow, and paired spectral processing. Positive CRPS skill is observed for '+f'{count} of {len(skill)}'+' evaluated raw-flow fields. These are point estimates for the available subset, without a claim of full-test superiority.\n\n'
    for i,f in enumerate(FIELDS):
        v=raw[f];c=s['sources']['coarse']['fields'][f];reg=s['sources']['regression']['fields'][f]
        results+=names[i]+': raw-flow ensemble-mean RMSE is '+number(v['rmse_pooled'],factors[i])+' '+units[i]+', compared with '+number(c['rmse_pooled'],factors[i])+' for coarse input and '+number(reg['rmse_pooled'],factors[i])+' for regression. Raw empirical CRPS skill relative to coarse input is '+number(v['crpss_vs_coarse'],100)+r'\%. '
        if 'spectral' in methods:
            v=s['sources']['spectral']['fields'][f];results+='With spectral processing, RMSE is '+number(v['rmse_pooled'],factors[i])+' and CRPS skill is '+number(v['crpss_vs_coarse'],100)+r'\%. '
        results+='\n\n'
    results+=figure('fig02_skill',scope+' CRPS skill and pooled RMSE reduction relative to coarse input. All seven evaluated fields include derived wind speed.','fig:skill')
    results+='\\subsection{Calibration and conditional variability}\nRaw empirical central 90'+r'\% coverage ranges from '+number(min(cov))+' to '+number(max(cov))+r'\% across fields. Finite-member empirical quantile intervals are not population quantiles; both spread and bias contribute to coverage. '+f"The snapshot uses {s['settings']['members']} members. "+'Figures~\\ref{fig:calibration} and \\ref{fig:ranks} assess interval coverage, corrected spread/error, and randomized-tie observation ranks.\n\n'
    results+=figure('fig03_calibration',scope+' Empirical central 90\\% interval coverage and corrected spread-to-RMSE ratio.','fig:calibration')
    results+=figure('fig05_rank_histograms',scope+' Area-weighted randomized-tie ranks, averaged equally across saved hours.','fig:ranks')
    results+='\\subsection{Spatial structure, precipitation, and processing}\nThe spectral, reliability, FSS, and gradient diagnostics use the same subset and member pairing as the score table. No object-based storm displacement or humidity-front width is inferred from marginal spectra. Spectral processing changes amplitude rather than establishing fine-scale phase accuracy. Table~\\ref{tab:ablation} reports its paired effects.\n\n'
    for stem,label,caption in [
        ('fig04_spectral_structure','fig:spatial','Member power relative to reference power across index-space spatial frequencies.'),
        ('fig06_precipitation_reliability','fig:reliability','Precipitation exceedance reliability; insets show issued raw-flow probability mass. Empty bins carry no reliability estimate.'),
        ('fig07_precipitation_fss','fig:fss','Pooled precipitation FSS versus neighborhood width in grid pixels. Solid curves use ensemble means; faint dotted curves pool member numerator/denominator terms.'),
        ('fig08_filtering','fig:filtering','Paired raw/spectral gradient recovery and CRPS changes. No confidence intervals are shown for this interim subset.')]:
        results+=figure(stem,scope+' '+caption,label)
    results+='\\subsection{Representation hypothesis and computational record}\\label{sec:qualitative}\nDevelopment comparisons motivated direct square-root precipitation generation because organized rain patterns appeared better retained than under earlier residual-rain configurations. Sharper moisture features remained difficult. Those observations are hypotheses for matched training ablations, not causal conclusions from this interim evaluation. Joint network training does not establish thermodynamic consistency.\n\n'
    results+=f"Saved records contain {s['sampling_seconds']/3600:.2f} summed worker-hours of ensemble sampling and {s['total_worker_seconds']/3600:.2f} summed worker-hours of total evaluation. "
    results+=f"Mean recorded sampling time is {s['sampling_seconds']/n:.1f} s per test hour ({s['sampling_seconds']/n/s['settings']['members']:.1f} s per full-domain member). "
    results+='These sums exclude unsaved/interrupted work and startup; parallel worker time is not elapsed wall time. No dynamical-model speedup or hardware-independent efficiency claim follows.\n\n'+processing_table
    a=source.index('\\section{Preliminary results}') if '\\section{Preliminary results}' in source else source.index('\\section{Interim results}')
    b=source.index('\\section{Discussion}',a);source=source[:a]+results+source[b:]
    source=source.replace('The observed combination of improved CRPS, slightly degraded RMSE, and negative bias is therefore a substantive evaluation outcome, not a contradiction to be resolved by selecting a single favorable score.',
        'CRPS, ensemble-mean error, and physical-amount bias measure different properties; the interim table reports them together rather than selecting a single favorable score.')
    source=source.replace('The preliminary ensemble is informative relative to deterministic baselines under empirical CRPS, yet its uncertainty intervals remain too narrow or biased in the selected cases.',
        'The interim probability scores and empirical coverage describe different aspects of conditional performance and require full-test confirmation.')
    source=source.replace('If recipes were selected using the three current cases, an additional independent test sample is required.',
        'This snapshot preserves the parent run settings; no interim result is used to retune the recipe.')
    source=source.replace('summarized in the reserved Table~', 'summarized in Table~')
    if 'The saved presentation scorecard instead multiplies' in source:
        a=source.index('The saved presentation scorecard instead multiplies')
        b=source.index('Similarly, the empirical CRPS',a)
        source=source[:a]+('The interim aggregation uses the population-to-ensemble-mean variance correction stated above. '
            'Neither spread/error ratio alone proves calibration. Pooling second moments across hours precedes forming the aggregate ratio.\n\n')+source[b:]
    source=source.replace('because eight members poorly resolve tails',
        f"because {s['settings']['members']} members still provide limited tail resolution")
    source=source.replace('This selected example illustrates the method rather than complete-test skill.',
        'These workflow maps illustrate an earlier development diagnostic; the quantitative interim results use the separate frozen evaluation checkpoint in Table~\\ref{tab:settings}.')
    a=source.index('The available three-case processed ensemble') if 'The available three-case processed ensemble' in source else source.index('% BEGIN INTERIM CONCLUSIONS')
    b=source.index('\\section*{Author contributions}',a)
    conclusion=(f'{scope} Raw-flow empirical CRPS skill is positive for {count} of {len(skill)} evaluated fields; coverage spans '+number(min(cov))+'--'+number(max(cov))+r'\%. '
        +'These descriptive point estimates neither establish representative full-test skill nor observational accuracy. The paired raw/spectral diagnostics assess sharpening together with error and calibration, while controlled target-representation and conditioning ablations remain outstanding.\n\n'
        +'The full '+str(N)+'-hour evaluation remains in progress. Global coverage is future work; the present scope is CONUS. Final conclusions require the complete frozen test run, longer independent periods, and observational validation.\n\n'
        +'\\section*{Code and data availability}\nThe implementation is available in the \\model{} repository at \\url{https://github.com/afahadabdullah/MERRA21C-ML}. '
        +'This interim result preserves a frozen parent manifest, completed-hour JSON, available example fields, pooled summary, plotting data, and snapshot/code hashes. The high-resolution reference is project-hosted on NASA NCCS Discover; source-data access and permanent archival identifiers must be documented before submission.\n\n')
    source=source[:a]+'% BEGIN INTERIM CONCLUSIONS\n'+conclusion+source[b:]
    # Optional predeclared case maps in the appendix: all are also exported.
    map_stems=[]
    if meta['available_map_cases']:
        first=meta['available_map_cases'][0]
        map_stems=[f'fig09_{first}_{f}' for f in ('t2m','precip')]
        appendix='\\section{Saved interim map examples}\n'
        for i,stem in enumerate(map_stems):
            appendix+=figure(stem,scope+' Predeclared saved example '+first.replace('_',r'\_')+'. The same raw member, baselines, reference, and ensemble means use shared physical scales. This illustration is not evidence of population-wide organization.','fig:interim_example_'+str(i))
        if '% BEGIN INTERIM EXAMPLES' in source:source=apply_block(source,'EXAMPLES',appendix)
        else:source=source.replace('\\begin{thebibliography}', '% BEGIN INTERIM EXAMPLES\n'+appendix+'% END INTERIM EXAMPLES\n\\begin{thebibliography}',1)
    stems=['interim_test_coverage','fig02_skill','fig03_calibration','fig05_rank_histograms',
        'fig04_spectral_structure','fig06_precipitation_reliability','fig07_precipitation_fss','fig08_filtering',*map_stems]
    assets=path.parent/'figures'/'interim';assets.mkdir(parents=True,exist_ok=True)
    inline=''
    for stem in stems:
        name='MERRAFlow_interim_'+stem+'.pdf';target=assets/name
        shutil.copyfile(root/'paper_figures'/f'{stem}.pdf',target)
        shutil.copyfile(root/'paper_figures'/f'{stem}.png',target.with_suffix('.png'))
        inline+='\\begin{filecontents*}[overwrite]{'+name+'}\n'+inline_pdf(target)+'\\end{filecontents*}\n'
    if '% BEGIN INTERIM FIGURES' in source:source=apply_block(source,'FIGURES',inline)
    else:source=source.replace('\\begin{document}', '% BEGIN INTERIM FIGURES\n'+inline+'% END INTERIM FIGURES\n\\begin{document}',1)
    # No old numeric case claims should survive outside the embedded images.
    clean=re.sub(r'\\begin\{filecontents\*\}.*?\\end\{filecontents\*\}', '', source, flags=re.S)
    for stale in ('three-case processed ensemble', 'The saved three-case scorecard', '3822 s'):
        if stale in clean:raise ValueError(f'Stale case-study claim remains: {stale}')
    backup=root/'manuscript_before_update.tex'
    if not backup.exists():shutil.copyfile(path,backup)
    temporary=path.with_name(path.name+'.interim.tmp')
    temporary.write_text(source);temporary.replace(path)
    (root/'manuscript_update.json').write_text(json.dumps(dict(manuscript=str(path),snapshot_id=meta['snapshot_id'],hours=n,expected_hours=N),indent=2)+'\n')
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


def bundle(root, manuscript):
    root=Path(root);path=Path(manuscript);target=root/'MERRAFlow_interim_paper.zip'
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
        z.write(path,'manuscript/'+path.name)
        for p in (path.parent/'figures').rglob('*.pdf'):z.write(p,'manuscript/figures/'+str(p.relative_to(path.parent/'figures')))
        for name in ('snapshot.json','manifest.json','summary.json','manuscript_update.json'):
            z.write(root/name,'analysis/'+name)
        for p in (root/'cases').glob('*.json'):z.write(p,'analysis/cases/'+p.name)
        for p in (root/'paper_figures').iterdir():
            if p.suffix in ('.pdf','.csv','.json','.md'):z.write(p,'analysis/paper_figures/'+p.name)
        for p in (root/'pdf').glob('*.pdf'):z.write(p,'pdf/'+p.name)
        z.writestr('README.txt','INTERIM MERRA-Flow paper. Numerical results cover only the frozen subset in snapshot.json.\n'
            +'The complete snapshot directory retains example NPZ fields; this lightweight paper bundle omits them.\n'
            +'Compile the saved manuscript source normally or with the single-source desktop compiler.\n')
    return target


def main():
    p=argparse.ArgumentParser(description=__doc__)
    mode=p.add_mutually_exclusive_group();mode.add_argument('--evaluation');mode.add_argument('--snapshot')
    p.add_argument('--output',help='New immutable snapshot directory; defaults to a timestamped sibling of the evaluation')
    p.add_argument('--manuscript',default='manuscript/MERRA21C_ML_arxiv.tex',help='Existing manuscript .tex to update in place')
    p.add_argument('--skip-maps',action='store_true',help='Do not copy/render saved example ensembles')
    p.add_argument('--dpi',type=int,default=300)
    p.add_argument('--compile',action='store_true',help='Export a PDF if tectonic or pdflatex is installed')
    a=p.parse_args()
    if not Path(a.manuscript).is_file():p.error('--manuscript must be an existing source file')
    if a.dpi<72:p.error('--dpi must be at least 72')
    if not a.snapshot:
        a.evaluation=a.evaluation or 'runs/merraflow_v4_1/evaluation/all_test_best_m16_s64'
        a.output=a.output or (a.evaluation+'_interim_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    root=snapshot(a.evaluation,a.output,not a.skip_maps) if a.evaluation else Path(a.snapshot)
    render(root,a.dpi);update_paper(root,a.manuscript)
    if a.compile:compile_paper(root,a.manuscript)
    archive=bundle(root,a.manuscript)
    print(f'Interim snapshot and figures: {root}\nUpdated manuscript: {a.manuscript}\nPaper bundle: {archive}',flush=True)


if __name__=='__main__':main()
