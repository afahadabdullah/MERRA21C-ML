"""Paper figures from saved verification JSON and lossless example fields only.

No model, source archive, GPU, external downloads, or metric recomputation.
Use --workflow-only to draw Figure 1 before evaluation is available.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, TwoSlopeNorm
import numpy as np

from .evaluate_all_v4_1 import FIELDS, atomic_json, digest, field, read_manifest

LABELS = ('2 m temperature', 'Precipitation', 'Surface pressure', '10 m zonal wind',
          '10 m meridional wind', '2 m specific humidity', '10 m wind speed')
SHORT = ('Temperature', 'Precipitation', 'Pressure', 'Zonal wind', 'Meridional wind', 'Humidity', 'Wind speed')
UNITS = ('K', r'mm h$^{-1}$', 'hPa', r'm s$^{-1}$', r'm s$^{-1}$', r'g kg$^{-1}$', r'm s$^{-1}$')
FACTORS = (1., 1., .01, 1., 1., 1000., 1.)
METHODS = ('coarse', 'regression', 'raw', 'spectral')
NAMES = dict(coarse='Coarse input', regression='Regression', raw='Raw flow', spectral='Spectral flow')
COLORS = dict(coarse='#7b818a', regression='#b27821', raw='#0072b2', spectral='#d55e00')
STYLES = dict(coarse=':', regression='--', raw='-', spectral='-.')


def style():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.titlesize': 9,
        'axes.labelsize': 8, 'legend.fontsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .65,
        'lines.linewidth': 1.5, 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'savefig.dpi': 600, 'svg.fonttype': 'none'})


def export(fig, out, stem, dpi=600):
    """Vector labels/lines; rasterized meteorological fields at requested DPI."""
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'svg', 'png'):
        fig.savefig(out/f'{stem}.{ext}', dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def workflow(out, dpi=600, case=None):
    from .workflow_maps_v4_1 import render
    if case is None:
        raise ValueError('Map workflow requires --workflow-data or saved evaluation fields')
    return render(case,out,dpi)


def load_results(evaluation):
    root = Path(evaluation); manifest = read_manifest(root)
    completion = json.loads((root/'completion.json').read_text())
    summary = json.loads((root/'summary.json').read_text())
    if (not completion['complete'] or not summary['complete'] or
            summary['run_id'] != manifest['run_id'] or summary['hours'] != len(manifest['entries']) or
            completion['completed'] != len(manifest['entries'])):
        raise ValueError('Paper figures require a complete summary matching the frozen manifest')
    return root, manifest, summary


def methods(summary):
    return [m for m in METHODS if m in summary['sources']]


def panel(ax, i, title):
    ax.set_title(f'({chr(97+i)}) {title}', loc='left', pad=7)
    ax.grid(axis='y', color='.88', lw=.45, zorder=0)


def legend(fig, axs):
    handles, labels = axs.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, 1.12), ncol=4, frameon=False)


def skill(summary, out, dpi):
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.15), layout='constrained')
    x = np.arange(len(FIELDS)); ms = methods(summary)
    for j, (metric, title) in enumerate((('crps', 'CRPS skill relative to coarse input'),
                                        ('rmse_pooled', 'Pooled RMSE reduction'))):
        for k, m in enumerate(ms[1:]):
            base = np.array([summary['sources']['coarse']['fields'][f][metric] for f in FIELDS])
            values = np.array([summary['sources'][m]['fields'][f][metric] for f in FIELDS])
            reduction = np.divide(base-values, base, out=np.full(len(base), np.nan), where=base>0)*100
            pos = x+(k-(len(ms)-2)/2)*.23
            axs[j].bar(pos, reduction, width=.22, color=COLORS[m], label=NAMES[m], zorder=3)
            if metric == 'crps':
                ci = summary['bootstrap'].get('comparisons', {}).get(m+'_minus_coarse', {})
                for c, f in enumerate(FIELDS):
                    if f in ci and base[c] > 0:
                        lo, hi = -np.array(ci[f]['crps']['interval_95'])[::-1]/base[c]*100
                        axs[j].plot([pos[c], pos[c]], [lo, hi], color='.15', lw=.85, zorder=5)
        axs[j].axhline(0, color='.25', lw=.65)
        axs[j].set_xticks(x, SHORT, rotation=45, ha='right')
        axs[j].set_ylabel('Reduction (%)'); panel(axs[j], j, title)
    legend(fig, axs)
    export(fig, out, 'fig02_skill', dpi)


def calibration(summary, out, dpi):
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3), layout='constrained')
    x = np.arange(len(FIELDS))
    for m in methods(summary):
        if m in ('coarse', 'regression'): continue
        vals = summary['sources'][m]['fields']
        for ax, metric, factor in ((axs[0], 'coverage_90', 100), (axs[1], 'spread_skill_corrected', 1)):
            ax.plot(x, [vals[f][metric]*factor for f in FIELDS], marker='o', ms=3.5,
                    color=COLORS[m], ls=STYLES[m], label=NAMES[m])
    axs[0].axhline(90, color='.4', ls=':', lw=1); axs[0].set_ylabel('Coverage (%)')
    axs[1].axhline(1, color='.4', ls=':', lw=1); axs[1].set_ylabel('Corrected spread / RMSE')
    for i, ax in enumerate(axs):
        ax.set_xticks(x, SHORT, rotation=45, ha='right')
        panel(ax, i, ('Empirical central 90% intervals', 'Ensemble spread and error')[i])
    legend(fig, axs); export(fig, out, 'fig03_calibration', dpi)


def structure(summary, out, dpi):
    fig, axs = plt.subplots(2, 4, figsize=(10.2, 5.2), layout='constrained')
    for c, f in enumerate(FIELDS):
        ax = axs.flat[c]
        for m in methods(summary):
            s = summary['sources'][m]['fields'][f]['spectrum']
            ref, p = np.asarray(s['reference']), np.asarray(s['members'])
            values = np.divide(p, ref, out=np.full_like(p, np.nan), where=ref>0)
            ax.semilogx(s['frequency_cycles_pixel'], values, color=COLORS[m], ls=STYLES[m], label=NAMES[m])
        ax.axhline(1, color='.35', lw=.7)
        ax.set_xlabel('Frequency (cycles / grid pixel)'); ax.set_ylabel('Member power / reference power')
        panel(ax, c, SHORT[c])
    axs.flat[-1].axis('off'); legend(fig, axs)
    export(fig, out, 'fig04_spectral_structure', dpi)
    fig, axs = plt.subplots(2, 4, figsize=(10.2, 5.2), layout='constrained')
    for c, f in enumerate(FIELDS):
        ax = axs.flat[c]
        for m in ('raw', 'spectral'):
            if m not in summary['sources']: continue
            rank = summary['sources'][m]['fields'][f]['rank_histogram_hour_mean']
            ax.plot(np.arange(len(rank)), np.array(rank)*100, marker='o', ms=2.5,
                    color=COLORS[m], ls=STYLES[m], label=NAMES[m])
        ax.axhline(100/(summary['settings']['members']+1), color='.35', ls=':', lw=.8)
        ax.set_xlabel('Observation rank'); ax.set_ylabel('Area-weighted frequency (%)')
        panel(ax, c, SHORT[c])
    axs.flat[-1].axis('off'); legend(fig, axs)
    export(fig, out, 'fig05_rank_histograms', dpi)


def precipitation(summary, out, dpi):
    thresholds = ('0.1', '1.0', '5.0', '10.0', '25.0')
    fig, axs = plt.subplots(2, 3, figsize=(9.3, 5.6), layout='constrained')
    for i, threshold in enumerate(thresholds):
        ax = axs.flat[i]
        for m in ('raw', 'spectral'):
            if m not in summary['sources']: continue
            r = summary['sources'][m]['precipitation'][threshold]
            ax.plot(r['probability_bins'], r['reliability_frequency'], marker='o', ms=3,
                    color=COLORS[m], ls=STYLES[m], label=NAMES[m])
            # Inset shows forecast probability mass, so sparse high-probability bins are visible.
            inset = ax.inset_axes([.57, .1, .38, .25]) if m == 'raw' else None
            if inset is not None:
                mass = np.asarray(r['reliability_area']); mass = mass/mass.sum()
                inset.bar(r['probability_bins'], mass, width=.8/len(mass), color=COLORS[m])
                inset.set(xticks=[0, 1], yticks=[], title='Issued area fraction')
                inset.title.set_fontsize(6); inset.tick_params(labelsize=6)
        ax.plot([0, 1], [0, 1], ls=':', color='.35', lw=.8)
        ax.set(xlim=(-.03, 1.03), ylim=(-.03, 1.03), xlabel='Forecast probability', ylabel='Observed frequency')
        panel(ax, i, f'Rain ≥ {float(threshold):g} mm h$^{{-1}}$')
    axs.flat[-1].axis('off'); legend(fig, axs)
    export(fig, out, 'fig06_precipitation_reliability', dpi)
    fig, axs = plt.subplots(1, 3, figsize=(9, 2.9), layout='constrained')
    for i, threshold in enumerate(('1.0', '5.0', '10.0')):
        ax = axs[i]
        for m in methods(summary):
            r = summary['sources'][m]['precipitation'][threshold]['fss']
            windows = sorted(r, key=int)
            ax.plot([int(w) for w in windows], [r[w]['ensemble_mean'] for w in windows],
                    marker='o', ms=3, color=COLORS[m], ls=STYLES[m], label=NAMES[m])
            if m in ('raw', 'spectral'):
                ax.plot([int(w) for w in windows], [r[w]['members'] for w in windows],
                        color=COLORS[m], ls=':', alpha=.6, lw=.9)
        ax.set(ylim=(0, 1.03), xlabel='Neighborhood width (grid pixels)', ylabel='Pooled FSS')
        panel(ax, i, f'Rain ≥ {float(threshold):g} mm h$^{{-1}}$')
    legend(fig, axs); export(fig, out, 'fig07_precipitation_fss', dpi)


def filtering(summary, out, dpi):
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3), layout='constrained')
    for c, f in enumerate(FIELDS):
        raw = summary['sources']['raw']['fields'][f]
        for m in ('raw', 'spectral'):
            if m not in summary['sources']: continue
            v = summary['sources'][m]['fields'][f]
            g = v['gradient_p99_reference']
            axs[0].plot(c, v['gradient_p99_members_mean']/g if g>0 else np.nan,
                        'o' if m=='raw' else 's', ms=4, color=COLORS[m], label=NAMES[m] if c==0 else None)
        ci = summary['bootstrap'].get('comparisons', {}).get('spectral_minus_raw', {}).get(f, {}).get('crps')
        if 'spectral' in summary['sources'] and raw['crps']>0:
            delta = (summary['sources']['spectral']['fields'][f]['crps']/raw['crps']-1)*100
            axs[1].plot(c, delta, 'o', color=COLORS['spectral'], ms=4)
            if ci:
                axs[1].plot([c,c], np.array(ci['interval_95'])/raw['crps']*100, color='.2', lw=.8)
    axs[0].axhline(1, color='.4', ls=':', lw=.8)
    axs[1].axhline(0, color='.4', ls=':', lw=.8)
    axs[0].set_ylabel('Member gradient p99 / reference gradient p99')
    axs[1].set_ylabel('Spectral − raw CRPS (% of raw CRPS)')
    for i, ax in enumerate(axs):
        ax.set_xticks(range(len(FIELDS)), SHORT, rotation=45, ha='right')
        panel(ax, i, ('Recovery of fine-scale gradients', 'Paired effect of spectral adjustment')[i])
    legend(fig, axs); export(fig, out, 'fig08_filtering', dpi)


def map_norm(values, c):
    """One shared value scale across truth, baselines and generated fields."""
    if c == 1:
        edges = np.array([0, .1, .5, 1, 2, 5, 10, 20, 30, 50, 75, 100])
        cmap = plt.get_cmap('YlGnBu', len(edges)-1).with_extremes(under='white', over='#4d004b')
        return BoundaryNorm(edges, cmap.N), cmap
    lo, hi = np.quantile(values[0], [.005, .995])
    if c in (3, 4):
        limit = max(abs(lo), abs(hi), 1e-8)
        return TwoSlopeNorm(0, -limit, limit), 'RdBu_r'
    return matplotlib.colors.Normalize(lo, max(hi, lo+1e-8)), 'viridis' if c in (5,6) else 'plasma'


def saved_maps(root, manifest, out, dpi):
    selected = manifest['settings'].get('saved_cases', [])
    if not selected: return []
    item = manifest['saved_grid']
    if item['path'] != 'fields/grid.npz' or digest(root/item['path']) != item['sha256']:
        raise ValueError('Saved coordinate grid changed')
    with np.load(root/item['path'], allow_pickle=False) as g:
        lat, lon = g['lat'], g['lon']
    produced = []
    for case_id in selected:
        record = json.loads((root/'cases'/f'{case_id}.json').read_text())
        if record['run_id'] != manifest['run_id']: raise ValueError('Foreign map record')
        arrays = {}
        for name, item in record['saved_fields'].items():
            if item['path'] != f'fields/{case_id}_{name}.npz' or digest(root/item['path']) != item['sha256']:
                raise ValueError('Saved map data changed')
            with np.load(root/item['path'], allow_pickle=False) as f:
                if str(f['run_id']) != manifest['run_id'] or f['seeds'].tolist() != record['seeds']:
                    raise ValueError('Saved member provenance mismatch')
                # Keep only needed arrays from one example at a time.
                arrays[name] = {k: f[k] for k in f.files if k not in ('run_id','seeds')}
        raw = arrays['raw']
        for c, field_name in enumerate(FIELDS):
            means = [field(raw[k], c)*FACTORS[c] for k in ('truth','coarse','regression')]
            ens = field(raw['ensemble'], c)*FACTORS[c]
            means += [ens[0], ens.mean(0)]
            titles = ['HWT reference', 'Coarse (on HWT grid)', 'Regression', 'Raw member 1', 'Raw ensemble mean']
            if 'spectral' in arrays:
                means += [(field(arrays['spectral']['ensemble'], c)*FACTORS[c]).mean(0)]
                titles += ['Spectral ensemble mean']
            else:
                means += [ens.std(0)]; titles += ['Raw ensemble spread']
            norm, cmap = map_norm(means, c)
            fig, axs = plt.subplots(2, 3, figsize=(10.2, 5.4), layout='constrained')
            for i, (ax, data, title) in enumerate(zip(axs.flat, means, titles)):
                n, color = (matplotlib.colors.Normalize(0, max(float(np.quantile(data,.995)),1e-8)), 'magma') if title.endswith('spread') else (norm,cmap)
                image = ax.imshow(data, origin='lower', interpolation='nearest', cmap=color, norm=n, rasterized=True)
                # Geographic context from the saved coordinates; native LCC index axes.
                for geo in (lat,lon):
                    levels = np.arange(np.ceil(geo.min()/10)*10, geo.max(), 10)
                    if len(levels):
                        contour = ax.contour(geo, levels=levels, colors='.35', linewidths=.3, alpha=.6)
                        ax.clabel(contour, fmt='%d°', fontsize=5, inline=True)
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_title(f'({chr(97+i)}) {title}', loc='left')
                if title.endswith('spread'): fig.colorbar(image, ax=ax, shrink=.7, label=UNITS[c])
            shared = matplotlib.cm.ScalarMappable(norm=norm,cmap=cmap)
            fig.colorbar(shared, ax=list(axs.flat), shrink=.78, pad=.015, label=UNITS[c], extend='both')
            fig.suptitle(f'{LABELS[c]} | {record["time"].replace("T"," ")} UTC', fontsize=11)
            stem = f'fig09_{case_id}_{field_name}'
            export(fig, out, stem, dpi); produced.append(stem)
            # Ensemble uncertainty and paired filtering, with common spread/error scales.
            spread = ens.std(0)
            vals = [ens.mean(0)-means[0], spread]
            title2 = ['Raw mean − reference', 'Raw ensemble spread']
            if 'spectral' in arrays:
                filtered = field(arrays['spectral']['ensemble'],c)*FACTORS[c]
                vals += [filtered.mean(0)-means[0], filtered.std(0)]
                title2 += ['Spectral mean − reference', 'Spectral ensemble spread']
            fig, axes = plt.subplots(len(vals)//2, 2, figsize=(7.2, 2.5*(len(vals)//2)), layout='constrained', squeeze=False)
            limit = max(max(float(np.quantile(abs(v),.995)) for v in vals[::2]),1e-8)
            spreadmax = max(max(float(np.quantile(v,.995)) for v in vals[1::2]),1e-8)
            for i,(ax,v,title) in enumerate(zip(axes.flat,vals,title2)):
                n = TwoSlopeNorm(0,-limit,limit) if i%2==0 else matplotlib.colors.Normalize(0,spreadmax)
                im = ax.imshow(v,origin='lower',interpolation='nearest',norm=n,cmap='RdBu_r' if i%2==0 else 'magma',rasterized=True)
                ax.set(xticks=[],yticks=[],title=f'({chr(97+i)}) {title}')
                fig.colorbar(im,ax=ax,shrink=.8,label=UNITS[c],extend='both')
            fig.suptitle(f'{LABELS[c]} | {record["time"].replace("T"," ")} UTC',fontsize=10)
            stem=f'fig10_{case_id}_{field_name}_uncertainty';export(fig,out,stem,dpi);produced.append(stem)
        del arrays
    return produced


def captions(summary, maps):
    return '\n'.join([
        '# Paper figure captions and conventions', '',
        'Figure 1. MERRA-Flow v4.1 map-based CONUS workflow with native GEOS-FP cells and matched regional temperature zooms, six generated high-resolution fields, hybrid generative targets, halo-supervised conditional flow matching, and synchronized ensemble inference. The zoom panels use identical extent/color scale without smoothing. Native source fields are used when saved; interpolated coarse inputs are explicitly labeled otherwise. Model prediction maps use the first fixed raw member for all fields, and fixed temperature members, from the first predeclared case. HRRR illustration bundles are rejected. Optional raw/spectral filtering is paired without retraining. P₀ = 1 mm h⁻¹.', '',
        f'Figures 2–8 summarize all {summary["hours"]} eligible held-out hours, with fixed-grid area weighting and equal hour weighting. They verify the high-resolution GEOS replay, not independent observations. Missing source hours and exclusions are recorded in the manifest.', '',
        'Figure 2. CRPS skill and pooled RMSE reduction relative to the coarse input. CRPS whiskers transform the paired temporal-block bootstrap intervals for absolute score differences using the fixed full-test coarse score; they are not bootstrap intervals of the skill-score ratio itself. RMSE bars have no confidence intervals.', '',
        'Figure 3. Empirical central 90% interval coverage and corrected ensemble spread/error. Dotted lines indicate 90% and unity. Finite-ensemble empirical quantile intervals do not imply exactly 90% coverage under ideal calibration.', '',
        'Figure 4. Time-averaged member power relative to reference power, computed from detrended Hann-tapered radial spectra. Frequency is in cycles per native grid pixel; these are index-space spectra, not exact isotropic geographic spectra. Members, rather than ensemble means, show resolved variability.', '',
        'Figure 5. Randomized-tie, area-weighted observation ranks. Dotted lines indicate uniform ranks. Each hour is weighted equally.', '',
        'Figure 6. Precipitation reliability at five thresholds. Empty bins have no observed frequency. Insets show the raw ensemble forecast-probability area fraction; sparse bins should be interpreted cautiously. No reliability confidence intervals are asserted.', '',
        'Figure 7. Fractions skill scores pooled over all hours. Main curves use ensemble-mean event fields; faint dotted curves use pooled member terms. Neighborhood widths are native grid pixels. All-dry undefined scores are omitted.', '',
        'Figure 8. Mean hourly 99th-percentile member gradient magnitude relative to the reference, and paired spectral-minus-raw CRPS changes. Lower CRPS changes indicate improvement. Whiskers, when available, use paired block intervals with a fixed raw CRPS denominator.', '',
        f'Bootstrap: {summary["bootstrap"]["block_hours"]}-hour nonoverlapping elapsed-time blocks; {summary["bootstrap"]["blocks"]} blocks; {summary["bootstrap"]["draws"]} draws. Status: {summary["bootstrap"]["status"]}. Intervals are not adjusted for multiple comparisons.', '',
        'Figures 9–10. Predeclared saved examples on the native HWT LCC array, with nearest-neighbor rendering and saved latitude/longitude contours where available. The coarse input is interpolated to this grid. All value panels share one scale; wind components use a symmetric scale. Temperature maps use K, pressure hPa, humidity g kg⁻¹, rain mm h⁻¹, and winds m s⁻¹. Error and spread scales match across raw/spectral variants. Raw member 1 is fixed rather than selected for visual quality. Case maps are illustrative and do not replace complete-test scores.', '',
        'Plot scales: rain uses fixed bins; state map limits use reference-only 0.5/99.5 percentiles for visualization. These choices change no evaluation metric. Float32 fields are saved losslessly; only plotted images are rasterized.', '',
        'Example figure stems: '+', '.join(maps), ''])


def make(evaluation=None, output='paper_figures_v4_1', dpi=600, workflow_only=False, skip_maps=False, workflow_data=None):
    if dpi < 72: raise ValueError('DPI must be at least 72')
    style(); out=Path(output); out.mkdir(parents=True,exist_ok=True)
    provenance=dict(plotter_sha256=digest(__file__),matplotlib=matplotlib.__version__,numpy=np.__version__,
                    dpi=dpi,workflow_only=workflow_only,skip_maps=skip_maps)
    if workflow_only:
        from .workflow_maps_v4_1 import load_bundle
        if not workflow_data: raise ValueError('--workflow-only needs --workflow-data; use scripts/make_workflow_figure.py for the public map illustration')
        workflow(out,dpi,load_bundle(workflow_data));atomic_json(out/'plot_manifest.json',provenance);return out
    if evaluation is None: raise ValueError('--evaluation is required unless --workflow-only')
    root, manifest, summary=load_results(evaluation)
    previous=out/'plot_manifest.json'
    if previous.exists():
        old=json.loads(previous.read_text())
        if old.get('run_id') not in (None,manifest['run_id']):
            raise ValueError('Use a separate figure directory for a different evaluation run')
    from .workflow_maps_v4_1 import from_evaluation,load_bundle
    if workflow_data:
        workflow(out,dpi,load_bundle(workflow_data))
    elif manifest['settings'].get('saved_cases',[]) and not skip_maps:
        workflow(out,dpi,from_evaluation(root,manifest))
    skill(summary,out,dpi);calibration(summary,out,dpi)
    structure(summary,out,dpi);precipitation(summary,out,dpi);filtering(summary,out,dpi)
    maps=[] if skip_maps else saved_maps(root,manifest,out,dpi)
    # Bundle every plotted diagnostic without reading the source archive again.
    atomic_json(out/'figure_data.json',summary)
    with (out/'table01_all_variables.csv').open('w',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['method','field','unit','rmse','mae','bias','crps','fair_crps','coverage_90','spread_skill_corrected'])
        for m in methods(summary):
            for c,name in enumerate(FIELDS):
                vals=summary['sources'][m]['fields'][name]
                writer.writerow([m,name,UNITS[c],*[None if vals[k] is None else vals[k]*FACTORS[c] for k in
                    ('rmse_pooled','mae','bias','crps','fair_crps')],vals['coverage_90'],vals['spread_skill_corrected']])
    (out/'captions.md').write_text(captions(summary,maps))
    provenance.update(run_id=manifest['run_id'],checkpoint_sha256=manifest['checkpoint_sha256'],
        summary_sha256=digest(root/'summary.json'),manifest_sha256=digest(root/'manifest.json'),
        evaluation=str(root.resolve()),saved_cases=manifest['settings'].get('saved_cases',[]),
        files={p.name:digest(p) for p in sorted(out.iterdir()) if p.suffix in ('.pdf','.svg','.png')})
    atomic_json(out/'plot_manifest.json',provenance)
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluation');p.add_argument('--output',default='paper_figures_v4_1')
    p.add_argument('--dpi',type=int,default=600);p.add_argument('--workflow-only',action='store_true')
    p.add_argument('--skip-maps',action='store_true')
    p.add_argument('--workflow-data',help='Saved workflow_data.npz for map-based Figure 1')
    print(make(**vars(p.parse_args())))


if __name__=='__main__':main()
