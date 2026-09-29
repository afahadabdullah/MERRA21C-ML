"""Presentation figures for v4.1: maps and skill diagnostics on a few cases.

For each case (--samples random hours of the split, plus the --wettest hour(s), or
--timestamps), the full domain is sampled with --members members (the production tiled
sampler; --post spectral applies the spectral fix to every member). Then, per variable
(rain, t2m, ps, u10m, v10m, q2m), ONE figure each:

  cases/<id>/conus_<var>.{png,pdf}     GEOS-FP 0.25° | truth 2 km | member 1 | ensemble mean
                                       (zoom windows outlined)
  cases/<id>/zoom<k>_<var>.{png,pdf}   the same four panels on a zoom window (the rain event
                                       first, then --zooms-1 random windows)
  cases/<id>/uncertainty_<var>.{png,pdf}  |ensemble mean − truth| beside the ensemble spread, same
                                       scale: where the model is unsure is where it is wrong
  cases/<id>/diversity_<var>.{png,pdf} truth + 4 members + spread on the rain-event zoom
                                       (member diversity)
  cases/<id>/scorecard.{png,pdf,csv}   this case's scores
  cases/<id>/reveal_<var>.png          GEOS-FP | truth | one prediction (rain, t2m, 10 m wind speed;
                                       --animate): the slide before the "click"
  cases/<id>/members_<var>.gif         the same three panels with the prediction cycling through all
                                       members: one input, many plausible 2 km outcomes

GEOS-FP is drawn as its original ~25 km cells, every other panel on the native 2 km
Lambert grid, all with nearest-neighbour imshow (no smoothing); one colour scale per
figure. Skill diagnostics, averaged over all cases (per-case numbers in the CSV files):

  skill_crps.{png,pdf}           CRPS relative to GEOS-FP (1 = GEOS-FP; lower is better):
                                 GEOS-FP, frozen regression mean, one member, ensemble
  skill_rmse.{png,pdf}           the same for RMSE (ensemble = ensemble-mean RMSE)
  spread_skill.{png,pdf}         ensemble spread / ensemble-mean RMSE (1 = calibrated)
  scorecard.{png,pdf,csv}        table: CRPS, CRPS skill score, RMSE, bias, correlation,
                                 spread/skill, 90 % interval coverage
  spectra.{png,pdf}              radial power spectra vs wavelength (km)
  spectra_ratio.{png,pdf}        the same as a ratio to truth
  rank_histograms.{png,pdf}      ensemble rank histograms (flat = reliable)
  precip_diagnostics.{png,pdf}   FSS vs scale, rain-rate distribution, Q-Q, reliability
  overview.{png,pdf}             one-slide summary: CRPS skill, calibration, rain spectrum
  metrics.json

Everything uses large fonts, short titles and one consistent colour per source, for slides.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import time
import numpy as np
import torch

from .config import write_json
from .metrics import weighted_mean
from .train import device_for
from .v4 import TARGETS
from .v4_1 import load_config
from .evaluate_v4_1 import (resolve_checkpoint, load_model, select_cases, member_seed, DomainSampler, Canvas,
                            native_fields, event_window, random_windows, case_metrics, _coarse_panel, _native_shown,
                            _norm, _display, DISPLAY, FSS_SCALES)
from .validation_v4_1 import RAIN_EDGES, RAIN_LEVELS, _rain_norm

FIELDS = ('precip', 't2m', 'ps', 'u10m', 'v10m', 'q2m')
LABEL = {'precip': 'Rain rate', 't2m': '2 m temperature', 'ps': 'Surface pressure', 'u10m': '10 m zonal wind',
         'v10m': '10 m meridional wind', 'q2m': '2 m specific humidity'}
SHORT = {'precip': 'Rain', 't2m': 't2m', 'ps': 'ps', 'u10m': 'u10m', 'v10m': 'v10m', 'q2m': 'q2m'}
SOURCES = ('coarse', 'regression', 'member_0', 'ensemble')
SOURCE_LABEL = {'coarse': 'GEOS-FP 0.25°', 'regression': 'Regression mean', 'member_0': 'One member',
                'ensemble': 'Ensemble', 'truth': 'Truth 2 km', 'members': 'Members', 'ensemble_mean': 'Ensemble mean'}
COLOR = {'truth': '#1A1A1A', 'coarse': '#9AA3AD', 'regression': '#6FA8C7', 'member_0': '#E0823D', 'members': '#E0823D',
         'ensemble': '#1F4E79', 'ensemble_mean': '#1F4E79'}


def style(plt):
    plt.rcParams.update({'font.size': 14, 'axes.titlesize': 16, 'axes.labelsize': 15, 'xtick.labelsize': 13,
                         'ytick.labelsize': 13, 'legend.fontsize': 13, 'figure.titlesize': 19,
                         'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
                         'figure.facecolor': 'white', 'savefig.facecolor': 'white', 'axes.titleweight': 'bold'})


def save(fig, path, dpi, pdf, plt):
    fig.savefig(path.with_suffix('.png'), dpi=dpi)
    if pdf:
        fig.savefig(path.with_suffix('.pdf'), dpi=dpi)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Maps
# ----------------------------------------------------------------------------

def colour_scale(canvas, item, name, ys, xs, shown):
    """(cmap, norm) for a variable: rain levels, else quantiles of truth, GEOS-FP and native cells."""
    if name == 'precip':
        return _rain_norm()
    pool = [f[ys, xs].ravel() for f in shown]
    native = _native_shown(item, name)
    if native is not None:
        lat, lon = canvas.lat[ys, xs], canvas.lon[ys, xs]
        keep = np.ix_((native['lat'] >= lat.min()) & (native['lat'] <= lat.max()),
                      (native['lon'] >= lon.min()) & (native['lon'] <= lon.max()))
        pool.append(native['values'][keep].ravel())
    return DISPLAY[name][4], _norm(name, np.concatenate(pool))


def uncertainty_figure(canvas, item, name, heading, path, plt, dpi, pdf):
    """|ensemble mean − truth| and ensemble spread on one scale: a useful spread is large where errors are."""
    from matplotlib.gridspec import GridSpec
    c = TARGETS.index(name)
    ens = item['ensemble'][:, c]
    error = np.abs(_display(name, ens.mean(0)-item['truth'][c], difference=True))
    spread = _display(name, ens.std(0), difference=True)
    top = max(float(np.quantile(np.concatenate([error.ravel(), spread.ravel()]), .995)), 1e-6)
    h, w = error.shape
    panel = 7.2
    fig = plt.figure(figsize=(2*panel+.8, panel*h/w+1.9), constrained_layout=True)
    grid = GridSpec(2, 2, figure=fig, height_ratios=[1, .055])
    area = item['area']
    unit = DISPLAY[name][1]
    image = None
    for j, (field, title) in enumerate(((error, 'Error of the ensemble mean |mean − truth|'),
                                        (spread, 'Ensemble spread (standard deviation)'))):
        ax = canvas.axes(fig, grid[0, j])
        image = canvas.show(ax, field, None, cmap='magma_r', norm=plt.Normalize(0, top), left=j == 0, bottom=True)
        ax.set_title(f'{title}\ndomain mean {weighted_mean(field, area):.3g} {unit}', fontsize=15)
    bar = fig.colorbar(image, cax=fig.add_subplot(grid[1, :]), orientation='horizontal', extend='max')
    bar.set_label(f'{DISPLAY[name][0]} ({unit})')
    fig.suptitle(heading)
    save(fig, path, dpi, pdf, plt)


def members_figure(canvas, item, name, window, heading, path, plt, dpi, pdf, count=4):
    """Truth, several members and the spread on one zoom: are the members different, and realistic?"""
    from matplotlib.gridspec import GridSpec
    c = TARGETS.index(name)
    ys, xs = canvas.region(window, item['truth'][c].shape)
    ens = item['ensemble'][:, c]
    count = min(count, len(ens))
    fields = [item['truth'][c]]+[ens[k] for k in range(count)]
    shown = fields if name == 'precip' else [_display(name, f) for f in fields]
    cmap, norm = colour_scale(canvas, item, name, ys, xs, shown[:1])
    spread = _display(name, ens.std(0), difference=True)
    panel = 4.6
    cols = count+2
    fig = plt.figure(figsize=(cols*panel+.8, panel+1.9), constrained_layout=True)
    grid = GridSpec(2, cols, figure=fig, height_ratios=[1, .055])
    image = None
    for j, field in enumerate(shown):
        ax = canvas.axes(fig, grid[0, j])
        image = canvas.show(ax, field, window, cmap=cmap, norm=norm, left=j == 0, bottom=True)
        ax.set_title('Truth 2 km' if j == 0 else f'Member {j}')
    kwargs = dict(ticks=RAIN_LEVELS, extend='both') if name == 'precip' else dict(extend='both')
    bar = fig.colorbar(image, cax=fig.add_subplot(grid[1, :cols-1]), orientation='horizontal', **kwargs)
    bar.set_label(f'{DISPLAY[name][0]} ({DISPLAY[name][1]})')
    ax = canvas.axes(fig, grid[0, cols-1])
    top = max(float(np.quantile(spread[ys, xs], .995)), 1e-6)
    spread_image = canvas.show(ax, spread, window, cmap='magma_r', norm=plt.Normalize(0, top), bottom=True)
    ax.set_title(f'Spread ({len(ens)} members)')
    bar = fig.colorbar(spread_image, cax=fig.add_subplot(grid[1, cols-1]), orientation='horizontal', extend='max')
    bar.set_label(DISPLAY[name][1])
    fig.suptitle(heading)
    save(fig, path, dpi, pdf, plt)


def _field_for(item, name, which):
    """Physical field(s) for a TARGETS name or the derived 'wind_speed' (which: coarse|truth|members)."""
    if name == 'wind_speed':
        u, v = TARGETS.index('u10m'), TARGETS.index('v10m')
        if which == 'members':
            return np.hypot(item['ensemble'][:, u], item['ensemble'][:, v])
        return np.hypot(item[which][u], item[which][v])
    c = TARGETS.index(name)
    return item['ensemble'][:, c] if which == 'members' else item[which][c]


def _image_data(canvas, field, window):
    """What canvas.show would put in its image for this field and window (for animation frames)."""
    ys, xs = canvas.region(window, field.shape)
    data = field[ys, xs]
    if canvas.mode == 'lcc' and canvas.dx < 0:
        data = data[:, ::-1]
    return data


def members_animation(canvas, item, name, window, heading, folder, plt, dpi, fps=1.5):
    """reveal_<name>.png (GEOS-FP | truth | member 1) and members_<name>.gif with the prediction panel
    cycling through every member while GEOS-FP and truth stay fixed."""
    from matplotlib import animation
    from matplotlib.gridspec import GridSpec
    ys, xs = canvas.region(window, item['truth'][0].shape)
    convert = (lambda f: f) if name in ('precip', 'wind_speed') else (lambda f: _display(name, f))
    coarse, truth = convert(_field_for(item, name, 'coarse')), convert(_field_for(item, name, 'truth'))
    members = [convert(m) for m in _field_for(item, name, 'members')]
    cmap, norm = colour_scale(canvas, item, name, ys, xs, [truth, coarse])
    panel = 5.6
    fig = plt.figure(figsize=(3*panel+.8, panel+2.0), constrained_layout=True)
    grid = GridSpec(2, 3, figure=fig, height_ratios=[1, .055])
    ax = canvas.axes(fig, grid[0, 0])
    image, label, _, _ = _coarse_panel(canvas, ax, item, name, coarse, window, cmap, norm, True, True)
    ax.set_title('GEOS-FP 0.25° (input)' if 'native' in label.lower() else 'GEOS-FP (input, regridded)')
    ax = canvas.axes(fig, grid[0, 1])
    canvas.show(ax, truth, window, cmap=cmap, norm=norm, bottom=True)
    ax.set_title('Truth 2 km')
    ax = canvas.axes(fig, grid[0, 2])
    moving = canvas.show(ax, members[0], window, cmap=cmap, norm=norm, bottom=True)
    title = ax.set_title('Prediction 2 km')
    kwargs = dict(ticks=RAIN_LEVELS, extend='both') if name == 'precip' else dict(extend='both')
    bar = fig.colorbar(moving, cax=fig.add_subplot(grid[1, :]), orientation='horizontal', **kwargs)
    bar.set_label(f'{DISPLAY[name][0]} ({DISPLAY[name][1]})')
    fig.suptitle(heading)
    fig.savefig(folder/f'reveal_{name}.png', dpi=dpi)

    def frame(k):
        data = _image_data(canvas, members[k], window)
        if hasattr(moving, 'set_data'):
            moving.set_data(data)
        else:   # pcolormesh fallback (no LCC grid): same subsampling as Canvas.show
            step = max(1, max(data.shape)//900)
            moving.set_array(data[::step, ::step].ravel())
        title.set_text(f'Member {k+1} of {len(members)}')
        return [moving, title]
    anim = animation.FuncAnimation(fig, frame, frames=len(members), blit=False)
    anim.save(folder/f'members_{name}.gif', writer=animation.PillowWriter(fps=fps), dpi=max(80, dpi//3))
    plt.close(fig)


def map_figure(canvas, item, name, window, heading, path, plt, dpi, pdf, boxes=()):
    """GEOS-FP | truth | member 1 | ensemble mean for one variable, one colour scale."""
    from matplotlib.gridspec import GridSpec
    c = TARGETS.index(name)
    ys, xs = canvas.region(window, item['truth'][c].shape)
    h, w = ys.stop-ys.start, xs.stop-xs.start
    panel = 5.4
    fig = plt.figure(figsize=(4*panel+.8, panel*h/w+1.9), constrained_layout=True)
    grid = GridSpec(2, 4, figure=fig, height_ratios=[1, .055])
    ens = item['ensemble'][:, c]
    fields = [item['coarse'][c], item['truth'][c], ens[0], ens.mean(0)]
    title, unit = DISPLAY[name][0], DISPLAY[name][1]
    shown = fields if name == 'precip' else [_display(name, f) for f in fields]
    cmap, norm = colour_scale(canvas, item, name, ys, xs, shown[:2])
    image = None
    titles = [None, 'Truth 2 km', 'One member', f'Ensemble mean ({len(ens)})']
    for j, field in enumerate(shown):
        ax = canvas.axes(fig, grid[0, j])
        if j == 0:
            image, label, _, _ = _coarse_panel(canvas, ax, item, name, field, window, cmap, norm, True, True)
            ax.set_title('GEOS-FP 0.25°' if 'native' in label.lower() else 'GEOS-FP (regridded)')
        else:
            image = canvas.show(ax, field, window, cmap=cmap, norm=norm, bottom=True)
            ax.set_title(titles[j])
        for box, text in boxes:
            canvas.box(ax, box, text)
    cax = fig.add_subplot(grid[1, 1:3])
    kwargs = dict(ticks=RAIN_LEVELS, extend='both') if name == 'precip' else dict(extend='both')
    bar = fig.colorbar(image, cax=cax, orientation='horizontal', **kwargs)
    bar.set_label(f'{title} ({unit})')
    fig.suptitle(heading)
    save(fig, path, dpi, pdf, plt)


# ----------------------------------------------------------------------------
# Aggregation and skill figures
# ----------------------------------------------------------------------------

def average_reports(reports):
    """Mean of the per-case continuous scores: {field: {source: {metric: value}}}."""
    out = {}
    for name in FIELDS:
        out[name] = {}
        for source in SOURCES:
            keys = reports[0][name][source].keys()
            out[name][source] = {k: float(np.mean([r[name][source][k] for r in reports
                                                   if r[name][source][k] is not None]))
                                 for k in keys if any(r[name][source][k] is not None for r in reports)}
    return out


def spread_skill(entry, members):
    return entry['spread']*np.sqrt((members+1)/members)/max(entry['rmse'], 1e-30)


def plot_relative(mean, metric, ylabel, title, path, plt, dpi, pdf):
    names = list(FIELDS)
    x = np.arange(len(names))
    width = .2
    fig, ax = plt.subplots(figsize=(13, 6), constrained_layout=True)
    for k, source in enumerate(SOURCES):
        values = [mean[n][source][metric]/max(mean[n]['coarse'][metric], 1e-30) for n in names]
        bars = ax.bar(x+(k-1.5)*width, values, width, color=COLOR[source], label=SOURCE_LABEL[source])
        if source == 'ensemble':
            for bar_, v in zip(bars, values):
                ax.text(bar_.get_x()+bar_.get_width()/2, v+.02, f'{100*(1-v):+.0f}%', ha='center', va='bottom',
                        fontsize=12, color=COLOR['ensemble'], fontweight='bold')
    ax.axhline(1, color='#555555', lw=1.2, ls='--')
    ax.set_xticks(x, [SHORT[n] for n in names])
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(1.25, ax.get_ylim()[1]))
    ax.set_title(title, loc='left')
    ax.legend(ncol=4, loc='upper left', bbox_to_anchor=(0, -.08))
    save(fig, path, dpi, pdf, plt)


def plot_spread_skill(mean, members, path, plt, dpi, pdf):
    names = list(FIELDS)
    values = [spread_skill(mean[n]['ensemble'], members) for n in names]
    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    bars = ax.bar([SHORT[n] for n in names], values, color=COLOR['ensemble'], width=.55)
    for bar_, v in zip(bars, values):
        ax.text(bar_.get_x()+bar_.get_width()/2, v+.02, f'{v:.2f}', ha='center', va='bottom', fontsize=13)
    ax.axhline(1, color='#555555', lw=1.2, ls='--')
    ax.text(len(names)-.5, 1.02, 'calibrated', ha='right', va='bottom', fontsize=12, color='#555555')
    ax.set_ylim(0, max(1.3, max(values)+.15))
    ax.set_ylabel('Spread / RMSE of ensemble mean')
    ax.set_title('Ensemble calibration: spread vs error (1 = calibrated)', loc='left')
    save(fig, path, dpi, pdf, plt)


def scorecard_rows(mean, members):
    rows = []
    for n in FIELDS:
        unit = DISPLAY[n][1]
        factor = DISPLAY[n][3]
        m, e, cz = mean[n], mean[n]['ensemble'], mean[n]['coarse']
        rows.append(dict(field=n, unit=unit,
                         crps_geosfp=cz['crps']*factor, crps_ensemble=e['crps']*factor,
                         crpss=1-e['crps']/max(cz['crps'], 1e-30),
                         rmse_geosfp=cz['rmse']*factor, rmse_ensemble_mean=e['rmse']*factor,
                         bias_ensemble_mean=e['bias']*factor,
                         correlation=float('nan') if e.get('correlation') is None else float(e['correlation']),
                         spread_skill=spread_skill(e, members), coverage_90=e['coverage_90'],
                         rmse_member=m['member_0']['rmse']*factor))
    return rows


def _fmt(value, spec):
    if value is None or not np.isfinite(value):
        return 'n/a'
    return format(value, spec)


def plot_scorecard(rows, path, plt, dpi, pdf, heading):
    columns = [('Field', lambda r: f'{SHORT[r["field"]]} ({r["unit"]})'),
               ('CRPS\nGEOS-FP', lambda r: _fmt(r['crps_geosfp'], '.3g')),
               ('CRPS\nensemble', lambda r: _fmt(r['crps_ensemble'], '.3g')),
               ('CRPS skill\nvs GEOS-FP', lambda r: _fmt(100*r['crpss'], '+.0f')+'%'),
               ('RMSE\nGEOS-FP', lambda r: _fmt(r['rmse_geosfp'], '.3g')),
               ('RMSE\nens. mean', lambda r: _fmt(r['rmse_ensemble_mean'], '.3g')),
               ('Bias\nens. mean', lambda r: _fmt(r['bias_ensemble_mean'], '+.2g')),
               ('Corr.\nens. mean', lambda r: _fmt(r['correlation'], '.3f')),
               ('Spread /\nskill', lambda r: _fmt(r['spread_skill'], '.2f')),
               ('90 %\ncoverage', lambda r: _fmt(100*r['coverage_90'], '.0f')+'%')]
    cells = [[fmt(r) for _, fmt in columns] for r in rows]
    fig, ax = plt.subplots(figsize=(18, 1.0+.72*(len(rows)+1)), constrained_layout=True)
    ax.axis('off')
    table = ax.table(cellText=cells, colLabels=[c for c, _ in columns], loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1, 2.6)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('#D5DAE0')
        if row == 0:
            cell.set_facecolor('#1F4E79')
            cell.get_text().set_color('white')
            cell.get_text().set_fontweight('bold')
        elif row % 2 == 0:
            cell.set_facecolor('#F2F5F8')
        if row > 0 and col == 3:
            cell.get_text().set_fontweight('bold')
            cell.get_text().set_color('#1F4E79')
    ax.set_title(heading, loc='left', fontsize=17)
    save(fig, path, dpi, pdf, plt)


def write_csv(rows, path):
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        for r in rows:
            writer.writerow({k: (f'{v:.6g}' if isinstance(v, float) else v) for k, v in r.items()})


def plot_overview(mean, spectra, members, dx_km, tag, path, plt, dpi, pdf):
    """One slide: ensemble CRPS skill vs GEOS-FP, calibration, and the rain spectrum."""
    fig, axes = plt.subplots(1, 3, figsize=(22, 6.6), constrained_layout=True,
                             gridspec_kw=dict(width_ratios=[1.15, 1, 1.1]))
    names = list(FIELDS)
    skill = [100*(1-mean[n]['ensemble']['crps']/max(mean[n]['coarse']['crps'], 1e-30)) for n in names]
    ax = axes[0]
    bars = ax.bar([SHORT[n] for n in names], skill, color=COLOR['ensemble'], width=.6)
    for bar_, v in zip(bars, skill):
        ax.text(bar_.get_x()+bar_.get_width()/2, v+(1 if v >= 0 else -1), f'{v:+.0f}%', ha='center',
                va='bottom' if v >= 0 else 'top', fontsize=14, fontweight='bold', color=COLOR['ensemble'])
    ax.axhline(0, color='#555555', lw=1)
    ax.set_ylabel('CRPS improvement over GEOS-FP (%)')
    ax.set_title('Skill: better than the input', loc='left')
    ax = axes[1]
    ratio = [spread_skill(mean[n]['ensemble'], members) for n in names]
    ax.bar([SHORT[n] for n in names], ratio, color=COLOR['members'], width=.6)
    for k, v in enumerate(ratio):
        ax.text(k, v+.02, f'{v:.2f}', ha='center', va='bottom', fontsize=14)
    ax.axhline(1, color='#555555', lw=1.2, ls='--')
    ax.set_ylim(0, max(1.3, max(ratio)+.15))
    ax.set_ylabel('Spread / error')
    ax.set_title('Calibration: 1 = honest spread', loc='left')
    ax = axes[2]
    s = spectra['precip']
    keep = s['freq'] > 0
    wavelength = dx_km/s['freq'][keep]
    for key, label, lw in (('truth', 'Truth 2 km', 3), ('coarse', 'GEOS-FP (regridded)', 2.2),
                           ('members', 'Member', 2.4), ('ensemble_mean', 'Ensemble mean', 2.2)):
        ax.plot(wavelength, s[key][keep], color=COLOR[key], lw=lw, label=label,
                ls='--' if key == 'ensemble_mean' else '-')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.invert_xaxis()
    ax.set_xlabel('Wavelength (km)')
    ax.set_ylabel('Rain power')
    ax.set_title('Detail: members keep truth-like small scales', loc='left')
    ax.legend()
    fig.suptitle(f'v4.1 at a glance · {tag}')
    save(fig, path, dpi, pdf, plt)


def mean_spectra(diagnostics):
    out = {}
    for n in FIELDS:
        entries = [d['spectra'][n] for d in diagnostics]
        out[n] = {k: np.mean([e[k] for e in entries], axis=0) for k in entries[0]}
    return out


def plot_spectra(spectra, dx_km, path, plt, dpi, pdf, ratio=False):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10.5), constrained_layout=True)
    lines = (('truth', 'Truth 2 km', 2.8, '-'), ('coarse', 'GEOS-FP (regridded)', 2.0, '-'),
             ('regression', 'Regression mean', 1.8, ':'), ('members', 'Member', 2.2, '-'),
             ('ensemble_mean', 'Ensemble mean', 2.0, '--'))
    for ax, n in zip(axes.flat, FIELDS):
        s = spectra[n]
        keep = s['freq'] > 0
        wavelength = dx_km/s['freq'][keep]
        for key, label, lw, ls in lines:
            if ratio and key == 'truth':
                ax.axhline(1, color=COLOR['truth'], lw=lw, label='Truth')
                continue
            values = s[key][keep]/(s['truth'][keep] if ratio else 1)
            ax.plot(wavelength, values, color=COLOR.get(key, '#555555'), lw=lw, ls=ls, label=label)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.invert_xaxis()
        ax.set_xlabel('Wavelength (km)')
        ax.set_ylabel('Power / truth' if ratio else 'Power')
        ax.set_title(LABEL[n], loc='left')
        ax.grid(True, which='major', ls='--', alpha=.35)
        if ratio:
            ax.set_ylim(1e-3, 3)
    axes.flat[0].legend(loc='lower left')
    fig.suptitle('Power spectra relative to truth (1 = truth-like at that scale)' if ratio else
                 'Radial power spectra: where each source loses variance')
    save(fig, path, dpi, pdf, plt)


def plot_ranks(diagnostics, members, path, plt, dpi, pdf):
    ranks = np.sum([d['ranks'] for d in diagnostics], axis=0)
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)
    for ax, n in zip(axes.flat, FIELDS):
        r = ranks[TARGETS.index(n)]
        r = r/max(r.sum(), 1e-30)
        ax.bar(np.arange(len(r)), r, color=COLOR['ensemble'], width=.8)
        ax.axhline(1/len(r), color='#555555', lw=1.2, ls='--')
        ax.set_title(LABEL[n], loc='left')
        ax.set_xlabel('Rank of truth among members')
        ax.set_ylabel('Frequency')
    fig.suptitle(f'Rank histograms ({members} members): flat = reliable, U = under-dispersed, ∩ = over-dispersed')
    save(fig, path, dpi, pdf, plt)


def plot_precip(reports, diagnostics, dx_km, members, path, plt, dpi, pdf):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
    scales_km = np.array(FSS_SCALES)*dx_km
    ax = axes[0, 0]
    for thr, ls in (('1.0', '-'), ('5.0', '--')):
        for key, color, label in (('ensemble_mean', COLOR['ensemble'], 'Ensemble mean'),
                                  ('members', COLOR['members'], 'Members'),
                                  ('coarse', COLOR['coarse'], 'GEOS-FP')):
            values = np.nanmean(np.array([[np.nan if v is None else v for v in r['fss'][thr][key]]
                                          for r in reports], dtype='float64'), axis=0)
            ax.plot(scales_km, values, color=color, ls=ls, lw=2.4, marker='o', ms=5,
                    label=f'{label}, ≥{float(thr):g} mm/h')
    ax.set_xscale('log')
    ax.set_ylim(0, 1)
    ax.set_xlabel('Neighbourhood size (km)')
    ax.set_ylabel('Fractions skill score')
    ax.set_title('Rain: FSS vs scale', loc='left')
    ax.legend(fontsize=11)
    ax = axes[0, 1]
    centers = np.sqrt(np.maximum(RAIN_EDGES[:-1], 1e-2)*RAIN_EDGES[1:])
    for key, label in (('truth', 'Truth 2 km'), ('coarse', 'GEOS-FP'), ('regression', 'Regression mean'),
                       ('ensemble', 'Members')):
        values = np.mean([d['hist'][key] for d in diagnostics], axis=0)
        ax.plot(centers[1:], values[1:], color=COLOR.get(key if key != 'ensemble' else 'members'), lw=2.4,
                marker='o', ms=4, label=label)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Rain rate (mm/h)')
    ax.set_ylabel('Area fraction')
    ax.set_title('Rain-rate distribution', loc='left')
    ax.legend()
    ax = axes[1, 0]
    probs = np.linspace(.5, .99999, 400)
    truth = np.concatenate([d['qq']['truth'] for d in diagnostics])
    tq = np.quantile(truth, probs)
    for key, label in (('coarse', 'GEOS-FP'), ('regression', 'Regression mean'), ('members', 'Members')):
        values = np.quantile(np.concatenate([d['qq'][key] for d in diagnostics]), probs)
        ax.plot(tq, values, color=COLOR.get(key), lw=2.4, label=label)
    top = float(max(tq.max(), 1))
    ax.plot([0, top], [0, top], color=COLOR['truth'], lw=1.2, ls='--', label='1:1')
    ax.set_xlabel('Truth quantile (mm/h)')
    ax.set_ylabel('Predicted quantile (mm/h)')
    ax.set_title('Rain Q-Q (50th to 99.999th percentile)', loc='left')
    ax.legend()
    ax = axes[1, 1]
    rel = diagnostics[0]['reliability']['1.0']
    area = np.sum([d['reliability']['1.0']['area'] for d in diagnostics], axis=0)
    observed = np.sum([d['reliability']['1.0']['observed'] for d in diagnostics], axis=0)
    p = np.arange(len(rel['area']))/(len(rel['area'])-1)
    ok = area > 0
    ax.plot([0, 1], [0, 1], color=COLOR['truth'], lw=1.2, ls='--', label='Perfect')
    ax.plot(p[ok], observed[ok]/area[ok], color=COLOR['ensemble'], lw=2.6, marker='o', ms=7, label='Ensemble')
    ax.set_xlabel(f'Forecast probability ({members} members)')
    ax.set_ylabel('Observed frequency')
    ax.set_title('Rain ≥ 1 mm/h reliability', loc='left')
    ax.legend()
    fig.suptitle('Precipitation diagnostics (all cases)')
    save(fig, path, dpi, pdf, plt)


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------

def _clean(value):
    """JSON-safe copy: NaN/inf -> null."""
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def run(cfg, checkpoint='latest', output=None, split='test', samples=2, wettest=1, timestamps=None, members=8,
        steps=64, zooms=2, zoom_size=320, post='spectral', seed=317, batch=32, threads=8, dpi=300, pdf=True,
        use_cartopy=True, map_features=True, weights='ema', animate=('precip', 't2m', 'wind_speed'), log=print):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    style(plt)
    if members < 2:
        raise ValueError('Use at least two members')
    started = time.monotonic()
    path = resolve_checkpoint(cfg, checkpoint)
    device = device_for(cfg['train']['device'])
    archive, model, conditioner, saved = load_model(cfg, path, device, weights)
    job = os.environ.get('SLURM_JOB_ID') or time.strftime('%Y%m%d_%H%M%S')
    out = Path(output) if output else (Path(cfg['train']['output'])/'evaluation'/f'present_{path.stem}_{split}_{job}')
    out.mkdir(parents=True, exist_ok=True)
    cases = select_cases(archive, split, timestamps, samples, wettest, seed, log)
    log(f'{path.name} (epoch {saved["epoch"]+1}); {members} members × {steps} steps; post {post}; cases: '
        + ', '.join(f'{c["entry"]["id"]} ({c["reason"]})' for c in cases))
    canvas = Canvas(archive, use_cartopy, map_features, log)
    area = np.asarray(archive.static['area'], dtype='float64')
    dx_km = float(np.sqrt(np.median(area))/1000)
    h, w = archive.shape
    clim = None
    if post == 'spectral':
        from .explore_inference_v4_1 import climatology, spectral_fix
        clim = climatology(archive, cases[0]['entry'], (0, h, 0, w), count=12, log=log)
    rng = np.random.default_rng(seed)
    reports, diagnostics, case_rows = [], [], []
    for number, case in enumerate(cases, 1):
        entry = case['entry']
        t0 = time.monotonic()
        sampler = DomainSampler(model, conditioner, archive, entry, cfg, device, batch, threads)
        ensemble = []
        for m in range(members):
            member = sampler.sample(member_seed(cfg, entry, m), steps)
            if clim is not None:
                member = spectral_fix(member, clim, dx_km, max_gain=1.3)
            ensemble.append(member)
            log(f'  [{number}/{len(cases)}] {entry["id"]}: member {m+1}/{members} ({time.monotonic()-t0:.0f}s)')
        ensemble = np.stack(ensemble)
        truth = np.asarray(archive.physical_truth(entry), dtype='float32')
        coarse, regression = sampler.coarse, sampler.regression
        del sampler
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        report, diag = case_metrics(ensemble, truth, coarse, regression, area, rng)
        native, _ = native_fields(entry, canvas.lat, canvas.lon, log=log)
        item = dict(truth=truth, coarse=coarse, regression=regression, ensemble=ensemble, area=area, native=native)
        windows = [(event_window(truth[1], zoom_size), 'A')]
        windows += [(wnd, chr(ord('B')+k)) for k, wnd in enumerate(random_windows(
            archive.shape, zoom_size, max(0, zooms-1), np.random.SeedSequence([seed, number])))]
        folder = out/'cases'/entry['id']
        folder.mkdir(parents=True, exist_ok=True)
        when = entry['time'][:16].replace('T', ' ')
        for name in FIELDS:
            unit = DISPLAY[name][1]
            map_figure(canvas, item, name, None, f'{LABEL[name]} ({unit}) · {when} UTC', folder/f'conus_{name}',
                       plt, dpi, pdf, boxes=windows)
            for wnd, tag in windows:
                cy, cx = (wnd[0].start+wnd[0].stop)//2, (wnd[1].start+wnd[1].stop)//2
                where = f'{float(canvas.lat[cy, cx]):.1f}°N {abs(float(canvas.lon[cy, cx])):.1f}°W'
                map_figure(canvas, item, name, wnd, f'{LABEL[name]} ({unit}) · {when} UTC · zoom {tag}, {where}',
                           folder/f'zoom{tag}_{name}', plt, dpi, pdf)
            uncertainty_figure(canvas, item, name, f'{LABEL[name]} · where the model is unsure vs where it is wrong · '
                               f'{when} UTC', folder/f'uncertainty_{name}', plt, dpi, pdf)
            members_figure(canvas, item, name, windows[0][0], f'{LABEL[name]} ({unit}) · {when} UTC · '
                           f'zoom A: truth and individual members', folder/f'diversity_{name}', plt, dpi, pdf)
        for name in animate:
            members_animation(canvas, item, name, windows[0][0], f'{LABEL.get(name, "10 m wind speed")} · {when} UTC · '
                              f'one input, many plausible 2 km outcomes', folder, plt, dpi)
        rows = scorecard_rows({n: report[n] for n in FIELDS}, members)
        write_csv(rows, folder/'scorecard.csv')
        plot_scorecard(rows, folder/'scorecard', plt, dpi, pdf, f'Scores · {when} UTC · {members} members')
        for r in rows:
            case_rows.append(dict(case=entry['id'], **r))
        reports.append(report)
        diagnostics.append(diag)
        log(f'[{number}/{len(cases)}] {entry["id"]} done in {time.monotonic()-t0:.0f}s → {folder}')
        del ensemble, item

    mean = average_reports(reports)
    tag = f'{len(reports)} case{"s" if len(reports) > 1 else ""} · {members} members · epoch {saved["epoch"]+1}'
    plot_relative(mean, 'crps', 'CRPS relative to GEOS-FP', f'Probabilistic skill vs GEOS-FP ({tag}); '
                  'labels: ensemble improvement', out/'skill_crps', plt, dpi, pdf)
    plot_relative(mean, 'rmse', 'RMSE relative to GEOS-FP', f'Error of the best estimate vs GEOS-FP ({tag})',
                  out/'skill_rmse', plt, dpi, pdf)
    plot_spread_skill(mean, members, out/'spread_skill', plt, dpi, pdf)
    rows = scorecard_rows(mean, members)
    write_csv(rows, out/'scorecard.csv')
    write_csv(case_rows, out/'scorecard_cases.csv')
    plot_scorecard(rows, out/'scorecard', plt, dpi, pdf, f'Scores averaged over {tag}')
    spectra = mean_spectra(diagnostics)
    plot_spectra(spectra, dx_km, out/'spectra', plt, dpi, pdf)
    plot_spectra(spectra, dx_km, out/'spectra_ratio', plt, dpi, pdf, ratio=True)
    plot_ranks(diagnostics, members, out/'rank_histograms', plt, dpi, pdf)
    plot_overview(mean, spectra, members, dx_km, tag, out/'overview', plt, dpi, pdf)
    plot_precip(reports, diagnostics, dx_km, members, out/'precip_diagnostics', plt, dpi, pdf)
    write_json(out/'metrics.json', _clean(json.loads(json.dumps(dict(
        checkpoint=str(path), epoch=saved['epoch']+1, weights=weights, split=split, members=members, steps=steps,
        post=post, cases=[dict(id=c['entry']['id'], time=c['entry']['time'], reason=c['reason']) for c in cases],
        mean=mean, scorecard=rows, seconds=time.monotonic()-started), default=float))))
    log(f'Presentation figures written to {out} ({time.monotonic()-started:.0f}s)')
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', default='configs/discover_v4_1.yaml')
    parser.add_argument('--checkpoint', default='latest', help='latest (default) | best | <epoch> | <path>')
    parser.add_argument('--weights', choices=('ema', 'raw'), default='ema')
    parser.add_argument('--split', choices=('val', 'test'), default='test')
    parser.add_argument('--samples', type=int, default=2, help='Random hours of the split (default 2)')
    parser.add_argument('--wettest', type=int, default=1, help='Wettest hours added (default 1)')
    parser.add_argument('--timestamps', nargs='+', help='Specific hours instead of random/wettest')
    parser.add_argument('--members', type=int, default=8)
    parser.add_argument('--steps', type=int, default=64)
    parser.add_argument('--zooms', type=int, default=2, help='Zoom windows per case: the rain event + random ones')
    parser.add_argument('--zoom-size', type=int, default=320, help='Zoom window in px (default 320 ≈ 600 km)')
    parser.add_argument('--post', choices=('spectral', 'none'), default='spectral')
    parser.add_argument('--animate', nargs='*', default=['precip', 't2m', 'wind_speed'],
                        choices=['precip', 't2m', 'ps', 'u10m', 'v10m', 'q2m', 'wind_speed'],
                        help='Variables for reveal_*.png + members_*.gif (default precip t2m wind_speed; none: skip)')
    parser.add_argument('--seed', type=int, default=317)
    parser.add_argument('--dpi', type=int, default=300)
    parser.add_argument('--no-pdf', action='store_true')
    parser.add_argument('--no-cartopy', action='store_true')
    parser.add_argument('--no-map-features', action='store_true')
    parser.add_argument('--cartopy-data-dir')
    parser.add_argument('--output')
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--threads', type=int, default=8)
    a = parser.parse_args()
    if a.cartopy_data_dir:
        import cartopy
        cartopy.config['pre_existing_data_dir'] = a.cartopy_data_dir
    run(load_config(a.config), a.checkpoint, a.output, a.split, a.samples, a.wettest, a.timestamps, a.members,
        a.steps, a.zooms, a.zoom_size, a.post, a.seed, a.batch, a.threads, a.dpi, not a.no_pdf, not a.no_cartopy,
        not a.no_map_features, a.weights, tuple(a.animate), log=lambda message: print(message, flush=True))


if __name__ == '__main__':
    main()
