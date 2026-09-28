"""Inference-time sharpening ablation for v4.1 (no retraining).

Every method runs on the same case(s), from the same member noise, through the
same synchronized tiled Heun sampler as ``evaluate_v4_1`` / ``cli_v4_1 predict``.

Methods (ids for ``--methods``; knob values come from the CLI):

  baseline        v4 inference as trained: uniform Heun, Hann blending
  more_steps      2x Heun steps. Control: if this barely changes the spectrum,
                  discretization is not what limits sharpness, and schedule
                  tweaks (time_warp) cannot help much either
  time_warp       ODE times t = 1-(1-tau)^gamma, shorter steps near the data end
  churn           EDM-style stochastic sampling: before each Heun step in
                  --churn-range, re-noise from sigma to sigma*(1+churn) along the
                  model's own path, then integrate. Re-injected noise is resolved
                  by the model again, restoring small-scale variance an
                  imperfect deterministic velocity field loses
  autoguide       autoguidance (Karras et al. 2024): v = v_weak + w (v - v_weak)
                  with an earlier kept checkpoint of this run as the weak model.
                  Extrapolates away from the less-trained model's (blurrier)
                  prediction; costs one extra forward pass per evaluation
  residual_scale  amplify each field's departure from the frozen regression;
                  rain is scaled in sqrt1p space around the regression rain (the
                  old version scaled rain z itself: a domain-wide wet bias)
  wet_cutoff      set rain below --dry-cutoff mm/h to 0 (drizzle cleanup: changes
                  wet area, not sharpness)
  tukey_window    flat-top Tukey tile blending (ablation; see make_window)
  combined        knobs of the methods listed in --combine, applied together

What counts as "better": sharper is not automatically better. A method helps if
it moves the rain power spectrum toward truth at fine scales WITHOUT worsening
the ensemble CRPS, the large-scale (~25 km block-mean) RMSE or the domain-mean
rain bias, and without worsening the CRPS of the other fields (every method
acts on all six outputs; each field gets the same scores and spectra). Pixel MAE/RMSE of single members always punish sharper fields
(double penalty) and are reported only for completeness.

Outputs (default <train.output>/evaluation/sharpness_<ckpt>_<split>_<sha>_<job>/):
  cases/<id>/maps/<var>_conus.{png,pdf}    per variable (precip, t2m, ps, u10m, v10m, q2m,
                                           wind_speed): truth, coarse, member 1 of each method
  cases/<id>/maps/<var>_zoom.{png,pdf}     the same around the strongest rain feature
  cases/<id>/sharpness_all_fields_zoom.png every field (rain, t2m, ps, u10m, v10m, q2m, wind speed)
                                           × truth, coarse and each method, in the rain-event zoom
  cases/<id>/sharpness_spectra.png         rain PSD and PSD ratio to truth vs wavelength
  cases/<id>/sharpness_spectra_all_fields.png  PSD ratio to truth for every field
  cases/<id>/metrics.json                  all scores for the case
  summary_tradeoff.png                     rain fine-scale power vs CRPS / large-scale RMSE / bias
  summary_scorecard.png                    methods × fields: CRPS change vs baseline, fine-scale power
Every figure is written as PNG and PDF at --dpi (default 300; --no-pdf for PNG only).
  summary_metrics.json, report.md          case-mean scores, verdicts, run settings
"""
import argparse
from copy import deepcopy
import gc
import json
import os
from pathlib import Path
import time
import numpy as np
import torch

from .config import write_json
from .metrics import weighted_mean, radial_psd, crps_ensemble, fss
from .train import device_for
from .train_v2 import file_hash_v2
from .v4 import TARGETS
from .v4_1 import load_config
from .evaluate_v4_1 import (resolve_checkpoint, load_model, load_guide, DomainSampler, select_cases,
                            member_seed, event_window, Canvas, native_fields, _rain_norm, RAIN_LEVELS,
                            _stamp, _display, _norm, _coarse_panel, DISPLAY)

KNOBS = dict(steps_factor=1, time_warp_gamma=1.0, churn=0.0, guide_weight=1.0,
             residual_scale=1.0, dry_cutoff=0.0, window_type='hann')
METHOD_IDS = ('baseline', 'more_steps', 'time_warp', 'churn', 'autoguide', 'residual_scale',
              'wet_cutoff', 'tukey_window', 'combined')
TRAJECTORY_KNOBS = ('steps_factor', 'time_warp_gamma', 'churn', 'guide_weight', 'window_type')
COLORS = ['#2166ac', '#8c8c8c', '#74add1', '#1a9850', '#7b3294', '#f46d43', '#fdae61', '#abd9e9',
          '#d73027', '#01665e', '#c51b7d']
STATE_FIELDS = ('t2m', 'ps', 'u10m', 'v10m', 'q2m', 'wind_speed')
FIELDS = ('precip',)+STATE_FIELDS
STATE_CRPS_TOLERANCE = 2.   # % CRPS worsening per non-rain field that blocks a 'recommended' verdict
FINE_KM, MESO_KM = (0., 30.), (30., 150.)   # wavelength bands for power ratios
LARGE_SCALE_KM = 25.                        # ~GEOS-FP cell: block size for large-scale RMSE
FSS_RAIN = ((1., 25.), (5., 25.), (1., 100.))  # (mm/h threshold, neighbourhood km)


def build_methods(names, combine, steps, warp_gamma, churn, guide_weight, residual_scale,
                  dry_cutoff, tukey_alpha, steps_factor=2):
    """Method specs (knobs + labels) for the requested ids, in order."""
    single = {
        'baseline': dict(label='Baseline (v4 inference)', desc=f'uniform Heun {steps} steps, Hann'),
        'more_steps': dict(label=f'{steps_factor}× steps', desc=f'uniform Heun {steps*steps_factor} steps',
                           steps_factor=steps_factor),
        'time_warp': dict(label=f'Time warp γ={warp_gamma:g}', desc='t = 1-(1-τ)^γ',
                          time_warp_gamma=warp_gamma),
        'churn': dict(label=f'Churn {churn:g}', desc='EDM stochastic re-noising between Heun steps',
                      churn=churn),
        'autoguide': dict(label=f'Autoguidance w={guide_weight:g}', desc='v_weak + w (v - v_weak)',
                          guide_weight=guide_weight),
        'residual_scale': dict(label=f'Residual ×{residual_scale:g}',
                               desc='departure from frozen regression (rain in sqrt space)',
                               residual_scale=residual_scale),
        'wet_cutoff': dict(label=f'Cutoff <{dry_cutoff:g} mm/h', desc='drizzle set to 0', dry_cutoff=dry_cutoff),
        'tukey_window': dict(label=f'Tukey window α={tukey_alpha:g}', desc='flat-top tile blending',
                             window_type='tukey'),
    }
    methods = []
    for name in names:
        if name == 'combined':
            parts = [p for p in combine if p in single and p != 'baseline']
            if not parts:
                continue
            spec = dict(label='Combined', desc=' + '.join(single[p]['label'] for p in parts))
            for p in parts:
                spec.update({k: v for k, v in single[p].items() if k in KNOBS})
            spec['components'] = parts
        elif name in single:
            spec = deepcopy(single[name])
        else:
            raise ValueError(f'Unknown method {name!r}; choose from {", ".join(METHOD_IDS)}')
        methods.append(dict(KNOBS, id=name, **spec))
    for i, m in enumerate(methods):
        m['color'] = COLORS[i % len(COLORS)]
        m['tukey_alpha'] = tukey_alpha
    return methods


def trajectory_key(method):
    return tuple(method[k] for k in TRAJECTORY_KNOBS)


def generate(sampler, methods, seeds, steps, churn_range, log=print):
    """{method id: (members, 6, h, w)} with one ODE solve per distinct trajectory
    per member; post-processing variants reuse that solve."""
    ensembles = {m['id']: [] for m in methods}
    seconds = {m['id']: 0. for m in methods}
    for number, seed in enumerate(seeds, 1):
        cores = {}
        for m in methods:
            key = trajectory_key(m)
            sampler.set_window(m['window_type'], m['tukey_alpha'])
            t0 = time.monotonic()
            if key not in cores:
                cores[key] = sampler.integrate(seed, steps*m['steps_factor'], m['time_warp_gamma'],
                                               m['churn'], churn_range, m['guide_weight'])
            ensembles[m['id']].append(sampler.decode(cores[key], m['residual_scale'], m['dry_cutoff']))
            seconds[m['id']] += time.monotonic()-t0
        log(f'  member {number}/{len(seeds)}: {len(cores)} ODE solves for {len(methods)} methods')
    sampler.set_window('hann')
    return {k: np.stack(v) for k, v in ensembles.items()}, seconds


# ----------------------------------------------------------------------------
# Scores
# ----------------------------------------------------------------------------

def mean_psd(fields):
    fields = np.asarray(fields)
    fields = fields[None] if fields.ndim == 2 else fields
    freq, psd = radial_psd(fields[0])
    for f in fields[1:]:
        psd = psd+radial_psd(f)[1]
    return freq, psd/len(fields)


def band_ratio(freq, psd, truth_psd, dx_km, band):
    """Geometric-mean power ratio to truth over a wavelength band (km)."""
    wavelength = dx_km/np.maximum(freq, 1e-12)
    keep = (freq > 0) & (wavelength >= band[0]) & (wavelength < band[1]) & (truth_psd > 0)
    if not keep.any():
        return None
    return float(np.exp(np.mean(np.log(np.maximum(psd[keep], 1e-30)/truth_psd[keep]))))


def block_mean(field, k):
    h, w = field.shape[-2:]
    h, w = h//k*k, w//k*k
    x = field[..., :h, :w]
    return x.reshape(*x.shape[:-2], h//k, k, w//k, k).mean((-3, -1))


def odd_pixels(km, dx_km):
    n = max(1, int(round(km/dx_km)))
    return n if n % 2 else n+1


def gradient_mean(field, area):
    gy, gx = np.gradient(np.asarray(field, dtype='float64'))
    return weighted_mean(np.hypot(gy, gx), area)


def score_rain(ensemble, truth, area, dx_km, truth_psd):
    """Rain scores of an ensemble (members, h, w) against truth (h, w)."""
    k = max(1, int(round(LARGE_SCALE_KM/dx_km)))
    area_k = block_mean(area, k)
    truth_mean = weighted_mean(truth, area)
    means = [weighted_mean(m, area) for m in ensemble]
    freq, psd = mean_psd(ensemble)
    truth_grad = gradient_mean(truth, area)
    s = dict(
        mean_rain=float(np.mean(means)),
        bias_pct=float(100*(np.mean(means)-truth_mean)/truth_mean) if truth_mean > 0 else None,
        wet_frac_0p1=float(np.mean([weighted_mean(m >= .1, area) for m in ensemble])),
        wet_frac_1=float(np.mean([weighted_mean(m >= 1, area) for m in ensemble])),
        p99=float(np.mean([np.quantile(m, .99) for m in ensemble])),
        p99_9=float(np.mean([np.quantile(m, .999) for m in ensemble])),
        peak=float(np.mean([m.max() for m in ensemble])),
        fine_power_ratio=band_ratio(freq, psd, truth_psd, dx_km, FINE_KM),
        meso_power_ratio=band_ratio(freq, psd, truth_psd, dx_km, MESO_KM),
        gradient_ratio=float(np.mean([gradient_mean(m, area) for m in ensemble])/truth_grad) if truth_grad > 0 else None,
        member_mae=float(np.mean([weighted_mean(abs(m-truth), area) for m in ensemble])),
        member_rmse=float(np.mean([np.sqrt(weighted_mean((m-truth)**2, area)) for m in ensemble])),
        crps=float(weighted_mean(crps_ensemble(ensemble, truth), area)),
        large_scale_rmse=float(np.mean([np.sqrt(weighted_mean((block_mean(m, k)-block_mean(truth, k))**2, area_k))
                                        for m in ensemble])),
    )
    for threshold, km in FSS_RAIN:
        values = [fss(m, truth, threshold, odd_pixels(km, dx_km), area) for m in ensemble]
        values = [v for v in values if v is not None]
        s[f'fss_{threshold:g}mm_{km:g}km'] = float(np.mean(values)) if values else None
    return s, freq, psd


def field_of(stack, name):
    """One field from (..., 6, h, w); wind_speed is derived from u10m/v10m."""
    if name == 'wind_speed':
        return np.hypot(stack[..., TARGETS.index('u10m'), :, :], stack[..., TARGETS.index('v10m'), :, :])
    return stack[..., TARGETS.index(name), :, :]


def score_states(ensemble, truth, area, dx_km):
    """Scores for every non-rain field of (members, 6, h, w) vs truth (6, h, w),
    plus the member-mean power spectrum of each field."""
    k = max(1, int(round(LARGE_SCALE_KM/dx_km)))
    area_k = block_mean(area, k)
    scores, spectra = {}, {}
    for name in STATE_FIELDS:
        ens, t = field_of(ensemble, name), field_of(truth, name)
        freq, truth_psd = radial_psd(t)
        _, psd = mean_psd(ens)
        truth_grad = gradient_mean(t, area)
        tk = block_mean(t, k)
        scores[name] = dict(
            bias=float(np.mean([weighted_mean(m, area) for m in ens])-weighted_mean(t, area)),
            member_rmse=float(np.mean([np.sqrt(weighted_mean((m-t)**2, area)) for m in ens])),
            crps=float(weighted_mean(crps_ensemble(ens, t), area)),
            large_scale_rmse=float(np.mean([np.sqrt(weighted_mean((block_mean(m, k)-tk)**2, area_k)) for m in ens])),
            fine_power_ratio=band_ratio(freq, psd, truth_psd, dx_km, FINE_KM),
            meso_power_ratio=band_ratio(freq, psd, truth_psd, dx_km, MESO_KM),
            gradient_ratio=float(np.mean([gradient_mean(m, area) for m in ens])/truth_grad) if truth_grad > 0 else None)
        spectra[name] = psd
    return scores, spectra


def evaluate_case(sampler, archive, entry, methods, seeds, steps, churn_range, dx_km, log=print):
    truth = np.asarray(archive.physical_truth(entry), dtype='float32')
    coarse = sampler.coarse
    area = np.asarray(archive.static['area'], dtype='float64')
    ensembles, seconds = generate(sampler, methods, seeds, steps, churn_range, log)
    t_freq, t_psd = radial_psd(truth[1])
    references = {}
    for name, field in (('truth', truth), ('coarse', coarse), ('regression', sampler.regression)):
        s, _, psd = score_rain(field[None, 1], truth[1], area, dx_km, t_psd)
        states, spectra = score_states(field[None], truth, area, dx_km)
        references[name] = dict(scores=s, psd=psd, rain=field[1], states=states, spectra=spectra, fields=field)
    results = {}
    for m in methods:
        ens = ensembles[m['id']]
        s, _, psd = score_rain(ens[:, 1], truth[1], area, dx_km, t_psd)
        s['seconds_per_member'] = round(seconds[m['id']]/len(seeds), 1)
        states, spectra = score_states(ens, truth, area, dx_km)
        results[m['id']] = dict(scores=s, states=states, spectra=spectra, psd=psd,
                                rain=ens[0, 1], fields=ens[0], method=m)
        log(f'  {m["label"]:<28} fine-power {_fmt(s["fine_power_ratio"])} · CRPS {s["crps"]:.4f} · '
            f'LS-RMSE {s["large_scale_rmse"]:.4f} · bias {_fmt(s["bias_pct"], "+.1f")}% · '
            f'p99.9 {s["p99_9"]:.2f} (truth {references["truth"]["scores"]["p99_9"]:.2f})')
    return results, references, t_freq


def _fmt(value, spec='.3f'):
    return 'n/a' if value is None else format(value, spec)


# ----------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------

def _grid(n, cols=4):
    return (n+cols-1)//cols, cols


FIGURES = dict(dpi=300, pdf=True)   # set by compare_sharpness(dpi=, pdf=)


def save_figure(fig, path, plt):
    """PNG at FIGURES['dpi'], plus a PDF whose map rasters use the same dpi
    (full-CONUS panels are ~native resolution at 300 dpi, so zooming stays sharp)."""
    path = Path(path)
    fig.savefig(path.with_suffix('.png'), dpi=FIGURES['dpi'])
    if FIGURES['pdf']:
        fig.savefig(path.with_suffix('.pdf'), dpi=FIGURES['dpi'])
    fig.clf()
    plt.close(fig)
    gc.collect()  # large Cartopy/Agg rasters; free them before the next map


def plot_maps(canvas, results, references, native, name, heading, path, plt, window=None):
    """One variable: truth, coarse (native GEOS-FP cells when available) and member 1
    of every method, over full CONUS or a zoom window."""
    title, unit, *_, cmap, _ = DISPLAY[name]
    panels = _field_panels(results, references)
    shown = [field_of(stack, name) for _, stack in panels]
    ys, xs = window if window is not None else (slice(None), slice(None))
    if name == 'precip':
        cmap, norm = _rain_norm()
    else:
        shown = [_display(name, f) for f in shown]
        norm = _norm(name, np.concatenate([shown[0][ys, xs][::2, ::2].ravel(), shown[1][ys, xs][::2, ::2].ravel()]))
    item = dict(native=native or {})
    rows, cols = _grid(len(panels))
    fig = plt.figure(figsize=(6.2*cols, (4.6 if window is None else 5.6)*rows), constrained_layout=True)
    from matplotlib.gridspec import GridSpec
    grid = GridSpec(rows, cols, figure=fig)
    image, axes = None, []
    for k, ((label, _), field) in enumerate(zip(panels, shown)):
        r, c = divmod(k, cols)
        ax = canvas.axes(fig, grid[r, c])
        axes.append(ax)
        left, bottom = c == 0, r == rows-1 or k+cols >= len(panels)
        if k == 1:
            shown_image, label, *_ = _coarse_panel(canvas, ax, item, name, field, window, cmap, norm, left, bottom)
        else:
            shown_image = canvas.show(ax, field, window, cmap=cmap, norm=norm, left=left, bottom=bottom)
            label = label if k == 0 else f'{label} · member 1'
        image = shown_image if shown_image is not None else image
        ax.set_title(label, fontsize=10.5, fontweight='bold')
        sub = field[ys, xs]
        if name == 'precip':
            if k == 0:
                stats = references['truth']['scores']
            elif k == 1:
                stats = references['coarse']['scores']
            else:
                stats = list(results.values())[k-2]['scores']
            if window is None:
                text = (f'mean {stats["mean_rain"]:.3f} · p99.9 {stats["p99_9"]:.1f} · max {stats["peak"]:.0f} mm/h · '
                        f'wet {stats["wet_frac_0p1"]:.0%}')
                if k:
                    text += f'\nfine-power ×{_fmt(stats["fine_power_ratio"], ".2f")} · CRPS {stats["crps"]:.3f}'
            else:
                text = f'local mean {float(sub.mean()):.2f} · max {float(sub.max()):.1f} mm/h'
        else:
            gy, gx = np.gradient(np.asarray(sub, dtype='float64'))
            text = f'mean {float(sub.mean()):.4g} · |∇| {float(np.hypot(gy, gx).mean()):.3g} {unit}/px'
            if k >= 2 and window is None:
                st = list(results.values())[k-2]['states'][name]
                text += f'\nfine-power ×{_fmt(st["fine_power_ratio"], ".2f")} · CRPS {st["crps"]:.4g}'
        _stamp(ax, text)
    fig.colorbar(image, ax=axes, location='bottom', shrink=.45, pad=.01, extend='both',
                 ticks=RAIN_LEVELS if name == 'precip' else None, label=f'{title} ({unit})')
    where = 'Full CONUS' if window is None else f'Rain-event zoom ({ys.stop-ys.start}×{xs.stop-xs.start} px)'
    fig.suptitle(f'{heading}\n{title} · {where}: inference sharpening ablation (same noise for every method)',
                 fontsize=13, fontweight='bold')
    save_figure(fig, path, plt)


def plot_spectra(results, references, t_freq, dx_km, heading, path, plt):
    fig, (ax, ratio_ax) = plt.subplots(1, 2, figsize=(18, 7), constrained_layout=True)
    valid = t_freq > 0
    wavelength = dx_km/t_freq[valid]
    t_psd = references['truth']['psd'][valid]
    ax.loglog(wavelength, np.maximum(t_psd, 1e-30), color='k', lw=2.6, label='Ground truth', zorder=10)
    for name, style in (('coarse', dict(color='#9e9e9e', ls=':')), ('regression', dict(color='#9e9e9e', ls='--'))):
        psd = references[name]['psd'][valid]
        ax.loglog(wavelength, np.maximum(psd, 1e-30), lw=1.4, label=name.capitalize(), **style)
        ratio_ax.semilogx(wavelength, psd/np.maximum(t_psd, 1e-30), lw=1.4, label=name.capitalize(), **style)
    for r in results.values():
        m = r['method']
        width = 2.4 if m['id'] in ('baseline', 'combined') else 1.5
        psd = r['psd'][valid]
        ax.loglog(wavelength, np.maximum(psd, 1e-30), color=m['color'], lw=width, label=m['label'])
        ratio_ax.semilogx(wavelength, psd/np.maximum(t_psd, 1e-30), color=m['color'], lw=width, label=m['label'])
    for a in (ax, ratio_ax):
        a.invert_xaxis()
        a.set_xlabel('Wavelength (km)')
        a.grid(True, which='both', ls='--', alpha=.4)
        for edge in (FINE_KM[1], MESO_KM[1]):
            a.axvline(edge, color='0.6', lw=.8)
    ratio_ax.axhline(1, color='k', lw=1.6)
    ratio_ax.set_ylim(0, 2)
    ax.set_ylabel('Rain PSD (mm h⁻¹)²')
    ratio_ax.set_ylabel('Power ratio to truth (1 = truth-like variance at that scale)')
    ax.set_title('Rain radial power spectrum (member mean)')
    ratio_ax.set_title(f'Ratio to truth · fine < {FINE_KM[1]:g} km < meso < {MESO_KM[1]:g} km')
    ax.legend(fontsize=8.5, loc='lower left')
    fig.suptitle(heading, fontsize=13, fontweight='bold')
    save_figure(fig, path, plt)


def _field_panels(results, references):
    panels = [('Truth', references['truth']['fields']), ('Coarse', references['coarse']['fields'])]
    return panels+[(r['method']['label'], r['fields']) for r in results.values()]


def plot_all_fields_zoom(canvas, results, references, native, window, heading, path, plt):
    """Every field (rows) × truth, coarse and member 1 of each method (columns) in one window."""
    from matplotlib.gridspec import GridSpec
    ys, xs = window
    panels = _field_panels(results, references)
    fig = plt.figure(figsize=(3.3*len(panels)+1.2, 3.1*len(FIELDS)), constrained_layout=True)
    grid = GridSpec(len(FIELDS), len(panels), figure=fig)
    item = dict(native=native or {})
    for r, name in enumerate(FIELDS):
        title, unit, *_, cmap, _ = DISPLAY[name]
        shown = [field_of(stack, name) for _, stack in panels]
        if name == 'precip':
            cmap, norm = _rain_norm()
        else:
            shown = [_display(name, f) for f in shown]
            norm = _norm(name, np.concatenate([shown[0][ys, xs].ravel(), shown[1][ys, xs].ravel()]))
        bottom, row = r == len(FIELDS)-1, []
        for c, ((label, _), field) in enumerate(zip(panels, shown)):
            ax = canvas.axes(fig, grid[r, c])
            row.append(ax)
            if c == 1:
                image, label, *_ = _coarse_panel(canvas, ax, item, name, field, window, cmap, norm, False, bottom)
            else:
                image = canvas.show(ax, field, window, cmap=cmap, norm=norm, left=c == 0, bottom=bottom)
            ax.set_title(f'{label}' if r == 0 or c == 1 else '', fontsize=9, fontweight='bold' if r == 0 else None)
            sub = field[ys, xs]
            if name == 'precip':
                _stamp(ax, f'mean {float(sub.mean()):.2f} · max {float(sub.max()):.1f}')
            else:
                gy, gx = np.gradient(np.asarray(sub, dtype='float64'))
                _stamp(ax, f'mean {float(sub.mean()):.4g} · |∇| {float(np.hypot(gy, gx).mean()):.3g}')
        if image is not None:
            fig.colorbar(image, ax=row, location='right', shrink=.9, pad=.005,
                         label=f'{title} ({unit})', extend='both', ticks=RAIN_LEVELS if name == 'precip' else None)
    fig.suptitle(f'{heading}\nAll fields, rain-event zoom ({ys.stop-ys.start}×{xs.stop-xs.start} px): '
                 'truth, coarse and member 1 of every method (same noise); |∇| = mean gradient per pixel',
                 fontsize=13, fontweight='bold')
    save_figure(fig, path, plt)


def plot_all_spectra(results, references, t_freq, dx_km, heading, path, plt):
    """Power ratio to truth vs wavelength for every field."""
    fig, axes = plt.subplots(2, 4, figsize=(24, 10.5), constrained_layout=True)
    valid = t_freq > 0
    wavelength = dx_km/t_freq[valid]

    def spectrum(entry, name):
        return (entry['psd'] if name == 'precip' else entry['spectra'][name])[valid]

    for ax, name in zip(axes.flat, FIELDS):
        truth = np.maximum(spectrum(references['truth'], name), 1e-30)
        for ref, style in (('coarse', dict(color='#9e9e9e', ls=':')), ('regression', dict(color='#9e9e9e', ls='--'))):
            ax.semilogx(wavelength, spectrum(references[ref], name)/truth, lw=1.3, label=ref.capitalize(), **style)
        for r in results.values():
            m = r['method']
            ax.semilogx(wavelength, spectrum(r, name)/truth, color=m['color'],
                        lw=2.3 if m['id'] in ('baseline', 'combined') else 1.4, label=m['label'])
        ax.axhline(1, color='k', lw=1.5)
        for edge in (FINE_KM[1], MESO_KM[1]):
            ax.axvline(edge, color='0.6', lw=.8)
        ax.invert_xaxis()
        ax.set_ylim(0, 2)
        ax.set_title(f'{DISPLAY[name][0]}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Wavelength (km)')
        ax.set_ylabel('Power ratio to truth')
        ax.grid(True, which='both', ls='--', alpha=.4)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].axis('off')
    axes.flat[-1].legend(handles, labels, loc='center', fontsize=10)
    fig.suptitle(f'{heading}\nPower ratio to truth for every field (member mean; 1 = truth-like variance at that scale)',
                 fontsize=13, fontweight='bold')
    save_figure(fig, path, plt)


def plot_scorecard(summary, methods, path, plt):
    """Methods × fields: CRPS change vs baseline, and fine-scale power ratio to truth."""
    from matplotlib.colors import TwoSlopeNorm, LogNorm
    others = [m for m in methods if m['id'] != 'baseline']
    base = summary['methods'].get('baseline')

    def score(mid, name, key):
        return summary['methods'][mid][key] if name == 'precip' else summary['states'][mid][name][key]

    fig, axes = plt.subplots(1, 2, figsize=(21, 1.1+.62*len(methods)), constrained_layout=True)
    labels = [DISPLAY[n][0] for n in FIELDS]
    if base and others:
        delta = np.array([[100*(score(m['id'], n, 'crps')-score('baseline', n, 'crps'))/score('baseline', n, 'crps')
                           if score('baseline', n, 'crps') else np.nan for n in FIELDS] for m in others])
        bound = max(float(np.nanmax(abs(delta))) if np.isfinite(delta).any() else 1., 1.)
        image = axes[0].imshow(delta, cmap='RdBu_r', norm=TwoSlopeNorm(0, -bound, bound), aspect='auto')
        for (i, j), v in np.ndenumerate(delta):
            axes[0].text(j, i, 'n/a' if np.isnan(v) else f'{v:+.1f}%', ha='center', va='center', fontsize=9)
        axes[0].set_yticks(range(len(others)), [m['label'] for m in others])
        fig.colorbar(image, ax=axes[0], label='CRPS change vs baseline (%) · blue = better')
    else:
        axes[0].axis('off')
    axes[0].set_xticks(range(len(FIELDS)), labels, rotation=20, ha='right')
    axes[0].set_title('Skill: ensemble CRPS change vs baseline', fontweight='bold')
    ratio = np.array([[score(m['id'], n, 'fine_power_ratio') or np.nan for n in FIELDS] for m in methods], dtype=float)
    finite = ratio[np.isfinite(ratio) & (ratio > 0)]
    spread = min(max(float(np.max(abs(np.log10(finite)))) if finite.size else .3, .05), 1.)  # colours cap at ×0.1/×10
    image = axes[1].imshow(np.where(ratio > 0, ratio, np.nan), cmap='PuOr_r', aspect='auto',
                           norm=LogNorm(10**-spread, 10**spread))
    for (i, j), v in np.ndenumerate(ratio):
        axes[1].text(j, i, 'n/a' if np.isnan(v) else f'×{v:.3g}', ha='center', va='center', fontsize=9)
    axes[1].set_yticks(range(len(methods)), [m['label'] for m in methods])
    axes[1].set_xticks(range(len(FIELDS)), labels, rotation=20, ha='right')
    axes[1].set_title(f'Sharpness: power ratio to truth at < {FINE_KM[1]:g} km (×1 = truth)', fontweight='bold')
    fig.colorbar(image, ax=axes[1], label='< 1 too smooth · > 1 too noisy')
    fig.suptitle(f'All-field scorecard, mean over {summary["cases"]} case(s) × {summary["members"]} member(s)',
                 fontsize=13, fontweight='bold')
    save_figure(fig, path, plt)


def plot_tradeoff(summary, methods, path, plt):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5), constrained_layout=True)
    base = summary['methods'].get('baseline')
    specs = [('crps', 'Rain CRPS (mm/h, lower is better)'),
             ('large_scale_rmse', f'Rain RMSE of ~{LARGE_SCALE_KM:g} km block means (lower is better)'),
             ('bias_pct', 'Domain-mean rain bias (%)')]
    for ax, (key, label) in zip(axes, specs):
        for i, m in enumerate(methods):
            s = summary['methods'][m['id']]
            x, y = s.get('fine_power_ratio'), s.get(key)
            if x is None or y is None:
                continue
            ax.scatter(x, y, s=90, color=m['color'], edgecolor='k', zorder=3, label=m['label'])
            ax.annotate(m['label'], (x, y), textcoords='offset points', xytext=(6, 6-10*(i % 3)), fontsize=8)
        ax.axvline(1, color='k', lw=1.2, ls='--')
        if base and base.get(key) is not None and key != 'bias_pct':
            ax.axhline(base[key], color='0.5', lw=1, ls=':')
        if key == 'bias_pct':
            ax.axhline(0, color='0.5', lw=1, ls=':')
        ax.set_xscale('log')
        ax.set_xlabel(f'Fine-scale rain power ratio to truth (< {FINE_KM[1]:g} km; 1 = truth)')
        ax.set_ylabel(label)
        ax.grid(True, which='both', ls='--', alpha=.4)
    axes[0].legend(fontsize=8, loc='best')
    fig.suptitle(f'Sharpness vs skill, mean over {summary["cases"]} case(s) × {summary["members"]} member(s): '
                 'good methods move right toward 1 without rising', fontsize=13, fontweight='bold')
    save_figure(fig, path, plt)


# ----------------------------------------------------------------------------
# Summary and report
# ----------------------------------------------------------------------------

def _mean(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def _gmean(values):
    values = [v for v in values if v is not None and v > 0]
    return float(np.exp(np.mean(np.log(values)))) if values else None


def summarize(cases, methods, members):
    ratio_keys = ('fine_power_ratio', 'meso_power_ratio', 'gradient_ratio')

    def combine(score_dicts):
        keys = score_dicts[0].keys()
        return {k: (_gmean if k in ratio_keys else _mean)([d[k] for d in score_dicts]) for k in keys}

    summary = dict(cases=len(cases), members=members, methods={}, references={}, states={})
    for m in methods:
        summary['methods'][m['id']] = combine([c['results'][m['id']]['scores'] for c in cases])
        summary['states'][m['id']] = {name: combine([c['results'][m['id']]['states'][name] for c in cases])
                                      for name in STATE_FIELDS}
    summary['reference_states'] = {}
    for name in ('truth', 'coarse', 'regression'):
        summary['references'][name] = combine([c['references'][name]['scores'] for c in cases])
        summary['reference_states'][name] = {f: combine([c['references'][name]['states'][f] for c in cases])
                                             for f in STATE_FIELDS}
    return summary


def verdicts(summary, methods, crps_tolerance=1., ls_tolerance=2., bias_tolerance=10.):
    """Plain-language verdict per method against the baseline (tolerances in %)."""
    base = summary['methods'].get('baseline')
    if base is None:
        return {}
    result = {}
    for m in methods:
        if m['id'] == 'baseline':
            continue
        s = summary['methods'][m['id']]
        issues, gains = [], []
        if s['fine_power_ratio'] and base['fine_power_ratio']:
            before, after = abs(np.log(base['fine_power_ratio'])), abs(np.log(s['fine_power_ratio']))
            if after < before-.02:
                gains.append(f'fine-scale power ×{base["fine_power_ratio"]:.2f} → ×{s["fine_power_ratio"]:.2f} (closer to truth)')
            elif after > before+.02:
                issues.append(f'fine-scale power ×{base["fine_power_ratio"]:.2f} → ×{s["fine_power_ratio"]:.2f} (further from truth)')
        dcrps = 100*(s['crps']-base['crps'])/base['crps'] if base['crps'] else 0.
        dls = 100*(s['large_scale_rmse']-base['large_scale_rmse'])/base['large_scale_rmse'] if base['large_scale_rmse'] else 0.
        (issues if dcrps > crps_tolerance else gains if dcrps < -crps_tolerance else []).append(f'CRPS {dcrps:+.1f}%')
        (issues if dls > ls_tolerance else gains if dls < -ls_tolerance else []).append(f'large-scale RMSE {dls:+.1f}%')
        if s['bias_pct'] is not None and abs(s['bias_pct']) > bias_tolerance and \
                abs(s['bias_pct']) > abs(base['bias_pct'] or 0)+2:
            issues.append(f'rain bias {s["bias_pct"]:+.1f}%')
        for name in STATE_FIELDS:
            b, v = summary['states']['baseline'][name]['crps'], summary['states'][m['id']][name]['crps']
            change = 100*(v-b)/b if b else 0.
            if change > STATE_CRPS_TOLERANCE:
                issues.append(f'{name} CRPS {change:+.1f}%')
            elif change < -STATE_CRPS_TOLERANCE:
                gains.append(f'{name} CRPS {change:+.1f}%')
        sharper = any(g.startswith('fine-scale') for g in gains)
        verdict = ('recommended' if sharper and not issues else
                   'trade-off' if sharper else
                   'worse' if issues else 'no clear effect')
        result[m['id']] = dict(verdict=verdict, gains=gains, issues=issues)
    return result


def write_report(out, summary, cases, methods, settings, judged):
    L = ['# v4.1 inference sharpening ablation', '',
         f'Checkpoint `{settings["checkpoint"]}` (epoch {settings["epoch"]}), split `{settings["split"]}`, '
         f'{summary["cases"]} case(s) × {summary["members"]} member(s), {settings["steps"]} Heun steps. '
         'Every method uses the same member noise.', '']
    if settings.get('guide'):
        L += [f'Autoguidance weak model: `{settings["guide"]}` (epoch {settings["guide_epoch"]}).', '']
    elif settings.get('guide_note'):
        L += [settings['guide_note'], '']
    L += ['## Methods', '', '| Method | What it does |', '|---|---|']
    L += [f'| {m["label"]} | {m["desc"]} |' for m in methods]
    L += ['', '## Verdicts (mean over cases, vs baseline)', '',
          'Recommended = fine-scale rain power moves toward truth with rain CRPS within ±1 %, large-scale RMSE '
          f'within ±2 %, no added rain bias beyond 10 % and no other field\'s CRPS worse by more than '
          f'{STATE_CRPS_TOLERANCE:g} %.', '',
          '| Method | Verdict | Gains | Costs |', '|---|---|---|---|']
    for m in methods:
        if m['id'] in judged:
            v = judged[m['id']]
            L.append(f'| {m["label"]} | **{v["verdict"]}** | {"; ".join(v["gains"]) or "–"} | {"; ".join(v["issues"]) or "–"} |')
    head = ('| | Fine power ×truth | Meso power ×truth | Gradient ×truth | CRPS | LS-RMSE | Bias % | '
            'Wet ≥0.1 | p99.9 | Peak | FSS 1mm 25km | FSS 5mm 25km | Member MAE | s/member |')
    rule = '|---|' + '---:|'*14

    def row(name, s, seconds=None, bold=False):
        cells = [_fmt(s['fine_power_ratio'], '.2f'), _fmt(s['meso_power_ratio'], '.2f'), _fmt(s['gradient_ratio'], '.2f'),
                 f'{s["crps"]:.4f}', f'{s["large_scale_rmse"]:.4f}', _fmt(s['bias_pct'], '+.1f'),
                 f'{s["wet_frac_0p1"]:.1%}', f'{s["p99_9"]:.2f}', f'{s["peak"]:.1f}',
                 _fmt(s.get('fss_1mm_25km')), _fmt(s.get('fss_5mm_25km')), f'{s["member_mae"]:.4f}',
                 '–' if seconds is None else f'{seconds:.0f}']
        name = f'**{name}**' if bold else name
        return f'| {name} | ' + ' | '.join(cells) + ' |'

    def table(block, refs):
        lines = [head, rule, row('Ground truth', refs['truth'], bold=True), row('Coarse input', refs['coarse']),
                 row('Frozen regression', refs['regression'])]
        lines += [row(m['label'], block[m['id']], block[m['id']].get('seconds_per_member')) for m in methods]
        return lines

    L += ['', '## Rain scores, mean over cases', '',
          'Power and gradient ratios are geometric means over cases (1 = truth). CRPS equals member MAE when '
          'there is one member. Member MAE/RMSE punish sharper fields (double penalty); judge sharpening by the '
          'spectra, CRPS and large-scale RMSE instead.', '']
    L += table(summary['methods'], summary['references'])
    L += ['', '## Other fields, mean over cases', '',
          'Fine/meso = power ratio to truth below 30 km / at 30–150 km (1 = truth). Bias and RMSE in physical '
          'units (K, Pa, m/s, kg/kg). LS-RMSE = RMSE of ~25 km block means.', '']
    for name in STATE_FIELDS:
        L += [f'### {DISPLAY[name][0]} (`{name}`)', '',
              '| | Fine ×truth | Meso ×truth | Gradient ×truth | CRPS | LS-RMSE | Member RMSE | Bias |',
              '|---|' + '---:|'*7]
        rows = [(r.capitalize(), summary['reference_states'][r][name]) for r in ('coarse', 'regression')]
        rows += [(m['label'], summary['states'][m['id']][name]) for m in methods]
        for label, st in rows:
            L.append(f'| {label} | {_fmt(st["fine_power_ratio"], ".2f")} | {_fmt(st["meso_power_ratio"], ".2f")} | '
                     f'{_fmt(st["gradient_ratio"], ".2f")} | {st["crps"]:.4g} | {st["large_scale_rmse"]:.4g} | '
                     f'{st["member_rmse"]:.4g} | {st["bias"]:+.3g} |')
        L.append('')
    for c in cases:
        L += ['', f'## Case `{c["id"]}` ({c["reason"]})', '']
        L += table({k: v['scores'] for k, v in c['results'].items()},
                   {k: v['scores'] for k, v in c['references'].items()})
    L += ['', '## Settings', '', '```json', json.dumps(settings, indent=2, default=str), '```']
    (out/'report.md').write_text('\n'.join(L)+'\n')


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------

def compare_sharpness(cfg, checkpoint='best', output=None, split='test', samples=0, wettest=1, timestamps=None,
                      steps=None, members=4, methods=METHOD_IDS, combine=('churn', 'autoguide', 'residual_scale'),
                      warp_gamma=1.5, churn=0.1, churn_range=(0.1, 0.8), guide_checkpoint='auto', guide_weight=1.5,
                      residual_scale=1.10, dry_cutoff=0.1, tukey_alpha=0.3, steps_factor=2,
                      zoom_size=256, dpi=300, pdf=True, weights='ema', seed=317, batch=32, threads=8, use_cartopy=True, map_features=True, use_native=True,
                      log=print):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from .validation_v4_1 import _style
    _style(plt)
    if members < 1:
        raise ValueError('members must be >= 1')
    FIGURES.update(dpi=dpi, pdf=pdf)

    path = resolve_checkpoint(cfg, checkpoint)
    device = device_for(cfg['train']['device'])
    archive, model, conditioner, saved = load_model(cfg, path, device, weights)
    digest = file_hash_v2(path)
    steps = steps or cfg['inference']['steps']
    methods = list(methods)
    guide, guide_note = None, None
    if 'autoguide' in methods or ('combined' in methods and 'autoguide' in combine):
        guide = load_guide(cfg, guide_checkpoint, archive, saved, device, log)
        if guide is None:
            guide_note = 'Autoguidance skipped: no earlier kept checkpoint of this run.'
            methods = [m for m in methods if m != 'autoguide']
            combine = tuple(c for c in combine if c != 'autoguide')
    specs = build_methods(methods, combine, steps, warp_gamma, churn, guide_weight, residual_scale,
                          dry_cutoff, tukey_alpha, steps_factor)
    if not specs:
        raise ValueError('No methods to compare')

    job = os.environ.get('SLURM_JOB_ID') or time.strftime('%Y%m%d_%H%M%S')
    out = Path(output) if output else (Path(cfg['train']['output'])/'evaluation'/
                                      f'sharpness_{path.stem}{"_raw" if weights == "raw" else ""}_{split}_{digest[:8]}_{job}')
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'{out} is not empty; pass a fresh --output')
    out.mkdir(parents=True, exist_ok=True)
    label = checkpoint if not Path(str(checkpoint)).is_file() else path.stem
    log(f'Sharpness ablation: {path} (epoch {saved["epoch"]+1}); device {device}; {steps} steps; '
        f'{members} member(s); output {out}')
    log('Methods: '+'; '.join(f'{m["label"]} [{m["desc"]}]' for m in specs))
    cases = select_cases(archive, split, timestamps, samples, wettest, seed, log)
    log('Cases: '+', '.join(f'{c["entry"]["id"]} ({c["reason"]})' for c in cases))
    canvas = Canvas(archive, use_cartopy, map_features, log)
    area = np.asarray(archive.static['area'], dtype='float64')
    dx_km = float(np.sqrt(np.median(area))/1000)

    reports = []
    for number, case in enumerate(cases, 1):
        entry = case['entry']
        started = time.monotonic()
        sampler = DomainSampler(model, conditioner, archive, entry, cfg, device, batch, threads,
                                guide=guide['model'] if guide else None)
        log(f'\n[{number}/{len(cases)}] {entry["id"]}: {len(sampler.tiles)} tiles ready in {time.monotonic()-started:.0f}s')
        seeds = [member_seed(cfg, entry, k) for k in range(members)]
        results, references, t_freq = evaluate_case(sampler, archive, entry, specs, seeds, steps,
                                                    churn_range, dx_km, log)
        del sampler
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        folder = out/'cases'/entry['id']
        folder.mkdir(parents=True, exist_ok=True)
        heading = (f'v4.1 · epoch {saved["epoch"]+1} ({label}) · {split} {entry["time"].replace("T", " ")[:16]} UTC · '
                   f'{case["reason"]}')
        native = native_fields(entry, canvas.lat, canvas.lon, log=log)[0] if use_native else {}
        window = event_window(references['truth']['rain'], zoom_size)
        (folder/'maps').mkdir(exist_ok=True)
        for name in FIELDS:
            log(f'  rendering {name} maps...')
            plot_maps(canvas, results, references, native, name, heading, folder/'maps'/f'{name}_conus.png', plt)
            plot_maps(canvas, results, references, native, name, heading, folder/'maps'/f'{name}_zoom.png', plt,
                      window=window)
        log('  rendering all-field zoom and spectra...')
        plot_all_fields_zoom(canvas, results, references, native, window, heading,
                             folder/'sharpness_all_fields_zoom.png', plt)
        plot_spectra(results, references, t_freq, dx_km, heading, folder/'sharpness_spectra.png', plt)
        plot_all_spectra(results, references, t_freq, dx_km, heading, folder/'sharpness_spectra_all_fields.png', plt)
        record = dict(id=entry['id'], time=entry['time'], reason=case['reason'],
                      references={k: dict(scores=v['scores'], states=v['states']) for k, v in references.items()},
                      results={k: dict(scores=v['scores'], states=v['states']) for k, v in results.items()},
                      seconds=round(time.monotonic()-started, 1))
        write_json(folder/'metrics.json', record)
        reports.append(record)
        log(f'[{number}/{len(cases)}] {entry["id"]} done in {record["seconds"]:.0f}s → {folder}')

    summary = summarize(reports, specs, members)
    judged = verdicts(summary, specs)
    settings = dict(checkpoint=str(path.resolve()), checkpoint_sha256=digest, epoch=saved['epoch']+1, weights=weights, split=split,
                    steps=steps, members=members, grid_km=dx_km, churn_range=list(churn_range),
                    guide=str(guide['path']) if guide else None, guide_epoch=guide['epoch'] if guide else None,
                    guide_note=guide_note, fine_band_km=list(FINE_KM), meso_band_km=list(MESO_KM),
                    large_scale_km=LARGE_SCALE_KM,
                    methods=[{k: v for k, v in m.items() if k != 'color'} for m in specs])
    plot_tradeoff(summary, specs, out/'summary_tradeoff.png', plt)
    plot_scorecard(summary, specs, out/'summary_scorecard.png', plt)
    write_report(out, summary, reports, specs, settings, judged)
    write_json(out/'summary_metrics.json', dict(settings=settings, summary=summary, verdicts=judged))
    log('\nVerdicts vs baseline:')
    for m in specs:
        if m['id'] in judged:
            v = judged[m['id']]
            log(f'  {m["label"]:<28} {v["verdict"]:<16} {"; ".join(v["gains"] + v["issues"])}')
    log(f'Sharpness ablation written to {out}')
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', default='configs/discover_v4_1.yaml')
    parser.add_argument('--checkpoint', default='best', help='best (default) | latest | <epoch number> | <path>')
    parser.add_argument('--weights', choices=('ema', 'raw'), default='ema', help='ema (default: exponential moving average) | raw (the optimizer weights at that checkpoint)')
    parser.add_argument('--latest', action='store_true', help='Shortcut for --checkpoint latest')
    parser.add_argument('--output', help='Fresh output directory (default under <train.output>/evaluation/)')
    parser.add_argument('--split', choices=('val', 'test'), default='test')
    parser.add_argument('--samples', type=int, default=0, help='Seeded random hours')
    parser.add_argument('--wettest', type=int, default=1, help='Wettest hours (≥12 h apart)')
    parser.add_argument('--timestamps', nargs='+', help='Exact IDs or ISO times')
    parser.add_argument('--steps', type=int, help='Heun steps (default: inference.steps)')
    parser.add_argument('--members', type=int, default=4, help='Members per method (CRPS needs ≥ 2; default 4)')
    parser.add_argument('--methods', default=','.join(METHOD_IDS), help=f'Comma list from: {", ".join(METHOD_IDS)}')
    parser.add_argument('--combine', default='churn,autoguide,residual_scale',
                        help='Methods whose knobs the "combined" method applies together')
    parser.add_argument('--warp-gamma', type=float, default=1.5, help='Time-warp exponent (default 1.5)')
    parser.add_argument('--steps-factor', type=int, default=2, help='Step multiplier for more_steps (default 2)')
    parser.add_argument('--churn', type=float, default=0.1, help='Churn: noise-level increase per step (default 0.1)')
    parser.add_argument('--churn-range', type=float, nargs=2, default=(0.1, 0.8), metavar=('TMIN', 'TMAX'),
                        help='Flow times with churn (0 noise .. 1 data; default 0.1 0.8)')
    parser.add_argument('--guide-checkpoint', default='auto',
                        help='Autoguidance weak model: auto (kept epoch nearest 1/3 of main) | <epoch> | <path>')
    parser.add_argument('--guide-weight', type=float, default=1.5, help='Autoguidance weight w (1 = off; default 1.5)')
    parser.add_argument('--residual-scale', type=float, default=1.10, help='Residual scale (default 1.10)')
    parser.add_argument('--dry-cutoff', type=float, default=0.1, help='Wet cutoff in mm/h (default 0.1)')
    parser.add_argument('--tukey-alpha', type=float, default=0.3, help='Tukey taper fraction (default 0.3)')
    parser.add_argument('--zoom-size', type=int, default=256, help='Rain-event zoom edge in grid pixels')
    parser.add_argument('--dpi', type=int, default=300, help='Figure resolution for PNG and PDF map rasters (default 300)')
    parser.add_argument('--no-pdf', action='store_true', help='PNG only (PDFs of every figure by default)')
    parser.add_argument('--seed', type=int, default=317, help='Case selection seed')
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--cartopy-data-dir', help='Natural Earth cache')
    parser.add_argument('--no-cartopy', action='store_true')
    parser.add_argument('--no-map-features', action='store_true')
    parser.add_argument('--no-native', action='store_true')
    args = parser.parse_args()
    if args.cartopy_data_dir:
        import cartopy
        cartopy.config['pre_existing_data_dir'] = args.cartopy_data_dir
    split = lambda text: tuple(s.strip() for s in text.split(',') if s.strip())
    cfg = load_config(args.config)
    compare_sharpness(
        cfg, checkpoint='latest' if args.latest else args.checkpoint, output=args.output, split=args.split,
        samples=args.samples, wettest=args.wettest, timestamps=args.timestamps, steps=args.steps,
        members=args.members, methods=split(args.methods), combine=split(args.combine),
        warp_gamma=args.warp_gamma, churn=args.churn, churn_range=tuple(args.churn_range),
        guide_checkpoint=args.guide_checkpoint, guide_weight=args.guide_weight,
        residual_scale=args.residual_scale, dry_cutoff=args.dry_cutoff, tukey_alpha=args.tukey_alpha,
        steps_factor=args.steps_factor, zoom_size=args.zoom_size, dpi=args.dpi, pdf=not args.no_pdf, weights=args.weights, seed=args.seed, batch=args.batch, threads=args.threads,
        use_cartopy=not args.no_cartopy, map_features=not args.no_map_features, use_native=not args.no_native,
        log=lambda message: print(message, flush=True))


if __name__ == '__main__':
    main()
