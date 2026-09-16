"""Instrument the production sampler without changing its scales or updates."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from scipy.ndimage import label

from .config import write_json
from .config_v2 import validate_config_v2, v2_path
from .dataset_v2 import ArchiveV2
from .inference import starts
from .inference_v2 import load_models_v2, sample_frame_v2
from .physics_v2 import TARGETS_V2, transform_v2, inverse_v2
from .precip_audit_v2 import audit_case, deterministic_scores
from .train import device_for
from .train_v2 import file_hash_v2


def field_summary(value):
    """Unweighted grid statistics; preserve nonfinite counts in failure reports."""
    value = np.asarray(value)
    finite = value[np.isfinite(value)].astype('float64')
    result = {'nonfinite_pixels': int(value.size-finite.size), 'pixels': int(value.size)}
    if finite.size:
        result.update(min=float(finite.min()), max=float(finite.max()), mean=float(finite.mean()),
                      rms=float(np.sqrt(np.mean(finite**2))), negative_fraction=float(np.mean(finite < 0)),
                      quantiles=dict(zip(('p01', 'p50', 'p99', 'p999'),
                                         map(float, np.quantile(finite, [.01, .5, .99, .999])))))
    return result


def structure_scores(rain, area, patch):
    """Connected wet objects and gradients near core edges; not a seam proof."""
    result = {}
    for threshold in (.1, 1., 5.):
        objects, count = label(rain >= threshold)  # four-neighbor connectivity
        sizes = np.bincount(objects.ravel())
        small = (objects > 0) & (sizes[objects] <= 4)
        wet_area = float(area[rain >= threshold].sum())
        result[str(threshold)] = {'objects': int(count), 'wet_area_fraction': wet_area/float(area.sum()),
                                  'wet_area_in_objects_le_4_pixels': float(area[small].sum())/wet_area if wet_area else None}
    edge_scores = {}
    for axis, name in ((0, 'y'), (1, 'x')):
        boundary = np.zeros(rain.shape[axis]-1, dtype=bool)
        for start in starts(rain.shape[axis], patch['size'], patch['stride']):
            for edge in (start, start+patch['size']):
                if 0 < edge < rain.shape[axis]:
                    boundary[edge-1] = True
        gradient = np.abs(np.diff(rain.astype('float64'), axis=axis))
        mask = np.broadcast_to(boundary[:, None] if axis == 0 else boundary[None, :], gradient.shape)
        edge_scores[name] = {'core_edge_mean_abs_difference': float(gradient[mask].mean()) if mask.any() else None,
                            'other_mean_abs_difference': float(gradient[~mask].mean()) if (~mask).any() else None}
    return {'wet_objects': result, 'neighbor_differences_mm_h': edge_scores,
            'limits': 'Object counts depend on threshold and wet area. Core-edge gradients depend on weather location; '
                      'neither proves a tiling defect. Pixel sizes are not physical distances.'}


class SamplerTrace:
    def __init__(self, archive, entry, cfg, scale):
        self.archive, self.entry, self.cfg = archive, entry, cfg
        self.scale = float(scale.detach().cpu().reshape(-1)[1])
        self.rs, self.rm = float(archive.rs[1, 0, 0]), float(archive.rm[1, 0, 0])
        p = cfg['patch']
        ys, xs = starts(archive.shape[0], p['size'], p['stride']), starts(archive.shape[1], p['size'], p['stride'])
        truth = archive.array(entry, 'truth')[1]
        wet = max(((y, x) for y in ys for x in xs), key=lambda pos:
                  float(truth[pos[0]:pos[0]+p['size'], pos[1]:pos[1]+p['size']].mean()))
        self.traced = {(ys[0], xs[0]), (ys[len(ys)//2], xs[len(xs)//2]), wet}
        self.trajectory, self.tiles, self.snapshots, self.fields = [], [], {}, {}
        self.mass, self.first, self.second = [np.zeros(archive.shape, dtype='float64') for _ in range(3)]
        self.count = np.zeros(archive.shape, dtype='int16')

    def trace_tile(self, y, x):
        return (y, x) in self.traced

    def on_ode(self, y, x, step, time, state, mean):
        size, halo = self.cfg['patch']['size'], self.cfg['patch']['halo']
        rain_state = state[0, 1, halo:halo+size, halo:halo+size].float().cpu().numpy().copy()
        record = {'tile_y': y, 'tile_x': x, 'step': step, 'time': time, 'state': field_summary(rain_state)}
        self.trajectory.append(record)
        steps = self.cfg['inference']['steps']
        if step in {0, steps//4, steps//2, 3*steps//4, steps}:
            self.snapshots[f'y{y}_x{x}_step{step}'] = rain_state
        if record['state']['nonfinite_pixels']:
            raise FloatingPointError(f'Nonfinite ODE state: tile {y},{x}, step {step}')

    def on_tile(self, y, x, endpoint, prediction, mean, window):
        if endpoint is None:
            raise ValueError('Flow diagnostics require a flow model')
        p = self.cfg['patch']
        halo, size = p['halo'], p['size']
        raw = endpoint[0, 1, halo:halo+size, halo:halo+size].float().cpu().numpy()
        self.tiles.append({'y': y, 'x': x, 'flow_endpoint': field_summary(raw),
                           'normalized_correction': field_summary(prediction[1]-mean[1])})
        if not np.isfinite(raw).all() or not np.isfinite(prediction).all():
            raise FloatingPointError(f'Nonfinite flow endpoint/prediction at tile {y},{x}')
        region = np.s_[y:y+size, x:x+size]
        # Measure disagreement of tile predictions in log-rainfall units before
        # blending. Baseline and residual mean are common at each physical pixel.
        local = prediction[1].astype('float64')*self.rs
        self.mass[region] += window
        self.first[region] += window*local
        self.second[region] += window*local**2
        self.count[region] += 1

    def on_frame(self, prediction, mean, baseline):
        log_rain = baseline[1]+prediction[1]*self.rs+self.rm
        log_mean = baseline[1]+mean[1]*self.rs+self.rm
        truth_log = np.log1p(self.archive.array(self.entry, 'truth')[1]/self.archive.stats['precip_log_scale'])
        target = (truth_log-baseline[1]-self.rm)/self.rs
        self.fields = {
            'coarse_log_precip': baseline[1], 'regression_normalized_residual': mean[1],
            'flow_normalized_residual': prediction[1],
            'flow_normalized_correction': prediction[1]-mean[1],
            'flow_endpoint_reconstructed_after_blending': (prediction[1]-mean[1])/self.scale,
            'HWT_required_endpoint_after_blending': (target-mean[1])/self.scale,
            'flow_log_correction': (prediction[1]-mean[1])*self.rs,
            'flow_log_precip_before_clipping': log_rain,
            'regression_log_precip_before_clipping': log_mean,
            'tile_prediction_log_std': np.sqrt(np.maximum(self.second/self.mass-(self.first/self.mass)**2, 0)).astype('float32'),
            'tile_coverage_count': self.count}

    def save(self, out, stamp):
        write_json(out/f'{stamp}_trace_v2.json', {
            'traced_tiles': [list(p) for p in sorted(self.traced)], 'trajectory': self.trajectory,
            'tiles': self.tiles, 'fields': {name: field_summary(value) for name, value in self.fields.items()},
            'limits': 'Intermediate ODE states are not forecasts. Only selected tiles have trajectories; all tiles have endpoint summaries. '
                      'The HWT-required endpoint is calculated relative to blended regression; it is not a demand that each random sample match HWT. '
                      'Overlap standard deviation uses weighted tile disagreement even in owner mode.'})
        np.savez_compressed(out/f'{stamp}_trajectory_v2.npz', **self.snapshots)


def save_fields(path, archive, entry, fields, attrs):
    ds = xr.Dataset({name: (('time', 'Ydim', 'Xdim'), np.asarray(value, dtype='float32')[None],
                           {'units': 'mm h-1' if name in ('precip', 'regression_precip', 'HWT_precip', 'coarse_precip') else '1'})
                     for name, value in fields.items()}, coords={'time': [np.datetime64(entry['time'])]}, attrs=attrs)
    for name in ('lat', 'lon', 'area'):
        ds[name] = (('Ydim', 'Xdim'), archive.static[name])
    ds.to_netcdf(path, engine='h5netcdf', encoding={name: {'zlib': True, 'complevel': 2} for name in fields})
    ds.close()


def plot_internal(path, trace, truth, rain):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fields = trace.fields
    panels = [('HWT rain', truth, False), ('Decoded flow rain', rain, False),
              ('Raw flow endpoint (reconstructed after blend)', fields['flow_endpoint_reconstructed_after_blending'], True),
              ('Required endpoint relative to HWT', fields['HWT_required_endpoint_after_blending'], True),
              ('Correction after flow_scale', fields['flow_normalized_correction'], True),
              ('Correction after archive residual std', fields['flow_log_correction'], True),
              ('Log rainfall before clipping/exp', fields['flow_log_precip_before_clipping'], True),
              ('Tile disagreement in log rainfall', fields['tile_prediction_log_std'], False),
              ('Tile coverage count', fields['tile_coverage_count'], False)]
    fig, axes = plt.subplots(3, 3, figsize=(16, 11), constrained_layout=True)
    rain_limit = max(.1, float(np.percentile(truth, 99.5)), float(np.percentile(rain, 99.5)))
    raw_limit = max(.1, *(float(np.percentile(np.abs(panels[i][1]), 99.5)) for i in (2, 3)))
    for i, (ax, (name, value, signed)) in enumerate(zip(axes.flat, panels)):
        limit = rain_limit if i < 2 else raw_limit if i in (2, 3) else max(.01, float(np.percentile(np.abs(value), 99.5)))
        if i == 8:
            limit = float(value.max())
        im = ax.imshow(value, origin='lower', interpolation='nearest', cmap='RdBu_r' if signed else 'viridis',
                       vmin=-limit if signed else 0, vmax=limit)
        peak = np.unravel_index(np.argmax(np.abs(value)), value.shape)
        ax.scatter(peak[1], peak[0], s=25, marker='*', c='yellow', edgecolors='black', linewidths=.4)
        ax.set(title=f'{name}\nrange [{value.min():.3g}, {value.max():.3g}]; saturated {np.mean(np.abs(value)>limit):.2%}', xticks=[], yticks=[])
        fig.colorbar(im, ax=ax, shrink=.7, extend='both' if signed else 'max', label='mm h⁻¹' if i < 2 else 'dimensionless')
    fig.suptitle(path.stem+'\n99.5th-percentile display limits; unmodified full fields in NetCDF; star = largest absolute value')
    fig.savefig(path, dpi=130)
    plt.close(fig)


def run_flow_diagnostic(cfg, checkpoint, output, timestamps, steps=(24, 48, 96), members=5,
                        split='test', plots=True):
    cfg = deepcopy(cfg)
    validate_config_v2(cfg)
    if split not in ('val', 'test') or members < 2 or len(set(steps)) != len(steps) or len(steps) < 2 or min(steps) < 1:
        raise ValueError('Require val/test, at least two members, and at least two distinct positive step counts')
    if not timestamps or len(set(timestamps)) != len(timestamps):
        raise ValueError('Require unique exact timestamps')
    steps = sorted(steps)
    cfg['inference']['members'] = members
    archive = ArchiveV2(cfg['data']['prepared'])
    entries = []
    for stamp in timestamps:
        matches = [e for e in archive.index['entries'] if e['split'] == split and stamp in (e['id'], e['time'])]
        if len(matches) != 1:
            raise ValueError(f'No unique {split} entry for {stamp}')
        entries.append(matches[0])
    out = v2_path(output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'{out} is not empty; use a fresh output')
    digest = file_hash_v2(checkpoint)
    device = device_for(cfg['train']['device'])
    mean, flow, scale, ckpt = load_models_v2(cfg, checkpoint, archive, device)
    if flow is None or file_hash_v2(checkpoint) != digest:
        raise ValueError('Require a stable flow checkpoint; checkpoint changed while loading or is regression-only')
    if scale.shape != (1, 5, 1, 1) or not torch.isfinite(scale).all() or not (scale > 0).all():
        raise ValueError('Invalid checkpoint flow_scale: require finite positive [1,5,1,1]')
    if not np.isfinite(archive.rs).all() or not (archive.rs > 0).all() or not np.isfinite(archive.rm).all():
        raise ValueError('Invalid archive residual normalization')
    out.mkdir(parents=True, exist_ok=True)
    scales = scale.detach().cpu().reshape(-1).tolist()
    manifest = {'checkpoint': str(Path(checkpoint).resolve()), 'checkpoint_sha256': digest,
                'checkpoint_epoch': ckpt['epoch']+1, 'regression_sha256': ckpt['regression_sha256'],
                'dataset_fingerprint': archive.index['fingerprint'], 'stats': archive.stats,
                'flow_scale_by_channel': dict(zip(TARGETS_V2, scales)),
                'effective_log_scale_precip': scales[1]*float(archive.rs[1, 0, 0]),
                'config': cfg, 'steps': steps, 'members': members, 'split': split,
                'timestamps': [e['id'] for e in entries],
                'formula': 'L = log1p(coarse/s) + residual_mean + residual_std * (regression + flow_scale * endpoint); P = s * expm1(max(L,0))',
                'weights_loaded_once': True, 'normalization_changed': False}
    write_json(out/'manifest_v2.json', manifest)
    print(f'Checkpoint: {checkpoint}; epoch {ckpt["epoch"]+1}; SHA256 {digest}', flush=True)
    print(f'Precip flow_scale={scales[1]:.6g}; residual_std={archive.rs[1,0,0]:.6g}; '
          f'effective log scale={manifest["effective_log_scale_precip"]:.6g}', flush=True)
    for entry in entries:
        case = out/entry['id']
        case.mkdir()
        truth = np.asarray(archive.array(entry, 'truth')[1])
        baseline = np.asarray(archive.array(entry, 'baseline')[1])
        area = archive.static['area']
        seeds = [int(np.random.SeedSequence([cfg['inference']['seed'], int(entry['id'].replace('_', '')), m]).generate_state(1)[0]) for m in range(members)]
        # Check the actual stored residual against the transform equation.
        reconstructed = transform_v2(archive.array(entry, 'truth'), archive.stats['precip_log_scale'])-transform_v2(archive.array(entry, 'baseline'), archive.stats['precip_log_scale'])
        saved_residual = archive.array(entry, 'residual')
        consistent = bool(np.allclose(reconstructed, saved_residual, rtol=1e-5, atol=1e-5))
        roundtrip = inverse_v2(transform_v2(archive.array(entry, 'truth'), archive.stats['precip_log_scale']), archive.stats['precip_log_scale'])
        checks = {'stored_residual_matches_transform': consistent,
                  'stored_residual_max_abs_error': float(np.max(np.abs(reconstructed-saved_residual))),
                  'rain_roundtrip_max_abs_error_mm_h': float(np.max(np.abs(roundtrip[1]-truth)))}
        write_json(case/'normalization_checks_v2.json', checks)
        if not consistent:
            raise ValueError(f'{entry["id"]}: stored residual does not match transform equation')
        previous, previous_steps, reports, convergence = None, None, {}, {}
        for nsteps in steps:
            cfg['inference']['steps'] = nsteps
            destination = case/f'steps_{nsteps:03d}_v2'
            destination.mkdir()
            prediction_dir = destination/'predictions_v2'
            prediction_dir.mkdir()
            rain_members, endpoints, regression = [], [], None
            for member, seed in enumerate(seeds):
                stamp = f'{entry["id"]}_m{member:03d}'
                print(f'{stamp}: {nsteps} Heun steps, seed={seed}', flush=True)
                trace = SamplerTrace(archive, entry, cfg, scale)
                try:
                    values, deterministic, _ = sample_frame_v2(mean, flow, scale, archive, entry, cfg, device, seed, diagnostics=trace)
                except Exception as error:
                    trace.save(destination, stamp)
                    if trace.fields:
                        np.savez_compressed(destination/f'{stamp}_failed_fields_v2.npz', **trace.fields)
                    write_json(destination/f'{stamp}_failure_v2.json', {'error': str(error), 'seed': seed, 'steps': nsteps})
                    raise
                trace.save(destination, stamp)
                attrs = dict(version='v2', stage='flow', checkpoint_sha256=digest, checkpoint_epoch=ckpt['epoch']+1,
                             regression_sha256=ckpt['regression_sha256'], dataset_fingerprint=archive.index['fingerprint'],
                             split=split, ensemble_member=member, seed=seed, ode_steps=nsteps,
                             blend=cfg['inference'].get('blend', 'weighted'),
                             target_alignment='HR midpoint snapshot approximates coarse hourly mean', conservation='none; audit only')
                save_fields(prediction_dir/f'{stamp}_v2.nc', archive, entry,
                            {**trace.fields, 'precip': values[1], 'regression_precip': deterministic[1]}, attrs)
                if plots:
                    plot_internal(destination/f'{stamp}_internal_v2.png', trace, truth, values[1])
                if regression is not None and not np.array_equal(regression, deterministic[1]):
                    raise ValueError('Regression changed across members')
                regression = deterministic[1].copy()
                rain_members.append(values[1].copy())
                endpoints.append(trace.fields['flow_endpoint_reconstructed_after_blending'].copy())
            ensemble, endpoint_stack = np.stack(rain_members), np.stack(endpoints)
            report = audit_case(ensemble, regression, baseline, truth, area, cfg['patch'], archive.stats['precip_log_scale'])
            report.update(id=entry['id'], time=entry['time'], split=split, seeds=seeds, checkpoint_sha256=digest, ode_steps=nsteps)
            report['structure'] = {name: structure_scores(field, area, cfg['patch']) for name, field in
                                   [('HWT', truth), ('regression', regression), ('flow_mean', ensemble.mean(0)),
                                    *[(f'member_{i:03d}', rain) for i, rain in enumerate(ensemble)]]}
            write_json(destination/'audit_v2.json', report)
            if plots:
                from .precip_plots_v2 import plot_case
                plot_case(destination, report, ensemble, regression, baseline, truth)
            if previous is not None:
                convergence[f'{previous_steps}_to_{nsteps}'] = [
                    {'member': m, 'seed': seeds[m],
                     'rain_change_rmse_mm_h': deterministic_scores(ensemble[m], previous[0][m], area)['rmse'],
                     'endpoint_change_rms': deterministic_scores(endpoint_stack[m], previous[1][m], area)['rmse']}
                    for m in range(members)]
            previous, previous_steps = (ensemble, endpoint_stack), nsteps
            reports[str(nsteps)] = {'scores': report['scores'], 'probabilistic': report['probabilistic'],
                                    'profiles': {k: {q: v for q, v in p.items() if q != 'exceedance'} for k, p in report['profiles'].items()}}
        save_fields(case/'reference_v2.nc', archive, entry, {'HWT_precip': truth, 'coarse_precip': baseline}, {'split': split})
        write_json(case/'summary_v2.json', {'checkpoint_sha256': digest, 'seeds': seeds, 'steps': reports,
                                          'same_seed_convergence': convergence, 'normalization_checks': checks})
        lines = ['# Flow inference diagnostic', '', f'Case {entry["id"]}; checkpoint `{Path(checkpoint).name}`; SHA256 `{digest}`.', '',
                 '| Heun steps | Regression RMSE | Flow-mean RMSE | Ensemble CRPS | Largest member rain |',
                 '|---|---:|---:|---:|---:|']
        for n, r in reports.items():
            maximum = max(p['max_mm_h'] for k, p in r['profiles'].items() if k.startswith('member_'))
            lines.append(f'| {n} | {r["scores"]["regression"]["rmse"]:.4g} | {r["scores"]["flow_mean"]["rmse"]:.4g} | {r["probabilistic"]["crps"]:.4g} | {maximum:.4g} |')
        lines += ['', 'Rain quantities are in mm/h. Same seeds, weights, precision, geometry and blend are used at every step count.', '',
                  '- Compare same_seed_convergence in summary_v2.json: decreasing changes suggest numerical convergence, not model accuracy.',
                  '- Large raw endpoints locate a problem before inverse-log decoding; large log corrections show how decoding amplifies it.',
                  '- Each steps directory contains audit_v2.json, internal maps, trajectories and complete member NetCDF fields.',
                  '- Check wet-object statistics and tile disagreement alongside HWT structure. These metrics do not establish a stitching defect.',
                  '- This diagnostic holds normalization fixed. Use validation cases before choosing a production setting.']
        (case/'report_v2.md').write_text('\n'.join(lines)+'\n')
    return out
