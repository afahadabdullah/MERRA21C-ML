"""Single-case search for the best inference-time recipe for crisp, calibrated members.

The fine-tunes showed that some members already have truth-like edges while
others do not, so the edge problem is at least partly in how members are
sampled and blended, not only in the weights. This script runs one case
through many sampling and post-processing recipes from the SAME member noise,
scores every recipe against truth, and ranks them: sharpest members that keep
the ensemble CRPS, bias and seams in check.

To keep ~20 recipes affordable, sampling is restricted to the tiles covering a
region around the front (default 384 px core + 96 px margin). Inside the
region this is the production synchronized tiled Heun sampler; with the
whole domain as region it reproduces ``DomainSampler.sample`` exactly.

Recipes (ids for --methods)
---------------------------
Tile blending (the v4 sampler averages up to 4 overlapping tile predictions):
  baseline      Heun, Hann blending (production)
  hann2, hann3  centre-weighted Hann^2 / Hann^3: the tile that sees a pixel with
                most context dominates, fewer predictions are averaged
  hard          no blending: every pixel takes the velocity of the tile whose
                centre is nearest (>= stride/2 px from any tile edge)
  shift_hard    no blending, and the tile grid alternates every step between the
                regular grid and one shifted by stride/2, so seams never stay put
                (the model heals the previous step's seams at a tile centre)
  shift_hann2   Hann^2 blending on the alternating grids
Stochastic / denoising during inference:
  time_warp     finer steps near the data end, t = 1-(1-tau)^gamma
  churn         EDM churn: re-noise sigma -> sigma(1+churn) before each Heun step
  langevin      Langevin corrector after each step in --langevin-range, using the
                score implied by the velocity, s = -(x - t v)/(1-t); refreshes
                eta*(1-t) of the noise and lets the model re-resolve fine detail
  restart       Restart sampling (Xu et al. 2023): after reaching t=1, re-noise to
                t=--restart-t and integrate again, --restart times
  temp          initial-noise temperature --temp (>1: more fine-scale energy)
  hf_boost      amplify the high-pass part of the velocity by --hf-boost for
                t >= --hf-start (mean-preserving detail amplification)
  autoguide     autoguidance with an earlier kept checkpoint (needs one in the run)
  autoguide_hf  the same guidance restricted to fine scales and mid-t (frequency-decoupled
                guidance, Kynkaanniemi et al. 2024 guidance interval + APG projection):
                v = v + (w-1) * [highpass(v - v_weak) minus its component along v],
                only for t in --ag-range; large scales (and so the mean) untouched
  sde_<e>       marginal-preserving SDE sampling (SiT, Ma et al. 2024):
                dx = [v - e*z]dt + sqrt(2 e (1-t)) dW with z = x - t v, for t in
                --sde-range; unlike churn it adds exactly the drift that keeps the
                marginals, so it adds detail without bias if the model is accurate
  shift4_hard   no blending with FOUR tile grids (offsets 0, 1/2, 1/4-3/4, 3/4-1/4 of
                the stride) cycled every step (SpotDiffusion-style moving seams)
  restart_shift DemoFusion/SDEdit-style: solve, re-noise to t=0.5, re-solve with
                shift4_hard tiling so every tile inherits one coherent large-scale state
  fk_edge       FK steering whose reward is the climatological p99.9 |grad| of the edge fields
                (the sharpest lines only) instead of p99 of all fields
  <fk>@<lam>    any FK recipe with strength lambda (in units of the reward spread between
                particles; default 2), e.g. fk_edge@1, fk_edge@3
  fk_steer      Feynman-Kac steering (Singhal et al. 2025): --fk-particles particles per
                member with Langevin noise; at t in --fk-times each particle's clean
                estimate is scored by the truth-free climatological sharpness match and
                particles are resampled with weights exp(lambda * reward increment);
                prunes blurry trajectories early instead of paying for full solves
  vguide_<s>    v10m-edge guidance INSIDE the sampler, strength s (--vguide-strengths):
                for t in --vguide-range, each member's clean estimate
                x1 = x + (1-t) v is converted to physical units; t2m and q2m
                (--vguide-fields) are guided-filtered with the member's own v10m
                as the edge guide, and the state is nudged toward it; the total
                strength s is spread over the guided steps (not applied per step).
                The model sees the sharpened state on the remaining steps and can
                repair it; the post-hoc filter (below) cannot.
Post-processing / selection (reuse baseline members; no extra solves):
  select_clim   draw --pool members, keep the --members whose sharpness is closest
                to the climatological sharpness for this case (truth-free:
                target = [truth/coarse p99 |grad| ratio over --clim-count training
                hours within --clim-days of the date] x this case's coarse p99 |grad|)
  select_sharp  keep the --members sharpest of the pool (same score, one-sided)
  prescreen     rank the pool's noise seeds by an 8-step solve, keep the best
                --members (tests whether cheap solves predict sharp members)
  spectral      raise each member's fine-scale spectrum toward the climatological
                truth spectrum (gain <= --spectral-max-gain, wavelengths below
                --spectral-cutoff-km; phases kept, mean unchanged)
  vpost_<s>     the post-hoc v10m guided filter, blended with strength s
Not implemented (need training): a real-vs-generated critic for importance
resampling / discriminator guidance (Kim et al. 2023), and ReNO/DNO noise optimization.
Automatic phase 2: the best feasible sampler recipe is also combined with
select_clim and spectral.

Ranking: sharpness gap G = mean of |log p99|grad| ratio|, |log fine-scale PSD
ratio| and |log edge step ratio| (members vs truth; 0 = truth-like, over-sharpening
is penalized too). Edge step: at the truth's ~24 sharpest edge points per field
(q2m, t2m, v10m, u10m, ps), the largest change across ~7.5 km within +-15 km of the
truth edge, member / truth (the dark-to-light line itself, not the broad ramp). A recipe is FEASIBLE if, vs baseline, the
mean CRPS change is <= --crps-tol % (and no field worse than --crps-tol-max %),
no field's |bias|/std grows by more than --bias-tol, and the seam index (edge
strength on tile boundaries vs everywhere) does not rise by more than 0.05.
One case is a screen: confirm the winner on the test set with
slurm_test_best_model_v4_1.sh / compare_sharpness.

Outputs (default <train.output>/evaluation/explore_<ckpt>_<case>_<job>/):
  report.md, metrics.json
  members_<var>.{png,pdf}   truth, coarse, member 1 of every recipe (region)
  edge_zoom.{png,pdf}       zoom on the truth's strongest q2m/t2m edges, local colour range
  edge_profiles.{png,pdf}   composite profiles across the truth's sharpest edges
  scorecard.{png,pdf}       sharpness gap, CRPS change, bias change, seams
  spectra.{png,pdf}         member PSD / truth PSD vs wavelength per field
  profiles.{png,pdf}        cross-front profiles (member 1 of each recipe)
  selection.{png,pdf}       truth-free selection score vs truth-based scores
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

from .config import write_json
from .dataset_v2 import crop_v2
from .inference import starts, blend_window
from .metrics import weighted_mean, crps_ensemble, radial_psd
from .train import device_for, autocast
from .v4 import TARGETS
from .v4_1 import load_config
from .evaluate_v4_1 import (resolve_checkpoint, load_model, load_guide, select_cases, member_seed, event_window,
                            make_window, make_time_steps, churn_time, DISPLAY, _display, _rain_norm)
from .diag_front_v4_1 import find_front, profile, slope_width

RAIN = TARGETS.index('precip')
FRONT_FIELDS = ('q2m', 't2m', 'v10m', 'ps')
EDGE_FIELDS = ('q2m', 't2m', 'v10m', 'u10m', 'ps')
EDGE_HALF_SPAN = 2   # px: local change across 2*2 px (~7.5 km at 1.9 km)
EDGE_SEARCH = 8      # px: a member's edge may sit up to ~15 km from truth's
EDGE_LENGTH = 16     # px: half-length of the edge profiles
FINE_BAND_KM = (4., 30.)
SEAM_TOL = .05

DEFAULTS = dict(kind='trajectory', steps=None, gamma=1., blend='hann', grids='A', churn=0., churn_range=(.1, .8),
                langevin=0., langevin_range=(.5, .97), restart=0, restart_t=.7, temp=1., hf=0., hf_sigma=2.,
                hf_start=.7, guide_weight=1., vguide=0., vguide_range=(.6, .97), sde=0., sde_range=(.2, .9),
                ag_hf=False, ag_sigma=8., ag_range=(.35, .85), restart_blend=None, restart_grids=None, then=(),
                fk_lambda=None, fk_reward='p99')
TRAJECTORY_IDS = ('baseline', 'hann2', 'hann3', 'hard', 'shift_hard', 'shift_hann2', 'shift4_hard', 'time_warp',
                  'churn', 'langevin', 'sde', 'restart', 'restart_shift', 'temp', 'hf_boost', 'autoguide',
                  'autoguide_hf', 'vguide', 'fk_steer', 'fk_edge')
GUIDED_IDS = ('autoguide', 'autoguide_hf')
GRID_OFFSETS = {'A': (0., 0.), 'B': (.5, .5), 'C': (.25, .75), 'D': (.75, .25)}   # fractions of the stride
POST_IDS = ('select_clim', 'select_sharp', 'prescreen', 'spectral', 'vpost')
METHOD_IDS = TRAJECTORY_IDS+POST_IDS


# ----------------------------------------------------------------------------
# Recipes
# ----------------------------------------------------------------------------

EXPANDED = ('sde', 'vguide', 'vpost')   # ids that expand to <id>_<strength>


def single_spec(part, o):
    """Spec of one recipe id, including expanded ids such as sde_1 or vguide_0.25."""
    if '@' in part:
        return build_methods((part,), o)[part]
    base, _, value = part.partition('_')
    if base in EXPANDED and value:
        key = {'sde': 'sde_strengths', 'vguide': 'vguide_strengths', 'vpost': 'vguide_strengths'}[base]
        return build_methods((base,), dict(o, **{key: (float(value),)}))[f'{base}_{float(value):g}']
    return build_methods((part,), o)[part]


def combine_methods(names, o):
    """Combined recipes, e.g. 'autoguide_hf+fk_steer+spectral': the sampling knobs of every
    sampler part together (FK steering if any part steers), then the post-processing parts
    applied in order to the combined members. Selection recipes cannot be combined."""
    o = dict(vars(o)) if not isinstance(o, dict) else o
    out = {}
    for name in names:
        parts = [q for q in name.split('+') if q]
        if len(parts) < 2:
            raise ValueError(f'A combination needs two or more recipes joined by "+": {name!r}')
        spec = dict(DEFAULTS, id=name, then=[], steps=o['steps'], source='baseline')
        labels = []
        for part in parts:
            s = single_spec(part, o)
            labels.append(s['label'])
            if s['kind'] == 'select':
                raise ValueError(f'{part} (member selection) cannot be combined; it has its own phase 2')
            if s['kind'] == 'post':
                spec['then'].append(s)
                continue
            if s['kind'] == 'fk':
                spec['kind'] = 'fk'
            for k, v in s.items():
                if k in DEFAULTS and k not in ('kind', 'then', 'steps') and v != DEFAULTS[k]:
                    spec[k] = v
        spec.update(label=' + '.join(labels), desc='combination: '+' + '.join(labels))
        out[name] = spec
    return out


def build_methods(ids, o):
    """Ordered {id: spec}. ``o``: options namespace/dict (CLI values)."""
    o = dict(vars(o)) if not isinstance(o, dict) else o
    table = {
        'baseline': dict(label='Baseline', desc='Heun, Hann blending (production)'),
        'hann2': dict(label='Hann²', desc='centre-weighted blending', blend='hann2'),
        'hann3': dict(label='Hann³', desc='strongly centre-weighted blending', blend='hann3'),
        'hard': dict(label='No blending', desc='nearest tile centre only', blend='hard'),
        'shift_hard': dict(label='Shifted, no blend', desc='alternating shifted grids, nearest tile only',
                           blend='hard', grids='AB'),
        'shift_hann2': dict(label='Shifted Hann²', desc='alternating shifted grids, Hann² blending',
                            blend='hann2', grids='AB'),
        'shift4_hard': dict(label='4 shifted grids, no blend', desc='four tile grids cycled, nearest tile only',
                            blend='hard', grids='ACBD'),
        'restart_shift': dict(label='Restart @0.5 shifted', desc='re-noise to t=0.5, re-solve on shifted grids',
                              restart=1, restart_t=.5, restart_blend='hard', restart_grids='ACBD'),
        'autoguide_hf': dict(label=f'HF autoguide w={o["guide_weight"]:g}',
                             desc='fine-scale, mid-t, projected autoguidance', guide_weight=o['guide_weight'],
                             ag_hf=True, ag_sigma=o['ag_sigma'], ag_range=tuple(o['ag_range'])),
        'fk_edge': dict(kind='fk', label=f'FK edge steering ×{o["fk_particles"]}',
                        desc='particles resampled toward climatological sharpness of the sharpest lines (p99.9 |∇|)',
                        fk_reward='edge', langevin=o['langevin'], langevin_range=(.2, .97)),
        'fk_steer': dict(kind='fk', label=f'FK steering ×{o["fk_particles"]}',
                         desc='particles resampled toward climatological sharpness',
                         langevin=o['langevin'], langevin_range=(.2, .97)),
        'time_warp': dict(label=f'Time warp γ={o["warp_gamma"]:g}', desc='finer late steps', gamma=o['warp_gamma']),
        'churn': dict(label=f'Churn {o["churn"]:g}', desc='EDM stochastic re-noising', churn=o['churn']),
        'langevin': dict(label=f'Langevin η={o["langevin"]:g}', desc='score-based corrector',
                         langevin=o['langevin'], langevin_range=tuple(o['langevin_range'])),
        'restart': dict(label=f'Restart {o["restart"]}×@{o["restart_t"]:g}', desc='re-noise and re-solve',
                        restart=o['restart'], restart_t=o['restart_t']),
        'temp': dict(label=f'Noise ×{o["temp"]:g}', desc='initial-noise temperature', temp=o['temp']),
        'hf_boost': dict(label=f'HF boost {o["hf_boost"]:g}', desc='high-pass velocity amplification',
                         hf=o['hf_boost'], hf_sigma=o['hf_sigma'], hf_start=o['hf_start']),
        'autoguide': dict(label=f'Autoguide w={o["guide_weight"]:g}', desc='earlier checkpoint as weak model',
                          guide_weight=o['guide_weight']),
        'select_clim': dict(kind='select', label='Select: climatology', desc=f'{o["members"]} of {o["pool"]} closest '
                            'to climatological sharpness', select='clim'),
        'select_sharp': dict(kind='select', label='Select: sharpest', desc=f'{o["members"]} sharpest of {o["pool"]}',
                             select='sharp'),
        'prescreen': dict(kind='select', label='Prescreen 8-step', desc='seeds ranked by an 8-step solve',
                          select='prescreen'),
        'spectral': dict(kind='post', label='Spectral fix', desc='fine-scale spectrum toward climatology',
                         post='spectral'),
    }
    methods = {}
    for name in ids:
        base, at, value = name.partition('@')
        if at:   # e.g. fk_steer@4: the recipe with FK strength lambda = 4
            spec = dict(build_methods((base,), o)[base], id=name)
            if spec['kind'] != 'fk':
                raise ValueError(f'"@lambda" applies to FK steering recipes only: {name!r}')
            spec.update(fk_lambda=float(value), label=f'{spec["label"]} λ={float(value):g}')
            methods[name] = spec
            continue
        stem, _, strength = name.partition('_')
        if stem in EXPANDED and strength and name not in table:   # e.g. vguide_0.5, sde_1, vpost_0.25
            key = {'sde': 'sde_strengths', 'vguide': 'vguide_strengths', 'vpost': 'vguide_strengths'}[stem]
            spec = build_methods((stem,), dict(o, **{key: (float(strength),)}))[f'{stem}_{float(strength):g}']
            methods[name] = dict(spec, id=name)
            continue
        if name == 'vguide':
            for s in o['vguide_strengths']:
                methods[f'vguide_{s:g}'] = dict(DEFAULTS, id=f'vguide_{s:g}', label=f'v10m guidance {s:g}',
                                                desc='in-sampler v10m-edge guidance', vguide=float(s),
                                                vguide_range=tuple(o['vguide_range']))
            continue
        if name == 'sde':
            for e in o['sde_strengths']:
                methods[f'sde_{e:g}'] = dict(DEFAULTS, id=f'sde_{e:g}', label=f'SDE ε={e:g}',
                                             desc='marginal-preserving SDE', sde=float(e),
                                             sde_range=tuple(o['sde_range']))
            continue
        if name == 'vpost':
            for s in o['vguide_strengths']:
                methods[f'vpost_{s:g}'] = dict(DEFAULTS, id=f'vpost_{s:g}', kind='post', post='vpost', strength=float(s),
                                               label=f'v10m post-filter {s:g}', desc='post-hoc v10m guided filter')
            continue
        if name not in table:
            raise ValueError(f'Unknown method {name!r}; choose from {", ".join(METHOD_IDS)}')
        methods[name] = dict(DEFAULTS, id=name, **table[name])
    if 'baseline' not in methods:
        methods = {'baseline': dict(DEFAULTS, id='baseline', **table['baseline']), **methods}
    for m in methods.values():
        m['steps'] = m['steps'] or o['steps']
        m.setdefault('source', 'baseline')
    return methods


# ----------------------------------------------------------------------------
# Torch filters
# ----------------------------------------------------------------------------

def gaussian_torch(x, sigma):
    """Separable Gaussian blur of (C, H, W) with replicate padding."""
    if sigma <= 0:
        return x
    r = max(1, int(round(3*sigma)))
    k = torch.exp(-.5*(torch.arange(-r, r+1, device=x.device, dtype=x.dtype)/sigma)**2)
    k = k/k.sum()
    c = x.shape[0]
    y = F.pad(x[None], (r, r, 0, 0), mode='replicate')
    y = F.conv2d(y, k.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c)
    y = F.pad(y, (0, 0, r, r), mode='replicate')
    return F.conv2d(y, k.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c)[0]


def guided_torch(p, guide, radius=8, eps=1e-2, sigma=2.):
    """Single-guide guided filter (He et al. 2010) of field ``p`` by ``guide`` (both H x W).

    Same design as diag_front_v4_1.guided_sharpen: the filtered field a*I + b takes the
    guide's edge shape with the field's own local amplitude; the field's texture finer
    than ``sigma`` is kept; the result is blended with the input by the local R^2, so
    where the field does not co-vary with the guide it is left unchanged.
    """
    p0 = p.double()
    ps = p0.std().clamp_min(1e-30)
    pm = p0.mean()
    q = (p0-pm)/ps
    I = guide.double()
    I = (I-I.mean())/I.std().clamp_min(1e-30)
    size = 2*radius+1

    def box(z):
        return F.avg_pool2d(F.pad(z[None, None], (radius,)*4, mode='replicate'), size, stride=1)[0, 0]
    mI, mq = box(I), box(q)
    cov = box(I*q)-mI*mq
    var = box(I*I)-mI*mI
    a = cov/(var+eps)
    b = mq-a*mI
    filtered = box(a)*I+box(b)
    r2 = box(torch.clamp(a*cov/torch.clamp(box(q*q)-mq*mq, min=1e-12), 0., 1.))
    residual = q-filtered
    texture = residual-gaussian_torch(residual[None], sigma)[0]
    out = r2*(filtered+texture)+(1-r2)*q
    return (out*ps+pm).to(p.dtype)


# ----------------------------------------------------------------------------
# Regional tiled sampler
# ----------------------------------------------------------------------------

def _shifted(values, offset, top):
    return sorted({values[0], *(min(v+offset, top) for v in values)})


class RegionEngine:
    """Synchronized tiled Heun sampler on the tiles covering ``region`` (+margin).

    Tile grids: A (the production grid) and B, C, D (A shifted by fractions of the
    stride, GRID_OFFSETS; built on demand). Blending modes: hann, hann2, hann3 (window powers) and hard (each pixel
    from the tile whose window is largest there, i.e. the nearest centre).
    """

    def __init__(self, model, conditioner, archive, entry, cfg, device, region, margin=96, batch=32, threads=8,
                 guide=None):
        p = cfg['patch']
        self.model, self.conditioner, self.archive, self.entry, self.cfg, self.device = \
            model, conditioner, archive, entry, cfg, device
        self.guide, self.batch, self.threads = guide, batch, threads
        self.size, self.halo, self.stride = p['size'], p['halo'], p['stride']
        self.width = self.size+2*self.halo
        self.h, self.w = archive.shape
        r0, r1, c0, c1 = region
        self.region = region
        rows = [y for y in starts(self.h, self.size, self.stride) if y < r1+margin and y+self.size > r0-margin]
        cols = [x for x in starts(self.w, self.size, self.stride) if x < c1+margin and x+self.size > c0-margin]
        self.rows, self.cols = {}, {}
        for name, (fy, fx) in GRID_OFFSETS.items():
            oy, ox = int(round(fy*self.stride)), int(round(fx*self.stride))
            self.rows[name] = _shifted(rows, oy, self.h-self.size) if oy else rows
            self.cols[name] = _shifted(cols, ox, self.w-self.size) if ox else cols
        ys = [y for v in self.rows.values() for y in v]
        xs = [x for v in self.cols.values() for x in v]
        self.Y0, self.Y1 = min(ys), max(ys)+self.width   # padded-grid box
        self.X0, self.X1 = min(xs), max(xs)+self.width
        self.coarse = np.asarray(archive.coarse(entry), dtype='float32')
        self.channels = archive.index['condition_channels']
        self.grids = {}
        self._weights = {}
        self.rs = conditioner.rs[0].cpu().numpy()
        self.rm = conditioner.rm[0].cpu().numpy()
        self.scale = conditioner.flow_scale[0].cpu().numpy()
        self.grid('A')
        window = torch.from_numpy(blend_window(self.width)).to(device)
        a = self.grids['A']
        blended = torch.zeros((len(TARGETS), self.Y1-self.Y0, self.X1-self.X0), device=device)
        weight = torch.zeros((1, self.Y1-self.Y0, self.X1-self.X0), device=device)
        for k, (y, x) in enumerate(a['local']):
            blended[:, y:y+self.width, x:x+self.width] += a['means'][k]*window
            weight[:, y:y+self.width, x:x+self.width] += window
        self.mean = (blended/weight.clamp_min(1e-12)).cpu().numpy()
        # Evaluation region and in-domain part of the box, in box coordinates.
        self.ey = slice(r0+self.halo-self.Y0, r1+self.halo-self.Y0)
        self.ex = slice(c0+self.halo-self.X0, c1+self.halo-self.X0)
        dy0, dy1 = max(self.halo, self.Y0), min(self.halo+self.h, self.Y1)
        dx0, dx1 = max(self.halo, self.X0), min(self.halo+self.w, self.X1)
        self.dy, self.dx = slice(dy0-self.Y0, dy1-self.Y0), slice(dx0-self.X0, dx1-self.X0)
        rows_u, cols_u = slice(dy0-self.halo, dy1-self.halo), slice(dx0-self.halo, dx1-self.halo)
        t = lambda a_: torch.from_numpy(np.ascontiguousarray(a_, dtype='float32')).to(device)
        self.t_scale, self.t_rs, self.t_rm = t(self.scale), t(self.rs), t(self.rm)
        self.t_mean_dom = t(self.mean[:, self.dy, self.dx])
        self.t_coarse_dom = t(self.coarse[:, rows_u, cols_u])

    @property
    def tile_count(self):
        return {k: len(g['local']) for k, g in self.grids.items()}

    def grid(self, name):
        if name in self.grids:
            return self.grids[name]
        p = self.cfg['patch']
        origins = [(y, x) for y in self.rows[name] for x in self.cols[name]]
        first = self.archive.inputs_with_original(self.entry, *origins[0], p)
        with ThreadPoolExecutor(max(1, self.threads)) as pool:
            rest = list(pool.map(lambda o: self.archive.inputs_with_original(self.entry, o[0], o[1], p), origins[1:]))
        inputs = [first, *rest]
        condition = torch.stack([i['condition'] for i in inputs]).to(self.device)
        context = torch.stack([i['context'] for i in inputs]).to(self.device)
        coarse = torch.from_numpy(np.stack([crop_v2(self.coarse, y, x, self.size, self.halo) for y, x in origins]))
        means = []
        with torch.no_grad():
            for s in range(0, len(origins), self.batch):
                b = dict(original_condition=condition[s:s+self.batch, :self.channels],
                         original_context=context[s:s+self.batch, :self.channels],
                         coarse=coarse[s:s+self.batch].to(self.device))
                with autocast(self.device, self.cfg['train']['precision']):
                    means.append(self.conditioner(b).float())
        self.grids[name] = dict(origins=origins, local=[(y-self.Y0, x-self.X0) for y, x in origins],
                                condition=condition, context=context, means=torch.cat(means))
        return self.grids[name]

    def weights(self, name, blend):
        key = (name, blend)
        if key in self._weights:
            return self._weights[key]
        g = self.grid(name)
        shape = (self.Y1-self.Y0, self.X1-self.X0)
        hann = torch.from_numpy(blend_window(self.width)).to(self.device)
        if blend == 'hard':
            best = torch.full(shape, -1., device=self.device)
            owner = torch.full(shape, -1, dtype=torch.long, device=self.device)
            for k, (y, x) in enumerate(g['local']):
                view = best[y:y+self.width, x:x+self.width]
                better = hann > view
                view[better] = hann[better]
                owner[y:y+self.width, x:x+self.width][better] = k
            tiles = torch.stack([(owner[y:y+self.width, x:x+self.width] == k).float()
                                 for k, (y, x) in enumerate(g['local'])])[:, None]
        else:
            tiles = torch.from_numpy(make_window(self.width, blend)).to(self.device)[None, None]
        norm = torch.zeros((1,)+shape, device=self.device)
        for k, (y, x) in enumerate(g['local']):
            norm[:, y:y+self.width, x:x+self.width] += tiles[k if tiles.shape[0] > 1 else 0]
        self._weights[key] = (tiles, norm)
        return self._weights[key]

    def noise(self, seed):
        full = np.random.default_rng(seed).standard_normal(
            (len(TARGETS), self.h+2*self.halo, self.w+2*self.halo)).astype('float32')
        return torch.from_numpy(np.ascontiguousarray(full[:, self.Y0:self.Y1, self.X0:self.X1])).to(self.device)

    @torch.no_grad()
    def velocity(self, x, t, spec, grid_name):
        g = self.grid(grid_name)
        tiles, norm = self.weights(grid_name, spec['blend'])
        guided = self.guide is not None and spec['guide_weight'] != 1. and not spec['ag_hf']
        hf_guided = (self.guide is not None and spec['ag_hf'] and spec['ag_range'][0] <= t <= spec['ag_range'][1])
        out = torch.zeros_like(x)
        w = self.width
        for s in range(0, len(g['local']), self.batch):
            chunk = g['local'][s:s+self.batch]
            xs = torch.stack([x[:, y:y+w, c:c+w] for y, c in chunk])
            tt = torch.full((len(chunk),), float(t), device=self.device)
            args = (xs, tt, g['condition'][s:s+len(chunk)], g['context'][s:s+len(chunk)], g['means'][s:s+len(chunk)])
            with autocast(self.device, self.cfg['train']['precision']):
                value = self.model(*args).float()
                if guided:
                    weak = self.guide(*args).float()
                    value = weak+spec['guide_weight']*(value-weak)
                if hf_guided:
                    value = value+(spec['guide_weight']-1)*self._hf_guidance(value, self.guide(*args).float(),
                                                                             spec['ag_sigma'])
            wt = tiles[s:s+len(chunk)] if tiles.shape[0] > 1 else tiles
            value = value*wt
            for k, (y, c) in enumerate(chunk):
                out[:, y:y+w, c:c+w] += value[k]
        out = torch.where(norm > 0, out/norm.clamp_min(1e-12), torch.zeros_like(out))
        if spec['hf'] > 0 and t >= spec['hf_start']:
            out = out+spec['hf']*(out-gaussian_torch(out, spec['hf_sigma']))
        return out

    @staticmethod
    def _hf_guidance(value, weak, sigma):
        """High-pass part of (value - weak), minus its component along value (APG), per tile/channel."""
        delta = value-weak
        n, c, hh, ww = delta.shape
        high = delta-gaussian_torch(delta.reshape(n*c, hh, ww), sigma).reshape(delta.shape)
        along = (high*value).sum((-2, -1), keepdim=True)/(value*value).sum((-2, -1), keepdim=True).clamp_min(1e-12)
        return high-along*value

    def vguide_step(self, x, v, t, weight, fields, radius, eps, sigma):
        """Nudge the state toward the member's v10m-edge-sharpened clean estimate."""
        x1 = (x+(1-t)*v)[:, self.dy, self.dx]
        phys = (x1*self.t_scale+self.t_mean_dom)*self.t_rs+self.t_rm+self.t_coarse_dom
        guide = phys[TARGETS.index('v10m')]
        for name in fields:
            c = TARGETS.index(name)
            delta = guided_torch(phys[c], guide, radius, eps, sigma)-phys[c]
            x[c, self.dy, self.dx] += weight*t*delta/(self.t_rs[c]*self.t_scale[c])
        return x

    @torch.no_grad()
    def _heun(self, x, times, spec, rng, opts, counter):
        for i in range(len(times)-1):
            grid = spec['grids'][counter[0] % len(spec['grids'])]
            counter[0] += 1
            t_cur, t_next = times[i], times[i+1]
            if spec['churn'] > 0 and 0 < t_cur and spec['churn_range'][0] <= t_cur <= spec['churn_range'][1]:
                t_hat = churn_time(t_cur, spec['churn'])
                sigma, sigma_hat = (1-t_cur)/t_cur, (1-t_hat)/t_hat
                fresh = torch.from_numpy(rng.standard_normal(x.shape, dtype=np.float32)).to(self.device)
                x = x*(t_hat/t_cur)+fresh*(t_hat*float(np.sqrt(sigma_hat**2-sigma**2)))
                t_cur = t_hat
            dt = t_next-t_cur
            first = self.velocity(x, t_cur, spec, grid)
            second = self.velocity(x+first*dt, t_next, spec, grid)
            start = x
            x = x+(first+second)*(dt/2)
            lo, hi = spec['sde_range']
            if spec['sde'] > 0 and lo <= t_cur <= hi:
                # Marginal-preserving SDE: extra drift -e*z (z = predicted noise) and noise sqrt(2e(1-t)dt).
                z = start-t_cur*first
                fresh = torch.from_numpy(rng.standard_normal(x.shape, dtype=np.float32)).to(self.device)
                x = x-spec['sde']*z*dt+float(np.sqrt(2*spec['sde']*(1-t_cur)*dt))*fresh
            lo, hi = spec['langevin_range']
            if spec['langevin'] > 0 and lo <= t_next <= hi and t_next < 1:
                v = self.velocity(x, t_next, spec, grid)
                score = -(x-t_next*v)/(1-t_next)
                delta = .5*(spec['langevin']*(1-t_next))**2
                fresh = torch.from_numpy(rng.standard_normal(x.shape, dtype=np.float32)).to(self.device)
                x = x+delta*score+float(np.sqrt(2*delta))*fresh
            lo, hi = spec['vguide_range']
            if spec['vguide'] > 0 and lo <= t_next <= hi and t_next < 1:
                # Strength spread over the guided interval: the steps' weights sum to spec['vguide'].
                weight = spec['vguide']*(t_next-t_cur)/max(hi-lo, 1e-6)
                x = self.vguide_step(x, second, t_next, weight, opts['vguide_fields'], opts['vguide_radius'],
                                     opts['vguide_eps'], opts['vguide_sigma'])
            if not bool(torch.isfinite(x).all()):
                raise FloatingPointError(f'Nonfinite trajectory ({spec["id"]})')
        return x

    @torch.no_grad()
    def integrate(self, seed, spec, opts=None):
        opts = opts or {}
        rng = np.random.default_rng(np.random.SeedSequence([seed, 7919]))
        x = self.noise(seed)*float(spec['temp'])
        counter = [0]
        x = self._heun(x, make_time_steps(spec['steps'], spec['gamma']), spec, rng, opts, counter)
        for _ in range(int(spec['restart'])):
            t_r = float(spec['restart_t'])
            fresh = torch.from_numpy(rng.standard_normal(x.shape, dtype=np.float32)).to(self.device)
            x = t_r*x+(1-t_r)*fresh
            n = max(2, int(round(spec['steps']*(1-t_r))))
            stage = dict(spec, churn=0., langevin=0., sde=0., blend=spec['restart_blend'] or spec['blend'],
                         grids=spec['restart_grids'] or spec['grids'])
            x = self._heun(x, list(np.linspace(t_r, 1., n+1)), stage, rng, opts, counter)
        return x

    @torch.no_grad()
    def fk_sample(self, seed, spec, opts, reward, particles=4, lam=10., times=(.3, .5, .7, .85)):
        """Feynman-Kac steering: resample particles by exp(lam * reward increment) of their
        clean estimates at ``times``; return (physical sample, effective sample sizes)."""
        seeds = [seed]+[int(np.random.SeedSequence([seed, 101, j]).generate_state(1)[0]) for j in range(1, particles)]
        states = [self.noise(s)*float(spec['temp']) for s in seeds]
        rngs = [np.random.default_rng(np.random.SeedSequence([s, 7919])) for s in seeds]
        counters = [[0] for _ in seeds]
        choose = np.random.default_rng(np.random.SeedSequence([seed, 303]))
        schedule = make_time_steps(spec['steps'], spec['gamma'])
        pending = sorted(times)
        previous = np.zeros(particles)
        ess = []

        def rewards(t):
            values = []
            for j, x in enumerate(states):
                v = self.velocity(x, t, spec, spec['grids'][counters[j][0] % len(spec['grids'])])
                values.append(reward(self.decode(x+(1-t)*v)))
            return np.array(values)
        for i in range(len(schedule)-1):
            for j in range(particles):
                states[j] = self._heun(states[j], schedule[i:i+2], spec, rngs[j], opts, counters[j])
            t = schedule[i+1]
            if pending and t >= pending[0]:
                while pending and t >= pending[0]:
                    pending.pop(0)
                r = rewards(t)
                logw = self._fk_logw(r-previous, lam)
                w = np.exp(logw-logw.max())
                w /= w.sum()
                ess.append(float(1/np.sum(w**2)))
                pick = choose.choice(particles, size=particles, p=w)
                states = [states[k].clone() for k in pick]
                previous = r[pick]
                counters = [[counters[k][0]] for k in pick]
                # Resampled copies need independent noise from here on.
                rngs = [np.random.default_rng(np.random.SeedSequence([seed, 7, i, j])) for j in range(particles)]
        final = [self.decode(x) for x in states]
        r = np.array([reward(f) for f in final])
        logw = self._fk_logw(r-previous, lam)
        w = np.exp(logw-logw.max())
        w /= w.sum()
        return final[int(choose.choice(particles, p=w))], ess

    @staticmethod
    def _fk_logw(increment, lam):
        """Log-weights from reward increments standardized across the particles: lambda is in
        units of the between-particle spread, so it gives real selection pressure whatever the
        reward's scale (raw increments of a region-mean reward are ~1e-2 and left the weights
        uniform). lambda=1: the best of 4 particles is typically ~2-3x as likely as the worst
        pair average; lambda=3: strongly favours the best."""
        spread = float(np.std(increment))
        return np.zeros_like(increment) if spread < 1e-12 else lam*(increment-np.mean(increment))/spread

    def decode(self, x):
        """Box state -> physical fields (6, R, C) on the evaluation region."""
        core = x[:, self.ey, self.ex].cpu().numpy() if torch.is_tensor(x) else x[:, self.ey, self.ex]
        mean = self.mean[:, self.ey, self.ex]
        r0, r1, c0, c1 = self.region
        value = (core*self.scale+mean)*self.rs+self.rm+self.coarse[:, r0:r1, c0:c1]
        z = np.maximum(core[RAIN], 0)
        value[RAIN] = self.archive.scale*z*(z+2)
        value[5] = np.clip(value[5], 0, 1)
        if not np.isfinite(value).all():
            raise FloatingPointError('Nonfinite physical output')
        return value.astype('float32')

    def sample(self, seed, spec, opts=None):
        return self.decode(self.integrate(seed, spec, opts))


# ----------------------------------------------------------------------------
# Sharpness, climatology, selection, post-processing
# ----------------------------------------------------------------------------

def sharp_view(c, field):
    """Field used for sharpness scores: rain in sqrt space (edges of all intensities)."""
    return np.sqrt(np.maximum(field, 0)) if c == RAIN else np.asarray(field, dtype='float64')


def p99_grad(field, q=.99):
    gy, gx = np.gradient(np.asarray(field, dtype='float64'))
    return float(np.quantile(np.hypot(gy, gx), q))


def member_p99(member, q=.99):
    return np.array([p99_grad(sharp_view(c, member[c]), q) for c in range(len(TARGETS))])


def _doy_distance(a, b):
    d = abs(a.timetuple().tm_yday-b.timetuple().tm_yday)
    return min(d, 365-d)


def climatology(archive, entry, region, count=24, days=45, seed=11, log=print):
    """Truth sharpness climatology over ``region`` from training hours near the case's date.

    Returns p99|grad| ratio truth/coarse per field (median over hours) and the median
    truth radial PSD per field, both truth-free with respect to the case itself.
    """
    r0, r1, c0, c1 = region
    when = datetime.fromisoformat(entry['time'][:19])
    pool = [e for e in archive.index['entries'] if e.get('split') == 'train']
    if not pool:
        pool = [e for e in archive.index['entries'] if e['id'] != entry['id'] and e.get('split') == entry.get('split')]
    near = [e for e in pool if _doy_distance(datetime.fromisoformat(e['time'][:19]), when) <= days]
    if len(near) < count:
        near = sorted(pool, key=lambda e: _doy_distance(datetime.fromisoformat(e['time'][:19]), when))[:4*count]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(near))
    ratios, ratios999, spectra, used = [], [], [], []
    for j in order:
        e = near[j]
        try:
            truth = np.asarray(archive.physical_truth(e), dtype='float32')[:, r0:r1, c0:c1]
            coarse = np.asarray(archive.coarse(e), dtype='float32')[:, r0:r1, c0:c1]
        except Exception:  # noqa: BLE001 - missing hour/history: skip
            continue
        t = member_p99(truth)
        c = member_p99(coarse)
        ratios.append(t/np.maximum(c, 1e-30))
        ratios999.append(member_p99(truth, .999)/np.maximum(member_p99(coarse, .999), 1e-30))
        spectra.append([radial_psd(sharp_view(k, truth[k]))[1] for k in range(len(TARGETS))])
        used.append(e['time'])
        if len(used) == count:
            break
    if not used:
        raise ValueError('No climatology hours available')
    freq = radial_psd(np.zeros((r1-r0, c1-c0)))[0]
    log(f'Climatology: {len(used)} hours within ~{days} days of {when:%m-%d} ({used[0][:10]} ...)')
    return dict(ratio=np.median(np.array(ratios), 0), ratio999=np.median(np.array(ratios999), 0), psd=np.median(np.array(spectra), 0), freq=freq, times=used)


def selection_scores(members, coarse, clim):
    """Truth-free sharpness scores per member: closeness to the climatological target
    (lower = better) and one-sided sharpness (higher = sharper)."""
    target = clim['ratio']*member_p99(coarse)
    logs = np.log(np.array([member_p99(m) for m in members])/np.maximum(target, 1e-30))
    return np.abs(logs).mean(1), logs.mean(1)


def spectral_fix(member, clim, dx_km, max_gain=1.5, cutoff_km=40.):
    """Raise fine-scale Fourier amplitudes toward the climatological truth spectrum.

    Gain = sqrt(target/member PSD), clipped to [1, max_gain], only for wavelengths below
    ``cutoff_km`` (smooth ramp from cutoff to cutoff/2). Phases and the mean are kept.
    """
    out = np.array(member, copy=True)
    h, w = member.shape[-2:]
    # Mirror-extend to 2h x 2w: the extension is periodic and continuous, so a large-scale
    # trend (e.g. surface pressure over terrain) does not become a spurious edge jump.
    fy, fx = np.fft.fftfreq(2*h), np.fft.rfftfreq(2*w)
    radius = np.sqrt(fy[:, None]**2+fx[None, :]**2)
    wavelength = dx_km/np.maximum(radius, 1e-12)
    ramp = np.clip((cutoff_km-wavelength)/(cutoff_km/2), 0, 1)
    for c in range(len(TARGETS)):
        f = sharp_view(c, member[c])
        freq, psd = radial_psd(f)
        gain_1d = np.sqrt(np.divide(clim['psd'][c], psd, out=np.ones_like(psd), where=psd > 0))
        gain_1d = np.clip(gain_1d, 1., max_gain)
        gain = np.interp(radius, freq, gain_1d)
        gain = 1+(gain-1)*ramp
        gain[0, 0] = 1.
        mean = f.mean()
        spec = np.fft.rfft2(np.pad(f-mean, ((0, h), (0, w)), mode='symmetric'))
        g = np.fft.irfft2(spec*gain, s=(2*h, 2*w))[:h, :w]+mean
        out[c] = np.maximum(g, 0)**2 if c == RAIN else g
    out[5] = np.clip(out[5], 0, 1)
    return out.astype('float32')


def vpost(member, strength, fields, radius, eps, sigma):
    out = np.array(member, copy=True)
    guide = torch.from_numpy(np.asarray(member[TARGETS.index('v10m')], dtype='float64'))
    for name in fields:
        c = TARGETS.index(name)
        p = torch.from_numpy(np.asarray(member[c], dtype='float64'))
        sharp = guided_torch(p, guide, radius, eps, sigma).numpy()
        out[c] = member[c]+strength*(sharp-member[c])
    return out.astype('float32')


# ----------------------------------------------------------------------------
# Scores
# ----------------------------------------------------------------------------

def seam_index(field, boundaries):
    """Mean |jump| across tile-boundary lines / mean |jump| everywhere (1 = no seams)."""
    rows, cols = boundaries
    f = np.asarray(field, dtype='float64')
    dy, dx = np.abs(np.diff(f, axis=0)), np.abs(np.diff(f, axis=1))
    parts = [dy[b-1] for b in rows if 1 <= b < f.shape[0]]+[dx[:, b-1] for b in cols if 1 <= b < f.shape[1]]
    if not parts:
        return 1.
    return float(np.mean(np.concatenate(parts))/max(np.mean(np.concatenate([dy.ravel(), dx.ravel()])), 1e-30))


def tile_boundaries(engine, grids='A'):
    """Row/col lines between neighbouring tile cores (nearest-centre switch lines),
    in evaluation-region coordinates, for the given grids."""
    r0, r1, c0, c1 = engine.region
    rows, cols = set(), set()
    half = engine.size/2
    for g in grids:
        ys, xs = engine.rows[g], engine.cols[g]
        rows |= {int(round((a+b)/2+half))-r0 for a, b in zip(ys[:-1], ys[1:])}
        cols |= {int(round((a+b)/2+half))-c0 for a, b in zip(xs[:-1], xs[1:])}
    return sorted(rows), sorted(cols)


def fine_psd_ratio(members, truth, dx_km, band=FINE_BAND_KM):
    freq, t = radial_psd(truth)
    m = np.mean([radial_psd(x)[1] for x in members], 0)
    wl = dx_km/np.maximum(freq, 1e-12)
    keep = (wl >= band[0]) & (wl <= band[1]) & (t > 0) & (m > 0)
    return float(np.exp(np.mean(np.log(m[keep]/t[keep])))) if keep.any() else None


def method_grids(spec, specs):
    """Tile grids whose boundaries a recipe's members can carry seams on."""
    if spec['kind'] in ('select', 'post'):
        return method_grids(specs.get(spec.get('source', 'baseline'), specs['baseline']), specs)
    return ''.join(sorted(set(spec['grids']+(spec.get('restart_grids') or ''))))


def truth_edges(field, count=24, quantile=.995, sigma=1., spacing=10, margin=EDGE_LENGTH+EDGE_SEARCH+2):
    """The truth's sharpest edge points (top ``quantile`` of the lightly smoothed gradient),
    at least ``spacing`` px apart, with the unit normal pointing toward higher values."""
    f = np.asarray(field, dtype='float64')
    f = gaussian_filter(f, sigma) if sigma > 0 else f
    gy, gx = np.gradient(f)
    g = np.hypot(gy, gx)
    h, w = g.shape
    inner = np.zeros_like(g)
    if h > 2*margin and w > 2*margin:
        inner[margin:h-margin, margin:w-margin] = g[margin:h-margin, margin:w-margin]
    if not (inner > 0).any():
        return []
    threshold = np.quantile(inner[inner > 0], quantile)
    points = []
    for i in np.argsort(inner, axis=None)[::-1]:
        if inner.flat[i] < threshold or len(points) == count:
            break
        y, x = divmod(int(i), w)
        if all((y-py)**2+(x-px)**2 >= spacing**2 for (py, px), _ in points):
            n = float(g[y, x]) or 1.
            points.append(((y, x), (float(gy[y, x])/n, float(gx[y, x])/n)))
    return points


def local_step(values, half=EDGE_HALF_SPAN, search=EDGE_SEARCH):
    """Largest change across 2*half px (in the truth's uphill direction) within +-search px of
    the profile centre: how much of the jump happens in a few km, allowing a small displacement."""
    c = len(values)//2
    return max(values[i+half]-values[i-half] for i in range(c-search, c+search+1))


def edge_scores(members, truth, edges, length=EDGE_LENGTH):
    """Edge contrast ratio (member local step / truth local step, median over the truth's edge
    points, mean over members; 1 = as sharp as truth) and composite profiles normalized by each
    point's truth jump (for plots)."""
    if not edges:
        return None, None
    ratios, composite, truth_composite = [], [], []
    for point, normal in edges:
        _, tp = profile(truth, point, normal, length, 2)
        step = local_step(tp)
        jump = float(np.mean(tp[-3:])-np.mean(tp[:3]))
        if step <= 0 or jump <= 0:
            continue
        rows = [profile(m, point, normal, length, 2)[1] for m in members]
        ratios.append([local_step(r)/step for r in rows])
        truth_composite.append((tp-tp.mean())/jump)
        composite.append((np.mean(rows, 0)-tp.mean())/jump)
    if not ratios:
        return None, None
    per_member = np.median(np.array(ratios), axis=0)
    return float(np.mean(per_member)), dict(truth=np.mean(truth_composite, 0).tolist(),
                                            members=np.mean(composite, 0).tolist(), per_member=per_member.tolist())


def score_ensemble(ens, truth, area, dx_km, front, boundaries, edges=None):
    """Per-field scores of an ensemble (M, 6, R, C) against truth (6, R, C)."""
    out = {}
    m = len(ens)
    for c, name in enumerate(TARGETS):
        e, t = ens[:, c].astype('float64'), truth[c].astype('float64')
        mean = e.mean(0)
        rmse = float(np.sqrt(weighted_mean((mean-t)**2, area)))
        spread = float(np.sqrt(weighted_mean(e.var(0, ddof=1), area))) if m > 1 else 0.
        sv = [sharp_view(c, x) for x in e]
        tv = sharp_view(c, t)
        bias = float(weighted_mean(mean-t, area))
        # A (near-)constant truth field has no meaningful normalized bias or sharpness ratio.
        flat = float(t.std()) <= 1e-6*(abs(float(t.mean()))+1e-12)
        tp99 = p99_grad(tv)
        out[name] = dict(
            crps=float(weighted_mean(crps_ensemble(e, t), area)), bias=bias, rmse=rmse,
            mae=float(weighted_mean(np.abs(mean-t), area)),
            bias_norm=None if flat else abs(bias)/float(t.std()),
            spread_skill=spread*np.sqrt((m+1)/m)/max(rmse, 1e-30) if m > 1 else None,
            p99_ratio=None if flat or tp99 <= 0 else float(np.mean([p99_grad(x) for x in sv])/tp99),
            fine_psd_ratio=None if flat else fine_psd_ratio(sv, tv, dx_km),
            seam=float(np.mean([seam_index(x, boundaries) for x in sv])), truth_seam=seam_index(tv, boundaries))
        if edges and name in edges:
            out[name]['edge_ratio'], out[name]['edge_profile'] = edge_scores(e, t, edges[name])
        if front is not None and name in FRONT_FIELDS:
            point, normal, length, band = front
            s, tp = profile(t, point, normal, length, band)
            widths = [slope_width(s, profile(x, point, normal, length, band)[1], dx_km) for x in e]
            widths = [w for w in widths if w is not None]
            out[name]['truth_width_km'] = slope_width(s, tp, dx_km)
            out[name]['member_width_km'] = float(np.median(widths)) if widths else None
    return out


def sharpness_gap(scores):
    """Mean |log| distance of member sharpness from truth: 0 = truth-like."""
    p99 = [abs(np.log(max(f['p99_ratio'], 1e-9))) for f in scores.values() if f['p99_ratio'] is not None]
    psd = [abs(np.log(f['fine_psd_ratio'])) for f in scores.values() if f['fine_psd_ratio']]
    edge = [abs(np.log(max(f['edge_ratio'], 1e-3))) for f in scores.values() if f.get('edge_ratio')]
    parts = dict(p99=float(np.mean(p99)) if p99 else None, psd=float(np.mean(psd)) if psd else None,
                 edge=float(np.mean(edge)) if edge else None)
    used = [v for v in parts.values() if v is not None]
    return float(np.mean(used)), parts


def judge(scores, base, crps_tol, crps_tol_max, bias_tol):
    """Feasibility vs baseline and the reasons when not feasible."""
    change = {k: 100*(scores[k]['crps']/max(base[k]['crps'], 1e-30)-1) for k in scores}
    reasons = []
    if np.mean(list(change.values())) > crps_tol:
        reasons.append(f'mean CRPS +{np.mean(list(change.values())):.1f}%')
    worst = max(change, key=change.get)
    if change[worst] > crps_tol_max:
        reasons.append(f'{worst} CRPS +{change[worst]:.1f}%')
    for k in scores:
        if scores[k]['bias_norm'] is None or base[k]['bias_norm'] is None:
            continue
        if scores[k]['bias_norm']-base[k]['bias_norm'] > bias_tol:
            reasons.append(f'{k} |bias| +{scores[k]["bias_norm"]-base[k]["bias_norm"]:.3f} std')
    # Seam excess over truth on the recipe's own tile boundaries (truth has no seams).
    seam = np.mean([f['seam']-f['truth_seam'] for f in scores.values()])
    base_seam = np.mean([f['seam']-f['truth_seam'] for f in base.values()])
    if seam-base_seam > SEAM_TOL:
        reasons.append(f'seam excess {seam:+.2f} vs {base_seam:+.2f}')
    return not reasons, reasons, change


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------

OPTION_DEFAULTS = dict(members=8, pool=16, steps=64, region=384, margin=96, clim_count=24, clim_days=45,
                       warp_gamma=2., churn=.2, langevin=.3, langevin_range=(.5, .97), restart=2, restart_t=.7,
                       temp=1.1, hf_boost=.3, hf_sigma=2., hf_start=.7, guide_checkpoint='auto', guide_weight=1.5,
                       vguide_strengths=(.25, .5, 1.), vguide_range=(.6, .97), vguide_fields=('t2m', 'q2m'),
                       vguide_radius=8, vguide_eps=1e-2, vguide_sigma=2., spectral_max_gain=1.5,
                       spectral_cutoff_km=40., prescreen_steps=8, sde_strengths=(.5, 1., 2.), sde_range=(.2, .9),
                       ag_sigma=8., ag_range=(.35, .85), combine=(), phase2=True, fk_particles=4, fk_lambda=2., fk_times=(.3, .5, .7, .85), crps_tol=1., crps_tol_max=3., bias_tol=.02,
                       profile_length=60, profile_band=24)


def run(cfg, checkpoint='latest', timestamp=None, split='val', methods=METHOD_IDS, center=None, center_latlon=None,
        anywhere=False, weights='ema', output=None, batch=32, threads=8, dpi=200, pdf=True, save_members=False,
        log=print, **options):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    o = dict(OPTION_DEFAULTS, **options)
    o['pool'] = max(o['pool'], o['members'])
    started = time.monotonic()
    path = resolve_checkpoint(cfg, checkpoint)
    device = device_for(cfg['train']['device'])
    archive, model, conditioner, saved = load_model(cfg, path, device, weights)
    cases = select_cases(archive, split, [timestamp] if timestamp else None, 0, 0 if timestamp else 1, 317, log)
    entry = cases[0]['entry']
    job = os.environ.get('SLURM_JOB_ID') or time.strftime('%Y%m%d_%H%M%S')
    out = Path(output) if output else (Path(cfg['train']['output'])/'evaluation'/
                                      f'explore_{path.stem}{"_raw" if weights == "raw" else ""}_{entry["id"]}_{job}')
    out.mkdir(parents=True, exist_ok=True)
    h, w = archive.shape
    area_full = np.asarray(archive.static['area'], dtype='float64')
    dx_km = float(np.sqrt(np.median(area_full))/1000)
    truth_full = np.asarray(archive.physical_truth(entry), dtype='float32')

    # Front and evaluation region
    if center is None and center_latlon is not None:
        lat, lon = np.asarray(archive.static['lat']), np.asarray(archive.static['lon'])
        dist = (lat-center_latlon[0])**2+((lon-center_latlon[1])*np.cos(np.deg2rad(center_latlon[0])))**2
        center = tuple(int(v) for v in np.unravel_index(np.argmin(dist), dist.shape))
    p = cfg['patch']
    width = p['size']+2*p['halo']
    search = None if anywhere or center is not None else event_window(truth_full[RAIN], min(384, h, w))
    _, point, normal = find_front(truth_full, archive.shape, min(width, h, w), 0, center, region=search)
    half = o['region']//2
    r0 = int(np.clip(point[0]-half, 0, max(0, h-o['region'])))
    c0 = int(np.clip(point[1]-half, 0, max(0, w-o['region'])))
    region = (r0, min(h, r0+o['region']), c0, min(w, c0+o['region']))
    local = (point[0]-r0, point[1]-c0)
    length = int(min(o['profile_length'], (region[1]-region[0])//3, (region[3]-region[2])//3))
    band = int(min(o['profile_band'], length))
    front = (local, normal, length, band)
    rows, cols = slice(region[0], region[1]), slice(region[2], region[3])
    truth, area = truth_full[:, rows, cols], area_full[rows, cols]
    coarse = np.asarray(archive.coarse(entry), dtype='float32')[:, rows, cols]

    spec_methods = build_methods(methods, o)
    guide = None
    spec_methods.update(combine_methods(o['combine'], o))
    guided_ids = [k for k, m in spec_methods.items() if m['guide_weight'] != 1.]
    if guided_ids:
        guide = load_guide(cfg, o['guide_checkpoint'], archive, saved, device, log)
        if guide is None:
            log(f'autoguidance recipes skipped (no earlier kept checkpoint in this run): {", ".join(guided_ids)}')
            for k in guided_ids:
                spec_methods.pop(k)
    engine = RegionEngine(model, conditioner, archive, entry, cfg, device, region, o['margin'], batch, threads,
                          guide['model'] if guide else None)
    log(f'Case {entry["id"]} ({cases[0]["reason"]}); {path.name} (epoch {saved["epoch"]+1}, {weights}); '
        f'front at {point}; region rows {region[0]}:{region[1]} cols {region[2]}:{region[3]} '
        f'({(region[1]-region[0])*dx_km:.0f}×{(region[3]-region[2])*dx_km:.0f} km); {engine.tile_count["A"]} tiles; '
        f'{len(spec_methods)} recipes, {o["members"]} members (pool {o["pool"]})')
    clim = climatology(archive, entry, region, o['clim_count'], o['clim_days'], log=log)

    seeds = [member_seed(cfg, entry, m) for m in range(o['pool'])]
    ensembles, seconds = {}, {}
    extras = {}

    target = clim['ratio']*member_p99(coarse)

    target999 = clim['ratio999']*member_p99(coarse, .999)
    edge_idx = [TARGETS.index(n) for n in EDGE_FIELDS]

    def reward(fields):
        """Truth-free: minus the mean |log| distance of p99|grad| from the climatological target."""
        return -float(np.mean(np.abs(np.log(member_p99(fields)/np.maximum(target, 1e-30)))))

    def reward_edge(fields):
        """Truth-free, sharp lines only: p99.9 |grad| of the edge fields vs the climatological target."""
        value = np.array([p99_grad(fields[c], .999) for c in edge_idx])
        return -float(np.mean(np.abs(np.log(value/np.maximum(target999[edge_idx], 1e-30)))))
    fk_ess = []

    def solve(spec, count):
        t0 = time.monotonic()
        members = []
        for k in range(count):
            if spec['kind'] == 'fk':
                member, ess = engine.fk_sample(seeds[k], spec, o, reward_edge if spec['fk_reward'] == 'edge' else reward,
                                               o['fk_particles'], spec['fk_lambda'] or o['fk_lambda'],
                                               tuple(o['fk_times']))
                fk_ess.append(ess)
                members.append(member)
            else:
                members.append(engine.sample(seeds[k], spec, o))
        seconds[spec['id']] = (time.monotonic()-t0)/count
        return np.stack(members)

    trajectory = [m for m in spec_methods.values() if m['kind'] in ('trajectory', 'fk')]
    solved = {}   # identical sampling knobs (e.g. a combination = recipe + post-processing) reuse members
    for spec in trajectory:
        count = o['pool'] if spec['id'] == 'baseline' else o['members']
        key = repr([(k, spec[k]) for k in sorted(DEFAULTS) if k != 'then'])
        if key in solved and len(ensembles[solved[key]]) >= min(count, o['members']) and spec['id'] != 'baseline':
            ens = ensembles[solved[key]][:count].copy()
            seconds[spec['id']] = seconds[solved[key]]
            ensembles[spec['id']] = ens
            log(f'  {spec["id"]:>14}: reuses the members of {solved[key]}')
            continue
        solved.setdefault(key, spec['id'])
        ens = solve(spec, count)
        if spec['id'] == 'baseline':
            extras['baseline_pool'] = ens
            ens = ens[:o['members']]
        ensembles[spec['id']] = ens
        log(f'  {spec["id"]:>14}: {seconds[spec["id"]]:.1f} s/member')

    pool = extras['baseline_pool']
    closeness, sharp = selection_scores(pool, coarse, clim)
    pool_truth_p99 = [float(np.mean(member_p99(mm)/np.maximum(member_p99(truth), 1e-30))) for mm in pool]
    pool_crps = [float(np.mean([weighted_mean(np.abs(mm[c]-truth[c]), area)/max(float(truth[c].std()), 1e-30)
                                for c in range(len(TARGETS))])) for mm in pool]
    selection = dict(closeness=closeness.tolist(), sharpness=sharp.tolist(), truth_p99_ratio=pool_truth_p99,
                     normalized_mae=pool_crps)
    if 'prescreen' in spec_methods:
        cheap = dict(spec_methods['baseline'], steps=o['prescreen_steps'], id='prescreen_solve')
        t0 = time.monotonic()
        quick = np.stack([engine.sample(s, cheap, o) for s in seeds])
        # Cost per kept member: one full solve plus the pool's cheap solves shared among the kept ones.
        seconds['prescreen'] = seconds['baseline']+(time.monotonic()-t0)/o['members']
        selection['prescreen_closeness'] = selection_scores(quick, coarse, clim)[0].tolist()
        from scipy.stats import spearmanr
        selection['prescreen_rank_correlation'] = float(spearmanr(selection['prescreen_closeness'], closeness)[0])
    chosen = {}

    def post(spec, source):
        if spec.get('select'):
            pool_ = extras.get(f'{source}_pool', pool)
            if spec['select'] == 'prescreen':
                key = np.array(selection['prescreen_closeness'])
            else:
                near, sharpness = selection_scores(pool_, coarse, clim)
                key = near if spec['select'] == 'clim' else -sharpness
            pick = np.sort(np.argsort(key, kind='stable')[:o['members']])
            chosen[spec['id']] = pick.tolist()
            seconds.setdefault(spec['id'], seconds[source]*o['pool']/o['members'])
            return pool_[pick]
        seconds.setdefault(spec['id'], seconds[source])
        return apply_post(spec, ensembles[source])

    def apply_post(spec, members):
        if spec['post'] == 'spectral':
            return np.stack([spectral_fix(mm, clim, dx_km, o['spectral_max_gain'], o['spectral_cutoff_km'])
                             for mm in members])
        return np.stack([vpost(mm, spec['strength'], o['vguide_fields'], o['vguide_radius'], o['vguide_eps'],
                               o['vguide_sigma']) for mm in members])

    for spec in trajectory:   # combinations: post-processing parts on their own members
        for step in spec.get('then', ()):
            ensembles[spec['id']] = apply_post(step, ensembles[spec['id']])

    for spec in spec_methods.values():
        if spec['kind'] not in ('trajectory', 'fk'):
            ensembles[spec['id']] = post(spec, 'baseline')
    if fk_ess:
        selection['fk_mean_ess'] = [float(np.mean([e[i] for e in fk_ess if len(e) > i]))
                                    for i in range(max(len(e) for e in fk_ess))]
        log('  FK steering mean effective sample size per resampling: '
            + ', '.join(f'{v:.2f}' for v in selection['fk_mean_ess']) + f' of {o["fk_particles"]}')

    edges = {name: truth_edges(truth[TARGETS.index(name)]) for name in EDGE_FIELDS}
    log('Truth edge points per field: ' + ', '.join(f'{k} {len(v)}' for k, v in edges.items()))

    def evaluate_all():
        scores = {k: score_ensemble(v, truth, area, dx_km, front,
                                    tile_boundaries(engine, method_grids(spec_methods[k], spec_methods)), edges)
                  for k, v in ensembles.items()}
        base = scores['baseline']
        table = {}
        for k, s in scores.items():
            gap, parts = sharpness_gap(s)
            ok, reasons, change = judge(s, base, o['crps_tol'], o['crps_tol_max'], o['bias_tol'])
            table[k] = dict(gap=gap, gap_parts=parts, feasible=ok, reasons=reasons, crps_change=change,
                            cost=seconds.get(k, 0.)/max(seconds['baseline'], 1e-9), fields=s)
        return table

    table = evaluate_all()
    # Phase 2: best sampler recipe + selection / spectral
    base_gap = table['baseline']['gap']
    candidates = [k for k, v in table.items() if v['feasible'] and k != 'baseline' and
                  spec_methods.get(k, {}).get('kind') in ('trajectory', 'fk') and v['gap'] < base_gap]
    winner = min(candidates, key=lambda k: table[k]['gap']) if candidates and o['phase2'] else None
    if winner:
        log(f'Phase 2: best sampler recipe {winner}; drawing its pool of {o["pool"]}')
        spec = spec_methods[winner]
        extra = solve(spec, o['pool'])[o['members']:] if o['pool'] > o['members'] else \
            np.zeros((0,)+ensembles[winner].shape[1:], dtype='float32')
        extras[f'{winner}_pool'] = np.concatenate([ensembles[winner], extra])
        for suffix, base_spec in (('select_clim', dict(kind='select', select='clim')),
                                  ('spectral', dict(kind='post', post='spectral'))):
            name = f'{winner}+{suffix}'
            s2 = dict(DEFAULTS, id=name, label=f'{spec["label"]} + {suffix.replace("_", " ")}',
                      desc='phase 2 combination', source=winner, **base_spec)
            spec_methods[name] = s2
            ensembles[name] = post(s2, winner)
        table = evaluate_all()

    order = sorted(table, key=lambda k: (not table[k]['feasible'], table[k]['gap']))
    best = next((k for k in order if table[k]['feasible']), 'baseline')
    metrics = dict(case=entry['id'], time=entry['time'], checkpoint=str(path), epoch=saved['epoch']+1, weights=weights,
                   front_point=list(point), normal=[float(v) for v in normal], region=list(region), grid_km=dx_km,
                   tiles=engine.tile_count, options={k: (list(v) if isinstance(v, tuple) else v) for k, v in o.items()},
                   methods={k: {kk: vv for kk, vv in spec_methods[k].items() if not callable(vv)} for k in table},
                   results=table, ranking=order, best=best, phase2_winner=winner, selection=selection,
                   selected=chosen, edges={k: [[list(p), list(n)] for p, n in v] for k, v in edges.items()},
                   climatology=dict(ratio=clim['ratio'].tolist(), times=clim['times']),
                   seconds=time.monotonic()-started)
    write_json(out/'metrics.json', json.loads(json.dumps(metrics, default=_jsonable)))
    if save_members:
        np.savez_compressed(out/'members.npz', truth=truth, coarse=coarse,
                            **{k.replace('+', '__'): v for k, v in ensembles.items()})

    heading = (f'v4.1 epoch {saved["epoch"]+1} ({path.stem}, {weights}) · {split} {entry["time"][:16]} · '
               f'inference recipes · region {(region[1]-region[0])*dx_km:.0f} km')

    def save(fig, stem):
        fig.savefig(out/f'{stem}.png', dpi=dpi)
        if pdf:
            fig.savefig(out/f'{stem}.pdf', dpi=dpi)
        plt.close(fig)
    for name in ('t2m', 'q2m', 'v10m', 'u10m', 'ps', 'precip'):
        save(plot_members(name, truth, coarse, ensembles, order, table, front, heading, plt, edges.get(name)),
             f'members_{name}')
    save(plot_edges(table, order, dx_km, heading, plt), 'edge_profiles')
    save(plot_edge_zoom(truth, ensembles, order, table, edges, dx_km, heading, plt), 'edge_zoom')
    save(plot_scorecard(table, order, heading, plt), 'scorecard')
    save(plot_spectra(ensembles, truth, order, dx_km, heading, plt), 'spectra')
    save(plot_profiles(ensembles, truth, order, front, dx_km, heading, plt), 'profiles')
    save(plot_selection(selection, heading, plt), 'selection')
    write_report(out, metrics, spec_methods)
    log(f'Best feasible recipe: {best} (gap {table[best]["gap"]:.3f} vs baseline {base_gap:.3f})')
    log(f'Inference exploration written to {out} ({time.monotonic()-started:.0f}s)')
    return out


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return str(value)


# ----------------------------------------------------------------------------
# Plots and report
# ----------------------------------------------------------------------------

def _conv(name):
    return (lambda f: f) if name == 'precip' else (lambda f: _display(name, f))


def _norm_for(name, pool):
    from matplotlib.colors import Normalize
    if name == 'precip':
        return _rain_norm()
    lo, hi = np.nanquantile(pool, [.01, .99])
    return DISPLAY[name][4], Normalize(float(lo), float(max(hi, lo+1e-9)))


def plot_members(name, truth, coarse, ensembles, order, table, front, heading, plt, edges=None):
    c = TARGETS.index(name)
    conv = _conv(name)
    panels = [('Truth', truth[c]), ('Coarse', coarse[c])]
    for k in order:
        mark = '✓' if table[k]['feasible'] else '✗'
        panels.append((f'{k} {mark} · gap {table[k]["gap"]:.2f}', ensembles[k][0, c]))
    cols = min(6, len(panels))
    rows = int(np.ceil(len(panels)/cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.3*cols, 3.5*rows), constrained_layout=True, squeeze=False)
    cmap, norm = _norm_for(name, np.concatenate([conv(truth[c]).ravel(), conv(coarse[c]).ravel()]))
    (cy, cx), (ny, nx), length, _ = front
    image = None
    for k, (ax, (label, field)) in enumerate(zip(axes.flat, panels)):
        image = ax.imshow(conv(field), origin='lower', cmap=cmap, norm=norm, interpolation='nearest')
        if edges and k == 0:   # truth: the edge points the edge score is measured at
            for (py, px), (ey, ex) in edges:
                ax.plot([px-EDGE_LENGTH*ex, px+EDGE_LENGTH*ex], [py-EDGE_LENGTH*ey, py+EDGE_LENGTH*ey],
                        color='m', lw=.9)
        elif not edges:
            ax.plot([cx-length*nx, cx+length*nx], [cy-length*ny, cy+length*ny], color='k', lw=.7, ls='--')
        ax.set_title(label, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in list(axes.flat)[len(panels):]:
        ax.set_visible(False)
    fig.colorbar(image, ax=list(axes.flat), shrink=.7, label=f'{DISPLAY[name][0]} ({DISPLAY[name][1]})')
    fig.suptitle(f'{heading}\n{DISPLAY[name][0]}: member 1 of each recipe, ranked (✓ feasible: CRPS/bias/seams '
                 f'within limits); gap 0 = truth-like sharpness'
                 + ('; magenta = truth edge segments scored' if edges else ''), fontsize=10)
    return fig


def plot_edges(table, order, dx_km, heading, plt):
    """Composite cross-edge profiles at the truth's sharpest edge points (normalized by the truth
    jump at each point): a truth-like member mean rises as steeply as truth."""
    fig, axes = plt.subplots(1, len(EDGE_FIELDS), figsize=(5.2*len(EDGE_FIELDS), 5), constrained_layout=True)
    cmap = plt.get_cmap('tab20')
    for ax, name in zip(axes, EDGE_FIELDS):
        base = table['baseline']['fields'][name].get('edge_profile')
        if not base:
            ax.set_visible(False)
            continue
        km = (np.arange(len(base['truth']))-len(base['truth'])//2)*dx_km
        ax.plot(km, base['truth'], color='k', lw=3, label='Truth', zorder=5)
        for i, k in enumerate(order):
            f = table[k]['fields'][name]
            if f.get('edge_profile'):
                ax.plot(km, f['edge_profile']['members'], lw=2 if k == 'baseline' else 1.1,
                        ls='--' if k == 'baseline' else '-', color='#2166ac' if k == 'baseline' else cmap(i % 20),
                        label=f'{k} ({f["edge_ratio"]:.2f})')
        ax.set_title(f'{name}: edge step ratio in legend (1 = truth)', fontsize=9)
        ax.set_xlabel('Distance across the truth edge (km)')
        ax.grid(True, ls='--', alpha=.4)
        ax.legend(fontsize=5.5)
    fig.suptitle(f'{heading}\nComposite profiles across the truth\'s sharpest edges (member mean, normalized by '
                 'truth jump)', fontsize=11)
    return fig


def plot_edge_zoom(truth, ensembles, order, table, edges, dx_km, heading, plt, fields=('q2m', 't2m'),
                   per_field=2, half=48, recipes=11):
    """Zoom on the truth's strongest edges, colours from truth's local 2-98 % range there (no
    saturation by the regional range): truth and member 1 of the best-ranked recipes."""
    names = [k for k in order[:recipes]]
    if 'baseline' not in names:
        names = names[:-1]+['baseline']
    rows = [(n, e) for n in fields for e in (edges.get(n) or [])[:per_field]]
    if not rows:
        fig = plt.figure(figsize=(4, 2))
        fig.text(.5, .5, 'no truth edges found', ha='center')
        return fig
    cols = 1+len(names)
    fig, axes = plt.subplots(len(rows), cols, figsize=(2.3*cols, 2.5*len(rows)+.8), constrained_layout=True,
                             squeeze=False)
    for r, (name, ((py, px), (ey, ex))) in enumerate(rows):
        c = TARGETS.index(name)
        conv = _conv(name)
        h, w = truth.shape[-2:]
        ys, xs = slice(max(0, py-half), min(h, py+half)), slice(max(0, px-half), min(w, px+half))
        window = conv(truth[c][ys, xs])
        lo, hi = np.quantile(window, [.02, .98])
        cmap = DISPLAY[name][4]
        panels = [('Truth', truth[c])]+[(f'{k}\nedge {table[k]["fields"][name].get("edge_ratio") or 0:.2f}',
                                         ensembles[k][0, c]) for k in names]
        for j, (label, field) in enumerate(panels):
            ax = axes[r, j]
            image = ax.imshow(conv(field[ys, xs]), origin='lower', cmap=cmap, vmin=lo, vmax=hi, interpolation='nearest')
            ly, lx = py-ys.start, px-xs.start
            ax.plot([lx-EDGE_LENGTH*ex, lx+EDGE_LENGTH*ex], [ly-EDGE_LENGTH*ey, ly+EDGE_LENGTH*ey], color='m', lw=.8)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(label if r == 0 or j == 0 else '', fontsize=7)
            if j == 0:
                ax.set_ylabel(f'{name} edge {r % per_field+1}', fontsize=8)
        fig.colorbar(image, ax=axes[r, :].tolist(), shrink=.8, pad=.01, label=DISPLAY[name][1])
    fig.suptitle(f'{heading}\nZoom ({2*half*dx_km:.0f} km) on the truth\'s strongest edges, local colour range; '
                 'member 1 of each recipe; "edge" = step ratio (1 = truth)', fontsize=10)
    return fig


def plot_scorecard(table, order, heading, plt):
    fields = list(next(iter(table.values()))['fields'])
    fig, axes = plt.subplots(1, 4, figsize=(24, max(5, .38*len(order)+2)), constrained_layout=True,
                             gridspec_kw=dict(width_ratios=[1, 1.4, 1.4, .8]))
    y = np.arange(len(order))[::-1]
    colors = ['#1a9850' if table[k]['feasible'] else '#bdbdbd' for k in order]
    colors = ['#2166ac' if k == 'baseline' else c for k, c in zip(order, colors)]
    axes[0].barh(y, [table[k]['gap'] for k in order], color=colors)
    axes[0].set_yticks(y, order, fontsize=8)
    axes[0].set_title('Sharpness gap (0 = truth-like)\ngreen feasible, grey not, blue baseline', fontsize=9)
    for ax, key, title, cmap, lim in ((axes[1], 'crps', 'CRPS change vs baseline (%)', 'RdBu_r', 5.),
                                      (axes[2], 'bias', 'Δ |bias| / truth std vs baseline', 'RdBu_r', .05)):
        if key == 'crps':
            data = np.array([[table[k]['crps_change'][f] for f in fields] for k in order])
        else:
            data = np.array([[np.nan if table[k]['fields'][f]['bias_norm'] is None else
                              table[k]['fields'][f]['bias_norm']-table['baseline']['fields'][f]['bias_norm']
                              for f in fields] for k in order])
        image = ax.imshow(data, cmap=cmap, vmin=-lim, vmax=lim, aspect='auto')
        ax.set_xticks(range(len(fields)), fields, fontsize=8)
        ax.set_yticks(range(len(order)), order, fontsize=8)
        for (i, j), v in np.ndenumerate(data):
            ax.text(j, i, f'{v:+.1f}' if key == 'crps' else f'{v:+.3f}', ha='center', va='center', fontsize=6.5)
        fig.colorbar(image, ax=ax, shrink=.6)
        ax.set_title(title, fontsize=9)
    seams = [np.mean([f['seam'] for f in table[k]['fields'].values()]) for k in order]
    axes[3].barh(y, seams, color=colors)
    axes[3].axvline(np.mean([f['truth_seam'] for f in table['baseline']['fields'].values()]), color='k', ls='--',
                    label='truth')
    axes[3].set_yticks(y, order, fontsize=8)
    axes[3].set_xlim(min(.8, min(seams)-.05), max(seams)+.05)
    axes[3].legend(fontsize=8)
    axes[3].set_title('Seam index (1 = no tile seams)', fontsize=9)
    fig.suptitle(f'{heading}\nScorecard', fontsize=11)
    return fig


def plot_spectra(ensembles, truth, order, dx_km, heading, plt):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    cmap = plt.get_cmap('tab20')
    for ax, (c, name) in zip(axes.flat, enumerate(TARGETS)):
        freq, t = radial_psd(sharp_view(c, truth[c]))
        wl = dx_km/np.maximum(freq, 1e-12)
        keep = (freq > 0) & (t > 0)
        for i, k in enumerate(order):
            m = np.mean([radial_psd(sharp_view(c, x[c]))[1] for x in ensembles[k]], 0)
            ax.plot(wl[keep], m[keep]/t[keep], lw=2.2 if k == 'baseline' else 1.1,
                    color='k' if k == 'baseline' else cmap(i % 20), label=k)
        ax.axhline(1, color='k', ls=':')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.invert_xaxis()
        ax.set_xlabel('Wavelength (km)')
        ax.set_title(f'{name}{" (sqrt rain)" if c == RAIN else ""}: member PSD / truth PSD', fontsize=9)
        ax.grid(True, which='both', ls='--', alpha=.3)
    axes.flat[0].legend(fontsize=6, ncol=2)
    fig.suptitle(f'{heading}\nPower spectra relative to truth (1 = truth-like at that scale)', fontsize=11)
    return fig


def plot_profiles(ensembles, truth, order, front, dx_km, heading, plt):
    point, normal, length, band = front
    fig, axes = plt.subplots(1, len(FRONT_FIELDS), figsize=(6*len(FRONT_FIELDS), 5.5), constrained_layout=True)
    cmap = plt.get_cmap('tab20')
    for ax, name in zip(axes, FRONT_FIELDS):
        c = TARGETS.index(name)
        conv = _conv(name)
        s, tp = profile(truth[c], point, normal, length, band)
        ax.plot(s*dx_km, conv(tp), color='k', lw=3, label='Truth', zorder=5)
        for i, k in enumerate(order):
            ax.plot(s*dx_km, conv(profile(ensembles[k][0, c], point, normal, length, band)[1]),
                    lw=2 if k == 'baseline' else 1, color='#2166ac' if k == 'baseline' else cmap(i % 20),
                    ls='--' if k == 'baseline' else '-', label=k)
        ax.set_xlabel('Distance across front (km)')
        ax.set_title(f'{DISPLAY[name][0]} ({DISPLAY[name][1]}) · member 1', fontsize=9)
        ax.grid(True, ls='--', alpha=.4)
    axes[0].legend(fontsize=6, ncol=2)
    fig.suptitle(f'{heading}\nCross-front profiles', fontsize=11)
    return fig


def plot_selection(selection, heading, plt):
    panels = [('closeness', 'truth_p99_ratio', 'Truth-free climatology score (lower = closer)',
               'Member p99|∇| / truth (mean over fields)'),
              ('closeness', 'normalized_mae', 'Truth-free climatology score', 'Member MAE / truth std (mean)')]
    if 'prescreen_closeness' in selection:
        panels.append(('prescreen_closeness', 'closeness', '8-step score', '64-step score'))
    fig, axes = plt.subplots(1, len(panels), figsize=(6*len(panels), 5), constrained_layout=True, squeeze=False)
    for ax, (xk, yk, xl, yl) in zip(axes[0], panels):
        ax.scatter(selection[xk], selection[yk], color='#2166ac')
        for i, (a, b) in enumerate(zip(selection[xk], selection[yk])):
            ax.annotate(str(i+1), (a, b), fontsize=7)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.grid(True, ls='--', alpha=.4)
    extra = (f' · 8-step vs 64-step rank correlation {selection["prescreen_rank_correlation"]:.2f}'
             if 'prescreen_rank_correlation' in selection else '')
    fig.suptitle(f'{heading}\nBaseline pool: does the truth-free score find the truth-like members?{extra}', fontsize=10)
    return fig


def _f(v, fmt='{:.2f}'):
    return 'n/a' if v is None else fmt.format(v)


def write_report(out, metrics, specs):
    t = metrics['results']
    base = t['baseline']
    L = [f'# Inference recipe search · case `{metrics["case"]}` · epoch {metrics["epoch"]} ({metrics["weights"]})', '',
         f'Front point {metrics["front_point"]}, region rows {metrics["region"][0]}:{metrics["region"][1]}, cols '
         f'{metrics["region"][2]}:{metrics["region"][3]} ({metrics["grid_km"]:.2f} km grid), tiles {metrics["tiles"]}, '
         f'{metrics["options"]["members"]} members (pool {metrics["options"]["pool"]}), '
         f'{metrics["options"]["steps"]} steps. One case: a screen, not a verdict.', '',
         f'**Best feasible recipe: `{metrics["best"]}`** (sharpness gap {t[metrics["best"]]["gap"]:.3f} vs baseline '
         f'{base["gap"]:.3f}; lower = closer to truth).', '',
         '## Ranking', '',
         'Gap = mean |log| distance of member sharpness from truth (p99 |∇|, 4–30 km power, and the edge step: '
         'change across ~7.5 km at the truth\'s sharpest edge points, members allowed ±15 km displacement). '
         f'Feasible: mean CRPS change ≤ {metrics["options"]["crps_tol"]}%, no field > '
         f'{metrics["options"]["crps_tol_max"]}%, |bias|/std growth ≤ {metrics["options"]["bias_tol"]}, '
         f'seam index growth ≤ {SEAM_TOL}. Cost = time per member / baseline.', '',
         '| Rank | Recipe | What | Feasible | Gap | p99 part | PSD part | Edge part | Mean ΔCRPS % | Worst ΔCRPS % | '
         'Seams | Cost | Why not |',
         '|---:|---|---|:-:|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for i, k in enumerate(metrics['ranking'], 1):
        r = t[k]
        change = list(r['crps_change'].values())
        worst = max(r['crps_change'], key=r['crps_change'].get)
        L.append(f'| {i} | `{k}` | {specs[k]["desc"]} | {"✓" if r["feasible"] else "✗"} | {r["gap"]:.3f} | '
                 f'{_f(r["gap_parts"]["p99"], "{:.3f}")} | {_f(r["gap_parts"]["psd"], "{:.3f}")} | '
                 f'{_f(r["gap_parts"]["edge"], "{:.3f}")} | {np.mean(change):+.2f} | {worst} {r["crps_change"][worst]:+.1f} | '
                 f'{np.mean([f["seam"] for f in r["fields"].values()]):.3f} | {r["cost"]:.1f}× | '
                 f'{"; ".join(r["reasons"])} |')
    fields = list(base['fields'])
    for title, key, fmt in (('p99 |∇| member ratio to truth (1 = truth-like)', 'p99_ratio', '{:.2f}'),
                            ('4–30 km power ratio to truth', 'fine_psd_ratio', '{:.2f}'),
                            ('CRPS', 'crps', '{:.4g}'), ('Bias (ensemble mean − truth)', 'bias', '{:+.3g}'),
                            ('Spread/skill (1 = calibrated)', 'spread_skill', '{:.2f}')):
        L += ['', f'## {title}', '', '| Recipe | ' + ' | '.join(fields) + ' |', '|---|' + '---:|'*len(fields)]
        for k in metrics['ranking']:
            L.append(f'| `{k}` | ' + ' | '.join(_f(t[k]['fields'][f][key], fmt) for f in fields) + ' |')
    L += ['', '## Edge step ratio at the truth\'s sharpest edges (1 = as sharp as truth; see edge_profiles.png)', '',
          '| Recipe | ' + ' | '.join(EDGE_FIELDS) + ' |', '|---|' + '---:|'*len(EDGE_FIELDS)]
    for k in metrics['ranking']:
        L.append(f'| `{k}` | ' + ' | '.join(_f(t[k]['fields'][f].get('edge_ratio')) for f in EDGE_FIELDS) + ' |')
    L += ['', 'Per member (min–max over members): the best member is the ceiling any selection or steering can reach.',
          '', '| Recipe | ' + ' | '.join(EDGE_FIELDS) + ' |', '|---|' + '---:|'*len(EDGE_FIELDS)]

    def span(k, f):
        prof = t[k]['fields'][f].get('edge_profile') or {}
        v = prof.get('per_member')
        return 'n/a' if not v else f'{min(v):.2f}–{max(v):.2f}'
    for k in metrics['ranking']:
        L.append(f'| `{k}` | ' + ' | '.join(span(k, f) for f in EDGE_FIELDS) + ' |')
    L += ['', '## Single-profile front max-slope width, member median (km) (legacy; one line, ramp-dominated)', '',
          '| Recipe | ' + ' | '.join(FRONT_FIELDS) + ' |', '|---|' + '---:|'*len(FRONT_FIELDS),
          '| truth | ' + ' | '.join(_f(base['fields'][f].get('truth_width_km'), '{:.1f}') for f in FRONT_FIELDS) + ' |']
    for k in metrics['ranking']:
        L.append(f'| `{k}` | ' + ' | '.join(_f(t[k]['fields'][f].get('member_width_km'), '{:.1f}')
                                            for f in FRONT_FIELDS) + ' |')
    sel = metrics['selection']
    L += ['', '## Member selection diagnostics (baseline pool)', '',
          f'Climatology from {len(metrics["climatology"]["times"])} training hours near the date. Selected pool members: '
          + ', '.join(f'{k}: {[i+1 for i in v]}' for k, v in metrics['selected'].items()) + '.']
    if 'fk_mean_ess' in sel:
        L.append(f'FK steering effective sample size per resampling (of {metrics["options"]["fk_particles"]} particles; '
                 f'near the particle count = no selection): ' + ', '.join(f'{v:.2f}' for v in sel['fk_mean_ess']) + '.')
    if 'prescreen_rank_correlation' in sel:
        L.append(f'8-step vs {metrics["options"]["steps"]}-step truth-free score rank correlation: '
                 f'{sel["prescreen_rank_correlation"]:.2f} (near 1 = cheap solves predict sharp seeds).')
    if metrics['phase2_winner']:
        L.append(f'Phase 2 combined the best sampler recipe `{metrics["phase2_winner"]}` with selection and spectral fix.')
    else:
        L.append('Phase 2 not run (switched off, or no sampler recipe was both feasible and sharper than baseline).')
    L += ['', 'Next: run the best recipe on the test set (several cases, 8 members) before adopting it.']
    (out/'report.md').write_text('\n'.join(L)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    d = OPTION_DEFAULTS
    parser.add_argument('--config', default='configs/discover_v4_1.yaml')
    parser.add_argument('--checkpoint', default='latest', help='best | latest (default) | <epoch> | <path>')
    parser.add_argument('--weights', choices=('ema', 'raw'), default='ema')
    parser.add_argument('--timestamp', help='Case ID or ISO time (default: wettest hour of the split)')
    parser.add_argument('--split', choices=('val', 'test'), default='val')
    parser.add_argument('--methods', default=','.join(METHOD_IDS), help=f'Comma list from: {", ".join(METHOD_IDS)}')
    parser.add_argument('--center', type=int, nargs=2, metavar=('ROW', 'COL'))
    parser.add_argument('--center-latlon', type=float, nargs=2, metavar=('LAT', 'LON'))
    parser.add_argument('--anywhere', action='store_true', help='Search the whole domain for the front')
    parser.add_argument('--members', type=int, default=d['members'])
    parser.add_argument('--pool', type=int, default=d['pool'], help='Members drawn for selection recipes')
    parser.add_argument('--steps', type=int, default=d['steps'])
    parser.add_argument('--region', type=int, default=d['region'], help='Evaluation region size (px)')
    parser.add_argument('--margin', type=int, default=d['margin'], help='Extra sampled margin around it (px)')
    parser.add_argument('--clim-count', type=int, default=d['clim_count'])
    parser.add_argument('--clim-days', type=int, default=d['clim_days'])
    parser.add_argument('--warp-gamma', type=float, default=d['warp_gamma'])
    parser.add_argument('--churn', type=float, default=d['churn'])
    parser.add_argument('--langevin', type=float, default=d['langevin'], help='Langevin corrector noise fraction eta')
    parser.add_argument('--langevin-range', type=float, nargs=2, default=d['langevin_range'])
    parser.add_argument('--restart', type=int, default=d['restart'])
    parser.add_argument('--restart-t', type=float, default=d['restart_t'])
    parser.add_argument('--temp', type=float, default=d['temp'])
    parser.add_argument('--hf-boost', type=float, default=d['hf_boost'])
    parser.add_argument('--hf-sigma', type=float, default=d['hf_sigma'])
    parser.add_argument('--hf-start', type=float, default=d['hf_start'])
    parser.add_argument('--guide-checkpoint', default=d['guide_checkpoint'])
    parser.add_argument('--guide-weight', type=float, default=d['guide_weight'])
    parser.add_argument('--vguide-strengths', type=float, nargs='+', default=list(d['vguide_strengths']))
    parser.add_argument('--vguide-range', type=float, nargs=2, default=d['vguide_range'])
    parser.add_argument('--vguide-fields', nargs='+', default=list(d['vguide_fields']))
    parser.add_argument('--vguide-radius', type=int, default=d['vguide_radius'])
    parser.add_argument('--vguide-eps', type=float, default=d['vguide_eps'])
    parser.add_argument('--sde-strengths', type=float, nargs='+', default=list(d['sde_strengths']))
    parser.add_argument('--sde-range', type=float, nargs=2, default=d['sde_range'])
    parser.add_argument('--ag-sigma', type=float, default=d['ag_sigma'],
                        help='High-pass Gaussian sigma (px) for autoguide_hf (default 8 px, ~80 km cutoff)')
    parser.add_argument('--ag-range', type=float, nargs=2, default=d['ag_range'])
    parser.add_argument('--fk-particles', type=int, default=d['fk_particles'])
    parser.add_argument('--fk-lambda', type=float, default=d['fk_lambda'])
    parser.add_argument('--fk-times', type=float, nargs='+', default=list(d['fk_times']))
    parser.add_argument('--combine', default='',
                        help='Comma list of combined recipes, each joined by "+", e.g. '
                             '"autoguide_hf+spectral,autoguide_hf+fk_steer+spectral"')
    parser.add_argument('--no-phase2', action='store_true', help='Skip the automatic phase-2 combinations')
    parser.add_argument('--spectral-max-gain', type=float, default=d['spectral_max_gain'])
    parser.add_argument('--spectral-cutoff-km', type=float, default=d['spectral_cutoff_km'])
    parser.add_argument('--crps-tol', type=float, default=d['crps_tol'])
    parser.add_argument('--crps-tol-max', type=float, default=d['crps_tol_max'])
    parser.add_argument('--bias-tol', type=float, default=d['bias_tol'])
    parser.add_argument('--save-members', action='store_true')
    parser.add_argument('--output')
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--dpi', type=int, default=200)
    parser.add_argument('--no-pdf', action='store_true')
    a = parser.parse_args()
    options = {k: getattr(a, k) for k in OPTION_DEFAULTS if hasattr(a, k) and k not in ('combine', 'phase2')}
    options['combine'] = tuple(c.strip() for c in a.combine.split(',') if c.strip())
    options['phase2'] = not a.no_phase2
    for k in ('langevin_range', 'vguide_range', 'vguide_strengths', 'vguide_fields', 'sde_strengths', 'sde_range',
              'ag_range', 'fk_times'):
        options[k] = tuple(options[k])
    run(load_config(a.config), a.checkpoint, a.timestamp, a.split, tuple(m for m in a.methods.split(',') if m),
        tuple(a.center) if a.center else None, tuple(a.center_latlon) if a.center_latlon else None, a.anywhere,
        a.weights, a.output, a.batch, a.threads, a.dpi, not a.no_pdf, a.save_members,
        log=lambda message: print(message, flush=True), **options)


if __name__ == '__main__':
    main()
