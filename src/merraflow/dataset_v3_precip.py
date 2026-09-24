"""Read v2 predictors, but expose only HWT precipitation as supervision.

History is strictly causal and never crosses a split or fills missing hours.

Rain encoding (fix 3, dry-margin): wet pixels use z = sqrt(1 + P/s) - 1 >= 0;
pixels below ``dry_threshold_mm_h`` are encoded at ``-dry_offset``. A generated
value must climb the whole margin before it decodes as rain, so small positive
diffusion noise over dry areas no longer leaks drizzle everywhere.

Targets (fix 6): ``midpoint_rate`` is the archived HWT :30 snapshot.
``hourly_mean_trapezoid`` is 1/4 P(:00) + 1/2 P(:30) + 1/4 P(:00 next hour)
from ``hwt_30mn_slv_LCC``, prepared by ``prepare_v3_precip`` into a separate
directory. It approximates the GEOS hourly mean window far better than one snapshot.

Patch proposals (fix 4): ``coarse`` scores candidate patches from the coarse
GEOS precipitation, an input. Oversampling by an input leaves p(target | input)
unchanged under input-only selection, but changes the population training risk.
Unit weights deliberately emphasize rainy inputs; they do not estimate uniform risk.
``truth`` keeps the old HWT-truth proposal with exact inverse weights.
"""
from datetime import datetime, timedelta
from pathlib import Path
import json
import numpy as np
import torch
from .dataset_v2 import ArchiveV2, PatchDatasetV2, crop_v2, box_means_v2

TARGET_KINDS = ('midpoint_rate', 'hourly_mean_trapezoid')
HOURLY_FILE = 'truth_hourly_v3_precip.npy'
HOURLY_INDEX = 'index_hourly_v3_precip.json'


def encode_rain(rain, scale, dry_threshold=0., dry_offset=0.):
    rain = np.maximum(rain, 0)
    value = rain/scale
    z = value/(np.sqrt(1+value)+1)
    if dry_offset > 0:
        z = np.where(rain < dry_threshold, -dry_offset, z)
    return z.astype('float32') if isinstance(z, np.ndarray) else z


def decode_rain(value, scale, wet_threshold=0.):
    value = np.asarray(value)
    if not np.isfinite(value).all():
        raise FloatingPointError('Nonfinite generated rainfall')
    wet = value > wet_threshold
    value = np.maximum(value, 0)
    rain = np.where(wet, scale*value*(value+2), 0.)
    if not np.isfinite(rain).all():
        raise FloatingPointError('Nonfinite generated rainfall')
    return rain.astype('float32')


def rain_codec(cfg):
    d = cfg['data']
    return dict(scale=d['rain_scale_mm_h'], dry_threshold=d.get('dry_threshold_mm_h', 0.),
                dry_offset=d.get('dry_offset', 0.), wet_threshold=d.get('wet_threshold_z', 0.))


class PrecipArchive(ArchiveV2):
    def __init__(self, cfg, verify_files=True):
        super().__init__(cfg['data']['prepared'])
        if self.stats.get('target_channels') != ['t2m', 'precip', 'ps', 'u10m', 'v10m']:
            raise ValueError('Unexpected source target ordering')
        data = self.index['data_config']
        if data.get('state_alignment') != 'midpoint_snapshot' or data.get('precip_source') != 'hwt_30mn_slv_LCC.PRECTOT':
            raise ValueError('Require documented HWT midpoint PRECTOT archive')
        if self.stats.get('training_coverage', {}).get('split') != 'train':
            raise ValueError('Archive must document training-only normalization')
        self.lags = cfg['data']['history_hours']
        codec = rain_codec(cfg)
        self.scale, self.dry_threshold, self.dry_offset, self.wet_threshold = (
            codec['scale'], codec['dry_threshold'], codec['dry_offset'], codec['wet_threshold'])
        self.target_kind = cfg['data']['target_kind']
        if self.target_kind not in TARGET_KINDS:
            raise ValueError(f'Unknown target_kind {self.target_kind}')
        self.by_time = {datetime.fromisoformat(e['time']): e for e in self.index['entries']}
        if len(self.by_time) != len(self.index['entries']):
            raise ValueError('Duplicate archive timestamps')
        self.channels = self.index['condition_channels']+(len(self.lags)-1)*len(self.cm)
        self.hourly_root, self.hourly_ids, self.hourly_fingerprint = None, None, None
        if self.target_kind == 'hourly_mean_trapezoid':
            self.hourly_root = Path(cfg['data']['hourly_targets'])
            index_path = self.hourly_root/HOURLY_INDEX
            if not index_path.exists():
                raise FileNotFoundError(f'{index_path} missing: run `cli_v3_precip prepare-hourly` then '
                                        '`finalize-hourly` (see docs/v3_precip.md)')
            index = json.loads(index_path.read_text())
            if index.get('archive_fingerprint') != self.index['fingerprint'] or index.get('method') != 'trapezoid_00_30_00':
                raise ValueError('Hourly targets were prepared for a different archive or method')
            from .prepare_v3_precip import validate_hourly_index
            validate_hourly_index(cfg, self, index, verify_files=verify_files)
            self.hourly_fingerprint = index['fingerprint']
            self.hourly_ids = set(index['completed'])

    def encode_rain(self, rain):
        return encode_rain(rain, self.scale, self.dry_threshold, self.dry_offset)

    def decode_rain(self, value):
        return decode_rain(value, self.scale, self.wet_threshold)

    def history(self, entry):
        time = datetime.fromisoformat(entry['time'])
        result = [self.by_time.get(time+timedelta(hours=lag)) for lag in self.lags]
        if any(e is None or e['split'] != entry['split'] for e in result):
            raise ValueError(f'Incomplete within-split hourly history: {entry["time"]}')
        return result

    def has_target(self, entry):
        return self.hourly_ids is None or entry['id'] in self.hourly_ids

    def eligible(self, split):
        result = []
        for entry in self.index['entries']:
            if entry['split'] == split and self.has_target(entry):
                try:
                    self.history(entry)
                except ValueError:
                    continue
                result.append(entry)
        if not result:
            raise ValueError(f'No {split} hours with complete within-split history')
        return result

    def condition(self, entry, y, x, size, halo=0):
        current = super().condition(entry, y, x, size, halo)
        history = [(crop_v2(self.array(e, 'condition'), y, x, size, halo)-self.cm)/self.cs
                   for e in self.history(entry)[:-1]]
        return np.concatenate([current, *history]).astype('float32')

    def truth_field(self, entry):
        """Full-domain (1, H, W) mm/h training/evaluation target (memory map when possible)."""
        if self.target_kind == 'midpoint_rate':
            return self.array(entry, 'truth')[1:2]
        return np.load(self.hourly_root/entry['id']/HOURLY_FILE, mmap_mode='r')

    def rain(self, entry, name, y, x, size, halo=0):
        # Slice the memory map before cropping; no non-rain target is read.
        source = self.truth_field(entry) if name == 'truth' else self.array(entry, name)[1:2]
        value = crop_v2(source, y, x, size, halo).copy()
        if not np.isfinite(value).all() or np.any(value < 0):
            raise ValueError(f'Invalid {name} precipitation at {entry["time"]}')
        return value


class PrecipDataset(PatchDatasetV2):
    def __init__(self, cfg, split, samples, seed=0):
        super().__init__(cfg['data']['prepared'], split, cfg['patch'], samples, seed)
        self.archive = PrecipArchive(cfg)
        self.entries = self.archive.eligible(split)
        self.proposal_kind = cfg['patch'].get('proposal', 'coarse')
        if self.proposal_kind not in ('coarse', 'truth'):
            raise ValueError('patch.proposal must be coarse or truth')

    def rain_score(self, entry):
        if self.proposal_kind == 'truth':
            if self.archive.target_kind != 'midpoint_rate':
                rain = np.asarray(self.archive.truth_field(entry)[0])
                return box_means_v2(np.log1p(rain), self.yy, self.xx, self.patch['size'])
            return super().rain_score(entry)
        # PatchDatasetV2.proposal caches the resulting q per entry and worker.
        rain = np.asarray(self.archive.array(entry, 'baseline')[1])
        return box_means_v2(np.log1p(np.maximum(rain, 0)), self.yy, self.xx, self.patch['size'])

    def __getitem__(self, index):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, int(index)]))
        entry = self.entries[rng.integers(len(self.entries))]
        q = self.proposal(entry)
        location = rng.choice(len(q), p=q)
        y, x = self.yy[location], self.xx[location]
        a, p = self.archive, self.patch
        condition, context = a.inputs(entry, y, x, p)
        truth = a.rain(entry, 'truth', y, x, p['size'], p['halo'])
        coarse = a.rain(entry, 'baseline', y, x, p['size'], p['halo'])
        baseline = a.encode_rain(coarse)
        target = a.encode_rain(truth)-baseline
        area = crop_v2(a.static['area'], y, x, p['size'])
        area_full = crop_v2(a.static['area'], y, x, p['size'], p['halo'])
        # Deliberate rain-emphasized population objective for coarse proposals.
        importance = 1. if self.proposal_kind == 'coarse' else 1/(len(q)*q[location])
        return dict(condition=condition, context=context, target=torch.from_numpy(target),
                    baseline=torch.from_numpy(baseline), truth=torch.from_numpy(truth),
                    coarse=torch.from_numpy(coarse), area=torch.from_numpy(area/area.mean()),
                    area_full=torch.from_numpy(area_full/area.mean()),
                    importance=torch.tensor(importance, dtype=torch.float32))
