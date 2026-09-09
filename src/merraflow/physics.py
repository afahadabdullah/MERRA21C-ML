"""Physical-space precipitation budgets on a finite-volume, center-assigned grid.

Each HR pixel belongs to the native cell containing its center. Budgets use the
sum of actual HR AREA within that footprint, including partial domain edges.
This is a discrete native-footprint constraint, NOT exact polygon intersections.
"""
import numpy as np


def native_groups(lat, lon, hr_lat, hr_lon):
    lat, lon = np.asarray(lat), np.asarray(lon)
    if lat.ndim != 1 or lon.ndim != 1 or min(len(lat), len(lon)) < 2:
        raise ValueError('Native grid must have 1D latitude/longitude coordinates')
    if np.any(np.diff(lat) <= 0) or np.any(np.diff(lon) <= 0):
        raise ValueError('Sort native coordinates before constructing footprints')
    def index(v, x):
        bounds = np.r_[v[0]-(v[1]-v[0])/2, (v[:-1]+v[1:])/2, v[-1]+(v[-1]-v[-2])/2]
        if np.any((x < bounds[0]) | (x > bounds[-1])):
            raise ValueError('Target grid extends beyond native coordinate bounds')
        return np.clip(np.searchsorted(bounds, x, side='right')-1, 0, len(v)-1)
    iy, ix = index(lat, hr_lat), index(lon, hr_lon)
    source_flat, inverse = np.unique(iy*len(lon)+ix, return_inverse=True)
    return inverse.reshape(hr_lat.shape).astype('int32'), source_flat


def group_sum(field, area, groups):
    return np.bincount(groups.ravel(), weights=(np.asarray(field, dtype='float64')*area).ravel(), minlength=int(groups.max())+1)


def project_precip(prediction, reference, area, groups, dry_threshold=0.0):
    """Nonnegative area-weighted projection; all-dry proposals use reference shape.

    Apply ONCE after full-domain stitching. Patch-level projection cannot conserve
    groups cut by patch boundaries. Supports one physical precipitation field.
    """
    p, r = np.asarray(prediction, dtype='float64'), np.asarray(reference, dtype='float64')
    if p.shape != r.shape or p.shape != area.shape or p.shape != groups.shape:
        raise ValueError('Prediction, budget, area and groups must have identical shape')
    if not all(np.isfinite(a).all() for a in (p, r, area)) or np.any(area <= 0) or np.any(r < 0):
        raise ValueError('Conservation requires finite fields, positive area, nonnegative reference')
    p = np.maximum(p, 0)
    p[p < dry_threshold] = 0
    target, current = group_sum(r, area, groups), group_sum(p, area, groups)
    scale = np.divide(target, current, out=np.zeros_like(target), where=current > 0)
    out = p * scale[groups]
    missing = (current == 0) & (target > 0)
    out = np.where(missing[groups], r, out)
    return out.astype('float32')


def budget_error(pred, ref, area, groups):
    expected, actual = group_sum(ref, area, groups), group_sum(pred, area, groups)
    wet = expected > 0
    return {'max_absolute_mm_m2_per_hour': float(np.max(np.abs(actual-expected))),
            'max_relative_wet': float(np.max(np.abs(actual[wet]-expected[wet])/expected[wet])) if wet.any() else 0.,
            'dry_group_leakage_mm_m2_per_hour': float(np.max(np.abs(actual[~wet]))) if (~wet).any() else 0.,
            'domain_relative': float(abs(actual.sum()-expected.sum())/expected.sum()) if expected.sum() else float(abs(actual.sum()))}


def transform_target(x, precip_scale=1., wind_scale=5.):
    x = np.asarray(x, dtype='float32').copy()
    x[1] = np.log1p(np.maximum(x[1], 0)/precip_scale)
    x[3] = np.log1p(np.maximum(x[3], 0)/wind_scale)
    return x


def inverse_target(x, precip_scale=1., wind_scale=5.):
    x = np.asarray(x, dtype='float32').copy()
    # Avoid exponent overflow; saturation is deliberately far outside training range.
    x[1] = np.expm1(np.clip(x[1], 0, 20))*precip_scale
    x[3] = np.expm1(np.clip(x[3], 0, 20))*wind_scale
    return x
