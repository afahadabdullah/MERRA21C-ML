"""Validate HWT surface fractions; explicitly mark unavailable inland-lake data."""
import warnings
import numpy as np
import xarray as xr
from scipy.ndimage import distance_transform_edt


def add_surface_features_v2(static, spec):
    # Import lazily: prepare_v2 imports this module too.
    from .prepare import field
    with xr.open_dataset(spec['path']) as ds:
        lat, lon = [field(ds, spec[k]) for k in ('lat', 'lon')]
        if spec.get('ocean'):
            if spec['ocean'] not in ds:
                raise ValueError(f'{spec["path"]}: missing {spec["ocean"]}; set data.static.path to the HWT static file')
            ocean = field(ds, spec['ocean'])
            lake_known = bool(spec.get('lake') and spec['lake'] in ds)
            if not lake_known and spec.get('require_lake', False):
                raise ValueError('Configured FRLAKE/lake fraction is missing')
            lake = field(ds, spec['lake']) if lake_known else np.zeros_like(ocean)
            land = 1-ocean-lake
            if not lake_known:
                warnings.warn('FROCEAN available but lake fraction absent: non-ocean includes unresolved inland lakes', stacklevel=2)
        else:
            land, lake = [field(ds, spec[k]) for k in ('land', 'lake')]
            lake_known = True
            ocean = 1-land-lake
    lon = (lon+180) % 360-180
    for key, value in [('lat', lat), ('lon', lon)]:
        if value.shape != static[key].shape or not np.allclose(value, static[key], atol=2e-5, rtol=0):
            raise ValueError(f'Static {key} does not match HWT grid; regrid explicitly before preparation')
    if (np.any(land < -1e-6) or np.any(lake < 0) or np.any(land > 1.000001) or np.any(lake > 1)
            or np.any(ocean < -1e-6) or np.any(ocean > 1.000001)
            or np.any(land+lake > 1.00001)):
        raise ValueError('Require disjoint solid-land and lake fractions in [0,1], summing to <=1')
    land, ocean = np.clip(land, 0, 1), np.clip(ocean, 0, 1)
    # Signed distance to any water boundary, in index-space pixels / 50.
    solid = land >= .5
    if solid.all() or not solid.any():
        distance = np.full(land.shape, 1. if solid.all() else -1.)
    else:
        distance = np.clip((distance_transform_edt(solid)-distance_transform_edt(~solid))/50, -1, 1)
    # Surface derivatives along grid axes use geodesic distances between centers.
    latr, lonr = np.deg2rad(lat), np.unwrap(np.deg2rad(lon), axis=1)
    slopes = []
    for axis in (0, 1):
        spacing = 6371000*np.hypot(np.gradient(latr, axis=axis),
                                  np.cos(latr)*np.gradient(lonr, axis=axis))
        if np.any(spacing <= 0):
            raise ValueError('Degenerate static grid spacing')
        slopes.append(np.gradient(static['elevation'], axis=axis)/spacing)
    known = np.full(land.shape, float(lake_known), dtype='float32')
    static.update(land_fraction=land, lake_fraction=lake, ocean_fraction=ocean, lake_fraction_known=known)
    static['features'] = np.concatenate([static['features'], np.stack([land, lake, distance, *slopes, ocean, known])]).astype('float32')
    return static
