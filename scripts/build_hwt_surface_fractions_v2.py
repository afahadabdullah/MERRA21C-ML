#!/usr/bin/env python3
"""Create HWT-LCC ocean and inland-water fractions from GSHHG shorelines.

The HWT HISTORY collections do not export FROCEAN or FRLAKE on their 3-km LCC
grid. This utility samples GSHHG's nested shoreline polygons over each HWT
cell and writes a single, coordinate-matched static file for v2 preparation.

Cartopy downloads GSHHG on first use when it is not already in its data cache.
For offline Discover jobs, set CARTOPY_DATA_DIR to a populated cache (or pass
--cartopy-data-dir) before running this script.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Union

import numpy as np
import xarray as xr


def two_dimensional(ds: xr.Dataset, name: str) -> tuple[np.ndarray, tuple[str, str]]:
    """Return a finite 2-D field, removing only singleton non-horizontal axes."""
    if name not in ds:
        raise KeyError(f'{ds.encoding.get("source", "grid")}: missing {name}')
    field = ds[name]
    for dim in tuple(field.dims):
        if dim not in ('Ydim', 'Xdim', 'y', 'x', 'lat', 'lon'):
            if field.sizes[dim] != 1:
                raise ValueError(f'{name}: expected singleton {dim}, got {field.sizes[dim]}')
            field = field.isel({dim: 0})
    if field.ndim != 2:
        raise ValueError(f'{name}: require a two-dimensional grid field')
    values = field.values.astype('float64')
    if not np.isfinite(values).all():
        raise ValueError(f'{name}: grid contains non-finite coordinates')
    return values, tuple(field.dims)


def load_hwt_grid(path: Union[str, Path], lat_name: str, lon_name: str):
    with xr.open_dataset(path) as ds:
        lat, dims = two_dimensional(ds, lat_name)
        lon, lon_dims = two_dimensional(ds, lon_name)
        if lon.shape != lat.shape or lon_dims != dims:
            raise ValueError('Latitude and longitude arrays must share two dimensions and shape')
        coords = {dim: ds[dim].load() for dim in dims if dim in ds and ds[dim].ndim == 1}
    return lat, lon, dims, coords


def gshhg_longitudes(lon: np.ndarray) -> np.ndarray:
    """Convert HWT 0-360 longitudes to the -180..180 convention of GSHHG."""
    return (lon+180) % 360-180


def sample_coordinates(lat: np.ndarray, lon: np.ndarray, first_row: int, last_row: int, supersample: int):
    """Bilinearly interpolate lat/lon at sub-cell centers in index space."""
    height, width = lat.shape
    fy = (np.arange(first_row * supersample, last_row * supersample)+.5)/supersample-.5
    fx = (np.arange(width * supersample)+.5)/supersample-.5
    y_base = np.floor(fy).astype(int)
    x_base = np.floor(fx).astype(int)
    y0, x0 = np.clip(y_base, 0, height-1), np.clip(x_base, 0, width-1)
    y1, x1 = np.clip(y_base+1, 0, height-1), np.clip(x_base+1, 0, width-1)
    wy = (fy-y_base)[:, None]
    wx = (fx-x_base)[None, :]

    def interpolate(values):
        top = values[y0[:, None], x0[None, :]]*(1-wx)+values[y0[:, None], x1[None, :]]*wx
        bottom = values[y1[:, None], x0[None, :]]*(1-wx)+values[y1[:, None], x1[None, :]]*wx
        return top*(1-wy)+bottom*wy

    return interpolate(lat), interpolate(lon)


def gshhg_geometries(lon: np.ndarray, lat: np.ndarray, scale: str):
    """Read only GSHHG polygons that can intersect the HWT domain."""
    from cartopy.io import shapereader
    from shapely.geometry import GeometryCollection
    from shapely import make_valid, union_all

    west, east = float(lon.min())-.25, float(lon.max())+.25
    south, north = float(lat.min())-.25, float(lat.max())+.25
    result = {}
    for level in range(1, 5):
        source = shapereader.gshhs(scale=scale, level=level)
        selected = []
        repaired = 0
        for record in shapereader.Reader(source).records():
            geometry = record.geometry
            left, bottom, right, top = geometry.bounds
            if right >= west and left <= east and top >= south and bottom <= north:
                if not geometry.is_valid:
                    geometry = make_valid(geometry)
                    repaired += 1
                selected.append(geometry)
        if level == 1 and not selected:
            raise ValueError(f'GSHHG has no land polygons overlapping the HWT grid; '
                             f'polygon longitude range {west:.2f}..{east:.2f}, '
                             f'latitude range {south:.2f}..{north:.2f}')
        # Repair invalid source polygons before merging overlaps; GEOS point
        # containment also requires the combined geometry to be topologically valid.
        result[level] = union_all(selected) if selected else GeometryCollection()
        print(f'GSHHG level {level}: {len(selected)} overlapping polygons, '
              f'{repaired} repaired', flush=True)
    return result


def fractions(lat: np.ndarray, lon: np.ndarray, levels, supersample: int, chunk_rows: int):
    """Return fractional land, lake, and ocean cover using GSHHG nesting parity."""
    from shapely import contains_xy, prepare

    if supersample < 1 or chunk_rows < 1:
        raise ValueError('supersample and chunk_rows must be positive')
    for geometry in levels.values():
        prepare(geometry)
    height, width = lat.shape
    counts = np.zeros((3, height, width), dtype='uint16')
    total = supersample**2
    for start in range(0, height, chunk_rows):
        stop = min(height, start+chunk_rows)
        sample_lat, sample_lon = sample_coordinates(lat, lon, start, stop, supersample)
        inside = {level: contains_xy(geometry, sample_lon, sample_lat)
                  for level, geometry in levels.items()}
        # GSHHG levels alternate land and water: L1 land, L2 lake, L3 island
        # in a lake, L4 pond in that island. Preserve that nesting explicitly.
        lake = (inside[2] & ~inside[3]) | inside[4]
        land = (inside[1] & ~inside[2]) | (inside[3] & ~inside[4])
        ocean = ~(land | lake)
        shape = (stop-start, supersample, width, supersample)
        for index, mask in enumerate((land, lake, ocean)):
            counts[index, start:stop] = mask.reshape(shape).sum(axis=(1, 3), dtype='uint16')
        print(f'sampled rows {start+1}-{stop}/{height}', flush=True)
    values = counts.astype('float32')/total
    if not np.allclose(values.sum(axis=0), 1, rtol=0, atol=1e-7):
        raise RuntimeError('Surface fractions do not sum to one')
    return values


def write_output(path: Union[str, Path], lat: np.ndarray, lon: np.ndarray, dims, coords, values: np.ndarray,
                 grid_path: Union[str, Path], scale: str, supersample: int):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    land, lake, ocean = values
    fields = {
        'lats': (dims, lat.astype('float32'), {'units': 'degrees_north'}),
        'lons': (dims, lon.astype('float32'), {'units': 'degrees_east'}),
        'FRLAND': (dims, land, {'units': '1', 'long_name': 'GIS-derived land and land-ice fraction'}),
        'FRLAKE': (dims, lake, {'units': '1', 'long_name': 'GIS-derived inland-lake and pond fraction'}),
        'FROCEAN': (dims, ocean, {'units': '1', 'long_name': 'GIS-derived open-ocean fraction'}),
    }
    dataset = xr.Dataset(fields, coords=coords,
                         attrs={'title': 'HWT LCC static surface fractions for MERRAflow v2',
                                'source': f'GSHHG full hierarchy (scale={scale})',
                                'grid_source': str(Path(grid_path).resolve()),
                                'fraction_method': f'{supersample}x{supersample} sub-cell sampling in HWT index space',
                                'classification': 'GSHHG level 1/3 land, 2/4 inland water, outside land ocean'})
    temporary = destination.with_suffix(destination.suffix+'.tmp')
    encoding = {name: {'zlib': True, 'complevel': 4, 'dtype': 'float32'} for name in fields}
    dataset.to_netcdf(temporary, engine='h5netcdf', encoding=encoding)
    os.replace(temporary, destination)
    return destination


def write_plot(path: Union[str, Path], values: np.ndarray):
    """Save a compact RGB preview without pretending index coordinates are lat/lon."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    land, lake, ocean = values
    # Keep the preview light even when the native HWT grid is ~1.9 million cells.
    stride = max(1, int(np.ceil(max(land.shape)/1200)))
    land, lake, ocean = (field[::stride, ::stride] for field in (land, lake, ocean))
    image = (land[..., None]*np.array([0.27, 0.55, 0.24])
             + lake[..., None]*np.array([0.20, 0.72, 0.95])
             + ocean[..., None]*np.array([0.05, 0.19, 0.50]))
    figure, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    axes[0].imshow(image, origin='lower', interpolation='nearest')
    axes[0].set(title='HWT-grid surface classification', xlabel='HWT grid column', ylabel='HWT grid row')
    axes[1].imshow(lake, origin='lower', interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
    axes[1].set(title='Inland-lake fraction', xlabel='HWT grid column', ylabel='HWT grid row')
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return Path(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grid', required=True, help='HWT sample or conus_03km_lcc_grid.nc4 file')
    parser.add_argument('--output', required=True, help='HWT-grid static NetCDF to create')
    parser.add_argument('--lat', default='lats')
    parser.add_argument('--lon', default='lons')
    parser.add_argument('--scale', choices=('c', 'l', 'i', 'h', 'f'), default='f',
                        help='GSHHG resolution; f is full resolution')
    parser.add_argument('--supersample', type=int, default=4,
                        help='Sub-cell samples per grid axis; 4 means 16 samples per HWT cell')
    parser.add_argument('--chunk-rows', type=int, default=8)
    parser.add_argument('--cartopy-data-dir', help='Existing Cartopy data cache for offline use')
    parser.add_argument('--plot', help='Optional PNG preview of ocean, land, and lake fractions')
    args = parser.parse_args()
    if args.cartopy_data_dir:
        import cartopy
        cartopy.config['pre_existing_data_dir'] = Path(args.cartopy_data_dir)
    lat, lon, dims, coords = load_hwt_grid(args.grid, args.lat, args.lon)
    polygon_lon = gshhg_longitudes(lon)
    print(f'HWT grid {lat.shape}: raw longitude {lon.min():.2f}..{lon.max():.2f}, '
          f'GSHHG longitude {polygon_lon.min():.2f}..{polygon_lon.max():.2f}, '
          f'latitude {lat.min():.2f}..{lat.max():.2f}', flush=True)
    levels = gshhg_geometries(polygon_lon, lat, args.scale)
    values = fractions(lat, polygon_lon, levels, args.supersample, args.chunk_rows)
    print(write_output(args.output, lat, lon, dims, coords, values, args.grid, args.scale, args.supersample))
    if args.plot:
        print(write_plot(args.plot, values))


if __name__ == '__main__':
    main()
