#!/usr/bin/env python3
"""
scripts/regrid_lowres.py
========================
Crops global GEOS-FP diagnostics (tavg1_2d_slv_Nx and tavg1_2d_flx_Nx) to CONUS
and regrids them onto the target 3 km LCC grid (1059 x 1799).

Method:
  - Bilinear interpolation for continuous fields (T2M, winds, pressure, humidity)
  - Conservative area-weighted interpolation for precipitation fluxes (PRECTOT, PRECCON, etc.)
  - Precomputes and caches xESMF regridding weights to speed up batch processing.

Usage on Discover:
    python scripts/regrid_lowres.py \
        --lowres_dir /gpfsm/dnb06/projects/p174/f5295_fp/diag/Y2025/M01 \
        --highres_sample /gpfsm/dnb08/projects/p38/HWT/CONUS02KM/Feature-c2160_L137/holding/hwt_30mn_slv_LCC/202501/Feature-c2160_L137.hwt_30mn_slv_LCC.20250131_2330z.nc4 \
        --output_dir /gpfsm/dnb10/projects/p311/ML_downscaling/data/lowres_lcc_1hr/202501 \
        --weights_dir /gpfsm/dnb10/projects/p311/ML_downscaling/data/weights \
        --date 20250108
"""

import os
import glob
import argparse
from pathlib import Path
import numpy as np
import xarray as xr
import xesmf as xe
from tqdm import tqdm


def get_grid_definitions(highres_sample_path: str, buffer_deg: float = 2.0):
    """Load target LCC grid and compute cropped source bounding box."""
    print(f"Loading high-resolution LCC grid from: {highres_sample_path}")
    ds_hr = xr.open_dataset(highres_sample_path)

    # 2D target coordinates in LCC
    target_lats = ds_hr["lats"].values  # (1059, 1799)
    target_lons = ds_hr["lons"].values  # (1059, 1799)

    # Standardize longitudes to [-180, 180]
    target_lons = np.where(target_lons > 180.0, target_lons - 360.0, target_lons)

    lat_min = float(np.min(target_lats) - buffer_deg)
    lat_max = float(np.max(target_lats) + buffer_deg)
    lon_min = float(np.min(target_lons) - buffer_deg)
    lon_max = float(np.max(target_lons) + buffer_deg)

    print(f"Target LCC Dimensions: Ydim={target_lats.shape[0]}, Xdim={target_lats.shape[1]}")
    print(f"CONUS Bounding Box with {buffer_deg}° buffer:")
    print(f"  Latitude:  [{lat_min:.2f}°, {lat_max:.2f}°]")
    print(f"  Longitude: [{lon_min:.2f}°, {lon_max:.2f}°]")

    # Create target xESMF grid dataset with CF attributes
    grid_target = xr.Dataset({
        "lat": (["y", "x"], target_lats, {"units": "degrees_north", "standard_name": "latitude"}),
        "lon": (["y", "x"], target_lons, {"units": "degrees_east", "standard_name": "longitude"}),
    })

    bbox = {
        "lat_min": lat_min, "lat_max": lat_max,
        "lon_min": lon_min, "lon_max": lon_max
    }

    return grid_target, bbox


def crop_lowres(ds: xr.Dataset, bbox: dict) -> xr.Dataset:
    """Crop global regular lat-lon dataset to CONUS bounding box."""
    # Ensure longitudes are in [-180, 180]
    if (ds["lon"].values > 180.0).any():
        ds = ds.assign_coords(lon=np.where(ds["lon"].values > 180.0, ds["lon"].values - 360.0, ds["lon"].values))
        ds = ds.sortby("lon")

    cropped = ds.sel(
        lat=slice(bbox["lat_min"], bbox["lat_max"]),
        lon=slice(bbox["lon_min"], bbox["lon_max"])
    )
    return cropped


def setup_regridders(sample_lowres_path: str, grid_target: xr.Dataset, bbox: dict, weights_dir: str):
    """Build and cache xESMF bilinear regridder for continuous conditioning fields."""
    os.makedirs(weights_dir, exist_ok=True)
    w_bilinear = os.path.join(weights_dir, "regrid_weights_bilinear_conus.nc")

    print("Initializing source cropped grid for regridding weights...")
    ds_src_raw = xr.open_dataset(sample_lowres_path)
    ds_src = crop_lowres(ds_src_raw, bbox)

    grid_src = xr.Dataset({
        "lat": (["lat"], ds_src["lat"].values, {"units": "degrees_north", "standard_name": "latitude"}),
        "lon": (["lon"], ds_src["lon"].values, {"units": "degrees_east", "standard_name": "longitude"}),
    })

    print(f"Source cropped grid size: lat={len(grid_src['lat'])}, lon={len(grid_src['lon'])}")

    print(f"Setting up Bilinear regridder (cached at {w_bilinear})...")
    regridder = xe.Regridder(
        grid_src, grid_target, method="bilinear",
        filename=w_bilinear, reuse_weights=os.path.exists(w_bilinear)
    )

    return regridder


def process_hourly_step(slv_path: str, flx_path: str, out_path: str,
                         bbox: dict, regridder):
    """Regrids one hourly step combining slv and flx into a single LCC NetCDF-4."""
    if os.path.exists(out_path):
        return

    # 1. Open datasets
    ds_slv_raw = xr.open_dataset(slv_path)
    ds_flx_raw = xr.open_dataset(flx_path)

    # 2. Crop to CONUS
    ds_slv = crop_lowres(ds_slv_raw, bbox)
    ds_flx = crop_lowres(ds_flx_raw, bbox)

    # 3. Bilinear regridding of continuous state
    t2m_lcc = regridder(ds_slv["T2M"])
    qv2m_lcc = regridder(ds_slv["QV2M"]) if "QV2M" in ds_slv else None
    u10m_lcc = regridder(ds_slv["U10M"]) if "U10M" in ds_slv else None
    v10m_lcc = regridder(ds_slv["V10M"]) if "V10M" in ds_slv else None
    ps_lcc = regridder(ds_slv["PS"]) if "PS" in ds_slv else None
    slp_lcc = regridder(ds_slv["SLP"]) if "SLP" in ds_slv else None
    tqv_lcc = regridder(ds_slv["TQV"]) if "TQV" in ds_slv else None

    # 4. Bilinear regridding of precipitation (with zero-clipping to prevent negative drizzle)
    prectot_raw = regridder(ds_flx["PRECTOT"]).values
    prectot_lcc = np.maximum(prectot_raw, 0.0)

    preccon_lcc = np.maximum(regridder(ds_flx["PRECCON"]).values, 0.0) if "PRECCON" in ds_flx else None
    preclsc_lcc = np.maximum(regridder(ds_flx["PRECLSC"]).values, 0.0) if "PRECLSC" in ds_flx else None
    precsno_lcc = np.maximum(regridder(ds_flx["PRECSNO"]).values, 0.0) if "PRECSNO" in ds_flx else None

    # Helper to safely convert either DataArray or ndarray to 3D float32 (time, Ydim, Xdim)
    def to_3d(val):
        arr = val.values if hasattr(val, "values") else np.asarray(val)
        arr = arr.astype(np.float32)
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]
        return arr

    # 5. Combine into single output dataset
    data_vars = {
        "T2M": (["time", "Ydim", "Xdim"], to_3d(t2m_lcc), {"units": "K", "long_name": "2-meter_air_temperature"}),
        "PRECTOT": (["time", "Ydim", "Xdim"], to_3d(prectot_lcc), {"units": "kg m-2 s-1", "long_name": "total_precipitation"}),
    }
    if qv2m_lcc is not None:
        data_vars["QV2M"] = (["time", "Ydim", "Xdim"], to_3d(qv2m_lcc), {"units": "kg kg-1", "long_name": "2-meter_specific_humidity"})
    if u10m_lcc is not None and v10m_lcc is not None:
        data_vars["U10M"] = (["time", "Ydim", "Xdim"], to_3d(u10m_lcc), {"units": "m s-1"})
        data_vars["V10M"] = (["time", "Ydim", "Xdim"], to_3d(v10m_lcc), {"units": "m s-1"})
    if ps_lcc is not None:
        data_vars["PS"] = (["time", "Ydim", "Xdim"], to_3d(ps_lcc), {"units": "Pa"})
    if slp_lcc is not None:
        data_vars["SLP"] = (["time", "Ydim", "Xdim"], to_3d(slp_lcc), {"units": "Pa"})
    if tqv_lcc is not None:
        data_vars["TQV"] = (["time", "Ydim", "Xdim"], to_3d(tqv_lcc), {"units": "kg m-2"})
    if preccon_lcc is not None:
        data_vars["PRECCON"] = (["time", "Ydim", "Xdim"], to_3d(preccon_lcc), {"units": "kg m-2 s-1"})
    if preclsc_lcc is not None:
        data_vars["PRECLSC"] = (["time", "Ydim", "Xdim"], to_3d(preclsc_lcc), {"units": "kg m-2 s-1"})

    ds_out = xr.Dataset(data_vars=data_vars, coords={"time": ds_slv["time"].values})
    
    # Save with compression
    encoding = {var: {"zlib": True, "complevel": 4} for var in ds_out.data_vars}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ds_out.to_netcdf(out_path, encoding=encoding)


def main():
    parser = argparse.ArgumentParser(description="Regrid GEOS-FP low-res diagnostics to 3 km LCC grid.")
    parser.add_argument("--lowres_dir", type=str, required=True, help="Directory containing tavg1_2d_slv_Nx and flx files")
    parser.add_argument("--highres_sample", type=str, required=True, help="Path to one sample high-res LCC file")
    parser.add_argument("--output_dir", type=str, required=True, help="Destination directory for regridded LCC files")
    parser.add_argument("--weights_dir", type=str, default="./data/weights", help="Directory to save/load regridding weights")
    parser.add_argument("--date", type=str, default=None, help="Specific date to process (e.g., 20250108), or all if not set")
    args = parser.parse_args()

    # Load grids
    grid_target, bbox = get_grid_definitions(args.highres_sample)

    # Find matching slv and flx files
    date_pattern = f"*{args.date}*.nc4" if args.date else "*.nc4"
    slv_files = sorted(glob.glob(os.path.join(args.lowres_dir, f"f5295_fp.tavg1_2d_slv_Nx.{date_pattern}")))
    print(f"Found {len(slv_files)} slv files to process.")

    if not slv_files:
        print("No matching slv files found! Exiting.")
        return

    # Build or load regridder
    regridder = setup_regridders(slv_files[0], grid_target, bbox, args.weights_dir)

    # Process all hours
    os.makedirs(args.output_dir, exist_ok=True)
    for slv_path in tqdm(slv_files, desc="Regridding hourly steps"):
        basename = os.path.basename(slv_path)
        timestamp = basename.split(".")[2]  # e.g., 20250108_0030z

        flx_name = f"f5295_fp.tavg1_2d_flx_Nx.{timestamp}.nc4"
        flx_path = os.path.join(args.lowres_dir, flx_name)
        if not os.path.exists(flx_path):
            print(f"Warning: Missing companion flux file: {flx_path}, skipping.")
            continue

        out_name = f"f5295_fp.lowres_lcc_1hr.{timestamp}.nc4"
        out_path = os.path.join(args.output_dir, out_name)

        process_hourly_step(slv_path, flx_path, out_path, bbox, regridder)

    print(f"\nAll hourly files regridded successfully to: {args.output_dir}")


if __name__ == "__main__":
    main()
