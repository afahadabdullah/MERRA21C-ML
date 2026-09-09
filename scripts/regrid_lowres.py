#!/usr/bin/env python3
"""
scripts/regrid_lowres.py
========================
Production-grade, Resumable & Parallel Low-Res GEOS-FP to 3 km LCC Regridding Pipeline.

Features:
  - Resumable: Automatically detects and skips existing, valid files.
  - Atomic writes: Writes to temporary files (*.tmp.nc4) and renames on success, preventing file corruption.
  - Comprehensive variable set:
      State (slv): T2M, TS, T10M, QV2M, QV10M, U10M, V10M, U2M, V2M, PS, SLP,
                   TQV, TQL, TQI, CLDPRS, CLDTMP, OMEGA500, PBLTOP
      Fluxes (flx): PRECTOT, PRECCON, PRECLSC, PRECANV, PRECSNO, PGENTOT, PREVTOT,
                    EVAP, EFLUX, HFLUX, PBLH, USTAR
  - Multi-worker Parallelism: Parallel processing across multiple CPU cores.
  - Caches xESMF bilinear weights to disk for instant loading.

Usage on Discover:
    # Single day test:
    python scripts/regrid_lowres.py --date 20250108 --num_workers 4

    # Full month parallel run:
    python scripts/regrid_lowres.py --num_workers 8
"""

import os
import glob
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import xarray as xr
import xesmf as xe
from tqdm import tqdm

# Global worker variables for ProcessPoolExecutor
_WORKER_REGRIDDER = None
_WORKER_BBOX = None

# Variables consumed by configs/discover.yaml and configs/a100_80gb.yaml.  A
# legacy output containing only T2M/PRECTOT is not complete enough to train.
MODEL_REQUIRED_SLV_VARS = {
    "T2M", "QV2M", "U10M", "V10M", "PS", "SLP", "TQV", "OMEGA500",
}
MODEL_REQUIRED_FLX_VARS = {"PRECTOT", "PRECCON", "PRECLSC"}
MODEL_REQUIRED_OUTPUT_VARS = MODEL_REQUIRED_SLV_VARS | MODEL_REQUIRED_FLX_VARS


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
    if (ds["lon"].values > 180.0).any():
        ds = ds.assign_coords(lon=np.where(ds["lon"].values > 180.0, ds["lon"].values - 360.0, ds["lon"].values))
        ds = ds.sortby("lon")

    return ds.sel(
        lat=slice(bbox["lat_min"], bbox["lat_max"]),
        lon=slice(bbox["lon_min"], bbox["lon_max"])
    )


def precompute_weights(sample_lowres_path: str, grid_target: xr.Dataset, bbox: dict, weights_path: str):
    """Generates and saves the bilinear regridding weights if not already present."""
    os.makedirs(os.path.dirname(weights_path), exist_ok=True)
    if os.path.exists(weights_path):
        print(f"Regridding weights already cached at: {weights_path}")
        return

    print("Building and caching xESMF Bilinear regridder weights...")
    ds_src_raw = xr.open_dataset(sample_lowres_path)
    ds_src = crop_lowres(ds_src_raw, bbox)

    grid_src = xr.Dataset({
        "lat": (["lat"], ds_src["lat"].values, {"units": "degrees_north", "standard_name": "latitude"}),
        "lon": (["lon"], ds_src["lon"].values, {"units": "degrees_east", "standard_name": "longitude"}),
    })

    # Build and cache to disk
    xe.Regridder(grid_src, grid_target, method="bilinear", filename=weights_path, reuse_weights=False)
    print(f"Weights successfully saved to: {weights_path}")


def init_worker(weights_path: str, highres_sample_path: str, sample_lowres_path: str, buffer_deg: float):
    """Initializes worker process with its own regridder instance from cached weights."""
    global _WORKER_REGRIDDER, _WORKER_BBOX
    grid_target, bbox = get_grid_definitions(highres_sample_path, buffer_deg)
    _WORKER_BBOX = bbox

    ds_src_raw = xr.open_dataset(sample_lowres_path)
    ds_src = crop_lowres(ds_src_raw, bbox)
    grid_src = xr.Dataset({
        "lat": (["lat"], ds_src["lat"].values, {"units": "degrees_north", "standard_name": "latitude"}),
        "lon": (["lon"], ds_src["lon"].values, {"units": "degrees_east", "standard_name": "longitude"}),
    })

    _WORKER_REGRIDDER = xe.Regridder(grid_src, grid_target, method="bilinear", filename=weights_path, reuse_weights=True)


def is_file_valid(filepath: str) -> bool:
    """Check if file exists, is non-empty, and has valid NetCDF-4 contents."""
    if not os.path.exists(filepath):
        return False
    if os.path.getsize(filepath) < 50_000:  # Suspiciously small (< 50KB)
        return False
    try:
        with xr.open_dataset(filepath) as ds:
            return MODEL_REQUIRED_OUTPUT_VARS.issubset(ds.data_vars)
    except Exception:
        return False


def to_3d(val):
    """Helper to convert either DataArray or ndarray to 3D float32 (time, Ydim, Xdim)."""
    arr = val.values if hasattr(val, "values") else np.asarray(val)
    arr = arr.astype(np.float32)
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    return arr


def process_single_step(item: tuple) -> str:
    """Worker task to process one hourly step."""
    global _WORKER_REGRIDDER, _WORKER_BBOX
    slv_path, flx_path, out_path = item

    # 1. Resumability: Skip if already done and valid
    if is_file_valid(out_path):
        return f"SKIPPED: {os.path.basename(out_path)}"

    regridder = _WORKER_REGRIDDER
    bbox = _WORKER_BBOX

    # 2. Open datasets
    with xr.open_dataset(slv_path) as ds_slv_raw, xr.open_dataset(flx_path) as ds_flx_raw:
        missing_slv = sorted(MODEL_REQUIRED_SLV_VARS - set(ds_slv_raw.data_vars))
        missing_flx = sorted(MODEL_REQUIRED_FLX_VARS - set(ds_flx_raw.data_vars))
        if missing_slv or missing_flx:
            details = []
            if missing_slv:
                details.append(f"{slv_path}: {missing_slv}")
            if missing_flx:
                details.append(f"{flx_path}: {missing_flx}")
            raise KeyError("Missing model-required source variables; " + "; ".join(details))
        ds_slv = crop_lowres(ds_slv_raw, bbox)
        ds_flx = crop_lowres(ds_flx_raw, bbox)

        # 3. Regrid continuous state fields (from slv)
        data_vars = {
            "T2M": (["time", "Ydim", "Xdim"], to_3d(regridder(ds_slv["T2M"])), {"units": "K", "long_name": "2-meter_air_temperature"}),
        }

        # Optional state variables
        state_vars = [
            ("TS", "K", "surface_skin_temperature"),
            ("QV2M", "kg kg-1", "2-meter_specific_humidity"),
            ("T10M", "K", "10-meter_air_temperature"),
            ("QV10M", "kg kg-1", "10-meter_specific_humidity"),
            ("U10M", "m s-1", "10-meter_eastward_wind"),
            ("V10M", "m s-1", "10-meter_northward_wind"),
            ("U2M", "m s-1", "2-meter_eastward_wind"),
            ("V2M", "m s-1", "2-meter_northward_wind"),
            ("PS", "Pa", "surface_pressure"),
            ("SLP", "Pa", "sea_level_pressure"),
            ("TQV", "kg m-2", "total_precipitable_water_vapor"),
            ("TQL", "kg m-2", "total_precipitable_liquid_water"),
            ("TQI", "kg m-2", "total_precipitable_ice_water"),
            ("CLDPRS", "Pa", "cloud_top_pressure"),
            ("CLDTMP", "K", "cloud_top_temperature"),
            ("OMEGA500", "Pa s-1", "omega_at_500_hPa"),
            ("PBLTOP", "Pa", "pbltop_pressure"),
        ]
        for var_name, units, long_name in state_vars:
            if var_name in ds_slv:
                data_vars[var_name] = (["time", "Ydim", "Xdim"], to_3d(regridder(ds_slv[var_name])), {"units": units, "long_name": long_name})

        # 4. Regrid flux fields (from flx) with non-negativity clipping for precip
        prectot_raw = regridder(ds_flx["PRECTOT"]).values
        data_vars["PRECTOT"] = (["time", "Ydim", "Xdim"], to_3d(np.maximum(prectot_raw, 0.0)), {"units": "kg m-2 s-1", "long_name": "total_precipitation"})

        flux_vars_clip = [
            ("PRECCON", "kg m-2 s-1", "convective_precipitation"),
            ("PRECLSC", "kg m-2 s-1", "nonanvil_large_scale_precipitation"),
            ("PRECANV", "kg m-2 s-1", "anvil_precipitation"),
            ("PRECSNO", "kg m-2 s-1", "snowfall"),
            ("PGENTOT", "kg m-2 s-1", "total_column_production_of_precipitation"),
            ("PREVTOT", "kg m-2 s-1", "total_column_re-evaporation_of_precipitation"),
            ("EVAP", "kg m-2 s-1", "evaporation_from_turbulence"),
        ]
        for var_name, units, long_name in flux_vars_clip:
            if var_name in ds_flx:
                clipped = np.maximum(regridder(ds_flx[var_name]).values, 0.0)
                data_vars[var_name] = (["time", "Ydim", "Xdim"], to_3d(clipped), {"units": units, "long_name": long_name})

        # Other fluxes (can be positive or negative)
        other_fluxes = [
            ("EFLUX", "W m-2", "total_latent_energy_flux"),
            ("HFLUX", "W m-2", "sensible_heat_flux_from_turbulence"),
            ("PBLH", "m", "planetary_boundary_layer_height"),
            ("USTAR", "m s-1", "surface_velocity_scale"),
        ]
        for var_name, units, long_name in other_fluxes:
            if var_name in ds_flx:
                data_vars[var_name] = (["time", "Ydim", "Xdim"], to_3d(regridder(ds_flx[var_name])), {"units": units, "long_name": long_name})

        ds_out = xr.Dataset(data_vars=data_vars, coords={"time": ds_slv["time"].values})

        # 5. Atomic write via temporary file
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        tmp_path = f"{out_path}.tmp.{os.getpid()}.nc4"
        encoding = {var: {"zlib": True, "complevel": 4} for var in ds_out.data_vars}

        ds_out.to_netcdf(tmp_path, encoding=encoding)
        os.replace(tmp_path, out_path)

    return f"COMPLETED: {os.path.basename(out_path)}"


def main():
    parser = argparse.ArgumentParser(description="Production-grade, resumable regridder for GEOS-FP to 3 km LCC.")
    parser.add_argument(
        "--lowres_root",
        type=str,
        default="/gpfsm/dnb06/projects/p174/f5295_fp/diag",
        help="Root directory containing YYYYY/MM subdirectories",
    )
    parser.add_argument(
        "--year",
        type=str,
        default="2025",
        help="Year to process (default: 2025)",
    )
    parser.add_argument(
        "--month",
        type=str,
        default=None,
        help="Month to process (e.g. 1 to 12). If omitted, processes all available months of the year.",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Filter for specific date (e.g. 20250108). Optional.",
    )
    parser.add_argument(
        "--highres_sample",
        type=str,
        default="/gpfsm/dnb08/projects/p38/HWT/CONUS02KM/Feature-c2160_L137/holding/hwt_30mn_slv_LCC/202501/Feature-c2160_L137.hwt_30mn_slv_LCC.20250131_2330z.nc4",
        help="Sample high-res file for target grid geometry",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="/gpfsm/dnb10/projects/p311/ML_downscaling/data/lowres_lcc_1hr",
        help="Root destination directory for regridded LCC files (subfolders YYYYMM created automatically)",
    )
    parser.add_argument(
        "--weights_path",
        type=str,
        default="/gpfsm/dnb10/projects/p311/ML_downscaling/data/weights/regrid_weights_bilinear_conus.nc",
        help="Path for cached xESMF regridding weights",
    )
    parser.add_argument("--num_workers", type=int, default=8, help="Number of parallel worker processes")
    args = parser.parse_args()

    # 1. Determine input directories to scan
    year_dir = os.path.join(args.lowres_root, f"Y{args.year}")
    if not os.path.exists(year_dir):
        # Fallback if lowres_root is already a monthly or direct path
        if os.path.exists(args.lowres_root):
            month_dirs = [args.lowres_root]
        else:
            raise FileNotFoundError(f"Directory not found: {year_dir}")
    elif args.month is not None:
        m_int = int(args.month)
        m_dir = os.path.join(year_dir, f"M{m_int:02d}")
        month_dirs = [m_dir] if os.path.exists(m_dir) else []
    else:
        month_dirs = sorted(glob.glob(os.path.join(year_dir, "M*")))

    print(f"=== Regridding GEOS-FP for Year {args.year} ===")
    print(f"Scanning {len(month_dirs)} monthly directories: {[os.path.basename(d) for d in month_dirs]}")

    # 2. Collect all slv files across selected months
    slv_files = []
    pattern = f"*{args.date}*.nc4" if args.date else "*.nc4"
    for m_dir in month_dirs:
        m_files = sorted(glob.glob(os.path.join(m_dir, f"f5295_fp.tavg1_2d_slv_Nx.{pattern}")))
        slv_files.extend(m_files)

    print(f"Total matching slv files found across selected period: {len(slv_files)}")
    if not slv_files:
        print("No input files found. Exiting.")
        return

    # 3. Grid geometry & precompute weights once
    grid_target, bbox = get_grid_definitions(args.highres_sample)
    precompute_weights(slv_files[0], grid_target, bbox, args.weights_path)

    # 4. Build task list & check existing files (resumable)
    tasks = []
    skipped_count = 0
    for slv_path in slv_files:
        basename = os.path.basename(slv_path)
        timestamp = basename.split(".")[2]  # e.g., 20250108_0030z
        month_tag = timestamp[:6]          # e.g., 202501

        m_dir = os.path.dirname(slv_path)
        flx_path = os.path.join(m_dir, f"f5295_fp.tavg1_2d_flx_Nx.{timestamp}.nc4")
        if not os.path.exists(flx_path):
            continue

        out_dir = os.path.join(args.output_root, month_tag)
        out_path = os.path.join(out_dir, f"f5295_fp.lowres_lcc_1hr.{timestamp}.nc4")

        if is_file_valid(out_path):
            skipped_count += 1
            continue
        tasks.append((slv_path, flx_path, out_path))

    print(f"Tasks to execute: {len(tasks)} (Already completed and verified: {skipped_count})")
    if not tasks:
        print("All requested files are already processed and verified! Nothing to do.")
        return

    # 5. Parallel execution with ProcessPoolExecutor
    print(f"Launching processing pool with {args.num_workers} parallel workers...")
    with ProcessPoolExecutor(
        max_workers=args.num_workers,
        initializer=init_worker,
        initargs=(args.weights_path, args.highres_sample, slv_files[0], 2.0)
    ) as executor:
        futures = {executor.submit(process_single_step, task): task for task in tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Regridding progress"):
            res = future.result()

    print(f"\nAll operations completed successfully! Output stored in: {args.output_root}")


if __name__ == "__main__":
    main()
