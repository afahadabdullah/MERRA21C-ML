#!/usr/bin/env python3
"""
scripts/plot_diagnostics.py
===========================
Diagnostic comparison tool for MERRA21C-ML downscaling project.

Compares across 4 columns:
  Column 1: Raw Low-Res Predictor (Native 0.25° GEOS-FP grid, discrete raw pixels)
  Column 2: Low-Res Interpolated / Regridded Predictor (Bilinear to 3 km LCC)
  Column 3: High-Res Ground Truth (HWT 3 km LCC simulation)
  Column 4: Difference / Bias (Interpolated Low-Res - High-Res Target)

Rows:
  Row 1: 2m Temperature (K -> °C)
  Row 2: Total Precipitation Rate (kg m-2 s-1 -> mm hr-1)

Display:
  - Uses imshow / pcolormesh flat rasterization without smoothing or contour interpolation,
    so raw 25 km discrete pixels and 3 km resolved features are crisply preserved.
  - Cartopy publication-quality maps with CONUS Lambert Conformal Conic projections.
  - Regional Orographic Zoom panel over complex terrain (Southern/Central Rockies).
"""

import os
import re
import glob
import random
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless supercomputing
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature


def parse_args():
    parser = argparse.ArgumentParser(description="Plot diagnostic comparisons between raw low-res, regridded low-res, and high-res data.")
    parser.add_argument(
        "--lowres_regrid_dir",
        type=str,
        default="data/lowres_lcc_1hr",
        help="Root directory of regridded low-res NetCDF files.",
    )
    parser.add_argument(
        "--lowres_raw_root",
        type=str,
        default="/gpfsm/dnb06/projects/p174/f5295_fp/diag",
        help="Root directory of raw native GEOS-FP diagnostics (Y2025/MXX).",
    )
    parser.add_argument(
        "--highres_dir",
        type=str,
        default="/gpfsm/dnb08/projects/p38/HWT/CONUS02KM/Feature-c2160_L137/holding/hwt_30mn_slv_LCC",
        help="Root directory of high-res HWT NetCDF files.",
    )
    parser.add_argument(
        "--static_grid",
        type=str,
        default="data/static_grid/hwt_static_3km_grid.nc",
        help="Path to static grid file with elevation, lats, and lons.",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Specific date to plot in YYYYMMDD format (e.g. 20250108).",
    )
    parser.add_argument(
        "--hour",
        type=int,
        default=None,
        help="Specific hour to plot (0-23). If None with --date, picks first available.",
    )
    parser.add_argument(
        "--random_samples",
        type=int,
        default=2,
        help="Number of random matching timestamp pairs to plot if --date is not specified.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="plots/diagnostics",
        help="Directory where diagnostic figures will be saved.",
    )
    return parser.parse_args()


def extract_timestamp_from_filename(filename: str):
    """Extract YYYYMMDD_HHMM from filename regardless of prefixes."""
    match = re.search(r"(\d{8})_(\d{4})z?", filename, re.IGNORECASE)
    if match:
        date_part, time_part = match.group(1), match.group(2)
        try:
            return datetime.strptime(f"{date_part}_{time_part}", "%Y%m%d_%H%M")
        except ValueError:
            return None
    return None


def find_matching_triplets(regrid_dir: str, raw_root: str, highres_dir: str):
    """
    Find timestamps where regridded low-res, raw native GEOS-FP, and high-res data are all present.
    """
    print(f"Scanning for available regridded low-res files in: {regrid_dir}")
    regrid_files = sorted(glob.glob(os.path.join(regrid_dir, "**", "*.nc4"), recursive=True))
    if not regrid_files:
        regrid_files = sorted(glob.glob(os.path.join(regrid_dir, "*.nc4")))

    print(f"Found {len(regrid_files)} low-res regridded files.")
    if not regrid_files:
        return []

    triplets = []
    for lr_path in regrid_files:
        fname = os.path.basename(lr_path)
        dt = extract_timestamp_from_filename(fname)
        if dt is None:
            continue

        ym = dt.strftime("%Y%m")
        # 1. High-res target candidates (HH30z or HH00z)
        hr_cand_30 = os.path.join(
            highres_dir, ym, f"Feature-c2160_L137.hwt_30mn_slv_LCC.{dt.strftime('%Y%m%d_%H30')}z.nc4"
        )
        hr_cand_00 = os.path.join(
            highres_dir, ym, f"Feature-c2160_L137.hwt_30mn_slv_LCC.{dt.strftime('%Y%m%d_%H00')}z.nc4"
        )
        hr_path = hr_cand_30 if os.path.exists(hr_cand_30) else (hr_cand_00 if os.path.exists(hr_cand_00) else None)

        if hr_path is None:
            continue

        # 2. Raw low-res files (slv and flx)
        # Location: raw_root/Y2025/M01/f5295_fp.tavg1_2d_slv_Nx.YYYYMMDD_HH30z.nc4
        raw_month_dir = os.path.join(raw_root, f"Y{dt.year}", f"M{dt.month:02d}")
        raw_slv = os.path.join(raw_month_dir, f"f5295_fp.tavg1_2d_slv_Nx.{dt.strftime('%Y%m%d_%H%M')}z.nc4")
        raw_flx = os.path.join(raw_month_dir, f"f5295_fp.tavg1_2d_flx_Nx.{dt.strftime('%Y%m%d_%H%M')}z.nc4")

        # Fallback if raw_month_dir doesn't exist or files slightly different
        if not (os.path.exists(raw_slv) and os.path.exists(raw_flx)):
            # Try flat or alternate search
            raw_slv = None
            raw_flx = None

        triplets.append({
            "regrid_path": lr_path,
            "highres_path": hr_path,
            "raw_slv": raw_slv,
            "raw_flx": raw_flx,
            "datetime": dt
        })

    print(f"Found {len(triplets)} matching sets between low-res and high-res.")
    return triplets


def load_static_grid(static_path: str, sample_hr_file: str):
    """Load or extract static lats, lons, and elevation."""
    if os.path.exists(static_path):
        print(f"Loading static grid from: {static_path}")
        ds = xr.open_dataset(static_path)
        lats = ds["lats"].values
        lons = ds["lons"].values
        elev = ds["elevation"].values if "elevation" in ds else None
        return lats, lons, elev

    print(f"Static grid not found at {static_path}. Extracting dynamically from {sample_hr_file}...")
    with xr.open_dataset(sample_hr_file) as ds:
        lats = ds["lats"].values
        lons = ds["lons"].values
        elev = None
        if "HGT_SFC" in ds:
            hgt = ds["HGT_SFC"].values
            if hgt.ndim == 3:
                hgt = hgt[0]
            elev = hgt / 9.80665
    return lats, lons, elev


def add_map_features(ax):
    """Add standard geographic boundaries for publication-quality CONUS maps."""
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.8, edgecolor="#222222", zorder=3)
    ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.5, edgecolor="#555555", linestyle=":", zorder=3)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.8, edgecolor="#222222", zorder=3)
    ax.add_feature(cfeature.LAKES.with_scale("50m"), facecolor="none", edgecolor="#333333", linewidth=0.5, zorder=3)


def plot_comparison_panel(item, lats, lons, elev, out_dir):
    """
    Creates a 2x4 comprehensive diagnostic panel:
      Col 1: Raw Low-Res (Native 0.25° GEOS-FP, discrete raster pixels)
      Col 2: Interpolated Low-Res (Bilinear to 3 km LCC)
      Col 3: High-Res Ground Truth (HWT 3 km simulation)
      Col 4: Difference (Interpolated Low-Res - High-Res Target)
    """
    os.makedirs(out_dir, exist_ok=True)
    dt = item["datetime"]
    stamp = dt.strftime("%Y%m%d_%H%Mz")
    print(f"\n---> Plotting 4-Column Diagnostic Panel for {stamp}...")

    # 1. Read Low-Res Regridded Fields
    with xr.open_dataset(item["regrid_path"]) as ds_lr:
        lr_t2m = ds_lr["T2M"].values
        if lr_t2m.ndim == 3:
            lr_t2m = lr_t2m[0]
        lr_t2m_c = lr_t2m - 273.15  # Convert to Celsius

        lr_precip = ds_lr["PRECTOT"].values
        if lr_precip.ndim == 3:
            lr_precip = lr_precip[0]
        lr_precip_mm = lr_precip * 3600.0  # kg m-2 s-1 -> mm/hr

    # 2. Read High-Res Fields
    with xr.open_dataset(item["highres_path"]) as ds_hr:
        t_var = "TMP_2M" if "TMP_2M" in ds_hr else "T2M"
        hr_t2m = ds_hr[t_var].values
        if hr_t2m.ndim == 3:
            hr_t2m = hr_t2m[0]
        hr_t2m_c = hr_t2m - 273.15

        p_var = "PRECTOT" if "PRECTOT" in ds_hr else ("APCP" if "APCP" in ds_hr else None)
        if p_var is None:
            raise KeyError("Neither PRECTOT nor APCP found in high-res dataset!")
        hr_precip = ds_hr[p_var].values
        if hr_precip.ndim == 3:
            hr_precip = hr_precip[0]
        hr_precip_mm = hr_precip * 3600.0 if p_var == "PRECTOT" else hr_precip

    # 3. Read Raw Native Low-Res Fields (if available)
    raw_t2m_c, raw_precip_mm, raw_lats, raw_lons = None, None, None, None
    if item["raw_slv"] and os.path.exists(item["raw_slv"]) and item["raw_flx"] and os.path.exists(item["raw_flx"]):
        with xr.open_dataset(item["raw_slv"]) as ds_r_slv:
            # Crop roughly to CONUS extent with buffer
            lons_norm = np.where(ds_r_slv["lon"].values > 180.0, ds_r_slv["lon"].values - 360.0, ds_r_slv["lon"].values)
            ds_r_slv = ds_r_slv.assign_coords(lon=lons_norm).sortby("lon")
            ds_crop_slv = ds_r_slv.sel(lat=slice(20.0, 55.0), lon=slice(-130.0, -65.0))
            raw_lats = ds_crop_slv["lat"].values
            raw_lons = ds_crop_slv["lon"].values
            r_t = ds_crop_slv["T2M"].values
            if r_t.ndim == 3:
                r_t = r_t[0]
            raw_t2m_c = r_t - 273.15

        with xr.open_dataset(item["raw_flx"]) as ds_r_flx:
            lons_norm = np.where(ds_r_flx["lon"].values > 180.0, ds_r_flx["lon"].values - 360.0, ds_r_flx["lon"].values)
            ds_r_flx = ds_r_flx.assign_coords(lon=lons_norm).sortby("lon")
            ds_crop_flx = ds_r_flx.sel(lat=slice(20.0, 55.0), lon=slice(-130.0, -65.0))
            r_p = ds_crop_flx["PRECTOT"].values
            if r_p.ndim == 3:
                r_p = r_p[0]
            raw_precip_mm = np.maximum(r_p * 3600.0, 0.0)

    # 4. Differences (Low-Res Interpolated - High-Res Target)
    diff_t2m = lr_t2m_c - hr_t2m_c
    diff_precip = lr_precip_mm - hr_precip_mm

    # Normalization & Colormaps
    t_min = float(np.percentile(hr_t2m_c[~np.isnan(hr_t2m_c)], 1))
    t_max = float(np.percentile(hr_t2m_c[~np.isnan(hr_t2m_c)], 99))
    t_cmap = "coolwarm"

    precip_levels = [0.05, 0.2, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 35.0, 50.0]
    precip_cmap = plt.cm.get_cmap("YlGnBu").copy()
    precip_cmap.set_under("white")
    precip_norm = mcolors.BoundaryNorm(boundaries=precip_levels, ncolors=precip_cmap.N, extend="max")

    diff_t_norm = mcolors.TwoSlopeNorm(vmin=-10.0, vcenter=0.0, vmax=10.0)
    diff_t_cmap = "bwr"

    diff_p_norm = mcolors.TwoSlopeNorm(vmin=-15.0, vcenter=0.0, vmax=15.0)
    diff_p_cmap = "BrBG"

    # Setup Projections & Bounds
    proj = ccrs.LambertConformal(central_longitude=-96.0, central_latitude=37.5, standard_parallels=(30, 45))
    data_crs = ccrs.PlateCarree()

    # Determine CONUS map extent from high-res grid
    extent = [-125.0, -66.5, 23.0, 50.5]

    fig, axes = plt.subplots(
        2, 4,
        figsize=(28, 12),
        subplot_kw={"projection": proj},
        constrained_layout=True
    )

    fig.suptitle(
        f"MERRA21C-ML Multi-Scale Comparison (Raw 25 km vs Interpolated vs 3 km Ground Truth)\nTimestamp: {dt.strftime('%Y-%m-%d %H:%M UTC')}",
        fontsize=19,
        fontweight="bold",
        y=0.98,
    )

    # For imshow on PlateCarree / LCC:
    # To plot raw discrete grid cells without smoothing or bilinear contour interpolation,
    # pcolormesh with shading="nearest" or "auto" or imshow with extent produces crisp non-interpolated pixels.
    step = 2
    lons_sub = lons[::step, ::step]
    lats_sub = lats[::step, ::step]

    # =========================================================================
    # ROW 1: TEMPERATURE (2m, °C)
    # =========================================================================

    # [1, 1] Col 1: Raw Low-Res (Native 0.25° GEOS-FP)
    ax = axes[0, 0]
    add_map_features(ax)
    ax.set_extent(extent, crs=data_crs)
    if raw_t2m_c is not None:
        # Use imshow with PlateCarree coordinate bounds to see crisp discrete 25 km pixels
        im1 = ax.imshow(
            raw_t2m_c, origin="lower",
            extent=[raw_lons.min(), raw_lons.max(), raw_lats.min(), raw_lats.max()],
            transform=data_crs, cmap=t_cmap, vmin=t_min, vmax=t_max,
            interpolation="nearest"
        )
        ax.set_title("[1] Raw Native Low-Res: T2M\n(GEOS-FP ~25 km Discrete Grid)", fontsize=13, fontweight="bold")
        cb = fig.colorbar(im1, ax=ax, orientation="horizontal", pad=0.04, shrink=0.75)
        cb.set_label("2m Temperature (°C)", fontsize=11)
    else:
        ax.set_title("[1] Raw Native Low-Res (Path not found)", fontsize=13)

    # [1, 2] Col 2: Interpolated Low-Res (Bilinear to 3 km LCC)
    ax = axes[0, 1]
    add_map_features(ax)
    ax.set_extent(extent, crs=data_crs)
    im2 = ax.pcolormesh(
        lons_sub, lats_sub, lr_t2m_c[::step, ::step],
        transform=data_crs, cmap=t_cmap, vmin=t_min, vmax=t_max,
        shading="nearest"  # Explicit discrete pixel boundaries, no contouring
    )
    ax.set_title("[2] Low-Res Interpolated: T2M\n(Bilinear on 3 km LCC Grid)", fontsize=13, fontweight="bold")
    cb = fig.colorbar(im2, ax=ax, orientation="horizontal", pad=0.04, shrink=0.75)
    cb.set_label("2m Temperature (°C)", fontsize=11)

    # [1, 3] Col 3: High-Res Ground Truth (HWT 3 km Simulation)
    ax = axes[0, 2]
    add_map_features(ax)
    ax.set_extent(extent, crs=data_crs)
    im3 = ax.pcolormesh(
        lons_sub, lats_sub, hr_t2m_c[::step, ::step],
        transform=data_crs, cmap=t_cmap, vmin=t_min, vmax=t_max,
        shading="nearest"
    )
    ax.set_title("[3] High-Res Ground Truth: TMP_2M\n(HWT Simulation, Resolved 3 km LCC)", fontsize=13, fontweight="bold")
    cb = fig.colorbar(im3, ax=ax, orientation="horizontal", pad=0.04, shrink=0.75)
    cb.set_label("2m Temperature (°C)", fontsize=11)

    # [1, 4] Col 4: Difference (Low-Res - High-Res)
    ax = axes[0, 3]
    add_map_features(ax)
    ax.set_extent(extent, crs=data_crs)
    im4 = ax.pcolormesh(
        lons_sub, lats_sub, diff_t2m[::step, ::step],
        transform=data_crs, cmap=diff_t_cmap, norm=diff_t_norm,
        shading="nearest"
    )
    rmse_t = np.sqrt(np.nanmean(diff_t2m**2))
    mae_t = np.nanmean(np.abs(diff_t2m))
    ax.set_title(f"[4] Difference (Interpolated - High-Res)\nRMSE: {rmse_t:.2f} °C | MAE: {mae_t:.2f} °C", fontsize=13, fontweight="bold")
    cb = fig.colorbar(im4, ax=ax, orientation="horizontal", pad=0.04, shrink=0.75)
    cb.set_label("Δ T2M (°C)", fontsize=11)

    # =========================================================================
    # ROW 2: PRECIPITATION (Rate, mm hr-1)
    # =========================================================================

    # [2, 1] Col 1: Raw Low-Res PRECTOT
    ax = axes[1, 0]
    add_map_features(ax)
    ax.set_extent(extent, crs=data_crs)
    if raw_precip_mm is not None:
        im5 = ax.imshow(
            raw_precip_mm, origin="lower",
            extent=[raw_lons.min(), raw_lons.max(), raw_lats.min(), raw_lats.max()],
            transform=data_crs, cmap=precip_cmap, norm=precip_norm,
            interpolation="nearest"
        )
        ax.set_title("[1] Raw Native Low-Res: PRECTOT\n(GEOS-FP ~25 km Discrete Grid)", fontsize=13, fontweight="bold")
        cb = fig.colorbar(im5, ax=ax, orientation="horizontal", pad=0.04, shrink=0.75)
        cb.set_label("Precipitation Rate (mm hr⁻¹)", fontsize=11)
    else:
        ax.set_title("[1] Raw Native Low-Res (Path not found)", fontsize=13)

    # [2, 2] Col 2: Interpolated Low-Res PRECTOT
    ax = axes[1, 1]
    add_map_features(ax)
    ax.set_extent(extent, crs=data_crs)
    im6 = ax.pcolormesh(
        lons_sub, lats_sub, lr_precip_mm[::step, ::step],
        transform=data_crs, cmap=precip_cmap, norm=precip_norm,
        shading="nearest"
    )
    ax.set_title("[2] Low-Res Interpolated: PRECTOT\n(Bilinear on 3 km LCC Grid)", fontsize=13, fontweight="bold")
    cb = fig.colorbar(im6, ax=ax, orientation="horizontal", pad=0.04, shrink=0.75)
    cb.set_label("Precipitation Rate (mm hr⁻¹)", fontsize=11)

    # [2, 3] Col 3: High-Res Ground Truth PRECTOT
    ax = axes[1, 2]
    add_map_features(ax)
    ax.set_extent(extent, crs=data_crs)
    im7 = ax.pcolormesh(
        lons_sub, lats_sub, hr_precip_mm[::step, ::step],
        transform=data_crs, cmap=precip_cmap, norm=precip_norm,
        shading="nearest"
    )
    ax.set_title("[3] High-Res Ground Truth: PRECTOT\n(HWT Simulation, Resolved 3 km LCC)", fontsize=13, fontweight="bold")
    cb = fig.colorbar(im7, ax=ax, orientation="horizontal", pad=0.04, shrink=0.75)
    cb.set_label("Precipitation Rate (mm hr⁻¹)", fontsize=11)

    # [2, 4] Col 4: Difference PRECTOT
    ax = axes[1, 3]
    add_map_features(ax)
    ax.set_extent(extent, crs=data_crs)
    im8 = ax.pcolormesh(
        lons_sub, lats_sub, diff_precip[::step, ::step],
        transform=data_crs, cmap=diff_p_cmap, norm=diff_p_norm,
        shading="nearest"
    )
    mae_p = np.nanmean(np.abs(diff_precip))
    max_hr_p = np.nanmax(hr_precip_mm)
    ax.set_title(f"[4] Difference (Interpolated - High-Res)\nMAE: {mae_p:.2f} mm/hr | High-Res Max: {max_hr_p:.1f} mm/hr", fontsize=13, fontweight="bold")
    cb = fig.colorbar(im8, ax=ax, orientation="horizontal", pad=0.04, shrink=0.75)
    cb.set_label("Δ Precip (mm hr⁻¹)", fontsize=11)

    out_file = os.path.join(out_dir, f"diagnostic_4col_{stamp}.png")
    plt.savefig(out_file, dpi=200)
    plt.close()
    print(f"Saved 4-column diagnostic multi-panel to: {out_file}")

    # Also generate regional orographic zoom with discrete nearest rendering
    plot_orographic_zoom(
        dt, lons, lats, elev, lr_t2m_c, hr_t2m_c, diff_t2m,
        raw_t2m_c, raw_lats, raw_lons, out_dir
    )


def plot_orographic_zoom(dt, lons, lats, elev, lr_t2m, hr_t2m, diff_t2m, raw_t2m, raw_lats, raw_lons, out_dir):
    """
    Detailed 1x5 regional zoom over the Intermountain West / Colorado Rockies
    showing the discrete pixel contrast from 25 km to 3 km without any contour smoothing.
    """
    stamp = dt.strftime("%Y%m%d_%H%Mz")
    bbox_lon = (-112.0, -102.0)
    bbox_lat = (35.0, 43.0)

    mask = (
        (lons >= bbox_lon[0]) & (lons <= bbox_lon[1]) &
        (lats >= bbox_lat[0]) & (lats <= bbox_lat[1])
    )
    if not np.any(mask):
        return

    proj = ccrs.PlateCarree()
    fig, axes = plt.subplots(1, 5, figsize=(30, 6), subplot_kw={"projection": proj}, constrained_layout=True)
    fig.suptitle(
        f"Regional Orographic Zoom: Southern/Central Rockies ({stamp})\nDiscrete Pixel Comparison Showing Sub-Grid Resolution & Topography",
        fontsize=16, fontweight="bold"
    )

    t_min = float(np.percentile(hr_t2m[mask], 2))
    t_max = float(np.percentile(hr_t2m[mask], 98))

    # 1. Elevation
    ax = axes[0]
    add_map_features(ax)
    ax.set_extent([bbox_lon[0], bbox_lon[1], bbox_lat[0], bbox_lat[1]], crs=proj)
    if elev is not None:
        im0 = ax.pcolormesh(lons, lats, elev, transform=proj, cmap="terrain", vmin=500, vmax=4000, shading="nearest")
        ax.set_title("[1] High-Res Topography (m)\n(3 km Resolved Ridge & Valley)", fontsize=11, fontweight="bold")
        cb = fig.colorbar(im0, ax=ax, orientation="horizontal", pad=0.06, shrink=0.8)
        cb.set_label("Elevation (m)", fontsize=10)
    else:
        ax.set_title("Elevation (Not Available)", fontsize=11)

    # 2. Raw Native Low-Res (25 km Discrete Pixels)
    ax = axes[1]
    add_map_features(ax)
    ax.set_extent([bbox_lon[0], bbox_lon[1], bbox_lat[0], bbox_lat[1]], crs=proj)
    if raw_t2m is not None:
        im1 = ax.imshow(
            raw_t2m, origin="lower",
            extent=[raw_lons.min(), raw_lons.max(), raw_lats.min(), raw_lats.max()],
            transform=proj, cmap="coolwarm", vmin=t_min, vmax=t_max,
            interpolation="nearest"
        )
        ax.set_title("[2] Raw Low-Res T2M (°C)\n(Discrete ~25 km Pixels)", fontsize=11, fontweight="bold")
        cb = fig.colorbar(im1, ax=ax, orientation="horizontal", pad=0.06, shrink=0.8)
        cb.set_label("T2M (°C)", fontsize=10)
    else:
        ax.set_title("Raw Low-Res (Not Available)", fontsize=11)

    # 3. Interpolated Low-Res (Bilinear)
    ax = axes[2]
    add_map_features(ax)
    ax.set_extent([bbox_lon[0], bbox_lon[1], bbox_lat[0], bbox_lat[1]], crs=proj)
    im2 = ax.pcolormesh(lons, lats, lr_t2m, transform=proj, cmap="coolwarm", vmin=t_min, vmax=t_max, shading="nearest")
    ax.set_title("[3] Interpolated Low-Res T2M (°C)\n(Bilinear on 3 km LCC Grid)", fontsize=11, fontweight="bold")
    cb = fig.colorbar(im2, ax=ax, orientation="horizontal", pad=0.06, shrink=0.8)
    cb.set_label("T2M (°C)", fontsize=10)

    # 4. High-Res Ground Truth (3 km Resolved)
    ax = axes[3]
    add_map_features(ax)
    ax.set_extent([bbox_lon[0], bbox_lon[1], bbox_lat[0], bbox_lat[1]], crs=proj)
    im3 = ax.pcolormesh(lons, lats, hr_t2m, transform=proj, cmap="coolwarm", vmin=t_min, vmax=t_max, shading="nearest")
    ax.set_title("[4] High-Res Ground Truth (°C)\n(3 km Fine Thermal Structure)", fontsize=11, fontweight="bold")
    cb = fig.colorbar(im3, ax=ax, orientation="horizontal", pad=0.06, shrink=0.8)
    cb.set_label("TMP_2M (°C)", fontsize=10)

    # 5. Difference (Low - High)
    ax = axes[4]
    add_map_features(ax)
    ax.set_extent([bbox_lon[0], bbox_lon[1], bbox_lat[0], bbox_lat[1]], crs=proj)
    norm = mcolors.TwoSlopeNorm(vmin=-8.0, vcenter=0.0, vmax=8.0)
    im4 = ax.pcolormesh(lons, lats, diff_t2m, transform=proj, cmap="bwr", norm=norm, shading="nearest")
    ax.set_title("[5] Orographic Bias (Low - High)\n(Peak warming / Valley cooling)", fontsize=11, fontweight="bold")
    cb = fig.colorbar(im4, ax=ax, orientation="horizontal", pad=0.06, shrink=0.8)
    cb.set_label("Δ T2M (°C)", fontsize=10)

    out_file = os.path.join(out_dir, f"diagnostic_orographic_zoom_{stamp}.png")
    plt.savefig(out_file, dpi=200)
    plt.close()
    print(f"Saved regional orographic zoom to: {out_file}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    triplets = find_matching_triplets(args.lowres_regrid_dir, args.lowres_raw_root, args.highres_dir)
    if not triplets:
        print(f"\nNo matched files found between regridded data and high-res data!")
        return

    selected = []
    if args.date:
        for item in triplets:
            dt = item["datetime"]
            if dt.strftime("%Y%m%d") == args.date:
                if args.hour is None or dt.hour == args.hour:
                    selected.append(item)
        if not selected:
            print(f"No matched triplets found for date {args.date} (hour={args.hour}).")
            return
    else:
        num = min(args.random_samples, len(triplets))
        random.seed(42)
        selected = random.sample(triplets, num)

    print(f"\nSelected {len(selected)} sample(s) for diagnostic plotting.")

    sample_hr = selected[0]["highres_path"]
    lats, lons, elev = load_static_grid(args.static_grid, sample_hr)

    for item in selected:
        plot_comparison_panel(item, lats, lons, elev, args.output_dir)

    print("\n=== All diagnostic plots generated successfully! ===")
    print(f"Check output images in: {args.output_dir}")


if __name__ == "__main__":
    main()
