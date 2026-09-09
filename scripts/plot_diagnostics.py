#!/usr/bin/env python3
"""
scripts/plot_diagnostics.py
===========================
Diagnostic comparison tool for MERRA21C-ML downscaling project.

Generates:
  1. Full CONUS Static Topography & Grid Metrics Map (3 km LCC Elevation, Slope, Area)
  2. Multi-Scale 4-Column Diagnostic Panel:
       Col 1: Raw Low-Res Predictor (Native 0.25° GEOS-FP grid, discrete raster pixels)
       Col 2: Low-Res Interpolated Predictor (Bilinear to 3 km LCC)
       Col 3: High-Res Ground Truth (HWT 3 km LCC simulation)
       Col 4: Difference / Bias (Interpolated Low-Res minus High-Res)
     Rows:
       Row 1: 2m Temperature (K -> °C)
       Row 2: Total Precipitation Rate (kg m-2 s-1 -> mm hr-1)
       Row 3: 10m Wind Speed (m s-1)
       Row 4: 2m Specific Humidity (g kg-1)
  3. Regional Orographic Zoom Panel over complex terrain (Southern/Central Rockies).
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
        raw_month_dir = os.path.join(raw_root, f"Y{dt.year}", f"M{dt.month:02d}")
        raw_slv = os.path.join(raw_month_dir, f"f5295_fp.tavg1_2d_slv_Nx.{dt.strftime('%Y%m%d_%H%M')}z.nc4")
        raw_flx = os.path.join(raw_month_dir, f"f5295_fp.tavg1_2d_flx_Nx.{dt.strftime('%Y%m%d_%H%M')}z.nc4")

        if not (os.path.exists(raw_slv) and os.path.exists(raw_flx)):
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
    """Load or dynamically extract static lats, lons, elevation, and area."""
    if os.path.exists(static_path):
        print(f"Loading static grid from: {static_path}")
        ds = xr.open_dataset(static_path)
        lats = ds["lats"].values
        lons = ds["lons"].values
        elev = ds["elevation"].values if "elevation" in ds else None
        area = ds["AREA"].values if "AREA" in ds else None
        return lats, lons, elev, area

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
        area = ds["AREA"].values[0] if "AREA" in ds else None
    return lats, lons, elev, area


def add_map_features(ax):
    """Add standard geographic boundaries for publication-quality CONUS maps."""
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.8, edgecolor="#222222", zorder=3)
    ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.5, edgecolor="#555555", linestyle=":", zorder=3)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.8, edgecolor="#222222", zorder=3)
    ax.add_feature(cfeature.LAKES.with_scale("50m"), facecolor="none", edgecolor="#333333", linewidth=0.5, zorder=3)


def plot_static_topography(lats, lons, elev, area, out_dir):
    """
    Dedicated diagnostic plot for 3 km High-Res Static Orography & Grid Metrics:
      [A] Full CONUS 3 km Elevation (m) with terrain colormap
      [B] Terrain Slope / Gradient (|∇z| in m/km) highlighting mountain ridges & canyons
      [C] Grid Cell Area (km²) verifying LCC spatial distortion
      [D] Topography Elevation Distribution (Histogram / CDF)
    """
    if elev is None:
        print("Static elevation data not available to plot.")
        return

    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "diagnostic_static_topography_3km.png")
    print(f"\n---> Generating Full CONUS Static Topography Map: {out_file}...")

    proj = ccrs.LambertConformal(central_longitude=-96.0, central_latitude=37.5, standard_parallels=(30, 45))
    data_crs = ccrs.PlateCarree()
    extent = [-125.0, -66.5, 23.0, 50.5]

    # Compute terrain slope |grad(z)|
    # dy ~ 3000m, dx ~ 3000m
    gy, gx = np.gradient(elev, 3000.0, 3000.0)
    slope_m_per_km = np.sqrt(gy**2 + gx**2) * 1000.0  # m / km

    # Area in km²
    area_km2 = area / 1e6 if area is not None else np.full_like(elev, 9.0)

    step = 2
    lons_sub = lons[::step, ::step]
    lats_sub = lats[::step, ::step]

    fig = plt.figure(figsize=(24, 14), constrained_layout=True)
    fig.suptitle("MERRA21C-ML: High-Resolution 3 km LCC Static Grid & Topography Diagnostics", fontsize=18, fontweight="bold")

    # 1. CONUS Elevation Map
    ax1 = fig.add_subplot(2, 2, 1, projection=proj)
    add_map_features(ax1)
    ax1.set_extent(extent, crs=data_crs)
    pcm1 = ax1.pcolormesh(
        lons_sub, lats_sub, elev[::step, ::step],
        transform=data_crs, cmap="terrain", vmin=0, vmax=3800, shading="auto"
    )
    ax1.set_title("[A] CONUS High-Resolution Topography (Elevation in meters)", fontsize=13, fontweight="bold")
    cb1 = fig.colorbar(pcm1, ax=ax1, orientation="horizontal", pad=0.04, shrink=0.75)
    cb1.set_label("Surface Elevation (m)", fontsize=11)

    # 2. Terrain Slope / Orographic Forcing Map
    ax2 = fig.add_subplot(2, 2, 2, projection=proj)
    add_map_features(ax2)
    ax2.set_extent(extent, crs=data_crs)
    pcm2 = ax2.pcolormesh(
        lons_sub, lats_sub, slope_m_per_km[::step, ::step],
        transform=data_crs, cmap="magma_r", vmin=0, vmax=150, shading="auto"
    )
    ax2.set_title("[B] Topographic Slope Magnitude |∇z| (m / km)", fontsize=13, fontweight="bold")
    cb2 = fig.colorbar(pcm2, ax=ax2, orientation="horizontal", pad=0.04, shrink=0.75)
    cb2.set_label("Slope (m / km)", fontsize=11)

    # 3. Grid Cell Area
    ax3 = fig.add_subplot(2, 2, 3, projection=proj)
    add_map_features(ax3)
    ax3.set_extent(extent, crs=data_crs)
    pcm3 = ax3.pcolormesh(
        lons_sub, lats_sub, area_km2[::step, ::step],
        transform=data_crs, cmap="viridis", vmin=6.0, vmax=11.0, shading="auto"
    )
    ax3.set_title("[C] Native LCC Grid Cell Area (km²)", fontsize=13, fontweight="bold")
    cb3 = fig.colorbar(pcm3, ax=ax3, orientation="horizontal", pad=0.04, shrink=0.75)
    cb3.set_label("Area (km²)", fontsize=11)

    # 4. Statistical Distribution of Topography & Slope
    ax4 = fig.add_subplot(2, 2, 4)
    elev_valid = elev[~np.isnan(elev)]
    elev_conus = elev_valid[elev_valid > 0]
    ax4.hist(elev_conus, bins=80, density=True, color="#2b5c8f", alpha=0.75, edgecolor="black", linewidth=0.5)
    ax4.set_title("[D] Probability Density Function of CONUS Elevation", fontsize=13, fontweight="bold")
    ax4.set_xlabel("Elevation (meters)", fontsize=11)
    ax4.set_ylabel("Probability Density", fontsize=11)
    ax4.grid(True, linestyle="--", alpha=0.5)
    ax4.axvline(np.mean(elev_conus), color="red", linestyle="--", label=f"Mean: {np.mean(elev_conus):.0f} m")
    ax4.axvline(np.median(elev_conus), color="orange", linestyle=":", label=f"Median: {np.median(elev_conus):.0f} m")
    ax4.axvline(np.max(elev_conus), color="darkred", linestyle="-", label=f"Max: {np.max(elev_conus):.0f} m")
    ax4.legend(fontsize=11)

    plt.savefig(out_file, dpi=200)
    plt.close()
    print(f"Saved static topography diagnostics to: {out_file}")


def plot_comparison_panel(item, lats, lons, elev, out_dir):
    """
    Creates a comprehensive 4-row x 4-column diagnostic panel:
      Row 1: 2m Temperature (K -> °C)
      Row 2: Total Precipitation Rate (kg m-2 s-1 -> mm hr-1)
      Row 3: 10m Wind Speed (m s-1)
      Row 4: 2m Specific Humidity (g kg-1)
      
    Columns:
      Col 1: Raw Native Low-Res (GEOS-FP 0.25° grid)
      Col 2: Interpolated Low-Res (Bilinear to 3 km LCC)
      Col 3: High-Res Ground Truth (HWT 3 km LCC)
      Col 4: Difference / Bias (Interpolated - High-Res)
    """
    os.makedirs(out_dir, exist_ok=True)
    dt = item["datetime"]
    stamp = dt.strftime("%Y%m%d_%H%Mz")
    print(f"\n---> Plotting 4-Row x 4-Column Diagnostic Panel for {stamp}...")

    # 1. Read Low-Res Regridded Fields
    with xr.open_dataset(item["regrid_path"]) as ds_lr:
        def get_lr(var):
            if var not in ds_lr:
                return None
            v = ds_lr[var].values
            return v[0] if v.ndim == 3 else v

        lr_t2m_c = get_lr("T2M") - 273.15
        lr_precip_mm = get_lr("PRECTOT") * 3600.0
        lr_u10 = get_lr("U10M")
        lr_v10 = get_lr("V10M")
        lr_wspd = np.sqrt(lr_u10**2 + lr_v10**2) if (lr_u10 is not None and lr_v10 is not None) else None
        lr_qv = get_lr("QV2M")
        lr_qv_g = lr_qv * 1000.0 if lr_qv is not None else None  # g/kg

    # 2. Read High-Res Fields
    with xr.open_dataset(item["highres_path"]) as ds_hr:
        def get_hr(var_candidates):
            for var in var_candidates:
                if var in ds_hr:
                    v = ds_hr[var].values
                    return v[0] if v.ndim == 3 else v
            return None

        hr_t2m_raw = get_hr(["TMP_2M", "T2M"])
        hr_t2m_c = hr_t2m_raw - 273.15 if hr_t2m_raw is not None else None

        hr_p_raw = get_hr(["PRECTOT", "APCP"])
        hr_precip_mm = hr_p_raw * 3600.0 if ("PRECTOT" in ds_hr) else hr_p_raw

        hr_wspd = get_hr(["SPEED", "WSPD_10M"])
        if hr_wspd is None:
            u = get_hr(["UGRD_10M", "U10M"])
            v = get_hr(["VGRD_10M", "V10M"])
            if u is not None and v is not None:
                hr_wspd = np.sqrt(u**2 + v**2)

        hr_qv = get_hr(["SPFH_2M", "QV2M"])
        hr_qv_g = hr_qv * 1000.0 if hr_qv is not None else None

    # 3. Read Raw Native Low-Res Fields
    raw_t2m_c, raw_precip_mm, raw_wspd, raw_qv_g, raw_lats, raw_lons = None, None, None, None, None, None
    if item["raw_slv"] and os.path.exists(item["raw_slv"]) and item["raw_flx"] and os.path.exists(item["raw_flx"]):
        with xr.open_dataset(item["raw_slv"]) as ds_r_slv:
            lons_norm = np.where(ds_r_slv["lon"].values > 180.0, ds_r_slv["lon"].values - 360.0, ds_r_slv["lon"].values)
            ds_r_slv = ds_r_slv.assign_coords(lon=lons_norm).sortby("lon")
            ds_crop_slv = ds_r_slv.sel(lat=slice(20.0, 55.0), lon=slice(-130.0, -65.0))
            raw_lats = ds_crop_slv["lat"].values
            raw_lons = ds_crop_slv["lon"].values

            def get_crop(ds, vname):
                if vname not in ds:
                    return None
                val = ds[vname].values
                return val[0] if val.ndim == 3 else val

            raw_t = get_crop(ds_crop_slv, "T2M")
            if raw_t is not None:
                raw_t2m_c = raw_t - 273.15
            ru10 = get_crop(ds_crop_slv, "U10M")
            rv10 = get_crop(ds_crop_slv, "V10M")
            if ru10 is not None and rv10 is not None:
                raw_wspd = np.sqrt(ru10**2 + rv10**2)
            rqv = get_crop(ds_crop_slv, "QV2M")
            if rqv is not None:
                raw_qv_g = rqv * 1000.0

        with xr.open_dataset(item["raw_flx"]) as ds_r_flx:
            lons_norm = np.where(ds_r_flx["lon"].values > 180.0, ds_r_flx["lon"].values - 360.0, ds_r_flx["lon"].values)
            ds_r_flx = ds_r_flx.assign_coords(lon=lons_norm).sortby("lon")
            ds_crop_flx = ds_r_flx.sel(lat=slice(20.0, 55.0), lon=slice(-130.0, -65.0))
            r_p = get_crop(ds_crop_flx, "PRECTOT")
            if r_p is not None:
                raw_precip_mm = np.maximum(r_p * 3600.0, 0.0)

    # 4. Compute Differences
    diff_t2m = (lr_t2m_c - hr_t2m_c) if (lr_t2m_c is not None and hr_t2m_c is not None) else None
    diff_precip = (lr_precip_mm - hr_precip_mm) if (lr_precip_mm is not None and hr_precip_mm is not None) else None
    diff_wspd = (lr_wspd - hr_wspd) if (lr_wspd is not None and hr_wspd is not None) else None
    diff_qv = (lr_qv_g - hr_qv_g) if (lr_qv_g is not None and hr_qv_g is not None) else None

    # Setup Projection & Bounds
    proj = ccrs.LambertConformal(central_longitude=-96.0, central_latitude=37.5, standard_parallels=(30, 45))
    data_crs = ccrs.PlateCarree()
    extent = [-125.0, -66.5, 23.0, 50.5]

    fig, axes = plt.subplots(
        4, 4,
        figsize=(28, 22),
        subplot_kw={"projection": proj},
        constrained_layout=True
    )
    fig.suptitle(
        f"MERRA21C-ML Multi-Variable Downscaling Diagnostics (25 km -> 3 km LCC)\nTimestamp: {dt.strftime('%Y-%m-%d %H:%M UTC')}",
        fontsize=20, fontweight="bold", y=0.99
    )

    step = 2
    lons_sub = lons[::step, ::step]
    lats_sub = lats[::step, ::step]

    # Helper for 4-column row plotting
    def plot_row(row_idx, raw_var, lr_var, hr_var, diff_var, var_title, unit, cmap, norm, diff_cmap, diff_norm, vmin=None, vmax=None):
        # Col 1: Raw Low-Res
        ax = axes[row_idx, 0]
        add_map_features(ax)
        ax.set_extent(extent, crs=data_crs)
        if raw_var is not None and raw_lons is not None:
            kw = {"norm": norm} if norm else {"vmin": vmin, "vmax": vmax}
            im = ax.imshow(
                raw_var, origin="lower",
                extent=[raw_lons.min(), raw_lons.max(), raw_lats.min(), raw_lats.max()],
                transform=data_crs, cmap=cmap, interpolation="nearest", **kw
            )
            ax.set_title(f"[1] Raw GEOS-FP (25 km)\n{var_title}", fontsize=12, fontweight="bold")
            cb = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.04, shrink=0.75)
            cb.set_label(f"{var_title} ({unit})", fontsize=10)
        else:
            ax.set_title(f"[1] Raw GEOS-FP (Not Found)", fontsize=12)

        # Col 2: Interpolated Low-Res
        ax = axes[row_idx, 1]
        add_map_features(ax)
        ax.set_extent(extent, crs=data_crs)
        if lr_var is not None:
            kw = {"norm": norm} if norm else {"vmin": vmin, "vmax": vmax}
            im = ax.pcolormesh(
                lons_sub, lats_sub, lr_var[::step, ::step],
                transform=data_crs, cmap=cmap, shading="auto", **kw
            )
            ax.set_title(f"[2] Interpolated Low-Res (3 km LCC)\n{var_title}", fontsize=12, fontweight="bold")
            cb = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.04, shrink=0.75)
            cb.set_label(f"{var_title} ({unit})", fontsize=10)

        # Col 3: High-Res Ground Truth
        ax = axes[row_idx, 2]
        add_map_features(ax)
        ax.set_extent(extent, crs=data_crs)
        if hr_var is not None:
            kw = {"norm": norm} if norm else {"vmin": vmin, "vmax": vmax}
            im = ax.pcolormesh(
                lons_sub, lats_sub, hr_var[::step, ::step],
                transform=data_crs, cmap=cmap, shading="auto", **kw
            )
            ax.set_title(f"[3] High-Res Target (HWT 3 km LCC)\n{var_title}", fontsize=12, fontweight="bold")
            cb = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.04, shrink=0.75)
            cb.set_label(f"{var_title} ({unit})", fontsize=10)

        # Col 4: Difference
        ax = axes[row_idx, 3]
        add_map_features(ax)
        ax.set_extent(extent, crs=data_crs)
        if diff_var is not None:
            im = ax.pcolormesh(
                lons_sub, lats_sub, diff_var[::step, ::step],
                transform=data_crs, cmap=diff_cmap, norm=diff_norm, shading="auto"
            )
            rmse = np.sqrt(np.nanmean(diff_var**2))
            mae = np.nanmean(np.abs(diff_var))
            ax.set_title(f"[4] Difference (Low - High)\nRMSE: {rmse:.2f} | MAE: {mae:.2f} {unit}", fontsize=12, fontweight="bold")
            cb = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.04, shrink=0.75)
            cb.set_label(f"Δ {var_title} ({unit})", fontsize=10)

    # --- ROW 1: 2m Temperature ---
    t_min = float(np.percentile(hr_t2m_c[~np.isnan(hr_t2m_c)], 1)) if hr_t2m_c is not None else -15.0
    t_max = float(np.percentile(hr_t2m_c[~np.isnan(hr_t2m_c)], 99)) if hr_t2m_c is not None else 35.0
    plot_row(
        0, raw_t2m_c, lr_t2m_c, hr_t2m_c, diff_t2m,
        "2m Temperature", "°C", "coolwarm", None, "bwr",
        mcolors.TwoSlopeNorm(vmin=-10.0, vcenter=0.0, vmax=10.0), vmin=t_min, vmax=t_max
    )

    # --- ROW 2: Precipitation Rate ---
    precip_levels = [0.05, 0.2, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 35.0, 50.0]
    p_cmap = plt.cm.get_cmap("YlGnBu").copy()
    p_cmap.set_under("white")
    p_norm = mcolors.BoundaryNorm(boundaries=precip_levels, ncolors=p_cmap.N, extend="max")
    plot_row(
        1, raw_precip_mm, lr_precip_mm, hr_precip_mm, diff_precip,
        "Total Precipitation", "mm hr⁻¹", p_cmap, p_norm, "BrBG",
        mcolors.TwoSlopeNorm(vmin=-15.0, vcenter=0.0, vmax=15.0)
    )

    # --- ROW 3: 10m Wind Speed ---
    plot_row(
        2, raw_wspd, lr_wspd, hr_wspd, diff_wspd,
        "10m Wind Speed", "m s⁻¹", "viridis", None, "PuOr",
        mcolors.TwoSlopeNorm(vmin=-8.0, vcenter=0.0, vmax=8.0), vmin=0.0, vmax=25.0
    )

    # --- ROW 4: 2m Specific Humidity ---
    plot_row(
        3, raw_qv_g, lr_qv_g, hr_qv_g, diff_qv,
        "2m Specific Humidity", "g kg⁻¹", "YlGn", None, "RdBu",
        mcolors.TwoSlopeNorm(vmin=-4.0, vcenter=0.0, vmax=4.0), vmin=0.0, vmax=20.0
    )

    out_file = os.path.join(out_dir, f"diagnostic_multivar_4col_{stamp}.png")
    plt.savefig(out_file, dpi=200)
    plt.close()
    print(f"Saved 4-row x 4-column multi-variable diagnostic to: {out_file}")

    # Regional Orographic Zoom
    plot_orographic_zoom(
        dt, lons, lats, elev, lr_t2m_c, hr_t2m_c, diff_t2m,
        raw_t2m_c, raw_lats, raw_lons, out_dir
    )


def plot_orographic_zoom(dt, lons, lats, elev, lr_t2m, hr_t2m, diff_t2m, raw_t2m, raw_lats, raw_lons, out_dir):
    """
    Detailed 1x5 regional zoom over the Intermountain West / Colorado Rockies
    showing the discrete pixel contrast from 25 km to 3 km alongside high-res topography.
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
        f"Regional Orographic Zoom: Southern/Central Rockies ({stamp})\nDemonstrating 3 km Topography vs Discrete ~25 km GEOS-FP Thermodynamics",
        fontsize=16, fontweight="bold"
    )

    t_min = float(np.percentile(hr_t2m[mask], 2)) if hr_t2m is not None else -10.0
    t_max = float(np.percentile(hr_t2m[mask], 98)) if hr_t2m is not None else 25.0

    # 1. Elevation
    ax = axes[0]
    add_map_features(ax)
    ax.set_extent([bbox_lon[0], bbox_lon[1], bbox_lat[0], bbox_lat[1]], crs=proj)
    if elev is not None:
        im0 = ax.pcolormesh(lons, lats, elev, transform=proj, cmap="terrain", vmin=500, vmax=4000, shading="auto")
        ax.set_title("[1] High-Res Topography (m)\n(3 km Resolved Ridge & Valley)", fontsize=11, fontweight="bold")
        cb = fig.colorbar(im0, ax=ax, orientation="horizontal", pad=0.06, shrink=0.8)
        cb.set_label("Elevation (m)", fontsize=10)
    else:
        ax.set_title("Elevation (Not Available)", fontsize=11)

    # 2. Raw Native Low-Res (25 km Discrete Pixels)
    ax = axes[1]
    add_map_features(ax)
    ax.set_extent([bbox_lon[0], bbox_lon[1], bbox_lat[0], bbox_lat[1]], crs=proj)
    if raw_t2m is not None and raw_lons is not None:
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
    if lr_t2m is not None:
        im2 = ax.pcolormesh(lons, lats, lr_t2m, transform=proj, cmap="coolwarm", vmin=t_min, vmax=t_max, shading="auto")
        ax.set_title("[3] Interpolated Low-Res T2M (°C)\n(Bilinear on 3 km LCC Grid)", fontsize=11, fontweight="bold")
        cb = fig.colorbar(im2, ax=ax, orientation="horizontal", pad=0.06, shrink=0.8)
        cb.set_label("T2M (°C)", fontsize=10)

    # 4. High-Res Ground Truth (3 km Resolved)
    ax = axes[3]
    add_map_features(ax)
    ax.set_extent([bbox_lon[0], bbox_lon[1], bbox_lat[0], bbox_lat[1]], crs=proj)
    if hr_t2m is not None:
        im3 = ax.pcolormesh(lons, lats, hr_t2m, transform=proj, cmap="coolwarm", vmin=t_min, vmax=t_max, shading="auto")
        ax.set_title("[4] High-Res Ground Truth (°C)\n(3 km Fine Thermal Structure)", fontsize=11, fontweight="bold")
        cb = fig.colorbar(im3, ax=ax, orientation="horizontal", pad=0.06, shrink=0.8)
        cb.set_label("TMP_2M (°C)", fontsize=10)

    # 5. Difference (Low - High)
    ax = axes[4]
    add_map_features(ax)
    ax.set_extent([bbox_lon[0], bbox_lon[1], bbox_lat[0], bbox_lat[1]], crs=proj)
    if diff_t2m is not None:
        norm = mcolors.TwoSlopeNorm(vmin=-8.0, vcenter=0.0, vmax=8.0)
        im4 = ax.pcolormesh(lons, lats, diff_t2m, transform=proj, cmap="bwr", norm=norm, shading="auto")
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

    # Load static grid / elevation
    sample_hr = triplets[0]["highres_path"]
    lats, lons, elev, area = load_static_grid(args.static_grid, sample_hr)

    # 1. Plot full CONUS Static Topography diagnostics (Elevation, Slope, Area, PDF)
    plot_static_topography(lats, lons, elev, area, args.output_dir)

    # 2. Select timestamps for multi-variable diagnostics
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

    print(f"\nSelected {len(selected)} sample(s) for diagnostic multi-variable plotting.")

    for item in selected:
        plot_comparison_panel(item, lats, lons, elev, args.output_dir)

    print("\n=== All diagnostic plots generated successfully! ===")
    print(f"Check output images in: {args.output_dir}")


if __name__ == "__main__":
    main()
