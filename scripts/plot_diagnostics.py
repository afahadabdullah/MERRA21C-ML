#!/usr/bin/env python3
"""
scripts/plot_diagnostics.py
===========================
Diagnostic comparison tool for MERRA21C-ML downscaling project.

Compares:
  1. Low-Resolution Regridded Predictors (GEOS-FP interpolated to 3 km LCC)
  2. High-Resolution Ground Truth (HWT 30-min LCC simulations)
  3. Static Orography / Elevation (from HWT HGT_SFC)
  4. Differences (Bias = LowRes - HighRes, Ratio, etc.)

Features:
  - Robust timestamp parsing supporting any prefix or directory structure.
  - Generates publication-ready Cartopy maps with CONUS Lambert Conformal / PlateCarree projections.
  - Automatically matches dates/timestamps between low-res and high-res or picks random paired samples.
  - Plots:
      a) Overview multi-panel: T2M (Low-Res vs High-Res vs Diff) & PRECTOT (Low-Res vs High-Res vs Diff)
      b) Orographic analysis: High-Res Topography vs Low-Res T2M / Precip correlation
      c) Regional zoom into complex terrain (e.g., Rocky Mountains, Sierra Nevada, or Great Plains storm)
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
    parser = argparse.ArgumentParser(description="Plot diagnostic comparisons between low-res regridded and high-res data.")
    parser.add_argument(
        "--lowres_dir",
        type=str,
        default="data/lowres_lcc_1hr",
        help="Root directory of regridded low-res NetCDF files.",
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
        help="Specific hour to plot (0-23). If None with --date, picks 18Z or first available.",
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


def find_matching_pairs(lowres_dir: str, highres_dir: str):
    """
    Find timestamps where both regridded low-res and high-res data are present.
    Matches any low-res *.nc4 with high-res hwt_30mn_slv_LCC files.
    """
    print(f"Scanning for available regridded low-res files in: {lowres_dir}")
    lowres_files = sorted(glob.glob(os.path.join(lowres_dir, "**", "*.nc4"), recursive=True))
    if not lowres_files:
        lowres_files = sorted(glob.glob(os.path.join(lowres_dir, "*.nc4")))

    print(f"Found {len(lowres_files)} low-res regridded files.")
    if not lowres_files:
        return []

    print(f"Sample low-res file: {lowres_files[0]}")

    pairs = []
    for lr_path in lowres_files:
        fname = os.path.basename(lr_path)
        dt = extract_timestamp_from_filename(fname)
        if dt is None:
            continue

        ym = dt.strftime("%Y%m")
        # Try both HH30z and HH00z candidates in high-res directory
        hr_candidate_30 = os.path.join(
            highres_dir, ym, f"Feature-c2160_L137.hwt_30mn_slv_LCC.{dt.strftime('%Y%m%d_%H30')}z.nc4"
        )
        hr_candidate_00 = os.path.join(
            highres_dir, ym, f"Feature-c2160_L137.hwt_30mn_slv_LCC.{dt.strftime('%Y%m%d_%H00')}z.nc4"
        )

        if os.path.exists(hr_candidate_30):
            pairs.append((lr_path, hr_candidate_30, dt))
        elif os.path.exists(hr_candidate_00):
            pairs.append((lr_path, hr_candidate_00, dt))

    print(f"Found {len(pairs)} matching timestamp pairs between low-res and high-res.")
    if not pairs:
        # Check what high-res monthly folders actually exist
        if os.path.exists(highres_dir):
            hr_dirs = sorted([d for d in os.listdir(highres_dir) if os.path.isdir(os.path.join(highres_dir, d))])
            print(f"Available high-res monthly folders in {highres_dir}: {hr_dirs}")
            sample_lr_months = sorted(list({os.path.basename(f).split('.')[2][:6] for f in lowres_files[:50] if len(os.path.basename(f).split('.')) >= 3}))
            print(f"Sample months in low-res regridded files: {sample_lr_months}")
    return pairs


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
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.8, edgecolor="#222222")
    ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.5, edgecolor="#555555", linestyle=":")
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.8, edgecolor="#222222")
    ax.add_feature(cfeature.LAKES.with_scale("50m"), facecolor="none", edgecolor="#333333", linewidth=0.5)


def plot_comparison_panel(lr_path, hr_path, dt, lats, lons, elev, out_dir):
    """
    Creates a 2x3 comprehensive diagnostic panel:
      Row 1: 2m Temperature (K -> °C)
        - [A] Low-Res Regridded (GEOS-FP, ~25 km physics on 3 km LCC)
        - [B] High-Res Ground Truth (HWT c2160, resolved 3 km simulation)
        - [C] Temperature Difference (Low-Res - High-Res)
      Row 2: Total Precipitation Rate (kg m-2 s-1 -> mm/hr)
        - [D] Low-Res Regridded Precipitation
        - [E] High-Res Ground Truth Precipitation
        - [F] Precipitation Difference (Low-Res - High-Res)
    """
    os.makedirs(out_dir, exist_ok=True)
    stamp = dt.strftime("%Y%m%d_%H%Mz")
    print(f"\n---> Plotting Diagnostic Multi-Panel for {stamp}...")

    # 1. Read Low-Res Fields
    with xr.open_dataset(lr_path) as ds_lr:
        lr_t2m = ds_lr["T2M"].values
        if lr_t2m.ndim == 3:
            lr_t2m = lr_t2m[0]
        lr_t2m_c = lr_t2m - 273.15  # Convert to Celsius

        lr_precip = ds_lr["PRECTOT"].values
        if lr_precip.ndim == 3:
            lr_precip = lr_precip[0]
        lr_precip_mm = lr_precip * 3600.0  # kg m-2 s-1 -> mm/hr

    # 2. Read High-Res Fields
    with xr.open_dataset(hr_path) as ds_hr:
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

    # 3. Differences
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

    # Setup Projection
    proj = ccrs.LambertConformal(central_longitude=-96.0, central_latitude=37.5, standard_parallels=(30, 45))
    data_crs = ccrs.PlateCarree()

    fig, axes = plt.subplots(
        2, 3,
        figsize=(22, 12),
        subplot_kw={"projection": proj},
        constrained_layout=True
    )

    fig.suptitle(
        f"MERRA21C-ML Regridding & Spatial Alignment Diagnostic\nTimestamp: {dt.strftime('%Y-%m-%d %H:%M UTC')}",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )

    # Subsample factor for responsive rendering of 1059x1799 grid
    step = 2
    lons_sub = lons[::step, ::step]
    lats_sub = lats[::step, ::step]

    # --- ROW 1: TEMPERATURE ---
    ax = axes[0, 0]
    add_map_features(ax)
    pcm1 = ax.pcolormesh(
        lons_sub, lats_sub, lr_t2m_c[::step, ::step],
        transform=data_crs, cmap=t_cmap, vmin=t_min, vmax=t_max, shading="auto"
    )
    ax.set_title("[A] Low-Res Predictor: T2M (GEOS-FP regridded to 3 km)", fontsize=13, fontweight="semibold")
    cbar1 = fig.colorbar(pcm1, ax=ax, orientation="horizontal", pad=0.04, shrink=0.75)
    cbar1.set_label("2m Temperature (°C)", fontsize=11)

    ax = axes[0, 1]
    add_map_features(ax)
    pcm2 = ax.pcolormesh(
        lons_sub, lats_sub, hr_t2m_c[::step, ::step],
        transform=data_crs, cmap=t_cmap, vmin=t_min, vmax=t_max, shading="auto"
    )
    ax.set_title("[B] High-Res Target: TMP_2M (HWT Simulation, 3 km LCC)", fontsize=13, fontweight="semibold")
    cbar2 = fig.colorbar(pcm2, ax=ax, orientation="horizontal", pad=0.04, shrink=0.75)
    cbar2.set_label("2m Temperature (°C)", fontsize=11)

    ax = axes[0, 2]
    add_map_features(ax)
    pcm3 = ax.pcolormesh(
        lons_sub, lats_sub, diff_t2m[::step, ::step],
        transform=data_crs, cmap=diff_t_cmap, norm=diff_t_norm, shading="auto"
    )
    rmse_t = np.sqrt(np.nanmean(diff_t2m**2))
    mae_t = np.nanmean(np.abs(diff_t2m))
    ax.set_title(f"[C] Temperature Difference (Low - High)\nRMSE: {rmse_t:.2f} °C | MAE: {mae_t:.2f} °C", fontsize=13, fontweight="semibold")
    cbar3 = fig.colorbar(pcm3, ax=ax, orientation="horizontal", pad=0.04, shrink=0.75)
    cbar3.set_label("Δ T2M (°C)", fontsize=11)

    # --- ROW 2: PRECIPITATION ---
    ax = axes[1, 0]
    add_map_features(ax)
    pcm4 = ax.pcolormesh(
        lons_sub, lats_sub, lr_precip_mm[::step, ::step],
        transform=data_crs, cmap=precip_cmap, norm=precip_norm, shading="auto"
    )
    ax.set_title("[D] Low-Res Predictor: PRECTOT (GEOS-FP regridded)", fontsize=13, fontweight="semibold")
    cbar4 = fig.colorbar(pcm4, ax=ax, orientation="horizontal", pad=0.04, shrink=0.75)
    cbar4.set_label("Precipitation Rate (mm hr⁻¹)", fontsize=11)

    ax = axes[1, 1]
    add_map_features(ax)
    pcm5 = ax.pcolormesh(
        lons_sub, lats_sub, hr_precip_mm[::step, ::step],
        transform=data_crs, cmap=precip_cmap, norm=precip_norm, shading="auto"
    )
    ax.set_title("[E] High-Res Target: PRECTOT (HWT Simulation)", fontsize=13, fontweight="semibold")
    cbar5 = fig.colorbar(pcm5, ax=ax, orientation="horizontal", pad=0.04, shrink=0.75)
    cbar5.set_label("Precipitation Rate (mm hr⁻¹)", fontsize=11)

    ax = axes[1, 2]
    add_map_features(ax)
    pcm6 = ax.pcolormesh(
        lons_sub, lats_sub, diff_precip[::step, ::step],
        transform=data_crs, cmap=diff_p_cmap, norm=diff_p_norm, shading="auto"
    )
    mae_p = np.nanmean(np.abs(diff_precip))
    max_hr_p = np.nanmax(hr_precip_mm)
    ax.set_title(f"[F] Precipitation Difference (Low - High)\nMAE: {mae_p:.2f} mm/hr | High-Res Max: {max_hr_p:.1f} mm/hr", fontsize=13, fontweight="semibold")
    cbar6 = fig.colorbar(pcm6, ax=ax, orientation="horizontal", pad=0.04, shrink=0.75)
    cbar6.set_label("Δ Precip (mm hr⁻¹)", fontsize=11)

    out_file = os.path.join(out_dir, f"diagnostic_comparison_{stamp}.png")
    plt.savefig(out_file, dpi=200)
    plt.close()
    print(f"Saved diagnostic multi-panel to: {out_file}")

    plot_orographic_zoom(
        dt, lons, lats, elev, lr_t2m_c, hr_t2m_c, diff_t2m, lr_precip_mm, hr_precip_mm, out_dir
    )


def plot_orographic_zoom(dt, lons, lats, elev, lr_t2m, hr_t2m, diff_t2m, lr_p, hr_p, out_dir):
    """
    Detailed 1x4 regional zoom over the Intermountain West / Colorado Rockies
    highlighting the impact of 3 km high-resolution topography on temperature and precipitation downscaling.
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
    fig, axes = plt.subplots(1, 4, figsize=(24, 6), subplot_kw={"projection": proj}, constrained_layout=True)
    fig.suptitle(
        f"Regional Orographic Zoom: Southern/Central Rockies ({stamp})\nDemonstrating Sub-Grid Topography Impact on ML Targets",
        fontsize=16, fontweight="bold"
    )

    ax = axes[0]
    add_map_features(ax)
    ax.set_extent([bbox_lon[0], bbox_lon[1], bbox_lat[0], bbox_lat[1]], crs=proj)
    if elev is not None:
        pcm0 = ax.pcolormesh(lons, lats, elev, transform=proj, cmap="terrain", vmin=500, vmax=4000, shading="auto")
        ax.set_title("3 km High-Res Topography (m)", fontsize=12, fontweight="semibold")
        cbar = fig.colorbar(pcm0, ax=ax, orientation="horizontal", pad=0.06, shrink=0.8)
        cbar.set_label("Elevation (m)", fontsize=10)
    else:
        ax.set_title("Elevation (Not Available)", fontsize=12)

    ax = axes[1]
    add_map_features(ax)
    ax.set_extent([bbox_lon[0], bbox_lon[1], bbox_lat[0], bbox_lat[1]], crs=proj)
    t_min = float(np.percentile(hr_t2m[mask], 2))
    t_max = float(np.percentile(hr_t2m[mask], 98))
    pcm1 = ax.pcolormesh(lons, lats, lr_t2m, transform=proj, cmap="coolwarm", vmin=t_min, vmax=t_max, shading="auto")
    ax.set_title("Low-Res Regridded T2M (°C)\n(Smooth 25 km thermodynamics)", fontsize=12, fontweight="semibold")
    cbar = fig.colorbar(pcm1, ax=ax, orientation="horizontal", pad=0.06, shrink=0.8)
    cbar.set_label("T2M (°C)", fontsize=10)

    ax = axes[2]
    add_map_features(ax)
    ax.set_extent([bbox_lon[0], bbox_lon[1], bbox_lat[0], bbox_lat[1]], crs=proj)
    pcm2 = ax.pcolormesh(lons, lats, hr_t2m, transform=proj, cmap="coolwarm", vmin=t_min, vmax=t_max, shading="auto")
    ax.set_title("High-Res Ground Truth T2M (°C)\n(Sharp valleys & ridge cooling)", fontsize=12, fontweight="semibold")
    cbar = fig.colorbar(pcm2, ax=ax, orientation="horizontal", pad=0.06, shrink=0.8)
    cbar.set_label("TMP_2M (°C)", fontsize=10)

    ax = axes[3]
    add_map_features(ax)
    ax.set_extent([bbox_lon[0], bbox_lon[1], bbox_lat[0], bbox_lat[1]], crs=proj)
    norm = mcolors.TwoSlopeNorm(vmin=-8.0, vcenter=0.0, vmax=8.0)
    pcm3 = ax.pcolormesh(lons, lats, diff_t2m, transform=proj, cmap="bwr", norm=norm, shading="auto")
    ax.set_title("Orographically-Driven T2M Error\n(ΔT: Valleys too cold, peaks too warm)", fontsize=12, fontweight="semibold")
    cbar = fig.colorbar(pcm3, ax=ax, orientation="horizontal", pad=0.06, shrink=0.8)
    cbar.set_label("Δ T2M (°C)", fontsize=10)

    out_file = os.path.join(out_dir, f"diagnostic_orographic_zoom_{stamp}.png")
    plt.savefig(out_file, dpi=200)
    plt.close()
    print(f"Saved regional orographic zoom to: {out_file}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    pairs = find_matching_pairs(args.lowres_dir, args.highres_dir)
    if not pairs:
        print(f"\nNo matched pairs found yet!")
        return

    selected_pairs = []
    if args.date:
        for lr, hr, dt in pairs:
            if dt.strftime("%Y%m%d") == args.date:
                if args.hour is None or dt.hour == args.hour:
                    selected_pairs.append((lr, hr, dt))
        if not selected_pairs:
            print(f"No matched pairs found for date {args.date} (hour={args.hour}).")
            return
    else:
        num = min(args.random_samples, len(pairs))
        random.seed(42)
        selected_pairs = random.sample(pairs, num)

    print(f"\nSelected {len(selected_pairs)} sample(s) for diagnostic plotting.")

    sample_hr = selected_pairs[0][1]
    lats, lons, elev = load_static_grid(args.static_grid, sample_hr)

    for lr_path, hr_path, dt in selected_pairs:
        plot_comparison_panel(lr_path, hr_path, dt, lats, lons, elev, args.output_dir)

    print("\n=== All diagnostic plots generated successfully! ===")
    print(f"Check output images in: {args.output_dir}")


if __name__ == "__main__":
    main()
