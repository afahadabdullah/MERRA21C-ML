#!/usr/bin/env python3
"""
Data Audit & Grid Extraction Script for MERRA21C-ML
Usage on Discover:
    python scripts/data_audit.py --hwt_dir /gpfsm/dnb08/projects/p38/HWT/CONUS02KM/Feature-c2160_L137/holding/hwt_30mn_slv_LCC \
                                 --output_dir ./data/static_grid
"""

import os
import glob
import argparse
from pathlib import Path
import numpy as np
import netCDF4 as nc

def audit_directory(hwt_dir: str):
    print(f"=== Auditing High-Resolution Directory: {hwt_dir} ===")
    if not os.path.exists(hwt_dir):
        print(f"ERROR: Directory does not exist: {hwt_dir}")
        return None, []

    # Find monthly folders
    months = sorted([d for d in os.listdir(hwt_dir) if os.path.isdir(os.path.join(hwt_dir, d)) and d.isdigit()])
    print(f"Found {len(months)} monthly directories: {months}")

    all_files = []
    for m in months:
        m_dir = os.path.join(hwt_dir, m)
        files = sorted(glob.glob(os.path.join(m_dir, "*.nc4")))
        print(f"  Month {m}: {len(files)} files")
        all_files.extend(files)

    print(f"Total netCDF-4 files found: {len(all_files)}")
    if not all_files:
        return None, []
    
    return all_files[0], all_files

def extract_static_grid(sample_file: str, output_path: str):
    print(f"\n=== Extracting Static Grid & Orography from: {sample_file} ===")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with nc.Dataset(sample_file, "r") as src:
        # Dimensions
        ydim = src.dimensions["Ydim"].size
        xdim = src.dimensions["Xdim"].size
        print(f"Grid dimensions: Ydim={ydim}, Xdim={xdim}")

        lats = src.variables["lats"][:]
        lons = src.variables["lons"][:]
        area = src.variables["AREA"][0, :, :]
        
        # Surface geopotential height to elevation in meters
        if "HGT_SFC" in src.variables:
            hgt_sfc = src.variables["HGT_SFC"][0, :, :]
            elevation = hgt_sfc / 9.80665  # m^2/s^2 to m
        else:
            elevation = np.zeros((ydim, xdim), dtype=np.float32)

    # Save to static NetCDF
    with nc.Dataset(output_path, "w", format="NETCDF4") as dst:
        dst.createDimension("Ydim", ydim)
        dst.createDimension("Xdim", xdim)

        v_lat = dst.createVariable("lats", "f4", ("Ydim", "Xdim"), zlib=True)
        v_lat.units = "degrees_north"
        v_lat[:] = lats

        v_lon = dst.createVariable("lons", "f4", ("Ydim", "Xdim"), zlib=True)
        v_lon.units = "degrees_east"
        v_lon[:] = lons

        v_area = dst.createVariable("AREA", "f4", ("Ydim", "Xdim"), zlib=True)
        v_area.units = "m2"
        v_area[:] = area

        v_elev = dst.createVariable("elevation", "f4", ("Ydim", "Xdim"), zlib=True)
        v_elev.units = "m"
        v_elev.long_name = "surface_elevation_from_HGT_SFC"
        v_elev[:] = elevation

    print(f"Static grid successfully saved to: {output_path}")
    print(f"  Lat range: [{np.min(lats):.2f}, {np.max(lats):.2f}]")
    print(f"  Lon range: [{np.min(lons):.2f}, {np.max(lons):.2f}]")
    print(f"  Elevation range: [{np.nanmin(elevation):.1f}m, {np.nanmax(elevation):.1f}m]")

def main():
    parser = argparse.ArgumentParser(description="Audit HWT holdings and extract static grid.")
    parser.add_argument(
        "--hwt_dir",
        type=str,
        default="/gpfsm/dnb08/projects/p38/HWT/CONUS02KM/Feature-c2160_L137/holding/hwt_30mn_slv_LCC",
        help="Path to high-resolution 30-min surface collection",
    )
    parser.add_argument(
        "--output_grid",
        type=str,
        default="data/static_grid/conus_lcc_grid.nc",
        help="Path where static grid file should be written",
    )
    args = parser.parse_args()

    sample_file, all_files = audit_directory(args.hwt_dir)
    if sample_file:
        extract_static_grid(sample_file, args.output_grid)

if __name__ == "__main__":
    main()
