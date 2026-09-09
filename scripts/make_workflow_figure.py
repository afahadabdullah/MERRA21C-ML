#!/usr/bin/env python3
"""Create a publication-style, data-backed MERRA21C-ML workflow figure.

Without --archive, download a fixed real-weather case: NASA GEOS-FP at
2025-07-04 10:30 UTC and NOAA HRRR valid at 11:00 UTC during the central-Texas
flood. HRRR is public visual context because HWT is not public. Pass --archive
on Discover to render the same diagram from the exact project data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import urllib.request
import zipfile

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io import shapereader
import cfgrib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np
import xarray as xr


INK = "#102a43"
NAVY = "#0b3558"
BLUE = "#1769aa"
TEAL = "#118c91"
GOLD = "#d9981e"
MUTED = "#5f7386"
LIGHT = "#edf3f7"
LINE = "#b8c8d4"
PROJECTION = ccrs.LambertConformal(
    central_longitude=-97.5, central_latitude=38.5, standard_parallels=(33, 45)
)
PC = ccrs.PlateCarree()
EXTENT = (-125, -66.5, 24, 50.5)

NASA_URL = (
    "https://portal.nccs.nasa.gov/datashare/gmao/geos-fp/das/Y2025/M07/D04/"
    "GEOS.fp.asm.tavg1_2d_slv_Nx.20250704_1030.V01.nc4"
)
HRRR_URL = (
    "https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.20250704/conus/"
    "hrrr.t10z.wrfsfcf01.grib2"
)
HRRR_MESSAGES = {
    "pres": ":PRES:surface:1 hour fcst:",
    "t2m": ":TMP:2 m above ground:1 hour fcst:",
    "u10": ":UGRD:10 m above ground:1 hour fcst:",
    "v10": ":VGRD:10 m above ground:1 hour fcst:",
    "apcp": ":APCP:surface:0-1 hour acc fcst:",
}
STATES_URL = (
    "https://naturalearth.s3.amazonaws.com/50m_cultural/"
    "ne_50m_admin_1_states_provinces_lines.zip"
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", help="Prepared archive; omit for public weather case")
    parser.add_argument("--predictions", help="Inference directory used with --archive")
    parser.add_argument("--timestamp", help="Archive id; defaults to first usable hour")
    parser.add_argument("--member", type=int, default=0)
    parser.add_argument("--cache", default="data/workflow_public_case")
    parser.add_argument("--output", default="docs/assets/merraflow-workflow.png")
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args()


def download(url, destination, byte_range=None):
    destination = Path(destination)
    if destination.exists() and destination.stat().st_size > 1000:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        url, headers={"User-Agent": "MERRA21C-ML workflow figure"}
    )
    if byte_range:
        request.add_header("Range", f"bytes={byte_range}")
    temporary = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(request, timeout=120) as source, open(temporary, "wb") as target:
        shutil.copyfileobj(source, target)
    temporary.replace(destination)
    return destination


def state_shapefile(cache):
    folder = cache / "natural-earth-states"
    shape = folder / "ne_50m_admin_1_states_provinces_lines.shp"
    if shape.exists():
        return shape
    archive = download(STATES_URL, cache / "natural-earth-states.zip")
    folder.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        source.extractall(folder)
    return shape


def read_grib(path):
    dataset = cfgrib.open_dataset(path, backend_kwargs={"indexpath": ""})
    return dataset, next(iter(dataset.data_vars))


def hrrr_ranges(index_path):
    records = []
    for line in Path(index_path).read_text().splitlines():
        fields = line.split(":", 2)
        if len(fields) >= 3:
            records.append((int(fields[1]), line))
    ranges = {}
    for name, signature in HRRR_MESSAGES.items():
        for position, (start, line) in enumerate(records):
            if signature in line:
                if position + 1 >= len(records):
                    raise ValueError(f"Cannot determine end byte for {line}")
                ranges[name] = f"{start}-{records[position + 1][0] - 1}"
                break
        else:
            raise ValueError(f"HRRR message not found: {signature}")
    return ranges


def load_public_case(cache):
    cache = Path(cache)
    geos_path = download(NASA_URL, cache / Path(NASA_URL).name)
    index_path = download(HRRR_URL + ".idx", cache / "hrrr_20250704_f01.idx")
    grib = {
        name: download(HRRR_URL, cache / f"hrrr_20250704_{name}.grib2", span)
        for name, span in hrrr_ranges(index_path).items()
    }
    with xr.open_dataset(geos_path) as source:
        geos = source.sel(
            lon=slice(EXTENT[0] - 1, EXTENT[1] + 1),
            lat=slice(EXTENT[2] - 1, EXTENT[3] + 1),
        ).isel(time=0).load()
    hrrr = {}
    latitude = longitude = None
    for key, path in grib.items():
        dataset, variable = read_grib(path)
        hrrr[key] = dataset[variable].values.astype("float32")
        latitude = dataset.latitude.values.astype("float32")
        longitude = (((dataset.longitude.values + 180) % 360) - 180).astype("float32")
        dataset.close()
    return {
        "input_lat": geos.lat.values,
        "input_lon": geos.lon.values,
        "input": np.stack(
            [geos.T2M.values, geos.PS.values, np.hypot(geos.U10M.values, geos.V10M.values)]
        ),
        "input_u": geos.U10M.values,
        "input_v": geos.V10M.values,
        "target_lat": latitude,
        "target_lon": longitude,
        "target": np.stack(
            [hrrr["t2m"], hrrr["apcp"], hrrr["pres"], np.hypot(hrrr["u10"], hrrr["v10"])]
        ),
        "timestamp": "2025-07-04 10:30–11:00 UTC",
        "input_source": "NASA GEOS-FP 0.25° analysis",
        "target_source": "NOAA HRRR 3 km public visual context",
        "footer": (
            "Maps: NASA GEOS-FP 2025-07-04 10:30 UTC and NOAA HRRR valid 11:00 UTC, "
            "during the central-Texas flood. HRRR is visual context only—not HWT data or a prediction."
        ),
        "public_proxy": True,
        "state_shape": state_shapefile(cache),
    }


def load_project_case(archive_path, predictions_path, timestamp, member):
    archive = Path(archive_path)
    with open(archive / "index.json") as source:
        index = json.load(source)
    entries = index["entries"]
    if timestamp:
        entries = [entry for entry in entries if entry["id"] == timestamp]
    if not entries:
        raise ValueError("No matching prepared timestamp")
    entry = entries[0]
    shard = archive / entry["id"]
    baseline = np.load(shard / "baseline.npy")
    target = np.load(shard / "target.npy")
    with np.load(archive / "static.npz") as source:
        lat, lon = source["lat"], source["lon"]
    prediction_note = ""
    if predictions_path:
        prediction = Path(predictions_path) / f"{entry['id']}_m{member:03d}.nc"
        if prediction.exists():
            prediction_note = f" • prediction available: {prediction.name}"
    return {
        "input_lat": lat,
        "input_lon": lon,
        "input": baseline[[0, 2, 3]],
        "input_u": None,
        "input_v": None,
        "target_lat": lat,
        "target_lon": lon,
        "target": target,
        "timestamp": entry["time"],
        "input_source": "project GEOS-FP predictors on HWT LCC",
        "target_source": "project HWT 3 km targets",
        "footer": f"Maps rendered from prepared project archive {archive} • {entry['id']}{prediction_note}",
        "public_proxy": False,
        "state_shape": None,
    }


def add_states(ax, state_shape):
    if state_shape:
        ax.add_geometries(
            list(shapereader.Reader(state_shape).geometries()), PC,
            facecolor="none", edgecolor="#42576a", linewidth=0.32,
            alpha=0.72, zorder=5,
        )


def map_panel(
    fig, bounds, lon, lat, field, title, unit, cmap, limits,
    state_shape=None, vectors=None, stride=1, norm=None,
):
    left, bottom, width, _ = bounds
    ax = fig.add_axes(bounds, projection=PROJECTION)
    ax.set_extent(EXTENT, crs=PC)
    values = np.asarray(field)[::stride, ::stride]
    if np.ndim(lon) == 1:
        xx, yy = np.meshgrid(lon, lat)
    else:
        xx, yy = lon, lat
    xx = np.asarray(xx)[::stride, ::stride]
    yy = np.asarray(yy)[::stride, ::stride]
    color_options = {"norm": norm} if norm is not None else {
        "vmin": limits[0], "vmax": limits[1]
    }
    mesh = ax.pcolormesh(
        xx, yy, values, transform=PC, cmap=cmap, shading="auto",
        rasterized=True, **color_options,
    )
    ax.coastlines("50m", color=INK, linewidth=0.55, zorder=6)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), edgecolor=INK, linewidth=0.45, zorder=6)
    add_states(ax, state_shape)
    if vectors is not None:
        u, v = vectors
        qstride = max(1, int(np.ceil(np.shape(u)[-1] / 20)))
        qx, qy = np.meshgrid(lon[::qstride], lat[::qstride])
        ax.quiver(
            qx, qy, u[::qstride, ::qstride], v[::qstride, ::qstride],
            transform=PC, color="#21394c", scale=210, width=0.0022,
            headwidth=3.2, alpha=0.78, zorder=7,
        )
    ax.set_title(title, loc="left", fontsize=9.2, color=INK, pad=5, fontweight="semibold")
    cax = fig.add_axes([left + 0.012, bottom - 0.017, width - 0.024, 0.008])
    colorbar = fig.colorbar(mesh, cax=cax, orientation="horizontal")
    colorbar.ax.tick_params(labelsize=6.2, length=2, colors=MUTED, pad=1)
    colorbar.outline.set_linewidth(0.45)
    colorbar.ax.set_xlabel(unit, fontsize=6.5, color=MUTED, labelpad=1)


def panel_frame(fig, left, width, label, title, subtitle):
    bottom, top = 0.09, 0.875
    fig.patches.append(
        Rectangle(
            (left, bottom), width, top - bottom, transform=fig.transFigure,
            facecolor="white", edgecolor=LINE, linewidth=0.9, zorder=-5,
        )
    )
    fig.text(left + 0.012, top - 0.025, label, color=BLUE, fontsize=12,
             fontweight="bold", va="center")
    fig.text(left + 0.043, top - 0.025, title, color=INK, fontsize=11.5,
             fontweight="bold", va="center")
    fig.lines.append(
        plt.Line2D(
            [left + 0.012, left + width - 0.012], [top - 0.047, top - 0.047],
            transform=fig.transFigure, color=BLUE, lw=1.4,
        )
    )
    fig.text(left + 0.012, top - 0.066, subtitle, color=MUTED, fontsize=7.4, va="top")


def stage_arrow(fig, x0, x1, y=0.485):
    fig.patches.append(
        FancyArrowPatch(
            (x0, y), (x1, y), transform=fig.transFigure, arrowstyle="-|>",
            mutation_scale=18, color=NAVY, linewidth=1.4, zorder=20,
        )
    )


def data_stage(fig, left, width, case):
    panel_frame(fig, left, width, "A", "REAL COARSE INPUTS",
                case["input_source"] + " • " + case["timestamp"])
    lon, lat = case["input_lon"], case["input_lat"]
    temperature = case["input"][0] - 273.15
    wind = case["input"][2]
    map_panel(
        fig, (left + 0.016, 0.50, width - 0.032, 0.267), lon, lat, temperature,
        "2 m temperature + 10 m wind", "°C", "RdYlBu_r",
        (np.nanpercentile(temperature, 2), np.nanpercentile(temperature, 98)),
        case["state_shape"],
        None if case["input_u"] is None else (case["input_u"], case["input_v"]),
    )
    map_panel(
        fig, (left + 0.016, 0.188, width - 0.032, 0.235), lon, lat, wind,
        "10 m wind speed", "m s⁻¹", "magma", (0, np.nanpercentile(wind, 99)),
        case["state_shape"],
    )
    fig.text(
        left + 0.018, 0.125,
        "11 dynamic predictors\nT2M • PRECTOT • PS • U/V • humidity • SLP • ω₅₀₀",
        color=INK, fontsize=7.5, linespacing=1.45, va="bottom",
    )


def alignment_stage(fig, left, width, case):
    panel_frame(fig, left, width, "B", "PAIR + ALIGN",
                "exact hourly pairing • common HWT Lambert grid")
    y = 0.752
    fig.lines.append(
        plt.Line2D(
            [left + 0.035, left + width - 0.035], [y, y],
            transform=fig.transFigure, color=BLUE, lw=2,
        )
    )
    for position, time, note, color in [
        (0.10, "00:00", "window start", MUTED),
        (0.50, "00:30", "GEOS-FP + HWT state", BLUE),
        (0.90, "01:00", "APCP end label", TEAL),
    ]:
        px = left + position * width
        fig.patches.append(plt.Circle((px, y), 0.006, transform=fig.transFigure, color=color, zorder=3))
        fig.text(px, y - 0.020, time, ha="center", fontsize=7.2, color=color, fontweight="bold")
        fig.text(px, y - 0.038, note, ha="center", fontsize=5.9, color=MUTED)
    coarse = case["input"][0] - 273.15
    fine = case["target"][0] - 273.15
    limits = (
        min(np.nanpercentile(coarse, 2), np.nanpercentile(fine, 2)),
        max(np.nanpercentile(coarse, 98), np.nanpercentile(fine, 98)),
    )
    half = (width - 0.046) / 2
    map_panel(
        fig, (left + 0.014, 0.438, half, 0.205), case["input_lon"], case["input_lat"],
        coarse, "~25 km information", "°C", "RdYlBu_r", limits, case["state_shape"],
    )
    map_panel(
        fig, (left + 0.032 + half, 0.438, half, 0.205), case["target_lon"],
        case["target_lat"], fine, "~3 km structure", "°C", "RdYlBu_r", limits,
        case["state_shape"], stride=4 if case["public_proxy"] else 1,
    )
    fig.text(left + 0.016, 0.372, "PREPROCESS", color=TEAL, fontsize=8.5, fontweight="bold")
    for index, text in enumerate([
        "validate time, units, grids, finite values",
        "derive wind speed and temporal/static features",
        "fit normalization on training hours only",
        "write memory-mapped hourly shards",
    ]):
        yy = 0.337 - index * 0.044
        fig.patches.append(plt.Circle((left + 0.023, yy + 0.004), 0.004,
                                      transform=fig.transFigure, color=TEAL))
        fig.text(left + 0.035, yy, text, fontsize=7.0, color=INK, va="bottom")
    fig.text(left + 0.016, 0.135, "CHRONOLOGICAL SPLIT", fontsize=7.5,
             color=INK, fontweight="bold")
    cursor, total = left + 0.016, width - 0.032
    for fraction, color, label in [
        (0.64, BLUE, "TRAIN"), (0.04, LIGHT, "48 h"), (0.13, TEAL, "VAL"),
        (0.04, LIGHT, "48 h"), (0.15, NAVY, "TEST"),
    ]:
        section = total * fraction
        fig.patches.append(Rectangle((cursor, 0.108), section, 0.022,
                                     transform=fig.transFigure, facecolor=color,
                                     edgecolor="white", linewidth=0.3))
        fig.text(cursor + section / 2, 0.119, label,
                 color="white" if color != LIGHT else MUTED, fontsize=5.4,
                 ha="center", va="center", fontweight="bold")
        cursor += section


def tensor_stage(fig, left, width, case):
    note = "HWT variables in production"
    if case["public_proxy"]:
        note += " • maps use public HRRR context"
    panel_frame(fig, left, width, "C", "CONDITIONS → TARGETS", note)
    fig.text(left + 0.018, 0.754, "CONDITION TENSOR  c", fontsize=8.8,
             color=INK, fontweight="bold")
    cursor, y0, total = left + 0.018, 0.710, width - 0.036
    for count, color, label in [
        (11, BLUE, "weather"), (6, GOLD, "static"),
        (6, TEAL, "time"), (4, NAVY, "baseline"),
    ]:
        section = total * count / 27
        fig.patches.append(Rectangle((cursor, y0), section, 0.031,
                                     transform=fig.transFigure, facecolor=color,
                                     edgecolor="white", linewidth=0.4))
        fig.text(cursor + section / 2, y0 + 0.0155, f"{count} {label}", color="white",
                 fontsize=6.0, ha="center", va="center", fontweight="bold")
        cursor += section
    fig.text(left + width / 2, 0.687, "27 normalized channels at every LCC pixel",
             ha="center", fontsize=6.6, color=MUTED)
    fields = case["target"]
    specifications = [
        (fields[0] - 273.15, "Temperature", "°C", "RdYlBu_r", None),
        (fields[1], "1-h precipitation • Texas flood", "mm h⁻¹", "YlGnBu",
         (0, np.nanpercentile(fields[1], 99.9))),
        (fields[2] / 100, "Surface pressure", "hPa", "Spectral_r", None),
        (fields[3], "10 m wind speed", "m s⁻¹", "magma",
         (0, np.nanpercentile(fields[3], 99))),
    ]
    map_width = (width - 0.050) / 2
    for index, (field, title, unit, cmap, limits) in enumerate(specifications):
        if limits is None:
            limits = (np.nanpercentile(field, 2), np.nanpercentile(field, 98))
        col, row = index % 2, index // 2
        precipitation_norm = None
        if index == 1:
            precipitation_norm = PowerNorm(gamma=0.42, vmin=0, vmax=limits[1])
        map_panel(
            fig,
            (left + 0.014 + col * (map_width + 0.022), 0.437 - row * 0.258,
             map_width, 0.185),
            case["target_lon"], case["target_lat"], field, title, unit, cmap, limits,
            case["state_shape"], stride=4 if case["public_proxy"] else 1,
            norm=precipitation_norm,
        )
    fig.text(
        left + width / 2, 0.112,
        "x₁ = standardize[ transform(target) − transform(coarse baseline) ]",
        ha="center", fontsize=7.1, color=INK, fontweight="semibold",
    )


def model_stage(fig, left, width):
    panel_frame(fig, left, width, "D", "CONDITIONAL FLOW MATCHING",
                "patch-memory bounded • EMA model for validation and sampling")
    fig.text(left + 0.018, 0.757, "STRAIGHT CONDITIONAL PATH", color=INK,
             fontsize=8.6, fontweight="bold")
    nodes = [
        (0.12, "x₀", "Gaussian\nnoise", BLUE),
        (0.50, "xₜ", "noisy target\nat time t", TEAL),
        (0.88, "vθ", "U-Net\nvelocity", NAVY),
    ]
    y = 0.680
    for fraction, symbol, note, color in nodes:
        cx = left + fraction * width
        fig.patches.append(plt.Circle((cx, y), 0.026, transform=fig.transFigure,
                                      facecolor="white", edgecolor=color, linewidth=1.7))
        fig.text(cx, y + 0.004, symbol, ha="center", va="center", fontsize=11,
                 color=color, fontweight="bold")
        fig.text(cx, y - 0.045, note, ha="center", va="top", fontsize=6.2, color=MUTED)
    stage_arrow(fig, left + 0.21 * width, left + 0.41 * width, y)
    stage_arrow(fig, left + 0.59 * width, left + 0.79 * width, y)
    fig.text(left + width / 2, 0.585, "xₜ = (1 − t)x₀ + tx₁     •     t ~ U(0,1)",
             ha="center", fontsize=8.0, color=INK)
    fig.text(left + 0.018, 0.542, "VELOCITY U-NET", color=INK,
             fontsize=8.6, fontweight="bold")
    center = left + width / 2
    for index, (yy, block_width) in enumerate(zip(
        [0.500, 0.466, 0.432, 0.398], [0.060, 0.050, 0.040, 0.030]
    )):
        fig.patches.append(Rectangle((center - 0.076 - block_width / 2, yy),
                                     block_width, 0.023, transform=fig.transFigure,
                                     facecolor=BLUE, alpha=0.90 - index * 0.11,
                                     edgecolor="none"))
        fig.patches.append(Rectangle((center + 0.076 - block_width / 2, yy),
                                     block_width, 0.023, transform=fig.transFigure,
                                     facecolor=TEAL, alpha=0.90 - index * 0.11,
                                     edgecolor="none"))
        fig.lines.append(plt.Line2D([center - 0.046, center + 0.046],
                                    [yy + 0.0115, yy + 0.0115],
                                    transform=fig.transFigure, color=LINE, lw=0.7))
    fig.patches.append(Rectangle((center - 0.018, 0.363), 0.036, 0.025,
                                 transform=fig.transFigure, facecolor=NAVY,
                                 edgecolor="none"))
    fig.text(left + 0.020, 0.382, "downsample", fontsize=6.0, color=BLUE)
    fig.text(left + width - 0.020, 0.382, "upsample", fontsize=6.0,
             color=TEAL, ha="right")
    fig.text(center, 0.345, "GroupNorm residual blocks • time-conditioned scale/shift",
             ha="center", fontsize=6.4, color=MUTED)
    fig.text(left + 0.018, 0.304, "CORE-ONLY PHYSICAL LOSS", color=INK,
             fontsize=8.6, fontweight="bold")
    outer_x, outer_y, outer_size = left + 0.024, 0.190, 0.095
    fig.patches.append(Rectangle((outer_x, outer_y), outer_size, outer_size,
                                 transform=fig.transFigure, facecolor=LIGHT,
                                 edgecolor=BLUE, linewidth=1.1))
    inset = 0.017
    fig.patches.append(Rectangle((outer_x + inset, outer_y + inset),
                                 outer_size - 2 * inset, outer_size - 2 * inset,
                                 transform=fig.transFigure, facecolor="#76b9d6",
                                 edgecolor=NAVY, linewidth=1.0))
    fig.text(outer_x + outer_size / 2, outer_y + outer_size / 2, "128² core",
             ha="center", va="center", fontsize=7, color="white", fontweight="bold")
    fig.text(outer_x + outer_size / 2, outer_y - 0.015, "160² input + halo",
             ha="center", fontsize=6.2, color=MUTED)
    fig.text(left + 0.137, 0.255, "AREA-weighted", fontsize=7.1,
             color=INK, fontweight="bold")
    fig.text(left + 0.137, 0.225, "channel-weighted MSE", fontsize=7.1, color=INK)
    fig.text(left + 0.137, 0.195, "AdamW • cosine • clipping", fontsize=7.1, color=INK)
    fig.text(
        left + 0.018, 0.125,
        "Inference: Gaussian member → 24-step Heun ODE → overlap blend → precipitation budget projection",
        fontsize=6.8, color=MUTED, wrap=True,
    )


def render(args):
    case = (
        load_project_case(args.archive, args.predictions, args.timestamp, args.member)
        if args.archive else load_public_case(args.cache)
    )
    fig = plt.figure(figsize=(23.5, 11.6), facecolor="#f8fafc")
    fig.text(0.5, 0.955, "MERRA21C-ML: PHYSICS-CONSTRAINED CONUS DOWNSCALING",
             ha="center", va="center", color=NAVY, fontsize=24, fontweight="bold")
    fig.text(
        0.5, 0.921,
        "native GEOS-FP information  →  HWT 3 km Lambert grid  →  residual conditional flow matching  →  four physical targets",
        ha="center", va="center", color=MUTED, fontsize=10.5,
    )
    margin, gap = 0.018, 0.020
    width = (1 - 2 * margin - 3 * gap) / 4
    lefts = [margin + index * (width + gap) for index in range(4)]
    data_stage(fig, lefts[0], width, case)
    alignment_stage(fig, lefts[1], width, case)
    tensor_stage(fig, lefts[2], width, case)
    model_stage(fig, lefts[3], width)
    for index in range(3):
        stage_arrow(fig, lefts[index] + width + 0.003, lefts[index + 1] - 0.003)
    fig.text(0.5, 0.045, case["footer"], ha="center", va="center",
             fontsize=7.2, color=MUTED)
    fig.text(
        0.5, 0.022,
        "All meteorological panels are projected from numerical data by this Python script; boxes and arrows are schematic.",
        ha="center", va="center", fontsize=6.8, color=MUTED, style="italic",
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=args.dpi, facecolor=fig.get_facecolor(),
                bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    render(parse_args())
