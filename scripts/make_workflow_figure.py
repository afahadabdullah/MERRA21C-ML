#!/usr/bin/env python3
"""Render the README workflow figure from prepared fields and model output.

The figure is intentionally data-backed: every heat map is read from the
prepared archive or an inference NetCDF.  The boxes and arrows are schematic.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
import xarray as xr


NAVY = "#082f55"
BLUE = "#1469a8"
CYAN = "#0b94a5"
GOLD = "#e4a11b"
RED = "#c83e4d"
INK = "#17324d"
MUTED = "#60758a"
PAPER = "#f7f9fb"
BORDER = "#c8d5e1"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", default="runs/smoke_verified/prepared")
    parser.add_argument("--predictions", default="runs/smoke_verified/predictions")
    parser.add_argument("--run-dir", default="runs/smoke_verified/run")
    parser.add_argument("--timestamp", help="Archive id (YYYYMMDD_HHMM); defaults to first predicted hour")
    parser.add_argument("--member", type=int, default=0)
    parser.add_argument("--output", default="docs/assets/merraflow-workflow.png")
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def load_inputs(args):
    archive = Path(args.archive)
    predictions = Path(args.predictions)
    with open(archive / "index.json") as handle:
        index = json.load(handle)
    if args.timestamp:
        tag = args.timestamp
    else:
        candidates = sorted(predictions.glob(f"*_m{args.member:03d}.nc"))
        if not candidates:
            raise FileNotFoundError(f"No member {args.member} predictions in {predictions}")
        tag = candidates[0].name.split("_m", 1)[0]
    entry = next((item for item in index["entries"] if item["id"] == tag), None)
    if entry is None:
        raise ValueError(f"Timestamp {tag} is absent from {archive / 'index.json'}")
    shard = archive / tag
    arrays = {name: np.load(shard / f"{name}.npy") for name in ("condition", "baseline", "truth", "target", "residual")}
    with np.load(archive / "static.npz") as static_file:
        static = {name: static_file[name] for name in static_file.files}
    prediction_path = predictions / f"{tag}_m{args.member:03d}.nc"
    with xr.open_dataset(prediction_path) as dataset:
        generated = np.stack([dataset[name].isel(time=0).values for name in ("t2m", "precip", "ps", "wind10m")])
    history_path = Path(args.run_dir) / "history.jsonl"
    history = []
    if history_path.exists():
        history = [json.loads(line) for line in history_path.read_text().splitlines() if line.strip()]
    return index, entry, arrays, static, generated, history, prediction_path


def add_background(fig):
    ax = fig.add_axes([0, 0, 1, 1], zorder=-20)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    for value in np.linspace(0, 1, 81):
        ax.axvline(value, color="#dfe7ee", lw=0.35, alpha=0.42)
    for value in np.linspace(0, 1, 46):
        ax.axhline(value, color="#dfe7ee", lw=0.35, alpha=0.42)


def add_panel(fig, left, bottom, width, height, title):
    body = FancyBboxPatch(
        (left, bottom), width, height,
        boxstyle="round,pad=0.003,rounding_size=0.009",
        transform=fig.transFigure, facecolor="white", edgecolor=BORDER,
        linewidth=1.1, zorder=-5,
    )
    fig.patches.append(body)
    header_h = 0.047
    header = FancyBboxPatch(
        (left, bottom + height - header_h), width, header_h,
        boxstyle="round,pad=0.003,rounding_size=0.009",
        transform=fig.transFigure, facecolor=NAVY, edgecolor=NAVY,
        linewidth=0, zorder=-4,
    )
    fig.patches.append(header)
    fig.patches.append(Rectangle(
        (left, bottom + height - header_h), width, header_h / 2,
        transform=fig.transFigure, facecolor=NAVY, edgecolor="none", zorder=-3,
    ))
    fig.text(left + width / 2, bottom + height - header_h / 2, title,
             ha="center", va="center", color="white", fontsize=10.5, fontweight="bold")
    return left + 0.012, bottom + 0.018, width - 0.024, height - header_h - 0.03


def add_card(fig, bounds, title, accent=BLUE, subtitle=None):
    x, y, w, h = bounds
    fig.patches.append(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.006",
        transform=fig.transFigure, facecolor="#fbfcfd", edgecolor=BORDER,
        linewidth=0.8, zorder=-2,
    ))
    fig.patches.append(Rectangle(
        (x, y), 0.004, h, transform=fig.transFigure,
        facecolor=accent, edgecolor="none", zorder=-1,
    ))
    fig.text(x + 0.009, y + h - 0.016, title, ha="left", va="top",
             color=accent, fontsize=8.1, fontweight="bold")
    if subtitle:
        fig.text(x + 0.009, y + h - 0.035, subtitle, ha="left", va="top",
                 color=MUTED, fontsize=6.3)


def map_axes(fig, bounds, field, cmap, title, unit="", vmin=None, vmax=None, norm=None, contour=False):
    x, y, w, h = bounds
    ax = fig.add_axes([x, y, w, h])
    image = ax.imshow(field, origin="lower", cmap=cmap, interpolation="nearest", aspect="auto", vmin=vmin, vmax=vmax, norm=norm)
    if contour and np.ptp(field) > 0:
        ax.contour(field, levels=6, colors="white", linewidths=0.35, alpha=0.58)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, color=INK, fontsize=6.8, pad=2.5, fontweight="semibold")
    for spine in ax.spines.values():
        spine.set_color("#8da2b6")
        spine.set_linewidth(0.55)
    cax = fig.add_axes([x + w + 0.002, y, 0.004, h])
    cb = fig.colorbar(image, cax=cax)
    cb.ax.tick_params(labelsize=4.8, length=1.5, colors=MUTED)
    cb.outline.set_linewidth(0.4)
    if unit:
        cb.ax.set_title(unit, fontsize=4.8, color=MUTED, pad=2)
    return ax


def add_arrow(fig, x0, x1, y):
    arrow = FancyArrowPatch(
        (x0, y), (x1, y), transform=fig.transFigure,
        arrowstyle="Simple,tail_width=0.8,head_width=10,head_length=10",
        color=NAVY, linewidth=0, mutation_scale=1, zorder=20,
    )
    fig.patches.append(arrow)


def add_input_panel(fig, bounds, arrays, static):
    x, y, w, h = bounds
    gap = 0.018
    card_h = (h - 2 * gap) / 3
    cards = [
        ("GEOS-FP COARSE WEATHER", "11 regridded hourly predictors", BLUE, arrays["condition"][0], "turbo", "T2M", "K"),
        ("NATIVE PRECIPITATION", "~25 km budget mapped to footprints", CYAN, arrays["baseline"][1], "Blues", "PRECTOT budget", "mm h⁻¹"),
        ("HWT + STATIC LCC GRID", "3 km labels, elevation, area, lat/lon", GOLD, static["elevation"], "terrain", "Topography", "m"),
    ]
    for idx, (title, subtitle, accent, field, cmap, label, unit) in enumerate(cards):
        cy = y + h - (idx + 1) * card_h - idx * gap
        add_card(fig, (x, cy, w, card_h), title, accent, subtitle)
        map_axes(fig, (x + 0.025, cy + 0.018, w - 0.062, card_h - 0.066), field, cmap, "", unit, contour=True)


def add_prepare_panel(fig, bounds, arrays, entry, index):
    x, y, w, h = bounds
    fig.text(x + w / 2, y + h - 0.025, "TIME + GRID ALIGNMENT", ha="center", va="center", color=NAVY, fontsize=8.3, fontweight="bold")
    ty = y + h - 0.085
    fig.lines.append(plt.Line2D([x + 0.025, x + w - 0.025], [ty, ty], transform=fig.transFigure, color=BLUE, lw=1.6))
    for pos, label, color in [(0.08, "00:00\nwindow start", MUTED), (0.50, ":30\nLR + HR state", BLUE), (0.92, "01:00\nAPCP end label", CYAN)]:
        px = x + pos * w
        fig.patches.append(plt.Circle((px, ty), 0.005, transform=fig.transFigure, color=color, zorder=2))
        fig.text(px, ty - 0.015, label, ha="center", va="top", color=color, fontsize=5.9)
    map_y = y + h * 0.42
    map_h = h * 0.245
    temp_min = min(arrays["baseline"][0].min(), arrays["truth"][0].min())
    temp_max = max(arrays["baseline"][0].max(), arrays["truth"][0].max())
    map_axes(fig, (x + 0.008, map_y, w * 0.41, map_h), arrays["baseline"][0], "turbo", "Coarse baseline", "K", temp_min, temp_max)
    map_axes(fig, (x + w * 0.53, map_y, w * 0.41, map_h), arrays["truth"][0], "turbo", "HWT truth", "K", temp_min, temp_max)
    fig.text(x + w / 2, map_y - 0.018, "same LCC grid • paired by exact timestamp", ha="center", va="top", color=MUTED, fontsize=6.1, style="italic")

    split_y = y + h * 0.24
    fig.text(x + 0.005, split_y + 0.046, "CHRONOLOGICAL SPLITS", color=NAVY, fontsize=7.2, fontweight="bold")
    sx, sw, sh = x + 0.005, w - 0.01, 0.024
    train_w, gap_w, val_w, test_w = 0.62, 0.045, 0.14, 0.15
    cursor = sx
    for frac, color, label in [(train_w, BLUE, "TRAIN"), (gap_w, "#e9eef3", ""), (val_w, CYAN, "VAL"), (gap_w, "#e9eef3", ""), (test_w, NAVY, "TEST")]:
        width = sw * frac / (train_w + 2 * gap_w + val_w + test_w)
        fig.patches.append(Rectangle((cursor, split_y), width, sh, transform=fig.transFigure, facecolor=color, edgecolor="white", lw=0.4))
        if label:
            fig.text(cursor + width / 2, split_y + sh / 2, label, ha="center", va="center", color="white", fontsize=5.7, fontweight="bold")
        cursor += width
    split = entry.get("split", "unknown")
    fig.text(x + 0.005, split_y - 0.016, f"shown hour: {entry['id']}  •  split: {split}", color=MUTED, fontsize=5.9)

    box_y = y + 0.018
    add_card(fig, (x, box_y, w, h * 0.16), "PREPARED ARCHIVE", CYAN)
    fig.text(x + 0.012, box_y + h * 0.105, "memory-mapped hourly shards", color=INK, fontsize=6.5)
    fig.text(x + 0.012, box_y + h * 0.074, "train-only normalization statistics", color=INK, fontsize=6.5)
    fig.text(x + 0.012, box_y + h * 0.043, f"{index['condition_channels']} condition channels", color=INK, fontsize=6.5)


def add_tensor_panel(fig, bounds, arrays):
    x, y, w, h = bounds
    fig.text(x + w / 2, y + h - 0.023, "CONDITION TENSOR", ha="center", color=NAVY, fontsize=8.2, fontweight="bold")
    labels = ["11 weather", "6 static", "6 time", "4 baseline"]
    colors = [BLUE, GOLD, CYAN, NAVY]
    bx = x + 0.006
    total_w = w - 0.012
    for idx, (label, color) in enumerate(zip(labels, colors)):
        ww = total_w * [0.38, 0.205, 0.205, 0.21][idx]
        fig.patches.append(Rectangle((bx, y + h - 0.081), ww, 0.029, transform=fig.transFigure, facecolor=color, edgecolor="white", lw=0.5))
        fig.text(bx + ww / 2, y + h - 0.0665, label, ha="center", va="center", color="white", fontsize=5.7, fontweight="bold")
        bx += ww
    fig.text(x + w / 2, y + h - 0.102, "normalized spatial conditions • [27, H, W]", ha="center", color=MUTED, fontsize=6.0)

    fig.text(x + w / 2, y + h - 0.145, "FOUR ML TARGETS", ha="center", color=NAVY, fontsize=8.2, fontweight="bold")
    target = arrays["target"]
    specs = [
        (0, "Temperature", "K", "turbo", None),
        (1, "Precipitation", "mm h⁻¹", "Blues", None),
        (2, "Surface pressure", "Pa", "viridis", None),
        (3, "10 m wind speed", "m s⁻¹", "magma", None),
    ]
    map_w, map_h = w * 0.41, h * 0.205
    for idx, (channel, title, unit, cmap, norm) in enumerate(specs):
        col, row = idx % 2, idx // 2
        mx = x + 0.008 + col * w * 0.52
        my = y + h * 0.37 - row * h * 0.265
        map_axes(fig, (mx, my, map_w, map_h), target[channel], cmap, title, unit, norm=norm)
    fig.text(x + w / 2, y + 0.024, "x₁ = standardized transformed\n(target − coarse baseline)", ha="center", va="bottom", color=INK, fontsize=6.6, fontweight="semibold")


def add_flow_panel(fig, bounds, history):
    x, y, w, h = bounds
    fig.text(x + w / 2, y + h - 0.025, "STRAIGHT-PATH FLOW MATCHING", ha="center", color=NAVY, fontsize=8.2, fontweight="bold")
    fy = y + h - 0.105
    positions = [x + 0.025, x + w * 0.37, x + w * 0.70]
    boxes = [("x₀", "Gaussian noise", BLUE), ("xₜ", "(1−t)x₀ + tx₁", CYAN), ("vθ", "conditional U-Net", NAVY)]
    for px, (symbol, subtitle, color) in zip(positions, boxes):
        fig.patches.append(FancyBboxPatch((px, fy), w * 0.22, 0.065, boxstyle="round,pad=0.003,rounding_size=0.006", transform=fig.transFigure, facecolor="#f4f8fb", edgecolor=color, linewidth=1.1))
        fig.text(px + w * 0.11, fy + 0.043, symbol, ha="center", va="center", color=color, fontsize=11, fontweight="bold")
        fig.text(px + w * 0.11, fy + 0.017, subtitle, ha="center", va="center", color=MUTED, fontsize=5.4)
    add_arrow(fig, positions[0] + w * 0.22, positions[1], fy + 0.032)
    add_arrow(fig, positions[1] + w * 0.22, positions[2], fy + 0.032)
    fig.text(x + w / 2, fy - 0.022, "t ~ Uniform(0, 1)  •  predict velocity x₁ − x₀", ha="center", color=INK, fontsize=6.2)

    patch_y = y + h * 0.44
    add_card(fig, (x + 0.006, patch_y, w - 0.012, h * 0.225), "PATCH TRAINING", GOLD)
    outer_x = x + w * 0.08
    inner_x = x + w * 0.18
    outer = Rectangle((outer_x, patch_y + 0.032), w * 0.34, h * 0.13, transform=fig.transFigure, facecolor="#eaf3fa", edgecolor=BLUE, lw=1.0)
    inner = Rectangle((inner_x, patch_y + 0.053), w * 0.26, h * 0.088, transform=fig.transFigure, facecolor="#74b9dc", edgecolor=NAVY, lw=1.0)
    fig.patches.extend([outer, inner])
    fig.text(inner_x + w * 0.13, patch_y + h * 0.097, "128 × 128\nscored core", ha="center", va="center", color="white", fontsize=6.0, fontweight="bold")
    fig.text(x + w * 0.66, patch_y + h * 0.125, "160 × 160 input", ha="center", color=INK, fontsize=6.4, fontweight="bold")
    fig.text(x + w * 0.66, patch_y + h * 0.087, "16-pixel context halo", ha="center", color=MUTED, fontsize=5.8)
    fig.text(x + w * 0.66, patch_y + h * 0.052, "AREA + channel weighted MSE", ha="center", color=MUTED, fontsize=5.8)

    ax = fig.add_axes([x + 0.025, y + 0.062, w - 0.05, h * 0.24])
    if history:
        epochs = [row["epoch"] + 1 for row in history]
        ax.plot(epochs, [row["train_loss"] for row in history], "o-", color=BLUE, lw=1.4, ms=3.5, label="train")
        ax.plot(epochs, [row["val_loss"] for row in history], "o-", color=CYAN, lw=1.4, ms=3.5, label="EMA validation")
        ax.legend(frameon=False, fontsize=5.3, loc="best")
    ax.set_title("Optimization trace", fontsize=6.9, color=INK, pad=3, fontweight="semibold")
    ax.set_xlabel("epoch", fontsize=5.3, color=MUTED)
    ax.set_ylabel("flow loss", fontsize=5.3, color=MUTED)
    ax.tick_params(labelsize=4.8, colors=MUTED, length=2)
    ax.grid(alpha=0.22, lw=0.5)
    for spine in ax.spines.values():
        spine.set_color(BORDER)
    fig.text(x + w / 2, y + 0.023, "AdamW • warmup + cosine • accumulation • clipping • EMA", ha="center", color=MUTED, fontsize=5.6)


def add_output_panel(fig, bounds, generated, prediction_path):
    x, y, w, h = bounds
    fig.text(x + w / 2, y + h - 0.023, "EMA ENSEMBLE SAMPLE", ha="center", color=NAVY, fontsize=8.2, fontweight="bold")
    fig.text(x + w / 2, y + h - 0.050, "Gaussian field → Heun ODE → overlap blend", ha="center", color=MUTED, fontsize=6.2)
    specs = [
        (0, "Generated T2M", "K", "turbo"),
        (1, "Generated precip", "mm h⁻¹", "Blues"),
        (2, "Generated PS", "Pa", "viridis"),
        (3, "Generated wind", "m s⁻¹", "magma"),
    ]
    map_w, map_h = w * 0.41, h * 0.205
    for idx, (channel, title, unit, cmap) in enumerate(specs):
        col, row = idx % 2, idx // 2
        mx = x + 0.008 + col * w * 0.52
        my = y + h * 0.57 - row * h * 0.26
        map_axes(fig, (mx, my, map_w, map_h), generated[channel], cmap, title, unit)

    card_y = y + h * 0.115
    add_card(fig, (x, card_y, w, h * 0.17), "PHYSICAL OUTPUT CONTRACT", RED)
    fig.text(x + 0.012, card_y + h * 0.105, "nonnegative precipitation", color=INK, fontsize=6.3)
    fig.text(x + 0.012, card_y + h * 0.073, "native-footprint AREA budget", color=INK, fontsize=6.3)
    fig.text(x + 0.012, card_y + h * 0.041, "compressed NetCDF per member", color=INK, fontsize=6.3)
    fig.text(x + w / 2, y + 0.035, prediction_path.name, ha="center", color=NAVY, fontsize=5.9, fontweight="bold")
    fig.text(x + w / 2, y + 0.017, "4 physical fields + provenance + mass audit", ha="center", color=MUTED, fontsize=5.4)


def render(args):
    index, entry, arrays, static, generated, history, prediction_path = load_inputs(args)
    fig = plt.figure(figsize=(19, 10.4), facecolor=PAPER)
    add_background(fig)
    fig.text(0.5, 0.965, "MERRA21C-ML DOWNSCALING WORKFLOW", ha="center", va="center", color=NAVY, fontsize=23, fontweight="bold")
    fig.text(0.5, 0.932, "GEOS-FP ~25 km information  →  HWT ~3 km LCC targets  →  probabilistic high-resolution fields", ha="center", va="center", color=MUTED, fontsize=10.3, style="italic")

    margin, gap, bottom, top = 0.012, 0.017, 0.075, 0.89
    width = (1 - 2 * margin - 4 * gap) / 5
    headers = ["1  INPUT DATA", "2  ALIGN + PREPARE", "3  MODEL TENSORS", "4  FLOW TRAINING", "5  GENERATED FIELDS"]
    panel_bounds = []
    for idx, header in enumerate(headers):
        left = margin + idx * (width + gap)
        panel_bounds.append(add_panel(fig, left, bottom, width, top - bottom, header))
        if idx:
            add_arrow(fig, left - gap + 0.002, left - 0.002, bottom + (top - bottom) * 0.52)

    add_input_panel(fig, panel_bounds[0], arrays, static)
    add_prepare_panel(fig, panel_bounds[1], arrays, entry, index)
    add_tensor_panel(fig, panel_bounds[2], arrays)
    add_flow_panel(fig, panel_bounds[3], history)
    add_output_panel(fig, panel_bounds[4], generated, prediction_path)

    source_text = " ".join([str(args.archive), *[str(value) for value in entry.values()]])
    example = "verified synthetic smoke fixture" if "smoke" in source_text.lower() else "production archive"
    footer = (
        f"Field panels plotted directly from {example} files • timestamp {entry['id']} • "
        "arrows/boxes are schematic • no generative image model used"
    )
    fig.text(0.5, 0.027, footer, ha="center", va="center", color=MUTED, fontsize=6.5)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=args.dpi, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    render(parse_args())
