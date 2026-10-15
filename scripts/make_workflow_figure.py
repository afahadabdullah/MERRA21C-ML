#!/usr/bin/env python3
"""Create a publication-style, data-backed MERRA21C-ML workflow figure.

Without --evaluation or --bundle, download a fixed real-weather case: NASA GEOS-FP at
2025-07-04 10:30 UTC and NOAA HRRR valid at 11:00 UTC during the central-Texas
flood. HRRR is public visual context because HWT is not public. Pass --evaluation
on Discover to render the same diagram from saved v4.1 evaluation fields.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import urllib.request


import numpy as np
import xarray as xr


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
    "q2m": ":SPFH:2 m above ground:1 hour fcst:",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', help='Complete v4.1 evaluation with saved example fields')
    parser.add_argument('--bundle', help='Previously saved workflow_data.npz; no source data reads')
    parser.add_argument('--cache', default='data/workflow_public_case')
    parser.add_argument('--output', default='output/figures/workflow_v4_1', help='Output directory')
    parser.add_argument('--zoom', default='-110,-102,35,41.5', help='west,east,south,north; use --zoom=...')
    parser.add_argument('--dpi', type=int, default=300)
    parser.add_argument('--no-features', action='store_true', help='No coast/state files or downloads')
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




def read_grib(path):
    import cfgrib
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
        "native": {name: dict(values=geos[var].values.astype("float32"), lat=geos.lat.values, lon=geos.lon.values)
                   for name,var in [("t2m","T2M"),("ps","PS"),("u10m","U10M"),("v10m","V10M"),("q2m","QV2M")] if var in geos},
        "input_u": geos.U10M.values,
        "input_v": geos.V10M.values,
        "target_lat": latitude,
        "target_lon": longitude,
        "target": np.stack(
            [hrrr["t2m"], hrrr["apcp"], hrrr["pres"], hrrr["u10"], hrrr["v10"], hrrr["q2m"]]
        ),
        "timestamp": "2025-07-04 10:30–11:00 UTC",
        "input_source": "NASA GEOS-FP 0.25° analysis",
        "target_source": "NOAA HRRR 3 km public visual context",
        "footer": (
            "Maps: NASA GEOS-FP 2025-07-04 10:30 UTC and NOAA HRRR valid 11:00 UTC, "
            "during the central-Texas flood. HRRR is visual context only—not HWT data or a prediction."
        ),
        "public_proxy": True,
        "state_shape": None,
    }



def render(args):
    from merraflow.workflow_maps_v4_1 import render as render_maps, save_bundle, load_bundle, from_evaluation
    from merraflow.evaluate_all_v4_1 import read_manifest
    if args.evaluation and args.bundle:
        raise ValueError('Choose --evaluation or --bundle')
    if args.evaluation:
        case=from_evaluation(args.evaluation,read_manifest(args.evaluation))
    elif args.bundle:
        case=load_bundle(args.bundle)
    else:
        cached=Path(args.cache)/'workflow_v4_1.npz'
        if cached.exists():
            case=load_bundle(cached)
        else:
            old=load_public_case(args.cache)
            case=dict(hr=old['target'],lat=old['target_lat'],lon=old['target_lon'],native=old['native'],
                      time=old['timestamp'],native_label='GEOS-FP native 0.25° × 0.3125°',
                      reference_label='HRRR 3 km: public illustration only',public_proxy=True,
                      state_shape=None,
                      footer='Maps: GEOS-FP 4 July 2025, 10:30 UTC; HRRR valid 11:00 UTC. Public visual context only, not HWT data or model predictions.')
            save_bundle(cached,case)
    zoom=tuple(float(x) for x in args.zoom.split(','))
    if len(zoom)!=4 or zoom[0]>=zoom[1] or zoom[2]>=zoom[3]:raise ValueError('Invalid zoom extent')
    render_maps(case,args.output,args.dpi,zoom,features=not args.no_features)
    print(Path(args.output)/'fig01_workflow.png')


if __name__ == '__main__':
    render(parse_args())
