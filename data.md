# Dataset Documentation: High-Resolution & Low-Resolution Reanalysis

This document records the exact file paths, grid definitions, variables, and temporal specifications for the high-resolution target data and low-resolution predictor data on the NASA NCCS Discover supercomputer.

---

## 1. High-Resolution Reanalysis (Target Data)

* **Experiment:** `Feature-c2160_L137` (GEOSgcm-v11.5.1 replay to GEOS-FP)
* **Title:** `CONUS02km_137L_replay_to_GEOS-FP`
* **File Format:** NetCDF-4 (written by `MAPL_PFIO`)
* **Physical Root Path on Discover:**
  ```bash
  /gpfsm/dnb08/projects/p38/HWT/CONUS02KM/Feature-c2160_L137/holding/
  ```
  *(Often symlinked from `/discover/nobackup/projects/gmao/osse2/HWT/CONUS02KM/...`)*

### 1.1 Spatial Grid Specifications
* **Grid Type:** Lambert Conformal Conic (LCC) over CONUS
* **Dimensions:**
  - `Ydim = 1059` (Grid rows)
  - `Xdim = 1799` (Grid columns)
  - Total grid points per horizontal slice: **1,905,141** (~1.9 million points, ~2–3 km nominal resolution)
* **Coordinate Variables:**
  - `lons(Ydim, Xdim)`: 2D longitude array (`degrees_east`)
  - `lats(Ydim, Xdim)`: 2D latitude array (`degrees_north`)
  - `AREA(time, Ydim, Xdim)`: Grid cell area in $\text{m}^2$ (essential for mass-conserving regridding and scale normalization)
  - `Xdim(Xdim)`, `Ydim(Ydim)`: Coordinate dimensions for GrADS compatibility

---

### 1.2 Available High-Resolution Collections

#### Collection 1: `hwt_30mn_slv_LCC` (Surface & Single-Level Fields)
* **Cadence:** Every 30 minutes (`time_increment = 3000` minutes $\implies$ `0000z`, `0030z`, `0100z`, `0130z`, ...)
* **Directory Pattern:**
  ```bash
  /gpfsm/dnb08/projects/p38/HWT/CONUS02KM/Feature-c2160_L137/holding/hwt_30mn_slv_LCC/YYYYMM/
  ```
* **Filename Pattern:**
  ```bash
  Feature-c2160_L137.hwt_30mn_slv_LCC.YYYYMMDD_HHMMz.nc4
  ```
* **Key Variables for Downscaling:**
  - `TMP_2M`: 2-meter air temperature ($\text{K}$) — **Primary Temperature Target**
  - `PRECTOT`: Total precipitation rate ($\text{kg}\ \text{m}^{-2}\ \text{s}^{-1}$) — **Primary Precipitation Rate Target**
  - `HGT_SFC`: Surface geopotential height ($\text{m}^2\ \text{s}^{-2}$) $\implies$ Elevation $z = \text{HGT\_SFC} / 9.80665\ \text{m}$ (Static Topography)
  - `PRES_SFC`: Surface pressure ($\text{Pa}$)
  - `SPFH_2M`: 2-meter specific humidity ($\text{kg}\ \text{kg}^{-1}$)
  - `RH_2M`: Near-surface relative humidity ($\%$)
  - `DPT_2M`: 2-meter dew point temperature ($\text{K}$)
  - `UGRD_10M`, `VGRD_10M`, `SPEED`, `GUST`: 10-meter wind fields ($\text{m}\ \text{s}^{-1}$)
  - `PRECCON`: Deep convective precipitation rate ($\text{kg}\ \text{m}^{-2}\ \text{s}^{-1}$)
  - `PRECLSC`: Non-anvil large scale precipitation rate ($\text{kg}\ \text{m}^{-2}\ \text{s}^{-1}$)
  - `RAIN`, `SNOW`, `ICE`, `FRZR`: Precipitation phase separation ($\text{kg}\ \text{m}^{-2}\ \text{s}^{-1}$)
  - `CAPE`, `CIN`: Convective Available Potential Energy and Inhibition ($\text{J}\ \text{kg}^{-1}$)
  - `REFC`: Maximum composite radar reflectivity ($\text{dBZ}$)

---

#### Collection 2: `hwt_01hr_acc_LCC` (Hourly Accumulated Fields)
* **Cadence:** Hourly (`time_increment = 10000` minutes $\implies$ `0000z`, `0100z`, `0200z`, ...)
* **Directory Pattern:**
  ```bash
  /gpfsm/dnb08/projects/p38/HWT/CONUS02KM/Feature-c2160_L137/holding/hwt_01hr_acc_LCC/YYYYMM/
  ```
* **Filename Pattern:**
  ```bash
  Feature-c2160_L137.hwt_01hr_acc_LCC.YYYYMMDD_HH00z.nc4
  ```
* **Key Variables:**
  - `APCP`: Total precipitation accumulation over the 1-hour window ($\text{mm}$)
  - `ACPCP`: Deep convective precipitation accumulation ($\text{kg}\ \text{m}^{-2}\ \text{s}^{-1}$)
  - `NCPCP`: Non-anvil large scale precipitation accumulation ($\text{kg}\ \text{m}^{-2}\ \text{s}^{-1}$)
  - `SNOWACCUM`: Total snowfall accumulation ($\text{mm}$)
  - `KUCHERA_RATIO`: Kuchera snow-to-liquid ratio

---

#### Collection 3: `hwt_30mn_prs_LCC` (3D Pressure Level Fields)
* **Cadence:** Every 30 minutes
* **Vertical Levels:** 11 isobaric levels (`lev = 11`, in $\text{hPa}$)
* **Directory Pattern:**
  ```bash
  /gpfsm/dnb08/projects/p38/HWT/CONUS02KM/Feature-c2160_L137/holding/hwt_30mn_prs_LCC/YYYYMM/
  ```
* **Filename Pattern:**
  ```bash
  Feature-c2160_L137.hwt_30mn_prs_LCC.YYYYMMDD_HHMMz.nc4
  ```
* **Variables:** `TMP` ($\text{K}$), `SPFH` ($\text{kg}\ \text{kg}^{-1}$), `RH`, `UGRD`, `VGRD`, `HGT`, `QG` (graupel), `QR` (rain), `QS` (snow), `QI` (ice), `QL` (liquid water), `DIVG`, `VORT`.

---

## 2. Low-Resolution Reanalysis (Predictor Data)

* **Source:** GEOS-FP (`f5295_fp`) Forward Processing Diagnostics
* **Physical Root Path on Discover:**
  ```bash
  /gpfsm/dnb06/projects/p174/f5295_fp/diag/Y2025/M01/
  ```

### 2.1 Spatial Grid Specifications
* **Grid Type:** Native Regular Latitude–Longitude
* **Dimensions:**
  - `lat = 721` ($\Delta \text{lat} = 180^\circ / 720 = 0.25^\circ$)
  - `lon = 1152` ($\Delta \text{lon} = 360^\circ / 1152 = 0.3125^\circ$)
  - Total grid points globally: **830,592**

### 2.2 Temporal Frequency & Alignment Strategy
* **Cadence:** **Hourly (1-hour time-averaged)** at `:30z` (`0030z`, `0130z`, `0230z`, ..., `2330z`, 24 files/day per collection).
* **Timestamps:** `T2M` and `PRECTOT` are synchronous, sharing the exact same hourly timestamp sequence.

### 2.3 Primary Low-Res Predictor Collections
1. **`f5295_fp.tavg1_2d_slv_Nx.YYYYMMDD_HH30z.nc4`** (Single-Level Diagnostics):
   - **`T2M`**: 2-meter air temperature ($\text{K}$) — *Primary Temperature Predictor*
   - `U10M`, `V10M`: 10-meter eastward and northward wind components ($\text{m}\ \text{s}^{-1}$)
   - `Q2M`: 2-meter specific humidity ($\text{kg}\ \text{kg}^{-1}$)
   - `SLP`: Sea level pressure ($\text{Pa}$)
   - `PS`: Surface pressure ($\text{Pa}$)
   - `TS`: Surface skin temperature ($\text{K}$)

2. **`f5295_fp.tavg1_2d_flx_Nx.YYYYMMDD_HH30z.nc4`** (Surface Flux Diagnostics):
   - **`PRECTOT`**: Total precipitation rate ($\text{kg}\ \text{m}^{-2}\ \text{s}^{-1}$) — *Primary Precipitation Predictor*
   - `PRECCON`: Convective precipitation rate ($\text{kg}\ \text{m}^{-2}\ \text{s}^{-1}$)
   - `PRECLSC`: Large scale precipitation rate ($\text{kg}\ \text{m}^{-2}\ \text{s}^{-1}$)
   - `EFLUX`, `HFLUX`: Latent and sensible heat fluxes ($\text{W}\ \text{m}^{-2}$)

---

## 3. Paired Dataset Architecture (Synchronous Hourly Mapping)

| Target Variable | Low-Res Predictor (0.25° Lat-Lon) | High-Res Target (3 km LCC) | Cadence & Alignment |
| :--- | :--- | :--- | :--- |
| **2m Air Temp** | **`T2M`** (from `tavg1_2d_slv_Nx`) | **`TMP_2M`** (from `hwt_30mn_slv_LCC`) | Synchronous Hourly |
| **Precipitation** | **`PRECTOT`** (from `tavg1_2d_flx_Nx`) | **`PRECTOT`** (from `hwt_30mn_slv_LCC`) | Synchronous Hourly ($\text{kg}\ \text{m}^{-2}\ \text{s}^{-1}$) |
| **Auxiliary Fields** | `U10M`, `V10M`, `SLP`, `PS`, `Q2M` | High-Res Orography `HGT_SFC` + LCC Grid Area | Orographic lift & thermodynamic conditioning |

---

## 4. Environment & Tooling on Discover

To load NetCDF binaries (`ncdump`), HDF5, and GEOS modules on Discover interactive nodes (e.g., `borgj149`):

```bash
# Source GEOS v12 modules:
source $g/GEOSv12/GEOSgcm/@env/g5_modules

# Or default GMAO environment:
source /discover/nobackup/projects/gmao/share/gmao_SIteam/Environments/GEOSenv
```

