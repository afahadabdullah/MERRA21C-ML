# MERRA21C-ML: High-Resolution Atmospheric Downscaling over CONUS Using Flow Matching

## 1. Executive Summary

This project implements a machine learning framework for statistical and generative downscaling of reanalysis and atmospheric model data over the Contiguous United States (CONUS). 

- **Input (Low Resolution):** MERRA21C / GEOS-FP reanalysis on a ~0.25° latitude/longitude grid (~25 km).
- **Target (High Resolution):** High-resolution regional reanalysis on a ~3 km Lambert Conformal Conic (LCC) grid.
- **Initial Target Variables:** 
  - 2-meter Temperature (`t2m` / T2M)
  - Precipitation (`precip` / PRECTOT / PRECTOTCORR)
- **Primary ML Framework:** Conditional Flow Matching (CFM) / Optimal Transport Flow Matching (OT-CFM) with deep generative architectures (e.g., U-Net / DiT backbones).
- **Computing Platform:** NASA Center for Climate Simulation (NCCS) **Discover** Supercomputer.
- **Project Directory & Conda Environment:**
  ```bash
  /gpfsm/dnb10/projects/p311/ML_downscaling/
  ```

---

## 2. Data Sources & Storage Paths (NASA Discover)

### 2.1 High-Resolution Target (Ground Truth: c2160 L137 HWT Replay)
* **Storage Root:**
  ```bash
  /gpfsm/dnb08/projects/p38/HWT/CONUS02KM/Feature-c2160_L137/holding/
  # (or symlinked to /discover/nobackup/projects/gmao/osse2/HWT/CONUS02KM/...)
  ```
* **Grid Specifications:**
  - Dimensions: `Ydim = 1059`, `Xdim = 1799` (~1.9M points/slice, 2-3 km LCC grid).
  - Coordinates: 2D `lats(Ydim, Xdim)`, `lons(Ydim, Xdim)`, and `AREA(time, Ydim, Xdim)` ($m^2$).
* **Collections & Exact Variable Names:**
  - **`hwt_30mn_slv_LCC`** (30-min single-level):
    - `TMP_2M`: 2-meter air temperature ($K$)
    - `PRECTOT`: Total precipitation rate ($kg\ m^{-2}\ s^{-1}$)
    - `HGT_SFC`: Surface geopotential height ($m^2\ s^{-2} \rightarrow z = \text{HGT\_SFC}/9.80665\ m$)
    - `PRES_SFC`: Surface pressure ($Pa$)
    - `UGRD_10M`, `VGRD_10M`, `SPEED`: 10m wind fields
  - **`hwt_01hr_acc_LCC`** (1-hour accumulated):
    - `APCP`: Total precipitation accumulation ($mm$)
    - `SNOWACCUM`: Snowfall accumulation ($mm$)

### 2.2 Low-Resolution Input (Predictors: GEOS-FP Diagnostics)
* **Storage Root:**
  ```bash
  /gpfsm/dnb06/projects/p174/f5295_fp/diag/Y2025/M01/
  ```
* **Grid:** Standard $1152 \times 721$ regular latitude-longitude grid ($0.3125^\circ \times 0.25^\circ$).
* **Cadence:** Hourly at `:30z` (`0030z`, `0130z`, ..., `2330z`).
* **Predictor Collections & Variables:**
  - **`tavg1_2d_slv_Nx`**: **`T2M`** (2m air temp, $\text{K}$), `U10M`, `V10M`, `SLP`, `PS`, `Q2M`
  - **`tavg1_2d_flx_Nx`**: **`PRECTOT`** (Total precip rate, $\text{kg}\ \text{m}^{-2}\ \text{s}^{-1}$), `PRECCON`, `PRECLSC`

### 2.3 Static Conditioning Fields (Extracted from `hwt_30mn_slv_LCC`)
* **High-Resolution Orography ($Z_{\text{HR}}$):** Derived once from `HGT_SFC / 9.80665`.
* **Grid Cell Area:** `AREA` array for local map scale factor and mass conservation weighting.
* **Geographic Encodings:** $\sin(\text{lat}), \cos(\text{lat}), \sin(\text{lon}), \cos(\text{lon})$.

---

## 3. Methodology & Modeling Architecture

### 3.1 Why Flow Matching for Downscaling?
Traditional downscaling methods (e.g., bilinear interpolation, CNN regression) suffer from over-smoothing (loss of spatial variance and extreme values). Diffusion models resolve this by generating realistic high-frequency textures and capturing multiscale variability, but standard diffusion requires hundreds of denoising steps.

**Conditional Flow Matching (CFM)** provides:
- Continuous Normalizing Flows (CNFs) with straight probability paths (via Optimal Transport paths).
- Faster, deterministic or stochastic sampling in significantly fewer function evaluations (10–25 ODE steps).
- Stable simulation-free training objectives minimizing regression error on the velocity vector field:
  $$\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t, x_0, x_1} \left[ \left\| v_\theta(x_t, t, c_{\text{LR}}) - (x_1 - x_0) \right\|^2 \right]$$
  where:
  - $x_1 \sim p_{\text{data}}$ is the high-resolution ground truth (3 km LCC).
  - $x_0 \sim p_0$ is a standard normal prior $\mathcal{N}(0, I)$ or a base-interpolated state.
  - $x_t = (1 - t)x_0 + t x_1$ represents the linear interpolation path ($t \in [0, 1]$).
  - $c_{\text{LR}}$ is the low-resolution conditioning field (MERRA21C/GEOS-FP).
  - $v_\theta$ is the parameterized neural vector field.

### 3.2 Network Backbone Options
1. **Multi-Scale Conditioning U-Net:**
   - Residual convolutional blocks with spatial self-attention at deep stages.
   - Cross-attention or adaptive group normalization (AdaGN) for time step $t$ and low-resolution context embeddings.
2. **Diffusion Transformer (DiT):**
   - Patchified input representations with spatial transformer blocks, scaling effectively across large patch sizes and multi-channel conditioning.

### 3.3 Variable Processing Strategies
- **2-meter Temperature (`t2m`):**
  - Standard continuous field with strong physical correlation to elevation.
  - Normalization: Z-score standard scaling $(\mu, \sigma)$ calculated over the training climatology.
- **Precipitation (`precip`):**
  - Highly intermittent, skewed, heavy-tailed distribution with significant zero-inflation (dry points).
  - Normalization & Transformation:
    - Log-transform / Power transform: $\tilde{y} = \log(1 + y / y_0)$ or Box-Cox transformation.
    - Specialized loss components (e.g., extreme value weighting, spectral/energy conservation terms).

---

## 4. Implementation Phases

### Phase 1: Data Audit, Preprocessing & Pairing Pipeline
- [ ] **Directory & Metadata Inspection:**
  - Inspect sample files from `/discover/nobackup/projects/gmao/osse2/HWT/CONUS02KM/...` and `/discover/nobackup/projects/gmao/geos_fp_arch/f5295_fp`.
  - Extract grid metadata: LCC projection parameters (standard parallels, central meridian, origin), bounding boxes, spatial dimensions, and timestamps.
- [ ] **Temporal Alignment & Colocation:**
  - Identify overlapping date ranges and observation frequencies (hourly vs 3-hourly).
  - Build an index/manifest mapping pairs of `(time_stamp, low_res_path, high_res_path)`.
- [ ] **Spatial Preprocessing & Patching:**
  - Crop or regrid low-resolution inputs over CONUS bounding domain.
  - Implement spatial patch extraction (e.g., $128 \times 128$ or $256 \times 256$ patches) to fit high-resolution training on modern GPU memory.
  - Compute global and seasonal normalization statistics $(\mu, \sigma, \min, \max)$.
- [ ] **High-Performance Storage Format:**
  - Store paired datasets as Zarr, HDF5, or WebDataset shards on `/discover/nobackup` for maximum multi-worker I/O throughput.

### Phase 2: Model Architecture & Training Framework
- [ ] **Flow Matching Framework Setup:**
  - Implement OT-CFM vector field generation and ODE integration (Euler, Midpoint, adaptive Runge-Kutta).
  - Design conditioning modules (bilinear/bicubic interpolated LR fields + static terrain fields concatenated or injected via cross-attention).
- [ ] **Distributed Multi-GPU Pipeline:**
  - Build PyTorch training loop using PyTorch Lightning or HuggingFace Accelerate with DDP.
  - Enable mixed precision (AMP with BF16 or FP16).
  - Create SLURM job submission scripts tailored for NCCS Discover GPU nodes (e.g., 4x A100 / V100 partitions).
- [ ] **Logging & Checkpointing:**
  - TensorBoard / MLflow / Weights & Biases logging for loss curves and sample validation snapshots.

### Phase 3: Evaluation, Metrics & Meteorological Verification
- [ ] **Point-wise Deterministic Metrics:**
  - Root Mean Square Error (RMSE), Mean Absolute Error (MAE), Mean Bias.
  - Pearson Spatial Correlation.
- [ ] **Distributional & Probabilistic Metrics:**
  - Continuous Ranked Probability Score (CRPS) across ensemble members/trajectories.
  - Power Spectral Density (PSD) analysis to verify that generated fields reproduce the correct physical kinetic energy / spatial variance spectra without excessive smoothing or unphysical noise.
- [ ] **Precipitation-Specific Validation:**
  - Fractions Skill Score (FSS) at multiple spatial scales and precipitation thresholds (e.g., 1 mm/hr, 5 mm/hr, 25 mm/hr).
  - Extreme tail comparison (95th, 99th, 99.9th percentiles).
- [ ] **Full-Domain Reconstruction:**
  - Implement full-CONUS inference with smooth patch stitching (overlapping Hann window blending) to prevent edge artifacts.

### Phase 4: Production & Export
- [ ] **CLI & Batch Inference Tools:**
  - Scripts to execute downscaling for arbitrary MERRA21C date ranges on Discover.
- [ ] **CF-Compliant Outputs:**
  - Output high-resolution downscaled products in standardized NetCDF-4/Zarr formats with complete metadata and projection attributes.

---

## 5. Directory Structure & Organization

```
MERRA21C-ML/
├── configs/                  # Experiment configurations (YAML/Hydra)
│   ├── data/                 # Data loading and grid configs
│   ├── model/                # Architecture configs (U-Net, DiT, Flow Matching)
│   └── train/                # Training hyperparameters & SLURM configs
├── src/
│   ├── data/                 # Dataset loaders, preprocessors, regridding
│   │   ├── dataset.py        # PyTorch Dataset for paired LR/HR samples
│   │   ├── grid.py           # Coordinate & projection conversions (lat-lon to LCC)
│   │   └── transforms.py     # Normalization, log-transforms, patching
│   ├── models/               # Model definitions
│   │   ├── flow_matching.py  # CFM / OT-CFM trajectory and ODE solvers
│   │   ├── unet.py           # Conditional U-Net backbone
│   │   └── dit.py            # Diffusion Transformer backbone
│   ├── evaluation/           # Verification & meteorological metrics
│   │   ├── metrics.py        # RMSE, MAE, Bias, CRPS
│   │   ├── spectral.py       # Power spectral density (PSD) calculation
│   │   └── precip_eval.py    # Fractions Skill Score (FSS), threshold statistics
│   └── utils/                # I/O, logging, and distributed training utilities
├── scripts/
│   ├── preprocess.py         # Data preprocessing and index generation
│   ├── train.py              # Main training entrypoint
│   ├── evaluate.py           # Evaluation on test years/seasons
│   ├── infer_conus.py        # Full-domain CONUS downscaling inference
│   └── slurm/                # SLURM submission scripts for Discover
├── notebooks/                # Exploratory analysis & visualization
├── plan.md                   # This project execution roadmap
└── README.md                 # Project overview & quickstart
```

---

## 6. Next Immediate Steps

1. **Verify Discover Paths & Access:**
   - Confirm permissions and inspect the directory structure and file listings under:
     - `/discover/nobackup/projects/gmao/osse2/HWT/CONUS02KM/Feature-c2160_L181/holding/`
     - `/discover/nobackup/projects/gmao/geos_fp_arch/f5295_fp`
2. **Inspect Variable Names & Time Steps:**
   - Confirm the exact NetCDF variable names for temperature and precipitation (e.g., `T2M`, `PRECTOT`, `PRECCON`).
3. **Draft Base Environment & Setup:**
   - Create environment configuration (`environment.yml` / `requirements.txt` / PyTorch on CUDA).
4. **Build Core Data Ingestion & Dataset Class:**
   - Implement the spatial alignment and patch extraction logic for LCC and regular lat-lon grids.
