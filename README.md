# Reef Imagery Pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Satellite imagery processing system for reef analysis and coastal bathymetry in the Algarve, Portugal. Combines Sentinel-2 data, high-resolution orthophotos (OrtoSat2023, DGT), and advanced physical-optical models for depth estimation and benthic visibility.

---

## 🏗️ System Architecture

```
reef_imagery_pipeline/
├── src/                    # Core package (physics + ML)
│   ├── reef_ml_predictor_acolite.py    # Main QAA + SDB model
│   ├── reef_ml_predictor.py            # STAC image ranking
│   ├── bathy_calibrator.py             # IH Isobath integration
│   ├── enhancer.py                     # Preprocessing + SNR
│   ├── utils.py                        # Raster I/O, Beer-Lambert
│   └── orchestrator_run.py             # Main orchestrator
│
├── scripts/                # Entry points and analysis
│   ├── reef_imagery_pipeline_v3.py     # Sentinel-2/DGT acquisition
│   ├── cdse_downloader_minimal.py      # CDSE download
│   ├── demo_bathy_live.py              # IH + SDB demo
│   ├── icesat2_algarve_bathy.py        # ICESat-2 validation
│   ├── sprint1_algarve_bathymetry.py   # Central Algarve bathymetry
│   ├── pedra_do_alto_best_images.py    # Automatic image selection
│   ├── save_refined_image*.py          # Temporal analyses
│   └── ...
│
├── tests/                  # Unit tests
├── archive/                # Legacy modules (v1, v2)
└── dashboard/              # Web visualization (Flask)
```

---

## 🔄 Data Pipeline

```mermaid
flowchart TB
    %% ============================================
    %% STYLING DEFINITIONS
    %% ============================================
    classDef input fill:#e1f5fe,stroke:#01579b,stroke-width:2px,color:#000
    classDef process fill:#fff3e0,stroke:#e65100,stroke-width:2px,color:#000
    classDef algorithm fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px,color:#000
    classDef calibration fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#000
    classDef output fill:#fce4ec,stroke:#c2185b,stroke-width:2px,color:#000
    classDef validation fill:#fff8e1,stroke:#ff8f00,stroke-width:2px,stroke-dasharray: 5 5,color:#000

    %% ============================================
    %% DATA INPUT LAYER
    %% ============================================
    subgraph INPUT_LAYER["📡 DATA ACQUISITION LAYER"]
        direction TB
        S2["🛰️ Sentinel-2 MSI<br/>10m resolution<br/>B02(490nm) | B03(560nm) | B04(665nm)"]
        ORTO["📷 High-Res Orthophotos<br/>OrtoSat2023 30cm<br/>DGT 2018/2021 25cm"]
        AUX["📊 Auxiliary Data<br/>Solar geometry | Cloud cover<br/>Wind | Tides"]
    end

    %% ============================================
    %% GROUND TRUTH LAYER
    %% ============================================
    subgraph GT_LAYER["🎯 GROUND TRUTH & VALIDATION"]
        direction TB
        IH["⚓ Instituto Hidrográfico<br/>Nautical Charts<br/>Isobaths 10/20/30m"]
        ICE["🛰️ ICESat-2 ATL03/ATL08<br/>Photon bathymetry<br/>Validation data"]
    end

    %% ============================================
    %% PREPROCESSING LAYER
    %% ============================================
    subgraph PREPROC_LAYER["🔧 PREPROCESSING"]
        direction LR
        VSI["VSI Stream<br/>COG Window Read"]
        DN["DN → BOA<br/>Reflectance scaling<br/>1/10000"]
        ACOLITE["ACOLITE<br/>Atmospheric correction<br/>Rayleigh + aerosol"]
    end

    %% ============================================
    %% PHYSICAL INVERSION LAYER
    %% ============================================
    subgraph PHYS_LAYER["⚙️ PHYSICAL INVERSION ENGINE"]
        direction TB
        
        subgraph QAA_BLOCK["Quasi-Analytical Algorithm (QAA)"]
            direction LR
            RRS["Rrs(λ)<br/>Remote sensing reflectance"]
            ABB["a(λ) + bb(λ)<br/>IOP inversion"]
            KD_QAA["Kd(λ) = a + bb<br/>Attenuation coefficient"]
        end
        
        subgraph SDB_BLOCK["Stumpf SDB Algorithm"]
            direction LR
            RATIO["ln(B02/B03)<br/>Blue/Green ratio"]
            DEPTH_CALC["Z = m₀ - m₁×ln(ratio)/ln(n)<br/>Depth estimation"]
        end
        
        subgraph QUALITY_BLOCK["Quality Assessment"]
            direction LR
            SNR["SNR Map<br/>μ/σ window analysis"]
            UNC["Uncertainty<br/>Kd saturation check"]
        end
    end

    %% ============================================
    %% CALIBRATION LAYER
    %% ============================================
    subgraph CAL_LAYER["🎚️ CALIBRATION & REFINEMENT"]
        direction TB
        IH_QUERY["ArcGIS REST Query<br/>Bounding box isobath fetch"]
        ISO_MATCH["Pixel-Isobath Matching<br/>Buffer sampling"]
        OPT_M0M1["Optimize m₀, m₁<br/>Minimize RMSE vs IH"]
        VALIDATE["Cross-validation<br/>Bias | RMSE | R²"]
    end

    %% ============================================
    %% OUTPUT LAYER
    %% ============================================
    subgraph OUTPUT_LAYER["📊 OUTPUT PRODUCTS"]
        direction TB
        
        subgraph RASTER_OUT["GeoTIFF Rasters"]
            SDB_R["sdb_depth_map.tif<br/>Float32 | EPSG:32629<br/>[0-40m]"]
            KD_R["kd_b02.tif | kd_b03.tif<br/>Attenuation maps"]
            CONF_R["confidence_map.tif<br/>SNR-based quality"]
        end
        
        subgraph VECTOR_OUT["Vector & Report"]
            META["metadata.json<br/>Processing lineage"]
            SCORE["visibility_score.json<br/>Quality metrics"]
        end
        
        subgraph VIS_OUT["Visualization"]
            QGIS_PROJ["reef_project_YYYYMMDD.qgs<br/>Styled QGIS project"]
            QML_STYLE["ratio_style.qml<br/>RdYlBu ramp 0.8-1.2"]
            PLOTS["analysis_plots.png<br/>Diagnostic figures"]
        end
    end

    %% ============================================
    %% FLOW CONNECTIONS
    %% ============================================
    S2 --> VSI
    ORTO --> VSI
    AUX --> ACOLITE
    
    VSI --> DN
    DN --> ACOLITE
    ACOLITE --> RRS
    
    RRS --> ABB --> KD_QAA
    RRS --> RATIO --> DEPTH_CALC
    ACOLITE --> SNR --> UNC
    
    KD_QAA --> SDB_R
    KD_QAA --> KD_R
    SNR --> CONF_R
    
    IH --> IH_QUERY --> ISO_MATCH --> OPT_M0M1 --> VALIDATE
    VALIDATE --> DEPTH_CALC
    OPT_M0M1 --> RATIO
    
    DEPTH_CALC --> SDB_R
    UNC --> SCORE
    KD_QAA --> SCORE
    
    SDB_R --> QGIS_PROJ
    KD_R --> QGIS_PROJ
    CONF_R --> QGIS_PROJ
    
    META --> QGIS_PROJ
    SCORE --> PLOTS
    
    ICE -.->|Independent validation| VALIDATE
    
    %% ============================================
    %% STYLE ASSIGNMENTS
    %% ============================================
    class S2,ORTO,AUX input
    class VSI,DN,ACOLITE,RATIO,RRS,ABB process
    class KD_QAA,DEPTH_CALC,SNR,UNC algorithm
    class IH,IH_QUERY,ISO_MATCH,OPT_M0M1,VALIDATE calibration
    class SDB_R,KD_R,CONF_R,META,SCORE,QGIS_PROJ,QML_STYLE,PLOTS output
    class ICE validation
```

---

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/3ruiruirui-sketch/reef-imagery-pipeline.git
cd reef-imagery-pipeline
pip install -r requirements_v3.txt
```

### 2. Basic Usage

#### Sentinel-2 + OrtoSat2023 Acquisition
```bash
python scripts/reef_imagery_pipeline_v3.py --step all \
    --lat 37.069071 --lon -8.210492 \
    --date 2024-10-15 \
    --output-dir reef_output_demo
```

#### Physical-Optical Processing (SDB + Kd)
```bash
python src/orchestrator_run.py --depth 16.0
```

#### IH Calibration Demo
```bash
python scripts/demo_bathy_live.py
```

---

## 📚 Core Modules

### `src/reef_ml_predictor_acolite.py`
Main physical inversion model based on:
- **QAA (Quasi-Analytical Algorithm)**: Kd inversion from Rrs
- **Stumpf SDB**: Bathymetry via B02/B03 log-ratio
- **IH Integration**: Calibration with official isobaths

**Key functions:**
- `run_predictor()` — Complete analysis pipeline
- `stumpf_sdb()` — Depth map generation
- `gordon_kd_inversion()` — Attenuation coefficient estimation
- `make_snr_map()` — Signal-to-noise ratio analysis

### `src/bathy_calibrator.py`
Integration with ArcGIS REST service from Instituto Hidrográfico:
- `fetch_isobaths_for_bbox()` — Query isobaths in area
- `calibrate_stumpf_from_isobaths()` — Derive m0/m1 coefficients
- `validate_sdb_vs_chart()` — Validate against IH data

### `src/enhancer.py`
Image preprocessing:
- `fetch_vsi_patch()` — Read via VSI (Virtual File System)
- NLM denoising + CLAHE
- SNR estimation

### `src/reef_ml_predictor.py` (Legacy)
Heuristic STAC image ranking based on:
- Cloud coverage
- Solar elevation
- Seasonal Kd490 coefficient

---

## 📊 Workflows

### Workflow 1: Complete Acquisition and Processing

```bash
# 1. Sentinel-2 acquisition
python scripts/reef_imagery_pipeline_v3.py \
    --step sentinel \
    --date 2024-09-30 \
    --output-dir reef_output_sep_2024

# 2. Physical processing
python -c "
from src.reef_ml_predictor_acolite import run_predictor
from src.utils import compute_metadata_stub

run_predictor(
    boa_b02_path='reef_output_sep_2024/S2_B02_20240930.tif',
    metadata=compute_metadata_stub('2024-09-30'),
    output_dir='reef_output_sep_2024/predictor',
    date='2024-09-30',
    b03_path='reef_output_sep_2024/S2_B03_20240930.tif',
    lat=37.069071, lon=-8.210492
)
"
```

### Workflow 2: Multi-Year Temporal Analysis

```bash
# Comparative analysis between 2022 and 2024
python scripts/save_refined_image_2022_09_26.py
python scripts/save_refined_image_2024_09_30.py

# Or use sprint1 for complete bathymetry
python scripts/sprint1_algarve_bathymetry.py
```

### Workflow 3: ICESat-2 Validation

```bash
# Search for ICESat-2 data in area
python scripts/icesat2_algarve_search.py

# Process and compare
python scripts/icesat2_algarve_bathy.py
```

---

## 📈 Expected Results

### Main Outputs

| File | Description | Format |
|------|-------------|--------|
| `S2_B02_YYYYMMDD.tif` | Sentinel-2 Blue band (10m) | GeoTIFF |
| `S2_B03_YYYYMMDD.tif` | Sentinel-2 Green band (10m) | GeoTIFF |
| `sdb_depth_map.tif` | SDB depth map | Float32 GeoTIFF |
| `kd_b02.tif` / `kd_b03.tif` | Diffuse attenuation coefficient | Float32 GeoTIFF |
| `confidence_map.tif` | Confidence map (SNR) | Float32 GeoTIFF |
| `visibility_score.json` | Benthic visibility metrics | JSON |
| `reef_project_YYYYMMDD.qgs` | Configured QGIS project | QGIS 3.x |

### Quality Metrics

| Metric | Expected Value | Description |
|--------|----------------|-------------|
| SDB resolution | 10m | Native Sentinel-2 |
| Maximum depth | ~30m | B02/B03 optical limit |
| RMSE vs IH | < 2m | After calibration |
| SNR threshold | > 3.0 | Acceptable quality |

---

## 🔬 Physical Methodology

### Stumpf SDB Model
```
Z = m0 - m1 * ln(B02/B03) / ln(n)
```
Where:
- `m0, m1`: calibrated coefficients (default: -16, 20)
- `n`: logarithmic scaling factor (default: 1000)
- `B02, B03`: BOA (Bottom-of-Atmosphere) reflectance

### QAA Inversion (Gordon et al.)
Kd estimation from surface reflectance:
```
Kd(λ) = a(λ) + bb(λ)
```
Where `a` is absorption and `bb` backscattering.

### IH Calibration
Adjustment of `m0, m1` via official isobath ground-truth (10m, 20m, 30m).

---

## 🛠️ Development

### Run Tests
```bash
python tests/test_bathy_calibrator.py
python tests/test_fft_cleanliness.py
python tests/test_stac.py
```

### Import Structure
```python
# From src/ (core package)
from src.reef_ml_predictor_acolite import run_predictor, stumpf_sdb
from src.bathy_calibrator import calibrate_stumpf_from_isobaths
from src.enhancer import fetch_vsi_patch
from src.utils import read_band, write_band

# From scripts/ (do not import between scripts)
# Run directly: python scripts/xxx.py
```

---

## 📖 Historical Documentation

- `README_v2.md` — Version 2 documentation (legacy, in `archive/`)
- `README_v3.md` — v3 downloader docs (now in `scripts/`)
- `SENTINEL_ANALYSIS_SUMMARY.md` — Detailed spectral analysis

---

## 🤝 Contributing

See `CONTRIBUTING.md` for development guidelines.

---

## 📄 License

MIT License — see LICENSE file for details.

---

## 🙋 Support

For questions or issues, open a GitHub ticket or contact the author via email.

---

**Last updated:** May 2026  
**Current version:** v3.1 (restructured)
