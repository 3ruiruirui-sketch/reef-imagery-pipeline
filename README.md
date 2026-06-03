# Reef Imagery Pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Data: Copernicus](https://img.shields.io/badge/data-Copernicus%20Sentinel--2-green.svg)](https://dataspace.copernicus.eu/)

---

## Overview

**Reef Imagery Pipeline** is a Python 3.10+ system that transforms free, open satellite and aerial imagery into shallow-water bathymetry and benthic visibility estimates for coastal reefs in the Algarve, Portugal. It combines Copernicus Sentinel-2 data, high-resolution national orthophotos (OrtoSat2023, DGT), hydrographic charts from Instituto Hidrográfico (IH), and physical-optical models to deliver actionable, site-scale information for divers, researchers, and coastal managers.

---

## Main Goal

The primary objective of this project is to provide an end-to-end, open-source pipeline for:

- **Predicting underwater visibility** at scuba diving and snorkelling sites along the Algarve coast
- **Mapping shallow reef structure and depth** using satellite-derived bathymetry (SDB)
- **Supporting reef monitoring and conservation** by integrating EO data with official hydrographic reference data
- **Bridging the gap** between sparse in situ observations and continuous, satellite-based coastal monitoring

The workflow is configurable so that other coastal regions can adapt the same approach by adjusting data sources, calibration areas, and processing parameters.

---

## Scientific Motivation

Shallow coastal reefs around the Algarve are heavily used for recreation and tourism, yet in situ visibility observations are sparse in both space and time. By combining physics-based optical inversion, machine learning, and open Earth observation data, this project aims to generate continuous, spatially explicit visibility and bathymetry products that support dive-condition assessment, reef habitat mapping, and long-term environmental monitoring.

---

## Scope of This Repository

This repository contains **version 3.1** of the Reef Imagery Pipeline:

| Directory / File | Description |
|---|---|
| `src/` | Core physics + ML package (QAA inversion, Stumpf SDB, IH integration) |
| `scripts/` | Acquisition and analysis entry points (Sentinel-2, OrtoSat2023, ICESat-2) |
| `dashboard/` | Flask-based web visualization and QA dashboard |
| `tests/` | Unit tests and regression checks |
| `archive/` / `old/` | Legacy v1/v2 modules and notebooks (kept for reproducibility) |

Additional documentation:
- `README_v3.md` — data acquisition and downloader details (v3)
- `README_bathy.md` — bathymetry calibration and validation notes
- `SENTINEL_ANALYSIS_SUMMARY.md` — detailed spectral analysis

---

## System Architecture

```
reef_imagery_pipeline/
├── src/                        # Core package (physics + ML)
│   ├── reef_ml_predictor_acolite.py  # Main QAA + SDB model
│   ├── reef_ml_predictor.py          # STAC image ranking
│   ├── bathy_calibrator.py           # IH Isobath integration
│   ├── enhancer.py                   # Preprocessing + SNR
│   ├── utils.py                      # Raster I/O, Beer-Lambert
│   └── orchestrator_run.py           # Main orchestrator
├── scripts/                    # Entry points and analysis
│   ├── reef_imagery_pipeline_v3.py   # Sentinel-2/DGT acquisition
│   ├── cdse_downloader_minimal.py    # CDSE download
│   └── demo_bathy_live.py            # Live demo
├── dashboard/                  # Flask web visualization
├── tests/                      # Unit tests
└── archive/                    # Legacy v1/v2 modules
```

---

## Quick Start

### 1. Installation

```bash
git clone https://github.com/3ruiruirui-sketch/reef-imagery-pipeline.git
cd reef-imagery-pipeline
pip install -r requirements_v3.txt
```

### 2. Basic Usage

**Acquisition:**
```bash
python scripts/reef_imagery_pipeline_v3.py \
  --step all \
  --lat 37.069071 --lon -8.210492 \
  --date 2024-10-15 \
  --output-dir reef_output_demo
```

**Processing:**
```bash
python src/orchestrator_run.py --depth 16.0
```

---

## Core Modules

- **`src/reef_ml_predictor_acolite.py`**: Implements QAA (Quasi-Analytical Algorithm) for Kd inversion, Stumpf SDB for bathymetry (log-ratio), and IH integration.
- **`src/bathy_calibrator.py`**: Integrates with Instituto Hidrográfico ArcGIS REST services to calibrate model coefficients and validate SDB results against official isobaths.
- **`src/enhancer.py`**: Handles image preprocessing, including VSI-based I/O, NLM denoising, and CLAHE contrast enhancement.

---

## Physical Methodology

- **Stumpf SDB Model:** `Z = m0 - m1 * ln(B02/B03) / ln(n)` (default m0=-16, m1=20)
- **QAA Inversion:** `Kd(λ) = a(λ) + bb(λ)` for estimating water column attenuation coefficients
- **IH Calibration:** Adjusts m0 and m1 parameters against official ground-truth isobaths (10m, 20m, 30m)

---

## Expected Results

| Output | Description | Format |
|---|---|---|
| `depth_map_*.tif` | Satellite-derived bathymetry | GeoTIFF |
| `kd_*.tif` | Kd attenuation coefficient maps | GeoTIFF |
| `confidence_map_*.tif` | Per-pixel model confidence | GeoTIFF |
| `visibility_score.json` | Per-site benthic visibility metrics for dive-condition assessment | JSON |
| `*.qgs` | QGIS project for visualization and QA | QGIS |

Expected accuracy: **RMSE vs IH isobaths < 2 m**.

---

## Data Sources & Acknowledgements

This project relies entirely on free and open Earth observation and hydrographic data. We gratefully acknowledge the following data providers:

- **Copernicus Sentinel-2** — satellite imagery accessed via the [Copernicus Data Space Ecosystem (CDSE)](https://dataspace.copernicus.eu/) under the European Union's Copernicus programme. Contains modified Copernicus Sentinel data.
- **OrtoSat2023** — high-resolution coastal orthophotos provided by [Direção-Geral do Território (DGT)](https://www.dgterritorio.gov.pt/), Portugal.
- **Bathymetric and isobath data** — from [Instituto Hidrográfico (IH)](https://www.hidrografico.pt/), Portugal, accessed via official ArcGIS REST services and used for model calibration and validation.
- **ICESat-2 ATL products** — from [NASA NSIDC](https://nsidc.org/data/icesat-2), used for independent depth validation.
- **QGIS** — open-source GIS platform used for visualization and manual QA/QC of pipeline outputs.

If you use results or derived products from this repository in scientific work, please cite the original data providers alongside this codebase.

---

## Other Documentation

- `README_v2.md` — Version 2 documentation (legacy, in `archive/`)
- `README_v3.md` — v3 downloader docs (now in `scripts/`)
- `SENTINEL_ANALYSIS_SUMMARY.md` — Detailed spectral analysis

---

## Contributing

See `CONTRIBUTING.md` for development guidelines.

---

## License

MIT License — see `LICENSE` file for details.

---

## Support

For questions or issues, open a GitHub ticket or contact the author via email.

---

*Last updated: June 2026 — Current version: v3.1 (restructured)*
