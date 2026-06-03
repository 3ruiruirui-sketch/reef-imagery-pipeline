# Reef Imagery Pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Data: Copernicus](https://img.shields.io/badge/data-Copernicus%20Sentinel--2-green.svg)](https://dataspace.copernicus.eu/)
[![IH Portugal](https://img.shields.io/badge/bathy-Instituto%20Hidrogr%C3%A1fico-blue.svg)](https://www.hidrografico.pt/)
[![DGT Portugal](https://img.shields.io/badge/ortho-DGT%20OrtoSat2023-orange.svg)](https://www.dgterritorio.gov.pt/)
[![ICESat-2](https://img.shields.io/badge/validation-ICESat--2%20NASA-lightgrey.svg)](https://icesat-2.gsfc.nasa.gov/)

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
| `data/` | Input CSVs, reference images, local cloud comparisons |
| `docs/` | Scientific figures and documentation assets |
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
│   ├── generate_reef_map.py          # Reef map generation
│   ├── phase_b_calibrate_icesat2.py  # ICESat-2 calibration
│   ├── validate_corrected_coords.py  # Coordinate validation
│   ├── watchdog.py                   # Pipeline watchdog
│   └── demo_bathy_live.py            # Live demo
├── data/                       # Reference data and CSVs
│   ├── local_cloud/                  # Local/cloud comparison logs
│   └── best_clear_water_images.csv   # Best imagery index
├── docs/                       # Figures and documentation
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

<table>
  <tr>
    <td align="center" width="160">
      <a href="https://dataspace.copernicus.eu/" target="_blank">
        <img src="https://www.copernicus.eu/sites/default/files/2019-08/Copernicus_logo.png" height="48" alt="Copernicus" />
      </a>
    </td>
    <td>
      <strong>Copernicus Sentinel-2</strong><br/>
      Satellite imagery accessed via the <a href="https://dataspace.copernicus.eu/">Copernicus Data Space Ecosystem (CDSE)</a>, under the European Union’s Copernicus programme.<br/>
      <em>Contains modified Copernicus Sentinel data.</em><br/>
      🔗 <a href="https://dataspace.copernicus.eu/">dataspace.copernicus.eu</a>
    </td>
  </tr>
  <tr>
    <td align="center" width="160">
      <a href="https://www.dgterritorio.gov.pt/" target="_blank">
        <img src="https://www.dgterritorio.gov.pt/sites/default/files/dgt_logo.png" height="48" alt="DGT" />
      </a>
    </td>
    <td>
      <strong>OrtoSat2023 — Direção-Geral do Território (DGT)</strong><br/>
      High-resolution coastal orthophotos for mainland Portugal.<br/>
      🔗 <a href="https://www.dgterritorio.gov.pt/">dgterritorio.gov.pt</a>
    </td>
  </tr>
  <tr>
    <td align="center" width="160">
      <a href="https://www.hidrografico.pt/" target="_blank">
        <img src="https://www.hidrografico.pt/resources/images/logo_ih.png" height="48" alt="Instituto Hidrográfico" />
      </a>
    </td>
    <td>
      <strong>Instituto Hidrográfico (IH)</strong><br/>
      Bathymetric charts and isobath data for Portuguese coastal waters, accessed via official ArcGIS REST services. Used for model calibration and validation.<br/>
      🔗 <a href="https://www.hidrografico.pt/">hidrografico.pt</a>
    </td>
  </tr>
  <tr>
    <td align="center" width="160">
      <a href="https://icesat-2.gsfc.nasa.gov/" target="_blank">
        <img src="https://icesat-2.gsfc.nasa.gov/sites/default/files/logo_0.png" height="48" alt="ICESat-2" />
      </a>
    </td>
    <td>
      <strong>ICESat-2 — NASA / NSIDC</strong><br/>
      ATL photon-counting lidar products used for independent shallow-water depth validation.<br/>
      🔗 <a href="https://icesat-2.gsfc.nasa.gov/">icesat-2.gsfc.nasa.gov</a> &nbsp;|&nbsp; <a href="https://nsidc.org/data/icesat-2">nsidc.org/data/icesat-2</a>
    </td>
  </tr>
  <tr>
    <td align="center" width="160">
      <a href="https://qgis.org/" target="_blank">
        <img src="https://qgis.org/styleguide/images/styleGuide/qgis-logo.svg" height="48" alt="QGIS" />
      </a>
    </td>
    <td>
      <strong>QGIS</strong><br/>
      Open-source GIS platform used for visualization and manual QA/QC of pipeline outputs.<br/>
      🔗 <a href="https://qgis.org/">qgis.org</a>
    </td>
  </tr>
</table>

> If you use results or derived products from this repository in scientific work, please cite the original data providers alongside this codebase.

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
