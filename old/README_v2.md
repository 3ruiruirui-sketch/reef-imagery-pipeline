# Reef Imagery Pipeline v2 — Usage Guide

## Overview
Production-grade local Python pipeline for Albufeira Reef imagery acquisition,
analysis, and QGIS-ready output. Includes a GEE JS bonus export script.

## Install
```bash
pip install -r requirements_v2.txt
```

## Recommended Run Order

### 1 — Probe DGT capabilities (confirm CoverageId)
```bash
python reef_imagery_pipeline_v2.py --step capabilities
```
Open `reef_output_v2/dgt_capabilities_2018.xml` and search for the correct
`<Identifier>` value for the 2018 25 cm orthophoto, then use it with `--coverage`.

### 2 — Run the full pipeline
```bash
python reef_imagery_pipeline_v2.py --step all --coverage <CoverageId_from_xml>
```

### 3 — Run individual steps
```bash
python reef_imagery_pipeline_v2.py --step sentinel --date 2024-10-15
python reef_imagery_pipeline_v2.py --step ratio
python reef_imagery_pipeline_v2.py --step qgis
```

## CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--step` | *(required)* | `capabilities`, `ortho`, `sentinel`, `ratio`, `qgis`, `gee`, `all` |
| `--date` | `2024-10-15` | Target Sentinel-2 acquisition date |
| `--lat` | `37.069071` | Reef centre latitude |
| `--lon` | `-8.210492` | Reef centre longitude |
| `--buffer-m` | `500` | Clip radius in metres |
| `--coverage` | `Ortofotomapa_2018` | DGT WCS CoverageId |
| `--output-dir` | `reef_output_v2` | Output folder |

## Outputs

| File | Description |
|------|-------------|
| `dgt_capabilities_2018.xml` | DGT WCS capabilities response |
| `dgt_ortho_2018_reef.tif` | Clipped 2018 orthophoto GeoTIFF |
| `dgt_ortho_error_response.xml` | Written if DGT returns a ServiceException |
| `S2_B02_YYYYMMDD.tif` | Sentinel-2 Blue band window (COG read) |
| `S2_B03_YYYYMMDD.tif` | Sentinel-2 Green band window (COG read) |
| `S2_meta_YYYYMMDD.json` | STAC item metadata + signed asset URLs |
| `ratio_B02_B03_YYYYMMDD.tif` | log(B02)/log(B03) GeoTIFF, EPSG:32629 |
| `ratio_analysis_YYYYMMDD.png` | Three-panel visualisation (B02, B03, ratio) |
| `reef_project_YYYYMMDD.qgs` | QGIS 3.x project file referencing local rasters |
| `ratio_style.qml` | RdYlBu colour ramp style (0.8–1.2) for QGIS |
| `gee_reef_export_YYYYMMDD.js` | GEE script — paste into code.earthengine.google.com |
| `pipeline.log` | Full timestamped log of every step |

## Troubleshooting

### DGT orthophoto returns XML instead of TIFF
The `dgt_ortho_error_response.xml` file will be written.  
Open it and check the `<ExceptionText>` — most likely the `--coverage` value is wrong.  
Re-run `--step capabilities`, find the correct `<Identifier>`, and retry.

### No Sentinel-2 scene found
Try a neighbouring OP20 date:
```
2021-10-31 | 2022-09-12 | 2024-01-03 | 2023-10-26
```

### Ratio step fails
The B02/B03 `.tif` files must exist in `--output-dir`. Run `--step sentinel` first.

### rasterio window read fails (CRS mismatch)
The pipeline reprojects the WGS-84 bounding box to the scene's native CRS automatically.
If you see a CRS error, ensure `rasterio` ≥ 1.3 and `pyproj` are installed.
