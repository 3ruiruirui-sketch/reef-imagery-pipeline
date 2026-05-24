# Bathymetric Reef Discovery Module

**File:** `scripts/reef_bathy_module.py`  
**Added to:** `scripts/reef_imagery_pipeline_v3.py` as `--step bathy`

Discovers underwater reef structures at the **Albufeira Reef**  
(lat=37.069071, lon=−8.210492, buffer ≈ 500 m, depth 0–50 m)  
by downloading free bathymetric data, computing morphological indices, and  
producing reef-candidate polygons as GeoJSON.

---

## Quick start

```bash
# Install dependencies (includes bathymetry requirements)
pip install -r requirements.txt
# Or install full package: pip install -e ".[dev]"

# Run bathy step only (uses all sources, default depth range)
python scripts/reef_imagery_pipeline_v3.py \
    --step bathy \
    --output-dir reef_Output_Master/reef_output_v3

# Full pipeline including bathy
python scripts/reef_imagery_pipeline_v3.py \
    --step all \
    --date 2024-10-15 \
    --bathy-source all \
    --depth-min -50 --depth-max -1

# Run bathy module standalone
python scripts/reef_bathy_module.py \
    --step bathy \
    --bathy-source emodnet \
    --depth-min -30 --depth-max -2 \
    --output-dir /tmp/bathy_test
```

---

## New CLI arguments (reef_imagery_pipeline_v3.py)

| Argument | Default | Description |
|---|---|---|
| `--bathy-source` | `all` | Data source: `emodnet`, `gebco`, `geomar`, `etopo`, `s2`, or `all` |
| `--depth-min` | `-50` | Minimum depth for reef candidate detection (m, negative below sea level) |
| `--depth-max` | `-1` | Maximum depth for reef candidate detection (m) |

---

## Data sources

### 1. EMODnet Bathymetry ★ (primary recommended)
- **URL:** https://ows.emodnet-bathymetry.eu/wcs
- **Access:** WCS 1.0.0 GET — no authentication
- **Resolution:** ~115 m (1/128°)
- **Coverage:** Full European seas including Algarve south coast
- **Product:** EMODnet DTM composite mean (`emodnet:mean`)
- **Notes:** Best-available free source for this area. Integrates survey data from IHM, SHOM, UK HO and others.

### 2. GEBCO 2024
- **URL:** https://download.gebco.net/
- **Access:** GET with bbox params — no authentication — returns ZIP + NetCDF4
- **Resolution:** ~460 m (15 arc-second)
- **Coverage:** Global
- **Notes:** Global reference product. Coarser than EMODnet but useful as independent validation. Module downloads a bbox-clipped sub-grid and converts to GeoTIFF via GDAL's NetCDF driver.

### 3. Portugal IHM / GEOMAR
- **URL:** https://geomar.hidrografico.pt/geoserver/geomar/wcs
- **Access:** WCS 1.0.0 GET — public GeoServer (coverage names may change)
- **Resolution:** ~25–100 m
- **Coverage:** Portuguese territorial waters
- **Notes:** The Instituto Hidrográfico de Marinha (IHM) manages GEOMAR / SEAMAP 2030 bathymetric survey data for Portugal. Module tries known coverage IDs and fails gracefully if unavailable. Check https://geomar.hidrografico.pt for current layer names.

### 4. NOAA ETOPO 2022
- **URL:** https://www.ncei.noaa.gov/thredds/wcs/etopo/etopo_60s_v2022.nc
- **Access:** THREDDS WCS 1.0.0 GET — no authentication
- **Resolution:** ~1.8 km (60 arc-second)
- **Coverage:** Global
- **Notes:** Coarse-resolution global relief model. Useful only for regional context; too coarse to resolve individual reef structures.

### 5. Sentinel-2 Shallow Depth Inversion ★ (free, no download)
- **Method:** Stumpf et al. (2003) log-ratio transform using existing B02 (blue) / B03 (green)
- **Formula:** `depth = m1 × ln(n×B02) / ln(n×B03) + m0`
- **Parameters:** m1=32.0, m0=−28.0, n=1500 (calibrated for Algarve clear water, Kd≈0.045 m⁻¹)
- **Resolution:** 10 m (same as Sentinel-2 optical bands)
- **Coverage:** Wherever S2 bands are already downloaded; optical depth limit ~20–25 m
- **Notes:** Best horizontal resolution. Requires calm, clear-water scenes (cloud < 5%). Run `--step sentinel` first. Depth values are relative; calibrate with ICESat-2 or echosounder data.

---

## Processing pipeline

```
Downloaded GeoTIFF
       │
       ▼
compute_bathy_indices()
   ├─ slope_deg        : local slope in degrees
   ├─ tri              : Terrain Ruggedness Index (Riley 1999)
   ├─ bpi_fine         : BPI 3×3 kernel (fine-scale relief)
   ├─ bpi_broad        : BPI 15×15 kernel (broad-scale position)
   └─ curvature        : Laplacian curvature
       │
       ▼
detect_reef_candidates()
   depth mask [depth_min, depth_max]
   + high rugosity (TRI > 70th percentile)
   + positive BPI (elevated vs surroundings, BPI_broad > 60th pct)
   + connected-component labelling (remove patches < 4 px)
   + rasterio.features.shapes → shapely polygons → GeoDataFrame
       │
       ▼
reef_candidates_YYYYMMDD.geojson
       │
       ▼
add_bathy_to_qgis()   → injects layers into existing reef_project_*.qgs
```

---

## Output files

All saved to `--output-dir` (same as main pipeline):

| File | Description |
|---|---|
| `bathy_emodnet_YYYYMMDD.tif` | EMODnet bathymetry GeoTIFF |
| `bathy_gebco_YYYYMMDD.tif` | GEBCO 2024 GeoTIFF |
| `bathy_geomar_YYYYMMDD.tif` | IHM GEOMAR GeoTIFF (if available) |
| `bathy_etopo_YYYYMMDD.tif` | NOAA ETOPO GeoTIFF |
| `bathy_s2_stumpf_YYYYMMDD.tif` | S2 depth inversion GeoTIFF |
| `bathy_emodnet_*_slope_deg.tif` | Slope raster |
| `bathy_emodnet_*_tri.tif` | TRI roughness raster |
| `bathy_emodnet_*_bpi_fine.tif` | BPI fine-scale raster |
| `bathy_emodnet_*_bpi_broad.tif` | BPI broad-scale raster |
| `bathy_emodnet_*_curvature.tif` | Laplacian curvature raster |
| `reef_candidates_YYYYMMDD.geojson` | Reef candidate polygons |
| `bathy.log` | Bathy-step log |

---

## Calibrating the S2 depth inversion

The Stumpf parameters (m0, m1) are empirical. To calibrate:

1. Collect depth ground truth (ICESat-2 ATL03, echosounder, charts).
2. Sample S2 B02/B03 ratio at known depth points.
3. Fit a linear regression: `depth ~ m1 * log_ratio + m0`.
4. Pass calibrated values via `compute_s2_depth_inversion(..., m0=..., m1=...)`.

A calibration script is available in `scripts/sprint1_algarve_bathymetry.py`.

---

## References

- Stumpf, R.P. et al. (2003). Determination of water depth with high-resolution satellite imagery over variable bottom types. *Limnology and Oceanography*, 48(1part2), 547–556.
- Riley, S.J. et al. (1999). A terrain ruggedness index that quantifies topographic heterogeneity. *Intermountain Journal of Sciences*, 5(1–4), 23–27.
- Lundblad, E. et al. (2006). A benthic terrain classification scheme for American Samoa. *Marine Geodesy*, 29(2), 89–111.
- EMODnet Bathymetry Consortium (2022). EMODnet Digital Bathymetry (DTM 2022). https://doi.org/10.12770/ff3aff8a-cff1-44a3-a2c8-1910bf109f85
- GEBCO Compilation Group (2024). GEBCO 2024 Grid. https://doi.org/10.5285/1c44ce99-0a0d-5f4f-e063-7086abc0ea0f
