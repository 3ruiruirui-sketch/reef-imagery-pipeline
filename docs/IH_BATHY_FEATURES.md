# IH/DGRM Bathymetry Feature Engineering — Implementation Notes

## What was added

A new module `src/ih_bathy_features.py` integrates the official **DGRM/IH**
ArcGIS REST bathymetric contour service into the reef imagery pipeline as
reusable, bathymetry-derived features for both historical model training and
operational daily prediction.

## Files changed

| File | Action | Why |
|------|--------|-----|
| `src/ih_bathy_features.py` | **Created** | Chunked ArcGIS downloader + persistent GeoPackage cache + EPSG:3763 metric feature engineering |
| `src/reef_ml_predictor_acolite.py` | **Modified** | Added `with_bathy_features` flag and 11 new columns to `summary.csv` |
| `tests/test_ih_bathy_features.py` | **Created** | 13 unit + integration tests |
| `docs/IH_BATHY_FEATURES.md` | **Created** | This document |

## How the ArcGIS layer is queried

**Service endpoint**
```
https://webgis.dgrm.mm.gov.pt/arcgis/rest/services/
    Dados_entidades_externas/Batimetrica_IH/MapServer/0
```

**Layer metadata**
- Title: "Isobatimetricas, Escala 1:150.000 (Fonte: IH)"
- Geometry: polyline
- Published CRS: EPSG:4326
- Source CRS: EPSG:3763
- Key attribute: `Depth` (metres)
- maxRecordCount: 1000

**Chunking strategy**
Large AOIs are automatically tiled into `0.10°` (~11 km) bbox chunks so each
request stays well under the 1000-record limit.  Results from all tiles are
merged and deduplicated by `(OBJECTID, Depth, first coordinate)` before
returning.

**Retry logic**
Transient failures trigger 3 retries with exponential back-off (2, 4, 6 s).

**Caching**
Downloaded isobaths are saved to a local GeoPackage (or JSON fallback) keyed
by a SHA-256 hash of the bbox + depth list.  Subsequent calls with the same
parameters hit the cache instantly.

## Bathymetry features generated

For any `(lon, lat)` point:

| Feature | Type | Description |
|---------|------|-------------|
| `nearest_isobath_distance_m` | float | Distance to closest IH contour (EPSG:3763) |
| `nearest_isobath_depth_m` | float | Depth label of that closest contour |
| `dist_to_isobath_10m` | float | Distance to 10 m isobath (or ∞) |
| `dist_to_isobath_20m` | float | Distance to 20 m isobath (or ∞) |
| `dist_to_isobath_30m` | float | Distance to 30 m isobath (or ∞) |
| `dist_to_isobath_50m` | float | Distance to 50 m isobath (or ∞) |
| `dist_to_isobath_100m` | float | Distance to 100 m isobath (or ∞) |
| `bathymetry_zone_class` | str | `very_shallow` / `shallow_reef` / `nearshore_mid` / `mid_depth` / `offshore` |
| `bathymetry_slope_proxy` | float | Std-dev of nearby contour depths (proxy for local gradient) |
| `contour_density_proxy` | float | Total contour length (m) / AOI area (km²) |
| `n_isobaths_in_aoi` | int | Number of unique polylines in the AOI |

All metric distances are computed in **EPSG:3763** (PT-TM06 / ETRS89-TM06)
for accurate metre-scale measurements.  If `pyproj` is unavailable a haversine
fallback is used.

## Integration with existing pipeline

### Standalone use

```python
from src.ih_bathy_features import BathyFeatureEngine

engine = BathyFeatureEngine(cache_dir="data/cache")
feats = engine.compute_features_for_point(lon=-8.21, lat=37.07)
```

### Predictor integration

```bash
python src/reef_ml_predictor_acolite.py \
    --boa-b02  S2_B02.tif \
    --date     2025-09-25 \
    --output   out/ \
    --with-bathy-features
```

When `--with-bathy-features` is passed and `lat/lon` are provided, the 11
new columns are appended to `summary.csv` automatically.

## Assumptions

1. The DGRM/IH ArcGIS REST service is publicly accessible (no auth required).
2. Algarve AOI is within the Portuguese coast coverage (Caminha → Guadiana).
3. `pyproj` is available for accurate EPSG:3763 reprojection (fallback to
   haversine if missing).
4. `geopandas` is available for GeoPackage cache I/O (fallback to JSON if
   missing).

## Limitations

- **Resolution**: IH contours are at 1:150,000 scale.  Small reef structures
  (<100 m) may not be represented.
- **Temporal**: Chart data is static (not updated daily).  Features reflect
  long-term bathymetry, not seasonal sand migration.
- **Coverage gaps**: Very nearshore (<5 m) may lack contours; very deep
  (>100 m) may be sparse.
- **Service availability**: If DGRM service is down, cached data is still
  usable, but stale contours will be used.

## How to run the new logic

### Quick demo (standalone)

```bash
cd /Users/ssoares/Downloads/PI-PROJE/reef_imagery_pipeline
python src/ih_bathy_features.py \
    --lon -8.210492 --lat 37.069071 \
    --buffer-m 5000
```

### Tests

```bash
python tests/test_ih_bathy_features.py
```

### With the full predictor

```bash
python src/reef_ml_predictor_acolite.py \
    --boa-b02  path/to/S2_B02.tif \
    --date     2025-09-25 \
    --output   out/ \
    --with-bathy-features
```

## Next recommended steps

1. **Collect ground-truth ROV / diver depth measurements** to validate the
   `dist_to_isobath_XXm` features against real-world positions.
2. **Train an ML model** using the new bathymetry features alongside
   Sentinel-2 SNR, Kd, and CMEMS/ERA5 variables.
3. **Add spatial cross-validation** — ensure the model generalises across
   different depth zones.
4. **Cache warming**: run a one-off crawl of the entire Algarve AOI to
   populate the cache and eliminate future network latency.
