# DGT Coastal Topography + Sentinel-2 Integration

Integration of Portuguese DGT (Direção Geral do Território) MDT-50cm LiDAR data with Sentinel-2 multispectral imagery for reef imagery pipeline analysis.

## Overview

This integration provides two main modules:

### 1. **CoastalTopographyAnalyzer** (`src/coastal_topography.py`)

Extracts terrain features from DGT MDT-50cm (0.5m resolution LiDAR DTM) around dive sites:

- **Slope** (degrees): terrain steepness, correlates with sediment resuspension patterns
- **Aspect** (degrees): terrain orientation (0°=N, 90°=E, 180°=S, 270°=W), indicates exposure to dominant waves/wind
- **Other stats**: mean, median, std, percentiles (90th) computed in buffers around each site

**Use case**: Static terrain features for reef visibility models — e.g., higher slope + westward aspect = more wave exposure = higher plume frequency.

### 2. **DGTSentinelIntegrator** (`src/dgt_sentinel_integrator.py`)

Downloads and aligns:

- MDT-50cm tiles from DGT STAC endpoint
- Sentinel-2 L2A bands from Copernicus (requires credentials)
- Outputs mosaic GeoTIFFs ready for multi-modal analysis

**Use case**: Combined terrain + multispectral features for advanced reef bathymetry and water quality models.

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or install with optional extras for development
pip install -e ".[dev]"
```

Required packages:
- `rasterio`, `geopandas`, `rasterstats`, `rioxarray` — geospatial data handling
- `requests` — HTTP API calls
- `numpy`, `pandas` — data processing
- `sentinelhub` (optional) — Sentinel-2 download (requires account)

### Extract Coastal Features

```python
from src.coastal_topography import CoastalTopographyAnalyzer

# Define dive sites (lat, lon)
sites = [
    ("pedra_sta_eulalia", 37.069081, -8.210242),
    ("albufeira_reef", 37.0690, -8.2105),
]

# Initialize analyzer for Algarve region
bbox = (-8.25, 37.04, -8.17, 37.10)  # Algarve coast
analyzer = CoastalTopographyAnalyzer(bbox=bbox, output_dir="./outputs/coastal")

# Extract features with 1 km buffer around each site
features = analyzer.extract_features_for_sites(sites, buffer_m=1000)

# Save to CSV / GeoJSON
analyzer.save_features(features, output_name="my_sites_features")
```

**Output files**:
- `coastal/algarve_coastal_features.csv` — tabular feature data
- `coastal/algarve_coastal_features.geojson` — georeferenced points
- `coastal/dem_mosaic_50cm.tif` — merged DEM
- `coastal/slope_50cm.tif`, `coastal/aspect_50cm.tif` — derived rasters

### Integrate with Sentinel-2

```python
from src.dgt_sentinel_integrator import DGTSentinelIntegrator

bbox = (-8.25, 37.04, -8.17, 37.10)
integrator = DGTSentinelIntegrator(bbox=bbox, output_dir="./outputs/integrated")

# Download MDT-50cm + optionally Sentinel-2 (requires credentials)
result = integrator.integrate(
    fetch_sentinel=False,  # Set True if you have sentinelhub credentials
    date_start="2024-07-01",
    date_end="2024-08-31",
    cloud_pct=30
)

print(result)  # Contains paths to MDT mosaic, Sentinel data, etc.
```

### Full Pipeline (All Survey Sites)

```bash
python scripts/integrate_dgt_sentinel.py \
    --output-dir ./outputs/algarve_integration \
    --buffer-m 1000 \
    --date-start 2024-07-01 \
    --date-end 2024-08-31
```

Analyzes all 15 Algarve survey sites and generates:
- Coastal topography features (slope, aspect)
- DGT MDT-50cm mosaic
- Integration report

## Data Sources

### DGT MDT-50cm

- **Source**: DGT STAC API — https://dgt-be.a.incd.pt:8081/collections/MDT-50cm/items
- **Coverage**: Entire Portugal
- **Resolution**: 0.5 m
- **Projection**: EPSG:3763 (ETRS89 / Portugal TM06) — metric coordinates
- **Type**: Raster, Float32, GeoTIFF
- **Nodata**: -999.0
- **Derivation**: LiDAR (ICESat-2 + ground surveys)

### Sentinel-2 L2A

- **Source**: Copernicus Open Access Hub or sentinelhub-py
- **Bands**: B02 (Blue, 10m), B03 (Green, 10m), B04 (Red, 10m), B08 (NIR, 10m), B11/B12 (SWIR, 20m)
- **Resolution**: 10m (multispectral), 20m (SWIR)
- **Projection**: UTM (native), can be reprojected to EPSG:3763
- **Requires**: Free Copernicus account for download

## Integration with Reef Imagery Pipeline

### Feature Engineering

1. **Load coastal features** into visibility model:
   ```python
   import pandas as pd
   features = pd.read_csv("outputs/coastal/algarve_coastal_features.csv")
   
   # Join with existing reef metrics by site_name
   reef_data = pd.merge(reef_df, features, on="site_name")
   ```

2. **Use as static predictors** for water quality / visibility:
   - `slope_mean`: terrain steepness → sediment plume potential
   - `aspect_mean`: terrain aspect → wave/wind exposure → resuspension
   - `slope_p90`: local peaks → drainage patterns

3. **Combine with Sentinel-2** multispectral data:
   - Stack MDT + S2 bands in EPSG:3763
   - Use for bathymetry (Stumpf SDB), benthic classification, water column analysis

### Integration Points

- **Visibility Model** (`src/ranking_model.py`): Add slope/aspect as static features
- **Drift Monitor** (`src/drift_monitor.py`): Incorporate terrain exposure for plume predictions
- **Bathymetry Calibration** (`src/bathy_calibrator.py`): Use terrain for local water dynamics
- **Dashboard** (`dashboard/`): Display slope/aspect as context layers

### Example: Add to Visibility Model

```python
# In src/ranking_model.py or a new module

import pandas as pd
from coastal_topography import CoastalTopographyAnalyzer

# Extract coastal features once
analyzer = CoastalTopographyAnalyzer(bbox=(...), cache_tiles=True)
features = analyzer.extract_features_for_sites(sites, buffer_m=1000)

# In model feature engineering:
X_static = features[["slope_mean", "slope_p90", "aspect_mean"]].values

# Combine with dynamic features (Sentinel-2, wind, swell, etc.)
X_combined = np.hstack([X_static, X_dynamic])

# Train visibility model
visibility_model.fit(X_combined, y_visibility)
```

## Coordinate Systems

- **Input (dive sites)**: WGS84 (EPSG:4326), lat/lon
- **Native DGT MDT**: ETRS89 / Portugal TM06 (EPSG:3763), meters
- **Output rasters**: EPSG:3763 (matches DGT native)
- **Feature extraction**: All computations in EPSG:3763 (metric)

**Note**: Slope/aspect are computed using grid gradient method in meters; results are in degrees and m/m respectively.

## Output Schema

### Coastal Features CSV

```
site_name,latitude,longitude,buffer_m,slope_mean,slope_median,slope_std,slope_min,slope_max,slope_percentile_90,aspect_mean,aspect_median,aspect_std,aspect_min,aspect_max,timestamp
pedra_sta_eulalia,37.069081,-8.210242,1000,8.5,7.2,4.1,0.1,21.3,15.2,185.3,182.1,12.5,45.0,359.0,2024-06-05T...
albufeira_reef,37.0690,-8.2105,1000,9.1,8.5,3.8,0.2,19.8,16.1,192.1,190.5,11.2,60.0,359.0,2024-06-05T...
```

### Integration Report JSON

```json
{
  "title": "Algarve Reef Imagery Pipeline: DGT + Sentinel Integration Report",
  "timestamp": "2024-06-05T...",
  "coastal_topography": {
    "status": "success",
    "sites_analyzed": 15,
    "tiles_downloaded": 8,
    "output_files": {
      "csv": "outputs/coastal_topography/algarve_coastal_features.csv",
      "geojson": "outputs/coastal_topography/algarve_coastal_features.geojson"
    }
  },
  "dgt_sentinel": {
    "status": "success",
    "mdt_mosaic_path": "outputs/dgt_sentinel/MDT_50cm_mosaic_algarve.tif",
    "integrated_path": "outputs/dgt_sentinel/integrated_mdt_sentinel_algarve.tif"
  }
}
```

## Troubleshooting

### No MDT tiles found

**Issue**: Query returns 0 features
- Check bbox is within Portugal territory (DGT coverage)
- Verify bbox format: (minlon, minlat, maxlon, maxlat)

### Slope/aspect all NaN

**Issue**: Nodata values not handled correctly
- Ensure DEM has nodata=-999.0 set in rasterio
- Check grid resolution (should be 0.5 m for DGT)

### sentinelhub authentication fails

**Issue**: Cannot download Sentinel-2
- Create account at https://dataspace.copernicus.eu
- Set up `~/.sentinelhub/config.json` with credentials
- Or use alternative: Copernicus Browser / CDSE

### Memory issues with large bbox

**Issue**: Mosaic creation OOM
- Reduce bbox size
- Process tiles individually or in sub-regions
- Use `rasterio` with block I/O

## Performance Notes

- **Coastal features for 15 sites, 1 km buffer**: ~5-10 min (first run), <1 min (cached)
- **MDT-50cm download**: ~2-5 min per tile (~50 MB each)
- **Slope/aspect computation**: ~30-60 sec per mosaic
- **Sentinel-2 download**: ~5-15 min per scene (depends on cloud processing)

## References

### Coastal Topography & Sediment Dynamics

- Mahadevan, A., & Archer, D. (2000). Modeling the impact of fronts and mesoscale circulation on the transport of plankton larvae. *Limnology and Oceanography*, 45(5), 1112-1124.
- Stumpf, R. P., Arnone, R. A., Gould Jr, R. W., Martinolich, P. M., & Ransibhaskar, V. (2003). A partially coupled ocean-atmosphere model for retrieval of water-leaving radiance from SeaWiFS or MODIS. *Journal of Geophysical Research*, 108(C5), 3-1.

### DGT & Portuguese Geospatial Data

- DGT SNIG portal: https://snig.dgterritorio.gov.pt
- DGT Copernicus info: https://www.dgterritorio.gov.pt/pt/web/dgterritorio/programa-copernicus
- STAC API docs: https://dgt-be.a.incd.pt:8081/docs

## Contributing

To extend this integration:

1. **Add new terrain metrics**: Implement in `CoastalTopographyAnalyzer.derive_*()` methods
2. **Support other satellite data**: Extend `DGTSentinelIntegrator` with Landsat, Copernicus DEM, etc.
3. **Optimize performance**: Use dask for large-scale parallel processing
4. **Validation**: Compare zonal stats with in-situ measurements at dive sites

## License

Same as parent `reef_imagery_pipeline` project.

---

**Last updated**: June 2024  
**Maintainer**: Reef Imagery Pipeline Team
