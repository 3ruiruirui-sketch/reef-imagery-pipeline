# Open Data Documentation for Algarve Underwater Visibility

This file documents open, legal, and sustainable data sources for the Algarve underwater visibility and coastal bathymetry project.

## 1. Purpose

- Describe trusted data sources used in the project.
- Record access methods, licensing, and sustainability for open-source integration.
- Provide practical Python examples for Sentinel/STAC-based acquisition.
- Clarify which datasets are openly available vs. restricted.

## 2. Recommended Open Data Sources

### Sentinel-2 MSI (Multispectral Instrument)

- Source: Copernicus Open Access Hub and Copernicus Data Space Ecosystem.
- Public access: free and open under the Copernicus data policy.
- Typical products: `S2MSI2A` / `sentinel-2-l2a`.
- Use case: coastal reflectance-based bathymetry, water clarity, suspended sediment, benthic visibility.
- Regions: Algarve coastal zone, nearshore shallow water.

### Sentinel-1 SAR

- Source: Copernicus Open Access / AWS / GCP public datasets.
- Public access: free and open; useful for sea-state and coastline masking.
- Use case: shoreline detection, open-water mask, wave condition filtering for optical scene selection.

### EMODnet Bathymetry

- Source: EMODnet Bathymetry portal.
- Public access: open EU data, often delivered as raster tiles and regional grids.
- Use case: dynamic calibration target for Stumpf-based depth inversion, coastal bathymetry reference.
- Note: EMODnet is best used as a calibration anchor, not a pixel-perfect ground truth for visible shallow-water depths.

### Copernicus Marine Environment Monitoring Service (CMEMS)

- Source: https://marine.copernicus.eu
- Public access: free registration required for many datasets.
- Use case: marine/optical auxiliary data such as water clarity, chlorophyll concentration, turbidity, currents, sea surface temperature, and tide proxies.
- Example products: `GLOBAL_ANALYSIS_FORECAST_PHY_001_024`, `MEDSEA_ANALYSISFORECAST_PHY_006_001`.

### Portuguese national and regional geospatial data

- Open orthophotos and terrain data may be available from:
  - Direção-Geral do Território (DGT) orthophotos
  - Carta Administrativa Oficial de Portugal (CAOP)
  - Portuguese government open geodata portals
- Use case: land mask, coastal infrastructure, orthophoto verification for image selection.
- License: depends on the source. Verify each portal.

### IPMA and national weather/ocean forecasts

- Source: Instituto Português do Mar e da Atmosfera (IPMA).
- Use case: atmospheric conditions, wind, rainfall, and sea-state data for candidate image selection.
- Note: IPMA is useful for context but not typically a direct source of underwater visibility metrics.

### ICESat-2 and validation datasets

- Source: NASA/ICESat-2 ATL08, ATL03.
- Use case: independent validation of bathymetry and coastal elevation.
- License: NASA open data.

## 3. Legal and Licensing Notes

- **Copernicus Sentinel data** are distributed under the Copernicus data policy and are free for any purpose, including commercial.
- **EMODnet data** are published under an EU open data license (often EUPL or compatible open terms).
- **CMEMS data** are free for registered users, but a valid account and acceptance of Terms of Use are required.
- **Portuguese hydrographic charts** from Instituto Hidrográfico are generally not openly licensed for unrestricted reuse; only use these when explicit permission or a data license is granted.
- Always document:
  - the data source URL
  - the product name or collection ID
  - the access method used
  - the license or terms of use

## 4. Recommended Access Methods

### 4.1 STAC / Planetary Computer

Use STAC search and the Microsoft Planetary Computer catalog for Sentinel data.

```python
from pystac_client import Client
from planetary_computer import sign

catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
bbox = [-8.35, 36.95, -8.15, 37.12]
search = catalog.search(
    collections=["sentinel-2-l2a"],
    bbox=bbox,
    datetime="2025-05-01/2025-10-31",
    query={"eo:cloud_cover": {"lt": 20}},
)
items = list(search.get_items())
print(f"Found {len(items)} items")

if items:
    item = items[0]
    b02_asset = item.assets["B02"]
    signed_href = sign(b02_asset.href)
    print("Signed B02 href:", signed_href)
```

The repository also includes a reusable helper module for Sentinel-2 STAC discovery:

- `src/stac_ingest.py` — scene search, least-cloudy ranking, and signed asset URL extraction
- `scripts/sentinel2_stac_ingest.py` — CLI example for the Algarve using either Planetary Computer or Earth Search STAC

Example usage:

```bash
python scripts/sentinel2_stac_ingest.py \
  --lat 37.068978 --lon -8.210328 \
  --start 2025-09-01 --end 2025-09-30 \
  --max-cloud 20 --catalog pc \
  --output sentinel2_scene.json
```

### 4.2 AWS / GCP Public Dataset Access

- Sentinel-2 L2A on AWS: `https://sentinel-s2-l2a.s3.amazonaws.com/`
- Sentinel-1 on AWS: `https://sentinel-s1-l1c.s3.amazonaws.com/`
- Sentinel-2 on Google Cloud: `https://storage.googleapis.com/gcp-public-data-sentinel-2/`

Use STAC or a published index to resolve the tile path; direct URL composition is possible once you know the tile name and acquisition date.

### 4.2.1 Verified STAC Endpoints

- Element84 / Earth Search STAC: `https://earth-search.aws.element84.com/v1`
- Microsoft Planetary Computer STAC: `https://planetarycomputer.microsoft.com/api/stac/v1`
- Sentinel Hub / Copernicus Data Space: `https://sh.dataspace.copernicus.eu/`

> Note: the repository defaults to Earth Search first because it is currently the most reliable STAC endpoint for Sentinel-2 discovery.
> Planetary Computer remains available as a fallback.

### 4.3 EMODnet Bathymetry Download

- EMODnet portal: https://emodnet.ec.europa.eu/en/bathymetry
- Data products are often delivered as GeoTIFFs or regional mosaics.
- For Algarve work, download the nearest coastal tile and reproject to the Sentinel-2 grid before calibration.

### 4.4 CMEMS and Marine Data

- CMEMS portal: https://marine.copernicus.eu/
- Register for an account to access API credentials.
- Many products are offered as NetCDF and can be consumed with `xarray` after installing `netCDF4`.
- Example: fetch SST, currents, or water clarity proxies for ancillary model inputs.

## 5. Practical Integration Guidance

### 5.1 Sentinel-2 for shallow-water visibility

- Use Sentinel-2 L2A surface reflectance (B02, B03, B04, optionally B08).
- Prioritize scenes with:
  - cloud cover < 20%
  - sun zenith angles < 70°
  - low aerosol and low sea-surface glint
- Compute the Stumpf log-ratio depth estimate and calibrate using EMODnet or other reference bathymetry.

### 5.2 EMODnet dynamic calibration

- Use EMODnet bathymetry as a calibration anchor within shallow coastal zones.
- Co-register EMODnet and Sentinel-2 footprints to the same CRS and resolution.
- Compare only water pixels; exclude land and out-of-water areas.
- Record calibration metadata in output JSON.

### 5.3 Validation and provenance

- Produce metadata records containing:
  - source dataset and collection ID
  - acquisition date/time
  - geographic bounding box
  - license or access terms
  - calibration method used
- For each derived product, keep a simple provenance statement such as:
  - "Sentinel-2 L2A image via Microsoft Planetary Computer STAC"
  - "EMODnet bathymetry tile downloaded from EMODnet portal"

## 6. Data Priorities for Algarve Work

1. Sentinel-2 L2A for optical water reflectance.
2. EMODnet Bathymetry for dynamic depth calibration.
3. Sentinel-1 SAR for water masks and sea-state stability.
4. CMEMS marine variables for auxiliary analysis when available.
5. Portuguese orthophotos and DGT data for coastal land/shoreline validation.
6. IH or ICESat-2 datasets only for validation if properly licensed.

## 7. Citation and Attribution

- Copernicus Sentinel data: "Contains modified Copernicus Sentinel data [2024]."
- EMODnet Bathymetry: "Contains EMODnet Bathymetry data products." Add the specific tile or dataset name when possible.
- CMEMS: "Contains Copernicus Marine Environment Monitoring Service (CMEMS) data."
- IPMA: cite IPMA as the source for weather/ocean conditions if used.
- Always keep a `LICENSE` or `CITATION` section in derived product metadata.

## 8. Notes on Restricted Data

- Instituto Hidrográfico nautical charts and official bathymetry are often licensed and not freely redistributable.
- Use restricted datasets only when you have a written data license.
- Prefer open, public datasets for reproducible research and publication.

## 9. Suggested Future Improvements

- Add direct STAC-driven Sentinel-2 scene selection in the pipeline.
- Add EMODnet tile download automation and tile caching.
- Add CMEMS auxiliary data ingestion with optional `netCDF4` support.
- Add a provenance JSON writer for each derived product.
