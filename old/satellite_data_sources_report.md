# Satellite Data Sources for Underwater Reef Detection (10-25m Depth)

> **Target**: Algarve coast, Portugal (37°N, -8°W)  
> **Depth range**: 10-25m below sea level  
> **Physics**: Blue light (400-480nm) penetrates deepest in clear ocean water. Coastal Blue / Aerosol bands at ~443nm are specifically optimized for bathymetry. In clear Atlantic water, practical penetration reaches 20-30m; in turbid coastal water drops to 2-5m.  
> **Report date**: 2026-05-29

---

## Quick-Reference Summary Table

| Source | Bands for Bathymetry | Resolution | Max Depth (clear) | Access | Algarve | Python API | Cost |
|--------|---------------------|-----------|-------------------|--------|---------|------------|------|
| Sentinel-2 MSI | B1(443nm), B2(490nm), B3(560nm) | 10-60m | ~20-25m | FREE | Yes | Yes | Free |
| Landsat 8/9 OLI | B1(443nm), B2(482nm), B3(561nm) | 30m | ~20m | FREE | Yes | Yes | Free |
| EMODnet Bathymetry | Derived DTM | ~115m | N/A (pre-computed) | FREE | Yes | WMS/WFS | Free |
| GEBCO | Derived DTM | ~450m | N/A (pre-computed) | FREE | Yes | NetCDF | Free |
| Planet SuperDove | Coastal Blue(443nm), Blue, Green, RE | 3m | ~25m | COMMERCIAL | Yes | Yes | $$$ |
| Planet NICFI | SuperDove tropics | 5m | ~25m | FREE-WITH-REG | **NO** (tropics) | Yes | Free |
| WorldView-2 (Vantor) | Coastal(427nm), Blue, Green | 1.8m (MS) | ~25-30m | COMMERCIAL | Yes | Yes | $$$$ |
| WorldView-3 (Vantor) | Coastal(427nm), Blue, Green, SWIR | 1.24m (MS) | ~25-30m | COMMERCIAL | Yes | Yes | $$$$ |
| Pléiades Neo (Airbus) | Coastal Blue, Blue, Green | 3m (MS) | ~20-25m | COMMERCIAL | Yes | Yes | $$$$ |
| PRISMA (ASI) | 250+ bands (400-2500nm) | 30m | ~20-25m | ACADEMIC-PROGRAM | Yes | Limited | Free (EU researchers) |
| EnMAP (DLR) | 230+ bands (420-2450nm) | 30m | ~20-25m | FREE-WITH-REG | Yes | Yes | Free (scientific) |
| DESIS (ISS/NASA) | 235 bands (400-1000nm) | 30m | ~20-25m | ACADEMIC-PROGRAM | Yes | Limited | Free (proposal) |
| Planet Tanager | 400+ bands (350-2500nm) | 30m | ~20-25m | COMMERCIAL | Yes | Yes | $$$$$ |
| ICESat-2 ATL03/ATL24 | Green laser (532nm) | 17m along-track | ~40m (lidar) | FREE-WITH-REG | Yes | Yes | Free |
| MODIS (Aqua/Terra) | B8-16 (412-869nm) | 250m-1km | ~15-20m | FREE | Yes | Yes | Free |
| VIIRS (Suomi-NPP) | M1-M5 (412-865nm) | 375-750m | ~15m | FREE | Yes | Yes | Free |

---

## A. Sources Available RIGHT NOW for Free (No Registration Required)

### A1. Sentinel-2 MSI (Copernicus / ESA)

| Field | Value |
|-------|-------|
| **Satellite** | Sentinel-2A, 2B, 2C (Copernicus programme, ESA) |
| **Bathymetry bands** | **B1 - Coastal Aerosol: 443nm (20nm FWHM), 60m** — primary bathymetry band. **B2 - Blue: 490nm (65nm), 10m** — secondary bathymetry. **B3 - Green: 560nm (35nm), 10m** — complementary for deeper features |
| **Max penetration** | ~20-25m in clear water (B1/B2 combined). Literature reports successful SDB to 25m in clear Mediterranean/Atlantic conditions |
| **Revisit** | 5 days (with both satellites) |
| **Covers Algarve** | Yes — full coverage, Path 29, Row 28/29 |
| **Access** | **FREE** — full, open, no restrictions |
| **Registration URL** | https://dataspace.copernicus.eu/ — register for API access (free) |
| **Python API** | `sentinelhub-py` (pip install sentinelhub), `sentinelsat`, Copernicus Data Space STAC API, openEO Python client |
| **Key strengths** | 10m blue/green bands, 5-day revisit, massive archive since 2015, surface reflectance products available |
| **Best for** | Wide-area SDB mapping, time-series analysis, multi-temporal change detection |
| **Cost** | Free |

**Download URLs:**
- Browser: https://browser.dataspace.copernicus.eu/
- API: https://dataspace.copernicus.eu/analyse/apis
- Documentation: https://documentation.dataspace.copernicus.eu/

---

### A2. Landsat 8/9 OLI/OLI-2 (NASA / USGS)

| Field | Value |
|-------|-------|
| **Satellite** | Landsat 8 (2013-present), Landsat 9 (2021-present), NASA/USGS |
| **Bathymetry bands** | **B1 - Coastal Aerosol: 443nm (20nm), 30m** — designed for bathymetry. **B2 - Blue: 482nm (60nm), 30m**. **B3 - Green: 561nm (57nm), 30m** |
| **Max penetration** | ~20m in clear water. Proven SDB applications to 15-20m in published literature |
| **Revisit** | 16 days (8 days with both satellites) |
| **Covers Algarve** | Yes — WRS-2 Path 204, Row 33/34 |
| **Access** | **FREE** via USGS EarthExplorer |
| **Registration URL** | https://ers.cr.usgs.gov/register/ (USGS ERS account) |
| **Python API** | `landsatxplore`, `earthaccess`, Google Earth Engine (`ee`), Microsoft Planetary Computer |
| **Key strengths** | Deep archive since 1972 (L7), 30m coastal band, well-validated for SDB |
| **Best for** | Long-term change detection, validated bathymetric algorithms |
| **Cost** | Free |

**Download URLs:**
- EarthExplorer: https://earthexplorer.usgs.gov/
- Earthdata: https://search.earthdata.nasa.gov/search
- Cloud: https://planetarycomputer.microsoft.com/

---

### A3. EMODnet Bathymetry (European Commission)

| Field | Value |
|-------|-------|
| **Source** | European Marine Observation and Data Network, EU-funded |
| **Type** | Pre-computed Digital Terrain Model (DTM), includes Satellite Derived Bathymetry |
| **Resolution** | ~115m (1/16 arc-minute), higher resolution HR-DTMs available for selected areas |
| **Max penetration** | N/A — already derived from multisource surveys + SDB from Sentinel-2/Landsat-8 |
| **Covers Algarve** | Yes — explicitly covered (Iberian Coast and Bay of Biscay region) |
| **Access** | **FREE**, open data, CC-BY license |
| **URL** | https://emodnet.ec.europa.eu/geoviewer |
| **Python API** | WMS/WFS/WCS OGC services; can be consumed via `owslib`, `geopandas`, `rasterio` |
| **Key strengths** | Pre-computed depth models, includes SDB data for coastal Spain/Portugal, quality-indexed |
| **Best for** | Ground truth reference, baseline bathymetry, validation |
| **Cost** | Free |

**Services:**
- Map viewer: https://emodnet.ec.europa.eu/geoviewer/
- WMS: `https://ows.emodnet-bathymetry.eu/wms`
- Download: Available in ESRI ASCII, NetCDF, GeoTIFF, CSV formats

---

### A4. GEBCO (General Bathymetric Chart of the Oceans)

| Field | Value |
|-------|-------|
| **Source** | IHO-IOC Joint Programme, maintained by GEBCO community |
| **Type** | Global bathymetric grid |
| **Resolution** | ~450m (15 arc-second) |
| **Covers Algarve** | Yes — global coverage |
| **Access** | **FREE** |
| **URL** | https://www.gebco.net/data_products/ |
| **Python API** | NetCDF download; `netCDF4`, `xarray`, `rasterio` |
| **Cost** | Free |

---

## B. Sources Requiring Free Registration

### B1. Planet NICFI Tropical Forest Program

| Field | Value |
|-------|-------|
| **Satellite** | Planet SuperDove constellation |
| **Bands** | 8-band including Coastal Blue (443nm), 3m resolution |
| **Coverage** | **TROPICS ONLY (±30° latitude)** |
| **Algarve coverage** | **NO** — 37°N is outside tropics |
| **Access** | Free with registration for non-commercial tropical forest monitoring |
| **URL** | https://www.planet.com/tropical-forest-observatory/ |
| **Note** | Excellent for tropical reef monitoring but **NOT applicable to Algarve project** |

### B2. NASA Earthdata

| Field | Value |
|-------|-------|
| **Data** | Landsat, MODIS, ICESat-2, and derived products |
| **Access** | FREE with registration |
| **URL** | https://urs.earthdata.nasa.gov/users/new |
| **Python API** | `earthaccess` (pip install earthaccess), `harmony-py` |
| **Key datasets** | MODIS L1/L2 ocean color (bathymetry-capable), ICESat-2 ATL24 bathymetry |
| **Cost** | Free |

### B3. ICESat-2 ATL03/ATL24 (NASA)

| Field | Value |
|-------|-------|
| **Satellite** | ICESat-2 (NASA), launched 2018 |
| **Sensor** | ATLAS — green laser (532nm) photon-counting lidar |
| **Resolution** | ~17m along-track, 0.7m vertical precision |
| **Max penetration** | ~40m in clear water (lidar, not passive optical) |
| **Covers Algarve** | Yes — global coverage, but limited track density |
| **Access** | FREE via NSIDC with NASA Earthdata registration |
| **URL** | https://nsidc.org/data/atl24 |
| **Python API** | `icepyx` (pip install icepyx) |
| **Key strengths** | Deepest penetration of any satellite, direct depth measurement, works regardless of sun angle |
| **Key limitation** | Point measurements along track (~17m footprints), not imagery. Sparse coverage for wide-area mapping |
| **Best for** | Depth validation/calibration, deep reef detection at isolated points |
| **Cost** | Free |

### B4. EnMAP (DLR / German Space Agency)

| Field | Value |
|-------|-------|
| **Satellite** | EnMAP (Environmental Mapping and Analysis Program), DLR, launched 2022 |
| **Sensor** | Hyperspectral imaging spectrometer |
| **Bands** | 230+ bands, VNIR: 420-1000nm (5nm sampling), SWIR: 900-2450nm (10nm) |
| **Resolution** | 30m |
| **Max penetration** | ~20-25m (VNIR bands in clear water) |
| **Covers Algarve** | Yes — global coverage on request |
| **Access** | **FREE for scientific use** — requires proposal via EnMAP Data Portal |
| **Registration URL** | https://enmap.org/data_access/ |
| **Python API** | `enpt` (EnMAP Processing Tool), standard rasterio/xarray |
| **Key strengths** | Hyperspectral (narrow bands optimal for water column correction), 30m, free for science |
| **Best for** | High-accuracy SDB with physics-based water column models |
| **Cost** | Free (scientific) |

### B5. MODIS Aqua/Terra (NASA)

| Field | Value |
|-------|-------|
| **Sensor** | MODIS (Moderate Resolution Imaging Spectroradiometer) |
| **Bands** | B8-B16 (412-869nm) at 1km, B1-B2 (645-858nm) at 250m |
| **Max penetration** | ~15-20m (very low resolution) |
| **Access** | FREE via NASA Earthdata, LAADS DAAC |
| **URL** | https://ladsweb.modaps.eosdis.nasa.gov/ |
| **Python API** | `earthaccess`, `pymodis` |
| **Key limitation** | 250m-1km resolution — too coarse for reef-scale mapping |
| **Best for** | Ocean color/health monitoring, water clarity estimation |
| **Cost** | Free |

---

## C. Sources with Free Demo / Trial

### C1. Planet Free Trial (Planet Insights Platform)

| Field | Value |
|-------|-------|
| **Data** | PlanetScope SuperDove (3m, 8-band including Coastal Blue 443nm) |
| **Trial terms** | Limited free trial with access to Planet Sandbox Data and API explorer |
| **Registration URL** | https://insights.planet.com/sign-up/ |
| **Python API** | `planet` (pip install planet), Planet Data API v2, Planet SDK |
| **Documentation** | https://docs.planet.com/ |
| **Algarve** | Yes — global daily coverage |
| **Cost after trial** | Contact sales; typical annual subscriptions start ~$10K-50K+ depending on area/quota |

### C2. Maxar/Vantor SecureWatch Trial

| Field | Value |
|-------|-------|
| **Data** | WorldView-2/3/LEGION archive (including coastal blue band) |
| **Trial** | Demo access available through Vantor Hub |
| **Registration URL** | https://hub.vantor.com |
| **Documentation** | https://discover.vantor.com/ |
| **Note** | Maxar rebranded as Vantor (2025). Access programs for government/research |
| **Cost after trial** | Commercial pricing, typically $10-30/km² for archive, tasking priced per collection |

---

## D. Academic / Research Programs

### D1. Planet Education & Research Program

| Field | Value |
|-------|-------|
| **Data** | PlanetScope (SuperDove 8-band), RapidEye archive |
| **Eligibility** | University researchers, students (non-commercial) |
| **Coverage** | 10,000+ users in 100+ countries |
| **Application URL** | https://www.planet.com/science/ |
| **Python API** | Full Planet SDK access |
| **Approval time** | Up to 3 weeks |
| **Term** | 1 year, renewable |
| **Algarve** | Yes |
| **Cost** | Free |

### D2. ESA EarthNet Programme (Third Party Missions)

| Field | Value |
|-------|-------|
| **Data** | PlanetScope, SkySat, RapidEye, plus other Third Party Mission data |
| **Eligibility** | Researchers in ESA Member States (includes Portugal), EU, China |
| **Application URL** | https://eoiam-idp.eo.esa.int/myaccount/login |
| **Process** | Submit Project Proposal, evaluated by ESA + data owner |
| **Approval time** | Up to 2 weeks, rolling deadlines |
| **Cost** | Free |

### D3. NASA CSDA (Commercial SmallSat Data Acquisition)

| Field | Value |
|-------|-------|
| **Data** | PlanetScope archive, RapidEye archive |
| **Eligibility** | US federally-funded researchers / NSF-funded |
| **Quota** | 5,000,000 km² per user initial allocation |
| **URL** | https://go.planet.com/nasa |
| **Cost** | Free |

### D4. DLR RESA (RapidEye Science Archive)

| Field | Value |
|-------|-------|
| **Data** | PlanetScope, RapidEye, SkySat |
| **Eligibility** | German researchers and institutions |
| **URL** | https://www.eo-lab.org/en/news_article/20241212/ |
| **Cost** | Free |

### D5. PRISMA Data Access (ASI — Italian Space Agency)

| Field | Value |
|-------|-------|
| **Satellite** | PRISMA (Precursore Iperspettrale della Missione Applicativa), ASI, launched 2019 |
| **Sensor** | Hyperspectral imaging spectrometer |
| **Bands** | 250+ bands, VNIR: 400-1010nm, SWIR: 920-2505nm |
| **Resolution** | 30m (hyperspectral), 5m (panchromatic) |
| **Max penetration** | ~20-25m (VNIR bands in clear water) |
| **Covers Algarve** | Yes — European/global coverage on request |
| **Access for EU researchers** | **FREE** via ASI proposal process |
| **Registration URL** | https://prisma.asi.it/ |
| **Process** | Submit data access request to ASI; Italian and EU researchers prioritized |
| **Python API** | Limited; HDF5 format, readable with `h5py`/`rasterio` |
| **Key strengths** | Free hyperspectral data for European coastal research, 30m resolution |
| **Cost** | Free (EU academic), tasking for commercial |

---

## E. Commercial Sources with Pricing

### E1. Planet SuperDove (Planet Labs PBC)

| Field | Value |
|-------|-------|
| **Satellite** | SuperDove constellation (~200+ satellites) |
| **Key bathymetry band** | **Coastal Blue: 431-452nm (20nm FWHM), 3m GSD** |
| **All bands** | 8-band: Coastal Blue(443nm), Blue(490nm), Green I(531nm), Green(565nm), Yellow(610nm), Red(665nm), Red Edge(705nm), NIR(865nm) |
| **Resolution** | 3m native (SuperRes: 2m with AI enhancement) |
| **Max penetration** | ~25m in clear water (Coastal Blue band) |
| **Revisit** | Near-daily global |
| **Covers Algarve** | Yes — daily coverage |
| **Python API** | `planet` SDK (pip install planet), Data API v2, Orders API |
| **Pricing** | Annual subscriptions from ~$10,000/yr (limited area) to $500K+ (global). Per-scene: not publicly listed — contact sales. Sandbox data available free for testing. |
| **Contact** | https://www.planet.com/contact-sales/ |
| **URL** | https://www.planet.com/pricing/ |

### E2. Vantor WorldView-2 (formerly Maxar)

| Field | Value |
|-------|-------|
| **Satellite** | WorldView-2 (DigitalGlobe/Vantor), launched 2009 |
| **Key bathymetry band** | **Coastal Blue: 427nm (48nm FWHM), 1.8m GSD** |
| **All bands** | 8-band: Coastal(427nm), Blue(478nm), Green(546nm), Yellow(608nm), Red(659nm), Red Edge(724nm), NIR(833nm), NIR2(949nm) |
| **Resolution** | 1.8m multispectral, 0.46m panchromatic |
| **Max penetration** | ~25-30m in clear water |
| **Covers Algarve** | Yes — extensive European archive |
| **Python API** | Vantor Hub API, `requests`-based |
| **Pricing** | Archive: ~$10-20/km². Tasking: $15-30/km². Volume discounts. Academic pricing via programs. |
| **URL** | https://www.vantor.com/product/worldview/2d/ |

### E3. Vantor WorldView-3

| Field | Value |
|-------|-------|
| **Satellite** | WorldView-3 (Vantor), launched 2014 |
| **Key bathymetry bands** | **Coastal Blue: 427nm, Blue, Green** + SWIR bands for atmospheric correction |
| **Resolution** | 1.24m multispectral, 0.31m panchromatic, 3.7m SWIR |
| **Max penetration** | ~25-30m |
| **Pricing** | Archive: ~$12-25/km². Tasking: $18-35/km². |
| **URL** | https://www.vantor.com/product/worldview/2d/ |

### E4. Airbus Pléiades Neo

| Field | Value |
|-------|-------|
| **Satellite** | Pléiades Neo 3 & 4 (Airbus Defence & Space), launched 2021-2022 |
| **Bands** | 4-band MS: Coastal Blue, Blue, Green, Red + Panchromatic |
| **Resolution** | 3m multispectral, 0.3m panchromatic |
| **Max penetration** | ~20-25m |
| **Covers Algarve** | Yes |
| **Pricing** | Archive: ~$10-20/km². Tasking: ~$20-35/km². |
| **Registration URL** | https://www.airbus.com/en/space-intelligence |
| **Python API** | Airbus OneAtlas API |

### E5. Planet SkySat

| Field | Value |
|-------|-------|
| **Satellite** | SkySat constellation (Planet Labs) |
| **Bands** | 4-band: Blue(485nm), Green(555nm), Red(660nm), NIR(830nm) |
| **Resolution** | 0.5m panchromatic, 1m multispectral |
| **Max penetration** | ~15-20m (no dedicated coastal blue band) |
| **Pricing** | Included in some Planet subscriptions, or per-tasking |
| **URL** | https://docs.planet.com/data/imagery/skysat/ |

---

## F. Hyperspectral Sources (Detailed)

### F1. PRISMA (ASI / Italy)

| Field | Value |
|-------|-------|
| **Operator** | Agenzia Spaziale Italiana (ASI) |
| **Bands** | 250+ bands, VNIR 400-1010nm (12nm sampling), SWIR 920-2505nm (12nm) |
| **Resolution** | 30m hyperspectral, 5m panchromatic |
| **Orbit** | Sun-synchronous, 615km altitude |
| **How to request** | 1. Register at https://prisma.asi.it/ 2. Submit data access proposal 3. Italian/EU researchers prioritized |
| **Format** | HDF5 |
| **Cost** | **Free** for Italian and EU researchers via ASI proposal |
| **Python** | `h5py` for reading, `rasterio` with GDAL HDF5 driver |

### F2. EnMAP (DLR / Germany)

| Field | Value |
|-------|-------|
| **Operator** | German Aerospace Center (DLR), launched 2022 |
| **Bands** | 230+ bands, VNIR 420-1000nm (5nm), SWIR 900-2450nm (10nm) |
| **Resolution** | 30m |
| **How to request** | 1. Register at https://enmap.org/ 2. Submit proposal via EnMAP Data Portal 3. Free for scientific use |
| **Format** | ENVI, GeoTIFF |
| **Cost** | **Free** for scientific projects |
| **Python** | `enpt` (EnMAP Processing Tool), `rasterio`, `xarray` |

### F3. DESIS (ISS / NASA + DLR)

| Field | Value |
|-------|-------|
| **Operator** | DLR / NASA, mounted on ISS (2018-present) |
| **Bands** | 235 bands, 400-1000nm (2.5nm sampling) |
| **Resolution** | 30m |
| **ISS orbit** | 51.6° inclination — **covers Algarve** |
| **How to request** | Via NASA Earthdata / ISS Research portal |
| **Cost** | **Free** via research proposal |

### F4. Planet Tanager (Commercial)

| Field | Value |
|-------|-------|
| **Operator** | Planet Labs |
| **Bands** | 400+ bands, 350-2500nm |
| **Resolution** | 30m |
| **Access** | Commercial — contact sales |
| **URL** | https://www.planet.com/products/hyperspectral/ |
| **Cost** | Premium commercial pricing — not publicly listed |

### F5. HySpex (Airborne — NOT satellite)

| Field | Value |
|-------|-------|
| **Operator** | NEO (Norway), now part of Teledyne |
| **Type** | Airborne hyperspectral sensors — NOT a satellite |
| **Bands** | VNIR + SWIR, up to 400+ bands |
| **URL** | https://www.hyspex.no/ |
| **Note** | Would require commissioning a flight campaign over Algarve. Not a satellite data source. |

---

## Recommended Action Plan for 10-25m Reef Detection on the Algarve Coast

### Tier 1 — Start immediately (FREE)

1. **Sentinel-2 via Copernicus Data Space** — Download B1/B2/B3 for the Algarve coast. Use `sentinelhub-py` or `sentinelsat` to automate. 10m blue bands can resolve reef features at 10-20m depth.
2. **Landsat 8/9 via USGS EarthExplorer** — B1/B2/B3 at 30m. Use for historical analysis (pre-2015).
3. **EMODnet Bathymetry DTM** — Download pre-computed bathymetry as ground truth/validation reference.
4. **ICESat-2 ATL24** — Download lidar depth profiles along Algarve coast for validation points. Use `icepyx`.

### Tier 2 — Register (free, same week)

5. **EnMAP hyperspectral** — Submit proposal for Algarve coastal area. Free for scientific use, 30m, 230+ bands gives best water column correction.
6. **NASA Earthdata** — Register for MODIS ocean color products (water clarity/chlorophyll).

### Tier 3 — Apply for academic access

7. **Planet Education & Research** — Apply for SuperDove 8-band access (3m, Coastal Blue 443nm). Approval in ~3 weeks.
8. **ESA EarthNet** — Apply for PlanetScope/SkySat access via ESA. Portugal is an ESA member state.
9. **PRISMA via ASI** — Apply for hyperspectral data. EU researchers eligible.

### Tier 4 — Commercial (if budget allows)

10. **Planet SuperDove subscription** — Best value for 3m daily coastal monitoring with Coastal Blue band.
11. **WorldView-2/3 archive** — Best resolution (1.8m/1.24m) with proven coastal blue band for deep bathymetry.
12. **Pléiades Neo** — 3m with coastal blue, good archive availability over Europe.

---

## Key Band Comparison for Bathymetry

| Satellite | Coastal Blue (nm) | Blue (nm) | Green (nm) | Resolution | Depth Advantage |
|-----------|-------------------|-----------|------------|-----------|----------------|
| Sentinel-2 B1/B2 | 443 | 490 | 560 | 10-60m | Good, free |
| Landsat 8/9 B1/B2 | 443 | 482 | 561 | 30m | Good, free |
| SuperDove | 443 | 490 | 565 | 3m | Best value commercial |
| WorldView-2/3 | 427 | 478 | 546 | 1.8m | Deepest penetration |
| Pléiades Neo | ~440 | ~490 | ~550 | 3m | Good |
| PRISMA | 400-1010 (continuous) | — | — | 30m | Hyperspectral = best physics |
| EnMAP | 420-1000 (continuous) | — | — | 30m | Hyperspectral = best physics |
| ICESat-2 | — | — | 532 (laser) | 17m along-track | Deepest overall (40m) |

---

## Python Library Quick Reference

| Library | Install | Purpose |
|---------|---------|---------|
| `sentinelhub` | `pip install sentinelhub` | Sentinel-2 data access via Sentinel Hub API |
| `sentinelsat` | `pip install sentinelsat` | Search/download from Copernicus Open Access Hub |
| `earthaccess` | `pip install earthaccess` | NASA Earthdata search/download |
| `planet` | `pip install planet` | Planet Labs Data/Orders/Tasking APIs |
| `icepyx` | `pip install icepyx` | ICESat-2 data access |
| `rasterio` | `pip install rasterio` | Raster I/O for GeoTIFF/NetCDF |
| `xarray` | `pip install xarray` | Multi-dimensional array analysis |
| `owslib` | `pip install OWSLib` | OGC Web Services (WMS/WFS/WCS) |
| `geopandas` | `pip install geopandas` | Vector data handling |
| `h5py` | `pip install h5py` | HDF5 file reading (PRISMA, MODIS) |
