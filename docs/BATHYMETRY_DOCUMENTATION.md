# Algarve Bathymetry & Topography Data Documentation

This document records the official bathymetric and topographic data sources integrated into the Algarve Underwater Visibility and Reef Imagery Pipeline. It provides data profiles, access methods, licensing, coordinate reference systems (CRS), and guidelines for combining these datasets to model benthic visibility, light attenuation, and wave-seabed interactions.

---

## 1. Primary Bathymetric & Topographic Datasets

### 1.1 Instituto Hidrográfico — SEAMAP 2030 Program
The official program by the Portuguese Navy's Hydrographic Institute (IH) to map the entire Portuguese maritime zone. It serves as the primary national reference and the ground-truth baseline for coastal bathymetry.

*   **Source Name:** SEAMAP 2030 (Mapeamento do Mar Português)
*   **Official URL:** [Instituto Hidrográfico - SEAMAP 2030](https://www.hidrografico.pt/en/activity/program-seamap-2030-mapping-of-the-portuguese-sea/)
*   **Access Portals:** [GeoMar Portal](https://geomar.hidrografico.pt/) & [Hidrográfico+ Portal](https://mais.hidrografico.pt/)
*   **SNIG Catalog ID:** [SEAMAP 2030 SNIG Record](https://snig.dgterritorio.gov.pt/rndg/srv/api/records/ebacc5a6-0504-4f97-94e5-4e5146dd0318)
*   **Access Method:** 
    *   *Visualization:* Interactive mapping via the GeoMar portal.
    *   *Web Services:* OGC WMS/WMTS/WCS endpoints served by IH GeoServer (`https://geomar.hidrografico.pt/geoserver/geomar/wcs`).
    *   *Direct Download:* Available for registered public entities and collaborative research partners. High-resolution raw surveys require a formal request.
*   **License & Reuse:** Free for public display and visualization. Commercial reuse or direct redistribution of the raw high-resolution grids requires explicit written permission and a data license from the Instituto Hidrográfico.
*   **Resolution:** Variable (5 m to 20 m in nearshore coastal areas; 100 m+ in deep offshore areas).
*   **Coordinate Reference System (CRS):** Native in PT-TM06 / ETRS89 (EPSG:3763); often served or reprojected in WGS 84 (EPSG:4326) for web services.
*   **Project Role:** Gold-standard ground truth for validation of Satellite-Derived Bathymetry (SDB) and verification of optical depth limits.
*   **Citation Format:** *Instituto Hidrográfico, Portugal. (2020). SEAMAP 2030 Bathymetric Grid. GeoMar Portal.*
*   **Access Limitations:** WCS coverages may occasionally undergo identifier changes on the GeoServer, requiring runtime fallback routines. High-resolution survey files are protected and require licensing.

---

### 1.2 Instituto Hidrográfico — 25 m Bathymetric Model (DTM)
A public bathymetric digital terrain model with continuous coverage for the Portuguese coastal zone, representing a standardized product compiled from multi-beam and single-beam hydrographic surveys.

*   **Source Name:** Modelo Batimétrico Resolução 25m (IH)
*   **Official URL:** [Instituto Hidrográfico](https://www.hidrografico.pt/)
*   **Access Method:** Accessible via WCS 1.0.0 and WMS protocols.
    *   *WCS Endpoint:* `https://geomar.hidrografico.pt/geoserver/geomar/wcs`
    *   *Known Coverage IDs:* `geomar:algarve_bathy`, `geomar:portugal_bathy`, `geomar:bathymetry`
*   **License & Reuse:** Publicly accessible for non-commercial research, academic usage, and visualization. Commercial products derived from this model require authorization.
*   **Resolution:** 25 meters (horizontal grid cell size).
*   **Coordinate Reference System (CRS):** Native PT-TM06 / ETRS89 (EPSG:3763).
*   **Project Role:** Local calibration anchor for Stumpf log-ratio depth inversion, providing a highly reliable coarse grid for regional wave-bottom interaction modeling.
*   **Citation Format:** *Instituto Hidrográfico, Portugal. (2015). Coastal Bathymetric Model 25m. Lisboa.*
*   **Access Limitations:** Public WCS can experience transient downtime during system maintenance; requires standard timeout handling and caching in the pipeline.

---

### 1.3 EMODnet Bathymetry / HRSM (High Resolution Seabed Mapping)
A harmonized, high-resolution bathymetric grid for all European sea basins, constructed by combining single/multibeam surveys, nautical charts, and satellite-derived bathymetry.

*   **Source Name:** EMODnet Digital Terrain Model (DTM)
*   **Official URL:** [EMODnet Bathymetry](https://emodnet.ec.europa.eu/en/bathymetry)
*   **Access Method:** Direct download of tiles in GeoTIFF/NetCDF, or programmatic querying via OGC WCS.
    *   *WCS Endpoint:* `https://ows.emodnet-bathymetry.eu/wcs`
    *   *WCS Layer Name:* `emodnet:mean` (Composite Mean Depth)
*   **License & Reuse:** Free and open for any use (commercial and non-commercial) under Creative Commons Attribution 4.0 International (CC-BY 4.0).
*   **Resolution:** 1/16 arc-minute (approximately 115 meters or 1/128°).
*   **Coordinate Reference System (CRS):** WGS 84 (EPSG:4326).
*   **Project Role:** Primary European benchmark layer and fallback calibration source. Used to co-register with Sentinel-2 tiles to dynamically fit Stumpf SDB model parameters in calm, clear scenes.
*   **Citation Format:** *EMODnet Bathymetry Consortium. (2022). EMODnet Digital Bathymetry (DTM). https://doi.org/10.12770/ff3aff8a-cff1-44a3-a2c8-1910bf109f85*
*   **Access Limitations:** Lower spatial resolution than national multibeam surveys, meaning it is unsuitable for resolving fine rocky outcrop crests or micro-reefs.

---

### 1.4 Coastal LiDAR Algarve 2011 (590_HR_Lidar_Algarve)
A high-resolution airborne topo-bathymetric LiDAR dataset capturing the shoreline morphology, intertidal zone, and very shallow coastal waters.

*   **Source Name:** Coastal LiDAR Algarve 2011 (590_HR_Lidar_Algarve)
*   **Official URL:** [EMODnet Geonetwork Entry](https://emodnet.ec.europa.eu/geonetwork/srv/api/records/SDN_CPRD_590_HR_Lidar_Algarve)
*   **Alternative Portals:** Direção-Geral do Território (DGT) / SNIG open data catalogs.
*   **Access Method:** 
    *   *DGT INCD API:* Programmatic tile fetching via DGT STAC endpoint (`https://dgt-be.a.incda.pt:8081/collections/MDT-50cm/items`).
    *   *Portal Download:* Bulk ZIP/TIFF downloads of local coastal sectors from DGT/SNIG.
*   **License & Reuse:** Open public data. Free for public administration, academic, research, and commercial purposes with appropriate attribution to DGT.
*   **Resolution:** 50 cm to 2 m horizontal resolution (highly precise).
*   **Coordinate Reference System (CRS):** PT-TM06 / ETRS89 (EPSG:3763).
*   **Project Role:** Land-sea boundary correction, intertidal zone modeling, tidal datum calibration, and high-resolution topo-bathy continuity mapping.
*   **Citation Format:** *Direção-Geral do Território, Portugal. (2011). Coastal LiDAR Survey Algarve. Lisboa.*
*   **Access Limitations:** Limited subtidal penetration (light attenuation typically restricts measurements to depths shallower than 3-5 meters depending on turbidity at the time of flight).

---

## 2. Summary Reference Matrix

| Dataset | Provider | Resolution | Native CRS | Primary Access | License | Project Role |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **SEAMAP 2030** | Instituto Hidrográfico | 5–20 m | EPSG:3763 | GeoMar Portal / WCS | Restricted (Commercial) / Free (Academic) | Ground Truth & Validation |
| **IH 25m Model** | Instituto Hidrográfico | 25 m | EPSG:3763 | GeoMar WCS | Free (Non-Commercial Research) | Calibration Anchor |
| **EMODnet DTM** | EU Consortium | ~115 m | EPSG:4326 | EMODnet WCS | Open Data (CC-BY 4.0) | Benchmark & Fallback |
| **LiDAR Algarve** | DGT / SNIG | 0.5–2 m | EPSG:3763 | DGT STAC / GeoTIFF | Open Data (CC-BY) | Shoreline & Intertidal |

---

## 3. Step-by-Step Access Instructions for Portals

### EMODnet Bathymetry Portal
1. Open the [EMODnet Bathymetry Viewer](https://portal.emodnet-bathymetry.eu/).
2. Zoom into the Algarve coastline (coordinates: `-8.5` to `-7.8` Lon, `36.9` to `37.1` Lat).
3. Open the "Download" panel on the right sidebar.
4. Select the desired format (e.g., GeoTIFF or NetCDF) and resolution (1/16 arc-minute).
5. Draw a custom bounding box or select the predefined tile covering the Algarve (`D7` or `E7`).
6. Download the grid file directly to the `data/` directory.

### DGT / SNIG LiDAR & Topo Portal
1. Visit the [SNIG Portal](https://snig.dgterritorio.gov.pt/).
2. Search for `MDT LiDAR Algarve` or `LiDAR 2011`.
3. Locate the download interface or use the open API endpoint: `https://dgt-be.a.incda.pt:8081/collections/MDT-50cm`.
4. Identify tiles covering your target zone (e.g. Faro/Carvoeiro).
5. Run the programmatic download script `ingest_lidar_dtm.py` specifying your bbox.

---

## 4. Algarve Coordinate Reference System Alignment

To avoid spatial offsets and distortion in distance-based calculations, the pipeline standardizes all geospatial raster and vector layers to a single local projected coordinate system:

*   **Standard Target CRS:** **PT-TM06 / ETRS89 (EPSG:3763)**
    *   *Type:* Projected Coordinate System (Transverse Mercator).
    *   *Coverage:* Portugal Mainland.
    *   *Unit:* Meter (enables exact grid cell sizing, slope angle, and area calculations).
    *   *WGS84 Equivalence:* Uses ETRS89 datum, which is compatible with WGS84 for coastal marine mapping.
*   **Alternative CRS for Sentinel-2 Co-Registration:** **WGS 84 / UTM Zone 29N (EPSG:32629)**
    *   *Usage:* Best for matching the native resolution of Sentinel-2 L2A tiles without rotating the grid cells, which preserves the original optical reflectance data.
