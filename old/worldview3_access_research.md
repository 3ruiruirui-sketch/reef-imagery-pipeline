# WorldView-3 Satellite Imagery Access Research
## Coastal/Reef Bathymetry Applications — Algarve Coast (37.045N, -8.175W)

> **Note:** Maxar has rebranded to **Vantor** (as of ~2025-2026). All references to Maxar APIs, docs, and portals now point to Vantor infrastructure. The old `maxar.com` and `docs.maxar.com` domains redirect to `vantor.com` and `developers.maxar.com` respectively.

---

## 1. Available Access Methods

### 1.1 Vantor Hub (formerly Maxar SecureWatch) — PRIMARY COMMERCIAL ACCESS
- **URL:** https://hub.vantor.com
- **Developer Portal:** https://developers.maxar.com
- **Type:** Commercial subscription (self-service)
- **What it provides:** Tasking, archive search, ordering, streaming, and download of WorldView imagery via web portal and REST APIs
- **Auth:** API key or OAuth2 bearer token
- **Python SDK:** `pip install maxar-platform` (replaces deprecated MGP-SDK)
- **Supports:** Discovery (STAC), Ordering, Streaming (OGC), Raster Analytics, Tasking

### 1.2 Vantor Discover (Archive Search Portal)
- **URL:** https://discover.vantor.com
- **Type:** Free browsing of available imagery catalog (ordering requires subscription)
- **What it provides:** Search/filter 125+ PB of imagery catalog via web UI
- **Note:** Requires JavaScript; the STAC-based catalog can be queried programmatically via the Discovery API

### 1.3 WorldView Access Programs (Premium Tasking)
Two tiers for priority satellite access:

| Feature | Direct Access | Rapid Access |
|---------|--------------|--------------|
| Tasking priority | Highest commercially available | High |
| Delivery latency | As fast as 15 min after collection | Within 6 hours |
| Ground infrastructure | Requires dedicated ground station | Cloud-based (no hardware) |
| Tasking window | Up to 15 min before imaging event | Up to 90 min before acquisition |
| Data ownership | Private | Private |
| Contact | sales@vantor.com | sales@vantor.com |

### 1.4 Vantor Open Data Program — DISASTER ONLY
- **URL:** https://www.vantor.com/open-data
- **License:** Creative Commons BY-NC 4.0
- **Scope:** **Disaster response only** — not applicable for routine coastal research
- **Activation:** Only during sudden-onset major disasters with FirstLook activation
- **Access:** Via https://discover.vantor.com when activated
- **Not suitable** for bathymetry research

### 1.5 Google Earth Engine (GEE)
- WorldView-2 and WorldView-3 imagery is **NOT freely available** on GEE
- GEE has Maxar/Vantor imagery only through specific programmatic partnerships or commercial licensing
- Sentinel-2 (free) is available on GEE and has coastal/bathymetry applications but lacks the Coastal Blue band

### 1.6 AWS Open Data
- **No free WorldView-3 data on AWS Open Data** program
- Maxar/Vantor imagery may be available through AWS Marketplace for commercial customers
- The open data on AWS is limited to disaster response activations (same as the Open Data Program)

### 1.7 Academic/Research Programs
- **PEaRS Program (Participation in Education and Research):** Historically offered by Maxar for academic access. The original URL (`maxar.com/maxar-news/maxar-opens-satellite-imagery-archive-for-research`) now redirects to the Vantor homepage. **The PEaRS program appears to have been discontinued or significantly restructured** following the Maxar→Vantor rebrand.
- **Current approach:** Contact Vantor sales directly at https://www.vantor.com/get-started/ and inquire about academic/research pricing
- **For European coastal research:** Contact Vantor's European partners/resellers via https://www.vantor.com/partner-directory/

### 1.8 Approved Resellers
- Vantor has an approved reseller network for smaller tasking orders
- **Partner directory:** https://www.vantor.com/partner-directory/
- Resellers may offer lower minimum order quantities suitable for research projects

---

## 2. API Endpoints and Authentication

### 2.1 Authentication

**Option A: API Key (Recommended)**
```bash
# Set environment variable
export MAXAR_API_KEY="your-api-key-here"
```
- Generate at: Hub account settings → API Keys
- Valid for up to 180 days
- Can be revoked at any time
- Works for data-related APIs (Discovery, Ordering, Streaming)

**Option B: OAuth2 Bearer Token**
```bash
# Exchange credentials for token
POST https://api.maxar.com/oauth/token
Content-Type: application/x-www-form-urlencoded

grant_type=password&username=YOUR_USERNAME&password=YOUR_PASSWORD
```
- Token lifespan: 2 hours (must be refreshed)
- Required for Admin and Auth APIs

### 2.2 Key API Endpoints

| Service | Endpoint Base | Description |
|---------|--------------|-------------|
| Authentication | `https://api.maxar.com/oauth/token` | Get bearer token |
| API Key | `https://api.maxar.com/apikey/v1/` | Manage API keys |
| Discovery (STAC) | `https://api.maxar.com/discovery/v1/` | Search catalog |
| Ordering | `https://api.maxar.com/ordering/v1/` | Order imagery |
| Streaming (OGC) | `https://api.maxar.com/streaming/v1/` | Stream imagery |
| Tasking | `https://api.maxar.com/tasking/v1/` | Task satellites |
| Raster Analytics | `https://api.maxar.com/raster-analytics/v1/` | On-the-fly processing |

### 2.3 STAC Discovery API

The Discovery API supports STAC-compliant search:

```
POST https://api.maxar.com/discovery/v1/search/stac
Authorization: Bearer YOUR_TOKEN
Content-Type: application/json

{
  "collections": ["wv03"],
  "intersects": {
    "type": "Point",
    "coordinates": [-8.175, 37.045]
  },
  "datetime": "2020-01-01T00:00:00Z/2026-12-31T23:59:59Z",
  "limit": 10
}
```

---

## 3. Pricing Tiers

Vantor does **not publish standard pricing publicly**. Pricing is based on:

- **Resolution tier:** 30cm, 50cm, 1m (archive); 30cm (tasking)
- **Area of interest size:** Smaller AOIs are cheaper
- **Delivery speed:** Standard vs. priority
- **Archive vs. Tasking:** Archive imagery is significantly cheaper than new tasking
- **Volume commitments:** Bulk discounts available

**Estimated ranges (from industry knowledge):**

| Product | Approximate Price Range |
|---------|----------------------|
| Archive 30cm (per sq km) | $10-25 USD/sq km |
| Archive 50cm (per sq km) | $5-15 USD/sq km |
| Tasking 30cm (per sq km) | $25-50+ USD/sq km |
| Streaming access (subscription) | Contact sales |
| Academic/research discounts | Contact sales — historically 50-80% discount |

**For bathymetry research on the Algarve coast (~50 sq km study area):**
- Archive 30cm: ~$500-1,250 USD
- Archive 50cm: ~$250-750 USD
- New tasking: ~$1,250-2,500+ USD

---

## 4. WorldView-3 Band Specifications for Bathymetry

### 4.1 Full Spectral Band Configuration

WorldView-3 has **8 multispectral bands + 8 SWIR bands**:

| Band # | Name | Center Wavelength (nm) | Bandwidth (nm) | Bathymetry Use |
|--------|------|----------------------|----------------|----------------|
| **1** | **Coastal Blue** | **427** | **47 (400-454)** | **PRIMARY — penetrates water deepest** |
| 2 | Blue | 478 | 54 (451-505) | Secondary bathymetry |
| 3 | Green | 546 | 63 (515-577) | Shallow water bathymetry |
| 4 | Yellow | 608 | 42 (587-628) | Water quality correction |
| 5 | Red | 659 | 58 (630-688) | Bottom type discrimination |
| 6 | Red Edge | 724 | 47 (701-749) | Vegetation (seagrass) |
| 7 | Near-IR1 | 833 | 92 (785-881) | Land/water boundary |
| 8 | Near-IR2 | 949 | 102 (897-1005) | Atmospheric correction |

### 4.2 SWIR Bands (WorldView-3 only)

| Band # | Name | Center Wavelength (nm) | Bandwidth (nm) |
|--------|------|----------------------|----------------|
| SWIR 1 | SWIR-C | 1210 | 40 |
| SWIR 2 | SWIR-M | 1570 | 40 |
| SWIR 3 | SWIR-L | 1660 | 40 |
| SWIR 4 | SWIR-1 | 1730 | 60 |
| SWIR 5 | SWIR-2 | 2165 | 70 |
| SWIR 6 | SWIR-3 | 2205 | 50 |
| SWIR 7 | SWIR-4 | 2260 | 60 |
| SWIR 8 | SWIR-5 | 2330 | 70 |

### 4.3 Spatial Resolution

| Band Type | GSD (Ground Sample Distance) |
|-----------|------------------------------|
| Panchromatic | 31 cm |
| Multispectral (Vis+NIR) | 1.24 m |
| SWIR | 3.7 m |

### 4.4 Bathymetry-Specific Notes

**Coastal Blue band (427nm) advantages for bathymetry:**
- Maximum water penetration depth: **~20-30m in clear water** (vs ~10-15m for standard Blue)
- Reduced water surface reflectance interference
- Combined with Blue (478nm) and Green (546nm) enables **physics-based bathymetry** algorithms (e.g., Lyzenga, Stumpf, SMACC)
- The triplet Coastal Blue/Blue/Green is ideal for **multi-band depth inversion**

**Recommended band combinations for reef/bathymetry:**
- **Depth estimation:** Bands 1 (Coastal Blue), 2 (Blue), 3 (Green) — Lyzenga method
- **Bottom type classification:** Bands 1-5 (Coastal through Red)
- **Seagrass detection:** Bands 3 (Green), 6 (Red Edge), 7 (NIR)
- **Water column correction:** Bands 1, 2, 3 with SWIR bands for atmospheric correction

---

## 5. Searching for Algarve Coast Imagery (37.045N, -8.175W)

### 5.1 Via Vantor Discover Web UI
1. Go to https://discover.vantor.com
2. Create a free account
3. Search by coordinates: `37.045, -8.175`
4. Filter by:
   - Satellite: WorldView-3
   - Date range: desired period
   - Cloud cover: < 10%
   - Off-nadir angle: < 20° (for bathymetry quality)

### 5.2 Via Python SDK
```python
from maxar_platform.session import session
from maxar_platform.catalog import Catalog

# Authenticate
session.login()  # or set MAXAR_API_KEY env var

# Search for WorldView-3 imagery over Algarve coast
catalog = Catalog()

results = catalog.search(
    collections=["wv03"],
    intersects={
        "type": "Point",
        "coordinates": [-8.175, 37.045]
    },
    datetime="2023-01-01T00:00:00Z/2026-12-31T23:59:59Z",
    maxcloudcover=10,
    limit=20
)

for item in results:
    print(f"ID: {item['id']}")
    print(f"Date: {item['properties']['datetime']}")
    print(f"Cloud: {item['properties'].get('eo:cloud_cover', 'N/A')}%")
    print(f"Bands: {item['properties'].get('maxar:spectral', 'N/A')}")
    print("---")
```

### 5.3 Via REST API (curl)
```bash
# Get API key first, then search
curl -X POST "https://api.maxar.com/discovery/v1/search/stac" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "collections": ["wv03"],
    "intersects": {
      "type": "Point",
      "coordinates": [-8.175, 37.045]
    },
    "datetime": "2023-01-01T00:00:00Z/2026-12-31T23:59:59Z",
    "limit": 10,
    "query": {
      "eo:cloud_cover": {"lt": 10}
    }
  }'
```

---

## 6. Python Code Examples

### 6.1 Install and Authenticate
```python
# Install the SDK
# pip install maxar-platform

from maxar_platform.session import session
from maxar_platform.catalog import Catalog
from maxar_platform.ordering import Ordering
from maxar_platform.raster_analytics import RasterAnalytics

# Option 1: API Key via environment variable (recommended)
# export MAXAR_API_KEY="your-key"

# Option 2: Interactive login
session.login()

# Option 3: Programmatic (not recommended for production)
# session.login(username="user", password="pass")
```

### 6.2 Search and Download Archive Imagery
```python
from maxar_platform.catalog import Catalog

catalog = Catalog()

# Search WorldView-3 over Algarve coast
results = catalog.search(
    collections=["wv03"],
    intersects={
        "type": "Polygon",
        "coordinates": [[
            [-8.25, 37.00],
            [-8.10, 37.00],
            [-8.10, 37.10],
            [-8.25, 37.10],
            [-8.25, 37.00]
        ]]
    },
    datetime="2024-01-01T00:00:00Z/2025-12-31T23:59:59Z",
    maxcloudcover=5,
    limit=10
)

# Convert to GeoDataFrame for analysis
gdf = results.to_geodataframe()
print(gdf[['id', 'datetime', 'eo:cloud_cover']].to_string())

# Order the best scene
from maxar_platform.ordering import Ordering
ordering = Ordering()

# Order with band selection (Coastal Blue, Blue, Green, Red, NIR)
order = ordering.order(
    item_id=results[0]['id'],
    product_type="standard",  # or "ortho" for orthorectified
    bands=["Coastal", "Blue", "Green", "Red", "NIR1"]
)
```

### 6.3 Stream Imagery Directly (No Full Download)
```python
from maxar_platform.streaming import Streaming

streaming = Streaming()

# Get a streaming URL for a specific area
bbox = [-8.25, 37.00, -8.10, 37.10]  # minx, miny, maxx, maxy

# Stream as COG (Cloud Optimized GeoTIFF)
url = streaming.get_url(
    collection="wv03",
    bbox=bbox,
    bands=["Coastal", "Blue", "Green"]
)

# Open directly with rasterio
import rasterio
with rasterio.open(url) as src:
    data = src.read()
    profile = src.profile
```

### 6.4 Bathymetry Processing Example
```python
import numpy as np
import rasterio
from rasterio.plot import show

# Assuming you have downloaded WV3 bands:
# B1: Coastal Blue (427nm)
# B2: Blue (478nm)
# B3: Green (546nm)

with rasterio.open("wv3_algarve_B1.tif") as src:
    coastal_blue = src.read(1).astype(np.float32)
    profile = src.profile

with rasterio.open("wv3_algarve_B2.tif") as src:
    blue = src.read(1).astype(np.float32)

with rasterio.open("wv3_algarve_B3.tif") as src:
    green = src.read(1).astype(np.float32)

# Stumpf bathymetry model (ratio-based)
# Depth ∝ (ln(R_blue) / ln(R_green))
# Using Coastal Blue instead of Blue for deeper penetration

ratio = np.log(coastal_blue + 1) / np.log(green + 1)

# Lyzenga multi-band approach
# Log-transformed ratios
ln_blue = np.log(blue + 1)
ln_green = np.log(green + 1)
ln_coastal = np.log(coastal_blue + 1)

# Simple linear combination (coefficients need calibration with in-situ data)
# depth = a * ln_coastal + b * ln_blue + c * ln_green + d
# For now, use ratio as proxy
depth_proxy = ratio * 5  # rough scaling, needs calibration

# Mask land (NIR threshold)
with rasterio.open("wv3_algarve_B7.tif") as src:
    nir = src.read(1)
    water_mask = nir < 1000  # threshold for water

depth_proxy[~water_mask] = np.nan

# Save result
profile.update(count=1, dtype='float32')
with rasterio.open("algarve_bathymetry.tif", "w", **profile) as dst:
    dst.write(depth_proxy, 1)
```

---

## 7. Alternative Free/Low-Cost Options for Coastal Bathymetry

If Vantor commercial access is not feasible, consider these alternatives:

### 7.1 Sentinel-2 (ESA, Free)
- **Access:** Copernicus Data Space, Google Earth Engine, AWS
- **Bathymetry bands:** B1 (443nm), B2 (490nm), B3 (560nm) — similar to WV3 Coastal/Blue/Green but at 10-60m resolution
- **Limitation:** 10m resolution vs 1.24m (multispectral) or 31cm (panchromatic) for WV3

### 7.2 Planet SkySat (Commercial)
- **Resolution:** 50cm multispectral
- **Access:** Academic program available
- **Bands:** Blue (450-515nm), Green (515-595nm), Red (605-695nm), NIR (740-900nm)
- **Limitation:** No dedicated Coastal Blue band

### 7.3 Pléiades/Pléiades Neo (Airbus)
- **Resolution:** 50cm (Pléiades), 30cm (Neo)
- **Bands:** Blue, Green, Red, NIR (no Coastal Blue)
- **Access:** Similar commercial model, academic programs available

### 7.4 Combination Approach (Recommended for Budget)
1. Use **Sentinel-2** for large-area bathymetry mapping (free, frequent revisit)
2. Use **WV3 archive** for high-resolution calibration areas (one-time purchase)
3. Cross-calibrate WV3-derived depths with Sentinel-2 for full coverage

---

## 8. Summary: Recommended Action Plan for Algarve Reef Bathymetry

| Step | Action | Cost | Timeline |
|------|--------|------|----------|
| 1 | Create free Vantor Discover account | Free | 1 day |
| 2 | Search WV3 archive for Algarve (37.045N, -8.175W) | Free | 1 day |
| 3 | Apply for Vantor Hub subscription | ~$500-2000/yr | 1-2 weeks |
| 4 | Contact sales for academic pricing | Varies | 1-2 weeks |
| 5 | Order best archive scene (low cloud, low tide) | ~$500-1250 | 1-3 days |
| 6 | Process Coastal Blue/Blue/Green for bathymetry | N/A | Ongoing |

**Key contacts:**
- Sales: https://www.vantor.com/get-started/
- Support: support@vantor.com
- Partners: https://www.vantor.com/partner-directory/
- Open Data: opendata@vantor.com
