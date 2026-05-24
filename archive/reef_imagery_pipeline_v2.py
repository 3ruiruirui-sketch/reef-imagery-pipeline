"""
reef_imagery_pipeline_v2.py
Production v2 — Albufeira Reef Imagery Acquisition Pipeline
============================================================
Steps:
  capabilities  → probe DGT WCS, save XML
  ortho         → download DGT 2018 25 cm orthophoto clip (real)
  sentinel      → STAC discovery + real B02/B03 download via Planetary Computer
  ratio         → log(B02)/log(B03) GeoTIFF + PNG visualisation
  qgis          → write QGIS project template + QML style
  gee           → write GEE JS bonus export script
  all           → run all steps above in order

Usage:
  python reef_imagery_pipeline_v2.py --step all
  python reef_imagery_pipeline_v2.py --step capabilities
  python reef_imagery_pipeline_v2.py --step all --coverage Ortofotomapa_2018_25cm --date 2024-10-15
"""

import argparse
import logging
import os
import sys
import json
import tempfile

import requests
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import transform_bounds
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pystac_client import Client
import planetary_computer as pc
from datetime import datetime

# Load .env from the UnderWater_Visibility sibling project (shared credentials)
try:
    from dotenv import load_dotenv
    _env_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "UnderWater_Visibility", ".env.save"
    )
    load_dotenv(dotenv_path=_env_path, override=False)
except ImportError:
    pass  # python-dotenv optional

CDSE_TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE"
    "/protocol/openid-connect/token"
)
CDSE_ODATA_URL = "https://download.dataspace.copernicus.eu/odata/v1"
CDSE_SEARCH_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_LAT        = 37.069071
DEFAULT_LON        = -8.210492
DEFAULT_DATE       = "2024-10-15"
DEFAULT_BUFFER_M   = 500        # metres (approx degree conversion used for WCS)
DEFAULT_COVERAGE   = "Ortos2018-RGB"
DEFAULT_OUTPUT_DIR = "reef_output_v2"
DGT_WCS_URL        = "https://cartografia.dgterritorio.gov.pt/wcs-inspire/ortos2018"
PC_STAC_URL        = "https://planetarycomputer.microsoft.com/api/stac/v1"

OP20_DATES = [
    "2024-10-15",
    "2021-10-31",
    "2022-09-12",
    "2024-01-03",
    "2023-10-26",
]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
log = logging.getLogger("reef_pipeline")

def setup_logging(output_dir: str) -> None:
    log_path = os.path.join(output_dir, "pipeline.log")
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout),
        ],
    )
    log.info("Pipeline v2 started — log: %s", log_path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def setup_output_dir(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)


def deg_offset(buffer_m: float) -> float:
    """Rough degree offset for a given metre buffer (valid near Portugal)."""
    return buffer_m / 111_320.0


def bbox(lat: float, lon: float, buffer_m: float):
    d = deg_offset(buffer_m)
    return lon - d, lat - d, lon + d, lat + d


def is_xml_error(content: bytes) -> bool:
    """Return True if the response body looks like a WCS/OWS ServiceException."""
    snippet = content[:2000].lower()
    return b"serviceexception" in snippet or b"exceptionreport" in snippet


def validate_tiff(path: str) -> bool:
    try:
        with rasterio.open(path) as ds:
            return ds.count > 0 and ds.width > 0
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Step: capabilities
# ---------------------------------------------------------------------------
def step_capabilities(output_dir: str, **_) -> None:
    log.info("→ Probing DGT WCS capabilities …")
    url = f"{DGT_WCS_URL}?service=WCS&request=GetCapabilities&version=2.0.1"
    out = os.path.join(output_dir, "dgt_capabilities_2018.xml")
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        with open(out, "wb") as f:
            f.write(r.content)
        log.info("   Saved → %s  (%d bytes)", out, len(r.content))
        log.info("   Inspect that file to confirm the exact CoverageId for the 2018 orthophoto.")
    except Exception as exc:
        log.error("   DGT capabilities fetch failed: %s", exc)


# ---------------------------------------------------------------------------
# Step: ortho
# ---------------------------------------------------------------------------
def step_ortho(output_dir: str, lat: float, lon: float,
               buffer_m: float, coverage: str, **_) -> None:
    log.info("→ Downloading DGT 2018 25 cm orthophoto clip (coverage=%s) …", coverage)

    # DGT WCS only supports EPSG:3763 — reproject bbox from WGS-84
    from pyproj import Transformer
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3763", always_xy=True)
    d = deg_offset(buffer_m)
    min_lon_wgs, min_lat_wgs = lon - d, lat - d
    max_lon_wgs, max_lat_wgs = lon + d, lat + d
    min_e, min_n = transformer.transform(min_lon_wgs, min_lat_wgs)
    max_e, max_n = transformer.transform(max_lon_wgs, max_lat_wgs)
    log.info("   EPSG:3763 bbox: E(%.1f, %.1f)  N(%.1f, %.1f)",
             min_e, max_e, min_n, max_n)

    # WCS 2.0.1 subset axes for EPSG:3763 are E and N
    url = (
        f"{DGT_WCS_URL}?service=WCS&request=GetCoverage&version=2.0.1"
        f"&coverageId={coverage}&format=image/tiff"
        f"&subset=E,http://www.opengis.net/def/crs/EPSG/0/3763({min_e:.2f},{max_e:.2f})"
        f"&subset=N,http://www.opengis.net/def/crs/EPSG/0/3763({min_n:.2f},{max_n:.2f})"
    )
    out = os.path.join(output_dir, "dgt_ortho_2018_reef.tif")
    log.info("   Request URL: %s", url)
    try:
        r = requests.get(url, timeout=180)
        r.raise_for_status()
        if is_xml_error(r.content):
            err_path = os.path.join(output_dir, "dgt_ortho_error_response.xml")
            with open(err_path, "wb") as f:
                f.write(r.content)
            log.warning(
                "   DGT returned a ServiceException. Check %s", err_path
            )
            return
        with open(out, "wb") as f:
            f.write(r.content)
        if validate_tiff(out):
            log.info("   ✓ Valid GeoTIFF saved → %s  (%d bytes)",
                     out, os.path.getsize(out))
        else:
            log.warning("   File saved but rasterio validation failed → %s", out)
    except Exception as exc:
        log.error("   Orthophoto download failed: %s", exc)


# ---------------------------------------------------------------------------
# CDSE authentication
# ---------------------------------------------------------------------------
def _cdse_token(username: str, password: str) -> str:
    """Obtain a short-lived Bearer token from Copernicus Data Space."""
    r = requests.post(
        CDSE_TOKEN_URL,
        data={
            "client_id": "cdse-public",
            "username": username,
            "password": password,
            "grant_type": "password",
        },
        timeout=30,
    )
    r.raise_for_status()
    token = r.json().get("access_token")
    if not token:
        raise RuntimeError(f"No access_token in CDSE response: {r.text[:200]}")
    log.info("   ✓ CDSE token obtained (expires ~600 s)")
    return token


def _cdse_download_band(token: str, product_name: str, band: str,
                        out_path: str, lat: float, lon: float,
                        buffer_m: float) -> bool:
    """Download a single band window from CDSE OData using a signed token."""
    # Search for the product node path
    search_url = (
        f"{CDSE_SEARCH_URL}?$filter=Name eq '{product_name}'"
        f"&$select=Id,Name&$top=1"
    )
    r = requests.get(search_url, timeout=30)
    r.raise_for_status()
    items = r.json().get("value", [])
    if not items:
        log.warning("   CDSE: product %s not found in catalogue", product_name)
        return False
    product_id = items[0]["Id"]

    # Build the granule band path (standard S2 SAFE structure)
    # product_name already ends in .SAFE — strip before building node URL
    safe_name = product_name if product_name.endswith(".SAFE") else product_name + ".SAFE"
    node_url = (
        f"{CDSE_ODATA_URL}/Products({product_id})/Nodes"
        f"({safe_name})/Nodes(GRANULE)/Nodes"
    )
    headers = {"Authorization": f"Bearer {token}"}
    try:
        r2 = requests.get(node_url, headers=headers, timeout=30)
        r2.raise_for_status()
        granules = r2.json().get("result", r2.json().get("value", []))
        if not granules:
            log.warning("   CDSE: no granules found for %s", product_name)
            return False
        granule_name = granules[0]["Name"]

        # Resolve the 10m band JP2 URL and stream it as a COG window
        band_node = (
            f"{CDSE_ODATA_URL}/Products({product_id})/Nodes"
            f"({safe_name})/Nodes(GRANULE)/Nodes({granule_name})"
            f"/Nodes(IMG_DATA)/Nodes(R10m)/Nodes"
        )
        r3 = requests.get(band_node, headers=headers, timeout=30)
        r3.raise_for_status()
        band_files = r3.json().get("result", r3.json().get("value", []))
        band_file = next(
            (f["Name"] for f in band_files if f"_{band}_" in f["Name"]), None
        )
        if not band_file:
            log.warning("   CDSE: %s band file not found in R10m folder", band)
            return False

        download_url = (
            f"{CDSE_ODATA_URL}/Products({product_id})/Nodes"
            f"({safe_name})/Nodes(GRANULE)/Nodes({granule_name})"
            f"/Nodes(IMG_DATA)/Nodes(R10m)/Nodes({band_file})/$value"
        )
        log.info("   CDSE: streaming %s from %s", band, download_url)
        with requests.get(download_url, headers=headers,
                          stream=True, timeout=300) as dl:
            dl.raise_for_status()
            tmp = out_path + ".tmp.jp2"
            with open(tmp, "wb") as f:
                for chunk in dl.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
        # Clip window and save as GeoTIFF
        _download_cog_window(tmp, out_path, lat, lon, buffer_m)
        os.remove(tmp)
        return True
    except Exception as exc:
        log.error("   CDSE band download failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Step: sentinel
# ---------------------------------------------------------------------------
def step_sentinel(output_dir: str, lat: float, lon: float,
                  date: str, buffer_m: float,
                  cdse_user: str = "", cdse_pass: str = "", **_):
    # Resolve credentials: CLI args > env vars > .env.save
    username = cdse_user or os.environ.get("CMEMS_USERNAME", "")
    password = cdse_pass or os.environ.get("CMEMS_PASSWORD", "")

    # --- Path A: CDSE authenticated download ---
    if username and password and username != "YOUR_CMEMS_USERNAME_HERE":
        log.info("→ Sentinel-2 via CDSE OData (user=%s, date=%s) …", username, date)
        try:
            token = _cdse_token(username, password)
            _sentinel_via_cdse(token, output_dir, lat, lon, date, buffer_m)
            return
        except Exception as exc:
            log.warning("   CDSE path failed (%s) — falling back to Planetary Computer", exc)

    # --- Path B: Planetary Computer (no auth, signed COG window) ---
    log.info("→ Sentinel-2 via Planetary Computer STAC (date=%s) …", date)
    _sentinel_via_pc(output_dir, lat, lon, date, buffer_m)


def _sentinel_via_cdse(token: str, output_dir: str, lat: float, lon: float,
                       date: str, buffer_m: float) -> None:
    """Search CDSE catalogue and download B02/B03 for the reef window."""
    d = deg_offset(buffer_m)
    bbox_wgs = f"{lon-d},{lat-d},{lon+d},{lat+d}"
    search_url = (
        f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
        f"?$filter=Collection/Name eq 'SENTINEL-2'"
        f" and Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType'"
        f" and att/OData.CSC.StringAttribute/Value eq 'S2MSI2A')"
        f" and ContentDate/Start ge {date}T00:00:00.000Z"
        f" and ContentDate/Start le {date}T23:59:59.999Z"
        f" and OData.CSC.Intersects(area=geography'SRID=4326;POINT({lon} {lat})')"
        f"&$orderby=ContentDate/Start asc&$top=5&$expand=Attributes"
    )
    r = requests.get(search_url, timeout=30)
    r.raise_for_status()
    products = r.json().get("value", [])
    if not products:
        log.warning("   CDSE: no S2 L2A scene found for %s", date)
        return

    product = products[0]
    product_id   = product["Id"]
    product_name = product["Name"]
    log.info("   CDSE scene: %s", product_name)

    date_str = date.replace("-", "")
    headers  = {"Authorization": f"Bearer {token}"}

    for band in ["B02", "B03"]:
        out_path = os.path.join(output_dir, f"S2_{band}_{date_str}.tif")
        ok = _cdse_download_band(token, product_name, band,
                                 out_path, lat, lon, buffer_m)
        if ok and validate_tiff(out_path):
            log.info("   ✓ %s → %s (%d bytes)", band, out_path, os.path.getsize(out_path))
        else:
            log.warning("   %s download incomplete — check CDSE quota/access", band)

    _write_s2_meta(output_dir, date_str, date, product_name, product_id)


def _sentinel_via_pc(output_dir: str, lat: float, lon: float,
                     date: str, buffer_m: float) -> None:
    """Planetary Computer path (no credentials required)."""
    catalog = Client.open(PC_STAC_URL, modifier=pc.sign_inplace)
    time_range = f"{date}T00:00:00Z/{date}T23:59:59Z"
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        intersects={"type": "Point", "coordinates": [lon, lat]},
        datetime=time_range,
    )
    items = list(search.items())
    if not items:
        log.warning("   No scene found for %s. Try OP20 dates: %s", date, OP20_DATES)
        return

    item     = items[0]
    date_str = date.replace("-", "")
    log.info("   PC scene: %s", item.id)

    for band in ["B02", "B03"]:
        asset = item.assets.get(band)
        if not asset:
            log.warning("   Asset %s missing", band)
            continue
        out_path = os.path.join(output_dir, f"S2_{band}_{date_str}.tif")
        try:
            _download_cog_window(asset.href, out_path, lat, lon, buffer_m)
            if validate_tiff(out_path):
                log.info("   ✓ %s → %s (%d bytes)",
                         band, out_path, os.path.getsize(out_path))
        except Exception as exc:
            log.error("   %s error: %s", band, exc)

    _write_s2_meta(output_dir, date_str, date, item.id,
                   item.properties.get("s2:granule_id", ""))


def _write_s2_meta(output_dir, date_str, date, name, id_):
    meta = {"scene": name, "id": id_, "date": date, "source": "cdse/pc"}
    path = os.path.join(output_dir, f"S2_meta_{date_str}.json")
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    log.info("   Metadata → %s", path)


def _download_cog_window(href: str, out_path: str,
                         lat: float, lon: float, buffer_m: float) -> None:
    """Read a windowed region from a Cloud-Optimised GeoTIFF and save locally."""
    d = deg_offset(buffer_m)
    with rasterio.open(href) as src:
        # Convert geographic bounds to pixel window
        from rasterio.crs import CRS
        from rasterio.warp import transform_bounds
        src_crs = src.crs
        west, south, east, north = (lon - d, lat - d, lon + d, lat + d)
        # Reproject bbox to raster CRS if needed
        if src_crs and src_crs.to_epsg() != 4326:
            west, south, east, north = transform_bounds(
                "EPSG:4326", src_crs, west, south, east, north
            )
        window = rasterio.windows.from_bounds(
            west, south, east, north, transform=src.transform
        )
        data = src.read(1, window=window)
        transform = src.window_transform(window)
        profile = src.profile.copy()
        profile.update(
            height=data.shape[0],
            width=data.shape[1],
            transform=transform,
            driver="GTiff",
            compress="lzw",
        )
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(data, 1)


# ---------------------------------------------------------------------------
# Step: ratio
# ---------------------------------------------------------------------------
def step_ratio(output_dir: str, date: str, **_) -> None:
    log.info("→ Computing log(B02)/log(B03) contrast ratio …")
    date_str = date.replace("-", "")
    b02_path = os.path.join(output_dir, f"S2_B02_{date_str}.tif")
    b03_path = os.path.join(output_dir, f"S2_B03_{date_str}.tif")

    if not (os.path.exists(b02_path) and os.path.exists(b03_path)):
        log.warning(
            "   B02/B03 files not found for %s. Run --step sentinel first.", date_str
        )
        return

    ratio_path = os.path.join(output_dir, f"ratio_B02_B03_{date_str}.tif")
    plot_path  = os.path.join(output_dir, f"ratio_analysis_{date_str}.png")

    try:
        with rasterio.open(b02_path) as src_b02, rasterio.open(b03_path) as src_b03:
            b02 = src_b02.read(1).astype(np.float32)
            b03 = src_b03.read(1).astype(np.float32)
            profile = src_b02.profile.copy()

        # Avoid log(0)
        b02 = np.where(b02 > 0, b02, np.nan)
        b03 = np.where(b03 > 0, b03, np.nan)
        ratio = np.log(b02) / np.log(b03)

        profile.update(dtype=rasterio.float32, count=1, compress="lzw", nodata=np.nan)
        with rasterio.open(ratio_path, "w", **profile) as dst:
            dst.write(ratio.astype(np.float32), 1)
        log.info("   ✓ Ratio GeoTIFF → %s", ratio_path)

        # PNG visualisation
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(b02, cmap="Blues_r");  axes[0].set_title("B02 (Blue)")
        axes[1].imshow(b03, cmap="Greens_r"); axes[1].set_title("B03 (Green)")
        im = axes[2].imshow(ratio, cmap="RdYlBu_r", vmin=0.8, vmax=1.2)
        axes[2].set_title("log(B02)/log(B03) Ratio")
        plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
        for ax in axes:
            ax.axis("off")
        fig.suptitle(f"Albufeira Reef — Sentinel-2 Analysis  {date}", fontsize=13)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        log.info("   ✓ Visualisation → %s", plot_path)
    except Exception as exc:
        log.error("   Ratio step failed: %s", exc)


# ---------------------------------------------------------------------------
# Step: qgis
# ---------------------------------------------------------------------------
def step_qgis(output_dir: str, date: str, lat: float, lon: float, **_) -> None:
    log.info("→ Writing QGIS project template & QML style …")
    date_str = date.replace("-", "")

    # Minimal QGIS 3.x project file referencing the local rasters
    ratio_file = os.path.abspath(
        os.path.join(output_dir, f"ratio_B02_B03_{date_str}.tif")
    )
    ortho_file = os.path.abspath(
        os.path.join(output_dir, "dgt_ortho_2018_reef.tif")
    )

    dgt_wms = "https://cartografia.dgterritorio.gov.pt/wms/ortos2018?service=WMS"
    qgz_content = f"""<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.28" projectname="Albufeira Reef — {date}">
  <projectlayers>
    <maplayer type="raster" autoRefreshEnabled="0">
      <id>ratio_{date_str}</id>
      <datasource>{ratio_file}</datasource>
      <layername>Ratio B02/B03 {date}</layername>
      <srs><spatialrefsys><authid>EPSG:32629</authid></spatialrefsys></srs>
    </maplayer>
    <maplayer type="raster" autoRefreshEnabled="0">
      <id>ortho_2018_local</id>
      <datasource>{ortho_file}</datasource>
      <layername>DGT Ortho 2018 (local — if downloaded)</layername>
      <srs><spatialrefsys><authid>EPSG:3763</authid></spatialrefsys></srs>
    </maplayer>
    <maplayer type="raster" autoRefreshEnabled="0">
      <id>dgt_wms_bg</id>
      <datasource>crs=EPSG:3857&amp;format=image/png&amp;layers=Ortos2018-RGB&amp;styles=&amp;url={dgt_wms}</datasource>
      <layername>DGT Ortho 2018 WMS (background)</layername>
      <srs><spatialrefsys><authid>EPSG:3857</authid></spatialrefsys></srs>
    </maplayer>
  </projectlayers>
</qgis>
"""
    qgz_path = os.path.join(output_dir, f"reef_project_{date_str}.qgs")
    with open(qgz_path, "w") as f:
        f.write(qgz_content)
    log.info("   ✓ QGIS project → %s", qgz_path)

    # QML style for the ratio layer (RdYlBu ramp, 0.8–1.2)
    qml_content = """<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.28">
  <pipe>
    <rasterrenderer type="singlebandpseudocolor" band="1" opacity="1">
      <rastershader>
        <colorrampshader colorRampType="INTERPOLATED" minimumValue="0.8" maximumValue="1.2">
          <item alpha="255" value="0.8"  label="0.80" color="#d73027"/>
          <item alpha="255" value="0.9"  label="0.90" color="#fc8d59"/>
          <item alpha="255" value="1.0"  label="1.00" color="#ffffbf"/>
          <item alpha="255" value="1.1"  label="1.10" color="#91bfdb"/>
          <item alpha="255" value="1.2"  label="1.20" color="#4575b4"/>
        </colorrampshader>
      </rastershader>
    </rasterrenderer>
  </pipe>
</qgis>
"""
    qml_path = os.path.join(output_dir, "ratio_style.qml")
    with open(qml_path, "w") as f:
        f.write(qml_content)
    log.info("   ✓ QML style → %s", qml_path)


# ---------------------------------------------------------------------------
# Step: gee (bonus)
# ---------------------------------------------------------------------------
def step_gee(output_dir: str, date: str, lat: float, lon: float, **_) -> None:
    log.info("→ Writing GEE JS bonus export script …")
    script = f"""// ============================================================
// GEE Export Script — Albufeira Reef Ratio  ({date})
// Paste into https://code.earthengine.google.com/
// ============================================================

var point  = ee.Geometry.Point([{lon}, {lat}]);
var buffer = point.buffer(500);          // 500 m radius

var col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(buffer)
    .filterDate('{date}', ee.Date('{date}').advance(1, 'day'))
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
    .sort('CLOUDY_PIXEL_PERCENTAGE');

var img = col.first();
print('Scene ID:', img.get('system:index'));

var b2    = img.select('B2').toFloat();
var b3    = img.select('B3').toFloat();
var ratio = b2.log().divide(b3.log()).rename('ratio');

// Visualise
Map.centerObject(buffer, 14);
Map.addLayer(img, {{bands:['B4','B3','B2'], min:0, max:3000}}, 'RGB');
Map.addLayer(ratio.clip(buffer),
  {{min:0.8, max:1.2, palette:['d73027','fc8d59','ffffbf','91bfdb','4575b4']}},
  'log(B2)/log(B3) Ratio');

// Export to Drive
Export.image.toDrive({{
  image:       ratio.clip(buffer),
  description: 'reef_ratio_{date.replace("-","")}',
  folder:      'reef_gee_exports',
  scale:       10,
  crs:         'EPSG:32629',
  region:      buffer,
  maxPixels:   1e9
}});
print('Export task created. Run from Tasks tab.');
"""
    out = os.path.join(output_dir, f"gee_reef_export_{date.replace('-','')}.js")
    with open(out, "w") as f:
        f.write(script)
    log.info("   ✓ GEE script → %s", out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
STEPS = ["capabilities", "ortho", "sentinel", "ratio", "qgis", "gee"]

def main():
    parser = argparse.ArgumentParser(
        description="Reef Imagery Pipeline v2 — Albufeira Reef",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--step",       required=True,
                        choices=STEPS + ["all"], help="Pipeline step to run")
    parser.add_argument("--date",       default=DEFAULT_DATE,
                        help="Target date (YYYY-MM-DD)")
    parser.add_argument("--lat",        type=float, default=DEFAULT_LAT)
    parser.add_argument("--lon",        type=float, default=DEFAULT_LON)
    parser.add_argument("--buffer-m",  type=float, default=DEFAULT_BUFFER_M,
                        help="Clip radius in metres")
    parser.add_argument("--coverage",   default=DEFAULT_COVERAGE,
                        help="DGT WCS CoverageId (check dgt_capabilities_2018.xml)")
    parser.add_argument("--output-dir",  default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cdse-user",   default="",
                        help="Copernicus Data Space email (overrides CMEMS_USERNAME env var)")
    parser.add_argument("--cdse-pass",   default="",
                        help="Copernicus Data Space password (overrides CMEMS_PASSWORD env var)")
    args = parser.parse_args()

    setup_output_dir(args.output_dir)
    setup_logging(args.output_dir)

    ctx = dict(
        output_dir=args.output_dir,
        lat=args.lat,
        lon=args.lon,
        date=args.date,
        buffer_m=args.buffer_m,
        coverage=args.coverage,
        cdse_user=args.cdse_user,
        cdse_pass=args.cdse_pass,
    )

    run = [args.step] if args.step != "all" else STEPS
    step_fns = {
        "capabilities": step_capabilities,
        "ortho":        step_ortho,
        "sentinel":     step_sentinel,
        "ratio":        step_ratio,
        "qgis":         step_qgis,
        "gee":          step_gee,
    }

    for s in run:
        log.info("=" * 60)
        step_fns[s](**ctx)

    log.info("=" * 60)
    log.info("Pipeline complete. Outputs in: %s", args.output_dir)


if __name__ == "__main__":
    main()
