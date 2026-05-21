"""
reef_imagery_pipeline_v3.py
Production v3 — Albufeira Reef Imagery Acquisition Pipeline
============================================================
Hardened Pass:
  - Token-first authentication for CDSE (Copernicus Data Space).
  - Optional TOTP 2FA support for token generation.
  - Safe environment token loading (CDSE_ACCESS_TOKEN).
  - Graceful automatic fallback to Planetary Computer STAC.
  - Multi-source Orthophoto Fused Track:
      1. OrtoSat2023 30 cm Pléiades Neo WMS GetMap (Zero-Auth Direct Download!)
         - Automatically queries the DGT WMS GetMap server.
         - Downloads high-resolution 30 cm True-Color and False-Color (NIR) clips.
         - Zero authentication required, bypasses SharePoint restrictions!
      2. OrtoSat2023 SharePoint Gated Fallback:
         - Auto-detects and clips local "OrtoSat2023_4824_FalsaCor.tif" if placed in output/!
      3. DGT 2018 WCS (25 cm, WGS-84 -> EPSG:3763 clip, optional via flag)
      4. Public DGT WMS (OrtoSat2023 True-Color, False-Color, and Ortho2018 background fallbacks in QGIS).

Usage:
  # Token-based execution (recommended)
  export CDSE_ACCESS_TOKEN="your_bearer_token"
  python reef_imagery_pipeline_v3.py --step all

  # Interactive/TOTP auth pass-through
  python reef_imagery_pipeline_v3.py --step sentinel --cdse-user user@mail.com --cdse-pass pass --cdse-totp 123456

  # Forcing WCS DGT 2018 orthophoto download clip (requires entitlement/payment)
  python reef_imagery_pipeline_v3.py --step ortho --enable-dgt-download
"""

import argparse
import logging
import os
import sys
import json
import requests
import numpy as np
import rasterio
from rasterio.warp import transform_bounds
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pystac_client import Client
import planetary_computer as pc
from datetime import datetime

# Load .env variables securely if python-dotenv is present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Configuration & Constants
# ---------------------------------------------------------------------------
DEFAULT_LAT        = 37.069071
DEFAULT_LON        = -8.210492
DEFAULT_DATE       = "2024-10-15"
DEFAULT_BUFFER_M   = 500
DEFAULT_COVERAGE   = "Ortos2018-RGB"
DEFAULT_OUTPUT_DIR = "reef_output_v3"
DGT_WCS_URL        = "https://cartografia.dgterritorio.gov.pt/wcs-inspire/ortos2018"
PC_STAC_URL        = "https://planetarycomputer.microsoft.com/api/stac/v1"

CDSE_TOKEN_URL     = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
CDSE_ODATA_URL     = "https://download.dataspace.copernicus.eu/odata/v1"
CDSE_SEARCH_URL    = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"

# WMS endpoints
DGT_WMS_2018_URL       = "https://cartografia.dgterritorio.gov.pt/wms/ortos2018?service=WMS"
ORTOSAT2023_WMS_URL    = "https://ortos.dgterritorio.gov.pt/wms/ortosat2023"
ORTOSAT2023_SHAREPOINT = (
    "https://dgterritorio.sharepoint.com/sites/EXT-ORTOSAT2023"
    "/_layouts/15/download.aspx?SourceUrl=/sites/EXT-ORTOSAT2023"
    "/Documentos%20Partilhados/2_OrtoSat2023_FalsaCor"
    "/1_Seccoes_OrtoSat2023_FalsaCor/Seccoes_4800"
    "/OrtoSat2023_4824_FalsaCor.tif"
)

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
    log.info("Pipeline v3 Hardened started — log: %s", log_path)

# ---------------------------------------------------------------------------
# Coordinate & File Helpers
# ---------------------------------------------------------------------------
def setup_output_dir(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

def deg_offset(buffer_m: float) -> float:
    return buffer_m / 111_320.0

def is_xml_error(content: bytes) -> bool:
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
    except Exception as exc:
        log.error("   DGT capabilities fetch failed: %s", exc)

# ---------------------------------------------------------------------------
# Step: ortho
# ---------------------------------------------------------------------------
def step_ortho(output_dir: str, lat: float, lon: float,
               buffer_m: float, coverage: str, enable_dgt_download: bool, **_) -> None:
    
    # ── 1. OrtoSat2023 WMS direct download (Zero-Auth 30 cm Pléiades Neo) ──
    log.info("→ Querying OrtoSat2023 WMS GetMap direct clips (30 cm Pléiades Neo) …")
    
    from pyproj import Transformer
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3763", always_xy=True)
    x_pt, y_pt = transformer.transform(lon, lat)
    min_e, max_e = x_pt - buffer_m, x_pt + buffer_m
    min_n, max_n = y_pt - buffer_m, y_pt + buffer_m
    
    # Target clip resolution matching 30 cm native scale:
    # 1000m buffer width / 0.3m per pixel = 3333 pixels
    pixel_size = int((buffer_m * 2) / 0.3)
    
    # Limit max size to 3333 to ensure server compatibility
    pixel_size = min(pixel_size, 3333)
    
    layers_2023 = {
        "FalsaCor":      "ortoSat2023-FalsaCor",
        "CorVerdadeira": "ortoSat2023-CorVerdadeira"
    }
    
    for key, layer_name in layers_2023.items():
        out_wms_clip = os.path.join(output_dir, f"OrtoSat2023_reef_clip_{key}.tif")
        log.info("   Downloading 2023 %s (30 cm resolution, %dx%d pixels) ...", key, pixel_size, pixel_size)
        
        params = {
            "service": "WMS",
            "version": "1.3.0",
            "request": "GetMap",
            "layers": layer_name,
            "styles": "",
            "crs": "EPSG:3763",
            "bbox": f"{min_e},{min_n},{max_e},{max_n}",
            "width": str(pixel_size),
            "height": str(pixel_size),
            "format": "image/tiff"
        }
        
        try:
            r = requests.get(ORTOSAT2023_WMS_URL, params=params, timeout=120)
            r.raise_for_status()
            if "image" in r.headers.get("Content-Type", "") and not is_xml_error(r.content):
                with open(out_wms_clip, "wb") as f:
                    f.write(r.content)
                if validate_tiff(out_wms_clip):
                    log.info("   ✓ Saved 2023 %s WMS clip → %s (%.1f MB)", key, out_wms_clip, os.path.getsize(out_wms_clip)/1e6)
                else:
                    log.warning("   2023 %s WMS download failed validation.", key)
            else:
                log.warning("   Server returned non-image content: %s", r.content[:500])
        except Exception as exc:
            log.error("   Failed to download 2023 %s direct WMS clip: %s", key, exc)

    # ── 1b. Orto2018 WMS direct download (Zero-Auth 25 cm Aerial) ──
    log.info("→ Querying Orto2018 WMS GetMap direct clips (25 cm Aerial) …")
    layers_2018 = {
        "FalsaCor":      "Ortos2018-IRG",
        "CorVerdadeira": "Ortos2018-RGB"
    }
    ortos2018_wms_url = "https://cartografia.dgterritorio.gov.pt/wms/ortos2018"

    for key, layer_name in layers_2018.items():
        out_wms_clip = os.path.join(output_dir, f"Orto2018_reef_clip_{key}.tif")
        # Calc pixel size for 25 cm native scale
        pixel_size_2018 = int((buffer_m * 2) / 0.25)
        pixel_size_2018 = min(pixel_size_2018, 3333)
        log.info("   Downloading 2018 %s (25 cm resolution, %dx%d pixels) ...", key, pixel_size_2018, pixel_size_2018)
        
        params = {
            "service": "WMS",
            "version": "1.3.0",
            "request": "GetMap",
            "layers": layer_name,
            "styles": "",
            "crs": "EPSG:3763",
            "bbox": f"{min_e},{min_n},{max_e},{max_n}",
            "width": str(pixel_size_2018),
            "height": str(pixel_size_2018),
            "format": "image/tiff"
        }
        
        try:
            r = requests.get(ortos2018_wms_url, params=params, timeout=120)
            r.raise_for_status()
            if "image" in r.headers.get("Content-Type", "") and not is_xml_error(r.content):
                with open(out_wms_clip, "wb") as f:
                    f.write(r.content)
                if validate_tiff(out_wms_clip):
                    log.info("   ✓ Saved 2018 %s WMS clip → %s (%.1f MB)", key, out_wms_clip, os.path.getsize(out_wms_clip)/1e6)
                else:
                    log.warning("   2018 %s WMS download failed validation.", key)
            else:
                log.warning("   Server returned non-image content: %s", r.content[:500])
        except Exception as exc:
            log.error("   Failed to download 2018 %s direct WMS clip: %s", key, exc)

    # ── 2. OrtoSat2023 SharePoint Gated Fallback (Direct Manual Download) ──
    local_source = os.path.join(output_dir, "OrtoSat2023_4824_FalsaCor.tif")
    local_clip   = os.path.join(output_dir, "OrtoSat2023_4824_FalsaCor_reef_clip.tif")

    if os.path.exists(local_source):
        log.info("   ✓ Found manual SharePoint download of OrtoSat2023 Sec 4824 at %s", local_source)
        try:
            log.info("   Clipping manual OrtoSat2023 high-res raster to reef buffer window...")
            _download_cog_window(local_source, local_clip, lat, lon, buffer_m)
            if validate_tiff(local_clip):
                log.info("   ✓ Manual OrtoSat2023 clip generated → %s (%d bytes)", local_clip, os.path.getsize(local_clip))
        except Exception as e:
            log.error("   Failed to clip manual OrtoSat2023 raster: %s", e)

    # ── 3. DGT 2018 WCS (Fallback WCS Clip) ──
    if not enable_dgt_download:
        log.info("→ DGT WCS 2018 download skipped (enable with --enable-dgt-download).")
        log.info("   Using default WMS visualization backgrounds instead in the QGIS template.")
        return

    log.info("→ Downloading DGT 2018 25 cm orthophoto clip (coverage=%s) …", coverage)
    url = (
        f"{DGT_WCS_URL}?service=WCS&request=GetCoverage&version=2.0.1"
        f"&coverageId={coverage}&format=image/tiff"
        f"&subset=E,http://www.opengis.net/def/crs/EPSG/0/3763({min_e:.2f},{max_e:.2f})"
        f"&subset=N,http://www.opengis.net/def/crs/EPSG/0/3763({min_n:.2f},{max_n:.2f})"
    )
    out = os.path.join(output_dir, "dgt_ortho_2018_reef.tif")
    try:
        r = requests.get(url, timeout=180)
        r.raise_for_status()
        if is_xml_error(r.content):
            err_path = os.path.join(output_dir, "dgt_ortho_error_response.xml")
            with open(err_path, "wb") as f:
                f.write(r.content)
            log.warning("   DGT returned a ServiceException. Check %s", err_path)
            return
        with open(out, "wb") as f:
            f.write(r.content)
        if validate_tiff(out):
            log.info("   ✓ Valid GeoTIFF saved → %s  (%d bytes)", out, os.path.getsize(out))
    except Exception as exc:
        log.error("   Orthophoto download failed: %s", exc)

# ---------------------------------------------------------------------------
# CDSE Token Generation / Validation
# ---------------------------------------------------------------------------
def _get_cdse_token(username: str = "", password: str = "", totp: str = "") -> str:
    """Fetch short-lived Bearer token from Copernicus Data Space (supporting optional TOTP)."""
    payload = {
        "client_id": "cdse-public",
        "username": username,
        "password": password,
        "grant_type": "password",
    }
    if totp:
        payload["totp"] = totp

    log.info("   Authenticating with CDSE oauth endpoint ...")
    r = requests.post(CDSE_TOKEN_URL, data=payload, timeout=30)
    r.raise_for_status()
    token = r.json().get("access_token")
    if not token:
        raise RuntimeError(f"No access_token found in response: {r.text[:200]}")
    log.info("   ✓ CDSE token successfully generated via credentials")
    return token

def _cdse_download_band(token: str, product_name: str, band: str,
                        out_path: str, lat: float, lon: float,
                        buffer_m: float) -> bool:
    """Download a single band window from CDSE OData using a Bearer token."""
    search_url = f"{CDSE_SEARCH_URL}?$filter=Name eq '{product_name}'&$select=Id,Name&$top=1"
    r = requests.get(search_url, timeout=30)
    r.raise_for_status()
    items = r.json().get("value", [])
    if not items:
        log.warning("   CDSE: product %s not found in catalogue", product_name)
        return False
    product_id = items[0]["Id"]

    safe_name = product_name if product_name.endswith(".SAFE") else product_name + ".SAFE"
    node_url = f"{CDSE_ODATA_URL}/Products({product_id})/Nodes({safe_name})/Nodes(GRANULE)/Nodes"
    headers = {"Authorization": f"Bearer {token}"}
    try:
        r2 = requests.get(node_url, headers=headers, timeout=30)
        r2.raise_for_status()
        granules = r2.json().get("result", r2.json().get("value", []))
        if not granules:
            log.warning("   CDSE: no granules found for %s", product_name)
            return False
        granule_name = granules[0]["Name"]

        band_node = (
            f"{CDSE_ODATA_URL}/Products({product_id})/Nodes"
            f"({safe_name})/Nodes(GRANULE)/Nodes({granule_name})"
            f"/Nodes(IMG_DATA)/Nodes(R10m)/Nodes"
        )
        r3 = requests.get(band_node, headers=headers, timeout=30)
        r3.raise_for_status()
        band_files = r3.json().get("result", r3.json().get("value", []))
        band_file = next((f["Name"] for f in band_files if f"_{band}_" in f["Name"]), None)
        if not band_file:
            log.warning("   CDSE: %s band file not found in R10m folder", band)
            return False

        download_url = (
            f"{CDSE_ODATA_URL}/Products({product_id})/Nodes"
            f"({safe_name})/Nodes(GRANULE)/Nodes({granule_name})"
            f"/Nodes(IMG_DATA)/Nodes(R10m)/Nodes({band_file})/$value"
        )
        log.info("   CDSE: streaming %s from %s", band, download_url)
        with requests.get(download_url, headers=headers, stream=True, timeout=300) as dl:
            dl.raise_for_status()
            tmp = out_path + ".tmp.jp2"
            with open(tmp, "wb") as f:
                for chunk in dl.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
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
                  cdse_token_str: str = "", cdse_user: str = "",
                  cdse_pass: str = "", cdse_totp: str = "", **_):
    token = cdse_token_str or os.environ.get("CDSE_ACCESS_TOKEN", "")

    # If no token is provided but credentials are, try generating a token
    if not token and cdse_user and cdse_pass:
        log.info("→ Attempting CDSE token generation...")
        try:
            token = _get_cdse_token(cdse_user, cdse_pass, cdse_totp)
        except Exception as exc:
            log.warning("   Credentials token generation failed: %s", exc)

    # --- Path A: CDSE with validated Token ---
    if token:
        log.info("→ Sentinel-2 via CDSE OData (Token Auth, date=%s) …", date)
        try:
            _sentinel_via_cdse(token, output_dir, lat, lon, date, buffer_m)
            return
        except Exception as exc:
            log.warning("   CDSE Token processing failed (%s) — falling back to Planetary Computer", exc)

    # --- Path B: Planetary Computer STAC Fallback ---
    log.info("→ Sentinel-2 via Planetary Computer STAC (date=%s) …", date)
    _sentinel_via_pc(output_dir, lat, lon, date, buffer_m)

def _sentinel_via_cdse(token: str, output_dir: str, lat: float, lon: float,
                       date: str, buffer_m: float) -> None:
    d = deg_offset(buffer_m)
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
        raise RuntimeError("No scene found in CDSE catalog")

    product = products[0]
    product_id   = product["Id"]
    product_name = product["Name"]
    log.info("   CDSE scene: %s", product_name)

    date_str = date.replace("-", "")
    for band in ["B02", "B03"]:
        out_path = os.path.join(output_dir, f"S2_{band}_{date_str}.tif")
        ok = _cdse_download_band(token, product_name, band, out_path, lat, lon, buffer_m)
        if ok and validate_tiff(out_path):
            log.info("   ✓ %s → %s (%d bytes)", band, out_path, os.path.getsize(out_path))
        else:
            raise RuntimeError(f"CDSE download of band {band} failed")

    _write_s2_meta(output_dir, date_str, date, product_name, product_id)

def _sentinel_via_pc(output_dir: str, lat: float, lon: float,
                     date: str, buffer_m: float) -> None:
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
                log.info("   ✓ %s → %s (%d bytes)", band, out_path, os.path.getsize(out_path))
        except Exception as exc:
            log.error("   %s error: %s", band, exc)

    _write_s2_meta(output_dir, date_str, date, item.id, item.properties.get("s2:granule_id", ""))

def _write_s2_meta(output_dir, date_str, date, name, id_):
    meta = {"scene": name, "id": id_, "date": date, "source": "cdse/pc"}
    path = os.path.join(output_dir, f"S2_meta_{date_str}.json")
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    log.info("   Metadata → %s", path)

def _download_cog_window(href: str, out_path: str,
                         lat: float, lon: float, buffer_m: float) -> None:
    d = deg_offset(buffer_m)
    with rasterio.open(href) as src:
        src_crs = src.crs
        west, south, east, north = (lon - d, lat - d, lon + d, lat + d)
        if src_crs and src_crs.to_epsg() != 4326:
            from rasterio.warp import transform_bounds
            west, south, east, north = transform_bounds("EPSG:4326", src_crs, west, south, east, north)
        window = rasterio.windows.from_bounds(west, south, east, north, transform=src.transform)
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
        log.warning("   B02/B03 files not found for %s. Run --step sentinel first.", date_str)
        return

    ratio_path = os.path.join(output_dir, f"ratio_B02_B03_{date_str}.tif")
    plot_path  = os.path.join(output_dir, f"ratio_analysis_{date_str}.png")

    try:
        with rasterio.open(b02_path) as src_b02, rasterio.open(b03_path) as src_b03:
            b02 = src_b02.read(1).astype(np.float32)
            b03 = src_b03.read(1).astype(np.float32)
            profile = src_b02.profile.copy()

        b02 = np.where(b02 > 0, b02, np.nan)
        b03 = np.where(b03 > 0, b03, np.nan)
        ratio = np.log(b02) / np.log(b03)

        profile.update(dtype=rasterio.float32, count=1, compress="lzw", nodata=np.nan)
        with rasterio.open(ratio_path, "w", **profile) as dst:
            dst.write(ratio.astype(np.float32), 1)
        log.info("   ✓ Ratio GeoTIFF → %s", ratio_path)

        # Plot output (using professional 2-98% dynamic percentile stretching!)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        b02_min, b02_max = np.nanpercentile(b02, 2), np.nanpercentile(b02, 98)
        b03_min, b03_max = np.nanpercentile(b03, 2), np.nanpercentile(b03, 98)
        r_min, r_max = np.nanpercentile(ratio, 2), np.nanpercentile(ratio, 98)
        
        axes[0].imshow(b02, cmap="Blues_r", vmin=b02_min, vmax=b02_max);  axes[0].set_title("B02 (Blue)")
        axes[1].imshow(b03, cmap="Greens_r", vmin=b03_min, vmax=b03_max); axes[1].set_title("B03 (Green)")
        im = axes[2].imshow(ratio, cmap="RdYlBu_r", vmin=r_min, vmax=r_max)
        axes[2].set_title("log(B02)/log(B03) Bathymetry")
        plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
        for ax in axes:
            ax.axis("off")
        fig.suptitle(f"Albufeira Reef — Sentinel-2 Dynamic Analysis  {date}", fontsize=13)
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

    ratio_file     = os.path.abspath(os.path.join(output_dir, f"ratio_B02_B03_{date_str}.tif"))
    ortho_file     = os.path.abspath(os.path.join(output_dir, "dgt_ortho_2018_reef.tif"))
    ortosat_fc     = os.path.abspath(os.path.join(output_dir, "OrtoSat2023_reef_clip_FalsaCor.tif"))
    ortosat_rgb    = os.path.abspath(os.path.join(output_dir, "OrtoSat2023_reef_clip_CorVerdadeira.tif"))
    orto2018_fc    = os.path.abspath(os.path.join(output_dir, "Orto2018_reef_clip_FalsaCor.tif"))
    orto2018_rgb   = os.path.abspath(os.path.join(output_dir, "Orto2018_reef_clip_CorVerdadeira.tif"))
    manual_clip    = os.path.abspath(os.path.join(output_dir, "OrtoSat2023_4824_FalsaCor_reef_clip.tif"))

    dgt_wms      = "https://cartografia.dgterritorio.gov.pt/wms/ortos2018?service=WMS"
    ortosat_wms  = "https://ortos.dgterritorio.gov.pt/wms/ortosat2023?service=WMS"

    qgz_content = f"""<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.28" projectname="Albufeira Reef — {date}">
  <projectlayers>
    <!-- Sentinel-2 log ratio -->
    <maplayer type="raster" autoRefreshEnabled="0">
      <id>ratio_{date_str}</id>
      <datasource>{ratio_file}</datasource>
      <layername>Ratio B02/B03 {date}</layername>
      <srs><spatialrefsys><authid>EPSG:32629</authid></spatialrefsys></srs>
    </maplayer>

    <!-- Local WMS OrtoSat2023 False Color Clip (Pleiades Neo 30 cm) -->
    <maplayer type="raster" autoRefreshEnabled="0">
      <id>ortosat_2023_wms_fc_clip</id>
      <datasource>{ortosat_fc}</datasource>
      <layername>OrtoSat 2023 FalsaCor (Local 30cm WMS Clip)</layername>
      <srs><spatialrefsys><authid>EPSG:3763</authid></spatialrefsys></srs>
    </maplayer>

    <!-- Local WMS OrtoSat2023 True Color Clip (Pleiades Neo 30 cm) -->
    <maplayer type="raster" autoRefreshEnabled="0">
      <id>ortosat_2023_wms_rgb_clip</id>
      <datasource>{ortosat_rgb}</datasource>
      <layername>OrtoSat 2023 CorVerdadeira (Local 30cm WMS Clip)</layername>
      <srs><spatialrefsys><authid>EPSG:3763</authid></spatialrefsys></srs>
    </maplayer>

    <!-- Local WMS Orto2018 False Color Clip (Aerial 25 cm) -->
    <maplayer type="raster" autoRefreshEnabled="0">
      <id>orto_2018_wms_fc_clip</id>
      <datasource>{orto2018_fc}</datasource>
      <layername>Orto 2018 FalsaCor (Local 25cm WMS Clip)</layername>
      <srs><spatialrefsys><authid>EPSG:3763</authid></spatialrefsys></srs>
    </maplayer>

    <!-- Local WMS Orto2018 True Color Clip (Aerial 25 cm) -->
    <maplayer type="raster" autoRefreshEnabled="0">
      <id>orto_2018_wms_rgb_clip</id>
      <datasource>{orto2018_rgb}</datasource>
      <layername>Orto 2018 CorVerdadeira (Local 25cm WMS Clip)</layername>
      <srs><spatialrefsys><authid>EPSG:3763</authid></spatialrefsys></srs>
    </maplayer>

    <!-- Local manual OrtoSat2023 False Color Clip (Pleiades Neo 30 cm) -->
    <maplayer type="raster" autoRefreshEnabled="0">
      <id>ortosat_2023_manual_clip</id>
      <datasource>{manual_clip}</datasource>
      <layername>OrtoSat 2023 Sec 4824 FalsaCor (Manual SharePoint Clip)</layername>
      <srs><spatialrefsys><authid>EPSG:3763</authid></spatialrefsys></srs>
    </maplayer>

    <!-- Local DGT 2018 WCS aerial clip -->
    <maplayer type="raster" autoRefreshEnabled="0">
      <id>ortho_2018_local</id>
      <datasource>{ortho_file}</datasource>
      <layername>DGT Ortho 2018 (Local 25cm WCS Clip)</layername>
      <srs><spatialrefsys><authid>EPSG:3763</authid></spatialrefsys></srs>
    </maplayer>

    <!-- OrtoSat2023 True Color WMS background -->
    <maplayer type="raster" autoRefreshEnabled="0">
      <id>ortosat_2023_wms_rgb</id>
      <datasource>crs=EPSG:3857&amp;format=image/png&amp;layers=ortoSat2023-CorVerdadeira&amp;styles=&amp;url={ortosat_wms}</datasource>
      <layername>OrtoSat 2023 True Color WMS (30cm Pleiades Neo Background)</layername>
      <srs><spatialrefsys><authid>EPSG:3857</authid></spatialrefsys></srs>
    </maplayer>

    <!-- OrtoSat2023 False Color WMS background -->
    <maplayer type="raster" autoRefreshEnabled="0">
      <id>ortosat_2023_wms_fc</id>
      <datasource>crs=EPSG:3857&amp;format=image/png&amp;layers=ortoSat2023-FalsaCor&amp;styles=&amp;url={ortosat_wms}</datasource>
      <layername>OrtoSat 2023 False Color WMS (30cm Pleiades Neo Background)</layername>
      <srs><spatialrefsys><authid>EPSG:3857</authid></spatialrefsys></srs>
    </maplayer>

    <!-- DGT 2018 Ortho WMS background -->
    <maplayer type="raster" autoRefreshEnabled="0">
      <id>dgt_wms_bg</id>
      <datasource>crs=EPSG:3857&amp;format=image/png&amp;layers=Ortos2018-RGB&amp;styles=&amp;url={dgt_wms}</datasource>
      <layername>DGT Ortho 2018 WMS (Background)</layername>
      <srs><spatialrefsys><authid>EPSG:3857</authid></spatialrefsys></srs>
    </maplayer>
  </projectlayers>
</qgis>
"""
    qgz_path = os.path.join(output_dir, f"reef_project_{date_str}.qgs")
    with open(qgz_path, "w") as f:
        f.write(qgz_content)
    log.info("   ✓ QGIS project → %s", qgz_path)

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
# Step: gee
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
        description="Reef Imagery Pipeline v3 Hardened — Albufeira Reef",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--step",       required=True, choices=STEPS + ["all"], help="Pipeline step to run")
    parser.add_argument("--date",       default=DEFAULT_DATE, help="Target date (YYYY-MM-DD)")
    parser.add_argument("--lat",        type=float, default=DEFAULT_LAT)
    parser.add_argument("--lon",        type=float, default=DEFAULT_LON)
    parser.add_argument("--buffer-m",  type=float, default=DEFAULT_BUFFER_M, help="Clip radius in metres")
    parser.add_argument("--coverage",   default=DEFAULT_COVERAGE, help="DGT WCS CoverageId")
    parser.add_argument("--output-dir",  default=DEFAULT_OUTPUT_DIR)

    # Hardened auth parameters
    parser.add_argument("--cdse-token",  default="", help="Copernicus Data Space Bearer Token (overrides env CDSE_ACCESS_TOKEN)")
    parser.add_argument("--cdse-user",   default="", help="Copernicus Data Space Username (if generating token)")
    parser.add_argument("--cdse-pass",   default="", help="Copernicus Data Space Password (if generating token)")
    parser.add_argument("--cdse-totp",   default="", help="Copernicus Data Space 2FA TOTP code (if generating token)")
    parser.add_argument("--enable-dgt-download", action="store_true", help="Force WCS download of the DGT 2018 orthophoto")

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
        cdse_token_str=args.cdse_token,
        cdse_user=args.cdse_user,
        cdse_pass=args.cdse_pass,
        cdse_totp=args.cdse_totp,
        enable_dgt_download=args.enable_dgt_download
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
