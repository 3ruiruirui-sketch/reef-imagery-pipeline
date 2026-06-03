#!/usr/bin/env python3
"""
fetch_sentinel1_sar.py — Sentinel-1 SAR Surface Roughness for Algarve Reefs
===============================================================================
Searches the Copernicus Data Space Ecosystem (CDSE) OData catalog for Sentinel-1
Ground Range Detected (GRD) IW products over the Algarve coast and extracts
surface roughness proxies for each reef site.

What this script produces:
  - A ranked list of available S1 GRD scenes for the target year and season
  - VV/VH backscatter sigma0 values at each reef site (extracted via STAC)
  - Surface roughness proxy: roughness = log10(σ0_VV / σ0_VH)
    • Low roughness (≈0) → calm sea surface → good visibility window
    • High roughness (>0.3) → wind-roughened surface → poor visibility
  - s1_roughness_report.json in the project output directory

Data sources:
  - Primary:  Copernicus CDSE OData API (free, no auth for catalog queries)
              https://catalogue.dataspace.copernicus.eu/odata/v1/Products
  - Download: Requires CDSE free account for actual file access
              https://dataspace.copernicus.eu
  - Fallback: Element84 Earth Search STAC (Sentinel-1 IW GRD via STAC)
              https://earth-search.aws.element84.com/v1

References:
  - Sentinel-1 SAR Backscatter Analysis Ready Data (Truckenbrodt et al. 2019)
  - Algarve sea surface state: CMEMS NWSHELF_ANALYSIS_FORECAST_PHY_004_013
  - ESA Sentinel-1 Product Definition: https://sentinel.esa.int/documents/247904/1877131/
"""

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

# ── Bootstrap project path ────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from scratch.generate_enhanced_v2 import SITES

# ── CDSE OData Catalog API ────────────────────────────────────────────────────
CDSE_ODATA_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"

# ── Element84 Earth Search (STAC fallback for public S1 data) ─────────────────
E84_STAC_URL = "https://earth-search.aws.element84.com/v1"
E84_S1_COLLECTION = "sentinel-1-grd"

# ── Algarve bounding box (WGS84) ─────────────────────────────────────────────
ALGARVE_BBOX = {
    "min_lon": -9.0,
    "min_lat": 36.9,
    "max_lon": -7.5,
    "max_lat": 37.5,
}

# Physical constants for roughness interpretation
ROUGHNESS_CALM_THRESHOLD  = 0.05   # below → calm, good diving conditions
ROUGHNESS_ROUGH_THRESHOLD = 0.25   # above → rough sea, poor visibility


def parse_args():
    p = argparse.ArgumentParser(
        description="Sentinel-1 SAR surface roughness proxy for Algarve reef sites",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--year", type=int, default=2025,
        help="Target year for S1 scene search"
    )
    p.add_argument(
        "--month-start", type=int, default=7, metavar="MONTH",
        help="Start month of search window (1-12)"
    )
    p.add_argument(
        "--month-end", type=int, default=9, metavar="MONTH",
        help="End month of search window (1-12)"
    )
    p.add_argument(
        "--sites", nargs="+", choices=list(SITES.keys()), default=list(SITES.keys()),
        help="Reef sites to analyse"
    )
    p.add_argument(
        "--max-results", type=int, default=10,
        help="Maximum number of S1 scenes to retrieve from catalog"
    )
    p.add_argument(
        "--out-dir", default=None,
        help="Output directory for s1_roughness_report.json (default: project root)"
    )
    return p.parse_args()


def search_cdse_s1_catalog(
    min_lon: float, min_lat: float, max_lon: float, max_lat: float,
    year: int, month_start: int, month_end: int,
    max_results: int = 10,
) -> list[dict]:
    """
    Query the CDSE OData API for Sentinel-1 GRD IW products intersecting
    the Algarve bounding box in the target time window.
    """
    # Build WKT polygon for bbox filter
    poly_wkt = (
        f"SRID=4326;POLYGON(("
        f"{min_lon} {min_lat},"
        f"{max_lon} {min_lat},"
        f"{max_lon} {max_lat},"
        f"{min_lon} {max_lat},"
        f"{min_lon} {min_lat}))"
    )

    # Pad month to 2 digits
    start_date = f"{year}-{month_start:02d}-01T00:00:00.000Z"
    last_day = 31 if month_end in [1, 3, 5, 7, 8, 10, 12] else 30
    end_date = f"{year}-{month_end:02d}-{last_day}T23:59:59.999Z"

    query_filter = (
        "Collection/Name eq 'SENTINEL-1' and "
        "Attributes/OData.CSC.StringAttribute/any(att:"
        "  att/Name eq 'productType' and att/Value eq 'GRD') and "
        "ContentDate/Start ge " + start_date + " and "
        "ContentDate/Start le " + end_date + " and "
        f"OData.CSC.Intersects(area=geography'{poly_wkt}')"
    )

    params = {
        "$filter": query_filter,
        "$top": max_results,
        "$orderby": "ContentDate/Start desc",
        "$expand": "Attributes",
        "$select": "Id,Name,ContentDate,Footprint,Attributes",
    }

    print(f"📡 Querying CDSE for Sentinel-1 GRD scenes "
          f"({year}-{month_start:02d} to {year}-{month_end:02d})...")

    try:
        resp = requests.get(CDSE_ODATA_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        products = data.get("value", [])
        print(f"   ✅ Found {len(products)} Sentinel-1 GRD scenes in CDSE catalog")
        return products
    except requests.RequestException as exc:
        print(f"   ❌ CDSE query failed: {exc}")
        return []


def search_stac_s1_scenes(
    lon: float, lat: float,
    year: int, month_start: int, month_end: int,
    max_results: int = 5,
) -> list[dict]:
    """
    Fallback: search Element84 Earth Search STAC for Sentinel-1 IW GRD scenes.
    Returns simplified item dicts with id, date, and asset hrefs.
    """
    try:
        from pystac_client import Client
    except ImportError:
        print("   ⚠️  pystac-client not installed — skipping STAC fallback")
        return []

    try:
        catalog = Client.open(E84_STAC_URL)
        start_dt = f"{year}-{month_start:02d}-01T00:00:00Z"
        end_dt = f"{year}-{month_end:02d}-28T23:59:59Z"  # conservative end
        search = catalog.search(
            collections=[E84_S1_COLLECTION],
            intersects={"type": "Point", "coordinates": [lon, lat]},
            datetime=f"{start_dt}/{end_dt}",
            max_items=max_results,
        )
        items = list(search.items())
        print(f"   ✅ STAC fallback: found {len(items)} S1 scenes at ({lon:.4f}, {lat:.4f})")
        return [
            {
                "id": it.id,
                "date": it.datetime.strftime("%Y-%m-%d") if it.datetime else "unknown",
                "assets": {k: v.href for k, v in it.assets.items()},
                "properties": dict(it.properties),
            }
            for it in items
        ]
    except Exception as exc:
        print(f"   ⚠️  STAC search failed: {exc}")
        return []


def roughness_from_sigma0(sigma0_vv: float, sigma0_vh: float) -> dict:
    """
    Compute surface roughness proxy from dual-polarisation sigma0 values.

    Cross-ratio roughness (Ulaby et al. 1986):
        roughness = log10(VV / VH)

    Physical interpretation:
        - VV: sensitive to large-scale wave slopes (Bragg scattering)
        - VH: mostly volumetric/multiple scattering (wind cells, foam)
        - Low VV/VH → calm specular surface → good diving visibility window
        - High VV/VH → rough surface → wind or swell present

    Returns roughness value and qualitative sea-state classification.
    """
    if sigma0_vv <= 0 or sigma0_vh <= 0:
        return {"roughness": None, "sea_state": "unknown", "vv": sigma0_vv, "vh": sigma0_vh}

    roughness = math.log10(sigma0_vv / sigma0_vh)

    if roughness < ROUGHNESS_CALM_THRESHOLD:
        sea_state = "calm"      # Beaufort ≤ 2
    elif roughness < ROUGHNESS_ROUGH_THRESHOLD:
        sea_state = "moderate"  # Beaufort 3-4
    else:
        sea_state = "rough"     # Beaufort ≥ 5

    return {
        "roughness": round(roughness, 4),
        "sea_state": sea_state,
        "vv_linear": round(sigma0_vv, 6),
        "vh_linear": round(sigma0_vh, 6),
        "vv_db": round(10 * math.log10(sigma0_vv), 2),
        "vh_db": round(10 * math.log10(sigma0_vh), 2),
    }


def extract_sigma0_at_point(item: dict, lon: float, lat: float) -> dict | None:
    """
    Extract approximate VV/VH sigma0 values at a geographic point from a STAC item.
    This requires the GRD COG to be publicly accessible (Element84 / AWS Open Data).

    Returns dict with vv, vh in linear scale, or None if extraction fails.
    """
    try:
        import rasterio
        import numpy as np
        from pyproj import Transformer

        assets = item.get("assets", {})

        # Common asset name patterns for S1 GRD in E84 STAC
        vv_href = (
            assets.get("vv") or assets.get("VV") or assets.get("measurement_vv")
        )
        vh_href = (
            assets.get("vh") or assets.get("VH") or assets.get("measurement_vh")
        )

        if not vv_href or not vh_href:
            return None

        env = rasterio.Env(AWS_NO_SIGN_REQUEST="YES", GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR")

        results = {}
        for pol, href in [("vv", vv_href), ("vh", vh_href)]:
            with env:
                with rasterio.open(href) as src:
                    tf = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
                    x, y = tf.transform(lon, lat)
                    # Sample a 3×3 pixel window for robustness
                    row, col = src.index(x, y)
                    window = rasterio.windows.Window(col - 1, row - 1, 3, 3)
                    data = src.read(1, window=window).astype(np.float32)
                    data_pos = data[data > 0]
                    if len(data_pos) == 0:
                        results[pol] = None
                    else:
                        # S1 L1 GRD pixel values are amplitude; power = amplitude²
                        amplitude_mean = float(np.mean(data_pos))
                        sigma0_linear = (amplitude_mean ** 2) / 1e8  # normalize
                        results[pol] = sigma0_linear

        if results.get("vv") and results.get("vh"):
            return results
        return None

    except Exception as exc:
        print(f"      ⚠️  Sigma0 extraction failed: {exc}")
        return None


def main():
    args = parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else _PROJECT_ROOT

    print("\n" + "═" * 70)
    print("  📡 SENTINEL-1 SAR Surface Roughness — Algarve Reefs")
    print(f"     Year: {args.year} | Window: {args.month_start:02d}–{args.month_end:02d}")
    print(f"     Sites: {', '.join(args.sites)}")
    print("═" * 70)

    # Step 1: Search CDSE catalog for available scenes
    products = search_cdse_s1_catalog(
        min_lon=ALGARVE_BBOX["min_lon"],
        min_lat=ALGARVE_BBOX["min_lat"],
        max_lon=ALGARVE_BBOX["max_lon"],
        max_lat=ALGARVE_BBOX["max_lat"],
        year=args.year,
        month_start=args.month_start,
        month_end=args.month_end,
        max_results=args.max_results,
    )

    # Build scene catalog summary
    scene_catalog = []
    for p in products:
        scene_name = p.get("Name", "")
        start_date = p.get("ContentDate", {}).get("Start", "")[:10]
        prod_id = p.get("Id", "")

        # Extract orbit mode from name (IW = Interferometric Wide Swath)
        mode = "IW" if "_IW_" in scene_name else "SM" if "_SM_" in scene_name else "EW"

        scene_catalog.append({
            "id": prod_id,
            "name": scene_name,
            "date": start_date,
            "mode": mode,
            "cdse_download_url": f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products({prod_id})/$value",
            "note": "Requires free CDSE account: https://dataspace.copernicus.eu/",
        })

    print(f"\n   📋 Catalog summary ({len(scene_catalog)} scenes):")
    for s in scene_catalog[:5]:
        print(f"      {s['date']}  {s['name'][:60]}")
    if len(scene_catalog) > 5:
        print(f"      ... and {len(scene_catalog)-5} more")

    # Step 2: Per-site STAC sigma0 extraction (Element84 public data)
    site_results = []

    for site_key in args.sites:
        site = SITES[site_key]
        lon, lat = site["lon"], site["lat"]
        print(f"\n🌊 {site['label']} ({lon:.5f}°E, {lat:.5f}°N)")

        # Try Element84 STAC for public S1 COG data
        stac_items = search_stac_s1_scenes(
            lon=lon, lat=lat,
            year=args.year,
            month_start=args.month_start,
            month_end=args.month_end,
            max_results=3,
        )

        roughness_records = []
        for item in stac_items[:3]:
            print(f"   Processing scene: {item['id'][:60]}...")
            sigma0 = extract_sigma0_at_point(item, lon, lat)

            if sigma0 and sigma0.get("vv") and sigma0.get("vh"):
                rdata = roughness_from_sigma0(sigma0["vv"], sigma0["vh"])
                rdata["scene_id"] = item["id"]
                rdata["scene_date"] = item.get("date", "?")
                roughness_records.append(rdata)
                print(f"      σ0_VV={rdata.get('vv_db', '?')}dB  "
                      f"σ0_VH={rdata.get('vh_db', '?')}dB  "
                      f"roughness={rdata.get('roughness', '?')}  "
                      f"sea_state={rdata.get('sea_state', '?')}")
            else:
                # Synthetic placeholder when COG access unavailable
                print(f"      ℹ️  Direct pixel extraction unavailable "
                      f"(scene requires CDSE credentials or is not COG). "
                      f"Recording metadata only.")
                roughness_records.append({
                    "scene_id": item["id"],
                    "scene_date": item.get("date", "?"),
                    "roughness": None,
                    "sea_state": "unavailable — requires CDSE credentials",
                    "note": "Download from: https://dataspace.copernicus.eu/",
                })

        # Summary statistics for this site
        valid_roughness = [r["roughness"] for r in roughness_records if r.get("roughness") is not None]
        site_summary = {
            "site": site_key,
            "label": site["label"],
            "coordinates": {"lon": lon, "lat": lat},
            "depth_m": site["depth_m"],
            "scenes_found": len(stac_items),
            "scenes_processed": len(roughness_records),
            "mean_roughness": round(sum(valid_roughness) / len(valid_roughness), 4) if valid_roughness else None,
            "visibility_window": (
                "good" if valid_roughness and sum(valid_roughness) / len(valid_roughness) < ROUGHNESS_CALM_THRESHOLD
                else "moderate" if valid_roughness
                else "unknown"
            ),
            "records": roughness_records,
        }
        site_results.append(site_summary)

    # Step 3: Build and save report
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "year": args.year,
        "search_window": f"{args.year}-{args.month_start:02d} to {args.year}-{args.month_end:02d}",
        "data_source": {
            "catalog": "Copernicus CDSE OData API (free, no auth)",
            "catalog_url": CDSE_ODATA_URL,
            "stac_source": "Element84 Earth Search v1 (public S1 GRD COG)",
            "stac_url": E84_STAC_URL,
            "credentials_required": "No for catalog search. Yes for data download (CDSE free account).",
            "registration": "https://dataspace.copernicus.eu/",
            "citation": (
                "Copernicus Sentinel data [year], processed by ESA. "
                "Accessed via Copernicus Data Space Ecosystem."
            ),
        },
        "scene_catalog": scene_catalog,
        "roughness_thresholds": {
            "calm": f"< {ROUGHNESS_CALM_THRESHOLD}",
            "moderate": f"{ROUGHNESS_CALM_THRESHOLD}–{ROUGHNESS_ROUGH_THRESHOLD}",
            "rough": f"> {ROUGHNESS_ROUGH_THRESHOLD}",
        },
        "sites": site_results,
        "interpretation": (
            "Surface roughness proxy = log10(σ0_VV / σ0_VH). "
            "Low values indicate calm sea conditions and higher likelihood of "
            "good underwater visibility. Correlate with wave height (CMEMS VHM0) "
            "for combined visibility scoring."
        ),
    }

    out_path = out_dir / f"s1_roughness_report_{args.year}.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    print(f"\n\n{'═'*70}")
    print(f"  ✅ Sentinel-1 SAR Analysis Complete")
    print(f"     Scenes found : {len(scene_catalog)}")
    print(f"     Sites analysed: {len(site_results)}")
    print(f"     Report saved  : {out_path}")
    print(f"{'═'*70}\n")

    if products:
        print("  📝 To download scenes (requires CDSE free account):")
        print("     1. Register at: https://dataspace.copernicus.eu/")
        print("     2. Use CDSE S3 API or Browser:")
        print("        https://browser.dataspace.copernicus.eu/")
        print("     3. Or use the official Python client:")
        print("        pip install cdsetool")
        print("        cdsetool download <product_id>")
        print()


if __name__ == "__main__":
    main()
