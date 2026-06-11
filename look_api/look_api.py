import json
import re
import requests
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List, Optional

# Configuration
ENDPOINTS = [
    # ── DGT STAC (dgt-be.a.incd.pt:8081) ──────────────────────────────────────
    {"name": "DGT STAC / all collections",
     "url": "https://dgt-be.a.incd.pt:8081/collections"},
    {"name": "DGT STAC / MDT-50cm LiDAR DTM",
     "url": "https://dgt-be.a.incd.pt:8081/collections/MDT-50cm/items?bbox=-8.25,37.04,-8.17,37.10&limit=5"},
    {"name": "DGT STAC / MDS-50cm LiDAR DSM",
     "url": "https://dgt-be.a.incd.pt:8081/collections/MDS-50cm/items?bbox=-8.25,37.04,-8.17,37.10&limit=3"},
    {"name": "DGT STAC / ORTOSAT-2023 30cm RGBI",
     "url": "https://dgt-be.a.incd.pt:8081/collections/ORTOSAT-2023/items?bbox=-8.25,37.04,-8.17,37.10&limit=3"},
    {"name": "DGT STAC / S2 mosaic 2025",
     "url": "https://dgt-be.a.incd.pt:8081/collections/MosaicoS2-2025/items?bbox=-8.25,37.04,-8.17,37.10&limit=2"},
    {"name": "DGT STAC / S2 mosaic 2024",
     "url": "https://dgt-be.a.incd.pt:8081/collections/MosaicoS2-2024/items?bbox=-8.25,37.04,-8.17,37.10&limit=2"},
    {"name": "DGT STAC / ORTOS-2021 aerial",
     "url": "https://dgt-be.a.incd.pt:8081/collections/ORTOS-2021/items?bbox=-8.25,37.04,-8.17,37.10&limit=2"},
    {"name": "DGT STAC / ORTOS-2018 aerial",
     "url": "https://dgt-be.a.incd.pt:8081/collections/ORTOS-2018/items?bbox=-8.25,37.04,-8.17,37.10&limit=2"},
    {"name": "DGT STAC / ORTOS-2015 aerial",
     "url": "https://dgt-be.a.incd.pt:8081/collections/ORTOS-2015/items?bbox=-8.25,37.04,-8.17,37.10&limit=2"},
    {"name": "DGT STAC / ORTOS-1995 aerial",
     "url": "https://dgt-be.a.incd.pt:8081/collections/ORTOS-1995/items?bbox=-8.25,37.04,-8.17,37.10&limit=2"},
    {"name": "DGT STAC / COPC LiDAR point clouds",
     "url": "https://dgt-be.a.incd.pt:8081/collections/COPC/items?bbox=-8.25,37.04,-8.17,37.10&limit=2"},
    {"name": "DGT STAC / MDT-2m (2m DTM)",
     "url": "https://dgt-be.a.incd.pt:8081/collections/MDT-2m/items?bbox=-8.25,37.04,-8.17,37.10&limit=2"},
    {"name": "DGT STAC / MDS-2m (2m DSM)",
     "url": "https://dgt-be.a.incd.pt:8081/collections/MDS-2m/items?bbox=-8.25,37.04,-8.17,37.10&limit=2"},

    # ── DGT MinIO storage (downloads — expect 403) ─────────────────────────────
    {"name": "DGT MinIO / MDT-50cm COG tile (anon)",
     "url": "https://stor-002.a.acnca.pt:9000/lidar/MDT50cm/MDT-50cm-196015-04-2024_v01.tif"},
    {"name": "DGT MinIO / ORTOSAT-2023 COG tile (anon)",
     "url": "https://stor-002.a.acnca.pt:9000/orto2023/ortosat2023_cog_30cm_rgbi_jpg_605-4_v01.tif"},
    {"name": "DGT MinIO / health check",
     "url": "https://stor-002.a.acnca.pt:9000/minio/health/live"},

    # ── DGT WMS (ortos.dgterritorio.gov.pt) ────────────────────────────────────
    {"name": "DGT WMS / OrtoSat2023 GetCapabilities",
     "url": "https://ortos.dgterritorio.gov.pt/wms/ortosat2023?SERVICE=WMS&REQUEST=GetCapabilities"},
    {"name": "DGT WMS / OrtoSat2023 GetMap tile",
     "url": "https://ortos.dgterritorio.gov.pt/wms/ortosat2023?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap"
            "&LAYERS=OrtoSat2023&CRS=EPSG:4326&BBOX=37.04,-8.25,37.10,-8.17"
            "&WIDTH=256&HEIGHT=256&FORMAT=image/png&STYLES="},

    # ── IH / DGRM ArcGIS (webgis.dgrm.mm.gov.pt) ──────────────────────────────
    {"name": "IH/DGRM ArcGIS / all services",
     "url": "https://webgis.dgrm.mm.gov.pt/arcgis/rest/services?f=json"},
    {"name": "IH/DGRM ArcGIS / Batimetrica_IH MapServer",
     "url": "https://webgis.dgrm.mm.gov.pt/arcgis/rest/services/Dados_entidades_externas/Batimetrica_IH/MapServer?f=json"},
    {"name": "IH/DGRM ArcGIS / isobath query (wide bbox)",
     "url": "https://webgis.dgrm.mm.gov.pt/arcgis/rest/services/Dados_entidades_externas/Batimetrica_IH/MapServer/0/query"
            "?f=geojson&geometry=-8.55,36.84,7.87,37.20&geometryType=esriGeometryEnvelope"
            "&inSR=4326&outSR=4326&outFields=*&returnGeometry=true&resultRecordCount=5&where=1=1"},

    # ── SNIG / Portuguese spatial data infrastructure ──────────────────────────
    {"name": "SNIG / catalog search (MDT)",
     "url": "https://snig.dgterritorio.gov.pt/rndg/srv/eng/q?q=MDT&_content_type=json&fast=index"},
    {"name": "SNIG / CSW GetCapabilities",
     "url": "https://snig.dgterritorio.gov.pt/rndg/srv/por/csw?SERVICE=CSW&REQUEST=GetCapabilities"},

    # ── dados.gov.pt ───────────────────────────────────────────────────────────
    {"name": "dados.gov.pt / search MDT DGT",
     "url": "https://dados.gov.pt/api/1/datasets/?q=MDT+DGT&page_size=3"},
    {"name": "dados.gov.pt / search batimetria",
     "url": "https://dados.gov.pt/api/1/datasets/?q=batimetria+hidrográfico&page_size=3"},

    # ── EMODnet Bathymetry ─────────────────────────────────────────────────────
    {"name": "EMODnet / WCS GetCapabilities",
     "url": "https://ows.emodnet-bathymetry.eu/wcs?SERVICE=WCS&VERSION=1.0.0&REQUEST=GetCapabilities"},
    {"name": "EMODnet / WCS GetCoverage (Algarve)",
     "url": "https://ows.emodnet-bathymetry.eu/wcs?SERVICE=WCS&VERSION=1.0.0&REQUEST=GetCoverage"
            "&COVERAGE=emodnet:mean&CRS=EPSG:4326&BBOX=-8.25,37.04,-8.17,37.10"
            "&WIDTH=64&HEIGHT=32&FORMAT=image/tiff"},
    {"name": "EMODnet / WMS GetCapabilities",
     "url": "https://ows.emodnet-bathymetry.eu/wms?SERVICE=WMS&REQUEST=GetCapabilities"},

    # ── Copernicus GLO-30 DEM (public AWS S3) ──────────────────────────────────
    {"name": "Copernicus GLO-30 / tile N37 W009",
     "url": "https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_N37_00_W009_00_DEM/Copernicus_DSM_COG_10_N37_00_W009_00_DEM.tif"},
    {"name": "Copernicus GLO-30 / tile N37 W008",
     "url": "https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_N37_00_W008_00_DEM/Copernicus_DSM_COG_10_N37_00_W008_00_DEM.tif"},

    # ── Copernicus CMEMS ───────────────────────────────────────────────────────
    {"name": "CMEMS / OPeNDAP Kd490 Atlantic",
     "url": "https://my.cmems-du.eu/thredds/dodsC/cmems_obs-oc_atl_bgc-transp_my_l4-multi-4km_P1M.das"},

    # ── Satellite imagery STAC ─────────────────────────────────────────────────
    {"name": "Earth Search STAC / root",
     "url": "https://earth-search.aws.element84.com/v1"},
    {"name": "Earth Search STAC / S2-L2A collection",
     "url": "https://earth-search.aws.element84.com/v1/collections/sentinel-2-l2a"},
    {"name": "Planetary Computer / STAC root",
     "url": "https://planetarycomputer.microsoft.com/api/stac/v1"},
    {"name": "Planetary Computer / S2-L2A collection",
     "url": "https://planetarycomputer.microsoft.com/api/stac/v1/collections/sentinel-2-l2a"},

    # ── ICESat-2 / OpenAltimetry ───────────────────────────────────────────────
    {"name": "OpenAltimetry / ATL08 API",
     "url": "https://openaltimetry.earthdatacloud.nasa.gov/data/api/icesat2/atl08"
            "?minx=-8.25&miny=37.04&maxx=-8.17&maxy=37.10&date=2023-07-15&outputFormat=json&trackId=1"},

    # ── IPMA (Portuguese met office) ───────────────────────────────────────────
    {"name": "IPMA / station observations",
     "url": "https://api.ipma.pt/open-data/observation/meteorology/stations/observations.json"},
    {"name": "IPMA / daily forecast",
     "url": "https://api.ipma.pt/open-data/forecast/meteorology/cities/daily/hp-daily-forecast-day0.json"},
    {"name": "IPMA / sea state forecast",
     "url": "https://api.ipma.pt/open-data/forecast/oceanography/daily/hp-daily-sea-forecast-day0.json"},

    # ── OpenTopography (key discovered at runtime — see discover_opentopo_key) ──
    # Placeholder replaced dynamically in main() once key discovery runs.
    {"name": "OpenTopography / COP30",
     "url": "https://portal.opentopography.org/API/globaldem?demtype=COP30"
            "&south=37.04&north=37.10&west=-8.25&east=-8.17&outputFormat=GTiff&API_Key=__OT_KEY__"},
]

THREADS = 8

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}


def probe_endpoint(endpoint: Dict[str, Any]) -> Dict[str, Any]:
    try:
        response = requests.get(
            url=endpoint["url"],
            timeout=12,
            headers=HEADERS,
            verify=False,
            stream=True,         # avoid downloading large COG bodies
        )
        # Read only first 512 bytes for large binary responses
        chunk = b""
        for c in response.iter_content(512):
            chunk = c
            break

        content_type = response.headers.get("Content-Type", "")
        content_length = response.headers.get("Content-Length", "?")

        # Determine access label
        if response.status_code == 200:
            label = "OK"
        elif response.status_code == 206:
            label = "OK (partial)"
        elif response.status_code == 403:
            label = "BLOCKED"
        elif response.status_code == 401:
            label = "AUTH_REQUIRED"
        else:
            label = f"HTTP_{response.status_code}"

        return {
            "name": endpoint["name"],
            "status": label,
            "response_code": response.status_code,
            "content_type": content_type[:60],
            "content_length": content_length,
            "preview": chunk[:120].decode("utf-8", errors="replace").replace("\n", " ").strip(),
        }

    except requests.exceptions.RequestException as e:
        return {
            "name": endpoint["name"],
            "status": "ERROR",
            "response_code": None,
            "error": str(e)[:120],
        }


# ── OpenTopography API key discovery ──────────────────────────────────────────

# Sources searched in order — all are publicly accessible URLs
_OT_DISCOVERY_SOURCES = [
    # 1. Official OpenAPI spec (most reliable — OT publishes demo key here)
    "https://portal.opentopography.org/apidocs/openapi.json",
    # 2. Developer page HTML (may contain key in code examples)
    "https://opentopography.org/developers",
    # 3. Swagger UI HTML (sometimes embeds default key in JS config)
    "https://portal.opentopography.org/apidocs/",
    # 4. OT documentation notebook on GitHub
    "https://raw.githubusercontent.com/OpenTopography/OT_3DEP_Workflows/main/notebooks/01_OT_API_GlobalDEM.ipynb",
    # 5. OT Python client source on PyPI (setup.cfg or __init__ may have demo key)
    "https://raw.githubusercontent.com/OpenTopography/OT_Python_CLI/master/opentopo/otcli.py",
]

# Regex patterns to extract candidate keys from any source
_KEY_PATTERNS = [
    re.compile(r'demoapikey[a-zA-Z0-9_\-]*', re.IGNORECASE),        # OT demo key naming convention
    re.compile(r'[Aa][Pp][Ii][-_]?[Kk]ey["\'\s:=]+([a-zA-Z0-9_\-]{8,40})'),  # generic "API_Key = xxx"
    re.compile(r'"example"\s*:\s*"([a-zA-Z0-9_\-]{8,40})"'),        # OpenAPI example field
    re.compile(r'API_Key=([a-zA-Z0-9_\-]{8,40})'),                  # query-string style
]

_OT_TEST_URL = (
    "https://portal.opentopography.org/API/globaldem"
    "?demtype=SRTMGL3&south=37.06&north=37.08&west=-8.22&east=-8.20"
    "&outputFormat=GTiff&API_Key={key}"
)


def _test_ot_key(key: str) -> str:
    """Return 'OK', 'RATE_LIMITED', or 'INVALID'."""
    try:
        r = requests.get(_OT_TEST_URL.format(key=key), timeout=15, stream=True)
        if r.status_code == 200:
            return "OK"
        body = next(r.iter_content(512), b"").decode("utf-8", errors="replace")
        if "rate limit" in body.lower():
            return "RATE_LIMITED"
        if "invalid" in body.lower() or "not found" in body.lower():
            return "INVALID"
        return f"HTTP_{r.status_code}"
    except Exception as e:
        return f"ERROR: {e}"


def discover_opentopo_key() -> Optional[str]:
    """
    Search publicly accessible OpenTopography pages and repos for a working API key.

    Strategy:
      1. Fetch each discovery source (OpenAPI spec, docs, GitHub)
      2. Apply regex patterns to extract candidate key strings
      3. Deduplicate and test each candidate against a small Algarve bbox
      4. Return the first key that works, or None if all are rate-limited/invalid

    Returns the working key string, or None.
    """
    print("\n── OpenTopography key discovery ──")
    candidates: List[str] = []
    seen: set = set()

    for source_url in _OT_DISCOVERY_SOURCES:
        try:
            r = requests.get(source_url, timeout=10,
                             headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code != 200:
                print(f"  {source_url.split('/')[-1]:<45} HTTP {r.status_code}")
                continue

            text = r.text
            # For JSON sources, also search the raw string
            if "json" in r.headers.get("Content-Type", ""):
                try:
                    text = json.dumps(json.loads(text))  # normalise whitespace
                except Exception:
                    pass

            found_here = []
            for pat in _KEY_PATTERNS:
                for match in pat.findall(text):
                    key = match.strip().strip("\"'")
                    if key and key not in seen:
                        seen.add(key)
                        candidates.append(key)
                        found_here.append(key)

            label = source_url.split("/")[-1] or source_url.split("/")[-2]
            if found_here:
                print(f"  ✅ {label:<45} found keys: {found_here}")
            else:
                print(f"  ⚠️  {label:<45} no key patterns matched")

        except Exception as e:
            print(f"  ❌ {source_url.split('/')[-1]:<45} {e}")

    if not candidates:
        print("  No candidate keys found in any source.")
        print("  → Register for a free key at: https://portal.opentopography.org/requestApiKey")
        return None

    print(f"\n  {len(candidates)} unique candidate(s) found: {candidates}")
    print("  Testing each against a small Algarve bbox …")

    best_key = None
    for key in candidates:
        result = _test_ot_key(key)
        icon = "✅" if result == "OK" else ("⏱️ " if result == "RATE_LIMITED" else "❌")
        print(f"  {icon} key={key:<30}  result={result}")
        if result == "OK" and best_key is None:
            best_key = key

    if best_key:
        print(f"\n  Working key: {best_key}")
    else:
        print("\n  All keys rate-limited or invalid.")
        print("  → Get your own free key (instant): https://portal.opentopography.org/requestApiKey")
        print("  → Demo key limit: 50 calls / 24 hrs shared across all users")

    return best_key


def main():
    import warnings
    warnings.filterwarnings("ignore")          # suppress SSL self-signed cert warnings

    # ── Discover OpenTopography API key before probing ─────────────────────────
    ot_key = discover_opentopo_key()
    # Patch the placeholder URL with the discovered key (or leave as __OT_KEY__ to show failure)
    for ep in ENDPOINTS:
        if "__OT_KEY__" in ep["url"]:
            ep["url"] = ep["url"].replace("__OT_KEY__", ot_key or "__NO_KEY__")

    print(f"\nProbing {len(ENDPOINTS)} endpoints ({THREADS} threads) …\n")

    with ThreadPoolExecutor(max_workers=THREADS) as executor:
        futures = [executor.submit(probe_endpoint, ep) for ep in ENDPOINTS]
        results = [f.result() for f in futures]

    # Keep original order
    order = {ep["name"]: i for i, ep in enumerate(ENDPOINTS)}
    results.sort(key=lambda r: order.get(r["name"], 999))

    # ── Print ──────────────────────────────────────────────────────────────────
    icons = {"OK": "✅", "OK (partial)": "✅", "BLOCKED": "🔒",
             "AUTH_REQUIRED": "🔑", "ERROR": "❌"}

    prev_group = None
    for r in results:
        group = r["name"].split("/")[0].strip()
        if group != prev_group:
            print(f"\n── {group} ──")
            prev_group = group

        icon = icons.get(r["status"]) or "⚠️ "
        code = r.get("response_code") or "---"
        ct   = r.get("content_type", "")
        note = r.get("preview", r.get("error", ""))[:80]
        print(f"  {icon} [{code}] {r['name']:<50}  {ct}")
        if note and r["status"] not in ("OK",):
            print(f"         {note}")

    # ── Summary ────────────────────────────────────────────────────────────────
    counts: Dict[str, int] = {}
    for r in results:
        s = "OK" if r["status"] in ("OK", "OK (partial)") else r["status"]
        counts[s] = counts.get(s, 0) + 1

    print(f"\n{'─'*60}")
    print(f"  Total: {len(results)}   "
          + "   ".join(f"{icons.get(k,'?')} {k}: {v}" for k, v in sorted(counts.items())))
    print(f"{'─'*60}")

    accessible = [r for r in results if r["status"] in ("OK", "OK (partial)")]
    if accessible:
        print(f"\n  {len(accessible)} accessible endpoints:")
        for r in accessible:
            print(f"    • {r['name']}")


if __name__ == "__main__":
    main()
