#!/usr/bin/env python3
"""
bvi_timeseries.py — Batch BVI trend computation for Algarve reef sites (2017–2025).

For each site in algarve_coastal_features.csv, queries Earth Search STAC for
cloud-free Sentinel-2 L2A scenes (June–September, cloud < 15%), reads a 2 km
COG window, and computes BVI via the physics + ranking model.

Results are cached per-scene so the script is safe to interrupt and rerun.

Outputs
-------
  outputs/bvi_timeseries/.cache/{site}/{scene_id}.json   (per-scene cache)
  outputs/bvi_timeseries/{site_name}.json                (per-site summary)
  outputs/bvi_timeseries/_index.json                     (all-sites index for dashboard)

Usage
-----
  python scripts/bvi_timeseries.py
  python scripts/bvi_timeseries.py --sites pedra_sta_eulalia galé --start 2020
  python scripts/bvi_timeseries.py --dry-run           # STAC queries only, no BVI
  python scripts/bvi_timeseries.py --cloud 20 --scenes-per-year 2
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

# ── project root on path ─────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("bvi_timeseries")
# Suppress urllib3 connection-pool noise from CMEMS background thread
logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)

# ── configuration ─────────────────────────────────────────────────────────────
SITES_CSV     = _ROOT / "outputs" / "coastal_topography" / "algarve_coastal_features.csv"
OUT_DIR       = _ROOT / "outputs" / "bvi_timeseries"
CACHE_DIR     = OUT_DIR / ".cache"
STAC_URL      = "https://earth-search.aws.element84.com/v1"
S2_COLLECTION = "sentinel-2-l2a"
WINDOW_PX     = 200        # pixels — 200 × 10 m = 2 km window
DEFAULT_DEPTH = 16.0       # m — Beer-Lambert path length
STAC_PAUSE_S  = 0.4        # polite delay between STAC requests


# ── site loader ───────────────────────────────────────────────────────────────

def load_sites(csv_path: Path, names: list[str] | None = None) -> list[dict]:
    import csv
    rows: list[dict] = []
    with open(csv_path) as fh:
        for row in csv.DictReader(fh):
            if names and row["site_name"] not in names:
                continue
            rows.append({
                "site_name":   row["site_name"],
                "lat":         float(row["latitude"]),
                "lon":         float(row["longitude"]),
                "slope_mean":  float(row.get("slope_mean", 5.0)),
                "aspect_mean": float(row.get("aspect_mean", 180.0)),
            })
    return rows


# ── STAC scene search ─────────────────────────────────────────────────────────

def search_summer_scenes(
    lat: float, lon: float, year: int,
    cloud_max: float = 15.0,
    limit: int = 1,
) -> list:
    """Return up to *limit* least-cloudy pystac.Item objects for Jun–Sep of *year*."""
    from pystac_client import Client
    client = Client.open(STAC_URL)
    date_range = f"{year}-06-01T00:00:00Z/{year}-09-30T23:59:59Z"
    try:
        results = client.search(
            collections=[S2_COLLECTION],
            intersects={"type": "Point", "coordinates": [lon, lat]},
            datetime=date_range,
            query={"eo:cloud_cover": {"lt": cloud_max}},
            max_items=50,
        )
        items = sorted(
            results.items(),
            key=lambda it: it.properties.get("eo:cloud_cover", 99.0),
        )
        return items[:limit]
    except Exception as exc:
        log.warning("STAC search failed year=%d lat=%.3f lon=%.3f: %s", year, lat, lon, exc)
        return []


def _band_href(item, band: str) -> Optional[str]:
    """Resolve B02/B03 asset href from a pystac Item (handles alias differences)."""
    aliases = {
        "B02": ["B02", "blue", "b02"],
        "B03": ["B03", "green", "b03"],
    }
    avail = {k.lower(): v for k, v in item.assets.items()}
    for alias in aliases.get(band, [band]):
        if alias.lower() in avail:
            return avail[alias.lower()].href
    return None


# ── COG windowed read ─────────────────────────────────────────────────────────

def read_cog_window(href: str, lat: float, lon: float, size_px: int = WINDOW_PX) -> Optional[np.ndarray]:
    """
    Range-request a square window from a Cloud-Optimised GeoTIFF.

    Uses GDAL /vsicurl/ so only the needed tiles are downloaded (~30–100 KB
    per band). Returns float32 array of shape (size_px, size_px) or None on error.
    """
    import rasterio
    import rasterio.windows
    from rasterio.warp import transform as warp_transform
    url = f"/vsicurl/{href}" if href.startswith("http") else href
    try:
        with rasterio.open(url) as src:
            # Sentinel-2 COGs are UTM, not WGS84 — must reproject before indexing
            xs, ys = warp_transform("EPSG:4326", src.crs, [lon], [lat])
            row, col = src.index(xs[0], ys[0])
            half = size_px // 2
            col_off = max(0, col - half)
            row_off = max(0, row - half)
            window = rasterio.windows.Window(col_off, row_off, min(size_px, src.width - col_off), min(size_px, src.height - row_off))  # type: ignore[call-arg]
            arr = src.read(1, window=window).astype(np.float32)
            nodata = src.nodata
            if nodata is not None:
                arr = np.where(arr == nodata, np.nan, arr)
            return arr
    except Exception as exc:
        log.warning("COG window read failed %s: %s", href[:60], exc)
        return None


# ── lightweight BVI computation ───────────────────────────────────────────────

def compute_bvi(
    b02: np.ndarray,
    b03: Optional[np.ndarray],
    month: int,
    site: dict,
) -> Optional[dict]:
    """
    Compute BVI from a small COG window without running the full pipeline.

    Physics chain: BOA conversion → Kd490 (band-ratio / seasonal prior) →
    Beer-Lambert transmittance → window statistics → ranking_model.predict_score().
    """
    from src.cmems_kd490 import get_kd490, get_zsd
    from src.ranking_model import predict_score, terrain_exposure_modifier

    # BOA conversion
    b02 = b02.copy()
    if np.nanmax(b02) > 2.0:
        b02 /= 10000.0
    if b03 is not None:
        b03 = b03.copy()
        if np.nanmax(b03) > 2.0:
            b03 /= 10000.0

    # Need enough valid water pixels
    valid = b02[np.isfinite(b02) & (b02 > 0) & (b02 < 0.25)]
    if valid.size < 400:
        log.debug("Too few valid pixels (%d) — skipping window", valid.size)
        return None

    # Kd490 — band-ratio if B03 available, else seasonal prior
    kd_seas = get_kd490(month)
    kd_b02 = kd_seas
    try:
        if b03 is not None:
            from src.reef_ml_predictor_acolite import estimate_kd_bandratio
            kd_b02, _ = estimate_kd_bandratio(b02, b03, kd_seas)
    except Exception:
        kd_b02 = kd_seas

    # Beer-Lambert two-way transmittance
    water_trans = float(np.exp(-2.0 * kd_b02 * DEFAULT_DEPTH))

    # Window statistics
    mu    = float(np.nanmean(valid))
    sigma = float(np.nanstd(valid))
    snr   = mu / (sigma + 1e-9)
    contrast = sigma / (mu + 1e-9)
    # FFT cleanliness proxy — high SNR = cleaner image
    cleanliness = float(np.clip(snr / 25.0, 0.0, 1.0))

    # Terrain from site CSV
    slope_mean  = site.get("slope_mean",  5.0)
    aspect_mean = site.get("aspect_mean", 180.0)
    slope_proxy = float(np.clip(slope_mean / 45.0, 0.0, 1.0))

    features = {
        "kd_b02":          kd_b02,
        "water_trans":     water_trans,
        "image_contrast":  contrast,
        "signal_strength": snr,
        "cleanliness":     cleanliness,
        "bathy_slope_proxy": slope_proxy,
    }

    try:
        _ps = predict_score(features)
        score = float(_ps.get("score", 0.0))
        mode = str(_ps.get("mode", "rf"))
    except Exception as exc:
        log.warning("predict_score failed: %s", exc)
        score = float(np.clip(water_trans * 0.5 + cleanliness * 0.3 + contrast * 0.2, 0, 1))
        mode = "heuristic"

    # Terrain exposure modifier
    modifier = terrain_exposure_modifier(slope_mean, aspect_mean)
    bvi = round(score * modifier, 4)

    zsd = get_zsd(month)
    return {
        "bvi":        bvi,
        "ranker_mode": mode,
        "kd_b02":     round(kd_b02, 4),
        "kd_seasonal": round(kd_seas, 4),
        "zsd_m":      round(zsd, 1) if zsd is not None else None,
        "water_trans": round(water_trans, 4),
        "snr":        round(snr, 2),
        "contrast":   round(contrast, 4),
        "n_pixels":   int(valid.size),
        "terrain_modifier": round(modifier, 3),
    }


# ── per-scene caching ─────────────────────────────────────────────────────────

def _cache_path(site_name: str, scene_id: str) -> Path:
    p = CACHE_DIR / site_name / f"{scene_id}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _load_cache(site_name: str, scene_id: str) -> Optional[dict]:
    p = _cache_path(site_name, scene_id)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return None


def _save_cache(site_name: str, scene_id: str, data: dict) -> None:
    _cache_path(site_name, scene_id).write_text(json.dumps(data, indent=2))


# ── main per-site processing ──────────────────────────────────────────────────

def process_site(site: dict, years: range, cloud_max: float, scenes_per_year: int, dry_run: bool) -> list[dict]:
    name = site["site_name"]
    lat, lon = site["lat"], site["lon"]
    records: list[dict] = []

    for year in years:
        log.info("[%s] year=%d — searching STAC …", name, year)
        items = search_summer_scenes(lat, lon, year, cloud_max=cloud_max, limit=scenes_per_year)
        time.sleep(STAC_PAUSE_S)

        if not items:
            log.info("[%s] year=%d — no scenes found", name, year)
            continue

        for item in items:
            scene_id  = item.id  # type: ignore[union-attr]
            props     = item.properties  # type: ignore[union-attr]
            date_str  = props.get("datetime", "")[:10]
            cloud_pct = round(props.get("eo:cloud_cover", 0.0), 1)
            month     = int(date_str[5:7]) if date_str else 7

            record_base = {
                "date":      date_str,
                "scene_id":  scene_id,
                "cloud_pct": cloud_pct,
                "month":     month,
            }

            if dry_run:
                log.info("[%s] DRY-RUN %s  cloud=%.1f%%", name, date_str, cloud_pct)
                records.append(record_base)
                continue

            # Check cache
            cached = _load_cache(name, scene_id)
            if cached:
                log.info("[%s] %s — cache hit ✓", name, date_str)
                records.append(cached)
                continue

            # Read COG windows
            b02_href = _band_href(item, "B02")
            b03_href = _band_href(item, "B03")
            if not b02_href:
                log.warning("[%s] %s — no B02 href", name, date_str)
                continue

            log.info("[%s] %s  cloud=%.1f%%  reading COG window …", name, date_str, cloud_pct)
            b02 = read_cog_window(b02_href, lat, lon)
            if b02 is None:
                log.warning("[%s] %s — B02 read failed", name, date_str)
                continue

            b03 = read_cog_window(b03_href, lat, lon) if b03_href else None

            bvi_data = compute_bvi(b02, b03, month, site)
            if bvi_data is None:
                log.info("[%s] %s — insufficient pixels", name, date_str)
                continue

            record = {**record_base, **bvi_data}
            _save_cache(name, scene_id, record)
            records.append(record)
            log.info("[%s] %s  BVI=%.3f  kd=%.4f  mode=%s",
                     name, date_str, record["bvi"], record["kd_b02"], record["ranker_mode"])

    return sorted(records, key=lambda r: r["date"])


def write_site_json(site: dict, records: list[dict]) -> Path:
    name = site["site_name"]
    out = {
        "site":     name,
        "lat":      site["lat"],
        "lon":      site["lon"],
        "updated":  datetime.utcnow().isoformat() + "Z",  # noqa: DTZ003
        "scenes":   records,
        # summary statistics (non-empty records only)
        "summary":  _summarise(records),
    }
    path = OUT_DIR / f"{name}.json"
    path.write_text(json.dumps(out, indent=2))
    log.info("Wrote %s  (%d scenes)", path.name, len(records))
    return path


def _summarise(records: list[dict]) -> dict:
    bvis = [r["bvi"] for r in records if "bvi" in r]
    if not bvis:
        return {}
    return {
        "n_scenes":  len(bvis),
        "bvi_mean":  round(float(np.mean(bvis)), 4),
        "bvi_min":   round(float(np.min(bvis)),  4),
        "bvi_max":   round(float(np.max(bvis)),  4),
        "bvi_trend": _trend_slope(records),
    }


def _trend_slope(records: list[dict]) -> Optional[float]:
    """Least-squares BVI trend per year (positive = improving visibility)."""
    pts = [(int(r["date"][:4]), r["bvi"]) for r in records if "bvi" in r and r.get("date")]
    if len(pts) < 2:
        return None
    xs = np.array([p[0] for p in pts], dtype=float)
    ys = np.array([p[1] for p in pts], dtype=float)
    xs -= xs.mean()
    slope = float(np.dot(xs, ys) / (np.dot(xs, xs) + 1e-12))
    return round(slope, 5)


def write_index(sites_done: list[dict]) -> Path:
    index = []
    for s in sites_done:
        p = OUT_DIR / f"{s['site_name']}.json"
        if p.exists():
            try:
                d = json.loads(p.read_text())
                index.append({
                    "site":    d["site"],
                    "lat":     d["lat"],
                    "lon":     d["lon"],
                    "summary": d.get("summary", {}),
                })
            except Exception:
                pass
    path = OUT_DIR / "_index.json"
    path.write_text(json.dumps(index, indent=2))
    log.info("Wrote _index.json  (%d sites)", len(index))
    return path


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sites",           nargs="*",         help="site names to process (default: all)")
    ap.add_argument("--start",           type=int, default=2017, help="first year (default: 2017)")
    ap.add_argument("--end",             type=int, default=2025, help="last year (default: 2025)")
    ap.add_argument("--cloud",           type=float, default=15.0, help="max cloud cover %% (default: 15)")
    ap.add_argument("--scenes-per-year", type=int,  default=1,    help="scenes per year per site (default: 1)")
    ap.add_argument("--dry-run",         action="store_true",      help="STAC only, no BVI computation")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Warm up CMEMS Kd490 with a 30-second timeout.
    # If CMEMS is slow or unavailable, the static table is used for all scenes.
    # NOTE: shutdown(wait=False) so the timeout actually fires — the default `with`
    # context manager would call shutdown(wait=True) and block until the thread ends.
    try:
        import concurrent.futures
        from src.cmems_kd490 import refresh_from_cmems
        _pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            fut = _pool.submit(refresh_from_cmems)
            ok = fut.result(timeout=30)
            log.info("CMEMS Kd490 %s", "loaded" if ok else "using static fallback")
        except concurrent.futures.TimeoutError:
            log.info("CMEMS Kd490 timeout (>30 s) — using static table for all scenes")
        finally:
            _pool.shutdown(wait=False)   # let thread finish in background; don't block
    except Exception as exc:
        log.debug("CMEMS refresh skipped: %s", exc)

    sites = load_sites(SITES_CSV, names=args.sites)
    if not sites:
        log.error("No sites found — check --sites names or %s", SITES_CSV)
        sys.exit(1)

    years = range(args.start, args.end + 1)
    log.info("Processing %d sites × %d years  (cloud<%.0f%%  dry_run=%s)",
             len(sites), len(years), args.cloud, args.dry_run)

    for site in sites:
        records = process_site(site, years, args.cloud, args.scenes_per_year, args.dry_run)
        if not args.dry_run:
            write_site_json(site, records)

    if not args.dry_run:
        write_index(sites)
        log.info("Done — results in %s", OUT_DIR)


if __name__ == "__main__":
    main()
