#!/usr/bin/env python3
"""
reef_detector.py — Find reef-like seafloor signatures in Sentinel-2 imagery.

Strategy (physically grounded, GPS-anchored):
  1. Download the 4 optical bands (B02/B03/B04/B08) for a scene over a bbox.
  2. Measure REAL clarity at the GPS reef pixel: NIR glint level + bottom contrast.
     (This answers "which scene has the better light" from the actual pixels,
     not from a precomputed SNR label.)
  3. Apply the Lyzenga depth-invariant transform: DII = X_B02 - (k02/k03)·X_B03,
     where Xi = ln(Ri - Lsw_i) and Lsw_i is the deep-water background.
     The k02/k03 attenuation ratio is estimated automatically per scene
     (Lyzenga 1978 variance method) — no hardcoded coefficient.
  4. Add local texture (rugosity proxy): reef is spatially heterogeneous,
     sand is smooth.
  5. Combine DII + texture into a reef-likelihood map over the WHOLE scene,
     anchored/validated at the 2 GPS-confirmed reef sites.
  6. Output: reef_likelihood.tif (GeoTIFF) + overlay PNG + report JSON.

The 2 GPS reef points are the only ground truth. Everything the map highlights
ELSEWHERE is a CANDIDATE with the same optical signature — to be confirmed by
dive survey, not asserted as fact.

Usage
-----
    python scripts/reef_detector.py \
        --scene S2A_29SNB_20230901_0_L2A \
        --scene S2B_29SNB_20250925_0_L2A
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.transform import from_bounds, rowcol

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Study area and ground truth ────────────────────────────────────────────────
# Western Algarve shelf (Quarteira → Armação de Pêra), includes open water to
# the south for reliable Lsw (deep-water background) estimation.
# Eastern sites (Tavira, Fuseta, Faro, Olhão) are on separate S2 tiles — run
# separately with tile 29SNC scenes.
DEFAULT_BBOX = (-8.45, 36.95, -8.05, 37.15)   # lon_min, lat_min, lon_max, lat_max

# GPS-confirmed dive sites are marked confirmed=True.
# Survey-hypothesis sites (from algarve_reef_survey.py) are unconfirmed candidates
# that share a reef-like bathymetric setting — to be validated by dive.
REEF_SITES = [
    # ── GPS dive-confirmed ─────────────────────────────────────────────────────
    {"name": "Pedra_Sta_Eulalia",    "lon": -8.2102,  "lat": 37.0691 },
    {"name": "Pedra_do_Alto",        "lon": -8.20982, "lat": 37.05815},
    # ── Fishing GPS (CarlosBarrinha.gpx, ≥3 repeated visits, 5–22 m depth) ────
    {"name": "fishing_N_eulalia_8m", "lon": -8.20294, "lat": 37.07504},  # 6 visits, 8 m
    {"name": "fishing_gale_16m",     "lon": -8.22965, "lat": 37.05822},  # 6 visits, 16 m
    {"name": "fishing_gale_15m_a",   "lon": -8.23384, "lat": 37.06220},  # 5 visits, 15 m
    {"name": "fishing_deep_E_19m",   "lon": -8.12089, "lat": 37.03144},  # 5 visits, 19 m
    {"name": "fishing_eulalia_11m",  "lon": -8.21477, "lat": 37.06629},  # 4 visits, 11 m
    {"name": "fishing_midshelf_19m", "lon": -8.17101, "lat": 37.04593},  # 4 visits, 19 m
    {"name": "fishing_gale_16m_b",   "lon": -8.23001, "lat": 37.05723},  # 4 visits, 16 m
    {"name": "fishing_gale_15m_b",   "lon": -8.23315, "lat": 37.06015},  # 4 visits, 15 m
    # ── Fishing GPS — EASTERN Algarve (tile 29SNA, top by reef_score) ───────────
    {"name": "e_faro_reef_12m_a",    "lon": -8.02339, "lat": 37.00917},  # 12 visits TRI=0.65 reef_score=26
    {"name": "e_faro_reef_11m",      "lon": -8.02498, "lat": 37.00989},  # 12 visits TRI=0.60 reef_score=24
    {"name": "e_tavira_13m",         "lon": -7.94326, "lat": 36.96016},  # 7 visits  TRI=0.82 reef_score=19
    {"name": "e_faro_reef_12m_b",    "lon": -8.02327, "lat": 37.00857},  # 10 visits TRI=0.57 reef_score=19
    {"name": "e_olhao_17m_a",        "lon": -7.97075, "lat": 36.96263},  # 12 visits TRI=0.47 reef_score=19
    {"name": "e_olhao_17m_b",        "lon": -7.97262, "lat": 36.96305},  # 12 visits TRI=0.45 reef_score=18
    {"name": "e_olhao_12m",          "lon": -7.94072, "lat": 36.95948},  # 11 visits TRI=0.47 reef_score=17
    {"name": "e_faro_16m_highTRI",   "lon": -7.99863, "lat": 36.98575},  # 5 visits  TRI=0.99 reef_score=17
    # ── Survey hypotheses (algarve_reef_survey.py, W cluster) ─────────────────
    {"name": "quarteira",            "lon": -8.1000,  "lat": 37.0650 },
    {"name": "vilamoura",            "lon": -8.1300,  "lat": 37.0750 },
    {"name": "olhos_de_agua",        "lon": -8.1600,  "lat": 37.0900 },
    {"name": "albufeira_reef",       "lon": -8.2105,  "lat": 37.0690 },
    {"name": "gale",                 "lon": -8.2296,  "lat": 37.0560 },
    {"name": "salgados",             "lon": -8.3000,  "lat": 37.0950 },
    {"name": "armacao_de_pera",      "lon": -8.3600,  "lat": 37.0700 },
]

# Names physically verified by divers. Fishing spots are high-confidence repeated
# GPS fixes but not dive-confirmed — keep separate for honest reporting.
CONFIRMED_REEF_NAMES = {"Pedra_Sta_Eulalia", "Pedra_do_Alto"}
FISHING_GPS_NAMES = {s["name"] for s in REEF_SITES if str(s["name"]).startswith("fishing_")}

# Earth Search asset names for the 10 m bands
_EARTH_SEARCH  = "https://earth-search.aws.element84.com/v1"
_S2_COLLECTION = "sentinel-2-l2a"
_BAND_ASSETS   = {"blue": "blue", "green": "green", "red": "red", "nir": "nir"}

OUTPUT_DIR = Path("outputs/reef_detector")

# Glint threshold: NIR BOA reflectance above this = surface contamination / land
GLINT_NIR_MAX = 0.015

# Optical depth window (m, negative = below MSL). Bottom is only visible in this
# range; beyond it the signal is sensor noise. Used to mask out deep water.
EMODNET_DEPTH = Path("outputs/reef_habitat/emodnet/depth_algarve.tif")
DEPTH_SHALLOW = -2.0     # exclude breaking-wave surf zone
DEPTH_DEEP    = -22.0    # Sentinel-2 optical floor in clear Algarve water


# ─────────────────────────────────────────────────────────────────────────────
# 1. Scene download (windowed COG read over bbox)
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_assets(scene_id: str) -> dict:
    """Return {asset_key: href} for a scene via Earth Search STAC."""
    from pystac_client import Client
    cat = Client.open(_EARTH_SEARCH)
    res = cat.search(collections=[_S2_COLLECTION], ids=[scene_id], max_items=1)
    items = list(res.items())
    if not items:
        raise RuntimeError(f"Scene not found in STAC: {scene_id}")
    it = items[0]
    return {
        "id":          it.id,
        "datetime":    it.properties["datetime"][:10],
        "cloud_cover": it.properties.get("eo:cloud_cover", -1),
        "platform":    it.properties.get("platform", "?"),
        "assets":      {k: v.href for k, v in it.assets.items()},
    }


def load_scene(scene_id: str, bbox: tuple) -> dict:
    """
    Download blue/green/red/nir BOA reflectance over *bbox* for *scene_id*.

    Returns dict with:
        bands     : dict of {band: (H,W) float32 reflectance}
        transform : rasterio Affine (in WGS-84)
        crs       : EPSG:4326
        meta      : scene metadata (id, date, cloud, platform)
    """
    meta = _resolve_assets(scene_id)
    log.info("Loading %s  date=%s  cloud=%.2f%%  platform=%s",
             meta["id"], meta["datetime"], meta["cloud_cover"], meta["platform"])

    lon_min, lat_min, lon_max, lat_max = bbox
    bands: dict[str, np.ndarray] = {}
    out_transform = None
    out_shape = None

    for band, asset_key in _BAND_ASSETS.items():
        href = meta["assets"].get(asset_key)
        if href is None:
            raise RuntimeError(f"No '{asset_key}' asset in {scene_id}")
        with rasterio.open(href) as src:
            # Project bbox corners into the scene CRS (UTM)
            from pyproj import Transformer
            tf = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
            x0, y0 = tf.transform(lon_min, lat_min)
            x1, y1 = tf.transform(lon_max, lat_max)
            row0, col0 = src.index(x0, y1)   # upper-left
            row1, col1 = src.index(x1, y0)   # lower-right
            row0, row1 = sorted((row0, row1))
            col0, col1 = sorted((col0, col1))
            # Clamp to valid raster bounds (bbox may extend beyond the tile)
            row0 = max(0, row0);  row1 = min(src.height, row1)
            col0 = max(0, col0);  col1 = min(src.width,  col1)
            window = Window.from_slices((row0, row1), (col0, col1))
            data = src.read(1, window=window).astype(np.float32) / 10000.0  # DN→reflectance
            bands[band] = data
            if out_shape is None:
                out_shape = data.shape
                # Build a WGS-84 transform for the output grid
                out_transform = from_bounds(lon_min, lat_min, lon_max, lat_max,
                                            data.shape[1], data.shape[0])

    # Align shapes (bands can differ by 1 px due to rounding)
    min_h = min(b.shape[0] for b in bands.values())
    min_w = min(b.shape[1] for b in bands.values())
    for k in bands:
        bands[k] = bands[k][:min_h, :min_w]
    out_transform = from_bounds(lon_min, lat_min, lon_max, lat_max, min_w, min_h)

    log.info("  Loaded %d×%d px  bands=%s", min_h, min_w, list(bands.keys()))
    return {"bands": bands, "transform": out_transform, "crs": "EPSG:4326", "meta": meta}


# ─────────────────────────────────────────────────────────────────────────────
# 2. Clarity measurement at GPS reef
# ─────────────────────────────────────────────────────────────────────────────

def measure_clarity(scene: dict, lon: float, lat: float, win: int = 3) -> dict:
    """
    Measure real water clarity at (lon, lat) from the scene pixels.

    Returns dict with:
        nir_glint      : mean NIR reflectance in window (low = clean, high = glint)
        blue, green, red : mean BOA reflectance at the reef
        bottom_contrast: (blue - green) — strong bottom signal deepens this gap
        valid          : whether the point fell inside the raster
    """
    bands     = scene["bands"]
    transform = scene["transform"]
    h, w = bands["blue"].shape
    row, col = rowcol(transform, lon, lat)
    if not (0 <= row < h and 0 <= col < w):
        return {"valid": False}

    half = win // 2
    r0, r1 = max(0, row - half), min(h, row + half + 1)
    c0, c1 = max(0, col - half), min(w, col + half + 1)

    def _m(b):
        return float(np.nanmean(bands[b][r0:r1, c0:c1]))

    blue, green, red, nir = _m("blue"), _m("green"), _m("red"), _m("nir")
    return {
        "valid":           True,
        "row":             int(row),
        "col":             int(col),
        "nir_glint":       round(nir, 5),
        "blue":            round(blue, 5),
        "green":           round(green, 5),
        "red":             round(red, 5),
        "bottom_contrast": round(blue - green, 5),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. Lyzenga depth-invariant index
# ─────────────────────────────────────────────────────────────────────────────

def load_depth_on_grid(scene: dict, emodnet_path: Path = EMODNET_DEPTH) -> np.ndarray | None:
    """
    Resample the EMODnet depth raster onto the scene's pixel grid.
    Returns (H,W) depth array (negative = below MSL), or None if unavailable.
    """
    if not emodnet_path.exists():
        log.warning("EMODnet depth not found at %s — depth masking disabled", emodnet_path)
        return None
    from rasterio.warp import reproject, Resampling
    h, w = scene["bands"]["blue"].shape
    dst = np.full((h, w), np.nan, dtype=np.float32)
    with rasterio.open(emodnet_path) as src:
        src_arr = src.read(1).astype(np.float32)
        src_nd  = src.nodata if src.nodata is not None else -9999.0
        src_arr[src_arr == src_nd] = np.nan
        reproject(
            source=src_arr, destination=dst,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=scene["transform"], dst_crs=scene["crs"],
            resampling=Resampling.bilinear,
        )
    return dst


def water_mask(bands: dict, depth: np.ndarray | None = None,
               nir_max: float = GLINT_NIR_MAX) -> np.ndarray:
    """
    Boolean mask of clean, optically-shallow water:
      - low NIR (no land, no sun glint)
      - positive blue/green
      - depth within the optical window [DEPTH_DEEP, DEPTH_SHALLOW] if depth given
    """
    nir, blue, green = bands["nir"], bands["blue"], bands["green"]
    m = (nir < nir_max) & (nir >= 0) & (blue > 0) & (green > 0)
    if depth is not None:
        m = m & (depth >= DEPTH_DEEP) & (depth <= DEPTH_SHALLOW)
    return m


def _attenuation_ratio(xi: np.ndarray, xj: np.ndarray) -> float:
    """
    Lyzenga (1978) automatic estimate of the attenuation-coefficient ratio
    k_i/k_j from the variance and covariance of the log-transformed bands:

        a       = (var(Xi) - var(Xj)) / (2·cov(Xi, Xj))
        k_i/k_j = a + sqrt(a² + 1)
    """
    vi = np.var(xi)
    vj = np.var(xj)
    cov = np.mean((xi - xi.mean()) * (xj - xj.mean()))
    if abs(cov) < 1e-9:
        return 1.0
    a = (vi - vj) / (2.0 * cov)
    return float(a + np.sqrt(a * a + 1.0))


def compute_dii(scene: dict, depth: np.ndarray | None = None) -> dict:
    """
    Compute the Lyzenga depth-invariant index DII over the scene.

    Returns dict with:
        dii         : (H,W) float32, NaN where not clean water
        mask        : water mask
        lsw         : deep-water background per band
        k_ratio     : estimated k_blue/k_green
    """
    bands = scene["bands"]
    # Shallow zone where we actually classify substrate
    mask  = water_mask(bands, depth)
    # Full water (all depths) for deep-water background estimation
    full_water = water_mask(bands, depth=None)

    blue, green = bands["blue"], bands["green"]

    # Deep-water background Lsw = 1st percentile over ALL clean water (darkest = deepest)
    lsw_blue  = float(np.percentile(blue[full_water],  1))
    lsw_green = float(np.percentile(green[full_water], 1))

    # Log transform; clip to avoid log of non-positive after background subtraction
    eps = 1e-6
    x_blue  = np.full(blue.shape,  np.nan, dtype=np.float64)
    x_green = np.full(green.shape, np.nan, dtype=np.float64)
    db = blue  - lsw_blue
    dg = green - lsw_green
    ok = mask & (db > eps) & (dg > eps)
    x_blue[ok]  = np.log(db[ok])
    x_green[ok] = np.log(dg[ok])

    # Auto attenuation ratio from the joint distribution of clean-water pixels
    k_ratio = _attenuation_ratio(x_blue[ok], x_green[ok])

    dii = np.full(blue.shape, np.nan, dtype=np.float32)
    dii[ok] = (x_blue[ok] - k_ratio * x_green[ok]).astype(np.float32)

    log.info("  Lyzenga: Lsw_blue=%.5f Lsw_green=%.5f  k_blue/k_green=%.4f  water_px=%d",
             lsw_blue, lsw_green, k_ratio, int(ok.sum()))
    return {"dii": dii, "mask": ok, "lsw": {"blue": lsw_blue, "green": lsw_green},
            "k_ratio": k_ratio}


# ─────────────────────────────────────────────────────────────────────────────
# 4. Texture (rugosity proxy)
# ─────────────────────────────────────────────────────────────────────────────

def local_stats(arr: np.ndarray, mask: np.ndarray, win: int = 31) -> tuple[np.ndarray, np.ndarray]:
    """
    NaN-aware local mean and standard deviation of *arr* within a win×win window.
    Computed via box filters using the E[x²]-E[x]² identity, ignoring masked-out
    pixels. Returns (mean, std), both NaN where insufficient valid neighbours.
    """
    from scipy.ndimage import uniform_filter
    a = np.where(mask, arr, 0.0).astype(np.float64)
    valid = mask.astype(np.float64)
    s1  = uniform_filter(a,       size=win, mode="constant")
    s2  = uniform_filter(a * a,   size=win, mode="constant")
    cnt = uniform_filter(valid,   size=win, mode="constant")
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = s1 / cnt
        var  = s2 / cnt - mean * mean
    var = np.clip(var, 0, None)
    std = np.sqrt(var)
    too_few = cnt < 0.15
    mean[too_few] = np.nan
    std[too_few]  = np.nan
    return mean.astype(np.float32), std.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Reef-likelihood combination
# ─────────────────────────────────────────────────────────────────────────────

def _robust_minmax(arr: np.ndarray, lo_pct: float = 2, hi_pct: float = 98) -> np.ndarray:
    """Scale to [0,1] using robust percentiles; NaN stays NaN."""
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return arr
    lo = np.percentile(finite, lo_pct)
    hi = np.percentile(finite, hi_pct)
    if hi - lo < 1e-9:
        return np.zeros_like(arr)
    out = (arr - lo) / (hi - lo)
    return np.clip(out, 0, 1)


def reef_likelihood(scene: dict, dii_result: dict, bg_win: int = 31) -> dict:
    """
    Detect reef as a LOCAL DARK ANOMALY in the green bottom signal.

    The diagnostic showed the confirmed reef sits at the ~10th percentile of green
    reflectance within its own ~800 m neighbourhood — it is darker than the
    surrounding sand at comparable depth. A global threshold can't capture this
    because brightness varies with depth across the shelf; a LOCAL contrast does.

    For each shallow-water pixel:
        darkness_z = (local_mean_green - green) / local_std_green
    Positive z = darker than local surroundings = reef-like. The large background
    window (bg_win px) removes the slow depth/brightness gradient, isolating
    compact dark structures (rock/reef) from smooth sand.

    Structure gate: reef is also spatially rough, so we multiply by a normalised
    fine-scale texture term to suppress isolated dark speckles.

    Returns dict with likelihood map (0–1) and the anchor diagnostics.
    """
    bands = scene["bands"]
    green = bands["green"]
    mask  = dii_result["mask"]            # shallow, clean-water pixels
    transform = scene["transform"]
    h, w = green.shape

    # Local background statistics (large window removes depth gradient)
    bg_mean, bg_std = local_stats(green, mask, win=bg_win)

    # Dark-anomaly z-score: how many local sigmas darker than surroundings
    eps = 1e-6
    darkness_z = (bg_mean - green) / (bg_std + eps)
    darkness_z[~mask] = np.nan

    # Fine-scale structure (3-px std) — reef is rough, sand is smooth
    _, fine_std = local_stats(green, mask, win=3)
    tex_norm = _robust_minmax(fine_std)

    z_norm = _robust_minmax(darkness_z)

    # Combine: dark AND structured. Geometric mean penalises pixels weak in either.
    likelihood = np.sqrt(np.clip(z_norm, 0, 1) * np.clip(tex_norm, 0, 1))
    likelihood[~mask] = np.nan

    # Anchor diagnostics at the GPS reef
    anchor_z = []
    for site in REEF_SITES:
        row, col = rowcol(transform, float(site["lon"]), float(site["lat"]))
        if 0 <= row < h and 0 <= col < w:
            v = np.nanmean(darkness_z[max(0, row-1):row+2, max(0, col-1):col+2])
            if np.isfinite(v):
                anchor_z.append(float(v))
    anchor_darkness_z = float(np.mean(anchor_z)) if anchor_z else float("nan")

    log.info("  Local dark-anomaly detector  bg_win=%dpx (~%dm)  anchor darkness_z=%.2f",
             bg_win, bg_win * 10, anchor_darkness_z)
    return {
        "likelihood":   likelihood.astype(np.float32),
        "darkness_z":   darkness_z.astype(np.float32),
        "z_norm":       z_norm.astype(np.float32),
        "tex_norm":     tex_norm.astype(np.float32),
        "anchor_darkness_z": round(anchor_darkness_z, 3),
        "bg_win_px":    bg_win,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. Validation at GPS anchors
# ─────────────────────────────────────────────────────────────────────────────

def validate_at_gps(scene: dict, like_result: dict) -> dict:
    """
    Report the likelihood value AND its scene percentile at each GPS reef.
    A working detector puts the confirmed reef in a high percentile.
    """
    like = like_result["likelihood"]
    transform = scene["transform"]
    h, w = like.shape
    finite = like[np.isfinite(like)]

    results: list[dict] = []
    for site in REEF_SITES:
        row, col = rowcol(transform, float(site["lon"]), float(site["lat"]))
        if not (0 <= row < h and 0 <= col < w):
            results.append({"site": str(site["name"]), "in_bbox": False})
            continue
        val = float(np.nanmean(like[max(0, row-1):row+2, max(0, col-1):col+2]))
        pct = float((finite < val).mean() * 100) if finite.size else 0.0
        results.append({
            "site":       str(site["name"]),
            "in_bbox":    True,
            "likelihood": round(val, 4),
            "percentile": round(pct, 1),
        })
        log.info("  Anchor %s: likelihood=%.3f (%.1fth percentile of scene)",
                 site["name"], val, pct)
    return {"anchors": results}


# ─────────────────────────────────────────────────────────────────────────────
# 7. Output (GeoTIFF + overlay PNG)
# ─────────────────────────────────────────────────────────────────────────────

def write_geotiff(arr: np.ndarray, scene: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = arr.shape
    profile = {
        "driver": "GTiff", "height": h, "width": w, "count": 1,
        "dtype": "float32", "crs": scene["crs"], "transform": scene["transform"],
        "nodata": -9999.0, "compress": "deflate",
    }
    out = np.where(np.isfinite(arr), arr, -9999.0).astype(np.float32)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(out, 1)
    log.info("  Wrote %s", path)


def write_overlay_png(scene: dict, like: np.ndarray, path: Path) -> None:
    """RGB true-colour with reef-likelihood contours/heat overlaid."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib unavailable — skipping overlay PNG")
        return

    bands = scene["bands"]
    rgb = np.dstack([bands["red"], bands["green"], bands["blue"]])
    rgb = np.clip(rgb / np.nanpercentile(rgb, 99), 0, 1)   # stretch

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(rgb)
    axes[0].set_title(f"{scene['meta']['datetime']} — True colour (B04/B03/B02)")
    axes[0].axis("off")

    axes[1].imshow(rgb)
    hm = axes[1].imshow(np.ma.masked_invalid(like), cmap="inferno", alpha=0.55, vmin=0, vmax=1)
    axes[1].set_title("Reef likelihood (GPS-anchored)")
    axes[1].axis("off")

    # Mark GPS reef sites
    transform = scene["transform"]
    h, w = like.shape
    for site in REEF_SITES:
        row, col = rowcol(transform, site["lon"], site["lat"])
        if 0 <= row < h and 0 <= col < w:
            for ax in axes:
                ax.plot(col, row, "co", ms=10, mfc="none", mew=2)
            axes[1].annotate(str(site["name"]).replace("_", " "), (col, row),
                             color="cyan", fontsize=8, xytext=(col+5, row-5))

    fig.colorbar(hm, ax=axes[1], fraction=0.046, pad=0.04, label="likelihood")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("  Wrote %s", path)


# ─────────────────────────────────────────────────────────────────────────────
# 8. Main
# ─────────────────────────────────────────────────────────────────────────────

def process_scene(scene_id: str, bbox: tuple) -> dict:
    """Full pipeline for one scene → clarity + likelihood map + validation."""
    scene = load_scene(scene_id, bbox)
    depth = load_depth_on_grid(scene)
    if depth is not None:
        shallow_px = int(((depth >= DEPTH_DEEP) & (depth <= DEPTH_SHALLOW)).sum())
        log.info("  Depth grid loaded; %d px in optical window [%.0f, %.0f m]",
                 shallow_px, DEPTH_DEEP, DEPTH_SHALLOW)

    # Clarity at each GPS reef
    clarity = {}
    for site in REEF_SITES:
        c = measure_clarity(scene, float(site["lon"]), float(site["lat"]))
        clarity[str(site["name"])] = c
        if c.get("valid"):
            log.info("  Clarity @ %s: NIR_glint=%.4f  blue=%.4f green=%.4f red=%.4f  bottom_contrast=%.4f",
                     site["name"], c["nir_glint"], c["blue"], c["green"], c["red"], c["bottom_contrast"])

    dii_result  = compute_dii(scene, depth)
    like_result = reef_likelihood(scene, dii_result)
    validation  = validate_at_gps(scene, like_result)

    # Outputs
    date = scene["meta"]["datetime"]
    write_geotiff(like_result["likelihood"], scene, OUTPUT_DIR / f"reef_likelihood_{date}.tif")
    write_geotiff(dii_result["dii"],          scene, OUTPUT_DIR / f"dii_{date}.tif")
    write_overlay_png(scene, like_result["likelihood"], OUTPUT_DIR / f"overlay_{date}.png")

    return {
        "scene":      scene["meta"],
        "clarity":    clarity,
        "k_ratio":    round(dii_result["k_ratio"], 4),
        "anchor_darkness_z": like_result["anchor_darkness_z"],
        "bg_win_px":  like_result["bg_win_px"],
        "validation": validation,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Reef detector — Lyzenga DII + texture")
    parser.add_argument("--scene", action="append", required=True,
                        help="Sentinel-2 scene ID (repeat for multiple)")
    parser.add_argument("--bbox", nargs=4, type=float, default=list(DEFAULT_BBOX),
                        metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"))
    args = parser.parse_args()

    bbox = tuple(args.bbox)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    reports = []
    for scene_id in args.scene:
        print(f"\n{'='*70}\nPROCESSING {scene_id}\n{'='*70}")
        try:
            reports.append(process_scene(scene_id, bbox))
        except Exception as exc:
            log.error("Failed on %s: %s", scene_id, exc, exc_info=True)

    # Cross-scene clarity comparison
    print(f"\n{'='*70}\nCLARITY COMPARISON (real pixels at GPS reef)\n{'='*70}")
    for r in reports:
        c = r["clarity"].get("Pedra_Sta_Eulalia", {})
        if c.get("valid"):
            print(f"  {r['scene']['datetime']}  ({r['scene']['platform']}):  "
                  f"NIR_glint={c['nir_glint']:.4f}  bottom_contrast={c['bottom_contrast']:.4f}  "
                  f"cloud={r['scene']['cloud_cover']:.2f}%")
    print(f"\n  Lower NIR_glint = cleaner water surface (less sun glint).")
    print(f"  Higher bottom_contrast = stronger seafloor signal reaching the sensor.")

    print(f"\n{'='*70}\nDETECTOR VALIDATION AT GPS REEF ANCHORS\n{'='*70}")
    for r in reports:
        print(f"  {r['scene']['datetime']}:")
        for a in r["validation"]["anchors"]:
            if a.get("in_bbox"):
                print(f"    {a['site']:22s} likelihood={a['likelihood']:.3f}  "
                      f"({a['percentile']:.1f}th percentile)")

    report_path = OUTPUT_DIR / "detector_report.json"
    with open(report_path, "w") as f:
        json.dump(reports, f, indent=2)
    print(f"\nReport: {report_path}")
    print(f"Maps:   {OUTPUT_DIR}/reef_likelihood_*.tif  +  overlay_*.png")


if __name__ == "__main__":
    main()
