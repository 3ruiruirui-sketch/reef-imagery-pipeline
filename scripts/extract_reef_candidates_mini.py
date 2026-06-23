#!/usr/bin/env python3
"""
extract_reef_candidates_mini.py — Reef-candidate miner (mini, SAFE MODE).

Reads the existing EMODnet TRI and depth rasters produced by the baseline
pipeline (``scripts/build_reef_dataset.py``) and emits a curated list of
candidate reef GPS locations for human review.

WHY THIS EXISTS
---------------
The baseline patch classifier is trained with only N=2 GPS-confirmed reef
sites, which is severely data-starved. The EMODnet TRI raster encodes
seafloor structural complexity at ~115 m resolution — high TRI correlates
with rocky reef. By raising the TRI threshold (above the baseline's 0.30 m)
and spatially clustering the surviving pixels, we can mine ~20–100 candidate
reef locations that a human can later curate and feed back into the training
set. The candidates are written under ``outputs/reef_habitat_mini/`` and are
NEVER auto-injected into ``scripts/build_reef_dataset.REEF_SITES``.

INPUTS (CLI args)
-----------------
All inputs are read-only. Outputs are isolated in ``outputs/reef_habitat_mini/``.

OUTPUTS
-------
- reef_candidates.csv      : cluster_id, lon, lat, mean_tri_m, mean_depth_m,
                             area_m2, n_pixels
- reef_candidates.geojson  : Point features, same fields (QGIS / Leaflet ready)
- reef_candidates_log.txt  : human-readable summary (thresholds, counts,
                             top-10 by TRI)
- candidates_overlay.png   : optional PNG (requires matplotlib)

USAGE
-----
    python scripts/extract_reef_candidates_mini.py \\
        --tri outputs/reef_habitat/emodnet/tri_algarve.tif \\
        --depth outputs/reef_habitat/emodnet/depth_algarve.tif \\
        --tri-min 0.8 --max-depth-m 25 --top 50 --min-cluster-px 3

SAFE MODE
---------
This script does NOT modify, overwrite, rename, move, or delete any
existing file. All outputs go to ``outputs/reef_habitat_mini/candidates/``
which is created if missing.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import logging
import sys
from collections import deque
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import rasterio
from rasterio.transform import rowcol

# ─────────────────────────────────────────────────────────────────────────────
# Project root on sys.path so that ``--reef-sites-module scripts.build_reef_dataset``
# can be imported even when the script is invoked from elsewhere.
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("extract_reef_candidates_mini")


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
# EMODnet mean grid is 1/16 arc-minute ≈ 115 m at 37 °N.
EMODNET_PX_M = 115.0
EMODNET_PX_AREA_M2 = EMODNET_PX_M * EMODNET_PX_M  # ≈ 13,225 m²


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Candidate:
    """A single reef-candidate cluster summary."""
    cluster_id: int
    lon: float
    lat: float
    mean_tri_m: float
    mean_depth_m: float
    area_m2: float
    n_pixels: int


# ─────────────────────────────────────────────────────────────────────────────
# Connected-components labelling (4-connectivity, inline BFS)
# ─────────────────────────────────────────────────────────────────────────────
def label_connected_components(mask: np.ndarray) -> np.ndarray:
    """
    Label connected components in a boolean mask using 4-connectivity BFS.

    Parameters
    ----------
    mask : 2-D boolean ndarray. ``True`` pixels belong to the foreground.

    Returns
    -------
    labels : int32 ndarray of the same shape. Background pixels are 0.
             Foreground components are labelled 1, 2, 3, ... in BFS-discovery
             order (top-to-bottom, left-to-right).
    """
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    if not mask.any():
        return labels

    next_label = 0
    # Pre-stack candidate seeds, scan row-major for determinism.
    rows, cols = np.where(mask)
    visited = np.zeros((h, w), dtype=bool)

    for r0, c0 in zip(rows.tolist(), cols.tolist()):
        if visited[r0, c0]:
            continue
        next_label += 1
        queue: deque[Tuple[int, int]] = deque([(r0, c0)])
        visited[r0, c0] = True
        labels[r0, c0] = next_label
        while queue:
            r, c = queue.popleft()
            # 4-connectivity
            if r > 0 and mask[r - 1, c] and not visited[r - 1, c]:
                visited[r - 1, c] = True
                labels[r - 1, c] = next_label
                queue.append((r - 1, c))
            if r + 1 < h and mask[r + 1, c] and not visited[r + 1, c]:
                visited[r + 1, c] = True
                labels[r + 1, c] = next_label
                queue.append((r + 1, c))
            if c > 0 and mask[r, c - 1] and not visited[r, c - 1]:
                visited[r, c - 1] = True
                labels[r, c - 1] = next_label
                queue.append((r, c - 1))
            if c + 1 < w and mask[r, c + 1] and not visited[r, c + 1]:
                visited[r, c + 1] = True
                labels[r, c + 1] = next_label
                queue.append((r, c + 1))

    return labels


# ─────────────────────────────────────────────────────────────────────────────
# Raster helpers
# ─────────────────────────────────────────────────────────────────────────────
def read_raster(path: Path) -> Tuple[np.ndarray, float, object, object]:
    """
    Read a single-band raster as float32.

    Returns
    -------
    data      : ndarray of float32
    nodata    : float nodata sentinel (or NaN if undefined)
    transform : rasterio affine transform
    crs       : rasterio CRS
    """
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        nodata = src.nodata
        if nodata is None:
            nodata = float("nan")
        nodata = float(nodata)
        transform = src.transform
        crs = src.crs
    return data, nodata, transform, crs


def build_candidate_mask(
    tri: np.ndarray,
    tri_nodata: float,
    depth: np.ndarray,
    depth_nodata: float,
    tri_min: float,
    max_depth_m: float,
) -> np.ndarray:
    """
    Build a boolean reef-candidate mask.

    True where:
        - TRI has valid data AND TRI >= tri_min
        - depth has valid data AND depth in [-max_depth_m, 0]
    """
    tri_valid = ~np.isnan(tri) & ~np.isnan(tri_nodata) & (tri != tri_nodata)
    depth_valid = ~np.isnan(depth) & ~np.isnan(depth_nodata) & (depth != depth_nodata)

    in_window = depth_valid & (depth >= -max_depth_m) & (depth <= 0.0)
    rough = tri_valid & (tri >= tri_min)

    return rough & in_window


def build_exclusion_mask(
    shape: Tuple[int, int],
    transform,
    reef_sites: Sequence[dict],
    buffer_deg: float,
) -> np.ndarray:
    """
    Return a boolean mask of pixels to EXCLUDE (True = exclude).

    Each REEF_SITES coordinate is mapped to a pixel, and all pixels within
    a square ``±ceil(buffer_deg / pixel_deg)`` window around that pixel are
    flagged. The buffer is approximate (square, not circular) — adequate
    for ~1 km exclusion around a known reef.
    """
    h, w = shape
    px_deg = abs(transform.a)  # pixel width in degrees (always positive)
    py_deg = abs(transform.e)  # pixel height in degrees (positive)
    r_buf = int(np.ceil(buffer_deg / max(py_deg, 1e-9)))
    c_buf = int(np.ceil(buffer_deg / max(px_deg, 1e-9)))

    exclude = np.zeros((h, w), dtype=bool)
    for site in reef_sites:
        try:
            row, col = rowcol(transform, site["lon"], site["lat"])
        except Exception:
            log.warning("Could not map reef site %s to pixel; skipping", site.get("name"))
            continue
        row = int(row)
        col = int(col)
        if not (0 <= row < h and 0 <= col < w):
            log.info("Reef site %s outside raster; skipping buffer", site.get("name"))
            continue
        r0 = max(0, row - r_buf)
        r1 = min(h, row + r_buf + 1)
        c0 = max(0, col - c_buf)
        c1 = min(w, col + c_buf + 1)
        exclude[r0:r1, c0:c1] = True

    return exclude


def summarize_components(
    labels: np.ndarray,
    tri: np.ndarray,
    depth: np.ndarray,
    transform,
    min_pixels: int,
) -> List[Candidate]:
    """
    For each labelled component ≥ ``min_pixels``, compute centroid in lon/lat,
    mean TRI, mean depth, area, and pixel count.
    """
    if labels.max() == 0:
        return []

    candidates: List[Candidate] = []
    for cid in range(1, int(labels.max()) + 1):
        rows, cols = np.where(labels == cid)
        n = int(rows.size)
        if n < min_pixels:
            continue

        mean_tri = float(np.nanmean(tri[rows, cols])) if n else 0.0
        mean_depth = float(np.nanmean(depth[rows, cols])) if n else 0.0

        # Centroid: use the mean pixel coordinate, then map that pixel to lon/lat.
        r_mean = float(rows.mean())
        c_mean = float(cols.mean())
        lon, lat = rasterio.transform.xy(transform, r_mean, c_mean, offset="center")

        candidates.append(
            Candidate(
                cluster_id=cid,
                lon=float(lon),
                lat=float(lat),
                mean_tri_m=mean_tri,
                mean_depth_m=mean_depth,
                area_m2=float(n) * EMODNET_PX_AREA_M2,
                n_pixels=n,
            )
        )

    return candidates


# ─────────────────────────────────────────────────────────────────────────────
# Output writers
# ─────────────────────────────────────────────────────────────────────────────
def write_csv(candidates: Sequence[Candidate], path: Path) -> None:
    """Write candidates to a CSV file."""
    fieldnames = ["cluster_id", "lon", "lat", "mean_tri_m",
                  "mean_depth_m", "area_m2", "n_pixels"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for c in candidates:
            w.writerow(asdict(c))


def write_geojson(candidates: Sequence[Candidate], path: Path) -> None:
    """Write candidates as Point features to a GeoJSON file."""
    features = []
    for c in candidates:
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [c.lon, c.lat]},
            "properties": {
                "cluster_id":   c.cluster_id,
                "mean_tri_m":   round(c.mean_tri_m, 4),
                "mean_depth_m": round(c.mean_depth_m, 4),
                "area_m2":      round(c.area_m2, 1),
                "n_pixels":     c.n_pixels,
            },
        })
    fc = {"type": "FeatureCollection", "features": features}
    path.write_text(json.dumps(fc, indent=2))


def write_log(
    path: Path,
    *,
    args: argparse.Namespace,
    n_pixels_total: int,
    n_pixels_after_exclusion: int,
    n_components: int,
    n_components_after_min: int,
    candidates: Sequence[Candidate],
    reef_sites: Sequence[dict],
) -> None:
    """Write a human-readable summary log."""
    lines: List[str] = []
    lines.append("Reef-candidate mining — log")
    lines.append("=" * 60)
    lines.append(f"  TRI raster       : {args.tri}")
    lines.append(f"  depth raster     : {args.depth}")
    lines.append(f"  reef-sites module: {args.reef_sites_module}")
    lines.append(f"  out-dir          : {args.out_dir}")
    lines.append("")
    lines.append("Thresholds / parameters")
    lines.append("-" * 60)
    lines.append(f"  tri-min          : {args.tri_min} m")
    lines.append(f"  max-depth-m      : {args.max_depth_m} m")
    lines.append(f"  min-cluster-px   : {args.min_cluster_px}")
    lines.append(f"  top              : {args.top}")
    lines.append(f"  exclude-buffer   : {args.exclude_buffer_deg} deg")
    lines.append(f"  EMODNET pixel    : {EMODNET_PX_M} m  (area={EMODNET_PX_AREA_M2:.0f} m²)")
    lines.append("")
    lines.append("Excluded reef sites")
    lines.append("-" * 60)
    for s in reef_sites:
        lines.append(f"  {s.get('name','?'):24s}  lon={s['lon']:.4f}  lat={s['lat']:.4f}")
    lines.append("")
    lines.append("Filtering funnel")
    lines.append("-" * 60)
    lines.append(f"  raw candidate pixels           : {n_pixels_total}")
    lines.append(f"  pixels after excluding reefs   : {n_pixels_after_exclusion}")
    lines.append(f"  connected components           : {n_components}")
    lines.append(f"  components ≥ min-cluster-px    : {n_components_after_min}")
    lines.append(f"  candidates written             : {len(candidates)}")
    lines.append("")
    lines.append("Top 10 candidates by mean TRI")
    lines.append("-" * 60)
    lines.append(f"  {'rank':>4s}  {'cid':>5s}  {'lon':>10s}  {'lat':>10s}  "
                 f"{'tri_m':>7s}  {'depth_m':>8s}  {'area_m2':>10s}  {'px':>4s}")
    for i, c in enumerate(candidates[:10], start=1):
        lines.append(
            f"  {i:>4d}  {c.cluster_id:>5d}  {c.lon:>10.5f}  {c.lat:>10.5f}  "
            f"{c.mean_tri_m:>7.3f}  {c.mean_depth_m:>8.3f}  "
            f"{c.area_m2:>10.0f}  {c.n_pixels:>4d}"
        )
    lines.append("")
    path.write_text("\n".join(lines))


def write_overlay_png(
    tri_path: Path,
    candidates: Sequence[Candidate],
    out_png: Path,
) -> bool:
    """
    Render a quick PNG overlay of the TRI raster with candidate points.

    Returns True on success, False if matplotlib is unavailable.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        log.warning("matplotlib not available — skipping PNG overlay: %s", e)
        return False

    try:
        with rasterio.open(tri_path) as src:
            tri = src.read(1).astype(np.float32)
            nodata = src.nodata
            if nodata is None:
                nodata = float("nan")
            extent = (
                src.bounds.left, src.bounds.right,
                src.bounds.bottom, src.bounds.top,
            )

        # Mask nodata for display
        tri_disp = np.where(tri == nodata, np.nan, tri)

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(tri_disp, extent=extent, origin="upper",
                       cmap="viridis", interpolation="nearest")
        if candidates:
            lons = [c.lon for c in candidates]
            lats = [c.lat for c in candidates]
            ax.scatter(lons, lats, s=22, edgecolor="red",
                       facecolor="none", linewidths=1.2, label="candidate")
            ax.legend(loc="lower right", fontsize=8)
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")
        ax.set_title(f"EMODnet TRI + reef candidates (n={len(candidates)})")
        cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cbar.set_label("TRI (m)")
        fig.tight_layout()
        fig.savefig(out_png, dpi=120)
        plt.close(fig)
        return True
    except Exception as e:
        log.warning("PNG overlay failed: %s", e)
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Reef-site import
# ─────────────────────────────────────────────────────────────────────────────
def load_reef_sites(module_path: str) -> List[dict]:
    """
    Import the REEF_SITES list from a module path (e.g. ``scripts.build_reef_dataset``).

    Returns a list of ``{"name", "lon", "lat"}`` dicts. Falls back to an empty
    list if the module cannot be imported or has no REEF_SITES attribute.
    """
    try:
        mod = importlib.import_module(module_path)
    except Exception as e:
        log.warning("Could not import reef-sites module '%s': %s — no exclusion applied",
                    module_path, e)
        return []

    sites = getattr(mod, "REEF_SITES", None)
    if not sites:
        log.warning("Module '%s' has no REEF_SITES attribute — no exclusion applied",
                    module_path)
        return []

    cleaned: List[dict] = []
    for s in sites:
        if "lon" in s and "lat" in s:
            cleaned.append({
                "name": s.get("name", "?"),
                "lon":  float(s["lon"]),
                "lat":  float(s["lat"]),
            })
    log.info("Loaded %d reef sites from %s for exclusion", len(cleaned), module_path)
    return cleaned


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────
def mine_candidates(args: argparse.Namespace) -> List[Candidate]:
    """Run the full candidate-mining pipeline. Returns the final candidate list."""
    tri_path = Path(args.tri)
    depth_path = Path(args.depth)

    log.info("Reading TRI raster: %s", tri_path)
    tri, tri_nodata, transform, crs = read_raster(tri_path)

    log.info("Reading depth raster: %s", depth_path)
    depth, depth_nodata, depth_transform, depth_crs = read_raster(depth_path)

    if tri.shape != depth.shape:
        raise ValueError(
            f"TRI shape {tri.shape} != depth shape {depth.shape}. "
            "Ensure both rasters are aligned."
        )
    if str(crs) != str(depth_crs) or transform != depth_transform:
        log.warning("TRI/depth CRS or transform mismatch — proceeding with TRI transform")

    log.info(
        "Building mask: TRI >= %.3f m  AND  depth in [-%.1f, 0] m",
        args.tri_min, args.max_depth_m,
    )
    mask = build_candidate_mask(
        tri, tri_nodata, depth, depth_nodata,
        tri_min=args.tri_min,
        max_depth_m=args.max_depth_m,
    )
    n_pixels_total = int(mask.sum())
    log.info("Raw candidate pixels: %d", n_pixels_total)
    if n_pixels_total == 0:
        log.warning("Mask is empty — no candidates produced (try lowering --tri-min)")
        return []

    # Exclusion buffer around known reef sites
    reef_sites = load_reef_sites(args.reef_sites_module)
    if reef_sites and args.exclude_buffer_deg > 0:
        excl = build_exclusion_mask(
            tri.shape, transform, reef_sites, args.exclude_buffer_deg,
        )
        n_excluded = int(excl.sum())
        mask = mask & ~excl
        log.info(
            "Excluded %d pixels within %.4f deg of %d known reef sites; "
            "%d candidate pixels remain",
            n_excluded, args.exclude_buffer_deg, len(reef_sites),
            int(mask.sum()),
        )

    n_pixels_after = int(mask.sum())
    if n_pixels_after == 0:
        log.warning("All candidate pixels were excluded — no candidates produced")
        return []

    # Connected-components labelling
    log.info("Labelling connected components (4-connectivity, inline BFS)")
    labels = label_connected_components(mask)
    n_components = int(labels.max())
    log.info("Connected components: %d", n_components)

    # Summarise and filter by min-cluster-px
    candidates = summarize_components(
        labels, tri, depth, transform, min_pixels=args.min_cluster_px,
    )
    n_after_min = len(candidates)
    log.info("Components ≥ %d pixels: %d", args.min_cluster_px, n_after_min)

    # Sort by mean TRI descending, keep top --top
    candidates.sort(key=lambda c: c.mean_tri_m, reverse=True)
    if len(candidates) > args.top:
        log.info("Keeping top %d of %d candidates by mean TRI", args.top, len(candidates))
        candidates = candidates[: args.top]

    # Persist the funnel counts on the namespace for the log writer.
    args._n_pixels_total = n_pixels_total  # type: ignore[attr-defined]
    args._n_pixels_after = n_pixels_after  # type: ignore[attr-defined]
    args._n_components = n_components       # type: ignore[attr-defined]
    args._n_after_min = n_after_min         # type: ignore[attr-defined]
    args._reef_sites = reef_sites           # type: ignore[attr-defined]

    return candidates


def write_outputs(args: argparse.Namespace, candidates: List[Candidate]) -> None:
    """Write CSV, GeoJSON, log, and optional PNG overlay."""
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "reef_candidates.csv"
    geojson_path = out_dir / "reef_candidates.geojson"
    log_path = out_dir / "reef_candidates_log.txt"
    png_path = out_dir / "candidates_overlay.png"

    write_csv(candidates, csv_path)
    log.info("Wrote CSV: %s (%d rows)", csv_path, len(candidates))

    write_geojson(candidates, geojson_path)
    log.info("Wrote GeoJSON: %s (%d features)", geojson_path, len(candidates))

    write_log(
        log_path,
        args=args,
        n_pixels_total=args._n_pixels_total,         # type: ignore[attr-defined]
        n_pixels_after_exclusion=args._n_pixels_after,  # type: ignore[attr-defined]
        n_components=args._n_components,             # type: ignore[attr-defined]
        n_components_after_min=args._n_after_min,    # type: ignore[attr-defined]
        candidates=candidates,
        reef_sites=args._reef_sites,                 # type: ignore[attr-defined]
    )
    log.info("Wrote log: %s", log_path)

    ok = write_overlay_png(Path(args.tri), candidates, png_path)
    if ok:
        log.info("Wrote PNG overlay: %s", png_path)
    else:
        log.info("PNG overlay skipped (matplotlib unavailable or write failed)")


def print_stdout_summary(candidates: Sequence[Candidate]) -> None:
    """Print the final top-10 summary block to stdout."""
    print("\n── Reef-candidate mining — top 10 by mean TRI ─────────────")
    print(f"  {'rank':>4s}  {'cid':>5s}  {'lon':>10s}  {'lat':>10s}  "
          f"{'mean_tri_m':>10s}")
    for i, c in enumerate(candidates[:10], start=1):
        print(f"  {i:>4d}  {c.cluster_id:>5d}  {c.lon:>10.5f}  {c.lat:>10.5f}  "
              f"{c.mean_tri_m:>10.4f}")
    print("────────────────────────────────────────────────────────────")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    """Construct the argparse parser."""
    p = argparse.ArgumentParser(
        prog="extract_reef_candidates_mini",
        description=(
            "Mine reef candidate GPS locations from the existing EMODnet "
            "TRI raster. SAFE MODE: never modifies existing files."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--tri",
        default="outputs/reef_habitat/emodnet/tri_algarve.tif",
        help="Path to EMODnet TRI GeoTIFF (read-only).",
    )
    p.add_argument(
        "--depth",
        default="outputs/reef_habitat/emodnet/depth_algarve.tif",
        help="Path to EMODnet depth GeoTIFF (read-only).",
    )
    p.add_argument(
        "--reef-sites-module",
        default="scripts.build_reef_dataset",
        help="Python module path exporting REEF_SITES (list of {name,lon,lat}). "
             "Used to EXCLUDE known reefs from candidates.",
    )
    p.add_argument(
        "--tri-min",
        type=float,
        default=0.80,
        help="Minimum TRI (m) for a pixel to count as reef-like. "
             "Hypothesis-to-test: 0.80 m is higher than the baseline's 0.30 m.",
    )
    p.add_argument(
        "--max-depth-m",
        type=float,
        default=25.0,
        help="Maximum optical depth (m). Pixels deeper than this are excluded.",
    )
    p.add_argument(
        "--min-cluster-px",
        type=int,
        default=3,
        help="Minimum number of connected pixels for a candidate cluster.",
    )
    p.add_argument(
        "--top",
        type=int,
        default=50,
        help="Maximum number of candidates to emit (sorted by mean TRI desc).",
    )
    p.add_argument(
        "--out-dir",
        default="outputs/reef_habitat_mini/candidates",
        help="Output directory for candidates (created if missing). "
             "SAFE MODE: must stay under outputs/reef_habitat_mini/.",
    )
    p.add_argument(
        "--exclude-buffer-deg",
        type=float,
        default=0.01,
        help="Buffer radius (degrees) around each known reef site to exclude.",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point. Returns the process exit code."""
    # ── SAFE MODE CHECK — must be the very first line of stdout ─────────────
    print("SAFE MODE CHECK: I will only create new mini-files and will not modify existing files.")

    parser = build_parser()
    args = parser.parse_args(argv)

    # Belt-and-braces: refuse to write into the baseline outputs directory.
    out_dir = Path(args.out_dir).resolve()
    forbidden = (Path("outputs/reef_habitat").resolve(),)
    for bad in forbidden:
        try:
            out_dir.relative_to(bad)
        except ValueError:
            continue
        raise SystemExit(
            f"REFUSED: --out-dir {out_dir} is inside baseline {bad}. "
            "SAFE MODE requires outputs/reef_habitat_mini/."
        )

    # Sanity check inputs exist
    for p in (Path(args.tri), Path(args.depth)):
        if not p.exists():
            log.error("Input raster not found: %s", p)
            return 2

    try:
        candidates = mine_candidates(args)
    except Exception as e:
        log.exception("Mining failed: %s", e)
        return 1

    write_outputs(args, candidates)
    print_stdout_summary(candidates)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())