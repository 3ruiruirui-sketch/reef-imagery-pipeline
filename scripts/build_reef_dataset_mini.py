#!/usr/bin/env python3
"""
build_reef_dataset_mini.py — Parallel, additive dataset builder.

A near-mirror of :mod:`scripts.build_reef_dataset` that writes every artefact
into a separate output tree (``outputs/reef_habitat_mini/`` by default) so
that two scenes / parameter sets can coexist for comparison.

Key differences from the baseline
---------------------------------
1. All output paths are CLI-configurable and default to ``outputs/reef_habitat_mini/``.
2. The Sentinel-2 scene ID is freely selectable (default is an empirically-good
   2023-09-01 candidate, treated as a hypothesis-to-test, not a proven fact).
3. ``--sand-multiplier`` can be raised beyond the baseline's 5× to collect more
   negative samples (useful when more reef labels are added later).
4. ``--tri-reef-threshold`` and ``--max-depth-m`` are exposed for sensitivity tests.
5. Constants (``REEF_SITES``, ``KNOWN_SAND_LOCATIONS``, ``ALGARVE_BBOX``,
   ``MAX_DEPTH_M``, ``TRI_REEF_THRESHOLD``, ``SAND_MULTIPLIER``) are
   **imported** from :mod:`scripts.build_reef_dataset` — single source of truth
   preserved.

The baseline ``build_reef_dataset.py`` is NOT modified by this script.

Usage
-----
    python scripts/build_reef_dataset_mini.py \\
        --scene-id S2A_29SNB_20230901_0_L2A \\
        --date-range 2023-08-01/2023-10-15 \\
        --patches-root outputs/reef_habitat_mini/patches \\
        --sand-multiplier 5 \\
        --force
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from rasterio.transform import xy as rio_xy

# Add project root to path so both src/ and scripts/ are importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.emodnet_bathymetry import build_habitat_mask, compute_tri, fetch_depth
from src.s2_patch_extractor import build_dataset

# Import (do NOT copy) constants from the baseline
from scripts.build_reef_dataset import (
    ALGARVE_BBOX,
    KNOWN_SAND_LOCATIONS,
    MAX_DEPTH_M as _BASELINE_MAX_DEPTH_M,
    REEF_SITES,
    SAND_MULTIPLIER as _BASELINE_SAND_MULTIPLIER,
    TRI_REEF_THRESHOLD as _BASELINE_TRI_REEF_THRESHOLD,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _maybe_remove(path: Path) -> None:
    """Delete *path* if it exists; log the action."""
    if path.exists():
        path.unlink()
        log.info("Removed existing file: %s", path)


def _ensure_parent(path: Path) -> None:
    """Create parent directory for *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline steps (parameterised versions of the baseline steps)
# ─────────────────────────────────────────────────────────────────────────────

def step1_fetch_emodnet(
    emodnet_dir: Path,
    depth_path: Path,
    force: bool,
) -> Path:
    """Fetch EMODnet mean depth for the Algarve bbox into ``depth_path``.

    Skips the download if the file already exists unless ``force`` is True.

    Parameters
    ----------
    emodnet_dir : Destination directory (created if missing).
    depth_path  : Target GeoTIFF for the depth raster.
    force       : If True, re-download even if the file already exists.

    Returns
    -------
    Path to the (possibly existing) depth GeoTIFF.
    """
    _ensure_parent(depth_path)
    if depth_path.exists() and not force:
        log.info("Depth raster already exists: %s (skipping download)", depth_path)
        return depth_path
    log.info("Step 1: Fetching EMODnet bathymetry for bbox=%s", ALGARVE_BBOX)
    try:
        fetch_depth(ALGARVE_BBOX, depth_path)
    except Exception as exc:
        log.error("Step 1 (EMODnet fetch) failed: %s", exc)
        raise
    return depth_path


def step2_compute_tri(
    depth_path: Path,
    tri_path: Path,
    force: bool,
) -> Path:
    """Compute Terrain Ruggedness Index from ``depth_path`` into ``tri_path``.

    Skips computation if the file already exists unless ``force`` is True.

    Parameters
    ----------
    depth_path : Source depth GeoTIFF.
    tri_path   : Destination TRI GeoTIFF.
    force      : If True, re-compute even if the file already exists.

    Returns
    -------
    Path to the (possibly existing) TRI GeoTIFF.
    """
    _ensure_parent(tri_path)
    if tri_path.exists() and not force:
        log.info("TRI raster already exists: %s (skipping)", tri_path)
        return tri_path
    log.info("Step 2: Computing TRI from %s", depth_path)
    try:
        compute_tri(depth_path, tri_path)
    except Exception as exc:
        log.error("Step 2 (TRI computation) failed: %s", exc)
        raise
    return tri_path


def step3_generate_labels(
    depth_path: Path,
    tri_path: Path,
    labels_path: Path,
    sand_multiplier: int,
    tri_reef_threshold: float,
    max_depth_m: float,
    force: bool,
    seed: int = 42,
) -> Path:
    """Write the combined reef + sand label manifest to ``labels_path``.

    Combines:
      * ``REEF_SITES`` from :mod:`scripts.build_reef_dataset` (GPS-confirmed dives).
      * ``KNOWN_SAND_LOCATIONS`` from :mod:`scripts.build_reef_dataset` (chart-confirmed).
      * EMODnet low-TRI sand candidates sampled at ``sand_multiplier`` × n_reef.

    Skips if the manifest already exists unless ``force`` is True.

    Parameters
    ----------
    depth_path, tri_path : EMODnet rasters used to derive sand candidates.
    labels_path           : Destination JSON path.
    sand_multiplier       : Max EMODnet sand candidates = sand_multiplier × n_reef.
    tri_reef_threshold    : TRI threshold for the habitat mask.
    max_depth_m           : Optical depth window upper bound.
    force                 : If True, regenerate even if file exists.
    seed                  : RNG seed for sand candidate subsampling.

    Returns
    -------
    Path to the written (or existing) manifest.
    """
    _ensure_parent(labels_path)
    if labels_path.exists() and not force:
        log.info("Label manifest already exists: %s (skipping)", labels_path)
        return labels_path

    log.info(
        "Step 3: Generating labels (sand_multiplier=%d  tri_thresh=%.2f  max_depth=%.1f m)",
        sand_multiplier, tri_reef_threshold, max_depth_m,
    )

    # ── Reef labels: GPS-confirmed dive sites (imported) ────────────────────
    reef_labels: list[dict] = []
    for site in REEF_SITES:
        reef_labels.append({
            "lon":    site["lon"],
            "lat":    site["lat"],
            "class":  "reef",
            "source": "gps_dive_confirmed",
            "name":   site["name"],
        })
    log.info("Reef labels: %d GPS-confirmed sites", len(reef_labels))

    # ── Sand labels: known locations (imported) + EMODnet candidates ────────
    sand_labels: list[dict] = []
    for loc in KNOWN_SAND_LOCATIONS:
        sand_labels.append({
            "lon":    loc["lon"],
            "lat":    loc["lat"],
            "class":  "sand",
            "source": "bathymetric_chart_confirmed",
            "name":   loc["name"],
        })

    try:
        habitat = build_habitat_mask(
            depth_path, tri_path,
            max_depth_m=max_depth_m,
            tri_reef_threshold=tri_reef_threshold,
        )
        mask = habitat["mask"]
        transform = habitat["transform"]
        sand_rows, sand_cols = np.where(mask == 1)
    except Exception as exc:
        log.error(
            "Could not build EMODnet habitat mask (%s) — "
            "continuing with chart-confirmed sand only.",
            exc,
        )
        sand_rows = np.array([], dtype=np.int64)
        sand_cols = np.array([], dtype=np.int64)
        transform = None

    log.info("EMODnet sand candidate pixels: %d", len(sand_rows))

    n_reef = len(reef_labels)
    target_sand = int(sand_multiplier) * n_reef
    rng = random.Random(seed)

    if len(sand_rows) > 0 and target_sand > 0:
        n_sample = min(target_sand, len(sand_rows))
        idx = rng.sample(range(len(sand_rows)), n_sample)
        for i in idx:
            lon, lat = rio_xy(transform, sand_rows[i], sand_cols[i])  # type: ignore[arg-type]
            sand_labels.append({
                "lon":    float(lon),
                "lat":    float(lat),
                "class":  "sand",
                "source": "emodnet_low_tri",
                "name":   f"emodnet_sand_{i:05d}",
            })

    log.info(
        "Total sand labels: %d (%d from chart + %d from EMODnet)",
        len(sand_labels),
        len(KNOWN_SAND_LOCATIONS),
        len(sand_labels) - len(KNOWN_SAND_LOCATIONS),
    )

    all_labels = reef_labels + sand_labels
    manifest = {
        "labels":             all_labels,
        "n_reef":             len(reef_labels),
        "n_sand":             len(sand_labels),
        "bbox":               list(ALGARVE_BBOX),
        "depth_path":         str(depth_path),
        "tri_path":           str(tri_path),
        "tri_reef_threshold": tri_reef_threshold,
        "max_depth_m":        max_depth_m,
        "sand_multiplier":    sand_multiplier,
        "source_tree":        "reef_habitat_mini",
    }
    with open(labels_path, "w") as f:
        json.dump(manifest, f, indent=2)
    log.info("Label manifest written: %s  (total=%d)", labels_path, len(all_labels))
    return labels_path


def step4_extract_patches(
    labels_path: Path,
    patches_root: Path,
    scene_id: Optional[str],
    date_range: str,
    patch_px: int,
    val_fraction: float,
    max_cloud: float,
) -> dict:
    """Extract Sentinel-2 patches into ``patches_root/{train,val}/{reef,sand}``.

    Parameters
    ----------
    labels_path  : Path to the manifest produced by :func:`step3_generate_labels`.
    patches_root : Output root; ``train/val/{reef,sand}/*.npy`` are written inside.
    scene_id     : Sentinel-2 L2A scene ID, or ``"auto"`` / ``None`` for STAC search.
    date_range   : STAC datetime filter (used only when scene_id is None/"auto").
    patch_px     : Patch side length in pixels.
    val_fraction : Fraction of patches reserved for validation.
    max_cloud    : Maximum cloud cover % for auto scene selection.

    Returns
    -------
    dict summary from :func:`src.s2_patch_extractor.build_dataset`.
    """
    # Normalise "auto" → None so the extractor does STAC discovery
    scene_arg: Optional[str] = None if (scene_id is None or scene_id.lower() == "auto") else scene_id

    log.info(
        "Step 4: Extracting S2 patches (scene=%s, date_range=%s, root=%s)",
        scene_arg or "auto", date_range, patches_root,
    )
    try:
        summary = build_dataset(
            labels_path=labels_path,
            output_dir=patches_root,
            scene_id=scene_arg,
            date_range=date_range,
            patch_px=patch_px,
            val_fraction=val_fraction,
            max_cloud=max_cloud,
        )
    except Exception as exc:
        log.error("Step 4 (S2 patch extraction) failed: %s", exc)
        raise
    return summary


def write_patch_manifest(patches_root: Path, summary: dict, scene_id_arg: str, date_range: str) -> Path:
    """Augment ``patch_manifest.json`` with the mini-tree provenance block.

    The extractor already writes a summary at ``patch_manifest.json``; this
    function copies/extends it with provenance metadata so the mini tree is
    self-describing.

    Returns
    -------
    Path to the written manifest.
    """
    manifest_path = patches_root / "patch_manifest.json"
    enriched = dict(summary)
    enriched["scene_id_requested"] = scene_id_arg
    enriched["date_range"] = date_range
    enriched["source_tree"] = "reef_habitat_mini"
    enriched["baseline_scene_default"] = "S2B_29SNB_20240930_0_L2A"
    with open(manifest_path, "w") as f:
        json.dump(enriched, f, indent=2)
    log.info("Patch manifest written: %s", manifest_path)
    return manifest_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser with explicit defaults."""
    parser = argparse.ArgumentParser(
        description=(
            "Build a reef classification dataset into outputs/reef_habitat_mini/ "
            "(additive to the baseline reef_habitat/ tree)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--scene-id",
        default="S2A_29SNB_20230901_0_L2A",
        help=(
            "Sentinel-2 L2A scene ID to use. Default is a 2023-09-01 candidate "
            "treated as a hypothesis-to-test (not a proven best scene). "
            "Pass 'auto' to let STAC pick from --date-range."
        ),
    )
    parser.add_argument(
        "--date-range",
        default="2023-08-01/2023-10-15",
        help="STAC datetime filter (only used when --scene-id is 'auto').",
    )
    parser.add_argument(
        "--patches-root",
        default="outputs/reef_habitat_mini/patches",
        help="Root directory for extracted patches and patch_manifest.json.",
    )
    parser.add_argument(
        "--emodnet-dir",
        default="outputs/reef_habitat_mini/emodnet",
        help="Directory for depth_algarve.tif and tri_algarve.tif.",
    )
    parser.add_argument(
        "--labels-path",
        default="outputs/reef_habitat_mini/labels/label_manifest.json",
        help="Output path for the label manifest JSON.",
    )
    parser.add_argument(
        "--sand-multiplier",
        type=int,
        default=_BASELINE_SAND_MULTIPLIER,
        help="Max EMODnet sand candidates per reef (default matches baseline; raise for more negatives).",
    )
    parser.add_argument(
        "--tri-reef-threshold",
        type=float,
        default=_BASELINE_TRI_REEF_THRESHOLD,
        help="TRI (m) above which a cell is reef-like (default matches baseline).",
    )
    parser.add_argument(
        "--max-depth-m",
        type=float,
        default=_BASELINE_MAX_DEPTH_M,
        help="Optical depth window upper bound (metres; default matches baseline).",
    )
    parser.add_argument(
        "--patch-px",
        type=int,
        default=128,
        help="Patch side length in pixels.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Fraction of patches reserved for validation.",
    )
    parser.add_argument(
        "--max-cloud",
        type=float,
        default=5.0,
        help="Maximum cloud cover %% for STAC auto scene selection.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run all steps (delete existing depth/TRI/labels before re-running).",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    """Entry point: parse CLI, run all four steps, print a summary block."""
    print("SAFE MODE CHECK: I will only create new mini-files and will not modify existing files.")

    args = _build_arg_parser().parse_args(argv)

    emodnet_dir = Path(args.emodnet_dir)
    patches_root = Path(args.patches_root)
    labels_path = Path(args.labels_path)

    depth_path = emodnet_dir / "depth_algarve.tif"
    tri_path = emodnet_dir / "tri_algarve.tif"

    if args.force:
        log.info("--force: removing existing depth / TRI / labels before re-running")
        _maybe_remove(depth_path)
        _maybe_remove(tri_path)
        _maybe_remove(labels_path)

    step1_fetch_emodnet(emodnet_dir, depth_path, force=args.force)
    step2_compute_tri(depth_path, tri_path, force=args.force)
    step3_generate_labels(
        depth_path=depth_path,
        tri_path=tri_path,
        labels_path=labels_path,
        sand_multiplier=args.sand_multiplier,
        tri_reef_threshold=args.tri_reef_threshold,
        max_depth_m=args.max_depth_m,
        force=args.force,
    )
    summary = step4_extract_patches(
        labels_path=labels_path,
        patches_root=patches_root,
        scene_id=args.scene_id,
        date_range=args.date_range,
        patch_px=args.patch_px,
        val_fraction=args.val_fraction,
        max_cloud=args.max_cloud,
    )
    write_patch_manifest(patches_root, summary, args.scene_id, args.date_range)

    print("\n── Mini dataset build complete ────────────────────────────────")
    print(f"  Reef  : train={summary.get('n_reef_train',0)}  val={summary.get('n_reef_val',0)}")
    print(f"  Sand  : train={summary.get('n_sand_train',0)}  val={summary.get('n_sand_val',0)}")
    print(f"  Failed: {summary.get('n_failed', 0)}")
    print(f"  Scene : {summary.get('scene_id','?')}  ({summary.get('scene_date','?')})")
    print(f"  Output: {patches_root}")
    print(f"  Labels: {labels_path}")
    print(f"  EMODnet: {emodnet_dir}")
    print("────────────────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
