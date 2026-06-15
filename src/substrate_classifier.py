"""
substrate_classifier.py — Benthic substrate classification for Algarve reefs.

Produces a 3-class map from Sentinel-2 BOA reflectance:

    0 = sand / bright substrate   (high B03, B03/B04 > 1.4)
    1 = seagrass / macroalgae     (elevated B08 NDVI > 0.05, if available)
    2 = rock-reef / dark substrate (dark, not seagrass)
   -1 = masked (land, cloud, optically deep)

This classification serves two purposes:
  1. Masks sandy pixels from IH/EMODnet calibration training (sand inflates
     apparent reflectance and biases Stumpf m0/m1 upward).
  2. Gives divers a benthic substrate layer alongside BVI scores — rock-reef
     pixels are more likely to have structural complexity and marine life.

Algorithm
---------
The spectral rules are adapted from Lyzenga (1981) and Kutser et al. (2020)
for Sentinel-2 in Algarve oligotrophic clear-water conditions (Kd490 ~0.06–0.2):

  • Depth mask: only classify pixels where SDB depth is 1–25 m (optically valid)
  • Sand: B03 (green) > 0.06 AND B03/B04 (green/red) > 1.4
           High green reflectance = bright sandy bottom
  • Seagrass (if B08 available): NDVI = (B08-B04)/(B08+B04) > 0.05
  • Rock-reef: everything else in the valid depth window

Usage:
    from src.substrate_classifier import classify_substrate, write_substrate_tiff

    classes = classify_substrate(
        b02=b02_arr, b03=b03_arr, b04=b04_arr,
        b08=b08_arr,         # optional — enables seagrass detection
        sdb_depth=depth_arr, # optional — restricts to valid optical depth
    )

    write_substrate_tiff(classes, ref_profile, "outputs/substrate.tif")
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# Class labels
SAND     = 0
SEAGRASS = 1
REEF     = 2
MASKED   = -1

# Class names for legend / dashboard
CLASS_NAMES = {SAND: "Sand", SEAGRASS: "Seagrass/Macroalgae", REEF: "Rock-reef", MASKED: "Masked"}

# Spectral thresholds (BOA reflectance, unitless 0–1)
_B03_SAND_MIN   = 0.06   # minimum B03 to be considered bright sand
_B03B04_SAND    = 1.4    # B03/B04 ratio threshold for sand
_NDVI_SEAGRASS  = 0.05   # minimum NDVI for seagrass / macroalgae

# Depth window for valid optical classification (metres, positive = below surface)
_DEPTH_MIN_M = 1.0
_DEPTH_MAX_M = 25.0


def classify_substrate(
    b02: np.ndarray,
    b03: np.ndarray,
    b04: np.ndarray,
    b08: np.ndarray | None = None,
    sdb_depth: np.ndarray | None = None,
    depth_min_m: float = _DEPTH_MIN_M,
    depth_max_m: float = _DEPTH_MAX_M,
) -> np.ndarray:
    """
    Classify each pixel into sand / seagrass / rock-reef.

    Parameters
    ----------
    b02, b03, b04 : BOA reflectance arrays (0–1 scale)
    b08           : NIR band (BOA, 0–1).  If provided, enables seagrass class.
    sdb_depth     : Depth array in metres (positive = below surface).
                    Used to mask pixels outside the 1–25 m optical window.
                    If None, no depth masking is applied.

    Returns
    -------
    int8 array: SAND=0, SEAGRASS=1, REEF=2, MASKED=-1
    """
    eps = 1e-6
    H, W = b03.shape
    classes = np.full((H, W), MASKED, dtype=np.int8)

    # Depth validity mask
    if sdb_depth is not None:
        in_depth = np.isfinite(sdb_depth) & (sdb_depth >= depth_min_m) & (sdb_depth <= depth_max_m)
    else:
        in_depth = np.ones((H, W), dtype=bool)

    # Basic reflectance validity (non-zero BOA in all bands)
    valid = in_depth & (b02 > eps) & (b03 > eps) & (b04 > eps)

    b03v = np.where(valid, b03, np.nan)
    b04v = np.where(valid, b04, np.nan)

    # Sand rule: bright green AND green/red > threshold
    sand_mask = (
        valid
        & (b03v > _B03_SAND_MIN)
        & (b03v / np.where(b04v > eps, b04v, eps) > _B03B04_SAND)
    )

    # Seagrass rule (requires B08)
    seagrass_mask = np.zeros((H, W), dtype=bool)
    if b08 is not None:
        b08v = np.where(valid, b08, np.nan)
        ndvi = (b08v - b04v) / (b08v + b04v + eps)
        seagrass_mask = valid & ~sand_mask & (ndvi > _NDVI_SEAGRASS)

    # Rock-reef: valid, not sand, not seagrass
    reef_mask = valid & ~sand_mask & ~seagrass_mask

    classes[reef_mask]     = REEF
    classes[seagrass_mask] = SEAGRASS
    classes[sand_mask]     = SAND

    n_sand     = int(sand_mask.sum())
    n_sea      = int(seagrass_mask.sum())
    n_reef     = int(reef_mask.sum())
    n_masked   = int((classes == MASKED).sum())
    n_total    = H * W
    log.info(
        "Substrate classification: sand=%d (%.1f%%)  seagrass=%d (%.1f%%)  "
        "reef=%d (%.1f%%)  masked=%d (%.1f%%)",
        n_sand, 100 * n_sand / n_total,
        n_sea,  100 * n_sea  / n_total,
        n_reef, 100 * n_reef / n_total,
        n_masked, 100 * n_masked / n_total,
    )

    return classes


def get_sand_mask(
    b02: np.ndarray,
    b03: np.ndarray,
    b04: np.ndarray,
    sdb_depth: np.ndarray | None = None,
) -> np.ndarray:
    """
    Boolean mask: True where a pixel is classified as sandy substrate.
    Convenience wrapper for IH calibration training — excludes sand pixels
    that would bias Stumpf m0/m1 toward high (bright) reflectance.
    """
    classes = classify_substrate(b02, b03, b04, sdb_depth=sdb_depth)
    return classes == SAND


def write_substrate_tiff(
    classes: np.ndarray,
    ref_profile: dict,
    output_path: str | Path,
) -> Path:
    """
    Write the substrate classification array to a single-band GeoTIFF.
    Uses int8 dtype; nodata=-1 (masked pixels).
    """
    try:
        import rasterio
    except ImportError as exc:
        raise ImportError("rasterio required for write_substrate_tiff") from exc

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    profile = ref_profile.copy()
    profile.update(dtype="int8", count=1, nodata=MASKED)

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(classes.astype("int8"), 1)
        dst.update_tags(
            classes="0=Sand,1=Seagrass,2=Rock-reef,-1=Masked",
            b03_sand_threshold=str(_B03_SAND_MIN),
            b03b04_sand_ratio=str(_B03B04_SAND),
            ndvi_seagrass_threshold=str(_NDVI_SEAGRASS),
        )

    log.info("Substrate classification written → %s", output_path)
    return output_path
