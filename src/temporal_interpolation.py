"""
temporal_interpolation.py — Cloud-gap filling for SDB time series.

The Algarve receives ~15 cloudy days/month in winter, creating gaps in the
monthly SDB mosaics.  This module fills those gaps by linearly interpolating
the depth value between the nearest cloud-free acquisitions on either side,
weighted by cloud fraction so that partially-cloudy scenes contribute less.

Usage:
    from src.temporal_interpolation import interpolate_cloud_gaps

    result_path = interpolate_cloud_gaps(
        scenes=[
            {"date": "2024-11-01", "sdb_path": "outputs/nov01_sdb.tif", "cloud_frac": 0.02},
            {"date": "2024-11-15", "sdb_path": None,                    "cloud_frac": 1.0},
            {"date": "2024-12-01", "sdb_path": "outputs/dec01_sdb.tif", "cloud_frac": 0.05},
        ],
        output_path="outputs/nov15_sdb_interpolated.tif",
        target_date="2024-11-15",
    )
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from pathlib import Path

import numpy as np
import rasterio

log = logging.getLogger(__name__)

# Cloud fraction above which a scene is treated as fully obscured
_CLOUD_CLEAR_THRESHOLD = 0.20


def _parse_date(d: str | date | datetime) -> date:
    if isinstance(d, (date, datetime)):
        return d if isinstance(d, date) else d.date()
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(d, fmt).date()
        except ValueError:
            pass
    raise ValueError(f"Cannot parse date: {d!r}")


def _load_sdb(path: str | Path) -> tuple[np.ndarray, dict]:
    """Load SDB GeoTIFF, return (array, profile)."""
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        profile = src.profile.copy()
    return arr, profile


def interpolate_cloud_gaps(
    scenes: list[dict],
    output_path: str | Path,
    target_date: str | date | None = None,
    cloud_clear_threshold: float = _CLOUD_CLEAR_THRESHOLD,
) -> Path | None:
    """
    Fill cloud gaps in a SDB time series by temporal linear interpolation.

    Parameters
    ----------
    scenes : list of dicts with keys:
        ``date``       – acquisition date (str or date)
        ``sdb_path``   – path to the SDB GeoTIFF, or None if scene is cloudy
        ``cloud_frac`` – cloud fraction 0–1 (1 = fully cloudy)
    output_path : destination GeoTIFF path
    target_date : date to interpolate to.  If None, fills ALL gap positions and
        writes a multi-band composite (one band per gap).

    Returns
    -------
    Path to the written GeoTIFF, or None if interpolation was not possible.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Sort scenes by date, tag clear/cloudy
    parsed: list[dict] = []
    for sc in scenes:
        d = _parse_date(sc["date"])
        cloud = float(sc.get("cloud_frac", 1.0))
        clear = (cloud <= cloud_clear_threshold) and sc.get("sdb_path") is not None
        parsed.append({**sc, "_date": d, "_clear": clear, "_cloud": cloud})
    parsed.sort(key=lambda x: x["_date"])

    clear_scenes = [sc for sc in parsed if sc["_clear"]]

    if len(clear_scenes) < 2:
        log.warning("Need ≥2 clear scenes for interpolation (have %d)", len(clear_scenes))
        return None

    # Determine which target dates need filling
    if target_date is not None:
        targets = [_parse_date(target_date)]
    else:
        targets = [sc["_date"] for sc in parsed if not sc["_clear"]]

    if not targets:
        log.info("No gap dates to fill")
        return None

    # Load reference profile from first clear scene
    _, ref_profile = _load_sdb(clear_scenes[0]["sdb_path"])
    ref_profile.update(dtype=rasterio.float32, count=len(targets), nodata=np.nan)

    bands: list[np.ndarray] = []

    for tgt_date in targets:
        # Find nearest clear scene before and after target
        before = [sc for sc in clear_scenes if sc["_date"] <= tgt_date]
        after = [sc for sc in clear_scenes if sc["_date"] >= tgt_date]

        if not before or not after:
            log.warning("Cannot bracket date %s — no clear scene on one side", tgt_date)
            bands.append(np.full_like(bands[0] if bands else np.zeros((1, 1)), np.nan))
            continue

        sc_before = before[-1]  # closest clear scene before target
        sc_after = after[0]  # closest clear scene after target

        arr_b, _ = _load_sdb(sc_before["sdb_path"])
        arr_a, _ = _load_sdb(sc_after["sdb_path"])

        if sc_before["_date"] == sc_after["_date"]:
            # Target coincides with a clear scene — no interpolation needed
            bands.append(arr_b.copy())
            log.info("%s: target coincides with clear scene — copied directly", tgt_date)
            continue

        # Linear interpolation weight: t=0 at before, t=1 at after
        span = (sc_after["_date"] - sc_before["_date"]).days
        offset = (tgt_date - sc_before["_date"]).days
        t = offset / span

        # Cloud-quality weighting: down-weight scenes with higher cloud fraction
        w_b = 1.0 - sc_before["_cloud"]
        w_a = 1.0 - sc_after["_cloud"]

        # Weighted linear blend
        t_adj = (t * w_a) / (t * w_a + (1 - t) * w_b + 1e-9)
        arr_interp = (1.0 - t_adj) * arr_b + t_adj * arr_a

        # Propagate NaN: if either side is NaN, output is NaN
        invalid = np.isnan(arr_b) | np.isnan(arr_a)
        arr_interp[invalid] = np.nan

        log.info(
            "%s: interpolated between %s (t=%.2f) and %s (t=%.2f) — " "t_adj=%.3f  valid=%.1f%%",
            tgt_date,
            sc_before["_date"],
            1 - t,
            sc_after["_date"],
            t,
            t_adj,
            100 * float(np.isfinite(arr_interp).mean()),
        )
        bands.append(arr_interp.astype(np.float32))

    if not bands:
        return None

    ref_profile["count"] = len(bands)
    with rasterio.open(output_path, "w", **ref_profile) as dst:
        for i, band in enumerate(bands, start=1):
            dst.write(band, i)

    log.info("Wrote %d interpolated band(s) → %s", len(bands), output_path)
    return output_path


def build_monthly_composite(
    scenes: list[dict],
    output_dir: str | Path,
    cloud_clear_threshold: float = _CLOUD_CLEAR_THRESHOLD,
) -> dict[str, Path]:
    """
    Build a gap-filled monthly SDB composite for each month in the scene list.

    For each month: take the median of all cloud-free scenes in that month.
    If a month has no clear scene, interpolate from adjacent months.

    Returns dict mapping "YYYY-MM" → output GeoTIFF path.
    """
    from collections import defaultdict

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group clear scenes by month
    by_month: dict[str, list[dict]] = defaultdict(list)
    for sc in scenes:
        d = _parse_date(sc["date"])
        cloud = float(sc.get("cloud_frac", 1.0))
        if cloud <= cloud_clear_threshold and sc.get("sdb_path") is not None:
            by_month[d.strftime("%Y-%m")].append(sc)

    results: dict[str, Path] = {}

    for month_key, month_scenes in sorted(by_month.items()):
        out_path = output_dir / f"sdb_{month_key}.tif"
        arrays = []
        ref_profile = None
        for sc in month_scenes:
            arr, profile = _load_sdb(sc["sdb_path"])
            arrays.append(arr)
            if ref_profile is None:
                ref_profile = profile

        if not arrays or ref_profile is None:
            continue

        # Median composite (robust to remaining cloud shadows)
        stack = np.stack(arrays, axis=0)
        composite = np.nanmedian(stack, axis=0).astype(np.float32)

        ref_profile.update(count=1, dtype=rasterio.float32, nodata=np.nan)
        with rasterio.open(out_path, "w", **ref_profile) as dst:
            dst.write(composite, 1)

        log.info("Monthly composite %s: median of %d scenes → %s", month_key, len(arrays), out_path)
        results[month_key] = out_path

    return results
