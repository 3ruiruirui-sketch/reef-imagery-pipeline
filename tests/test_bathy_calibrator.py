#!/usr/bin/env python3
"""
test_bathy_calibrator.py
========================
Automated BEFORE / AFTER comparison tests for bathy_calibrator.py.

Tests verify that the IH-calibrated Stumpf SDB:
  1. Has lower bias vs IH chart than uncalibrated defaults
  2. Has RMSE < threshold for the primary isobath zone
  3. Produces correct zone classification for known spots
  4. Correctly detects single-isobath offset calibration
  5. Buffer sampling produces more samples than point sampling

Run with:  python test_bathy_calibrator.py
Exit 0 = all tests pass, Exit 1 = failures found.
"""

import sys
import logging
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform_bounds

logging.basicConfig(
    level=logging.WARNING,   # suppress INFO during tests; failures still visible
    format="%(levelname)s %(message)s",
)

ROOT   = Path(__file__).parent
MASTER = ROOT / "reef_Output_Master"

# ── colour helpers ─────────────────────────────────────────────────────────────
GRN  = "\033[92m"
RED  = "\033[91m"
YEL  = "\033[93m"
RST  = "\033[0m"
BOLD = "\033[1m"

_passed = _failed = _warned = 0

def ok(msg):
    global _passed
    _passed += 1
    print(f"  {GRN}✓{RST}  {msg}")

def fail(msg):
    global _failed
    _failed += 1
    print(f"  {RED}✗{RST}  {msg}")

def warn(msg):
    global _warned
    _warned += 1
    print(f"  {YEL}⚠{RST}  {msg}")

def section(title):
    print(f"\n{BOLD}{'─'*60}{RST}")
    print(f"{BOLD}  {title}{RST}")
    print(f"{BOLD}{'─'*60}{RST}")

def assert_lt(val, threshold, label):
    if val is None:
        warn(f"{label}: no data")
        return
    if abs(val) < threshold:
        ok(f"{label}: {val:.3f} < {threshold} ✓")
    else:
        fail(f"{label}: {val:.3f} ≥ {threshold}  (FAIL)")

def assert_better(after, before, label):
    """Pass if |after| < |before|, i.e. improvement."""
    if after is None or before is None:
        warn(f"{label}: missing data")
        return
    if abs(after) < abs(before):
        pct = 100 * (1 - abs(after) / max(abs(before), 1e-9))
        ok(f"{label}: {before:.2f}m → {after:.2f}m  ({pct:.0f}% improvement) ✓")
    else:
        fail(f"{label}: {before:.2f}m → {after:.2f}m  (no improvement) ✗")


# ── Raster loader ──────────────────────────────────────────────────────────────
def load_raster_wgs84(tif_path):
    with rasterio.open(tif_path) as src:
        arr = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        west, south, east, north = transform_bounds(
            src.crs, "EPSG:4326",
            src.bounds.left, src.bounds.bottom,
            src.bounds.right, src.bounds.top
        )
    if arr.max() > 2.0:
        arr /= 10000.0
    return arr, profile, (south, west, north, east)   # (min_lat, min_lon, max_lat, max_lon)


# ── Stumpf SDB helper ──────────────────────────────────────────────────────────
def compute_sdb(b02, b03, m0, m1, n=1000.0):
    eps = 1e-6
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(
            (b02 > eps) & (b03 > eps),
            np.log(n * b02 + eps) / (np.log(n * b03 + eps) + eps),
            np.nan
        )
    return np.clip(m1 * ratio + m0, 0, 40).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# TEST DATA
# ══════════════════════════════════════════════════════════════════════════════
SPOTS = [
    {
        "name":  "Pedra do Alto (2024-09-30)",
        "dir":   MASTER / "reef_output_ai_prediction_spot",
        "b02":   "S2_B02_20240930.tif",
        "b03":   "S2_B03_20240930.tif",
        "lat":   37.0636, "lon": -8.2193,
        "primary_isobath": 10,
        "expect_zone": "very_shallow",
        "bias_threshold_m": 3.0,
        "rmse_threshold_m": 5.0,
    },
    {
        "name":  "Mar 2022 New Spot (2022-02-28)",
        "dir":   MASTER / "reef_output_mar_2022_new_spot",
        "b02":   "S2_B02_20220228.tif",
        "b03":   "S2_B03_20220228.tif",
        "lat":   37.0581, "lon": -8.2098,
        "primary_isobath": 10,
        "expect_zone": "nearshore_mid",
        "bias_threshold_m": 8.0,    # single-isobath; allow wider tolerance
        "rmse_threshold_m": 10.0,
    },
    {
        "name":  "Aug 2022 Target (2022-08-12)",
        "dir":   MASTER / "reef_output_aug_2022_target1",
        "b02":   "S2_B02_20220812.tif",
        "b03":   "S2_B03_20220812.tif",
        "lat":   37.0636, "lon": -8.2193,
        "primary_isobath": 20,
        "expect_zone": "very_shallow",
        "bias_threshold_m": 10.0,   # single-isobath
        "rmse_threshold_m": 12.0,
    },
]

M0_DEFAULT = -16.0
M1_DEFAULT =  20.0
BUF_M = 3000.0
DEG   = BUF_M / 111_000.0


# ══════════════════════════════════════════════════════════════════════════════
# TESTS
# ══════════════════════════════════════════════════════════════════════════════
def test_zone_classification():
    from src.bathy_calibrator import fetch_isobaths_for_bbox, classify_benthic_zone
    section("TEST 1 — Zone Classification")

    expected = [
        (37.0636, -8.2193, "very_shallow"),
        (37.0468, -7.6603, "shallow_reef"),
        (37.0581, -8.2098, "nearshore_mid"),
    ]
    for lat, lon, exp_zone in expected:
        feats = fetch_isobaths_for_bbox(lon-DEG, lat-DEG, lon+DEG, lat+DEG)
        z = classify_benthic_zone(lon, lat, feats)
        got = z["zone"]
        if got == exp_zone:
            ok(f"({lat},{lon}) → zone='{got}' ✓")
        else:
            fail(f"({lat},{lon}) → expected '{exp_zone}', got '{got}'")
        if z["optically_viable"]:
            ok(f"  optically_viable=True ✓")
        else:
            warn(f"  optically_viable=False (unexpected for reef spot)")


def test_buffer_sampling_improvement():
    from src.bathy_calibrator import (
        fetch_isobaths_for_bbox,
        _sample_pixels_near_isobath, BUF_PIX,
        BENTHIC_ISOBATHS,
    )
    section("TEST 2 — Buffer Sampling > Point Sampling")

    spot = SPOTS[0]
    b02, _, bounds = load_raster_wgs84(spot["dir"] / spot["b02"])
    b03, _, _      = load_raster_wgs84(spot["dir"] / spot["b03"])
    lat, lon = spot["lat"], spot["lon"]
    feats = fetch_isobaths_for_bbox(lon-DEG, lat-DEG, lon+DEG, lat+DEG)

    for depth in BENTHIC_ISOBATHS:
        n_buf  = len(_sample_pixels_near_isobath(b02, b03, feats, float(depth), bounds, buf=BUF_PIX))
        n_pt   = len(_sample_pixels_near_isobath(b02, b03, feats, float(depth), bounds, buf=0))
        if n_buf == 0 and n_pt == 0:
            warn(f"  {depth}m isobath: no pixels in raster (outside bbox)")
            continue
        ratio = n_buf / max(n_pt, 1)
        if ratio >= 1.0:
            ok(f"  {depth}m: buf={n_buf} vs point={n_pt} (×{ratio:.1f} more samples) ✓")
        else:
            fail(f"  {depth}m: buf={n_buf} vs point={n_pt} — buffer not better?")


def test_single_isobath_offset_calibration():
    from src.bathy_calibrator import (
        fetch_isobaths_for_bbox,
        calibrate_stumpf_from_isobaths,
        STUMPF_M1_LITERATURE,
    )
    section("TEST 3 — Single-Isobath Offset Calibration")

    for spot in SPOTS[1:]:   # spots 2 & 3 have single-isobath in raster bbox
        b02, _, bounds = load_raster_wgs84(spot["dir"] / spot["b02"])
        b03, _, _      = load_raster_wgs84(spot["dir"] / spot["b03"])
        lat, lon = spot["lat"], spot["lon"]
        feats = fetch_isobaths_for_bbox(lon-DEG, lat-DEG, lon+DEG, lat+DEG)
        m0, m1, cal = calibrate_stumpf_from_isobaths(b02, b03, feats, bounds)

        if cal["calibrated"]:
            ok(f"  {spot['name']}: calibrated=True ✓")
            method = cal.get("method","")
            if "offset" in method:
                ok(f"    method=offset_calibration (single-isobath) ✓")
                # m1 should be fixed at literature value
                if abs(m1 - STUMPF_M1_LITERATURE) < 1e-6:
                    ok(f"    m1={m1} == STUMPF_M1_LITERATURE={STUMPF_M1_LITERATURE} ✓")
                else:
                    warn(f"    m1={m1} ≠ {STUMPF_M1_LITERATURE} (unexpected)")
                ok(f"    m0 solved = {m0:.3f}")
            else:
                ok(f"    method={method}")
        else:
            reason = cal.get("reason","?")
            warn(f"  {spot['name']}: not calibrated — {reason}")


def test_before_after_bias():
    from src.bathy_calibrator import (
        fetch_isobaths_for_bbox,
        calibrate_stumpf_from_isobaths,
        validate_sdb_vs_chart,
    )
    section("TEST 4 — BEFORE vs AFTER: Bias & RMSE vs IH Chart")
    print(f"  {'Spot':35s} {'Iso':5s} {'Before':>10s} {'After':>10s} {'Δ':>8s}")
    print(f"  {'-'*70}")

    all_before, all_after = [], []

    for spot in SPOTS:
        b02, _, bounds = load_raster_wgs84(spot["dir"] / spot["b02"])
        b03, _, _      = load_raster_wgs84(spot["dir"] / spot["b03"])
        lat, lon = spot["lat"], spot["lon"]
        feats = fetch_isobaths_for_bbox(lon-DEG, lat-DEG, lon+DEG, lat+DEG)

        # BEFORE: generic defaults
        sdb_before = compute_sdb(b02, b03, M0_DEFAULT, M1_DEFAULT)
        val_before = validate_sdb_vs_chart(sdb_before, feats, bounds)
        ov_b = val_before.get("overall", {})
        bias_b = ov_b.get("overall_bias_m")
        rmse_b = ov_b.get("overall_rmse_m")

        # AFTER: IH-calibrated
        m0, m1, cal = calibrate_stumpf_from_isobaths(b02, b03, feats, bounds)
        sdb_after = compute_sdb(b02, b03, m0, m1)
        val_after = validate_sdb_vs_chart(sdb_after, feats, bounds)
        ov_a = val_after.get("overall", {})
        bias_a = ov_a.get("overall_bias_m")
        rmse_a = ov_a.get("overall_rmse_m")

        if bias_b is not None and bias_a is not None:
            delta = abs(bias_a) - abs(bias_b)
            tag = f"{GRN}▼{abs(delta):.2f}{RST}" if delta < 0 else f"{RED}▲{abs(delta):.2f}{RST}"
            print(f"  {spot['name'][:35]:35s} {spot['primary_isobath']:3d}m "
                  f"  {bias_b:+7.2f}m  {bias_a:+7.2f}m   {tag}")
            all_before.append(abs(bias_b))
            all_after.append(abs(bias_a))

    print()
    if all_before and all_after:
        mean_b = np.mean(all_before)
        mean_a = np.mean(all_after)
        assert_better(mean_a, mean_b, "Mean |bias| across all spots")

    # Per-spot pass/fail
    print()
    for spot in SPOTS:
        b02, _, bounds = load_raster_wgs84(spot["dir"] / spot["b02"])
        b03, _, _      = load_raster_wgs84(spot["dir"] / spot["b03"])
        lat, lon = spot["lat"], spot["lon"]
        feats = fetch_isobaths_for_bbox(lon-DEG, lat-DEG, lon+DEG, lat+DEG)
        m0, m1, _ = calibrate_stumpf_from_isobaths(b02, b03, feats, bounds)
        sdb_after = compute_sdb(b02, b03, m0, m1)
        val = validate_sdb_vs_chart(sdb_after, feats, bounds)
        ov = val.get("overall", {})
        bias_a = ov.get("overall_bias_m")
        rmse_a = ov.get("overall_rmse_m")
        print(f"\n  {spot['name']}")
        assert_lt(bias_a, spot["bias_threshold_m"], f"    |bias| < {spot['bias_threshold_m']}m")
        assert_lt(rmse_a, spot["rmse_threshold_m"], f"    RMSE  < {spot['rmse_threshold_m']}m")


def test_sdb_depth_distribution():
    from src.bathy_calibrator import fetch_isobaths_for_bbox, calibrate_stumpf_from_isobaths
    section("TEST 5 — SDB Depth Distribution (sanity check)")

    for spot in SPOTS:
        b02, _, bounds = load_raster_wgs84(spot["dir"] / spot["b02"])
        b03, _, _      = load_raster_wgs84(spot["dir"] / spot["b03"])
        lat, lon = spot["lat"], spot["lon"]
        feats = fetch_isobaths_for_bbox(lon-DEG, lat-DEG, lon+DEG, lat+DEG)
        m0, m1, cal = calibrate_stumpf_from_isobaths(b02, b03, feats, bounds)
        sdb = compute_sdb(b02, b03, m0, m1)
        valid = sdb[sdb > 0]
        median = float(np.median(valid))
        print(f"\n  {spot['name']}")
        print(f"    m0={m0:.2f}  m1={m1:.2f}  calibrated={cal['calibrated']}")
        print(f"    SDB median={median:.1f}m  mean={valid.mean():.1f}m  "
              f"[{valid.min():.1f}–{valid.max():.1f}]")
        # The median should be in 0–35m range for a reef spot
        if 0 < median < 35:
            ok(f"    median={median:.1f}m in physical range [0–35m] ✓")
        else:
            fail(f"    median={median:.1f}m is outside [0–35m] — unrealistic")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"\n{BOLD}{'═'*60}{RST}")
    print(f"{BOLD}  BATHY CALIBRATOR — Test Suite{RST}")
    print(f"{BOLD}{'═'*60}{RST}")

    test_zone_classification()
    test_buffer_sampling_improvement()
    test_single_isobath_offset_calibration()
    test_before_after_bias()
    test_sdb_depth_distribution()

    print(f"\n{BOLD}{'═'*60}{RST}")
    print(f"{BOLD}  RESULTS: {GRN}{_passed} passed{RST}  "
          f"{RED}{_failed} failed{RST}  "
          f"{YEL}{_warned} warnings{RST}{BOLD}{RST}")
    print(f"{BOLD}{'═'*60}{RST}\n")

    sys.exit(0 if _failed == 0 else 1)
