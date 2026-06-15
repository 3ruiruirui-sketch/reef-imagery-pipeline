"""
ensemble_calibrator.py — Weighted ensemble of Stumpf m0/m1 from multiple sources.

Replaces the sequential IH → EMODnet → ICESat-2 fallback chain with a
weighted combination that uses all available sources simultaneously.

Weight logic:
  • IH (nautical chart isobaths) — highest trust where samples are dense
  • EMODnet DTM                  — medium trust; wide coverage, coarser depth
  • ICESat-2 ATL03               — used when merged cache (1–34 m) is available;
                                   down-weighted when only deep-only cache exists

Usage:
    from src.ensemble_calibrator import ensemble_calibrate

    result = ensemble_calibrate(
        ih_result      = {"m0": -14.2, "m1": 22.1, "n_samples": 380, "calibrated": True},
        emodnet_result = {"m0": -12.8, "m1": 19.4, "rmse": 2.1, "calibration_samples": 5000},
        icesat2_result = {"m0": 191.0, "m1": -168.0, "rmse": 4.3, "calibration_samples": 594,
                          "status": "success"},
    )
    m0, m1 = result["m0"], result["m1"]
"""

from __future__ import annotations

import logging

import numpy as np

from src.constants import STUMPF_M0_DEFAULT, STUMPF_M1_DEFAULT

log = logging.getLogger(__name__)

# Base weights (normalised internally)
_W_IH       = 0.60
_W_EMODNET  = 0.30
_W_ICESAT2  = 0.10

# RMSE penalty: weight is multiplied by exp(-rmse / scale)
_RMSE_SCALE = 2.0   # metres — RMSE of 2 m halves the weight


def _rmse_weight(rmse: float | None) -> float:
    """Scale weight down for high-RMSE sources."""
    if rmse is None or rmse <= 0:
        return 1.0
    return float(np.exp(-rmse / _RMSE_SCALE))


def _ih_confidence(ih_result: dict) -> float:
    """
    IH confidence is driven by sample count.
    Saturates at 500 samples (typical dense isobath coverage).
    """
    n = ih_result.get("n_samples", 0) or ih_result.get("calibration_samples", 0)
    return float(np.clip(n / 500.0, 0.0, 1.0))


def ensemble_calibrate(
    ih_result: dict | None = None,
    emodnet_result: dict | None = None,
    icesat2_result: dict | None = None,
) -> dict:
    """
    Compute a weighted-average Stumpf m0/m1 from all available calibration
    sources.  Each source is weighted by its base weight × RMSE penalty ×
    sample-count confidence.

    Parameters
    ----------
    ih_result : dict from bathy_calibrator.calibrate_stumpf_from_isobaths
        Must have ``calibrated=True`` to be included.
    emodnet_result : dict from stumpf_emodnet_calibration.calibrate_stumpf_vs_emodnet
        Must have ``m0`` and ``m1`` keys.
    icesat2_result : dict from icesat2_calibrator.calibrate_from_icesat2
        Must have ``status="success"`` to be included.

    Returns
    -------
    dict with keys:
        m0, m1          – ensemble Stumpf coefficients
        weights         – per-source effective weights (normalised)
        sources_used    – list of source names included
        status          – "ensemble" | "single_source" | "default"
    """
    contributions: list[dict] = []

    # ── IH isobath ────────────────────────────────────────────────────────────
    if ih_result is not None and ih_result.get("calibrated"):
        rmse = ih_result.get("rmse_m")
        w = _W_IH * _rmse_weight(rmse) * _ih_confidence(ih_result)
        if w > 0:
            contributions.append({
                "name": "IH",
                "m0": float(ih_result["m0"]),
                "m1": float(ih_result["m1"]),
                "weight": w,
            })
            log.debug("IH contribution: m0=%.3f m1=%.3f w=%.3f", ih_result["m0"], ih_result["m1"], w)

    # ── EMODnet ───────────────────────────────────────────────────────────────
    if emodnet_result is not None and "m0" in emodnet_result and "m1" in emodnet_result:
        rmse = emodnet_result.get("rmse")
        w = _W_EMODNET * _rmse_weight(rmse)
        if w > 0:
            contributions.append({
                "name": "EMODnet",
                "m0": float(emodnet_result["m0"]),
                "m1": float(emodnet_result["m1"]),
                "weight": w,
            })
            log.debug("EMODnet contribution: m0=%.3f m1=%.3f w=%.3f", emodnet_result["m0"], emodnet_result["m1"], w)

    # ── ICESat-2 ATL03 ────────────────────────────────────────────────────────
    if icesat2_result is not None and icesat2_result.get("status") == "success":
        rmse = icesat2_result.get("rmse")
        n_samples = icesat2_result.get("calibration_samples", 0)
        # Discount heavily if RMSE > 3 m (likely deep-only biased)
        quality = 1.0 if (rmse is not None and rmse < 3.0) else 0.3
        w = _W_ICESAT2 * _rmse_weight(rmse) * quality * float(np.clip(n_samples / 200.0, 0.0, 1.0))
        if w > 0:
            contributions.append({
                "name": "ICESat-2",
                "m0": float(icesat2_result["m0"]),
                "m1": float(icesat2_result["m1"]),
                "weight": w,
            })
            log.debug("ICESat-2 contribution: m0=%.3f m1=%.3f w=%.3f", icesat2_result["m0"], icesat2_result["m1"], w)

    # ── Combine ───────────────────────────────────────────────────────────────
    if not contributions:
        log.warning("No calibration sources available — using defaults")
        return {
            "m0": STUMPF_M0_DEFAULT, "m1": STUMPF_M1_DEFAULT,
            "weights": {}, "sources_used": [], "status": "default",
        }

    total_w = sum(c["weight"] for c in contributions)
    norm_weights = {c["name"]: round(c["weight"] / total_w, 4) for c in contributions}

    m0 = sum(c["m0"] * c["weight"] for c in contributions) / total_w
    m1 = sum(c["m1"] * c["weight"] for c in contributions) / total_w

    status = "ensemble" if len(contributions) > 1 else "single_source"
    log.info(
        "Ensemble calibration [%s]: m0=%.3f  m1=%.3f  sources=%s  weights=%s",
        status, m0, m1,
        [c["name"] for c in contributions],
        norm_weights,
    )

    return {
        "m0": float(m0),
        "m1": float(m1),
        "weights": norm_weights,
        "sources_used": [c["name"] for c in contributions],
        "status": status,
    }
