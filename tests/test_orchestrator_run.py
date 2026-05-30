#!/usr/bin/env python3
"""
tests/test_orchestrator_run.py
================================
Unit tests for the physics and analysis helpers in src/orchestrator_run.py.

Functions tested:
    snell_optical_path()   — Snell's law + Beer–Lambert path geometry
    sunglint_correction()  — Hedley-style linear glint removal
    analyse_band()         — Full per-image feature extraction (mocked rasterio)

Run:
    python -m pytest tests/test_orchestrator_run.py -v
"""

import math
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

# Ensure project root is on the path
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.orchestrator_run import snell_optical_path, sunglint_correction, analyse_band
from src.constants import N_WATER


# ─────────────────────────────────────────────────────────────────────────────
# snell_optical_path
# ─────────────────────────────────────────────────────────────────────────────

class TestSnellOpticalPath:
    """Verify Snell's law refraction and Beer–Lambert path geometry."""

    def test_vertical_incidence(self):
        """SZA=0° (sun directly overhead) → optical path equals depth."""
        path_m, sza_w = snell_optical_path(0.0, 16.0)
        # cos(0) = 1 → path = depth / 1 = depth
        assert abs(path_m - 16.0) < 1e-6
        assert abs(sza_w) < 1e-6

    def test_typical_september_sza(self):
        """SZA≈40° (typical Algarve September) → path slightly longer than depth."""
        path_m, sza_w = snell_optical_path(40.498, 16.0)
        # Refracted angle is always < air angle due to n>1
        assert sza_w < 40.498
        # Path must be longer than depth (cos θ_water < 1)
        assert path_m > 16.0
        # Physical bound: path can't exceed depth / cos(SZA_water)
        expected_path = 16.0 / math.cos(math.radians(sza_w))
        assert abs(path_m - expected_path) < 1e-6

    def test_known_snell_refraction(self):
        """Cross-check refracted angle against Snell's law formula directly."""
        sza_air = 30.0
        _, sza_w = snell_optical_path(sza_air, 10.0)
        expected_sza_w = math.degrees(
            math.asin(math.sin(math.radians(sza_air)) / N_WATER)
        )
        assert abs(sza_w - expected_sza_w) < 1e-6

    def test_path_scales_with_depth(self):
        """Deeper target → proportionally longer optical path at same SZA."""
        path_10, _ = snell_optical_path(40.0, 10.0)
        path_20, _ = snell_optical_path(40.0, 20.0)
        # Ratio must equal depth ratio exactly (linear scaling)
        assert abs(path_20 / path_10 - 2.0) < 1e-9

    def test_returns_tuple_of_floats(self):
        path_m, sza_w = snell_optical_path(35.0, 16.0)
        assert isinstance(path_m, float)
        assert isinstance(sza_w, float)


# ─────────────────────────────────────────────────────────────────────────────
# sunglint_correction
# ─────────────────────────────────────────────────────────────────────────────

class TestSunglintCorrection:
    """Verify Hedley-style linear sunglint removal."""

    def _make_arrays(self, rows=40, cols=40, seed=42):
        rng = np.random.default_rng(seed)
        b02 = rng.uniform(0.02, 0.20, (rows, cols)).astype(np.float64)
        b03 = rng.uniform(0.01, 0.15, (rows, cols)).astype(np.float64)
        return b02, b03

    def test_no_negative_output(self):
        """Corrected array must never contain negative values."""
        b02, b03 = self._make_arrays()
        corrected = sunglint_correction(b02, b03)
        assert np.all(corrected >= 0), "Negative values found after glint correction"

    def test_output_shape_preserved(self):
        b02, b03 = self._make_arrays(30, 50)
        corrected = sunglint_correction(b02, b03)
        assert corrected.shape == b02.shape

    def test_slope_clamped_to_valid_range(self):
        """Even with extreme covariance the slope stays in [0, 2]."""
        # Make B02 = 100 * B03 to produce huge raw slope
        b03 = np.random.rand(40, 40) * 0.1
        b02 = b03 * 100.0
        corrected = sunglint_correction(b02, b03)
        # If slope were unclamped, almost all pixels would go deeply negative
        assert np.all(corrected >= 0)

    def test_all_zero_b03_returns_original(self):
        """If B03 is all zero there are no valid pixels → return B02 unchanged."""
        b02 = np.ones((10, 10)) * 0.1
        b03 = np.zeros((10, 10))
        corrected = sunglint_correction(b02, b03)
        np.testing.assert_array_equal(corrected, b02)

    def test_fewer_than_10_valid_pixels_returns_original(self):
        """With <10 valid pixels the function should pass through unchanged."""
        b02 = np.zeros((5, 5))
        b03 = np.zeros((5, 5))
        # Set only 5 pixels non-zero
        b02[:5, 0] = 0.05
        b03[:5, 0] = 0.03
        corrected = sunglint_correction(b02, b03)
        np.testing.assert_array_equal(corrected, b02)

    def test_correction_reduces_mean_in_correlated_scene(self):
        """When B02 and B03 are positively correlated, correction lowers mean."""
        rng = np.random.default_rng(0)
        base = rng.uniform(0.05, 0.15, (40, 40))
        b03 = base + rng.normal(0, 0.005, (40, 40))
        b02 = base * 1.5 + rng.normal(0, 0.005, (40, 40))  # correlated
        b02 = np.clip(b02, 0, None)
        b03 = np.clip(b03, 1e-6, None)
        corrected = sunglint_correction(b02, b03)
        assert np.mean(corrected) < np.mean(b02)


# ─────────────────────────────────────────────────────────────────────────────
# analyse_band
# ─────────────────────────────────────────────────────────────────────────────

def _make_mock_rasterio_open(arr_b02, arr_b03, crs_epsg=32629):
    """Return a context-manager-compatible mock for rasterio.open."""
    from rasterio.transform import from_origin
    from rasterio.crs import CRS

    transform = from_origin(
        west=-8.25, north=37.10,
        xsize=10 / 111320, ysize=10 / 111320,   # ~10m pixels in degrees
    )
    crs = CRS.from_epsg(crs_epsg)

    def _make_src(arr):
        src = MagicMock()
        src.__enter__ = lambda s: s
        src.__exit__ = MagicMock(return_value=False)
        src.crs = crs
        src.transform = transform
        src.height, src.width = arr.shape
        src.index = MagicMock(return_value=(arr.shape[0] // 2, arr.shape[1] // 2))
        src.read = MagicMock(return_value=arr)
        return src

    call_count = [0]
    sources = [_make_src(arr_b02), _make_src(arr_b03)]

    def _open(path, *a, **kw):
        idx = call_count[0] % 2
        call_count[0] += 1
        return sources[idx]

    return _open


class TestAnalyseBand:
    """Verify analyse_band() output structure and physical constraints."""

    EXPECTED_KEYS = {
        "date", "sza_air_deg", "sza_water_deg", "optical_path_m",
        "kd490_seasonal", "kd490_estimated", "kd_high_uncertainty",
        "water_transmittance_twoway", "b02_signal_mean", "b02_noise_std",
        "b02_cv", "SNR_mean_16m", "contrast_benthic_mean",
        "percent_pixels_useful", "percent_area_high_confidence",
        "visibility_score", "cleanliness", "cloud_cover",
    }

    def _run(self, b02_val=0.08, b03_val=0.06, cloud=1.0, month=9, sza=40.5):
        size = (100, 100)
        arr_b02 = (np.ones(size, dtype=np.float32) * b02_val * 10000).astype(np.float32)
        arr_b03 = (np.ones(size, dtype=np.float32) * b03_val * 10000).astype(np.float32)
        meta = {"date": "2025-09-25", "sza": sza, "cloud": cloud, "month": month}

        mock_open = _make_mock_rasterio_open(arr_b02, arr_b03)
        with patch("src.orchestrator_run.rasterio.open", side_effect=mock_open), \
             patch("src.orchestrator_run.Transformer") as mock_tf:
            mock_tf.from_crs.return_value.transform.return_value = (
                569000.0, 4102000.0
            )
            result = analyse_band(Path("fake_b02.tif"), Path("fake_b03.tif"),
                                  meta, depth=16.0)
        return result

    def test_returns_all_required_keys(self):
        result = self._run()
        for key in self.EXPECTED_KEYS:
            assert key in result, f"Missing key: {key}"

    def test_transmittance_in_unit_interval(self):
        result = self._run()
        t = result["water_transmittance_twoway"]
        assert 0.0 < t <= 1.0, f"Transmittance out of range: {t}"

    def test_snr_non_negative(self):
        result = self._run()
        assert result["SNR_mean_16m"] >= 0

    def test_contrast_in_unit_interval(self):
        result = self._run()
        c = result["contrast_benthic_mean"]
        assert 0.0 <= c <= 1.0, f"Contrast out of range: {c}"

    def test_visibility_score_in_unit_interval(self):
        result = self._run()
        v = result["visibility_score"]
        assert 0.0 <= v <= 1.0, f"Visibility score out of range: {v}"

    def test_cloud_cover_propagated(self):
        result = self._run(cloud=3.5)
        assert result["cloud_cover"] == 3.5

    def test_date_propagated(self):
        result = self._run()
        assert result["date"] == "2025-09-25"

    def test_high_cloud_reduces_usable_pixels(self):
        low_cloud = self._run(cloud=1.0)
        high_cloud = self._run(cloud=50.0)
        assert high_cloud["percent_pixels_useful"] < low_cloud["percent_pixels_useful"]

    def test_september_kd_used(self):
        """September (month=9) should use Kd≈0.045 per KD490_TABLE."""
        result = self._run(month=9)
        assert abs(result["kd490_seasonal"] - 0.045) < 1e-6

    def test_april_kd_used(self):
        """April (month=4) should use Kd≈0.065 per KD490_TABLE."""
        result = self._run(month=4)
        assert abs(result["kd490_seasonal"] - 0.065) < 1e-6

    def test_cleanliness_default_present(self):
        """cleanliness must always be present (default 5000 sentinel)."""
        result = self._run()
        assert "cleanliness" in result
        assert isinstance(result["cleanliness"], (int, float))

    def test_kd_uncertainty_flag_type(self):
        result = self._run()
        assert isinstance(result["kd_high_uncertainty"], bool)
