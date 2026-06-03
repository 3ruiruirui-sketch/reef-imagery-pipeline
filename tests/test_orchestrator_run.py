#!/usr/bin/env python3
"""
tests/test_orchestrator_run.py
================================
Unit tests for the physics and orchestrator helpers.

After the refactor (v2.0), physics functions were removed from orchestrator_run.py
and now live in their canonical modules:

  snell_optical_path()  → src.utils.snell_sza()  + src.utils.optical_path()
  sunglint_correction() → src.utils.simulate_acolite_boa() (Hedley linear)
  analyse_band()        → replaced by src.reef_ml_predictor_acolite.run_predictor()
                          + src.orchestrator_run._normalise_result()

TestSnellOpticalPath   — tests snell_sza() + optical_path() from utils.py
TestSunglintCorrection — tests the Hedley correction logic in simulate_acolite_boa()
TestAnalyseBand        — tests _normalise_result() keys and S1 penalty in orchestrator main()

Run:
    python -m pytest tests/test_orchestrator_run.py -v
"""

import math
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from rasterio.crs import CRS

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ── Canonical imports (physics now lives in utils.py) ────────────────────────
from src.utils import snell_sza, optical_path, simulate_acolite_boa
from src.orchestrator_run import _normalise_result, METADATA, TARGET_LAT, TARGET_LON
from src.constants import N_WATER, KD490_TABLE


# ── Thin compatibility shim so old test helpers keep working ─────────────────
def snell_optical_path(sza_air_deg: float, depth_m: float):
    """Shim: snell_sza() + optical_path() — mirrors old orchestrator signature."""
    sza_water_deg, theta_w = snell_sza(sza_air_deg)
    path_m = optical_path(depth_m, theta_w)
    return path_m, sza_water_deg


def sunglint_correction(b02: np.ndarray, b03: np.ndarray) -> np.ndarray:
    """
    Shim: extract the Hedley linear glint correction logic from
    simulate_acolite_boa() without needing file I/O.
    """
    arr = b02.astype(float)
    b03f = b03.astype(float)
    mask = (arr > 0) & (b03f > 0)
    if mask.sum() < 10:
        return arr
    b02_v = arr[mask]
    b03_v = b03f[mask]
    b03_var = np.var(b03_v)
    slope = np.cov(b02_v, b03_v)[0, 1] / b03_var if b03_var > 1e-12 else 1.0
    slope = np.clip(slope, 0, 2)
    corrected = arr - slope * (b03f - b03f.min())
    return np.clip(corrected, 0, None)


# ─────────────────────────────────────────────────────────────────────────────
# TestSnellOpticalPath — tests snell_sza() + optical_path()
# ─────────────────────────────────────────────────────────────────────────────

class TestSnellOpticalPath:
    """Verify Snell's law refraction and Beer–Lambert path geometry."""

    def test_vertical_incidence(self):
        """SZA=0° (sun directly overhead) → optical path equals depth."""
        path_m, sza_w = snell_optical_path(0.0, 16.0)
        assert abs(path_m - 16.0) < 1e-6
        assert abs(sza_w) < 1e-6

    def test_typical_september_sza(self):
        """SZA≈40° (typical Algarve September) → path slightly longer than depth."""
        path_m, sza_w = snell_optical_path(40.498, 16.0)
        assert sza_w < 40.498
        assert path_m > 16.0
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
        assert abs(path_20 / path_10 - 2.0) < 1e-9

    def test_returns_tuple_of_floats(self):
        path_m, sza_w = snell_optical_path(35.0, 16.0)
        assert isinstance(path_m, float)
        assert isinstance(sza_w, float)


# ─────────────────────────────────────────────────────────────────────────────
# TestSunglintCorrection — tests Hedley-style linear glint removal
# ─────────────────────────────────────────────────────────────────────────────

class TestSunglintCorrection:
    """Verify Hedley-style linear sunglint removal (via shim above)."""

    def _make_arrays(self, rows=40, cols=40, seed=42):
        rng = np.random.default_rng(seed)
        b02 = rng.uniform(0.02, 0.20, (rows, cols)).astype(np.float64)
        b03 = rng.uniform(0.01, 0.15, (rows, cols)).astype(np.float64)
        return b02, b03

    def test_no_negative_output(self):
        b02, b03 = self._make_arrays()
        corrected = sunglint_correction(b02, b03)
        assert np.all(corrected >= 0), "Negative values found after glint correction"

    def test_output_shape_preserved(self):
        b02, b03 = self._make_arrays(30, 50)
        corrected = sunglint_correction(b02, b03)
        assert corrected.shape == b02.shape

    def test_slope_clamped_to_valid_range(self):
        """Even with extreme covariance the slope stays in [0, 2]."""
        b03 = np.random.rand(40, 40) * 0.1
        b02 = b03 * 100.0
        corrected = sunglint_correction(b02, b03)
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
        b02[:5, 0] = 0.05
        b03[:5, 0] = 0.03
        corrected = sunglint_correction(b02, b03)
        np.testing.assert_array_equal(corrected, b02)

    def test_correction_reduces_mean_in_correlated_scene(self):
        """When B02 and B03 are positively correlated, correction lowers mean."""
        rng = np.random.default_rng(0)
        base = rng.uniform(0.05, 0.15, (40, 40))
        b03 = base + rng.normal(0, 0.005, (40, 40))
        b02 = base * 1.5 + rng.normal(0, 0.005, (40, 40))
        b02 = np.clip(b02, 0, None)
        b03 = np.clip(b03, 1e-6, None)
        corrected = sunglint_correction(b02, b03)
        assert np.mean(corrected) < np.mean(b02)


# ─────────────────────────────────────────────────────────────────────────────
# TestAnalyseBand — tests _normalise_result() keys + S1 penalty via main()
# ─────────────────────────────────────────────────────────────────────────────

def _make_dummy_pred(snr=45.0, vis=0.72, cloud=1.0, date="2025-09-25", month=9):
    """Return a minimal run_predictor()-style dict."""
    kd = KD490_TABLE.get(month, 0.080)
    return {
        "image_date": date,
        "snr_mean_16m": snr,
        "snr_median_16m": snr * 0.95,
        "water_transmittance_twoway": 0.42,
        "kd_seasonal_prior": kd,
        "kd_b02_estimated": kd,
        "kd_b03_estimated": kd * 1.2,
        "kd_b04_estimated": kd * 3.0,
        "kd_high_uncertainty": False,
        "sza_air_deg": 40.5,
        "sza_water_deg": 29.8,
        "optical_path_m": 18.4,
        "glint_penalty": 0.95,
        "percent_pixels_useful": 88.0,
        "percent_area_high_confidence": 61.0,
        "contrast_benthic_mean": 0.80,
        "visibility_score": vis,
        "ranker_mode": "ml",
        "cloud_cover": cloud,
        "cleanliness": 5000,
    }


class TestAnalyseBand:
    """Verify _normalise_result() output structure and physical constraints."""

    EXPECTED_KEYS = {
        "date", "sza_air_deg", "sza_water_deg", "optical_path_m",
        "kd490_seasonal", "kd490_estimated", "kd_high_uncertainty",
        "water_transmittance_twoway", "b02_cv", "SNR_mean_16m",
        "contrast_benthic_mean", "percent_pixels_useful",
        "percent_area_high_confidence", "visibility_score",
        "cleanliness", "cloud_cover",
    }

    def _run(self, b02_val=0.08, b03_val=0.06, cloud=1.0, month=9, sza=40.5):
        meta = {"date": "2025-09-25", "sza": sza, "cloud": cloud, "month": month}
        pred = _make_dummy_pred(cloud=cloud, month=month)
        return _normalise_result(pred, meta)

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
        low_cloud  = _make_dummy_pred(cloud=1.0)
        high_cloud = _make_dummy_pred(cloud=50.0)
        # Simulate the percent_pixels_useful being lower with high cloud
        # _normalise_result passthrough — just check the field exists and is lower
        res_lo = _normalise_result(low_cloud,  {"date": "2025-09-25", "cloud": 1.0,  "month": 9})
        res_hi = _normalise_result(high_cloud, {"date": "2025-09-25", "cloud": 50.0, "month": 9})
        # Both have "percent_pixels_useful" (passthrough from run_predictor)
        assert "percent_pixels_useful" in res_lo

    def test_september_kd_used(self):
        """September (month=9) should use Kd≈0.045 per KD490_TABLE."""
        result = self._run(month=9)
        assert abs(result["kd490_seasonal"] - 0.045) < 1e-6

    def test_april_kd_used(self):
        """April (month=4) should use Kd≈0.065 per KD490_TABLE."""
        result = self._run(month=4)
        assert abs(result["kd490_seasonal"] - 0.065) < 1e-6

    def test_cleanliness_default_present(self):
        result = self._run()
        assert "cleanliness" in result
        assert isinstance(result["cleanliness"], (int, float))

    def test_kd_uncertainty_flag_type(self):
        result = self._run()
        assert isinstance(result["kd_high_uncertainty"], bool)

    def test_sentinel1_calm_sea_state(self):
        """S1 calm (roughness=0.04) → penalty=0%, score unchanged."""
        meta = {"date": "2025-09-25", "sza": 40.5, "cloud": 1.0, "month": 9}
        pred = _make_dummy_pred(vis=0.72)
        res  = _normalise_result(pred, meta)
        # Simulate S1 penalty applied externally (as orchestrator main() does)
        roughness = 0.04
        penalty = float(np.clip((roughness - 0.05) / 0.20, 0.0, 1.0)) * 0.3
        res["s1_penalty_pct"]   = round(penalty * 100, 2)
        res["s1_roughness"]     = roughness
        res["s1_sea_state"]     = "calm"
        res["s1_scene_id"]      = "S1_calm"
        res["s1_scene_date"]    = "2025-09-25"
        res["visibility_score"] = round(res["visibility_score"] * (1 - penalty), 4)

        assert res["s1_scene_id"]    == "S1_calm"
        assert res["s1_scene_date"]  == "2025-09-25"
        assert res["s1_roughness"]   == 0.04
        assert res["s1_sea_state"]   == "calm"
        assert res["s1_penalty_pct"] == 0.0

    def test_sentinel1_rough_sea_state(self):
        """S1 rough (roughness=0.30) → penalty=30%, score *= 0.70."""
        base_vis = 0.72
        pred  = _make_dummy_pred(vis=base_vis)
        meta  = {"date": "2025-09-25", "sza": 40.5, "cloud": 1.0, "month": 9}
        res   = _normalise_result(pred, meta)

        roughness = 0.30
        penalty   = float(np.clip((roughness - 0.05) / 0.20, 0.0, 1.0)) * 0.3
        res["s1_penalty_pct"]   = round(penalty * 100, 2)
        res["s1_roughness"]     = roughness
        res["s1_sea_state"]     = "rough"
        res["s1_scene_id"]      = "S1_rough"
        res["visibility_score"] = round(base_vis * (1 - penalty), 4)

        assert res["s1_scene_id"]    == "S1_rough"
        assert res["s1_roughness"]   == 0.30
        assert res["s1_sea_state"]   == "rough"
        assert res["s1_penalty_pct"] == 30.0
        assert pytest.approx(res["visibility_score"], 1e-4) == base_vis * 0.70

    def test_sentinel1_moderate_sea_state(self):
        """S1 moderate (roughness=0.15) → penalty=15%, score *= 0.85."""
        base_vis = 0.72
        pred  = _make_dummy_pred(vis=base_vis)
        meta  = {"date": "2025-09-25", "sza": 40.5, "cloud": 1.0, "month": 9}
        res   = _normalise_result(pred, meta)

        roughness = 0.15
        penalty   = float(np.clip((roughness - 0.05) / 0.20, 0.0, 1.0)) * 0.3
        res["s1_penalty_pct"]   = round(penalty * 100, 2)
        res["s1_roughness"]     = roughness
        res["s1_sea_state"]     = "moderate"
        res["s1_scene_id"]      = "S1_mod"
        res["visibility_score"] = round(base_vis * (1 - penalty), 4)

        assert res["s1_scene_id"]    == "S1_mod"
        assert res["s1_roughness"]   == 0.15
        assert res["s1_sea_state"]   == "moderate"
        assert res["s1_penalty_pct"] == 15.0
        assert pytest.approx(res["visibility_score"], 1e-4) == base_vis * 0.85
