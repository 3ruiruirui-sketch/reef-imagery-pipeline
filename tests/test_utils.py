"""Tests for src/utils.py — physics utility functions."""
import math
import pytest
from src.utils import (
    snell_air_to_water,
    snell_sza,
    optical_path,
    beer_lambert_transmittance,
    get_kd490,
    compute_metadata_stub,
)


# ── snell_air_to_water ────────────────────────────────────────────────────────

class TestSnellAirToWater:
    def test_normal_incidence_gives_zero(self):
        assert snell_air_to_water(0.0) == pytest.approx(0.0, abs=1e-9)

    def test_30deg_incidence(self):
        # sin(30°)/1.333 = 0.5/1.333 = 0.375 → asin(0.375) ≈ 22.03°
        theta_w = snell_air_to_water(math.radians(30))
        assert math.degrees(theta_w) == pytest.approx(22.03, abs=0.1)

    def test_output_less_than_input(self):
        # Refracted angle in water is always smaller than air angle
        for deg in [10, 20, 30, 45, 60]:
            theta_air = math.radians(deg)
            theta_w = snell_air_to_water(theta_air)
            assert theta_w < theta_air

    def test_grazing_incidence_clamped(self):
        # Near 90° should not raise and should return a valid angle
        result = snell_air_to_water(math.radians(89.9))
        assert 0.0 <= result <= math.pi / 2


# ── snell_sza ─────────────────────────────────────────────────────────────────

class TestSnellSza:
    def test_returns_tuple(self):
        deg, rad = snell_sza(40.0)
        assert isinstance(deg, float)
        assert isinstance(rad, float)

    def test_consistent_with_snell_air_to_water(self):
        # snell_sza should agree with snell_air_to_water
        deg, rad = snell_sza(40.0)
        expected = snell_air_to_water(math.radians(40.0))
        assert rad == pytest.approx(expected, abs=1e-9)
        assert deg == pytest.approx(math.degrees(expected), abs=1e-6)

    def test_zero_sza(self):
        deg, rad = snell_sza(0.0)
        assert deg == pytest.approx(0.0, abs=1e-9)
        assert rad == pytest.approx(0.0, abs=1e-9)


# ── optical_path ──────────────────────────────────────────────────────────────

class TestOpticalPath:
    def test_vertical_ray(self):
        # theta = 0 → path = depth / cos(0) = depth
        assert optical_path(10.0, 0.0) == pytest.approx(10.0)

    def test_oblique_ray(self):
        # theta = 30° → path = 10 / cos(30°) ≈ 11.547
        path = optical_path(10.0, math.radians(30))
        assert path == pytest.approx(10.0 / math.cos(math.radians(30)), rel=1e-6)

    def test_zero_depth_gives_zero(self):
        assert optical_path(0.0, math.radians(20)) == pytest.approx(0.0)


# ── beer_lambert_transmittance ────────────────────────────────────────────────

class TestBeerLambertTransmittance:
    def test_zero_path_gives_one(self):
        assert beer_lambert_transmittance(0.042, 0.0) == pytest.approx(1.0)

    def test_zero_kd_gives_one(self):
        assert beer_lambert_transmittance(0.0, 20.0) == pytest.approx(1.0)

    def test_typical_algarve_10m(self):
        # kd=0.042 m⁻¹, two-way path 10m: T = exp(-2 * 0.042 * 10)
        expected = math.exp(-2 * 0.042 * 10)
        assert beer_lambert_transmittance(0.042, 10.0) == pytest.approx(expected, rel=1e-9)

    def test_transmittance_decreases_with_depth(self):
        kd = 0.042
        t_shallow = beer_lambert_transmittance(kd, 5.0)
        t_deep = beer_lambert_transmittance(kd, 20.0)
        assert t_shallow > t_deep

    def test_transmittance_decreases_with_turbidity(self):
        path = 10.0
        t_clear = beer_lambert_transmittance(0.042, path)
        t_turbid = beer_lambert_transmittance(0.10, path)
        assert t_clear > t_turbid

    def test_range_is_zero_to_one(self):
        for kd in [0.02, 0.042, 0.08, 0.15]:
            for depth in [1, 5, 10, 20, 40]:
                t = beer_lambert_transmittance(kd, depth)
                assert 0.0 < t <= 1.0


# ── get_kd490 ─────────────────────────────────────────────────────────────────

class TestGetKd490:
    def test_integer_key_lookup(self):
        table = {7: 0.042, 8: 0.042, 9: 0.046}
        assert get_kd490(7, table) == 0.042

    def test_string_key_fallback(self):
        # Some callers may pass string keys
        table = {"7": 0.042, "8": 0.042}
        assert get_kd490(7, table) == 0.042

    def test_missing_month_returns_default(self):
        table = {7: 0.042}
        result = get_kd490(11, table)
        assert result == 0.080  # hardcoded default in utils.get_kd490

    def test_int_cast_on_month(self):
        # String month should be cast to int for lookup
        table = {9: 0.046}
        assert get_kd490("9", table) == 0.046


# ── compute_metadata_stub ─────────────────────────────────────────────────────

class TestComputeMetadataStub:
    def test_known_date_returns_correct_sza(self):
        m = compute_metadata_stub("2025-09-25")
        assert m["solar_zenith_deg"] == pytest.approx(40.498)

    def test_unknown_date_returns_defaults(self):
        m = compute_metadata_stub("2000-01-01")
        assert m["solar_zenith_deg"] == pytest.approx(40.0)
        assert m["cloud_cover_pct"] == pytest.approx(2.0)

    def test_required_keys_present(self):
        m = compute_metadata_stub("2025-09-25")
        for key in ("date", "crs", "datum", "level", "solar_zenith_deg",
                    "solar_azimuth_deg", "cloud_cover_pct"):
            assert key in m

    def test_date_field_matches_input(self):
        m = compute_metadata_stub("2023-10-01")
        assert m["date"] == "2023-10-01"
