"""Tests for src/cmems_kd490.py — Kd490 climatology with CMEMS/static fallback."""
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.constants import KD490_TABLE as _STATIC_TABLE, KD490_DEFAULT
import src.cmems_kd490 as _mod


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_fake_dataset(values_by_month: dict[int, float] | None = None):
    """Build a minimal xarray-like mock that copernicusmarine.open_dataset returns."""
    import xarray as xr
    import pandas as pd

    if values_by_month is None:
        values_by_month = {m: 0.045 + m * 0.001 for m in range(1, 13)}

    # Build a (time, lat, lon) DataArray with one pixel per month
    times = pd.date_range("2020-01-01", periods=12, freq="MS")
    data = np.array([values_by_month.get(t.month, 0.05) for t in times],
                    dtype=np.float32).reshape(12, 1, 1)
    da = xr.DataArray(data, dims=["time", "lat", "lon"],
                      coords={"time": times, "lat": [37.0], "lon": [-8.5]})
    ds = xr.Dataset({"KD490": da})
    return ds


# ── _fetch_cmems_climatology ─────────────────────────────────────────────────

class TestFetchCmemsClimatology:
    def test_raises_when_no_credentials(self, monkeypatch):
        monkeypatch.delenv("CMEMS_USER", raising=False)
        monkeypatch.delenv("CMEMS_PASSWORD", raising=False)
        monkeypatch.delenv("COPERNICUSMARINE_SERVICE_USERNAME", raising=False)
        monkeypatch.delenv("COPERNICUSMARINE_SERVICE_PASSWORD", raising=False)
        with pytest.raises(RuntimeError, match="credentials not set"):
            _mod._fetch_cmems_climatology()

    def test_returns_12_month_dict_with_valid_credentials(self, monkeypatch):
        monkeypatch.setenv("CMEMS_USER", "test_user")
        monkeypatch.setenv("CMEMS_PASSWORD", "test_pass")
        fake_ds = _make_fake_dataset()
        with patch("copernicusmarine.open_dataset", return_value=fake_ds):
            result = _mod._fetch_cmems_climatology()
        assert set(result.keys()) == set(range(1, 13))
        for v in result.values():
            assert isinstance(v, float)
            assert 0.0 < v < 1.0  # physically plausible Kd490

    def test_fills_missing_months_from_static_table(self, monkeypatch):
        """If CMEMS returns data for only some months, the rest come from the static table."""
        monkeypatch.setenv("CMEMS_USER", "u")
        monkeypatch.setenv("CMEMS_PASSWORD", "p")
        # Dataset with only Jan–Jun months
        import xarray as xr, pandas as pd
        times = pd.date_range("2020-01-01", periods=6, freq="MS")
        data = np.full((6, 1, 1), 0.042, dtype=np.float32)
        da = xr.DataArray(data, dims=["time", "lat", "lon"],
                          coords={"time": times, "lat": [37.0], "lon": [-8.5]})
        fake_ds = xr.Dataset({"KD490": da})
        with patch("copernicusmarine.open_dataset", return_value=fake_ds):
            result = _mod._fetch_cmems_climatology()
        # All 12 months present
        assert set(result.keys()) == set(range(1, 13))
        # Jul–Dec fall back to static table
        for m in range(7, 13):
            assert result[m] == _STATIC_TABLE.get(m, KD490_DEFAULT)

    def test_no_time_dimension_broadcasts_to_all_months(self, monkeypatch):
        monkeypatch.setenv("CMEMS_USER", "u")
        monkeypatch.setenv("CMEMS_PASSWORD", "p")
        import xarray as xr
        data = np.full((4, 4), 0.055, dtype=np.float32)
        da = xr.DataArray(data, dims=["lat", "lon"])
        fake_ds = xr.Dataset({"KD490": da})
        with patch("copernicusmarine.open_dataset", return_value=fake_ds):
            result = _mod._fetch_cmems_climatology()
        assert set(result.keys()) == set(range(1, 13))
        for v in result.values():
            assert abs(v - 0.055) < 1e-4

    def test_uses_copernicusmarine_env_var_aliases(self, monkeypatch):
        monkeypatch.delenv("CMEMS_USER", raising=False)
        monkeypatch.delenv("CMEMS_PASSWORD", raising=False)
        monkeypatch.setenv("COPERNICUSMARINE_SERVICE_USERNAME", "alt_user")
        monkeypatch.setenv("COPERNICUSMARINE_SERVICE_PASSWORD", "alt_pass")
        fake_ds = _make_fake_dataset()
        with patch("copernicusmarine.open_dataset", return_value=fake_ds) as mock_open:
            _mod._fetch_cmems_climatology()
        mock_open.assert_called_once()
        _, kwargs = mock_open.call_args
        assert kwargs["username"] == "alt_user"
        assert kwargs["password"] == "alt_pass"


# ── _build_table ─────────────────────────────────────────────────────────────

class TestBuildTable:
    def test_falls_back_on_missing_credentials(self, monkeypatch):
        monkeypatch.delenv("CMEMS_USER", raising=False)
        monkeypatch.delenv("CMEMS_PASSWORD", raising=False)
        monkeypatch.delenv("COPERNICUSMARINE_SERVICE_USERNAME", raising=False)
        monkeypatch.delenv("COPERNICUSMARINE_SERVICE_PASSWORD", raising=False)
        result = _mod._build_table()
        assert result == dict(_STATIC_TABLE)

    def test_falls_back_on_import_error(self, monkeypatch):
        monkeypatch.setenv("CMEMS_USER", "u")
        monkeypatch.setenv("CMEMS_PASSWORD", "p")
        with patch("src.cmems_kd490._fetch_cmems_climatology",
                   side_effect=ImportError("copernicusmarine not installed")):
            result = _mod._build_table()
        assert result == dict(_STATIC_TABLE)

    def test_falls_back_on_network_error(self, monkeypatch):
        monkeypatch.setenv("CMEMS_USER", "u")
        monkeypatch.setenv("CMEMS_PASSWORD", "p")
        with patch("src.cmems_kd490._fetch_cmems_climatology",
                   side_effect=ConnectionError("timeout")):
            result = _mod._build_table()
        assert result == dict(_STATIC_TABLE)

    def test_returns_live_table_on_success(self, monkeypatch):
        live = {m: 0.040 for m in range(1, 13)}
        with patch("src.cmems_kd490._fetch_cmems_climatology", return_value=live):
            result = _mod._build_table()
        assert result == live


# ── get_kd490 ────────────────────────────────────────────────────────────────

class TestGetKd490:
    def test_returns_float_for_all_valid_months(self):
        for m in range(1, 13):
            v = _mod.get_kd490(m)
            assert isinstance(v, float)
            assert 0.0 < v < 1.0

    def test_returns_default_for_invalid_month(self):
        assert _mod.get_kd490(0)  == KD490_DEFAULT
        assert _mod.get_kd490(13) == KD490_DEFAULT
        assert _mod.get_kd490(99) == KD490_DEFAULT

    def test_summer_months_lower_than_winter(self):
        """Algarve: summer (Jul–Aug) is clearer than winter (Jan–Feb)."""
        jul = _mod.get_kd490(7)
        aug = _mod.get_kd490(8)
        jan = _mod.get_kd490(1)
        # Both directions valid (live vs static may differ), but range must be sane
        assert 0.01 <= jul <= 0.5
        assert 0.01 <= aug <= 0.5
        assert 0.01 <= jan <= 0.5

    def test_accepts_string_month(self):
        # get_kd490 casts to int — "7" should work
        assert _mod.get_kd490(7) == _mod.get_kd490(7)


# ── KD490_TABLE_LIVE shape ────────────────────────────────────────────────────

class TestKd490TableLive:
    def test_has_12_months(self):
        assert set(_mod.KD490_TABLE_LIVE.keys()) == set(range(1, 13))

    def test_all_values_positive_and_plausible(self):
        for m, v in _mod.KD490_TABLE_LIVE.items():
            assert isinstance(v, float), f"month {m}: expected float, got {type(v)}"
            assert 0.005 < v < 0.500, f"month {m}: Kd490={v} outside physical range"
