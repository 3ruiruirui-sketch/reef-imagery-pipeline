"""Tests for src/sh_downloader.py — dual Sentinel Hub endpoint support.

Covers endpoint selection, credential handling, and the auto-fallback chain
between the commercial SH endpoint and the free CDSE Sentinel Hub endpoint.
All HTTP traffic is mocked — no live network calls.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import rasterio
from rasterio.io import MemoryFile

import src.sh_downloader as sh

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _isolate_credentials(monkeypatch):
    """Wipe all SH/CDSE credential env vars and the in-process token cache
    so each test starts from a known-empty state."""
    for var in ("SH_CLIENT_ID", "SH_CLIENT_SECRET",
                "CDSE_CLIENT_ID", "CDSE_CLIENT_SECRET"):
        monkeypatch.delenv(var, raising=False)
    sh._token_cache["commercial"].token = None
    sh._token_cache["commercial"].expires_at = 0.0
    sh._token_cache["cdse"].token       = None
    sh._token_cache["cdse"].expires_at  = 0.0
    yield


def _fake_token_response(token: str = "fake-jwt-token") -> dict:
    return {"access_token": token, "expires_in": 3600}


def _fake_ok_response(body: bytes | dict | list = b"") -> MagicMock:
    resp = MagicMock()
    resp.ok = True
    resp.status_code = 200
    if isinstance(body, (dict, list)):
        resp.content = json.dumps(body).encode()
        resp.json.return_value = body
    else:
        resp.content = body
        resp.json.return_value = {}
    return resp


def _fake_error_response(status: int, body: str = "err") -> MagicMock:
    resp = MagicMock()
    resp.ok = False
    resp.status_code = status
    resp.text = body
    resp.content = body.encode()
    return resp


def _fake_tiff_bytes(width: int = 4, height: int = 4, bands: int = 2) -> bytes:
    """Build a minimal in-memory float32 GeoTIFF so the rasterio round-trip
    succeeds inside download_scene()."""
    data = np.zeros((bands, height, width), dtype="float32")
    transform = rasterio.transform.from_bounds(0, 0, 1, 1, width, height)
    with MemoryFile() as mem:
        with mem.open(
            driver="GTiff", height=height, width=width, count=bands,
            dtype="float32", crs="EPSG:4326", transform=transform,
        ) as dst:
            dst.write(data)
        return mem.read()


# ─────────────────────────────────────────────────────────────────────────────
# _available_endpoints / _endpoint_urls
# ─────────────────────────────────────────────────────────────────────────────

class TestEndpointRegistry:
    def test_no_credentials_returns_empty(self):
        assert sh._available_endpoints() == []

    def test_commercial_only(self, monkeypatch):
        monkeypatch.setenv("SH_CLIENT_ID", "x")
        monkeypatch.setenv("SH_CLIENT_SECRET", "y")
        assert sh._available_endpoints() == ["commercial"]

    def test_cdse_only(self, monkeypatch):
        monkeypatch.setenv("CDSE_CLIENT_ID", "x")
        monkeypatch.setenv("CDSE_CLIENT_SECRET", "y")
        assert sh._available_endpoints() == ["cdse"]

    def test_both_endpoints_available(self, monkeypatch):
        monkeypatch.setenv("SH_CLIENT_ID", "x")
        monkeypatch.setenv("SH_CLIENT_SECRET", "y")
        monkeypatch.setenv("CDSE_CLIENT_ID", "a")
        monkeypatch.setenv("CDSE_CLIENT_SECRET", "b")
        assert sh._available_endpoints() == ["commercial", "cdse"]

    def test_only_secret_missing_excludes_endpoint(self, monkeypatch):
        monkeypatch.setenv("SH_CLIENT_ID", "x")
        monkeypatch.setenv("CDSE_CLIENT_ID", "a")
        monkeypatch.setenv("CDSE_CLIENT_SECRET", "b")
        assert sh._available_endpoints() == ["cdse"]

    def test_endpoint_urls_resolve_commercial(self):
        assert sh._endpoint_urls("commercial") == "https://services.sentinel-hub.com"

    def test_endpoint_urls_resolve_cdse(self):
        assert sh._endpoint_urls("cdse") == "https://sh.dataspace.copernicus.eu"

    def test_endpoint_urls_rejects_auto(self):
        with pytest.raises(KeyError):
            sh._endpoint_urls("auto")


# ─────────────────────────────────────────────────────────────────────────────
# _fallback_chain
# ─────────────────────────────────────────────────────────────────────────────

class TestFallbackChain:
    def test_explicit_endpoint_with_creds(self, monkeypatch):
        monkeypatch.setenv("SH_CLIENT_ID", "x")
        monkeypatch.setenv("SH_CLIENT_SECRET", "y")
        assert sh._fallback_chain("commercial") == ["commercial"]

    def test_explicit_endpoint_without_creds_raises(self):
        with pytest.raises(RuntimeError, match="not set in the environment"):
            sh._fallback_chain("commercial")

    def test_auto_picks_commercial_when_both(self, monkeypatch):
        monkeypatch.setenv("SH_CLIENT_ID", "x")
        monkeypatch.setenv("SH_CLIENT_SECRET", "y")
        monkeypatch.setenv("CDSE_CLIENT_ID", "a")
        monkeypatch.setenv("CDSE_CLIENT_SECRET", "b")
        assert sh._fallback_chain("auto") == ["commercial", "cdse"]

    def test_auto_cdse_only(self, monkeypatch):
        monkeypatch.setenv("CDSE_CLIENT_ID", "a")
        monkeypatch.setenv("CDSE_CLIENT_SECRET", "b")
        assert sh._fallback_chain("auto") == ["cdse"]

    def test_auto_no_creds_raises(self):
        with pytest.raises(RuntimeError, match="No Sentinel Hub credentials"):
            sh._fallback_chain("auto")


# ─────────────────────────────────────────────────────────────────────────────
# _get_token / token cache
# ─────────────────────────────────────────────────────────────────────────────

class TestGetToken:
    def test_token_request_uses_correct_token_url_for_commercial(self, monkeypatch):
        monkeypatch.setenv("SH_CLIENT_ID", "id1")
        monkeypatch.setenv("SH_CLIENT_SECRET", "sec1")
        with patch("src.sh_downloader.requests.post",
                   return_value=_fake_ok_response(_fake_token_response("tok-c"))) as mock_post:
            tok = sh._get_token("commercial")
        assert tok == "tok-c"
        url = mock_post.call_args.args[0]
        assert url == sh._ENDPOINTS["commercial"]["token_url"]
        body = mock_post.call_args.kwargs["data"]
        assert body["grant_type"] == "client_credentials"
        assert body["client_id"] == "id1"
        assert body["client_secret"] == "sec1"

    def test_token_request_uses_correct_token_url_for_cdse(self, monkeypatch):
        monkeypatch.setenv("CDSE_CLIENT_ID", "id2")
        monkeypatch.setenv("CDSE_CLIENT_SECRET", "sec2")
        with patch("src.sh_downloader.requests.post",
                   return_value=_fake_ok_response(_fake_token_response("tok-d"))) as mock_post:
            tok = sh._get_token("cdse")
        assert tok == "tok-d"
        url = mock_post.call_args.args[0]
        assert url == sh._ENDPOINTS["cdse"]["token_url"]
        body = mock_post.call_args.kwargs["data"]
        assert body["client_id"] == "id2"

    def test_token_is_cached_and_not_refetched(self, monkeypatch):
        monkeypatch.setenv("SH_CLIENT_ID", "x")
        monkeypatch.setenv("SH_CLIENT_SECRET", "y")
        sh._token_cache["commercial"].token      = "cached-tok"
        sh._token_cache["commercial"].expires_at = 9_999_999_999
        with patch("src.sh_downloader.requests.post") as mock_post:
            assert sh._get_token("commercial") == "cached-tok"
            mock_post.assert_not_called()

    def test_missing_credentials_raises(self, monkeypatch):
        with pytest.raises(RuntimeError, match="SH_CLIENT_ID"):
            sh._get_token("commercial")


# ─────────────────────────────────────────────────────────────────────────────
# _post_with_fallback
# ─────────────────────────────────────────────────────────────────────────────

class TestPostWithFallback:
    def test_returns_first_endpoint_response_when_ok(self, monkeypatch):
        monkeypatch.setenv("SH_CLIENT_ID", "x")
        monkeypatch.setenv("SH_CLIENT_SECRET", "y")
        responses = [
            _fake_ok_response(_fake_token_response()),   # token fetch
            _fake_ok_response({"ok": True}),             # actual request
        ]
        with patch("src.sh_downloader.requests.post",
                   side_effect=responses) as mock_post:
            resp = sh._post_with_fallback("/api/v1/process", {"a": 1},
                                          endpoint="commercial", timeout=10)
        assert resp.json() == {"ok": True}
        assert mock_post.call_count == 2
        # First call: token URL. Second call: process URL.
        assert mock_post.call_args_list[0].args[0].endswith(
            "/auth/realms/main/protocol/openid-connect/token"
        )
        assert mock_post.call_args_list[1].args[0] == "https://services.sentinel-hub.com/api/v1/process"

    def test_falls_back_on_401_to_cdse(self, monkeypatch):
        monkeypatch.setenv("SH_CLIENT_ID", "x")
        monkeypatch.setenv("SH_CLIENT_SECRET", "y")
        monkeypatch.setenv("CDSE_CLIENT_ID", "a")
        monkeypatch.setenv("CDSE_CLIENT_SECRET", "b")

        responses = [
            _fake_ok_response(_fake_token_response()),      # 1: token for commercial
            _fake_error_response(401, "trial ran out"),      # 2: process on commercial
            _fake_ok_response(_fake_token_response()),      # 3: token for cdse
            _fake_ok_response({"fallback": "ok"}),           # 4: process on cdse
        ]
        with patch("src.sh_downloader.requests.post",
                   side_effect=responses) as mock_post:
            resp = sh._post_with_fallback("/api/v1/process", {"a": 1},
                                          endpoint="auto", timeout=10)
        assert resp.json() == {"fallback": "ok"}
        assert mock_post.call_count == 4
        # Sequence: commercial token → commercial process (401) → cdse token → cdse process
        assert "identity.dataspace.copernicus.eu" in mock_post.call_args_list[2].args[0]
        assert mock_post.call_args_list[3].args[0] == \
            "https://sh.dataspace.copernicus.eu/api/v1/process"

    def test_falls_back_on_403(self, monkeypatch):
        monkeypatch.setenv("SH_CLIENT_ID", "x")
        monkeypatch.setenv("SH_CLIENT_SECRET", "y")
        monkeypatch.setenv("CDSE_CLIENT_ID", "a")
        monkeypatch.setenv("CDSE_CLIENT_SECRET", "b")
        responses = [
            _fake_ok_response(_fake_token_response()),
            _fake_error_response(403),
            _fake_ok_response(_fake_token_response()),
            _fake_ok_response({}),
        ]
        with patch("src.sh_downloader.requests.post", side_effect=responses):
            resp = sh._post_with_fallback("/api/v1/process", {},
                                          endpoint="auto", timeout=10)
        assert resp.ok

    def test_falls_back_on_429_quota(self, monkeypatch):
        monkeypatch.setenv("SH_CLIENT_ID", "x")
        monkeypatch.setenv("SH_CLIENT_SECRET", "y")
        monkeypatch.setenv("CDSE_CLIENT_ID", "a")
        monkeypatch.setenv("CDSE_CLIENT_SECRET", "b")
        responses = [
            _fake_ok_response(_fake_token_response()),
            _fake_error_response(429),
            _fake_ok_response(_fake_token_response()),
            _fake_ok_response({}),
        ]
        with patch("src.sh_downloader.requests.post", side_effect=responses):
            resp = sh._post_with_fallback("/api/v1/process", {},
                                          endpoint="auto", timeout=10)
        assert resp.ok

    def test_does_not_fall_back_on_400_bad_request(self, monkeypatch):
        monkeypatch.setenv("SH_CLIENT_ID", "x")
        monkeypatch.setenv("SH_CLIENT_SECRET", "y")
        # Only commercial is configured — no fallback should be tried.
        responses = [
            _fake_ok_response(_fake_token_response()),
            _fake_error_response(400, "bad evalscript"),
        ]
        with (
            patch("src.sh_downloader.requests.post", side_effect=responses) as mock_post,
            pytest.raises(RuntimeError, match="HTTP 400"),
        ):
            sh._post_with_fallback("/api/v1/process", {}, endpoint="commercial", timeout=10)
        # token (1) + commercial process (1) = 2 calls, no fallback
        assert mock_post.call_count == 2

    def test_400_in_auto_mode_does_try_fallback(self, monkeypatch):
        """In auto mode with both endpoints available, any HTTP error on the
        first endpoint should attempt the fallback — a 400 might be endpoint-
        specific, and we want to use the free endpoint whenever possible."""
        monkeypatch.setenv("SH_CLIENT_ID", "x")
        monkeypatch.setenv("SH_CLIENT_SECRET", "y")
        monkeypatch.setenv("CDSE_CLIENT_ID", "a")
        monkeypatch.setenv("CDSE_CLIENT_SECRET", "b")
        responses = [
            _fake_ok_response(_fake_token_response()),
            _fake_error_response(400, "bad evalscript"),
            _fake_ok_response(_fake_token_response()),
            _fake_ok_response({}),
        ]
        with patch("src.sh_downloader.requests.post", side_effect=responses) as mock_post:
            resp = sh._post_with_fallback("/api/v1/process", {},
                                          endpoint="auto", timeout=10)
        assert resp.ok
        assert mock_post.call_count == 4

    def test_both_endpoints_fail_raises_with_detail(self, monkeypatch):
        monkeypatch.setenv("SH_CLIENT_ID", "x")
        monkeypatch.setenv("SH_CLIENT_SECRET", "y")
        monkeypatch.setenv("CDSE_CLIENT_ID", "a")
        monkeypatch.setenv("CDSE_CLIENT_SECRET", "b")
        responses = [
            _fake_ok_response(_fake_token_response()),   # token for commercial
            _fake_error_response(401, "trial ran out"),   # commercial fails
            _fake_ok_response(_fake_token_response()),   # token for cdse
            _fake_error_response(500, "server boom"),     # cdse also fails
        ]
        with (
            patch("src.sh_downloader.requests.post", side_effect=responses),
            pytest.raises(RuntimeError, match="All Sentinel Hub endpoints failed"),
        ):
            sh._post_with_fallback("/api/v1/process", {}, endpoint="auto", timeout=10)

    def test_explicit_endpoint_does_not_fallback(self, monkeypatch):
        monkeypatch.setenv("SH_CLIENT_ID", "x")
        monkeypatch.setenv("SH_CLIENT_SECRET", "y")
        monkeypatch.setenv("CDSE_CLIENT_ID", "a")
        monkeypatch.setenv("CDSE_CLIENT_SECRET", "b")
        responses = [
            _fake_ok_response(_fake_token_response()),
            _fake_error_response(401),
        ]
        with (
            patch("src.sh_downloader.requests.post", side_effect=responses) as mock_post,
            pytest.raises(RuntimeError, match="HTTP 401"),
        ):
            sh._post_with_fallback("/api/v1/process", {}, endpoint="commercial", timeout=10)
        # token (1) + commercial process (1) = 2 calls, no fallback attempted
        assert mock_post.call_count == 2


# ─────────────────────────────────────────────────────────────────────────────
# download_scene — integration with mocked HTTP
# ─────────────────────────────────────────────────────────────────────────────

class TestDownloadScene:
    def test_writes_geotiff_and_uses_process_url(self, monkeypatch, tmp_path):
        monkeypatch.setenv("SH_CLIENT_ID", "x")
        monkeypatch.setenv("SH_CLIENT_SECRET", "y")

        tiff_bytes = _fake_tiff_bytes(width=4, height=4, bands=2)
        responses = [
            _fake_ok_response(_fake_token_response()),  # token request
            _fake_ok_response(tiff_bytes),               # process request
        ]
        with patch("src.sh_downloader.requests.post", side_effect=responses) as mock_post:
            out = sh.download_scene(
                bbox=(-0.1, 37.0, -0.05, 37.05),
                date_range=("2024-09-01", "2024-09-30"),
                bands=["B02", "B04"],
                output_path=tmp_path / "scene.tif",
            )

        assert out == tmp_path / "scene.tif"
        assert out.exists()
        # Process call should be the 2nd call, hitting the SH process URL.
        process_call = mock_post.call_args_list[1]
        assert process_call.args[0] == "https://services.sentinel-hub.com/api/v1/process"
        with rasterio.open(out) as ds:
            assert ds.count == 2
            assert ds.dtypes[0] == "float32"
            assert ds.crs.to_epsg() == 4326

    def test_falls_back_to_cdse_when_commercial_401(self, monkeypatch, tmp_path):
        monkeypatch.setenv("SH_CLIENT_ID", "x")
        monkeypatch.setenv("SH_CLIENT_SECRET", "y")
        monkeypatch.setenv("CDSE_CLIENT_ID", "a")
        monkeypatch.setenv("CDSE_CLIENT_SECRET", "b")

        tiff_bytes = _fake_tiff_bytes(width=2, height=2, bands=4)
        # Sequence:
        #   1. token for commercial
        #   2. process on commercial → 401
        #   3. token for cdse
        #   4. process on cdse → tiff
        responses = [
            _fake_ok_response(_fake_token_response()),
            _fake_error_response(401),
            _fake_ok_response(_fake_token_response()),
            _fake_ok_response(tiff_bytes),
        ]
        with patch("src.sh_downloader.requests.post", side_effect=responses) as mock_post:
            out = sh.download_scene(
                bbox=(-0.1, 37.0, -0.05, 37.05),
                date_range=("2024-09-01", "2024-09-30"),
                bands=["B02", "B03", "B04", "B08"],
                output_path=tmp_path / "fallback.tif",
                endpoint="auto",
            )
        assert out.exists()
        # Fourth POST must hit the CDSE process URL.
        assert mock_post.call_args_list[3].args[0] == \
            "https://sh.dataspace.copernicus.eu/api/v1/process"

    def test_explicit_cdse_endpoint_skips_commercial(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CDSE_CLIENT_ID", "a")
        monkeypatch.setenv("CDSE_CLIENT_SECRET", "b")

        tiff_bytes = _fake_tiff_bytes(width=2, height=2, bands=1)
        responses = [
            _fake_ok_response(_fake_token_response()),
            _fake_ok_response(tiff_bytes),
        ]
        with patch("src.sh_downloader.requests.post", side_effect=responses) as mock_post:
            out = sh.download_scene(
                bbox=(-0.1, 37.0, -0.05, 37.05),
                date_range=("2024-09-01", "2024-09-30"),
                bands=["B02"],
                output_path=tmp_path / "cdse_only.tif",
                endpoint="cdse",
            )
        assert out.exists()
        # Token URL must be CDSE
        assert "identity.dataspace.copernicus.eu" in mock_post.call_args_list[0].args[0]
        # Process URL must be CDSE
        assert mock_post.call_args_list[1].args[0] == \
            "https://sh.dataspace.copernicus.eu/api/v1/process"


# ─────────────────────────────────────────────────────────────────────────────
# download_patch — same fallback, convenience wrapper
# ─────────────────────────────────────────────────────────────────────────────

class TestDownloadPatch:
    def test_uses_download_scene_under_the_hood(self, monkeypatch, tmp_path):
        monkeypatch.setenv("SH_CLIENT_ID", "x")
        monkeypatch.setenv("SH_CLIENT_SECRET", "y")
        tiff_bytes = _fake_tiff_bytes(width=2, height=2, bands=1)
        responses = [
            _fake_ok_response(_fake_token_response()),
            _fake_ok_response(tiff_bytes),
        ]
        with patch("src.sh_downloader.requests.post", side_effect=responses):
            out = sh.download_patch(
                lon=-8.2102, lat=37.0691,
                size_m=1280,
                date_range=("2024-09-01", "2024-09-30"),
                bands=["B02"],
                output_path=tmp_path / "patch.tif",
                endpoint="commercial",
            )
        assert out.exists()


# ─────────────────────────────────────────────────────────────────────────────
# search_scenes — fallback chain across endpoints
# ─────────────────────────────────────────────────────────────────────────────

class TestSearchScenes:
    def test_returns_features_sorted_by_cloud(self, monkeypatch):
        monkeypatch.setenv("SH_CLIENT_ID", "x")
        monkeypatch.setenv("SH_CLIENT_SECRET", "y")
        catalog = {
            "features": [
                {"id": "A", "properties": {"datetime": "2024-09-15",
                                           "eo:cloud_cover": 30.0},
                 "assets": {"B02": {}}},
                {"id": "B", "properties": {"datetime": "2024-09-10",
                                           "eo:cloud_cover": 5.0},
                 "assets": {"B02": {}}},
                {"id": "C", "properties": {"datetime": "2024-09-12",
                                           "eo:cloud_cover": 15.0},
                 "assets": {"B02": {}}},
            ]
        }
        responses = [
            _fake_ok_response(_fake_token_response()),
            _fake_ok_response(catalog),
        ]
        with patch("src.sh_downloader.requests.post", side_effect=responses):
            results = sh.search_scenes(
                bbox=(-0.1, 37.0, -0.05, 37.05),
                date_range=("2024-09-01", "2024-09-30"),
                max_cloud=20.0,
            )
        ids = [r["id"] for r in results]
        assert ids == ["B", "C"]
        assert all(r["cloud_cover"] <= 20.0 for r in results)

    def test_falls_back_to_cdse_when_commercial_catalog_fails(self, monkeypatch):
        monkeypatch.setenv("SH_CLIENT_ID", "x")
        monkeypatch.setenv("SH_CLIENT_SECRET", "y")
        monkeypatch.setenv("CDSE_CLIENT_ID", "a")
        monkeypatch.setenv("CDSE_CLIENT_SECRET", "b")
        catalog = {"features": []}
        responses = [
            _fake_ok_response(_fake_token_response()),
            _fake_error_response(403),
            _fake_ok_response(_fake_token_response()),
            _fake_ok_response(catalog),
        ]
        with patch("src.sh_downloader.requests.post", side_effect=responses) as mock_post:
            results = sh.search_scenes(
                bbox=(-0.1, 37.0, -0.05, 37.05),
                date_range=("2024-09-01", "2024-09-30"),
                endpoint="auto",
            )
        assert results == []
        assert mock_post.call_args_list[3].args[0] == \
            "https://sh.dataspace.copernicus.eu/api/v1/catalog/1.0.0/search"


# ─────────────────────────────────────────────────────────────────────────────
# get_stats — same fallback wiring
# ─────────────────────────────────────────────────────────────────────────────

class TestGetStats:
    def test_returns_stats_from_endpoint(self, monkeypatch):
        monkeypatch.setenv("SH_CLIENT_ID", "x")
        monkeypatch.setenv("SH_CLIENT_SECRET", "y")
        body = {
            "data": [{
                "interval": {"from": "2024-09-01T00:00:00Z",
                             "to":   "2024-10-01T00:00:00Z"},
                "outputs": {
                    "default": {
                        "bands": {
                            "B0": {"stats": {"mean": 0.05, "stDev": 0.01,
                                             "min": 0.0, "max": 0.5,
                                             "percentiles": {"10.0": 0.01,
                                                             "50.0": 0.05,
                                                             "90.0": 0.1},
                                             "sampleCount": 1000,
                                             "noDataCount": 10}},
                            "B1": {"stats": {"mean": 0.04, "stDev": 0.01,
                                             "min": 0.0, "max": 0.4,
                                             "percentiles": {"10.0": 0.01,
                                                             "50.0": 0.04,
                                                             "90.0": 0.09},
                                             "sampleCount": 1000,
                                             "noDataCount": 5}},
                        }
                    }
                }
            }]
        }
        responses = [
            _fake_ok_response(_fake_token_response()),
            _fake_ok_response(body),
        ]
        with patch("src.sh_downloader.requests.post", side_effect=responses):
            stats = sh.get_stats(
                bbox=(-0.1, 37.0, -0.05, 37.05),
                date_range=("2024-09-01", "2024-09-30"),
                bands=["B02", "B03"],
            )
        assert len(stats) == 1
        assert stats[0]["bands"]["B02"]["mean"] == pytest.approx(0.05, abs=1e-6)
        assert stats[0]["bands"]["B03"]["n_valid"] == 995


# ─────────────────────────────────────────────────────────────────────────────
# Regression tests for review findings
# ─────────────────────────────────────────────────────────────────────────────

class TestSearchScenesRegression:
    def test_total_network_failure_raises(self, monkeypatch, caplog):
        """When every configured endpoint raises RequestException, search_scenes
        must raise RuntimeError so callers don't silently treat the failure
        as 'no scenes found'."""
        import logging
        monkeypatch.setenv("SH_CLIENT_ID", "x")
        monkeypatch.setenv("SH_CLIENT_SECRET", "y")
        monkeypatch.setenv("CDSE_CLIENT_ID", "a")
        monkeypatch.setenv("CDSE_CLIENT_SECRET", "b")

        import requests as _requests
        responses = [
            _fake_ok_response(_fake_token_response()),
            _requests.ConnectionError("dns lookup failed"),
            _fake_ok_response(_fake_token_response()),
            _requests.ConnectionError("connection refused"),
        ]
        with (
            caplog.at_level(logging.WARNING),
            patch("src.sh_downloader.requests.post", side_effect=responses),
            pytest.raises(RuntimeError, match="network errors"),
        ):
            sh.search_scenes(
                bbox=(-0.1, 37.0, -0.05, 37.05),
                date_range=("2024-09-01", "2024-09-30"),
                endpoint="auto",
            )

    def test_200_with_empty_features_does_not_warn(self, monkeypatch, caplog):
        """A successful HTTP 200 with an empty `features` array is NOT a
        catalog-access failure — don't log a misleading warning."""
        import logging
        monkeypatch.setenv("SH_CLIENT_ID", "x")
        monkeypatch.setenv("SH_CLIENT_SECRET", "y")

        responses = [
            _fake_ok_response(_fake_token_response()),
            _fake_ok_response({"features": []}),
        ]
        with (
            caplog.at_level(logging.WARNING, logger="src.sh_downloader"),
            patch("src.sh_downloader.requests.post", side_effect=responses),
        ):
            results = sh.search_scenes(
                bbox=(-0.1, 37.0, -0.05, 37.05),
                date_range=("2024-09-01", "2024-09-30"),
                endpoint="commercial",
            )
        assert results == []
        # The misleading "trial may restrict catalog access" warning must NOT fire
        # on a 200+empty response.
        assert not any("trial may restrict catalog access" in rec.message
                       for rec in caplog.records), \
            f"unexpected warning: {[r.message for r in caplog.records]}"


class TestTokenCacheInvalidation:
    def test_failed_endpoint_token_is_cleared_on_auto_fallback(self, monkeypatch, tmp_path):
        """When auto-mode falls back from commercial to CDSE after a 401, the
        failed endpoint's cached token must be invalidated so subsequent calls
        don't re-use the known-bad token."""
        monkeypatch.setenv("SH_CLIENT_ID", "x")
        monkeypatch.setenv("SH_CLIENT_SECRET", "y")
        monkeypatch.setenv("CDSE_CLIENT_ID", "a")
        monkeypatch.setenv("CDSE_CLIENT_SECRET", "b")

        # Pre-populate commercial cache with a still-valid (but bad) token.
        # The first call will use it directly, get a 401, and trigger fallback.
        sh._token_cache["commercial"].token      = "bad-cached-token"
        sh._token_cache["commercial"].expires_at = 9_999_999_999

        tiff_bytes = _fake_tiff_bytes(width=2, height=2, bands=1)
        # Sequence: commercial process (uses cached bad token, gets 401,
        # cache is cleared), cdse token fetch, cdse process → tiff.
        responses = [
            _fake_error_response(401),
            _fake_ok_response(_fake_token_response("cdse-tok")),
            _fake_ok_response(tiff_bytes),
        ]
        with patch("src.sh_downloader.requests.post", side_effect=responses):
            sh.download_scene(
                bbox=(-0.1, 37.0, -0.05, 37.05),
                date_range=("2024-09-01", "2024-09-30"),
                bands=["B02"],
                output_path=tmp_path / "regression.tif",
                endpoint="auto",
            )

        # The bad token must have been cleared during the fallback so the next
        # call re-fetches instead of re-using the known-bad token.
        assert sh._token_cache["commercial"].token is None
        assert sh._token_cache["commercial"].expires_at == 0.0

    def test_next_call_refetches_cleared_token(self, monkeypatch, tmp_path):
        """After the token cache is cleared by fallback, the NEXT call must
        re-fetch the token rather than reusing the bad one."""
        monkeypatch.setenv("SH_CLIENT_ID", "x")
        monkeypatch.setenv("SH_CLIENT_SECRET", "y")
        sh._token_cache["commercial"].token      = "bad-cached-token"
        sh._token_cache["commercial"].expires_at = 9_999_999_999

        tiff_bytes = _fake_tiff_bytes(width=2, height=2, bands=1)
        # Second call should: (1) commercial token fetch (since cache was
        # cleared), (2) commercial process → ok.
        responses = [
            _fake_ok_response(_fake_token_response("fresh-c-tok")),
            _fake_ok_response(tiff_bytes),
        ]
        with patch("src.sh_downloader.requests.post", side_effect=responses) as mock_post:
            # Simulate the cache having been cleared by a previous fallback.
            sh._token_cache["commercial"].token      = None
            sh._token_cache["commercial"].expires_at = 0.0
            sh.download_scene(
                bbox=(-0.1, 37.0, -0.05, 37.05),
                date_range=("2024-09-01", "2024-09-30"),
                bands=["B02"],
                output_path=tmp_path / "refetch.tif",
                endpoint="commercial",
            )
        # First call should be to the token URL.
        assert "auth/realms/main/protocol/openid-connect/token" in \
            mock_post.call_args_list[0].args[0]
        # Second call should be to the process URL.
        assert mock_post.call_args_list[1].args[0] == \
            "https://services.sentinel-hub.com/api/v1/process"

    def test_search_scenes_clears_token_on_catalog_failure(self, monkeypatch):
        """search_scenes must invalidate the token for an endpoint that returns
        a 4xx so subsequent calls re-fetch rather than replaying a bad token."""
        monkeypatch.setenv("SH_CLIENT_ID", "x")
        monkeypatch.setenv("SH_CLIENT_SECRET", "y")
        sh._token_cache["commercial"].token      = "bad-catalog-tok"
        sh._token_cache["commercial"].expires_at = 9_999_999_999

        responses = [
            _fake_error_response(403, "catalog forbidden"),
        ]
        with patch("src.sh_downloader.requests.post", side_effect=responses):
            sh.search_scenes(
                bbox=(-0.1, 37.0, -0.05, 37.05),
                date_range=("2024-09-01", "2024-09-30"),
                endpoint="commercial",
            )

        cache = sh._token_cache["commercial"]
        assert cache.token is None
        assert cache.expires_at == 0.0

    def test_successful_endpoint_token_is_not_cleared(self, monkeypatch, tmp_path):
        """A successful response on the first endpoint must NOT touch the
        token cache — no reason to invalidate a working token."""
        monkeypatch.setenv("SH_CLIENT_ID", "x")
        monkeypatch.setenv("SH_CLIENT_SECRET", "y")
        sh._token_cache["commercial"].token      = "good-tok"
        sh._token_cache["commercial"].expires_at = 9_999_999_999

        tiff_bytes = _fake_tiff_bytes(width=2, height=2, bands=1)
        responses = [_fake_ok_response(tiff_bytes)]   # no token fetch needed
        with patch("src.sh_downloader.requests.post", side_effect=responses):
            sh.download_scene(
                bbox=(-0.1, 37.0, -0.05, 37.05),
                date_range=("2024-09-01", "2024-09-30"),
                bands=["B02"],
                output_path=tmp_path / "ok.tif",
                endpoint="commercial",
            )

        # The working token must be untouched.
        assert sh._token_cache["commercial"].token == "good-tok"
        assert sh._token_cache["commercial"].expires_at == 9_999_999_999


# ─────────────────────────────────────────────────────────────────────────────
# Imports — make sure the public API surface is intact
# ─────────────────────────────────────────────────────────────────────────────

def test_public_api_exports():
    """The functions documented in the module docstring must remain importable."""
    from src.sh_downloader import (
        S2_BANDS_L2A,
        download_patch,
        download_scene,
        get_stats,
        search_scenes,
    )
    assert callable(download_scene)
    assert callable(download_patch)
    assert callable(search_scenes)
    assert callable(get_stats)
    assert isinstance(S2_BANDS_L2A, list) and "B02" in S2_BANDS_L2A
