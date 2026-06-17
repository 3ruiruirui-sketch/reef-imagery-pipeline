"""
Integration tests: build_coastal_geojson artefact ↔ /api/coastal-features endpoint.

Validates that the GeoJSON produced by build_coastal_geojson() is consumed
correctly by the Flask dashboard endpoint — no network, no satellite access.

The tests use Flask's built-in test client so the endpoint code runs for real
without requiring a live server.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ─── fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def flask_client():
    """Return a Flask test client for dashboard/app.py."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("dashboard_app", ROOT / "dashboard" / "app.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    mod.app.config["TESTING"] = True
    with mod.app.test_client() as client:
        yield client


@pytest.fixture(scope="module")
def production_geojson() -> dict:
    """Load the production GeoJSON artefact."""
    path = ROOT / "outputs" / "coastal_topography" / "algarve_coastal_features.geojson"
    if not path.exists():
        pytest.skip("Production GeoJSON not found — run generate_coastal_features_dgt.py first")
    with open(path) as fh:
        return json.load(fh)


# ─── artefact contract tests ─────────────────────────────────────────────────

class TestProductionGeojsonContract:
    """The GeoJSON artefact on disk satisfies the dashboard's structural contract."""

    def test_is_feature_collection(self, production_geojson):
        assert production_geojson["type"] == "FeatureCollection"

    def test_crs_is_crs84(self, production_geojson):
        assert production_geojson["crs"]["properties"]["name"] == "urn:ogc:def:crs:OGC:1.3:CRS84"

    def test_has_fifteen_algarve_sites(self, production_geojson):
        assert len(production_geojson["features"]) == 15

    def test_all_features_are_points(self, production_geojson):
        for feat in production_geojson["features"]:
            assert feat["geometry"]["type"] == "Point"
            assert len(feat["geometry"]["coordinates"]) == 2

    def test_coordinates_are_lon_lat_order(self, production_geojson):
        """GeoJSON spec: coordinates are [longitude, latitude]."""
        for feat in production_geojson["features"]:
            lon, lat = feat["geometry"]["coordinates"]
            props = feat["properties"]
            assert lon == pytest.approx(float(props["longitude"]), abs=1e-6)
            assert lat == pytest.approx(float(props["latitude"]), abs=1e-6)

    def test_required_slope_aspect_keys(self, production_geojson):
        required = {
            "site_name", "latitude", "longitude", "buffer_m",
            "slope_min", "slope_max", "slope_mean", "slope_std",
            "slope_median", "slope_percentile_90",
            "aspect_min", "aspect_max", "aspect_mean", "aspect_std", "aspect_median",
        }
        for feat in production_geojson["features"]:
            assert required.issubset(feat["properties"].keys()), feat["properties"]["site_name"]

    def test_no_nan_values(self, production_geojson):
        """All numeric properties must be finite (no NaN from offshore sites)."""
        for feat in production_geojson["features"]:
            props = feat["properties"]
            for k, v in props.items():
                if isinstance(v, float):
                    assert v == v, f"NaN in {props['site_name']}.{k}"  # NaN != NaN

    def test_slope_mean_positive_for_all_sites(self, production_geojson):
        for feat in production_geojson["features"]:
            props = feat["properties"]
            assert props["slope_mean"] > 0.0, f"Zero slope at {props['site_name']}"

    def test_armacao_de_pera_not_all_zeros(self, production_geojson):
        """Regression: armacao_de_pera was all-zeros before snap-to-coast was added."""
        site = next(
            f["properties"] for f in production_geojson["features"]
            if f["properties"]["site_name"] == "armacao_de_pera"
        )
        assert site["slope_mean"] > 1.0, "armacao_de_pera snap-to-coast regression"


# ─── endpoint compatibility tests ────────────────────────────────────────────

class TestCoastalFeaturesEndpoint:
    """The /api/coastal-features endpoint serves the artefact correctly."""

    def test_endpoint_returns_200(self, flask_client):
        resp = flask_client.get("/api/coastal-features")
        assert resp.status_code == 200

    def test_response_is_feature_collection(self, flask_client):
        resp = flask_client.get("/api/coastal-features")
        data = json.loads(resp.data)
        assert data["type"] == "FeatureCollection"
        assert "features" in data

    def test_response_has_fifteen_features(self, flask_client):
        resp = flask_client.get("/api/coastal-features")
        data = json.loads(resp.data)
        assert len(data["features"]) == 15

    def test_site_filter_returns_one_match(self, flask_client):
        resp = flask_client.get("/api/coastal-features?site=armacao_de_pera")
        data = json.loads(resp.data)
        assert len(data["features"]) == 1
        assert data["features"][0]["properties"]["site_name"] == "armacao_de_pera"

    def test_site_filter_unknown_returns_empty(self, flask_client):
        resp = flask_client.get("/api/coastal-features?site=does_not_exist")
        data = json.loads(resp.data)
        assert data["features"] == []

    def test_endpoint_slope_mean_nonzero(self, flask_client):
        resp = flask_client.get("/api/coastal-features")
        data = json.loads(resp.data)
        for feat in data["features"]:
            assert feat["properties"]["slope_mean"] > 0.0, feat["properties"]["site_name"]
