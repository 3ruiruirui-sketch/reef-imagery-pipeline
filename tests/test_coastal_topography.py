"""Tests for src/coastal_topography.py — CoastalTopographyAnalyzer."""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.coastal_topography import CoastalTopographyAnalyzer


ALGARVE_BBOX = (-8.25, 37.04, -8.17, 37.10)


# ── Tile naming ────────────────────────────────────────────────────────────────

class TestGlo30TileName:
    def test_northern_western(self):
        name = CoastalTopographyAnalyzer._glo30_tile_name(37, -9)
        assert name == "Copernicus_DSM_COG_10_N37_00_W009_00_DEM"

    def test_southern_eastern(self):
        name = CoastalTopographyAnalyzer._glo30_tile_name(-5, 12)
        assert name == "Copernicus_DSM_COG_10_S05_00_E012_00_DEM"

    def test_equator_prime_meridian(self):
        name = CoastalTopographyAnalyzer._glo30_tile_name(0, 0)
        assert name == "Copernicus_DSM_COG_10_N00_00_E000_00_DEM"

    def test_algarve_tile(self):
        name = CoastalTopographyAnalyzer._glo30_tile_name(37, -8)
        assert "N37" in name and "W008" in name


class TestGlo30TilesForBbox:
    def test_single_tile(self):
        # bbox fully within one 1°×1° tile
        tiles = CoastalTopographyAnalyzer._glo30_tiles_for_bbox((-8.25, 37.04, -8.17, 37.10))
        assert tiles == [(37, -9)]

    def test_two_lon_tiles(self):
        tiles = CoastalTopographyAnalyzer._glo30_tiles_for_bbox((-8.5, 37.0, -7.5, 37.9))
        lons = {lon for _, lon in tiles}
        assert -9 in lons and -8 in lons

    def test_four_tiles_crossing_boundaries(self):
        tiles = CoastalTopographyAnalyzer._glo30_tiles_for_bbox((-8.5, 36.5, -7.5, 37.5))
        assert len(tiles) == 4

    def test_output_is_list_of_tuples(self):
        tiles = CoastalTopographyAnalyzer._glo30_tiles_for_bbox(ALGARVE_BBOX)
        assert isinstance(tiles, list)
        for t in tiles:
            assert isinstance(t, tuple) and len(t) == 2


# ── Initialisation ─────────────────────────────────────────────────────────────

class TestInit:
    def test_default_dem_source(self, tmp_path):
        a = CoastalTopographyAnalyzer(ALGARVE_BBOX, str(tmp_path))
        assert a.dem_source == "auto"

    def test_explicit_dem_source(self, tmp_path):
        a = CoastalTopographyAnalyzer(ALGARVE_BBOX, str(tmp_path), dem_source="srtm")
        assert a.dem_source == "srtm"

    def test_output_dirs_created(self, tmp_path):
        out = tmp_path / "costal_out"
        CoastalTopographyAnalyzer(ALGARVE_BBOX, str(out))
        assert out.exists()
        assert (out / "mdt_tiles").exists()

    def test_bbox_stored(self, tmp_path):
        a = CoastalTopographyAnalyzer(ALGARVE_BBOX, str(tmp_path))
        assert a.bbox == ALGARVE_BBOX


# ── STAC fetch (mocked) ────────────────────────────────────────────────────────

class TestFetchStacItems:
    def test_returns_list(self, tmp_path):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "features": [{"id": "tile-1", "assets": {"Data": {"href": "http://x/t.tif"}},
                          "geometry": {"type": "Polygon", "coordinates": [[[-8.2,37.0],[-8.1,37.0],[-8.1,37.1],[-8.2,37.1],[-8.2,37.0]]]}}],
            "context": {"returned": 1},
        }
        mock_resp.raise_for_status = MagicMock()
        with patch("requests.get", return_value=mock_resp):
            a = CoastalTopographyAnalyzer(ALGARVE_BBOX, str(tmp_path))
            items = a.fetch_stac_items(limit=1)
        assert len(items) == 1
        assert items[0]["id"] == "tile-1"

    def test_network_error_returns_empty(self, tmp_path):
        import requests as _req
        with patch("requests.get", side_effect=_req.RequestException("timeout")):
            a = CoastalTopographyAnalyzer(ALGARVE_BBOX, str(tmp_path))
            items = a.fetch_stac_items()
        assert items == []


# ── Credentials (mocked) ───────────────────────────────────────────────────────

class TestReadCdseCredentials:
    def test_reads_username_password(self, tmp_path):
        import base64
        cred_content = base64.b64encode(b"username = test@example.com\npassword = secret123\n").decode()
        cred_file = tmp_path / ".copernicusmarine-credentials"
        cred_file.write_text(cred_content)
        with patch.object(CoastalTopographyAnalyzer, "CDSE_CRED_FILE", cred_file):
            user, pwd = CoastalTopographyAnalyzer._read_cdse_credentials()
        assert user == "test@example.com"
        assert pwd == "secret123"

    def test_missing_file_raises(self, tmp_path):
        with patch.object(CoastalTopographyAnalyzer, "CDSE_CRED_FILE", tmp_path / "no_file"):
            with pytest.raises(FileNotFoundError):
                CoastalTopographyAnalyzer._read_cdse_credentials()


# ── Production artefact check ──────────────────────────────────────────────────

class TestProductionOutputs:
    """Verify the 15-site production run produced valid artefacts."""

    FEATURES_CSV = Path("outputs/coastal_topography/algarve_coastal_features.csv")
    FEATURES_GJ  = Path("outputs/coastal_topography/algarve_coastal_features.geojson")
    DEM_TIF      = Path("outputs/coastal_topography/dem_mosaic_50cm.tif")

    def test_csv_exists(self):
        if not self.FEATURES_CSV.exists():
            pytest.skip("Production run not yet executed")
        assert self.FEATURES_CSV.stat().st_size > 0

    def test_csv_has_15_sites(self):
        if not self.FEATURES_CSV.exists():
            pytest.skip("Production run not yet executed")
        import pandas as pd
        df = pd.read_csv(self.FEATURES_CSV)
        assert len(df) == 15

    def test_csv_slope_values_in_range(self):
        if not self.FEATURES_CSV.exists():
            pytest.skip("Production run not yet executed")
        import pandas as pd
        df = pd.read_csv(self.FEATURES_CSV)
        assert (df["slope_mean"] >= 0.0).all()
        assert (df["slope_mean"] <= 45.0).all()

    def test_csv_aspect_values_in_range(self):
        if not self.FEATURES_CSV.exists():
            pytest.skip("Production run not yet executed")
        import pandas as pd
        df = pd.read_csv(self.FEATURES_CSV)
        assert (df["aspect_mean"] >= 0.0).all()
        assert (df["aspect_mean"] <= 360.0).all()

    def test_geojson_exists_and_valid(self):
        if not self.FEATURES_GJ.exists():
            pytest.skip("Production run not yet executed")
        import json
        with open(self.FEATURES_GJ) as f:
            data = json.load(f)
        assert data["type"] == "FeatureCollection"
        assert len(data["features"]) == 15

    def test_dem_mosaic_exists(self):
        if not self.DEM_TIF.exists():
            pytest.skip("Production run not yet executed")
        import rasterio
        with rasterio.open(str(self.DEM_TIF)) as src:
            assert src.crs.to_epsg() == 3763
            assert src.count == 1
