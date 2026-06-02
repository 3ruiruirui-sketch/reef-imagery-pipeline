import os
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from scripts.reef_bathy_module import compute_s2_depth_inversion
from src.stumpf_emodnet_calibration import validate_reprojected_emodnet


def write_test_tif(path, arr, crs="EPSG:32629", transform=None, nodata=-9999.0):
    if transform is None:
        transform = Affine.translation(500000.0, 4100000.0) * Affine.scale(10.0, -10.0)
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "count": 1,
        "width": arr.shape[1],
        "height": arr.shape[0],
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
        "compress": "lzw",
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr.astype(np.float32), 1)


def test_validate_reprojected_emodnet_rejects_empty(tmp_path):
    empty_raster = tmp_path / "emodnet_empty.tif"
    arr = np.full((8, 8), np.nan, dtype=np.float32)
    write_test_tif(empty_raster, arr)

    with pytest.raises(ValueError, match="few valid pixels"):
        validate_reprojected_emodnet(str(empty_raster), min_valid_pixels=1)


def test_compute_s2_depth_inversion_uses_emodnet_calibration(tmp_path):
    b02_path = tmp_path / "S2_B02_test.tif"
    b03_path = tmp_path / "S2_B03_test.tif"
    emodnet_path = tmp_path / "bathy_emodnet_test.tif"

    b02 = np.full((12, 12), 0.07, dtype=np.float32)
    b03 = np.full((12, 12), 0.03, dtype=np.float32)
    emodnet = np.linspace(-6.0, -18.0, num=144, dtype=np.float32).reshape(12, 12)

    write_test_tif(b02_path, b02)
    write_test_tif(b03_path, b03)
    write_test_tif(emodnet_path, emodnet)

    output_path = compute_s2_depth_inversion(
        str(b02_path),
        str(b03_path),
        str(tmp_path),
        date_str="test",
    )

    assert output_path is not None
    assert os.path.exists(output_path)
    assert output_path.endswith("bathy_s2_stumpf_test.tif")

    with rasterio.open(output_path) as src:
        depth_arr = src.read(1)

    assert np.isfinite(depth_arr).any(), "Output depth raster should contain valid depth values"


def test_compute_s2_depth_inversion_falls_back_with_invalid_emodnet(tmp_path):
    b02_path = tmp_path / "S2_B02_test.tif"
    b03_path = tmp_path / "S2_B03_test.tif"
    emodnet_path = tmp_path / "bathy_emodnet_test.tif"

    b02 = np.full((12, 12), 0.07, dtype=np.float32)
    b03 = np.full((12, 12), 0.03, dtype=np.float32)
    emodnet = np.full((12, 12), np.nan, dtype=np.float32)
    emodnet[0, 0] = -10.0

    write_test_tif(b02_path, b02)
    write_test_tif(b03_path, b03)
    write_test_tif(emodnet_path, emodnet)

    output_path = compute_s2_depth_inversion(
        str(b02_path),
        str(b03_path),
        str(tmp_path),
        date_str="test",
    )

    assert output_path is not None
    assert os.path.exists(output_path)
    assert output_path.endswith("bathy_s2_stumpf_test.tif")

    with rasterio.open(output_path) as src:
        depth_arr = src.read(1)

    assert np.isfinite(depth_arr).any(), "Fallback output depth raster should contain valid depth values"
