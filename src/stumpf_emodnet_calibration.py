import numpy as np
import rasterio
from src.constants import STUMPF_LOG_EPSILON
from rasterio.warp import calculate_default_transform, reproject
from rasterio.enums import Resampling
from sklearn.linear_model import HuberRegressor
import logging

log = logging.getLogger(__name__)

def reproject_emodnet_to_s2(emodnet_path, s2_reference_path, output_emodnet_10m):
    """
    Reprojects and resamples the EMODnet bathymetry to exactly match
    the grid, CRS, and 10m resolution of the Sentinel-2 reference band.
    """
    log.info("Lendo imagem Sentinel-2 de referência (10m)...")
    with rasterio.open(s2_reference_path) as dst_ref:
        dst_crs = dst_ref.crs
        dst_transform = dst_ref.transform
        dst_width = dst_ref.width
        dst_height = dst_ref.height
        dst_profile = dst_ref.profile
        
    log.info(f"Reamostrando EMODnet para alinhar com Sentinel-2 (CRS: {dst_crs})...")
    with rasterio.open(emodnet_path) as src:
        # Atualizar o profile de destino com base no Sentinel-2
        dst_profile.update({
            'dtype': rasterio.float32,
            'count': 1,
            'nodata': np.nan if src.nodata is None else src.nodata
        })

        with rasterio.open(output_emodnet_10m, 'w', **dst_profile) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear
            )
    log.info(f"EMODnet reamostrado guardado em: {output_emodnet_10m}")
    return output_emodnet_10m


def validate_reprojected_emodnet(emodnet_path, min_valid_pixels: int = 500):
    """Validate the reprojected EMODnet raster before calibration."""
    with rasterio.open(emodnet_path) as src:
        arr = src.read(1)
    valid_pixels = int(np.count_nonzero(np.isfinite(arr)))
    if valid_pixels < min_valid_pixels:
        raise ValueError(
            f"Reprojected EMODnet raster has too few valid pixels: {valid_pixels} < {min_valid_pixels}"
        )
    # EMODnet bathymetry uses negative values for depth below sea level.
    # Check sign convention on marine pixels only (< 5 m elevation) to avoid
    # false positives on coastal tiles that include significant land area.
    valid_arr = arr[np.isfinite(arr)]
    marine_arr = valid_arr[valid_arr < 5.0]  # exclude obvious land (> 5 m)
    if marine_arr.size > 0 and np.all(marine_arr >= 0.0):
        raise ValueError(
            f"EMODnet depth raster has unexpected sign convention: "
            f"all {marine_arr.size} marine pixels are non-negative "
            f"(expected negative = below sea level). "
            f"Check that the raster was not sign-flipped before ingestion."
        )
    return True


def stumpf_log_ratio(b_blue, b_green, n=1000):
    """Calcula o rácio logarítmico entre as bandas Azul e Verde."""
    b_blue = np.clip(b_blue, STUMPF_LOG_EPSILON, None)
    b_green = np.clip(b_green, STUMPF_LOG_EPSILON, None)
    return np.log(n * b_blue) / np.log(n * b_green)

def calibrate_stumpf_vs_emodnet(
    s2_blue_path,
    s2_green_path,
    emodnet_10m_path,
    output_path,
    cmems_10m_path=None,
    depth_min_m: float = 5.0,
    depth_max_m: float = 20.0,
):
    """
    Calibrate Stumpf SDB using a blended depth prior (EMODnet + optional CMEMS SDB).

    When *cmems_10m_path* is provided the two priors are blended before regression
    (EMODnet 60 %, CMEMS 40 %).  The training window is automatically extended to
    depth_max_m=34 m when CMEMS Wave-Kinematics data is present, since it reaches
    deeper than the EMODnet-only 20 m default.

    Both rasters must be on the same 10 m S2 grid (negative = below sea level).
    """
    log.info("Loading rasters for Stumpf calibration...")
    with rasterio.open(s2_blue_path) as src:
        b_blue = src.read(1)
        profile = src.profile

    with rasterio.open(s2_green_path) as src:
        b_green = src.read(1)

    with rasterio.open(emodnet_10m_path) as src:
        emodnet_prior = src.read(1)

    # Optionally blend in CMEMS SDB
    cmems_prior = None
    if cmems_10m_path is not None:
        try:
            with rasterio.open(cmems_10m_path) as src:
                cmems_prior = src.read(1)
            log.info("CMEMS SDB prior loaded (%d valid pixels)",
                     int(np.isfinite(cmems_prior).sum()))
        except Exception as exc:
            log.warning("Could not load CMEMS prior — using EMODnet only: %s", exc)

    try:
        from src.cmems_sdb import blend_depth_priors
        depth_prior = blend_depth_priors(emodnet_prior, cmems_prior)
    except ImportError:
        depth_prior = emodnet_prior.copy()

    prior_source = "EMODnet+CMEMS" if cmems_prior is not None else "EMODnet"
    log.info("Depth prior source: %s", prior_source)

    # Calculate Stumpf ratio
    log.info("Calculating Stumpf log-ratio...")
    X_ratio = stumpf_log_ratio(b_blue, b_green)

    valid_mask = (
        (depth_prior <= -depth_min_m) &
        (depth_prior >= -depth_max_m) &
        np.isfinite(X_ratio) &
        np.isfinite(depth_prior)
    )

    X_train = X_ratio[valid_mask].reshape(-1, 1)
    y_train = depth_prior[valid_mask]

    if len(X_train) < 100:
        raise ValueError(
            f"Insufficient training points ({len(X_train)}) for calibration. "
            "Check cloud mask or bbox coverage."
        )

    log.info("Training robust regression model with %d samples (%s prior)...",
             len(X_train), prior_source)
    model = HuberRegressor()
    model.fit(X_train, y_train)

    m1 = model.coef_[0]
    m0 = model.intercept_
    log.info("Calibration: m0=%.3f, m1=%.3f", m0, m1)

    sdb_depth = (m1 * X_ratio) + m0

    # Mask: only output where prior is valid and within depth window
    invalid_mask = (
        ~np.isfinite(depth_prior) |
        (depth_prior >= -1.0) |
        (depth_prior < -depth_max_m)
    )
    sdb_depth = np.where(invalid_mask, np.nan, sdb_depth)

    log.info("Writing calibrated SDB depth to %s", output_path)
    profile.update(dtype=rasterio.float32, count=1, nodata=np.nan)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(sdb_depth.astype(rasterio.float32), 1)

    y_pred = model.predict(X_train)
    rmse = float(np.sqrt(np.mean((y_train - y_pred) ** 2)))
    log.info("Calibration RMSE: %.3f m", rmse)

    return {
        "m0": m0,
        "m1": m1,
        "rmse": rmse,
        "calibration_samples": int(len(X_train)),
        "depth_prior_source": prior_source,
        "depth_training_window_m": [depth_min_m, depth_max_m],
    }

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Exemplo de chamada:
    # calibrate_stumpf_vs_emodnet("b02.tif", "b03.tif", "emodnet_10m.tif", "sdb_stumpf_calibrated.tif")
