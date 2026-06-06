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

def calibrate_stumpf_vs_emodnet(s2_blue_path, s2_green_path, emodnet_10m_path, output_path):
    """
    Calibrate Stumpf SDB using EMODnet depth prior as ground truth via robust regression.
    """
    log.info(f"Loading rasters for Stumpf calibration...")
    with rasterio.open(s2_blue_path) as src:
        b_blue = src.read(1)
        profile = src.profile
        
    with rasterio.open(s2_green_path) as src:
        b_green = src.read(1)
        
    with rasterio.open(emodnet_10m_path) as src:
        depth_prior = src.read(1)

    # Calculate Stumpf ratio
    log.info("Calculating Stumpf log-ratio...")
    X_ratio = stumpf_log_ratio(b_blue, b_green)
    
    # Create mask for training: depths between 5m and 20m
    # Assuming depth_prior has negative values (e.g. -15 for 15m depth)
    valid_mask = (
        (depth_prior <= -5.0) &
        (depth_prior >= -20.0) &
        np.isfinite(X_ratio) &
        np.isfinite(depth_prior)
    )

    X_train = X_ratio[valid_mask].reshape(-1, 1)
    y_train = depth_prior[valid_mask]

    if len(X_train) < 100:
         raise ValueError("Pontos insuficientes para calibração. Verifica a zona ou a máscara de nuvens.")

    # Robust Linear Regression
    log.info(f"Training robust regression model with {len(X_train)} samples...")
    model = HuberRegressor()
    model.fit(X_train, y_train)

    m1 = model.coef_[0]
    m0 = model.intercept_
    log.info(f"Calibration successful: m0={m0:.3f}, m1={m1:.3f}")

    # Apply coefficients to the whole image
    sdb_depth = (m1 * X_ratio) + m0
    
    # Safety mask: Do not calculate where EMODnet is land, too deep, or invalid.
    invalid_mask = (
        ~np.isfinite(depth_prior) |
        (depth_prior >= -1.0) |
        (depth_prior < -20.0)
    )
    sdb_depth = np.where(invalid_mask, np.nan, sdb_depth)

    # Save output
    log.info(f"Writing calibrated SDB depth to {output_path}")
    profile.update(dtype=rasterio.float32, count=1, nodata=np.nan)
    with rasterio.open(output_path, 'w', **profile) as dst:
         dst.write(sdb_depth.astype(rasterio.float32), 1)

    # Calculate RMSE
    y_pred = model.predict(X_train)
    rmse = np.sqrt(np.mean((y_train - y_pred)**2))
    
    log.info(f"Calibration RMSE: {rmse:.3f}m")

    return {
        "m0": m0,
        "m1": m1,
        "rmse": rmse,
        "calibration_samples": len(X_train)
    }

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Exemplo de chamada:
    # calibrate_stumpf_vs_emodnet("b02.tif", "b03.tif", "emodnet_10m.tif", "sdb_stumpf_calibrated.tif")
