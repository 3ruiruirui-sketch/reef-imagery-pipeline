import os
import sys
import logging
import rasterio
from rasterio.enums import Resampling

# Adicionar a raiz do projeto ao path para conseguir importar do src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.stumpf_emodnet_calibration import calibrate_stumpf_vs_emodnet

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
log = logging.getLogger(__name__)

def reproject_emodnet_to_s2(emodnet_path, s2_reference_path, output_emodnet_10m):
    """
    Reprojects and resamples the EMODnet bathymetry to exactly match
    the grid, CRS, and 10m resolution of the Sentinel-2 reference band.
    """
    from rasterio.warp import calculate_default_transform, reproject
    
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

if __name__ == "__main__":
    import numpy as np # Import aqui para a função acima
    
    # Vamos usar as imagens existentes em outputs/santa_eulalia_multiband e outputs/santa_eulalia_12m
    # como dados de entrada para este teste isolado.
    
    S2_B02_PATH = "outputs/santa_eulalia_multiband/S2_B02_20241015.tif"
    S2_B03_PATH = "outputs/santa_eulalia_multiband/S2_B03_20241015.tif"
    EMODNET_RAW_PATH = "outputs/santa_eulalia_12m/bathy_emodnet_20260529.tif"
    
    # Ficheiros de output baseados na arquitetura
    INTERIM_EMODNET = "data/interim/emodnet_10m/test_emodnet_10m_santa_eulalia.tif"
    OUTPUT_STUMPF = "data/processed/bathy_10m_stumpf/test_sdb_stumpf_santa_eulalia.tif"
    
    if not all(os.path.exists(p) for p in [S2_B02_PATH, S2_B03_PATH, EMODNET_RAW_PATH]):
        log.error("Ficheiros de teste não encontrados. Certifica-te que estás na raiz do projeto.")
        sys.exit(1)

    log.info("=== PASSO 1: Preparar Dados (Reamostrar EMODnet) ===")
    reproject_emodnet_to_s2(EMODNET_RAW_PATH, S2_B02_PATH, INTERIM_EMODNET)
    
    log.info("\n=== PASSO 2: Executar Calibração Stumpf vs EMODnet ===")
    resultados = calibrate_stumpf_vs_emodnet(
        s2_blue_path=S2_B02_PATH,
        s2_green_path=S2_B03_PATH,
        emodnet_10m_path=INTERIM_EMODNET,
        output_path=OUTPUT_STUMPF
    )
    
    log.info("\n=== RESULTADOS DA CALIBRAÇÃO ===")
    log.info(f"Intercept (m0): {resultados['m0']:.3f}")
    log.info(f"Slope (m1):     {resultados['m1']:.3f}")
    log.info(f"RMSE (m):       {resultados['rmse']:.3f} m")
    log.info(f"Amostras:       {resultados['calibration_samples']}")
    log.info(f"\nMapa final gravado em: {OUTPUT_STUMPF}")
