#!/usr/bin/env python3
"""
hybrid_stac_physical_orchestrator.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Extracao VSI streaming das bandas B02/B03 para validar as 15 melhores datas 
previstas pela heuristica global nos ultimos 8 anos.
Sem downloads pesados! Lê apenas um BBox de 1km x 1km sobre as coordenadas.
"""

import argparse
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import cv2
import rasterio
from rasterio.windows import from_bounds
from pyproj import Transformer
import planetary_computer as pc
from pystac_client import Client
import warnings

# Suppress rasterio warnings about block sizes
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

# Re-use your physics logic
from src.reef_ml_predictor import calculate_physics_score
from src.reef_ml_predictor_acolite import make_snr_map, estimate_kd_bandratio
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("HybridVSI")

def fetch_stac_heuristic_shortlist(lat, lon, years=8, top_k=15):
    log.info(f"Fase 1: Pesquisando STAC para os ultimos {years} anos...")
    catalog = Client.open('https://planetarycomputer.microsoft.com/api/stac/v1', modifier=pc.sign_inplace)
    end_date = datetime.now()
    start_date = datetime(end_date.year - years, 1, 1)
    
    search = catalog.search(
        collections=['sentinel-2-l2a'],
        intersects={'type': 'Point', 'coordinates': [lon, lat]},
        datetime=f"{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}",
        query={"eo:cloud_cover": {"lt": 10}}  # Relaxed to 10% to catch coastal clears
    )
    
    items = list(search.items())
    log.info(f"Encontradas {len(items)} passagens com <10% nuvens globais (T29SNB).")
    
    data = []
    for item in items:
        props = item.properties
        # Filter high nodata (tile edge issues)
        if props.get('s2:nodata_pixel_percentage', 100) > 20: continue
        
        # Build a record compatible with our heuristic
        rec = {
            'date_str': item.datetime.strftime('%Y-%m-%d'),
            'datetime': item.datetime,
            'cloud_cover': props.get('eo:cloud_cover', 100),
            'sun_elevation': props.get('view:sun_elevation', 45),
            'item': item
        }
        data.append(rec)
        
    df = pd.DataFrame(data)
    if df.empty:
        log.error("Nenhuma cena viavel.")
        return []
        
    # Apply heuristic (using depth=16m)
    df['heuristic_score'] = df.apply(lambda r: calculate_physics_score(r, 16.0), axis=1)
    
    # Sort and take top_k unique dates
    df = df.sort_values('heuristic_score', ascending=False).drop_duplicates('date_str')
    shortlist = df.head(top_k)
    log.info(f"Shortlist heuristica pronta ({len(shortlist)} imagens).")
    return shortlist

def process_vsi_pixel_window(item, lat, lon):
    b02_href = item.assets["B02"].href
    b03_href = item.assets["B03"].href
    b08_href = item.assets["B08"].href  # NIR para glint
    
    # 1 km buffer => ~ 500m cada lado => raio ~ 500m
    BUFFER_M = 500.0
    
    env = rasterio.Env(AWS_NO_SIGN_REQUEST='YES', GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR')
    
    with env:
        try:
            with rasterio.open(b02_href) as src:
                tf = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
                x, y = tf.transform(lon, lat)
                window = from_bounds(x - BUFFER_M, y - BUFFER_M, x + BUFFER_M, y + BUFFER_M, src.transform)
                
                b02_arr = src.read(1, window=window).astype(np.float32)
                
            with rasterio.open(b03_href) as src:
                b03_arr = src.read(1, window=window).astype(np.float32)
                
            with rasterio.open(b08_href) as src:
                b08_arr = src.read(1, window=window).astype(np.float32)
                
        except Exception as e:
            log.warning(f"VSI read failed for {item.datetime}: {e}")
            return None
            
    # L2A DN -> Reflectance (factor 10000)
    b02_ref = np.clip(b02_arr / 10000.0, 0, 1.5)
    b03_ref = np.clip(b03_arr / 10000.0, 0, 1.5)
    b08_ref = np.clip(b08_arr / 10000.0, 0, 1.5)
    
    # Check if empty (nodata)
    if b02_ref.max() == 0 or np.all(np.isnan(b02_ref)):
        return None
        
    # Check clouds locally (B02 > 0.15 is likely cloud or extreme sunglint)
    # If the window is more than 50% clouds, skip
    cloud_mask = b02_ref > 0.15
    if cloud_mask.mean() > 0.5:
        return None
        
    # Simulate empirical sunglint correction (subtract 5% of 95th percentile)
    p95_b02 = np.percentile(b02_ref[b02_ref > 0], 95) if np.any(b02_ref > 0) else 0.0
    p95_b03 = np.percentile(b03_ref[b03_ref > 0], 95) if np.any(b03_ref > 0) else 0.0
    
    b02_corr = np.clip(b02_ref - 0.8 * p95_b02 * 0.05, 0, 1.0)
    b03_corr = np.clip(b03_ref - 0.8 * p95_b03 * 0.05, 0, 1.0)
    
    # Local SNR using make_snr_map (homogeneity - BAD for reef edges!)
    snr_map = make_snr_map(b02_corr, window=5)
    
    # BENTHIC EDGE CONTRAST (Sobel/Laplacian) - GOOD for reef edges!
    # 1. Subtração empírica total da superfície (NIR) para remover variância de ondas
    b02_puro = np.clip(b02_corr - b08_ref, 0, 1.0)
    
    # 2. Deteção de Arestas Geológicas (Laplacian of Gaussian)
    # Primeiro aplicamos um Blur para destruir o ruído de sensor (que engana o Laplacian).
    # O recife verdadeiro sobrevive ao Blur porque é uma estrutura macro.
    b02_blur = cv2.GaussianBlur(b02_puro, (5, 5), 0)
    laplacian = cv2.Laplacian(b02_blur, cv2.CV_32F)
    
    edge_var = float(np.var(laplacian)) * 1000000
    benthic_contrast = edge_var * float(np.nanmean(snr_map))
    
    # 3. Spatial frequency partition using 2D FFT (Calmness Index)
    f_transform = np.fft.fft2(b02_corr)
    f_shift = np.fft.fftshift(f_transform)
    power = np.abs(f_shift) ** 2
    h, w = b02_corr.shape
    cy, cx = h // 2, w // 2
    
    # Low frequency mask (macro geological structures)
    r_low = 5
    y, x = np.ogrid[:h, :w]
    mask_low = (x - cx)**2 + (y - cy)**2 <= r_low**2
    # High frequency mask (surface waves, ripples, glint)
    r_high = 15
    mask_high = (x - cx)**2 + (y - cy)**2 >= r_high**2
    
    low_power = np.sum(power[mask_low])
    high_power = np.sum(power[mask_high])
    fft_cleanliness = float(low_power / (high_power + 1e-12))
    
    # Smooth macro-scale image for Sobel/Contour calculation
    macro = cv2.GaussianBlur(b02_corr, (9, 9), 0)
    
    # 4. Geological Edge-Entropy (Entropy Gate)
    # Compute Sobel gradients to find structural contours
    sobelx = cv2.Sobel(macro, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(macro, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobelx**2 + sobely**2)
    
    if grad_mag.max() > grad_mag.min():
        grad_norm = ((grad_mag - grad_mag.min()) / (grad_mag.max() - grad_mag.min()) * 255).astype(np.uint8)
        hist = cv2.calcHist([grad_norm], [0], None, [256], [0, 256]).ravel()
        p = hist / np.sum(hist)
        p = p[p > 0]
        edge_entropy = float(-np.sum(p * np.log2(p)))
    else:
        edge_entropy = 0.0
    
    # Kd ratio
    kd_est, _ = estimate_kd_bandratio(b02_corr, b03_corr, 0.045)
    
    return {
        'snr_mean': float(np.nanmean(snr_map)),
        'benthic_contrast': benthic_contrast,
        'cleanliness': fft_cleanliness,
        'edge_entropy': edge_entropy,
        'kd_mean': float(kd_est),
        'raw_mean': float(np.mean(b02_ref)),
        'b02_mean_ref': float(np.nanmean(b02_corr)),
        'b03_mean_ref': float(np.nanmean(b03_corr))
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lat", type=float, default=37.05811)
    parser.add_argument("--lon", type=float, default=-8.20978)
    parser.add_argument("--years", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=15, help="Qtd imagens na shortlist para processar VSI")
    args = parser.parse_args()
    
    shortlist = fetch_stac_heuristic_shortlist(args.lat, args.lon, args.years, args.top_k)
    
    results = []
    
    log.info("Fase 2: VSI Streaming & Calculo Fisico Bêntico nos pìxeis reais...")
    for idx, row in shortlist.iterrows():
        item = row['item']
        date_str = row['date_str']
        sys.stdout.write(f"\r  Processando {date_str} via VSI... ")
        sys.stdout.flush()
        
        phys = process_vsi_pixel_window(item, args.lat, args.lon)
        if phys is not None:
            results.append({
                'date': date_str,
                'cloud_meta': row['cloud_cover'],
                'heuristic': row['heuristic_score'],
                'real_snr': phys['snr_mean'],
                'benthic_contrast': phys['benthic_contrast'],
                'cleanliness': phys['cleanliness'],
                'edge_entropy': phys['edge_entropy'],
                'kd_mean': phys['kd_mean'],
                'raw_mean': phys['raw_mean']
            })
            
    print("\n")
    if not results:
        log.error("Nenhuma imagem pode ser validada via VSI.")
        return
        
    df_res = pd.DataFrame(results)
    
    # Calculate senior visual-geophysical score:
    # 1. log10(cleanliness) * edge_entropy * (0.045 / kd_mean)
    # 2. Calmness Penalty: multiply by 0.1 if cleanliness < 5000 (rough/wavy water)
    # 3. Signal Penalty: multiply by 0.1 if raw_mean is not in [0.06, 0.15]
    df_res['score'] = np.log10(df_res['cleanliness']) * df_res['edge_entropy'] * (0.045 / df_res['kd_mean'])
    df_res.loc[df_res['cleanliness'] < 5000, 'score'] *= 0.1
    df_res.loc[(df_res['raw_mean'] < 0.06) | (df_res['raw_mean'] > 0.15), 'score'] *= 0.1
    
    # Sort by the final composite Score
    df_res = df_res.sort_values('score', ascending=False).reset_index(drop=True)
    
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"🏆 RESULTADOS FINAIS - BENTHIC VISIBILITY INDEX (ÚLTIMOS {args.years} ANOS) 🏆")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    for i, r in df_res.iterrows():
        if i >= 10: break  # print top 10
        print(f"{i+1:2}. {r['date']} | Score: {r['score']:.3f} | Calmness (Clean): {r['cleanliness']:8.1f} | EdgeEntropy: {r['edge_entropy']:.3f} | Kd: {r['kd_mean']:.4f} | SNR: {r['real_snr']:.2f}")
        print(f"   ↳ Nuvens STAC: {r['cloud_meta']:.2f}% | Refletância Média: {r['raw_mean']:.3f} | Heurística Antiga: {r['heuristic']:.1f}")

        
if __name__ == "__main__":
    main()
