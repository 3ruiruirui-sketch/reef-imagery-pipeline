"""
generate_enhanced_v2.py — Pipeline S2 Melhorado (Nível 1 + 2)
══════════════════════════════════════════════════════════════════
Melhorias sobre generate_green_eulalia.py:
  1. Ratio B02/B03 (Stumpf) em vez de só B02 (mais robusto)
  2. Correcção SZA por pixel (metadados da cena)
  3. Correcção de glint Hedley (2005) com B11 SWIR
  4. Filtro FFT para remover padrão de ondas
  5. Stretch percentil sobre máscara de água (evita oversaturation)
  6. Banda Green overlay para visualização natural

Uso: python scratch/generate_enhanced_v2.py
"""
import os
import sys
import math
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import rasterio
from rasterio.windows import Window
from pyproj import Transformer
import planetary_computer as pc
from pystac_client import Client
from skimage.restoration import denoise_nl_means, estimate_sigma
from numpy.fft import fft2, ifft2, fftshift
from pathlib import Path
from datetime import datetime as dt
import pandas as pd
import time

# ── Configuração ─────────────────────────────────────────────────────────────
SITES = {
    "pedra_santa_eulalia": {
        "lat": 37.068978,
        "lon": -8.210328,
        "depth_m": 14,
        "label": "Pedra de Santa Eulália",
    },
    "pedra_do_alto": {
        "lat": 37.04715,
        "lon": -8.45268,
        "depth_m": 16,
        "label": "Pedra do Alto",
    },
    "tartaruga": {
        "lat": 37.05295,
        "lon": -8.2708,
        "depth_m": 12,
        "label": "Tartaruga",
    },
}

# Escolher local e data
SITE_KEY   = "pedra_santa_eulalia"  # ← alterar conforme necessário
DATE_STR   = "2025-09-25"
BUFFER_M   = 600   # janela 1.2 km × 1.2 km
DPI        = 300   # 300 DPI para preview rápido, usar 1200 para publicação
OUT_DIR    = Path("/Users/ssoares/.gemini/antigravity-ide/brain/d6246231-ba06-4562-9142-e672e792f965")

# ── Colormaps ─────────────────────────────────────────────────────────────────
green_reef_cmap = LinearSegmentedColormap.from_list(
    "green_reef", ["#001100", "#002d08", "#004d1b", "#007a33", "#22b15b", "#6ce08c", "#baffd0"]
)
blue_depth_cmap = LinearSegmentedColormap.from_list(
    "blue_depth", ["#000814", "#001d3d", "#023e8a", "#0077b6", "#0096c7", "#48cae4", "#ade8f4"]
)

# ── Funções ───────────────────────────────────────────────────────────────────

def get_window(src, lon, lat, buffer_m):
    """Converte coordenadas geográficas para janela raster."""
    tf = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
    x, y = tf.transform(lon, lat)
    inv = ~src.transform
    col_min, row_max = inv * (x - buffer_m, y - buffer_m)
    col_max, row_min = inv * (x + buffer_m, y + buffer_m)
    return Window(int(col_min), int(row_min), int(col_max - col_min), int(row_max - row_min))


def read_band(item, band_name, window, env):
    """Lê uma banda Sentinel-2 e devolve em reflectância BOA."""
    with env:
        with rasterio.open(item.assets[band_name].href) as src:
            arr = src.read(1, window=window).astype(np.float32)
    return np.clip(arr / 10000.0, 0.0, 1.5)


def sza_correction(arr, sza_deg):
    """Normaliza reflectância para ângulo zenital solar."""
    cos_sza = math.cos(math.radians(max(sza_deg, 5.0)))
    return arr / cos_sza


def hedley_glint_correction(b_vis, b_swir):
    """
    Correcção de glint Hedley (2005) usando SWIR como proxy de glint.
    B11 (1610nm) é completamente absorvido pela água — qualquer sinal é glint.
    """
    b11_min = float(np.percentile(b_swir[b_swir > 0], 1)) if np.any(b_swir > 0) else 0.0
    glint = np.clip(b_swir - b11_min, 0.0, 1.0)
    # Factor empírico: ~0.9 para 490 nm (Blue), ~0.7 para 560 nm (Green)
    k_blue  = 0.9
    corrected = np.clip(b_vis - k_blue * glint, 0.0, 1.0)
    return corrected.astype(np.float32)


def fft_wave_filter(arr):
    """Remove padrão de ondas (linhas horizontais de baixa frequência)."""
    F = fftshift(fft2(arr.astype(np.float64)))
    rows, cols = F.shape
    # Máscara gaussiana suave (evita artefactos de ringing)
    mask = np.ones((rows, cols), np.float64)
    cx, cy = rows // 2, cols // 2
    # Só suprimir frequências horizontais de muito longa comprimento de onda
    # half_r = 1 pixel de frequência = suprime períodos > metade da cena
    half_r = max(1, rows // 80)
    mask[cx - half_r : cx + half_r, :] = 0.3  # atenuação suave, não corte total
    mask[cx, cy] = 1.0  # preservar DC
    F_filtered = F * mask
    filtered = np.real(ifft2(fftshift(F_filtered))).astype(np.float32)
    return np.clip(filtered, 0.0, None)


def stumpf_ratio(b02, b03, n=1000.0):
    """Ratio logarítmico de Stumpf — correlacionado com profundidade."""
    safe_b02 = np.clip(b02, 0.001, 1.5)
    safe_b03 = np.clip(b03, 0.001, 1.5)
    return np.log(n * safe_b02) / np.log(n * safe_b03)


def water_stretch(arr):
    """Stretch de contraste suave sobre pixels de água."""
    # Usar percentis 2-98 sobre toda a imagem (mais robusto em cenas de água)
    p2, p98 = np.percentile(arr[arr > 0], [2, 98]) if np.any(arr > 0) else (0, 1)
    stretched = np.clip((arr - p2) / (p98 - p2 + 1e-12), 0.0, 1.0)
    # Gamma 0.7 para realçar detalhes nas zonas escuras (fundos rochosos)
    return np.power(stretched, 0.7)


def denoise(arr):
    sigma = np.mean(estimate_sigma(arr))
    # h=1.2*sigma mais agressivo, patch maior para água homogénea
    denoised = denoise_nl_means(arr, h=1.2 * sigma, fast_mode=True,
                                patch_size=7, patch_distance=11)
    # Gaussian suavizante extra para eliminar salt-and-pepper residual
    denoised = cv2.GaussianBlur(denoised, (3, 3), sigmaX=0.8)
    return denoised


def clahe_enhance(arr, clip_limit=0.5, tile_size=8):
    """CLAHE suave — clipLimit baixo para não amplificar ruído em água."""
    arr16 = np.clip(arr * 65535, 0, 65535).astype(np.uint16)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    enhanced = clahe.apply(arr16).astype(np.float32) / 65535.0
    # 80% original + 20% CLAHE (blend muito conservador para água)
    return arr * 0.8 + enhanced * 0.2


def save_image(img_arr, title, out_path, cmap, dpi=300):
    """Guarda imagem sem margens com colormap."""
    fig, ax = plt.subplots(figsize=(7, 7), facecolor="#001108")
    ax.imshow(img_arr, cmap=cmap, interpolation="bilinear", vmin=0, vmax=1)
    ax.set_title(title, color="white", fontsize=10, pad=6, fontfamily="monospace")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0.04, top=0.96)
    fig.savefig(out_path, dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"  ✅ Guardado: {out_path.name} ({size_mb:.2f} MB)")
    return out_path


# ── Reusable Processing Function ──────────────────────────────────────────────
def process_site(site_key, date_str, buffer_m=600, out_dir=OUT_DIR, dpi=300):
    """
    Downloads, corrects, filters and processes Sentinel-2 imagery for a given site and date.
    Returns processed bands, ratio, normalized ratio, rgb, bounds, and metadata.
    """
    site = SITES[site_key]
    lat, lon = site["lat"], site["lon"]
    label = site["label"]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n🔍 Processando: {label} em {date_str}")
    print(f"   Coordenadas: {lat}°N, {lon}°W | Profundidade: {site['depth_m']} m\n")

    # 1. Pesquisar cena STAC — fallback multi-provider
    STAC_PROVIDERS = [
        {
            "name": "Planetary Computer",
            "url": "https://planetarycomputer.microsoft.com/api/stac/v1",
            "collection": "sentinel-2-l2a",
            "modifier": pc.sign_inplace,
            "signed": True,
        },
        {
            "name": "Element84 Earth Search",
            "url": "https://earth-search.aws.element84.com/v1",
            "collection": "sentinel-2-l2a",
            "modifier": None,
            "signed": False,
        },
    ]

    target_date = dt.strptime(date_str, "%Y-%m-%d").date()
    next_day = target_date + pd.Timedelta(days=1)
    items = []
    provider_used = None

    for provider in STAC_PROVIDERS:
        print(f"   📡 A tentar: {provider['name']}...")
        for attempt in range(1, 4):  # 3 tentativas por provider
            try:
                kwargs = {}
                if provider["modifier"]:
                    kwargs["modifier"] = provider["modifier"]
                catalog = Client.open(provider["url"], **kwargs)
                search = catalog.search(
                    collections=[provider["collection"]],
                    intersects={"type": "Point", "coordinates": [lon, lat]},
                    datetime=f"{target_date.isoformat()}/{next_day.isoformat()}"
                )
                items = list(search.items())
                provider_used = provider
                print(f"   ✅ {provider['name']}: {len(items)} cena(s) encontrada(s)")
                break
            except Exception as e:
                wait = 2 ** attempt
                print(f"   ⚠️  Tentativa {attempt}/3 falhou: {str(e)[:80]}")
                if attempt < 3:
                    time.sleep(wait)
        if items:
            break
        print(f"   ❌ {provider['name']} indisponível, a tentar próximo provider...")

    if not items:
        print("❌ Todos os providers STAC falharam. Verifique a ligação à internet.")
        sys.exit(1)

    # Escolher cena com menos nuvens
    items_sorted = sorted(items, key=lambda x: x.properties.get("eo:cloud_cover", 99))
    item = items_sorted[0]
    cloud_pct = item.properties.get("eo:cloud_cover", "?")
    sza_deg = item.properties.get("s2:mean_solar_zenith", 30.0)
    print(f"   Cena: {item.id}")
    print(f"   Nuvens: {cloud_pct}% | SZA: {sza_deg:.1f}°\n")

    env = rasterio.Env(AWS_NO_SIGN_REQUEST="YES", GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR")

    def _get_asset_href(item, pc_name: str, e84_fallbacks: list[str]) -> str:
        if pc_name in item.assets:
            return item.assets[pc_name].href
        for fb in e84_fallbacks:
            if fb in item.assets:
                return item.assets[fb].href
        raise KeyError(f"Asset '{pc_name}' não encontrado. Disponíveis: {list(item.assets.keys())}")

    # 2. Ler B02 e determinar janela (B02 é 10m)
    print("📡 A descarregar bandas S2...")
    b02_href = _get_asset_href(item, "B02", ["blue", "B02"])
    with env:
        with rasterio.open(b02_href) as src_b02:
            window = get_window(src_b02, lon, lat, buffer_m)
            b02_raw = src_b02.read(1, window=window).astype(np.float32)
            
            # Obter coordenadas geográficas precisas da janela de corte
            col_min, row_min = window.col_off, window.row_off
            col_max, row_max = col_min + window.width, row_min + window.height
            x_min, y_max = src_b02.transform * (col_min, row_min)
            x_max, y_min = src_b02.transform * (col_max, row_max)
            tf_to_wgs = Transformer.from_crs(src_b02.crs, "EPSG:4326", always_xy=True)
            min_lon, min_lat = tf_to_wgs.transform(x_min, y_min)
            max_lon, max_lat = tf_to_wgs.transform(x_max, y_max)
            bounds_wgs84 = (min_lat, min_lon, max_lat, max_lon)
            
            s2_transform = rasterio.windows.transform(window, src_b02.transform)
            s2_crs = src_b02.crs

    b02 = np.clip(b02_raw / 10000.0, 0.0, 1.5)
    print(f"   B02: {b02.shape} pixels ({b02.shape[0]*10}m × {b02.shape[1]*10}m)")

    # 3. Ler B03 (Green, 10m)
    b03_href = _get_asset_href(item, "B03", ["green", "B03"])
    with env:
        with rasterio.open(b03_href) as src:
            b03_raw = src.read(1, window=window).astype(np.float32)
    b03 = np.clip(b03_raw / 10000.0, 0.0, 1.5)
    print("   B03 (Green): OK")

    # 3b. Ler B04 (Red, 10m)
    b04_href = _get_asset_href(item, "B04", ["red", "B04"])
    with env:
        with rasterio.open(b04_href) as src:
            b04_raw = src.read(1, window=window).astype(np.float32)
    b04 = np.clip(b04_raw / 10000.0, 0.0, 1.5)
    print("   B04 (Red): OK")

    # 4. Ler B11 (SWIR 1610nm, 20m → resample para 10m)
    b11_href = _get_asset_href(item, "B11", ["swir16", "B11"])
    with env:
        with rasterio.open(b11_href) as src_b11:
            tf = Transformer.from_crs("EPSG:4326", src_b11.crs, always_xy=True)
            x, y = tf.transform(lon, lat)
            inv = ~src_b11.transform
            col_min_b11, row_max_b11 = inv * (x - buffer_m, y - buffer_m)
            col_max_b11, row_min_b11 = inv * (x + buffer_m, y + buffer_m)
            win_b11 = Window(int(col_min_b11), int(row_min_b11),
                             int(col_max_b11 - col_min_b11), int(row_max_b11 - row_min_b11))
            b11_raw = src_b11.read(1, window=win_b11).astype(np.float32)
    b11 = np.clip(b11_raw / 10000.0, 0.0, 1.5)
    b11_upsampled = cv2.resize(b11, (b02.shape[1], b02.shape[0]),
                               interpolation=cv2.INTER_LINEAR)
    print("   B11 (SWIR): OK (resample 20m→10m)")

    # ── PIPELINE DE PROCESSAMENTO ─────────────────────────────────────────────
    print("\n⚙️  Correcções físicas...")

    # 5. Correcção SZA
    b02_sza = sza_correction(b02, sza_deg)
    b03_sza = sza_correction(b03, sza_deg)
    print(f"   SZA correction ({sza_deg:.1f}°): aplicada")

    # 6. Glint Hedley com SWIR
    b02_glint = hedley_glint_correction(b02_sza, b11_upsampled)
    b03_glint = hedley_glint_correction(b03_sza, b11_upsampled)
    glint_removed = float(np.mean(b02_sza) - np.mean(b02_glint))
    print(f"   Glint Hedley: removido ~{glint_removed*100:.2f}% de reflectância espúria")

    # 7. Denoising NL-Means
    b02_dn = denoise(b02_glint)
    b03_dn = denoise(b03_glint)
    print("   Denoising NL-Means: aplicado")

    # 8. FFT wave filter
    b02_fft = fft_wave_filter(b02_dn)
    print("   FFT wave filter: aplicado")

    # 9. CLAHE
    b02_enh = clahe_enhance(b02_fft)
    b03_enh = clahe_enhance(b03_dn)
    print("   CLAHE: aplicado")

    # 10. Ratio de Stumpf (proxy de profundidade)
    ratio = stumpf_ratio(b02_enh, b03_enh)
    print("   Stumpf B02/B03 ratio: calculado")

    # ── STRETCH FINAL ─────────────────────────────────────────────────────────
    b02_final  = water_stretch(b02_enh)
    ratio_norm = water_stretch(np.clip(ratio - 0.9, 0, 0.5))  # normalizar ratio ≈ 1.0

    print("\n🎨 A renderizar imagens...")
    slug = site_key.replace("_", "-")
    date_slug = date_str.replace("-", "")

    # A. Blue (B02 melhorado com todos os filtros)
    out_blue = out_dir / f"{slug}_{date_slug}_blue_enhanced.png"
    save_image(b02_final, f"{label} · Blue B02 Enhanced · {date_str}", out_blue, blue_depth_cmap, dpi)

    # B. Green (B02 no colormap verde)
    out_green = out_dir / f"{slug}_{date_slug}_green_enhanced.png"
    save_image(b02_final, f"{label} · Green Reef · {date_str}", out_green, green_reef_cmap, dpi)

    # C. Depth Proxy (Stumpf ratio)
    out_depth = out_dir / f"{slug}_{date_slug}_depth_proxy.png"
    save_image(ratio_norm, f"{label} · Depth Proxy (Stumpf B02/B03) · {date_str}", out_depth, blue_depth_cmap, dpi)

    # D. RGB natural (B04=Red, B03=Green, B02=Blue) — True Colour Sentinel-2
    b04_sza   = sza_correction(b04, sza_deg)
    b04_glint = hedley_glint_correction(b04_sza, b11_upsampled)

    rgb_raw = np.stack([b04_glint, b03_glint, b02_glint], axis=-1).astype(np.float32)
    rgb_smooth = cv2.GaussianBlur(rgb_raw, (7, 7), sigmaX=1.5)

    vals_flat = rgb_smooth[rgb_smooth > 0]
    p2  = float(np.percentile(vals_flat, 2))
    p98 = float(np.percentile(vals_flat, 98))
    rgb_stretched = np.clip((rgb_smooth - p2) / (p98 - p2 + 1e-9), 0.0, 1.0)

    rgb = np.power(rgb_stretched, 0.5).astype(np.float32)
    rgb = np.clip(rgb, 0.0, 1.0)

    out_rgb = out_dir / f"{slug}_{date_slug}_natural_rgb.png"
    fig, ax = plt.subplots(figsize=(7, 7), facecolor="#000")
    ax.imshow(rgb, interpolation="bilinear")
    ax.set_title(f"{label} · True Colour S2 (B04/B03/B02) · {date_str}", color="white", fontsize=9, pad=6)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0.04, top=0.96)
    fig.savefig(out_rgb, dpi=dpi, facecolor="black", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"  ✅ Guardado: {out_rgb.name} ({os.path.getsize(out_rgb)/1e6:.2f} MB)")

    print(f"\n✅ Concluído! 4 imagens melhoradas em: {out_dir}")
    
    return {
        "b02": b02,
        "b03": b03,
        "b02_glint": b02_glint,
        "b03_glint": b03_glint,
        "b02_enh": b02_enh,
        "b03_enh": b03_enh,
        "b02_final": b02_final,
        "ratio": ratio,
        "ratio_norm": ratio_norm,
        "rgb": rgb,
        "bounds_wgs84": bounds_wgs84,
        "s2_transform": s2_transform,
        "s2_crs": s2_crs,
        "item": item,
        "cloud_pct": cloud_pct,
        "sza_deg": sza_deg
    }


def main():
    process_site(SITE_KEY, DATE_STR, BUFFER_M, OUT_DIR, DPI)


if __name__ == "__main__":
    main()
