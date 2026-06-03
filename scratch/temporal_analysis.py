#!/usr/bin/env python3
"""
temporal_analysis.py
========================================================================
Main temporal analysis script for the Pedra do Alto reef.
Performs historical trend analysis and change detection from 2017 to 2025.

Steps:
  1. Searches the best summer scene (July-September) for each year.
  2. Runs process_site() to extract the cleaned Stumpf depth proxy.
  3. Saves the arrays and renders a comparison grid for all years.
  4. Generates a line chart showing historical visibility/reflectance trends.
  5. Computes and plots the difference map (2025 - 2017) for change detection.
"""

import os
import sys
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from pystac_client import Client

# Add project root to sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

from scratch.generate_enhanced_v2 import process_site, SITES, OUT_DIR, blue_depth_cmap

SITE_KEY = "pedra_do_alto"
BUFFER_M = 600   # 1.2 km × 1.2 km
DPI      = 300

def find_best_date_for_year(year: int, lat: float, lon: float) -> str:
    """
    Queries Element84 Earth Search STAC catalog for the best clear-water scene
    in the summer window (July 1st to September 30th) of the given year.
    Returns the YYYY-MM-DD date string of the lowest cloud cover scene.
    """
    url = "https://earth-search.aws.element84.com/v1"
    print(f"   📡 A procurar melhor cena de {year} no Element84...")
    
    start_date = f"{year}-07-01"
    end_date = f"{year}-09-30"
    
    try:
        catalog = Client.open(url)
        # Search with a strict cloud cover threshold first (<5%)
        search = catalog.search(
            collections=["sentinel-2-l2a"],
            intersects={"type": "Point", "coordinates": [lon, lat]},
            datetime=f"{start_date}/{end_date}",
            query={"eo:cloud_cover": {"lt": 5.0}}
        )
        items = list(search.items())
        
        # Fallback to <20% cloud cover if none found
        if not items:
            search = catalog.search(
                collections=["sentinel-2-l2a"],
                intersects={"type": "Point", "coordinates": [lon, lat]},
                datetime=f"{start_date}/{end_date}",
                query={"eo:cloud_cover": {"lt": 20.0}}
            )
            items = list(search.items())
            
        if not items:
            print(f"   ⚠️  Nenhuma cena encontrada para {year} no verão. A tentar alargado (Junho-Outubro)...")
            search = catalog.search(
                collections=["sentinel-2-l2a"],
                intersects={"type": "Point", "coordinates": [lon, lat]},
                datetime=f"{year}-06-01/{year}-10-31",
                query={"eo:cloud_cover": {"lt": 30.0}}
            )
            items = list(search.items())
            
        if not items:
            print(f"   ❌ Nenhuma cena elegível para {year}.")
            return None
            
        # Sort by cloud cover
        items_sorted = sorted(items, key=lambda x: x.properties.get("eo:cloud_cover", 99.0))
        best_item = items_sorted[0]
        best_date = best_item.datetime.strftime("%Y-%m-%d")
        print(f"      Selected: {best_date} ({best_item.properties.get('eo:cloud_cover', 99.0):.2f}% clouds)")
        return best_date
        
    except Exception as e:
        print(f"   ⚠️  Erro na pesquisa STAC para {year}: {e}")
        # Return fallback hardcoded typical best clear dates if STAC query fails
        fallbacks = {
            2017: "2017-09-08",
            2018: "2018-08-03",
            2019: "2019-09-12",
            2020: "2020-09-06",
            2021: "2021-08-07",
            2022: "2022-09-26",
            2023: "2023-09-01",
            2024: "2024-09-15",
            2025: "2025-09-25"
        }
        print(f"      Using robust fallback date: {fallbacks[year]}")
        return fallbacks[year]

def main():
    site = SITES[SITE_KEY]
    lat, lon = site["lat"], site["lon"]
    label = site["label"]
    
    print("="*80)
    # Print a premium, high-impact header
    print(f"  🌊 ANALISE TEMPORAL DO RECIFE PEDRA DO ALTO (2017-2025)")
    print(f"  Localização: {lat}°N, {lon}°W | Profundidade de Referência: {site['depth_m']}m")
    print("="*80)
    
    years = list(range(2017, 2026))
    processed_data = {}
    
    # Step 1: Process each year
    for yr in years:
        print(f"\n⏳ [Ano {yr}]")
        best_date = find_best_date_for_year(yr, lat, lon)
        if not best_date:
            continue
            
        # Check if ratio array already exists to avoid re-downloading
        arr_path = OUT_DIR / f"pedra_do_alto_{yr}_ratio.npy"
        if arr_path.exists():
            print(f"   ✓ Carregado array persistido de {arr_path.name}")
            ratio = np.load(arr_path)
            from scratch.generate_enhanced_v2 import water_stretch
            ratio_norm = water_stretch(np.clip(ratio - 0.9, 0, 0.5))
            processed_data[yr] = {
                "date": best_date,
                "ratio_norm": ratio_norm,
                "ratio": ratio
            }
            continue

        # Process and retrieve arrays
        try:
            res = process_site(SITE_KEY, best_date, BUFFER_M, OUT_DIR, DPI)
            processed_data[yr] = {
                "date": best_date,
                "ratio_norm": res["ratio_norm"],
                "ratio": res["ratio"]
            }
            # Save raw array for persistence
            np.save(arr_path, res["ratio"])
        except Exception as e:
            print(f"   ❌ Falha ao processar {best_date}: {e}")
            
    if len(processed_data) < 2:
        print("❌ Dados insuficientes para gerar a análise temporal (mínimo de 2 anos necessários).")
        sys.exit(1)
        
    print("\n📊 A calcular estatísticas e tendências temporais...")
    
    # ── A. Grid Comparativo 3x3 ────────────────────────────────────────────────
    # We want a beautiful, publication-ready grid showing all processed years
    fig, axes = plt.subplots(3, 3, figsize=(15, 15), facecolor="#001108")
    axes = axes.ravel()
    
    for idx, yr in enumerate(years):
        ax = axes[idx]
        if yr in processed_data:
            data = processed_data[yr]
            im = ax.imshow(data["ratio_norm"], cmap=blue_depth_cmap, interpolation="bilinear", vmin=0, vmax=1)
            ax.set_title(f"Ano {yr} · {data['date']}", color="white", fontsize=11, fontfamily="monospace")
        else:
            ax.text(0.5, 0.5, f"Ano {yr}\nSem dados", color="red", ha="center", va="center")
            
        ax.axis("off")
        
    fig.suptitle(f"{label} · Evolução do Depth Proxy (Stumpf B02/B03) · 2017-2025", color="white", fontsize=16, y=0.96)
    fig.subplots_adjust(left=0.03, right=0.97, bottom=0.03, top=0.92, wspace=0.1, hspace=0.1)
    
    out_grid = OUT_DIR / "temporal_grid_pedra_alto.png"
    fig.savefig(out_grid, dpi=DPI, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Guardado Grid: {out_grid.name} ({os.path.getsize(out_grid)/1e6:.2f} MB)")
    
    # ── B. Gráfico de Tendência (Trend Line) ──────────────────────────────────
    # Calculate the average depth proxy value over time in the central reef zone (e.g. 60x60 pixels = 600m x 600m)
    trend_dates = []
    trend_means = []
    
    for yr in sorted(processed_data.keys()):
        ratio_arr = processed_data[yr]["ratio"]
        h, w = ratio_arr.shape
        cy, cx = h // 2, w // 2
        # Crop central 300m x 300m = 30x30 pixels
        central_crop = ratio_arr[cy - 15 : cy + 15, cx - 15 : cx + 15]
        mean_val = float(np.mean(central_crop))
        trend_dates.append(yr)
        trend_means.append(mean_val)
        
    fig, ax = plt.subplots(figsize=(10, 5), facecolor="#001108")
    ax.set_facecolor("#002211")
    
    ax.plot(trend_dates, trend_means, marker="o", color="#00ffcc", linewidth=2.5, markersize=8, label="Médio Central Reef")
    ax.set_title(f"{label} · Tendência Temporal do Depth Proxy (2017-2025)", color="white", fontsize=12, pad=15)
    ax.set_xlabel("Ano", color="white", fontsize=10)
    ax.set_ylabel("Valor Médio do Stumpf Ratio", color="white", fontsize=10)
    ax.tick_params(colors="white")
    ax.grid(True, linestyle="--", alpha=0.3, color="white")
    
    # Customize spine colors
    for spine in ax.spines.values():
        spine.set_color("white")
        
    out_trend = OUT_DIR / "temporal_trend.png"
    fig.savefig(out_trend, dpi=DPI, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Guardado Gráfico de Tendência: {out_trend.name} ({os.path.getsize(out_trend)/1e6:.2f} MB)")
    
    # ── C. Deteção de Mudanças (Change Detection Map) ─────────────────────────
    # Compute: Delta = 2025_ratio - 2017_ratio
    # Blue = increased depth proxy (darker seabed, higher ratio SNR)
    # Red = decreased depth proxy (increased brightness, sediment deposition or shallower)
    if 2017 in processed_data and 2025 in processed_data:
        r2017 = processed_data[2017]["ratio"]
        r2025 = processed_data[2025]["ratio"]
        
        # Ensure identical shapes
        if r2017.shape == r2025.shape:
            delta = r2025 - r2017
            # Smooth delta slightly to reduce high frequency noise
            delta_smooth = cv2.GaussianBlur(delta, (5, 5), 0)
            
            fig, ax = plt.subplots(figsize=(8, 7), facecolor="#001108")
            # Diverging colormap: seismic (Red = negative difference, Blue = positive difference)
            im = ax.imshow(delta_smooth, cmap="seismic", vmin=-0.08, vmax=0.08, interpolation="bilinear")
            ax.set_title(f"{label} · Deteção de Mudanças (2025 vs 2017)\nAzul: Fundo Escuro/Profundo | Vermelho: Mais Brilhante/Deposicional", color="white", fontsize=11, pad=10, fontfamily="monospace")
            ax.axis("off")
            
            cbar = fig.colorbar(im, ax=ax, orientation="vertical", pad=0.03, shrink=0.8)
            cbar.set_label("Diferença do Stumpf Ratio (Δ)", color="white", fontsize=10, labelpad=8)
            cbar.ax.yaxis.set_tick_params(color="white", labelcolor="white")
            cbar.outline.set_edgecolor("white")
            
            fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.92)
            out_change = OUT_DIR / "change_detection.png"
            fig.savefig(out_change, dpi=DPI, facecolor=fig.get_facecolor(), bbox_inches="tight")
            plt.close(fig)
            print(f"  ✅ Guardado Mapa de Mudanças: {out_change.name} ({os.path.getsize(out_change)/1e6:.2f} MB)")
            
    print("\n✅ Concluída com sucesso a Análise Temporal Histórica!")

if __name__ == "__main__":
    main()
