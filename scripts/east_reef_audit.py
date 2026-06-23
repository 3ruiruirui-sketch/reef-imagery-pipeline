#!/usr/bin/env python3
"""
east_reef_audit.py — Safari audit page for eastern Algarve fishing GPS spots.

Shows true-colour + green-reflectance crops (S2B_29SNA_20250925_0_L2A) for each
high-score eastern fishing spot. Crosshair = GPS coordinate.
Cyan circle = confirmed high-TRI fishing location.
"""
from __future__ import annotations
import base64, io, logging
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
from pyproj import Transformer
from pystac_client import Client

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

SCENE_ID  = "S2B_29SNA_20250925_0_L2A"
OUT_HTML  = Path("outputs/reef_detector/east_reef_audit.html")
HALF_PX   = 80   # 80 px × 10 m/px = 800 m half-window → 1600 m wide crop

SITES = [
    {"name": "e_faro_reef_11m",    "lon": -8.02498, "lat": 37.00989, "depth": 11, "visits": 12, "tri": 0.60, "pct": 71.5},
    {"name": "e_faro_reef_12m_a",  "lon": -8.02339, "lat": 37.00917, "depth": 12, "visits": 12, "tri": 0.65, "pct": 37.2},
    {"name": "e_faro_reef_12m_b",  "lon": -8.02327, "lat": 37.00857, "depth": 12, "visits": 10, "tri": 0.57, "pct": 54.1},
    {"name": "e_faro_16m_highTRI", "lon": -7.99863, "lat": 36.98575, "depth": 16, "visits":  5, "tri": 0.99, "pct": 44.9},
    {"name": "e_olhao_12m",        "lon": -7.94072, "lat": 36.95948, "depth": 12, "visits": 11, "tri": 0.47, "pct": 55.4},
    {"name": "e_tavira_13m",       "lon": -7.94326, "lat": 36.96016, "depth": 13, "visits":  7, "tri": 0.82, "pct": 60.2},
    {"name": "e_olhao_17m_a",      "lon": -7.97075, "lat": 36.96263, "depth": 17, "visits": 12, "tri": 0.47, "pct": 24.0},
    {"name": "e_olhao_17m_b",      "lon": -7.97262, "lat": 36.96305, "depth": 17, "visits": 12, "tri": 0.45, "pct":  8.3},
]


def _crop(href: str, lon: float, lat: float, half: int = HALF_PX):
    with rasterio.open(href) as src:
        tf = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        x, y = tf.transform(lon, lat)
        r, c = src.index(x, y)
        r0, r1 = max(0, r - half), min(src.height, r + half)
        c0, c1 = max(0, c - half), min(src.width,  c + half)
        cy, cx = r - r0, c - c0   # crosshair position within crop
        arr = src.read(1, window=Window.from_slices((r0, r1), (c0, c1))).astype(np.float32) / 10000.0
    return arr, cy, cx


def _resolve(scene_id: str) -> dict:
    cat = Client.open("https://earth-search.aws.element84.com/v1")
    it  = list(cat.search(collections=["sentinel-2-l2a"], ids=[scene_id]).items())[0]
    return {k: v.href for k, v in it.assets.items()}


def build_panel(assets: dict, site: dict) -> str:
    lon, lat = site["lon"], site["lat"]
    blue,  cy, cx = _crop(assets["blue"],  lon, lat)
    green, _,  _  = _crop(assets["green"], lon, lat)
    red,   _,  _  = _crop(assets["red"],   lon, lat)

    rgb = np.dstack([red, green, blue])
    p2, p98 = np.percentile(rgb, 2), np.percentile(rgb, 98)
    rgb = np.clip((rgb - p2) / (p98 - p2 + 1e-9), 0, 1)

    w_m = HALF_PX * 10 * 2

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(rgb)
    axes[0].set_title(f"True colour  ({w_m} m wide)", fontsize=9)
    axes[1].imshow(green, cmap="viridis_r")
    axes[1].set_title("Green reflectance — reef = DARK", fontsize=9)

    for ax in axes:
        ax.axhline(cy, color="cyan", lw=0.7, alpha=0.8)
        ax.axvline(cx, color="cyan", lw=0.7, alpha=0.8)
        ax.plot(cx, cy, "o", mfc="none", mec="cyan", ms=20, mew=2)
        ax.axis("off")

    title = (f"{site['name']}  |  {site['lat']:.5f}°N  {abs(site['lon']):.5f}°W  "
             f"|  {site['depth']} m depth  |  {site['visits']}× visits  "
             f"|  TRI={site['tri']:.2f}  |  detector={site['pct']:.1f}th pct")
    fig.suptitle(title, fontsize=8.5, color="#1a3a5c")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def main() -> None:
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    log.info("Resolving %s …", SCENE_ID)
    assets = _resolve(SCENE_ID)

    cards = []
    for site in SITES:
        log.info("  Rendering %s …", site["name"])
        try:
            b64 = build_panel(assets, site)
            cards.append((site, b64))
        except Exception as exc:
            log.warning("  Failed %s: %s", site["name"], exc)

    html_parts = ["""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<title>East Algarve Reef Audit — Sentinel-2 2025-09-25</title>
<style>
  body { font-family:-apple-system,sans-serif; background:#0a1628; color:#e8edf5;
         padding:20px; max-width:1150px; margin:0 auto; }
  h1 { color:#7ec8e3; font-size:1.3rem; }
  h2 { color:#a8c5da; font-size:1rem; margin-top:28px;
       border-bottom:1px solid #1e3a5c; padding-bottom:6px; }
  .card { background:#12213a; border-radius:10px; padding:14px; margin:14px 0; }
  .card img { width:100%; border-radius:6px; }
  .meta { font-size:0.8rem; color:#6b7f94; margin:6px 0; }
  .badge { display:inline-block; padding:2px 8px; border-radius:4px;
           font-size:0.75rem; margin:2px; }
  .good { background:#1a4a2e; color:#4ade80; }
  .mid  { background:#3a3a1a; color:#facc15; }
  .low  { background:#3a1a1a; color:#f87171; }
  .info { background:#1a2744; border:1px solid #1e3a5c; border-radius:8px;
          padding:12px; margin-bottom:20px; font-size:0.85rem; line-height:1.6; }
</style></head><body>
<h1>🔍 Eastern Algarve — Reef Audit (Sentinel-2 2025-09-25)</h1>
<div class="info">
  Scene: <strong>S2B_29SNA_20250925_0_L2A</strong> · cloud 1.1% · tile 29SNA<br>
  Source: <strong>CarlosBarrinha.gpx</strong> fishing waypoints — repeated visits at optical depth (5–22 m)<br>
  Cyan crosshair = GPS coordinate. In the green-reflectance panel: <strong>reef/rock appears DARK</strong>, sand appears bright.<br>
  TRI ≥ 0.30 = rocky/rugose seafloor. Higher percentile = stronger dark-anomaly reef signal.
</div>
<h2>Top fishing spots by detector score — please confirm: does the crosshair land on a dark reef structure?</h2>
"""]

    for site, b64 in cards:
        pct = site["pct"]
        badge_class = "good" if pct >= 60 else ("mid" if pct >= 40 else "low")
        html_parts.append(f"""<div class="card">
  <div class="meta">
    <strong>{site['name'].replace('_',' ')}</strong>
    &nbsp;·&nbsp; {site['lat']:.5f}°N &nbsp; {abs(site['lon']):.5f}°W
    &nbsp;·&nbsp; {site['depth']} m depth
    &nbsp;&nbsp;
    <span class="badge good">{site['visits']}× visits</span>
    <span class="badge {'good' if site['tri']>=0.50 else 'mid'}">TRI {site['tri']:.2f}</span>
    <span class="badge {badge_class}">detector {pct:.1f}th pct</span>
  </div>
  <img src="data:image/png;base64,{b64}" alt="{site['name']}">
</div>""")

    html_parts.append("</body></html>")
    OUT_HTML.write_text("\n".join(html_parts))
    log.info("Wrote %s", OUT_HTML)
    print(f"\n✓ Open in Safari:\n  file://{OUT_HTML.resolve()}")


if __name__ == "__main__":
    main()
