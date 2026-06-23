#!/usr/bin/env python3
"""
west_candidates_audit.py — Visual audit of top western Algarve reef candidates.

Renders true-colour + green-reflectance crops (S2B_29SNB_20250925_0_L2A) for
the top-ranked western candidates, with GPS-confirmed sites highlighted.
Output: outputs/reef_detector/west_candidates_audit.html
"""
from __future__ import annotations
import base64, io, json, logging
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

SCENE_ID = "S2B_29SNB_20250925_0_L2A"
OUT_HTML  = Path("outputs/reef_detector/west_candidates_audit.html")
HALF_PX   = 80   # 800 m half-window → 1600 m wide crop
TOP_N     = 12   # how many candidates to show


def _resolve(scene_id: str) -> dict:
    cat = Client.open("https://earth-search.aws.element84.com/v1")
    it  = list(cat.search(collections=["sentinel-2-l2a"], ids=[scene_id]).items())[0]
    return {k: v.href for k, v in it.assets.items()}


def _crop(href: str, lon: float, lat: float, half: int = HALF_PX):
    with rasterio.open(href) as src:
        tf = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        x, y = tf.transform(lon, lat)
        r, c = src.index(x, y)
        r0, r1 = max(0, r - half), min(src.height, r + half)
        c0, c1 = max(0, c - half), min(src.width,  c + half)
        cy, cx = r - r0, c - c0
        arr = src.read(1, window=Window.from_slices((r0, r1), (c0, c1))).astype(np.float32) / 10000.0
    return arr, cy, cx


def build_panel(assets: dict, lon: float, lat: float, title: str) -> str:
    blue,  cy, cx = _crop(assets["blue"],  lon, lat)
    green, _,  _  = _crop(assets["green"], lon, lat)
    red,   _,  _  = _crop(assets["red"],   lon, lat)

    rgb = np.dstack([red, green, blue])
    p2, p98 = np.percentile(rgb, 2), np.percentile(rgb, 98)
    rgb = np.clip((rgb - p2) / (p98 - p2 + 1e-9), 0, 1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(rgb)
    axes[0].set_title(f"True colour  ({HALF_PX*10*2} m wide)", fontsize=9)
    axes[1].imshow(green, cmap="viridis_r")
    axes[1].set_title("Green reflectance — reef = DARK", fontsize=9)
    for ax in axes:
        ax.axhline(cy, color="cyan", lw=0.7, alpha=0.8)
        ax.axvline(cx, color="cyan", lw=0.7, alpha=0.8)
        ax.plot(cx, cy, "o", mfc="none", mec="cyan", ms=20, mew=2)
        ax.axis("off")

    fig.suptitle(title, fontsize=8.5, color="#1a3a5c")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def main() -> None:
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)

    # Load top western candidates
    geojson = json.load(open("outputs/reef_detector/reef_candidates_west_2025-09-25.geojson"))
    feats = geojson["features"][:TOP_N]

    log.info("Resolving %s …", SCENE_ID)
    assets = _resolve(SCENE_ID)

    cards = []
    for f in feats:
        p = f["properties"]
        lon, lat = p["centroid"]
        gps = p.get("gps_confirmed_reef") or ""
        title = (f"Rank #{p['rank']}  |  {lat:.4f}°N  {abs(lon):.4f}°W  "
                 f"|  {p['area_ha']} ha  mean_like={p['mean_like']}"
                 + (f"  ★ {gps}" if gps else ""))
        log.info("  Rendering rank #%d  %s …", p["rank"], gps or "unknown")
        try:
            b64 = build_panel(assets, float(lon), float(lat), title)
            cards.append((p, b64))
        except Exception as exc:
            log.warning("  Failed rank #%d: %s", p["rank"], exc)

    html_parts = ["""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<title>West Algarve Candidates Audit — Sentinel-2 2025-09-25</title>
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
  .confirmed { background:#1a3a4a; color:#22d3ee; border:1px solid #22d3ee; }
  .info { background:#1a2744; border:1px solid #1e3a5c; border-radius:8px;
          padding:12px; margin-bottom:20px; font-size:0.85rem; line-height:1.6; }
</style></head><body>
<h1>🪸 Western Algarve — Top Candidates Audit (Sentinel-2 2025-09-25)</h1>
<div class="info">
  Scene: <strong>S2B_29SNB_20250925_0_L2A</strong> · cloud 1.25% · tile 29SNB · bbox (-8.26, 37.03, -8.16, 37.09)<br>
  Sorted by reef score (mean likelihood × log area). Cyan crosshair = candidate centroid.<br>
  In the green-reflectance panel: <strong>reef/rock appears DARK</strong>, sand appears bright.<br>
  ★ = GPS-confirmed dive site or high-confidence fishing GPS.
</div>
<h2>Top candidates — does the crosshair land on a dark reef structure?</h2>
"""]

    for p, b64 in cards:
        gps = p.get("gps_confirmed_reef") or ""
        ml  = p.get("mean_like", 0)
        badge_class = "good" if ml >= 0.42 else ("mid" if ml >= 0.36 else "low")
        html_parts.append(f"""<div class="card">
  <div class="meta">
    <strong>Rank #{p['rank']}</strong>
    &nbsp;·&nbsp; {p['centroid'][1]:.4f}°N &nbsp; {abs(p['centroid'][0]):.4f}°W
    &nbsp;·&nbsp; {p['area_ha']} ha
    &nbsp;&nbsp;
    <span class="badge {badge_class}">mean_like {p['mean_like']:.4f}</span>
    <span class="badge {'good' if p.get('score',0)>0.5 else 'mid'}">score {p.get('score',0):.3f}</span>
    {f'<span class="badge confirmed">★ {gps}</span>' if gps else ''}
  </div>
  <img src="data:image/png;base64,{b64}" alt="rank-{p['rank']}">
</div>""")

    html_parts.append("</body></html>")
    OUT_HTML.write_text("\n".join(html_parts))
    log.info("Wrote %s", OUT_HTML)
    print(f"\n✓ Open in Safari:\n  file://{OUT_HTML.resolve()}")


if __name__ == "__main__":
    main()
