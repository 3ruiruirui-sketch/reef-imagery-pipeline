#!/usr/bin/env python3
"""
batch_all_sites.py — Batch runner for all Algarve reef sites
=================================================================
Generates calibrated bathymetry maps and Sentinel-2 enhancements
for all configured reef sites. Each site runs independently with
full error recovery — a failure in one site does not block others.

Outputs per site (saved to OUT_DIR):
  • *_blue_enhanced.png
  • *_green_enhanced.png
  • *_depth_proxy.png
  • *_natural_rgb.png
  • *_bathy_absolute.png
  • *_bathy_contours.png
  • *_calibration_report.json

A combined batch_report.json is saved in the project root.

Usage:
  python scratch/batch_all_sites.py
  python scratch/batch_all_sites.py --date 2024-08-15 --dpi 1200
"""

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

# ── Bootstrap project path ────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from scratch.generate_enhanced_v2 import process_site, SITES, OUT_DIR
from scratch.calibrate_and_map import calibrate_from_emodnet, main as _calibrate_main

# ── Configuration ─────────────────────────────────────────────────────────────
# Default best-date per site (lowest cloud cover, summer, 2025)
DEFAULT_DATES = {
    "pedra_santa_eulalia": "2025-09-25",
    "pedra_do_alto":       "2025-09-25",
    "tartaruga":           "2025-09-25",
}

REPORT_PATH = _PROJECT_ROOT / "batch_report.json"


def parse_args():
    p = argparse.ArgumentParser(
        description="Batch runner: generate S2 enhancements + absolute bathymetry for all reef sites",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--date", default=None, metavar="YYYY-MM-DD",
        help="Override date for all sites (default: per-site best dates)"
    )
    p.add_argument(
        "--sites", nargs="+", choices=list(SITES.keys()), default=list(SITES.keys()),
        help="Subset of sites to process"
    )
    p.add_argument(
        "--buffer-m", type=int, default=600,
        help="AOI half-size in metres"
    )
    p.add_argument(
        "--dpi", type=int, default=300,
        help="Output image DPI"
    )
    p.add_argument(
        "--skip-calibration", action="store_true",
        help="Skip absolute bathymetry calibration (only generate S2 enhancements)"
    )
    return p.parse_args()


def run_site(site_key: str, date_str: str, buffer_m: int, dpi: int,
             skip_calibration: bool) -> dict:
    """
    Process a single site: S2 enhancements + absolute bathymetry calibration.
    Returns a result dict with status, timings, and file list.
    """
    t0 = time.time()
    site_label = SITES[site_key]["label"]
    result = {
        "site": site_key,
        "label": site_label,
        "date": date_str,
        "status": "pending",
        "error": None,
        "elapsed_s": 0.0,
        "outputs": [],
    }

    print(f"\n{'═'*70}")
    print(f"  🌊 {site_label}  ({site_key})")
    print(f"     Date: {date_str} | Buffer: {buffer_m}m | DPI: {dpi}")
    print(f"{'═'*70}")

    try:
        # Step 1: Sentinel-2 enhanced images
        res = process_site(site_key, date_str, buffer_m, OUT_DIR, dpi)

        slug = site_key.replace("_", "-")
        date_slug = date_str.replace("-", "")
        for suffix in ["blue_enhanced", "green_enhanced", "depth_proxy", "natural_rgb"]:
            p = OUT_DIR / f"{slug}_{date_slug}_{suffix}.png"
            if p.exists():
                result["outputs"].append(str(p))

        result["cloud_pct"] = res.get("cloud_pct", "?")
        result["sza_deg"] = res.get("sza_deg", "?")

        # Step 2: Absolute bathymetry calibration
        if not skip_calibration:
            print("\n📐 A executar calibração batimétrica absoluta...")

            # Inline calibration (re-uses already-downloaded arrays from process_site)
            from src.bathy_calibrator import (
                fetch_isobaths_for_bbox,
                calibrate_stumpf_from_isobaths,
                classify_benthic_zone,
            )
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np
            from rasterio.warp import reproject, Resampling
            import rasterio

            b02_enh = res["b02_enh"]
            b03_enh = res["b03_enh"]
            b02_final = res["b02_final"]
            ratio = res["ratio"]
            bounds_wgs84 = res["bounds_wgs84"]
            s2_transform = res["s2_transform"]
            s2_crs = res["s2_crs"]
            min_lat, min_lon, max_lat, max_lon = bounds_wgs84

            features, use_emodnet = [], False
            try:
                features = fetch_isobaths_for_bbox(min_lon, min_lat, max_lon, max_lat)
                if not features:
                    print("   ⚠️  Sem isóbatas IH no AOI → EMODnet fallback")
                    use_emodnet = True
            except Exception as e:
                print(f"   ⚠️  IH API falhou ({e}) → EMODnet fallback")
                use_emodnet = True

            emodnet_path = str(_PROJECT_ROOT / "scratch" / "regional_emodnet.tif")
            if use_emodnet:
                m0, m1, cal_report = calibrate_from_emodnet(
                    b02_enh, b03_enh, b02_final.shape, s2_transform, s2_crs, emodnet_path
                )
            else:
                m0, m1, cal_report = calibrate_stumpf_from_isobaths(
                    b02_enh, b03_enh, features, bounds_wgs84
                )

            depth_m = m1 * ratio + m0
            masked = np.ma.masked_where(
                (depth_m < 0.0) | (depth_m > 30.0) | np.isnan(depth_m), depth_m
            )

            # Bathy absolute map
            fig, ax = plt.subplots(figsize=(8.5, 7.5), facecolor="#001108")
            im = ax.imshow(masked, cmap="plasma_r", vmin=0, vmax=25, interpolation="bilinear")
            ax.set_title(
                f"{site_label} · Batimetria Absoluta · {date_str}",
                color="white", fontsize=11, pad=10, fontfamily="monospace"
            )
            ax.axis("off")
            cbar = fig.colorbar(im, ax=ax, orientation="vertical", pad=0.03, shrink=0.8)
            cbar.set_label("Profundidade (m)", color="white", fontsize=10, labelpad=8)
            cbar.ax.yaxis.set_tick_params(color="white", labelcolor="white")
            cbar.outline.set_edgecolor("white")
            fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.92)
            out_bathy = OUT_DIR / f"{slug}_{date_slug}_bathy_absolute.png"
            fig.savefig(out_bathy, dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches="tight")
            plt.close(fig)
            print(f"  ✅ {out_bathy.name}")
            result["outputs"].append(str(out_bathy))

            # Contour overlay
            fig, ax = plt.subplots(figsize=(7, 7), facecolor="#001108")
            from scratch.generate_enhanced_v2 import green_reef_cmap
            ax.imshow(b02_final, cmap=green_reef_cmap, interpolation="bilinear", vmin=0, vmax=1)
            levels = [2.0, 5.0, 10.0, 15.0, 20.0]
            valid = [l for l in levels if np.nanmin(depth_m) < l < np.nanmax(depth_m)]
            if valid:
                cs = ax.contour(depth_m, levels=valid, colors="#ff3366", linewidths=1.2, alpha=0.85)
                ax.clabel(cs, inline=True, fontsize=8, fmt="%1.0fm", colors="white")
            ax.set_title(
                f"{site_label} · Contornos · {date_str}",
                color="white", fontsize=10, pad=6, fontfamily="monospace"
            )
            ax.axis("off")
            out_cont = OUT_DIR / f"{slug}_{date_slug}_bathy_contours.png"
            fig.savefig(out_cont, dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches="tight")
            plt.close(fig)
            print(f"  ✅ {out_cont.name}")
            result["outputs"].append(str(out_cont))

            # Calibration report
            site_obj = SITES[site_key]
            zone = classify_benthic_zone(site_obj["lon"], site_obj["lat"], features) if features else {}
            report = {
                "site": site_key, "date": date_str,
                "m0": m0, "m1": m1,
                "isobaths_available": [f["depth"] for f in features] if features else "EMODnet",
                "calibration": cal_report, "benthic_zone": zone,
            }
            out_json = OUT_DIR / f"{slug}_{date_slug}_calibration_report.json"
            with open(out_json, "w", encoding="utf-8") as fh:
                json.dump(report, fh, indent=2, ensure_ascii=False)
            print(f"  ✅ {out_json.name}")
            result["outputs"].append(str(out_json))

        result["status"] = "success"
        result["elapsed_s"] = round(time.time() - t0, 1)
        print(f"\n  ✅ {site_label} concluído em {result['elapsed_s']}s")

    except Exception as exc:
        result["status"] = "error"
        result["error"] = str(exc)
        result["traceback"] = traceback.format_exc()
        result["elapsed_s"] = round(time.time() - t0, 1)
        print(f"\n  ❌ {site_label} falhou após {result['elapsed_s']}s: {exc}")

    return result


def main():
    args = parse_args()
    t_start = time.time()

    print("\n" + "🌊 " * 20)
    print("  BATCH RUNNER — Algarve Reef Imagery Pipeline")
    print("  Sites : " + ", ".join(args.sites))
    print("  DPI   : " + str(args.dpi))
    print("🌊 " * 20)

    results = []
    for site_key in args.sites:
        date_str = args.date or DEFAULT_DATES.get(site_key, "2025-09-25")
        result = run_site(
            site_key=site_key,
            date_str=date_str,
            buffer_m=args.buffer_m,
            dpi=args.dpi,
            skip_calibration=args.skip_calibration,
        )
        results.append(result)

    # ── Final Summary ──────────────────────────────────────────────────────────
    total_elapsed = round(time.time() - t_start, 1)
    n_ok  = sum(1 for r in results if r["status"] == "success")
    n_err = sum(1 for r in results if r["status"] == "error")

    batch_report = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "total_elapsed_s": total_elapsed,
        "sites_ok": n_ok,
        "sites_failed": n_err,
        "results": results,
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        json.dump(batch_report, fh, indent=2, ensure_ascii=False)

    print(f"\n\n{'━'*70}")
    print(f"  BATCH COMPLETO — {n_ok}/{len(results)} sites com sucesso em {total_elapsed}s")
    for r in results:
        icon = "✅" if r["status"] == "success" else "❌"
        err_note = f"  → {r['error']}" if r["error"] else ""
        print(f"  {icon}  {r['label']:<30} ({r['elapsed_s']}s){err_note}")
    print(f"\n  📄 Relatório: {REPORT_PATH}")
    print(f"{'━'*70}\n")

    if n_err > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
