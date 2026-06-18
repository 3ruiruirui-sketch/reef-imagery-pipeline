#!/usr/bin/env python3
"""
generate_coastal_features_dgt.py — Regenerate algarve_coastal_features.csv from
DGT 50 cm LiDAR, processed PER SITE.

Why per-site: the DGT MDT-50cm catalogue is tiled at ~1 km²; a single mosaic over
the full 15-site bounding box (~220×220 km) would need hundreds of tiles, far past
the STAC limit.  GLO-30 (global, 30 m) can do one big mosaic, but for 50 cm we
build a small mosaic around each site (buffer reaches the emerged coast — the reef
points themselves are offshore/submerged and not covered by topographic LiDAR).

Requires DGT_CDD_USERNAME / DGT_CDD_PASSWORD (load via `set -a; source .env; set +a`).
"""
from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.coastal_topography import CoastalTopographyAnalyzer  # noqa: E402
from src.utils import build_coastal_geojson  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("gen_coastal_dgt")

SURVEY_SITES = [
    ("tavira_west",       37.1050, -7.6800),
    ("ilha_de_tavira",    37.0950, -7.7200),
    ("fuseta",            37.0600, -7.7600),
    ("olhao_offshore",    37.0350, -7.8200),
    ("faro_east",         37.0100, -7.8800),
    ("praia_de_faro",     36.9800, -7.9400),
    ("ancão_peninsula",   36.9650, -8.0000),
    ("quarteira",         37.0650, -8.1000),
    ("vilamoura",         37.0750, -8.1300),
    ("olhos_de_agua",     37.0900, -8.1600),
    ("pedra_sta_eulalia", 37.069081, -8.210242),
    ("albufeira_reef",    37.0690, -8.2105),
    ("galé",              37.0560, -8.2296),
    ("salgados",          37.0950, -8.3000),
    ("armacao_de_pera",   37.0700, -8.3600),
]

OUT_DIR = Path("outputs/coastal_topography")
BUFFER_M = 2000          # feature-stats radius; reaches the emerged coast
BBOX_MARGIN_DEG = 0.022  # ~2.4 km half-box (must cover BUFFER_M); keeps tile count down


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[pd.DataFrame] = []
    work_root = OUT_DIR / "_per_site_dgt"

    for i, (name, lat, lon) in enumerate(SURVEY_SITES, 1):
        log.info("=" * 70)
        log.info("[%d/%d] %s  (%.5f N, %.5f W)", i, len(SURVEY_SITES), name, lat, lon)
        m = BBOX_MARGIN_DEG
        bbox = (lon - m, lat - m, lon + m, lat + m)
        analyzer = CoastalTopographyAnalyzer(
            bbox=bbox, output_dir=str(work_root / name), dem_source="dgt"
        )
        try:
            res = analyzer.run_analysis(
                sites=[(name, lat, lon)],
                buffer_m=BUFFER_M,
                output_name=f"{name}_features",
                include_chm=False,  # skip MDS/CHM — features only need slope/aspect
            )
            csv = res.get("output_files", {}).get("csv") if isinstance(res, dict) else None
            if csv and Path(csv).exists():
                rows.append(pd.read_csv(csv))
                log.info("  ✓ %s features extracted", name)
            else:
                log.warning("  ✗ %s — no features (no DGT coverage at this site?)", name)
        except Exception as exc:  # noqa: BLE001
            log.warning("  ✗ %s failed: %s", name, exc)
        finally:
            # Drop the heavy rasters/tiles immediately — we only keep the stats.
            shutil.rmtree(work_root / name, ignore_errors=True)

    if not rows:
        log.error("No sites produced features — aborting (production CSV untouched)")
        sys.exit(1)

    df = pd.concat(rows, ignore_index=True)
    out_csv = OUT_DIR / "algarve_coastal_features.csv"
    df.to_csv(out_csv, index=False)

    out_geojson = OUT_DIR / "algarve_coastal_features.geojson"
    n = build_coastal_geojson(out_csv, out_geojson)
    log.info("=" * 70)
    log.info("DONE: %d/%d sites → %s + %s (%d features)", len(df), len(SURVEY_SITES), out_csv.name, out_geojson.name, n)
    print(df[["site_name", "slope_mean", "slope_max", "aspect_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
