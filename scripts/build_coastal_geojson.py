#!/usr/bin/env python3
"""
build_coastal_geojson.py — Rebuild algarve_coastal_features.geojson from its CSV.

Use this after any VM-based coastal feature extraction or whenever the CSV is
updated outside the normal generate_coastal_features_dgt.py flow.

    python scripts/build_coastal_geojson.py
    python scripts/build_coastal_geojson.py --csv path/to/other.csv --out path/to/out.geojson

A timestamped backup (.bak_YYYYMMDDTHHMMSS) is created before overwriting.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils import build_coastal_geojson  # noqa: E402

DEFAULT_CSV = ROOT / "outputs" / "coastal_topography" / "algarve_coastal_features.csv"
DEFAULT_GEOJSON = ROOT / "outputs" / "coastal_topography" / "algarve_coastal_features.geojson"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=str(DEFAULT_CSV), help="Input CSV (default: outputs/coastal_topography/algarve_coastal_features.csv)")
    ap.add_argument("--out", default=str(DEFAULT_GEOJSON), help="Output GeoJSON (default: same dir, same stem)")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    geojson_path = Path(args.out)

    if not csv_path.exists():
        sys.exit(f"ERROR: CSV not found: {csv_path}")

    print(f"Reading  {csv_path}")
    n = build_coastal_geojson(csv_path, geojson_path)
    print(f"Written  {geojson_path}  ({n} features)")

    # List any backups created in the same directory
    backups = sorted(geojson_path.parent.glob(geojson_path.name + ".bak_*"))
    if backups:
        print(f"Backup   {backups[-1].name}")


if __name__ == "__main__":
    main()
