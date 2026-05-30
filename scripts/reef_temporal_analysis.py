"""
reef_temporal_analysis.py
=========================
Temporal consistency analysis of reef candidates across ALL available dates.

For each date that has both ratio_B02_B03_YYYYMMDD.tif and S2_B02/B03_YYYYMMDD.tif:
  1. Generate Stumpf SDB (bathy_s2_stumpf_YYYYMMDD.tif) if missing
  2. Generate reef candidates (reef_candidates_YYYYMMDD.geojson) if missing
  3. Cross-reference candidates vs ratio raster (z-score vs background)
  4. Aggregate across dates by spatial proximity

Outputs (written to reef_Output_Master/reef_output_v3/):
  - reef_temporal_consistency_YYYYMMDD.csv
  - reef_temporal_consistency.png

Usage:
    python scripts/reef_temporal_analysis.py
"""

import argparse
import sys
import os
from pathlib import Path as _Path, Path

# Insert project root so `src.*` imports resolve; insert scripts/ dir so sibling
# reef_bathy_module can be imported without installing the package.
_PROJECT_ROOT = str(_Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
_SCRIPTS_DIR = os.path.dirname(__file__)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

from datetime import datetime

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask as rio_mask
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shapely.geometry import box as shapely_box
from shapely.ops import unary_union

# Import bathy module functions
from reef_bathy_module import (
    compute_s2_depth_inversion,
    compute_bathy_indices,
    detect_reef_candidates,
    _validate_tiff,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

def _parse_args():
    parser = argparse.ArgumentParser(description="Reef temporal consistency analysis")
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory. Defaults to reef_Output_Master/reef_output_v3. "
             "For Santa Eulalia use: outputs/santa_eulalia_multiband_analysis/"
    )
    return parser.parse_args()

_args = _parse_args()
OUTPUT_DIR = Path(_args.output) if _args.output else BASE_DIR / "reef_Output_Master" / "reef_output_v3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TODAY      = datetime.utcnow().strftime("%Y%m%d")
CSV_OUT    = OUTPUT_DIR / f"reef_temporal_consistency_{TODAY}.csv"
PNG_OUT    = OUTPUT_DIR / "reef_temporal_consistency.png"

# Proximity threshold for matching candidates across dates (metres)
PROXIMITY_M = 50.0

# High-confidence threshold (z-score below this = strong benthic signal)
HC_THRESHOLD = -1.0

# ---------------------------------------------------------------------------
# Step 1 — Find all dates with ratio + S2_B02 + S2_B03
# ---------------------------------------------------------------------------

N_SCALE = 1000.0  # Stumpf log-ratio scaling factor


def ensure_ratio(date: str) -> bool:
    """
    Generate ratio_B02_B03_YYYYMMDD.tif from S2_B02 and S2_B03 if missing.
    ratio = ln(n * B02) / ln(n * B03)  (Stumpf log-ratio, n=1000)
    Returns True if file exists (pre-existing or newly created).
    """
    ratio_path = OUTPUT_DIR / f"ratio_B02_B03_{date}.tif"
    if ratio_path.exists():
        return True
    b02_path = OUTPUT_DIR / f"S2_B02_{date}.tif"
    b03_path = OUTPUT_DIR / f"S2_B03_{date}.tif"
    if not (b02_path.exists() and b03_path.exists()):
        return False
    try:
        import rasterio
        with rasterio.open(str(b02_path)) as src_b02:
            b02 = src_b02.read(1).astype(np.float32)
            profile = src_b02.profile.copy()
        with rasterio.open(str(b03_path)) as src_b03:
            b03 = src_b03.read(1).astype(np.float32)
        valid = (b02 > 0) & (b03 > 0)
        b02_s = np.where(valid, b02, np.nan)
        b03_s = np.where(valid, b03, np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_b = np.log(N_SCALE * b02_s)
            log_g = np.log(N_SCALE * b03_s)
            safe = (log_g > 0) & np.isfinite(log_b) & np.isfinite(log_g)
            ratio = np.where(safe, log_b / log_g, np.nan)
        profile.update(dtype=rasterio.float32, count=1, compress="lzw", nodata=np.nan)
        with rasterio.open(str(ratio_path), 'w', **profile) as dst:
            dst.write(ratio.astype(np.float32), 1)
        print(f"  [GEN] Created ratio: {ratio_path.name}")
        return True
    except Exception as e:
        print(f"  [FAIL] Could not create ratio for {date}: {e}")
        return False


def find_available_dates() -> list[str]:
    """Return sorted list of YYYYMMDD strings that have all required inputs."""
    import re
    # Ensure all dates with B02/B03 have ratio files
    b02_files = sorted(OUTPUT_DIR.glob("S2_B02_*.tif"))
    dates = []
    for bf in b02_files:
        m = re.search(r"S2_B02_(\d{8})\.tif$", bf.name)
        if not m:
            continue
        d = m.group(1)
        b03 = OUTPUT_DIR / f"S2_B03_{d}.tif"
        if not b03.exists():
            print(f"  [SKIP] {d}: missing S2_B03")
            continue
        if not ensure_ratio(d):
            print(f"  [SKIP] {d}: could not create ratio")
            continue
        dates.append(d)
    return sorted(dates)


# ---------------------------------------------------------------------------
# Step 2 — Ensure Stumpf SDB exists
# ---------------------------------------------------------------------------

def ensure_stumpf(date: str) -> str | None:
    """Return path to bathy_s2_stumpf_YYYYMMDD.tif, computing if needed."""
    out = OUTPUT_DIR / f"bathy_s2_stumpf_{date}.tif"
    if out.exists() and _validate_tiff(str(out)):
        print(f"  [OK] Stumpf bathy already exists: {out.name}")
        return str(out)

    b02 = str(OUTPUT_DIR / f"S2_B02_{date}.tif")
    b03 = str(OUTPUT_DIR / f"S2_B03_{date}.tif")
    print(f"  [GEN] Running Stumpf SDB for {date} …")
    result = compute_s2_depth_inversion(b02, b03, str(OUTPUT_DIR), date_str=date)
    if result and _validate_tiff(result):
        print(f"  [OK] Generated: {Path(result).name}")
        return result
    print(f"  [FAIL] Stumpf SDB failed for {date}")
    return None


# ---------------------------------------------------------------------------
# Step 3 — Ensure reef candidates exist
# ---------------------------------------------------------------------------

def ensure_candidates(date: str, stumpf_path: str) -> str | None:
    """Return path to reef_candidates_YYYYMMDD.geojson, computing if needed."""
    out = OUTPUT_DIR / f"reef_candidates_{date}.geojson"
    if out.exists():
        # Verify it has at least one feature
        try:
            gdf = gpd.read_file(str(out))
            if len(gdf) > 0:
                print(f"  [OK] Candidates already exist: {out.name} ({len(gdf)} polygons)")
                return str(out)
            else:
                print(f"  [WARN] Candidates file is empty for {date}, regenerating …")
        except Exception as e:
            print(f"  [WARN] Could not read candidates for {date}: {e}, regenerating …")

    base_name = Path(stumpf_path).stem
    indices_dir = str(OUTPUT_DIR)

    # Check if indices exist already (for 20241015 they are precomputed)
    tri_path  = OUTPUT_DIR / f"{base_name}_tri.tif"
    bpi_path  = OUTPUT_DIR / f"{base_name}_bpi_broad.tif"

    if not (tri_path.exists() and bpi_path.exists()):
        print(f"  [GEN] Computing bathy indices for {date} …")
        indices = compute_bathy_indices(stumpf_path, indices_dir)
        if not indices:
            print(f"  [FAIL] Bathy indices failed for {date}")
            return None
    else:
        print(f"  [OK] Bathy indices already exist for {date}")

    print(f"  [GEN] Detecting reef candidates for {date} …")
    result = detect_reef_candidates(
        stumpf_path, indices_dir, str(OUTPUT_DIR),
        depth_min=-50.0, depth_max=-1.0,
        date_str=date,
    )
    if result and Path(result).exists():
        try:
            gdf = gpd.read_file(result)
            print(f"  [OK] Generated candidates: {Path(result).name} ({len(gdf)} polygons)")
        except Exception:
            print(f"  [OK] Generated candidates: {Path(result).name}")
        return result
    print(f"  [FAIL] Candidate detection failed for {date}")
    return None


# ---------------------------------------------------------------------------
# Step 4 — Cross-analysis: per-candidate stats from ratio raster
# ---------------------------------------------------------------------------

def extract_pixels(geom, src_path: str) -> np.ndarray:
    """Return valid (non-NaN, non-inf) pixel values within geom."""
    with rasterio.open(src_path) as src:
        try:
            out_image, _ = rio_mask(src, [geom], crop=True, nodata=np.nan,
                                    all_touched=False)
            pixels = out_image[0].flatten()
            pixels = pixels[np.isfinite(pixels)]
        except Exception:
            pixels = np.array([], dtype="float32")
    return pixels


def run_cross_analysis(date: str, candidates_path: str) -> pd.DataFrame | None:
    """
    Cross-reference reef candidates vs the ratio raster for this date.
    Returns DataFrame with columns:
        candidate_id, centroid_x, centroid_y, mean_ratio, z_score, date
    """
    ratio_path = str(OUTPUT_DIR / f"ratio_B02_B03_{date}.tif")
    if not Path(ratio_path).exists():
        print(f"  [FAIL] Ratio raster missing for {date}")
        return None

    try:
        gdf = gpd.read_file(candidates_path)
    except Exception as e:
        print(f"  [FAIL] Could not read candidates for {date}: {e}")
        return None

    if len(gdf) == 0:
        print(f"  [SKIP] No candidates for {date}")
        return None

    gdf = gdf.reset_index(drop=True)
    gdf["candidate_id"] = [f"C{str(i).zfill(2)}" for i in range(len(gdf))]

    # Reproject if needed
    with rasterio.open(ratio_path) as src:
        raster_crs = src.crs
    if gdf.crs and gdf.crs.to_epsg() != raster_crs.to_epsg():
        gdf = gdf.to_crs(raster_crs)

    # Background stats
    with rasterio.open(ratio_path) as src:
        aoi_geom = shapely_box(*src.bounds)

    candidates_union = unary_union(gdf.geometry.values)
    bg_geom = aoi_geom.difference(candidates_union)
    bg_pixels = extract_pixels(bg_geom, ratio_path)

    if len(bg_pixels) == 0:
        print(f"  [WARN] No background pixels for {date}, using full raster stats")
        with rasterio.open(ratio_path) as src:
            arr = src.read(1).astype("float32")
            bg_pixels = arr[np.isfinite(arr)].flatten()

    bg_mean = float(np.mean(bg_pixels))
    bg_std  = float(np.std(bg_pixels, ddof=1)) if len(bg_pixels) > 1 else 1.0

    if bg_std == 0:
        bg_std = 1.0  # avoid divide-by-zero

    # Per-candidate stats
    records = []
    for _, row in gdf.iterrows():
        pix = extract_pixels(row.geometry, ratio_path)
        centroid = row.geometry.centroid
        if len(pix) == 0:
            mean_r = np.nan
            z_score = np.nan
        else:
            mean_r = float(np.mean(pix))
            z_score = (mean_r - bg_mean) / bg_std

        records.append({
            "candidate_id": row["candidate_id"],
            "centroid_x":   centroid.x,
            "centroid_y":   centroid.y,
            "mean_ratio":   mean_r,
            "z_score":      z_score,
            "date":         date,
            "bg_mean":      bg_mean,
            "bg_std":       bg_std,
            "pixel_count":  int(len(pix)),
        })

    df = pd.DataFrame(records).dropna(subset=["mean_ratio"])
    print(f"  [OK] {date}: {len(df)} candidates, bg_mean={bg_mean:.5f}, bg_std={bg_std:.6f}")
    hc = (df["z_score"] < HC_THRESHOLD).sum()
    print(f"       → {hc} high-confidence (z < {HC_THRESHOLD})")
    return df


# ---------------------------------------------------------------------------
# Step 5 — Aggregate across dates by spatial proximity
# ---------------------------------------------------------------------------

def cluster_by_proximity(all_records: pd.DataFrame, proximity_m: float = PROXIMITY_M
                          ) -> pd.DataFrame:
    """
    Group candidate detections across dates by centroid proximity.

    Returns a DataFrame with one row per unique spatial location:
        loc_id, easting, northing, n_dates_detected, n_dates_high_confidence,
        mean_zscore, best_zscore, best_date, all_dates
    Also appends loc_id back to all_records for per-date-per-loc plotting.
    """
    if len(all_records) == 0:
        return pd.DataFrame()

    coords = all_records[["centroid_x", "centroid_y"]].values
    labels = np.full(len(coords), -1, dtype=int)
    next_label = 0

    # Greedy proximity clustering
    for i in range(len(coords)):
        if labels[i] >= 0:
            continue
        labels[i] = next_label
        for j in range(i + 1, len(coords)):
            if labels[j] >= 0:
                continue
            dist = np.sqrt((coords[i, 0] - coords[j, 0])**2 +
                           (coords[i, 1] - coords[j, 1])**2)
            if dist <= proximity_m:
                labels[j] = next_label
        next_label += 1

    all_records = all_records.copy()
    all_records["loc_id"] = [f"L{str(l).zfill(3)}" for l in labels]

    # Aggregate per location
    rows = []
    for loc_id, grp in all_records.groupby("loc_id"):
        hc = (grp["z_score"] < HC_THRESHOLD).sum()
        best_row = grp.loc[grp["z_score"].idxmin()]
        rows.append({
            "loc_id":               loc_id,
            "easting":              grp["centroid_x"].mean(),
            "northing":             grp["centroid_y"].mean(),
            "n_dates_detected":     len(grp["date"].unique()),
            "n_dates_high_conf":    int(hc),
            "mean_zscore":          grp["z_score"].mean(),
            "best_zscore":          best_row["z_score"],
            "best_date":            best_row["date"],
            "all_dates":            ";".join(sorted(grp["date"].unique())),
        })

    locs = pd.DataFrame(rows)
    # Consistency score: n_hc × |mean_zscore| (larger = more consistent and more negative)
    locs["consistency_score"] = locs["n_dates_high_conf"] * np.abs(locs["mean_zscore"])
    locs = locs.sort_values("consistency_score", ascending=False).reset_index(drop=True)
    return locs, all_records


# ---------------------------------------------------------------------------
# Step 6 — Add WGS84 lat/lon to the locations DataFrame
# ---------------------------------------------------------------------------

def add_latlon(locs: pd.DataFrame, epsg_in: int = 32629) -> pd.DataFrame:
    """Convert easting/northing to WGS84 lat/lon and append to locs."""
    from pyproj import Transformer
    transformer = Transformer.from_crs(f"EPSG:{epsg_in}", "EPSG:4326", always_xy=True)
    lons, lats = transformer.transform(locs["easting"].values, locs["northing"].values)
    locs = locs.copy()
    locs["lon"] = lons
    locs["lat"] = lats
    return locs


# ---------------------------------------------------------------------------
# Step 7 — Summary plot
# ---------------------------------------------------------------------------

def make_consistency_plot(locs: pd.DataFrame, all_records: pd.DataFrame,
                           all_dates: list[str]) -> None:
    """
    X-axis: locations ranked by consistency score (left = most consistent)
    Y-axis: z-score per date (each date a different color line/marker)
    Horizontal line at z = HC_THRESHOLD
    """
    n_locs = len(locs)
    if n_locs == 0:
        print("  [WARN] No locations to plot")
        return

    # Color palette per date
    cmap = plt.cm.get_cmap("tab20", len(all_dates))
    date_colors = {d: cmap(i) for i, d in enumerate(sorted(all_dates))}

    fig, ax = plt.subplots(figsize=(max(12, n_locs * 0.45), 7))

    # Map loc_id → x position (rank order)
    loc_rank = {row["loc_id"]: i for i, row in locs.iterrows()}

    # Plot per-date z-scores
    for date in sorted(all_dates):
        date_data = all_records[all_records["date"] == date].copy()
        if len(date_data) == 0:
            continue
        # Only plot locations that appear in locs
        date_data = date_data[date_data["loc_id"].isin(loc_rank)]
        xs = [loc_rank[lid] for lid in date_data["loc_id"]]
        ys = date_data["z_score"].values
        yr = date.replace("20", "20")[2:]  # short year label: "241015"
        label = f"{date[:4]}-{date[4:6]}-{date[6:]}"
        ax.scatter(xs, ys, color=date_colors[date], alpha=0.75, s=35,
                   zorder=3, label=label)
        # Connect points from same date with a faint line
        pairs = sorted(zip(xs, ys))
        if len(pairs) > 1:
            px, py = zip(*pairs)
            ax.plot(px, py, color=date_colors[date], alpha=0.2, lw=0.8, zorder=2)

    # High-confidence threshold line
    ax.axhline(HC_THRESHOLD, color="#E53935", lw=1.8, ls="--",
               label=f"High-confidence threshold (z = {HC_THRESHOLD})", zorder=4)
    ax.axhline(0, color="#90A4AE", lw=1.0, ls=":", alpha=0.6)

    # X-axis labels: loc_id
    ax.set_xticks(range(n_locs))
    ax.set_xticklabels(locs["loc_id"].values, rotation=90, fontsize=7)

    ax.set_xlabel("Location (ranked by temporal consistency score)", fontsize=11)
    ax.set_ylabel("B02/B03 z-score (lower = stronger benthic signal)", fontsize=11)
    ax.set_title("Reef candidate temporal consistency — Albufeira Reef\n"
                 f"({len(all_dates)} dates · {n_locs} unique spatial locations · "
                 f"proximity threshold {PROXIMITY_M:.0f} m)",
                 fontsize=12, fontweight="bold")

    ax.legend(fontsize=7, loc="upper right", ncol=2, framealpha=0.85)
    ax.grid(axis="y", alpha=0.3)
    ax.invert_yaxis()  # most negative (strongest signal) at top

    plt.tight_layout()
    fig.savefig(str(PNG_OUT), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[OK] Plot saved → {PNG_OUT}")


# ---------------------------------------------------------------------------
# Step 8 — Find C16 match
# ---------------------------------------------------------------------------

def find_c16_match(locs: pd.DataFrame, all_records: pd.DataFrame) -> str:
    """Return the loc_id closest to C16's centroid from 20241015."""
    C16_X = 569930.93
    C16_Y = 4102927.78

    c16_records = all_records[
        (all_records["date"] == "20241015") &
        (all_records["candidate_id"] == "C16")
    ]
    if len(c16_records) == 0:
        # Fall back to closest loc centroid
        dists = np.sqrt((locs["easting"] - C16_X)**2 + (locs["northing"] - C16_Y)**2)
        closest_idx = dists.idxmin()
        return locs.loc[closest_idx, "loc_id"], float(dists.min())

    c16_loc = c16_records.iloc[0]["loc_id"]
    loc_row = locs[locs["loc_id"] == c16_loc]
    if len(loc_row) == 0:
        dists = np.sqrt((locs["easting"] - C16_X)**2 + (locs["northing"] - C16_Y)**2)
        closest_idx = dists.idxmin()
        return locs.loc[closest_idx, "loc_id"], float(dists.min())

    dist = np.sqrt((loc_row.iloc[0]["easting"] - C16_X)**2 +
                   (loc_row.iloc[0]["northing"] - C16_Y)**2)
    return c16_loc, float(dist)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("REEF TEMPORAL ANALYSIS — Albufeira Reef")
    print(f"Output dir: {OUTPUT_DIR}")
    print("=" * 70)

    # --- Step 1: Find dates ---
    dates = find_available_dates()
    print(f"\nFound {len(dates)} dates with all required inputs:")
    for d in dates:
        print(f"  {d}")

    if not dates:
        print("[ERROR] No valid dates found. Exiting.")
        sys.exit(1)

    # --- Steps 2–4: Process each date ---
    all_frames = []
    processed_dates = []
    skipped_dates = []

    for date in dates:
        print(f"\n{'─'*60}")
        print(f"Processing: {date}")
        print(f"{'─'*60}")

        # Step 2: Stumpf SDB
        stumpf_path = ensure_stumpf(date)
        if stumpf_path is None:
            print(f"  [SKIP] {date}: Stumpf SDB failed")
            skipped_dates.append((date, "Stumpf SDB failed"))
            continue

        # Step 3: Candidates
        candidates_path = ensure_candidates(date, stumpf_path)
        if candidates_path is None:
            print(f"  [SKIP] {date}: Candidate detection failed")
            skipped_dates.append((date, "Candidate detection failed"))
            continue

        # Step 4: Cross-analysis
        df = run_cross_analysis(date, candidates_path)
        if df is None or len(df) == 0:
            print(f"  [SKIP] {date}: Cross-analysis produced no results")
            skipped_dates.append((date, "Cross-analysis empty"))
            continue

        all_frames.append(df)
        processed_dates.append(date)

    print(f"\n{'='*70}")
    print(f"Processed: {len(processed_dates)} dates")
    if skipped_dates:
        print(f"Skipped:   {len(skipped_dates)} dates:")
        for d, reason in skipped_dates:
            print(f"  {d}: {reason}")

    if not all_frames:
        print("[ERROR] No data to aggregate. Exiting.")
        sys.exit(1)

    # --- Step 5: Aggregate ---
    all_records = pd.concat(all_frames, ignore_index=True)
    print(f"\nTotal candidate-date pairs: {len(all_records)}")

    result = cluster_by_proximity(all_records, PROXIMITY_M)
    if isinstance(result, tuple):
        locs, all_records = result
    else:
        locs = result

    print(f"Unique spatial locations (proximity ≤ {PROXIMITY_M} m): {len(locs)}")

    # --- Step 6: Add lat/lon ---
    try:
        locs = add_latlon(locs, epsg_in=32629)
    except Exception as e:
        print(f"  [WARN] Could not convert to WGS84: {e}")
        locs["lat"] = np.nan
        locs["lon"] = np.nan

    # --- Save CSV ---
    col_order = ["loc_id", "easting", "northing", "lat", "lon",
                 "n_dates_detected", "n_dates_high_conf",
                 "mean_zscore", "best_zscore", "best_date",
                 "consistency_score", "all_dates"]
    col_order = [c for c in col_order if c in locs.columns]
    locs[col_order].to_csv(str(CSV_OUT), index=False, float_format="%.6f")
    print(f"\n[OK] CSV saved → {CSV_OUT}")

    # --- Step 7: Plot ---
    make_consistency_plot(locs, all_records, processed_dates)

    # --- Step 8: Summary report ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Dates processed        : {len(processed_dates)}")
    print(f"  Date range             : {min(processed_dates)} → {max(processed_dates)}")
    print(f"  Dates skipped          : {len(skipped_dates)}")
    print(f"  Unique spatial locs    : {len(locs)}")
    print(f"  Proximity threshold    : {PROXIMITY_M} m")

    hc_locs = locs[locs["n_dates_high_conf"] > 0]
    print(f"  Locs with any HC date  : {len(hc_locs)}")

    print("\n--- Top 5 most consistent locations ---")
    top5 = locs.head(5)
    for _, row in top5.iterrows():
        print(f"  {row['loc_id']}: {row['n_dates_high_conf']}/{row['n_dates_detected']} HC dates, "
              f"mean_z={row['mean_zscore']:.3f}, best_z={row['best_zscore']:.3f} ({row['best_date']}), "
              f"score={row['consistency_score']:.3f}")
        if not np.isnan(row['lat']):
            print(f"    → lat={row['lat']:.5f}, lon={row['lon']:.5f}")

    print("\n--- C16 (20241015, strongest single-date signal) match ---")
    try:
        c16_loc_id, c16_dist = find_c16_match(locs, all_records)
        c16_row = locs[locs["loc_id"] == c16_loc_id]
        if len(c16_row) > 0:
            r = c16_row.iloc[0]
            print(f"  C16 → {c16_loc_id} (centroid distance {c16_dist:.1f} m)")
            print(f"  {r['n_dates_high_conf']}/{r['n_dates_detected']} HC dates, "
                  f"mean_z={r['mean_zscore']:.3f}, best_z={r['best_zscore']:.3f} ({r['best_date']})")
            rank = locs[locs["loc_id"] == c16_loc_id].index[0] + 1
            print(f"  Temporal rank: #{rank} out of {len(locs)} locations")
        else:
            print(f"  C16 maps to {c16_loc_id} but no rows found in locs")
    except Exception as e:
        print(f"  [WARN] C16 matching failed: {e}")

    print("\n" + "=" * 70)
    print("Done.")
    print(f"  CSV  → {CSV_OUT}")
    print(f"  Plot → {PNG_OUT}")
    print("=" * 70)


if __name__ == "__main__":
    main()
