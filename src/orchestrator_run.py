#!/usr/bin/env python3
"""
Orchestrator — Reef Benthic Visibility Pipeline
================================================
Orquestra: ACOLITE (ou fallback L2A) → run_predictor() → JSON report

Refactor v2.0 — physics delegation:
  - Removed 5 duplicate physics fns (snell, beer-lambert, sunglint, gdal-extract, analyse_band)
  - All physics now handled by reef_ml_predictor_acolite.run_predictor()
  - Added load_config() for YAML-driven paths (no more hardcoded paths)
  - Glint penalty sourced from GLINT_PENALTY constant (not hardcoded 0.60)

Uso: python3 -m src.orchestrator_run [--depth 16.0] [--config config.yaml]
"""

import os, json, subprocess, logging, argparse, shutil, shlex
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import rasterio

from src.constants import (
    N_WATER, CLOUD_THRESHOLD, SNR_THRESHOLD, KD490_TABLE, KD490_DEFAULT,
    GLINT_PENALTY, GLINT_PENALTY_DEFAULT
)
from src.reef_ml_predictor_acolite import run_predictor

try:
    from src.ih_bathy_features import get_bathy_features_for_summary
    HAS_IH_BATHY = True
except ImportError:
    HAS_IH_BATHY = False

# Drift monitoring (shadow mode — never blocks pipeline)
try:
    from src.drift_monitor import reset as drift_reset, log_summary as drift_log_summary
    from src.drift_export import export_to_file as drift_export_file
    from src.drift_history import export_history_json as drift_history_json
    from src.drift_history import export_history_csv as drift_history_csv
    from src.drift_report import export_html as drift_export_html
    HAS_DRIFT_MONITOR = True
except ImportError:
    HAS_DRIFT_MONITOR = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Config defaults ──────────────────────────────────────────────────────────
PROJECT_DIR   = Path(__file__).parent.parent

# Coastal terrain features (shadow mode — never blocks pipeline)
try:
    import pandas as _pd
    _COASTAL_CSV = PROJECT_DIR / "outputs" / "coastal_topography" / "algarve_coastal_features.csv"
    _TERRAIN_DF = _pd.read_csv(_COASTAL_CSV) if _COASTAL_CSV.exists() else None
    HAS_TERRAIN = _TERRAIN_DF is not None and not _TERRAIN_DF.empty
except Exception:
    _TERRAIN_DF = None
    HAS_TERRAIN = False

IMAGE_A_B02   = PROJECT_DIR / "reef_Output_Master/reef_output_pedra_to_gale_20250925/S2_B02_20250925.tif"
IMAGE_A_B03   = PROJECT_DIR / "reef_Output_Master/reef_output_pedra_to_gale_20250925/S2_B03_20250925.tif"
IMAGE_B_B02   = PROJECT_DIR / "reef_Output_Master/reef_output_ai_prediction_spot_2023/S2_B02_20231001.tif"
IMAGE_B_B03   = PROJECT_DIR / "reef_Output_Master/reef_output_ai_prediction_spot_2023/S2_B03_20231001.tif"
OUTPUT_DIR    = PROJECT_DIR / "reef_output_acolite_comparison"

METADATA = {
    "A": {"date": "2025-09-25", "sza": 40.498, "saa": 158.883, "cloud": 1.245, "level": "L2A", "month": 9},
    "B": {"date": "2023-10-01", "sza": 42.413, "saa": 160.459, "cloud": 0.007, "level": "L2A", "month": 10},
}

TARGET_LAT, TARGET_LON = 37.05815, -8.20982


# ── Config loader (YAML-driven, optional) ────────────────────────────────────
def load_config(config_path: str | Path | None = None) -> dict:
    """
    Load YAML config file if present. Falls back to module-level defaults.
    Expected keys: image_a_b02, image_a_b03, image_b_b02, image_b_b03,
                   output_dir, target_lat, target_lon, metadata.
    """
    if config_path is None:
        return {}
    try:
        import yaml  # type: ignore
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
        log.info("Loaded config from %s", config_path)
        return cfg
    except ImportError:
        log.warning("PyYAML not installed — ignoring config file %s", config_path)
        return {}
    except FileNotFoundError:
        log.warning("Config file not found: %s — using defaults", config_path)
        return {}
    except Exception as e:
        log.warning("Config file parse error (%s) — using defaults: %s", config_path, e)
        return {}


# ── Shell helpers ─────────────────────────────────────────────────────────────
def run_shell(cmd, check=True):
    """Run a command safely. Accepts str or list of args. Uses shell=False."""
    if isinstance(cmd, str):
        cmd_list = shlex.split(cmd)
    else:
        cmd_list = list(cmd)
    log.info("$ %s", " ".join(shlex.quote(c) for c in cmd_list))
    result = subprocess.run(cmd_list, shell=False, capture_output=True, text=True, timeout=600)
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed:\n{result.stderr}")
    return result

def acolite_available() -> bool:
    return shutil.which("acolite") is not None or shutil.which("acolite_cli") is not None

def snap_gpt_available() -> bool:
    return shutil.which("gpt") is not None and str(shutil.which("gpt")) != "/usr/sbin/gpt"

def run_acolite(input_path: Path, output_dir: Path):
    """Run ACOLITE BOA correction with sunglint removal."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "acolite_cli",
        "--input", str(input_path),
        "--output", str(output_dir),
        "--product", "BOA",
        "--sensor", "S2",
        "--proc", "water",
        "--sunglint", "true",
        "--aot-method", "image",
        "--output-format", "GeoTIFF",
    ]
    run_shell(cmd)
    boa = next(output_dir.glob("*BOA*.tif"), None)
    if not boa:
        raise FileNotFoundError(f"ACOLITE BOA output not found in {output_dir}")
    return boa


# ── Band extractor (merged gdal_extract_b02 + gdal_extract_b03) ──────────────
def extract_band(boa_tif: Path, band_num: int, out_path: Path) -> Path:
    """Extract a single band from a multi-band GeoTIFF using rasterio."""
    with rasterio.open(boa_tif) as src:
        profile = src.profile.copy()
        profile.update(count=1)
        data = src.read(band_num)   # rasterio is 1-based
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data, 1)
    log.info("Extracted band %d → %s", band_num, out_path)
    return out_path


# ── Result normaliser: run_predictor() → legacy key aliases ──────────────────
def _normalise_result(pred: dict, meta: dict) -> dict:
    """
    Thin adapter: maps run_predictor() output keys to the legacy aliases
    expected by _build_justification() and the JSON report builder.
    run_predictor() does not compute b02_cv directly — derive it as 1/SNR.
    """
    # Use explicit None check: dict.get(key, default) returns None when the key
    # IS present but its value is None, so a truthy/falsy test would be wrong.
    _snr_raw = pred.get("snr_mean_16m") if "snr_mean_16m" in pred else pred.get("SNR_mean_16m")
    snr = _snr_raw if _snr_raw is not None else 0.0
    out = dict(pred)
    # Legacy aliases
    out.setdefault("SNR_mean_16m",              snr)
    _month = meta.get("month")
    out.setdefault("kd490_seasonal",
                   KD490_TABLE[_month] if _month in KD490_TABLE
                   else pred.get("kd_seasonal_prior", KD490_DEFAULT))
    _kd_est = pred.get("kd_b02_estimated") if "kd_b02_estimated" in pred else pred.get("kd490_seasonal")
    out.setdefault("kd490_estimated", _kd_est if _kd_est is not None else KD490_DEFAULT)
    out.setdefault("date",                       meta["date"])
    out.setdefault("cloud_cover",                meta["cloud"])
    # b02_cv: coefficient of variation — proxy from SNR (CV = 1/SNR)
    out["b02_cv"] = round(1.0 / max(snr, 1e-6), 5)
    # Sentinel-1 fields (may be absent if run_predictor didn't query S1)
    out.setdefault("s1_scene_id",    "none")
    out.setdefault("s1_scene_date",  "none")
    out.setdefault("s1_roughness",   -1.0)
    out.setdefault("s1_sea_state",   "unknown")
    out.setdefault("s1_penalty_pct", 0.0)
    return out


# ── Output writers ────────────────────────────────────────────────────────────
def save_boa_copy(src_b02: Path, src_b03: Path, out_dir: Path, label: str) -> dict:
    """Copy pre-processed bands to output dir as 'BOA' equivalents."""
    out_dir.mkdir(parents=True, exist_ok=True)
    boa_b02  = out_dir / f"BOA_B02_{label}.tif"
    snr_map  = out_dir / f"SNR_map_{label}.tif"
    conf_map = out_dir / f"Confidence_map_{label}.tif"
    shutil.copy2(src_b02, boa_b02)

    with rasterio.open(src_b02) as src:
        profile = src.profile.copy()
        data = src.read(1).astype(float) / 10000.0
        from src.reef_ml_predictor_acolite import make_snr_map
        snr_px = make_snr_map(data, window=7)
        conf_px  = np.select([snr_px < 5, snr_px < 30], [0, 1], default=2).astype(np.uint8)

    profile.update(dtype=rasterio.float32, count=1)
    with rasterio.open(snr_map, "w", **profile) as dst:
        dst.write(snr_px, 1)
    profile.update(dtype=rasterio.uint8)
    with rasterio.open(conf_map, "w", **profile) as dst:
        dst.write(conf_px, 1)

    return {"boa_b02": str(boa_b02), "snr_map": str(snr_map), "confidence_map": str(conf_map)}


def _build_justification(winner: str, loser: str, results: dict, snr_diff: float, depth: float) -> str:
    """Construct human-readable justification safely (handles zero-division)."""
    w_cv = results[winner]["b02_cv"]
    l_cv = results[loser]["b02_cv"]
    cv_ratio = (l_cv / w_cv) if w_cv > 1e-9 else float("inf")
    snr_str  = f"+{snr_diff:.0f}%" if snr_diff != float("inf") else "+inf%"
    cv_str   = f"{cv_ratio:.1f}×"  if cv_ratio != float("inf") else "∞×"
    return (
        f"Image {winner} ({results[winner]['date']}) chosen. "
        f"SNR {results[winner]['SNR_mean_16m']:.1f} vs {results[loser]['SNR_mean_16m']:.1f} "
        f"({snr_str}). CV {w_cv:.4f} vs {l_cv:.4f} "
        f"({cv_str} more surface noise in loser). "
        f"Kd490={results[winner]['kd490_seasonal']:.3f} m⁻¹ | "
        f"Two-way transmittance={results[winner]['water_transmittance_twoway']:.4f} at {depth:.0f}m."
    )


def save_csv(results: dict, path: Path):
    import csv
    rows = [{"image_key": k, **r} for k, r in results.items()]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    log.info("CSV saved: %s", path)


# ── Main ──────────────────────────────────────────────────────────────────────
def main(depth: float = 16.0, config_path: str | None = None):
    # Apply YAML config overrides (if provided)
    cfg = load_config(config_path)
    b02_a_path  = Path(cfg["image_a_b02"])  if cfg.get("image_a_b02") else IMAGE_A_B02
    b03_a_path  = Path(cfg["image_a_b03"])  if cfg.get("image_a_b03") else IMAGE_A_B03
    b02_b_path  = Path(cfg["image_b_b02"])  if cfg.get("image_b_b02") else IMAGE_B_B02
    b03_b_path  = Path(cfg["image_b_b03"])  if cfg.get("image_b_b03") else IMAGE_B_B03
    out_dir     = Path(cfg["output_dir"])   if cfg.get("output_dir")   else OUTPUT_DIR
    target_lat  = float(cfg["target_lat"])  if cfg.get("target_lat")   else TARGET_LAT
    target_lon  = float(cfg["target_lon"])  if cfg.get("target_lon")   else TARGET_LON

    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("=== Reef Orchestrator — depth=%.1fm ===", depth)

    if HAS_DRIFT_MONITOR:
        try:
            drift_reset()
        except Exception:
            pass

    # Step 1: ACOLITE or direct L2A fallback
    use_acolite = acolite_available()
    log.info("ACOLITE: %s | SNAP/gpt: %s",
             "YES" if use_acolite else "NO (fallback L2A)",
             "YES" if snap_gpt_available() else "NO")

    if use_acolite:
        boa_a = run_acolite(b02_a_path.parent, out_dir / "acolite_A")
        boa_b = run_acolite(b02_b_path.parent, out_dir / "acolite_B")
        b02_a_path = extract_band(boa_a, 2, out_dir / "BOA_B02_A_raw.tif")
        b02_b_path = extract_band(boa_b, 2, out_dir / "BOA_B02_B_raw.tif")
        b03_a_path = extract_band(boa_a, 3, out_dir / "BOA_B03_A_raw.tif")
        b03_b_path = extract_band(boa_b, 3, out_dir / "BOA_B03_B_raw.tif")
    else:
        log.info("Using L2A BOA TIFFs directly (ACOLITE not installed)")

    # Step 2: Physics — delegate entirely to run_predictor()
    log.info("Running run_predictor() for both images...")
    meta_a = {
        "date": METADATA["A"]["date"],
        "solar_zenith_deg": METADATA["A"]["sza"],
        "solar_azimuth_deg": METADATA["A"]["saa"],
        "cloud_cover_pct": METADATA["A"]["cloud"],
    }
    meta_b = {
        "date": METADATA["B"]["date"],
        "solar_zenith_deg": METADATA["B"]["sza"],
        "solar_azimuth_deg": METADATA["B"]["saa"],
        "cloud_cover_pct": METADATA["B"]["cloud"],
    }

    pred_a = run_predictor(
        boa_b02_path=str(b02_a_path), metadata=meta_a, output_dir=str(out_dir / "pred_A"),
        date=METADATA["A"]["date"], b03_path=str(b03_a_path),
        lat=target_lat, lon=target_lon, depth_target=depth,
        with_bathy_features=HAS_IH_BATHY,
    )
    pred_b = run_predictor(
        boa_b02_path=str(b02_b_path), metadata=meta_b, output_dir=str(out_dir / "pred_B"),
        date=METADATA["B"]["date"], b03_path=str(b03_b_path),
        lat=target_lat, lon=target_lon, depth_target=depth,
        with_bathy_features=HAS_IH_BATHY,
    )

    res_a = _normalise_result(pred_a, METADATA["A"])
    res_b = _normalise_result(pred_b, METADATA["B"])

    # Apply Sentinel-1 roughness penalty (best-effort, uses existing S1 integration)
    for key, res, meta in [("A", res_a, METADATA["A"]), ("B", res_b, METADATA["B"])]:
        try:
            from scratch.fetch_sentinel1_sar import search_stac_s1_scenes, extract_sigma0_at_point, roughness_from_sigma0
            from datetime import timedelta
            t_date = datetime.strptime(meta["date"], "%Y-%m-%d")
            start_dt = t_date - timedelta(days=3)
            end_dt   = t_date + timedelta(days=3)
            s1_items = search_stac_s1_scenes(
                lon=target_lon, lat=target_lat,
                year=t_date.year,
                month_start=start_dt.month,
                month_end=end_dt.month,
                max_results=10,
            )
            valid = [i for i in s1_items
                     if start_dt <= datetime.strptime(i["date"], "%Y-%m-%d") <= end_dt]
            if valid:
                valid.sort(key=lambda x: abs((datetime.strptime(x["date"], "%Y-%m-%d") - t_date).total_seconds()))
                best = valid[0]
                sigma0 = extract_sigma0_at_point(best, target_lon, target_lat)
                if sigma0 and sigma0.get("vv") is not None and sigma0.get("vh") is not None:
                    r_data = roughness_from_sigma0(sigma0["vv"], sigma0["vh"])
                    if r_data.get("roughness") is not None:
                        roughness = r_data["roughness"]
                        penalty_factor = float(np.clip((roughness - 0.05) / 0.20, 0.0, 1.0)) * 0.3
                        res["s1_scene_id"]    = best["id"]
                        res["s1_scene_date"]  = best["date"]
                        res["s1_roughness"]   = roughness
                        res["s1_sea_state"]   = r_data["sea_state"]
                        res["s1_penalty_pct"] = round(penalty_factor * 100.0, 2)
                        res["visibility_score"] = round(
                            res.get("visibility_score", 0) * (1.0 - penalty_factor), 4)
                        log.info("[S1 %s] roughness=%.4f → penalty=%.2f%%", key, roughness, res["s1_penalty_pct"])
        except Exception as e:
            log.warning("Sentinel-1 penalty skipped for image %s: %s", key, e)

    # Apply coastal terrain exposure modifier (best-effort, never blocks pipeline)
    if HAS_TERRAIN:
        try:
            from src.ranking_model import terrain_exposure_modifier
            import math as _math
            # Nearest site by Euclidean distance in degrees (fast, ~1 km precision)
            df = _TERRAIN_DF.copy()
            df["_dist"] = (df["latitude"] - target_lat)**2 + (df["longitude"] - target_lon)**2
            row = df.loc[df["_dist"].idxmin()]
            terrain_feat = {
                "slope_mean":  float(np.nan_to_num(row.get("slope_mean") or 0.0, nan=0.0)),
                "aspect_mean": float(np.nan_to_num(row.get("aspect_mean") or 180.0, nan=180.0)),
            }
            mod = terrain_exposure_modifier(
                terrain_feat["slope_mean"], terrain_feat["aspect_mean"]
            )
            for key, res in [("A", res_a), ("B", res_b)]:
                original = res.get("visibility_score", 0.0)
                res["visibility_score"] = round(original * mod, 4)
                res["terrain_site"]     = str(row.get("site_name", "nearest"))
                res["terrain_modifier"] = round(mod, 4)
                res["terrain_slope"]    = terrain_feat["slope_mean"]
                res["terrain_aspect"]   = terrain_feat["aspect_mean"]
            log.info("[Terrain] site=%s slope=%.2f° aspect=%.1f° modifier=%.3f",
                     row.get("site_name"), terrain_feat["slope_mean"],
                     terrain_feat["aspect_mean"], mod)
        except Exception as e:
            log.warning("Terrain modifier skipped: %s", e)

    results = {"A": res_a, "B": res_b}

    # Step 3: ML ranking (already computed inside run_predictor via predict_score)
    score_a = res_a.get("visibility_score", 0.0)
    score_b = res_b.get("visibility_score", 0.0)

    winner = "A" if score_a >= score_b else "B"
    loser  = "B" if winner == "A" else "A"
    loser_snr = results[loser]["SNR_mean_16m"]
    snr_diff = ((results[winner]["SNR_mean_16m"] - loser_snr) / loser_snr * 100) if loser_snr > 0 else float("inf")

    warnings = []
    if res_a.get("kd_high_uncertainty"): warnings.append("Kd high uncertainty in Image A")
    if res_b.get("kd_high_uncertainty"): warnings.append("Kd high uncertainty in Image B")

    # Step 4: Save BOA copies + maps
    maps_a = save_boa_copy(b02_a_path, b03_a_path, out_dir, "A_20250925")
    maps_b = save_boa_copy(b02_b_path, b03_b_path, out_dir, "B_20231001")

    # Step 5: JSON + CSV output
    csv_path = out_dir / "summary_comparison.csv"
    save_csv(results, csv_path)

    report = {
        "chosen_image": results[winner]["date"],
        "scores": {"A": score_a, "B": score_b},
        "metrics": {"A": res_a, "B": res_b},
        "outputs": {
            "boa_b02_a": maps_a["boa_b02"],
            "boa_b02_b": maps_b["boa_b02"],
            "snr_map_a": maps_a["snr_map"],
            "snr_map_b": maps_b["snr_map"],
            "confidence_map_a": maps_a["confidence_map"],
            "confidence_map_b": maps_b["confidence_map"],
            "summary_csv": str(csv_path),
        },
        "justification": _build_justification(winner, loser, results, snr_diff, depth),
        "assumptions": [
            f"n_water=1.333 (Snell refraction)",
            f"depth_target={depth}m",
            f"Kd490_table: Sep/Oct=0.045, Jan/Feb=0.055, Apr/May=0.200, else=0.080",
            f"Sunglint: Hedley linear correction via simulate_acolite_boa() in utils.py",
            f"Cloud threshold={CLOUD_THRESHOLD}%",
            f"Physics engine: reef_ml_predictor_acolite.run_predictor()",
            f"Datum: WGS84/UTM Zone 29N",
            f"ACOLITE: {'used' if use_acolite else 'not installed — L2A BOA used directly'}",
        ],
        "warnings": warnings,
        "training_inputs_reef_ml_predictor": {
            "month_9_glint_penalty": GLINT_PENALTY.get(9, GLINT_PENALTY_DEFAULT),
            "month_10_glint_penalty": GLINT_PENALTY.get(10, GLINT_PENALTY_DEFAULT),
            "kd490_sep_oct": 0.045,
            "depth_target_m": depth,
            "n_water": N_WATER,
        },
    }

    report_path = out_dir / "orchestrator_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    log.info("=== DONE ===")
    log.info("Winner: %s | Score A=%.4f B=%.4f", report["chosen_image"], score_a, score_b)
    log.info("JSON → %s", report_path)
    log.info("CSV  → %s", csv_path)

    # Drift monitoring: batch-end summary + export (shadow mode)
    if HAS_DRIFT_MONITOR:
        try:
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
            batch_id = f"reef-orchestrator-{ts}-depth{int(depth)}"
            drift_log_summary()
            drift_export_file(batch_id=batch_id)
            drift_history_json()
            drift_history_csv()
            drift_export_html()
        except Exception as e:
            log.debug("Drift reporting (non-critical): %s", e)

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reef Benthic Visibility Orchestrator v2")
    parser.add_argument("--depth",  type=float, default=16.0, help="Target depth in metres")
    parser.add_argument("--config", type=str,   default=None, help="Path to YAML config file")
    parser.add_argument("--image-a-b02", type=str, help="Override Image A B02 path")
    parser.add_argument("--image-b-b02", type=str, help="Override Image B B02 path")
    args = parser.parse_args()
    if args.image_a_b02: IMAGE_A_B02 = Path(args.image_a_b02)
    if args.image_b_b02: IMAGE_B_B02 = Path(args.image_b_b02)
    main(depth=args.depth, config_path=args.config)
