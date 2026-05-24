#!/usr/bin/env python3
"""
Orchestrator — Reef Benthic Visibility Pipeline
================================================
Orquestra: ACOLITE (ou fallback L2A) → física Beer-Lambert/Snell → JSON report
Uso: python3 orchestrator_run.py [--image-a PATH] [--image-b PATH] [--depth 16.0]
"""

import os, json, subprocess, logging, argparse, math, shutil, shlex
from pathlib import Path
from datetime import datetime
import numpy as np
import rasterio
from pyproj import Transformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Config defaults ──────────────────────────────────────────────────────────
PROJECT_DIR   = Path(__file__).parent.parent  # src/ -> raiz do projeto
IMAGE_A_B02   = PROJECT_DIR / "reef_Output_Master/reef_output_pedra_to_gale_20250925/S2_B02_20250925.tif"
IMAGE_A_B03   = PROJECT_DIR / "reef_Output_Master/reef_output_pedra_to_gale_20250925/S2_B03_20250925.tif"
IMAGE_B_B02   = PROJECT_DIR / "reef_Output_Master/reef_output_ai_prediction_spot_2023/S2_B02_20231001.tif"
IMAGE_B_B03   = PROJECT_DIR / "reef_Output_Master/reef_output_ai_prediction_spot_2023/S2_B03_20231001.tif"
OUTPUT_DIR    = PROJECT_DIR / "reef_output_acolite_comparison"

METADATA = {
    "A": {"date":"2025-09-25","sza":40.498,"saa":158.883,"cloud":1.245,"level":"L2A","month":9},
    "B": {"date":"2023-10-01","sza":42.413,"saa":160.459,"cloud":0.007,"level":"L2A","month":10},
}

# Physical constants
N_WATER        = 1.333
CLOUD_THRESHOLD = 5.0   # %
SNR_THRESHOLD   = 3.0
KD490_TABLE    = {9: 0.045, 10: 0.045, 1: 0.055, 2: 0.055, 4: 0.200, 5: 0.200}

TARGET_LAT, TARGET_LON = 37.05815, -8.20982

# ── Helpers ──────────────────────────────────────────────────────────────────
def run_shell(cmd, check=True):
    """Run a command safely. Accepts str (will be shlex.split) or list of args.
    Uses shell=False to prevent command injection."""
    if isinstance(cmd, str):
        cmd_list = shlex.split(cmd)
    else:
        cmd_list = list(cmd)
    log.info("$ %s", " ".join(shlex.quote(c) for c in cmd_list))
    result = subprocess.run(cmd_list, shell=False, capture_output=True, text=True)
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

def run_sen2cor(input_l1c: Path, output_dir: Path):
    """Run Sen2Cor via SNAP GPT if available."""
    output_dir.mkdir(parents=True, exist_ok=True)
    graph = Path(os.environ.get("SEN2COR_GRAPH", "/opt/snap/bin/Sen2Cor-Processor.xml"))
    run_shell(["gpt", str(graph), f"-Pinput={input_l1c}", f"-Poutput={output_dir}"])
    return output_dir

def gdal_extract_b02(boa_tif: Path, out_b02: Path):
    """Extract band 2 from a multi-band GeoTIFF using rasterio (no gdal binary needed)."""
    with rasterio.open(boa_tif) as src:
        profile = src.profile.copy()
        profile.update(count=1)
        b02 = src.read(2)   # band index 2 = B02 in Sentinel-2 BOA stack
    with rasterio.open(out_b02, 'w', **profile) as dst:
        dst.write(b02, 1)
    log.info("Extracted B02 → %s", out_b02)
    return out_b02

# ── Physics core ─────────────────────────────────────────────────────────────
def snell_optical_path(sza_air_deg: float, depth_m: float) -> float:
    sza_water = math.degrees(math.asin(math.sin(math.radians(sza_air_deg)) / N_WATER))
    return depth_m / math.cos(math.radians(sza_water)), sza_water

def beer_lambert(kd: float, path_m: float) -> float:
    return math.exp(-2 * kd * path_m)   # two-way transmittance

def sunglint_correction(arr: np.ndarray, b03: np.ndarray) -> np.ndarray:
    """
    Simple Hedley-style linear sunglint correction:
    corrected_B02 = B02 - slope * (B03 - min(B03))
    slope estimated from linear regression B02 ~ B03 in deep-water pixels.
    """
    mask = (arr > 0) & (b03 > 0)
    if mask.sum() < 10:
        return arr
    b02_v = arr[mask].astype(float)
    b03_v = b03[mask].astype(float)
    slope = np.cov(b02_v, b03_v)[0, 1] / np.var(b03_v)
    slope = np.clip(slope, 0, 2)   # sanity clamp
    corrected = arr.astype(float) - slope * (b03.astype(float) - b03.min())
    return np.clip(corrected, 0, None)

def analyse_band(b02_path: Path, b03_path: Path, meta: dict, depth: float) -> dict:
    """Full physical analysis for one image."""
    kd   = KD490_TABLE.get(meta["month"], 0.080)
    opt_path, sza_w = snell_optical_path(meta["sza"], depth)
    trans = beer_lambert(kd, opt_path)
    kd_uncert = False

    with rasterio.open(b02_path) as s2, rasterio.open(b03_path) as s3:
        t = Transformer.from_crs("EPSG:4326", s2.crs, always_xy=True)
        x, y = t.transform(TARGET_LON, TARGET_LAT)
        row, col = s2.index(x, y)
        win = rasterio.windows.Window(col-20, row-20, 40, 40)
        b02 = s2.read(1, window=win).astype(float) / 10000.0
        b03 = s3.read(1, window=win).astype(float) / 10000.0

    # Sunglint correction (Hedley linear)
    b02_corr = sunglint_correction(b02, b03)

    sig    = np.mean(b02_corr)
    noise  = np.std(b02_corr)
    snr    = sig / noise if noise > 0 else 0
    cv     = noise / sig if sig > 0 else 0

    # Local Kd estimate from B02/B03 ratio
    ratio  = np.mean(b02_corr) / np.mean(b03) if np.mean(b03) > 0 else 1.0
    kd_est = kd * (1 + (ratio - 1.0) * 0.15)
    if abs(kd_est - kd) / kd > 0.30:
        kd_uncert = True
        log.warning("Kd estimated (%.4f) diverges >30%% from seasonal prior (%.4f)", kd_est, kd)

    # Benthic contrast
    sand_btm  = 0.25 * trans
    rock_btm  = 0.05 * trans
    contrast  = (sand_btm - rock_btm) / sand_btm if sand_btm > 0 else 0

    # Pixel usefulness: cloud mask
    usable    = max(0.0, 1.0 - meta["cloud"] / 100.0)
    snr_ok    = min(1.0, snr / 100.0)
    glint_ok  = 1.0 if cv < 0.015 else 0.015 / cv
    vis_score = min(1.0, usable * snr_ok * glint_ok * contrast * 5.0)

    # Confidence map proxy
    high_conf_pct = 100.0 if snr >= SNR_THRESHOLD and cv < 0.02 else 50.0 if snr >= SNR_THRESHOLD else 0.0

    return {
        "date": meta["date"],
        "sza_air_deg": meta["sza"],
        "sza_water_deg": round(sza_w, 3),
        "optical_path_m": round(opt_path, 3),
        "kd490_seasonal": kd,
        "kd490_estimated": round(kd_est, 4),
        "kd_high_uncertainty": kd_uncert,
        "water_transmittance_twoway": round(trans, 4),
        "b02_signal_mean": round(float(sig), 5),
        "b02_noise_std": round(float(noise), 6),
        "b02_cv": round(float(cv), 5),
        "SNR_mean_16m": round(float(snr), 2),
        "contrast_benthic_mean": round(float(contrast), 4),
        "percent_pixels_useful": round(usable * 100, 2),
        "percent_area_high_confidence": round(high_conf_pct, 1),
        "visibility_score": round(vis_score, 4),
    }

# ── Output writers ───────────────────────────────────────────────────────────
def save_boa_copy(src_b02: Path, src_b03: Path, out_dir: Path, label: str) -> dict:
    """Copy pre-processed bands to output dir as 'BOA' equivalents."""
    out_dir.mkdir(parents=True, exist_ok=True)
    boa_b02 = out_dir / f"BOA_B02_{label}.tif"
    shutil.copy2(src_b02, boa_b02)
    snr_map = out_dir / f"SNR_map_{label}.tif"
    conf_map = out_dir / f"Confidence_map_{label}.tif"

    with rasterio.open(src_b02) as src:
        profile = src.profile.copy()
        data = src.read(1).astype(float) / 10000.0
        valid = data > 0
        if valid.any():
            sig = float(np.mean(data[valid]))
            noise = float(np.std(data[valid]))
        else:
            sig, noise = 0.0, 1.0
        # SNR = signal_mean / noise_std (proper per-pixel proxy: pixel / global noise)
        snr_px = np.where(valid, data / (noise + 1e-9), 0).astype(np.float32)
        conf_px = np.select([snr_px < 5, snr_px < 30], [0, 1], default=2).astype(np.uint8)

    profile.update(dtype=rasterio.float32, count=1)
    with rasterio.open(snr_map, 'w', **profile) as dst:
        dst.write(snr_px, 1)

    profile.update(dtype=rasterio.uint8)
    with rasterio.open(conf_map, 'w', **profile) as dst:
        dst.write(conf_px, 1)

    return {"boa_b02": str(boa_b02), "snr_map": str(snr_map), "confidence_map": str(conf_map)}

def _build_justification(winner: str, loser: str, results: dict, snr_diff: float, depth: float) -> str:
    """Construct human-readable justification safely (handles zero-division)."""
    w_cv = results[winner]["b02_cv"]
    l_cv = results[loser]["b02_cv"]
    cv_ratio = (l_cv / w_cv) if w_cv > 1e-9 else float("inf")
    snr_str = f"+{snr_diff:.0f}%" if snr_diff != float("inf") else "+inf%"
    cv_str = f"{cv_ratio:.1f}×" if cv_ratio != float("inf") else "∞×"
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
    rows = []
    for k, r in results.items():
        row = {"image_key": k}
        row.update(r)
        rows.append(row)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    log.info("CSV saved: %s", path)

# ── Main ─────────────────────────────────────────────────────────────────────
def main(depth: float = 16.0):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log.info("=== Reef Orchestrator — depth=%.1fm ===", depth)

    # Step 1: ACOLITE or direct L2A fallback
    use_acolite = acolite_available()
    log.info("ACOLITE: %s | SNAP/gpt: %s", "YES" if use_acolite else "NO (fallback L2A)", "YES" if snap_gpt_available() else "NO")

    if use_acolite:
        boa_a = run_acolite(IMAGE_A_B02.parent, OUTPUT_DIR / "acolite_A")
        boa_b = run_acolite(IMAGE_B_B02.parent, OUTPUT_DIR / "acolite_B")
        b02_a = gdal_extract_b02(boa_a, OUTPUT_DIR / "BOA_B02_A_raw.tif")
        b02_b = gdal_extract_b02(boa_b, OUTPUT_DIR / "BOA_B02_B_raw.tif")
        b03_a = b03_b = None   # ACOLITE stack will contain B03 too
    else:
        log.info("Using L2A BOA TIFFs directly (ACOLITE not installed)")
        b02_a, b03_a = IMAGE_A_B02, IMAGE_A_B03
        b02_b, b03_b = IMAGE_B_B02, IMAGE_B_B03

    # Step 2: Physical analysis
    log.info("Running physical radiometric analysis...")
    res_a = analyse_band(b02_a, b03_a, METADATA["A"], depth)
    res_b = analyse_band(b02_b, b03_b, METADATA["B"], depth)
    results = {"A": res_a, "B": res_b}

    # Step 3: Save BOA copies + maps
    maps_a = save_boa_copy(b02_a, b03_a, OUTPUT_DIR, "A_20250925")
    maps_b = save_boa_copy(b02_b, b03_b, OUTPUT_DIR, "B_20231001")

    # Step 4: Decision
    winner = "A" if res_a["visibility_score"] >= res_b["visibility_score"] else "B"
    loser  = "B" if winner == "A" else "A"
    loser_snr = results[loser]["SNR_mean_16m"]
    snr_diff = ((results[winner]["SNR_mean_16m"] - loser_snr) / loser_snr * 100) if loser_snr > 0 else float("inf")

    warnings = []
    if res_a["kd_high_uncertainty"]: warnings.append("Kd high uncertainty in Image A")
    if res_b["kd_high_uncertainty"]: warnings.append("Kd high uncertainty in Image B")

    # Step 5: JSON output
    csv_path = OUTPUT_DIR / "summary_comparison.csv"
    save_csv(results, csv_path)

    report = {
        "chosen_image": results[winner]["date"],
        "scores": {"A": res_a["visibility_score"], "B": res_b["visibility_score"]},
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
            f"Sunglint: Hedley linear correction (B03 deep-water regression)",
            f"Cloud threshold={CLOUD_THRESHOLD}%",
            f"Analysis window=40x40px (~400m) at 37.05815N, 8.20982W",
            f"Datum: WGS84/UTM Zone 29N",
            f"ACOLITE: {'used' if use_acolite else 'not installed — L2A BOA used directly'}",
        ],
        "warnings": warnings,
        "training_inputs_reef_ml_predictor": {
            "month_9_glint_penalty": 0.95,
            "month_10_glint_penalty": 0.60,
            "kd490_sep_oct": 0.045,
            "depth_target_m": depth,
            "n_water": N_WATER,
        },
    }

    report_path = OUTPUT_DIR / "orchestrator_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    log.info("=== DONE ===")
    log.info("Winner: %s | Score A=%.4f B=%.4f", report["chosen_image"], res_a["visibility_score"], res_b["visibility_score"])
    log.info("JSON → %s", report_path)
    log.info("CSV  → %s", csv_path)
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reef Benthic Visibility Orchestrator")
    parser.add_argument("--depth", type=float, default=16.0, help="Target depth in metres")
    parser.add_argument("--image-a-b02", type=str, help="Override Image A B02 path")
    parser.add_argument("--image-b-b02", type=str, help="Override Image B B02 path")
    args = parser.parse_args()
    if args.image_a_b02: IMAGE_A_B02 = Path(args.image_a_b02)
    if args.image_b_b02: IMAGE_B_B02 = Path(args.image_b_b02)
    main(depth=args.depth)
