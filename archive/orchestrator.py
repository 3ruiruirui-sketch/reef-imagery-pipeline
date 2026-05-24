#!/usr/bin/env python3
"""
orchestrator.py — Main runner for the Reef Benthic Visibility Pipeline.

Usage (simulated ACOLITE mode, no external tools needed):
  python orchestrator.py \\
    --image-a reef_Output_Master/reef_output_pedra_to_gale_20250925/S2_B02_20250925.tif \\
    --image-b reef_Output_Master/reef_output_ai_prediction_spot_2023/S2_B02_20231001.tif \\
    --b03-a   reef_Output_Master/reef_output_pedra_to_gale_20250925/S2_B03_20250925.tif \\
    --b03-b   reef_Output_Master/reef_output_ai_prediction_spot_2023/S2_B03_20231001.tif \\
    --output  reef_output_acolite_comparison

When ACOLITE is installed: replace simulated BOA step by pointing
--image-a/--image-b directly at ACOLITE BOA GeoTIFFs; the rest is identical.
"""

import json, logging, argparse
import numpy as np
from pathlib import Path

from src.utils import simulate_acolite_boa, compute_metadata_stub
from src.reef_ml_predictor_acolite import run_predictor
from archive.comparator import compare_models

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Defaults (pre-filled for this project) ───────────────────────────────────
PROJECT = Path(__file__).parent
DEFAULTS = {
    "image_a": str(PROJECT / "reef_Output_Master/reef_output_pedra_to_gale_20250925/S2_B02_20250925.tif"),
    "image_b": str(PROJECT / "reef_Output_Master/reef_output_ai_prediction_spot_2023/S2_B02_20231001.tif"),
    "b03_a":   str(PROJECT / "reef_Output_Master/reef_output_pedra_to_gale_20250925/S2_B03_20250925.tif"),
    "b03_b":   str(PROJECT / "reef_Output_Master/reef_output_ai_prediction_spot_2023/S2_B03_20231001.tif"),
    "output":  str(PROJECT / "reef_output_acolite_comparison"),
    "date_a":  "2025-09-25",
    "date_b":  "2023-10-01",
}


def main(image_a, image_b, output, date_a, date_b,
         b03_a=None, b03_b=None, b04_a=None, b04_b=None,
         kd_prior: dict | None = None,
         cloud_threshold: float = 0.2,
         snr_threshold: float = 3.0,
         use_real_acolite: bool = False):

    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)

    kd_tbl = kd_prior or {"09": 0.045, "10": 0.045, "01": 0.055, "02": 0.055,
                           "04": 0.200, "05": 0.200}

    # ── Step 1: BOA correction ────────────────────────────────────────────────
    if use_real_acolite:
        # Real ACOLITE: images already BOA GeoTIFFs — pass straight through
        boa_a, boa_b = Path(image_a), Path(image_b)
        log.info("Using real ACOLITE BOA inputs directly.")
    else:
        log.info("Simulating ACOLITE BOA (Hedley sunglint + DN→reflectance)...")
        boa_a = out / f"acolite_sim_A_{date_a.replace('-','')}_B02.tif"
        boa_b = out / f"acolite_sim_B_{date_b.replace('-','')}_B02.tif"
        simulate_acolite_boa(image_a, boa_a, b03_tif=b03_a)
        simulate_acolite_boa(image_b, boa_b, b03_tif=b03_b)
        log.info("BOA A → %s", boa_a)
        log.info("BOA B → %s", boa_b)

    # ── Step 2: Metadata ──────────────────────────────────────────────────────
    meta_a = compute_metadata_stub(date_a)
    meta_b = compute_metadata_stub(date_b)

    # ── Step 3: Predictor (Gordon/QAA Kd + SDB + SNR + confidence) ───────────
    log.info("Running predictor on Image A (%s)...", date_a)
    dir_a = out / f"predictor_A_{date_a.replace('-','')}"
    res_a = run_predictor(
        str(boa_a), meta_a, str(dir_a),
        kd_prior=kd_tbl, cloud_threshold=cloud_threshold,
        snr_threshold=snr_threshold, date=date_a,
        b03_path=b03_a, b04_path=b04_a,
    )

    log.info("Running predictor on Image B (%s)...", date_b)
    dir_b = out / f"predictor_B_{date_b.replace('-','')}"
    res_b = run_predictor(
        str(boa_b), meta_b, str(dir_b),
        kd_prior=kd_tbl, cloud_threshold=cloud_threshold,
        snr_threshold=snr_threshold, date=date_b,
        b03_path=b03_b, b04_path=b04_b,
    )

    # ── Step 4: Compare & summary ─────────────────────────────────────────────
    log.info("Comparing outputs...")
    summary_csv = out / "comparison_summary.csv"
    compare_models(str(dir_a / "summary.csv"), str(dir_b / "summary.csv"), str(summary_csv))

    # ── Step 5: Final JSON report ─────────────────────────────────────────────
    winner = date_a if res_a["visibility_score"] >= res_b["visibility_score"] else date_b
    wr, lr = (res_a, res_b) if winner == date_a else (res_b, res_a)
    snr_diff = (wr["snr_mean_16m"] - lr["snr_mean_16m"]) / (lr["snr_mean_16m"] + 1e-9) * 100

    report = {
        "chosen_image": winner,
        "scores": {"A": res_a["visibility_score"], "B": res_b["visibility_score"]},
        "metrics": {"A": res_a, "B": res_b},
        "outputs": {
            "boa_b02_a": str(boa_a), "boa_b02_b": str(boa_b),
            "snr_map_a": res_a["snr_map"], "snr_map_b": res_b["snr_map"],
            "confidence_map_a": res_a["confidence_map"], "confidence_map_b": res_b["confidence_map"],
            "sdb_map_a": res_a.get("sdb_depth_map"), "sdb_map_b": res_b.get("sdb_depth_map"),
            "summary_csv": str(summary_csv),
        },
        "justification": (
            f"Image {winner}: score={wr['visibility_score']:.4f}, SNR={wr['snr_mean_16m']:.2f}, "
            f"Kd_B02={wr['kd_b02_estimated']:.4f} ({wr['kd_estimation_method']}). "
            f"vs rejected: score={lr['visibility_score']:.4f}, SNR={lr['snr_mean_16m']:.2f}. "
            f"SNR diff={snr_diff:+.0f}%, glint_penalty={wr['glint_penalty']}."
        ),
        "assumptions": [
            f"depth_target={DEFAULTS.get('depth', 16.0)}m",
            "n_water=1.333", f"cloud_threshold={cloud_threshold}%",
            f"snr_threshold={snr_threshold}",
            f"Kd method: Gordon/QAA (B02+B03) or band-ratio fallback",
            "SDB: Stumpf log-ratio (m0=-16, m1=20, n=1000) — calibrate with in-situ for production",
            "sunglint: Hedley linear correction via B03 deep-water regression",
            f"acolite_real={'yes' if use_real_acolite else 'no — simulated from L2A'}",
        ],
        "warnings": (
            ["Kd high uncertainty in A"] if res_a["kd_high_uncertainty"] else []
        ) + (
            ["Kd high uncertainty in B"] if res_b["kd_high_uncertainty"] else []
        ),
        "training_inputs_reef_ml_predictor": {
            "month_9_glint_penalty": 0.95, "month_10_glint_penalty": 0.60,
            "kd490_sep_oct": 0.045, "depth_target_m": 16.0, "n_water": 1.333,
        },
    }

    json_out = out / "orchestrator_report.json"
    with open(json_out, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False,
                  default=lambda o: bool(o) if isinstance(o, (bool, np.bool_)) else str(o))

    log.info("=== DONE ===")
    log.info("Winner: %s | Score A=%.4f  B=%.4f", winner, res_a["visibility_score"], res_b["visibility_score"])
    log.info("Report  → %s", json_out)
    log.info("CSV     → %s", summary_csv)
    return report


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Reef Benthic Visibility Orchestrator")
    p.add_argument("--image-a",  default=DEFAULTS["image_a"])
    p.add_argument("--image-b",  default=DEFAULTS["image_b"])
    p.add_argument("--b03-a",    default=DEFAULTS["b03_a"])
    p.add_argument("--b03-b",    default=DEFAULTS["b03_b"])
    p.add_argument("--b04-a",    default=None)
    p.add_argument("--b04-b",    default=None)
    p.add_argument("--output",   default=DEFAULTS["output"])
    p.add_argument("--date-a",   default=DEFAULTS["date_a"])
    p.add_argument("--date-b",   default=DEFAULTS["date_b"])
    p.add_argument("--kd-prior", default=None, help='JSON e.g. \'{"09":0.045,"04":0.200}\'')
    p.add_argument("--cloud-threshold", type=float, default=0.2)
    p.add_argument("--snr-threshold",   type=float, default=3.0)
    p.add_argument("--real-acolite",    action="store_true",
                   help="Images are already ACOLITE BOA outputs — skip simulation")
    args = p.parse_args()
    kd_prior = json.loads(args.kd_prior) if args.kd_prior else None
    main(
        args.image_a, args.image_b, args.output, args.date_a, args.date_b,
        b03_a=args.b03_a, b03_b=args.b03_b, b04_a=args.b04_a, b04_b=args.b04_b,
        kd_prior=kd_prior, cloud_threshold=args.cloud_threshold,
        snr_threshold=args.snr_threshold, use_real_acolite=args.real_acolite,
    )
