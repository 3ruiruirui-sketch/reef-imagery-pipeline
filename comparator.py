#!/usr/bin/env python3
"""
Comparator: compares predictor output CSVs and writes combined summary.
"""
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def compare_models(summary_csv_a: str, summary_csv_b: str, out_csv: str) -> str:
    a = pd.read_csv(summary_csv_a)
    b = pd.read_csv(summary_csv_b)

    a = a.add_prefix("A_")
    b = b.add_prefix("B_")
    merged = pd.concat([a.reset_index(drop=True), b.reset_index(drop=True)], axis=1)

    merged["delta_visibility_score"]   = merged["A_visibility_score"]      - merged["B_visibility_score"]
    merged["delta_snr_mean_16m"]       = merged["A_snr_mean_16m"]          - merged["B_snr_mean_16m"]
    merged["delta_percent_useful"]     = merged["A_percent_pixels_useful"]  - merged["B_percent_pixels_useful"]
    merged["chosen_image"]             = merged.apply(
        lambda r: r["A_image_date"] if r["A_visibility_score"] >= r["B_visibility_score"] else r["B_image_date"],
        axis=1
    )
    merged["justification"] = merged.apply(
        lambda r: (
            f"Image A ({r['A_image_date']}): score={r['A_visibility_score']:.4f}, "
            f"SNR={r['A_snr_mean_16m']:.2f}  vs  "
            f"Image B ({r['B_image_date']}): score={r['B_visibility_score']:.4f}, "
            f"SNR={r['B_snr_mean_16m']:.2f}. "
            f"Delta score={r['delta_visibility_score']:+.4f}, "
            f"delta SNR={r['delta_snr_mean_16m']:+.2f}."
        ), axis=1
    )

    merged.to_csv(out_csv, index=False)
    logging.info("Comparison CSV saved to %s", out_csv)
    return out_csv


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 4:
        compare_models(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        print("Usage: comparator.py <summary_a.csv> <summary_b.csv> <out.csv>")
