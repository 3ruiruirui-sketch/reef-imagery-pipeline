#!/usr/bin/env python3
"""
threshold_sweep.py — Precision/recall/F1 sweep over a binarisation threshold.

Given a directory of predicted-probability tiles and matching ground-truth
masks, sweep the decision threshold from 0.3 to 0.7 in steps of 0.05, compute
precision/recall/F1 at each threshold, and write:

  * a CSV  (one row per threshold)
  * a PNG  (precision/recall/F1 curves) — only if matplotlib is available

Tiles and masks are read as ``.npy`` arrays. A prediction file ``foo.npy`` is
paired with a mask of the same stem in ``--masks-dir`` (or a sibling
``foo_mask.npy`` if ``--masks-dir`` is omitted). Predictions are floats in
[0, 1]; masks are treated as boolean (``> 0`` is positive).

Usage::

    python scripts/threshold_sweep.py --pred-dir preds/ --masks-dir masks/ \
        --out-csv sweep.csv --out-png sweep.png

    # No data on hand? Generate a synthetic prediction/mask pair and sweep it:
    python scripts/threshold_sweep.py --demo --out-csv sweep_demo.csv

Dependency-light: numpy only is required; matplotlib is optional (guarded).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

# Threshold grid: 0.30, 0.35, ..., 0.70 (inclusive).
THRESHOLDS = [round(0.30 + 0.05 * i, 2) for i in range(9)]

CSV_COLUMNS = ["threshold", "tp", "fp", "fn", "tn", "precision", "recall", "f1"]


def _confusion_counts(probs: np.ndarray, truth: np.ndarray, thr: float):
    """Return (tp, fp, fn, tn) for a single threshold."""
    pred = probs >= thr
    pos = truth > 0
    tp = int(np.sum(pred & pos))
    fp = int(np.sum(pred & ~pos))
    fn = int(np.sum(~pred & pos))
    tn = int(np.sum(~pred & ~pos))
    return tp, fp, fn, tn


def _prf(tp: int, fp: int, fn: int):
    """Precision, recall, F1 with zero-division guards."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def sweep(probs: np.ndarray, truth: np.ndarray, thresholds=THRESHOLDS):
    """Sweep thresholds; return a list of per-threshold metric dicts."""
    probs = np.asarray(probs, dtype=float).ravel()
    truth = np.asarray(truth).ravel()
    rows = []
    for thr in thresholds:
        tp, fp, fn, tn = _confusion_counts(probs, truth, thr)
        precision, recall, f1 = _prf(tp, fp, fn)
        rows.append(
            {
                "threshold": thr,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
            }
        )
    return rows


def _load_pairs(pred_dir: Path, masks_dir: Path | None):
    """Yield (probs, truth) array pairs from a directory of .npy tiles."""
    pairs = []
    for pred_path in sorted(pred_dir.glob("*.npy")):
        if masks_dir is not None:
            mask_path = masks_dir / pred_path.name
        else:
            mask_path = pred_path.with_name(pred_path.stem + "_mask.npy")
        if not mask_path.exists():
            print(f"  skip (no mask): {pred_path.name}")
            continue
        probs = np.load(pred_path)
        truth = np.load(mask_path)
        if probs.shape != truth.shape:
            print(f"  skip (shape mismatch): {pred_path.name}")
            continue
        pairs.append((probs, truth))
    return pairs


def _make_demo_pair(seed: int = 0):
    """Synthetic prediction/mask pair: a blobby mask with calibrated-ish probs."""
    rng = np.random.default_rng(seed)
    n = 64
    yy, xx = np.mgrid[0:n, 0:n]
    # Two circular "reef" blobs as ground truth.
    truth = (((xx - 20) ** 2 + (yy - 20) ** 2) < 100) | (((xx - 45) ** 2 + (yy - 45) ** 2) < 64)
    truth = truth.astype(np.uint8)
    # Probabilities: high inside truth, low outside, plus noise — clipped to [0, 1].
    probs = np.where(truth > 0, 0.75, 0.25) + rng.normal(0, 0.12, size=(n, n))
    probs = np.clip(probs, 0.0, 1.0)
    return probs.astype(np.float32), truth


def _write_csv(rows, out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"Wrote CSV: {out_csv}")


def _write_png(rows, out_png: Path) -> bool:
    """Plot precision/recall/F1 curves. Returns False if matplotlib missing."""
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless / offline
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover - optional dep
        print(f"matplotlib unavailable — skipping PNG ({e})")
        return False

    thr = [r["threshold"] for r in rows]
    plt.figure(figsize=(7, 4.5))
    plt.plot(thr, [r["precision"] for r in rows], "-o", label="precision")
    plt.plot(thr, [r["recall"] for r in rows], "-s", label="recall")
    plt.plot(thr, [r["f1"] for r in rows], "-^", label="F1")
    plt.xlabel("threshold")
    plt.ylabel("score")
    plt.title("Threshold sweep")
    plt.ylim(-0.02, 1.02)
    plt.grid(True, alpha=0.3)
    plt.legend()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=120)
    plt.close()
    print(f"Wrote PNG: {out_png}")
    return True


def run_sweep(pairs, out_csv: Path, out_png: Path | None = None):
    """Aggregate confusion counts across all pairs, sweep, and write outputs."""
    if not pairs:
        raise SystemExit("No prediction/mask pairs found — nothing to sweep.")

    # Concatenate all pixels so metrics are computed over the full corpus.
    probs = np.concatenate([np.asarray(p, dtype=float).ravel() for p, _ in pairs])
    truth = np.concatenate([np.asarray(t).ravel() for _, t in pairs])

    rows = sweep(probs, truth)
    _write_csv(rows, out_csv)
    if out_png is not None:
        _write_png(rows, out_png)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--pred-dir", help="Directory of predicted-probability .npy tiles")
    ap.add_argument("--masks-dir", help="Directory of ground-truth mask .npy tiles")
    ap.add_argument("--demo", action="store_true", help="Use a synthetic pair (ignores --pred-dir)")
    ap.add_argument("--out-csv", default="threshold_sweep.csv", help="Output CSV path")
    ap.add_argument("--out-png", default="threshold_sweep.png", help="Output PNG path")
    ap.add_argument("--no-png", action="store_true", help="Skip PNG generation")
    args = ap.parse_args()

    if args.demo:
        print("Demo mode — generating one synthetic prediction/mask pair.")
        pairs = [_make_demo_pair()]
    else:
        if not args.pred_dir:
            raise SystemExit("--pred-dir is required (or pass --demo).")
        pred_dir = Path(args.pred_dir)
        masks_dir = Path(args.masks_dir) if args.masks_dir else None
        pairs = _load_pairs(pred_dir, masks_dir)

    out_png = None if args.no_png else Path(args.out_png)
    rows = run_sweep(pairs, Path(args.out_csv), out_png)

    best = max(rows, key=lambda r: r["f1"])
    print(f"Best F1={best['f1']:.4f} at threshold={best['threshold']}")


if __name__ == "__main__":
    main()
