#!/usr/bin/env python3
"""Real threshold sweep for the trained reef U-Net on the local labelled dataset.

Runs the trained `ReefUNetCPU` (smp Unet, resnet18 encoder, 1-channel bathymetry)
over every patch in ``data/master_ml_dataset/{images,masks}``, then sweeps the
decision threshold and reports pixel-level precision / recall / F1 — the
operating-point analysis the static ``PipelineConfig.threshold`` was missing.

This is the *real* counterpart to ``scripts/threshold_sweep.py`` (which only does
``--demo`` synthetic data). It needs torch + segmentation-models-pytorch, so run
it in your project ``.venv`` (not the cloud sandbox):

    source .venv/bin/activate
    python scripts/sweep_unet_local.py

Outputs (under ``outputs/threshold_sweep/``):
    - unet_threshold_sweep.csv   one row per threshold
    - unet_threshold_sweep.png   precision / recall / F1 curves

Notes
-----
* The encoder is forced to ``resnet18`` to match the saved weights — the default
  ``load_model()`` picks mobilenet and would raise a state_dict-mismatch error.
* Input normalization reuses ``src.reef_segmenter.normalize_depth`` (the same
  [0,1] depth scaling, window -40..0 m, used during training).
* The dataset is heavily imbalanced (~1.2% reef pixels), so accuracy is
  meaningless here — F1 / PR is the right lens.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

WEIGHTS_DEFAULT = _PROJECT_ROOT / "models" / "unet_reef_best_smp_resnet18.pth"
IMAGES_DEFAULT = _PROJECT_ROOT / "data" / "master_ml_dataset" / "images"
MASKS_DEFAULT = _PROJECT_ROOT / "data" / "master_ml_dataset" / "masks"
OUT_DIR_DEFAULT = _PROJECT_ROOT / "outputs" / "threshold_sweep"


def _load_unet(weights_path: Path):
    """Load the reef model via reef_segmenter's architecture-aware loader.

    Returns (model, applies_sigmoid). This auto-matches the checkpoint to the
    canonical ReefUNet or the vanilla smp.Unet(resnet18) fallback, so it loads
    whichever weights are present without an architecture mismatch.
    """
    from src.reef_segmenter import _build_and_load

    return _build_and_load(weights_path)


def _collect_probs_and_truth(model, applies_sigmoid, images_dir: Path, masks_dir: Path):
    """Run inference over every patch; return flat (probs, truth) arrays."""
    import rasterio
    import torch

    from src.reef_segmenter import normalize_depth

    image_files = sorted(glob.glob(os.path.join(str(images_dir), "*.tif")))
    if not image_files:
        raise FileNotFoundError(f"No image patches in {images_dir}")

    all_probs: list[np.ndarray] = []
    all_truth: list[np.ndarray] = []
    n_used = 0
    with torch.no_grad():
        for img_path in image_files:
            mask_path = os.path.join(str(masks_dir), os.path.basename(img_path))
            if not os.path.exists(mask_path):
                continue
            with rasterio.open(img_path) as s:
                depth = s.read(1).astype(np.float32)
            with rasterio.open(mask_path) as s:
                mask = s.read(1)

            norm = normalize_depth(depth)  # [0,1], same window as training
            x = torch.from_numpy(norm).float()[None, None, :, :]  # (1,1,H,W)
            out = model(x)
            if applies_sigmoid:  # smp.Unet returns logits; ReefUNet already sigmoids
                out = torch.sigmoid(out)
            probs = out[0, 0].cpu().numpy()

            all_probs.append(probs.ravel())
            all_truth.append((mask.ravel() > 0).astype(np.uint8))
            n_used += 1

    print(f"Ran inference on {n_used} labelled patches")
    return np.concatenate(all_probs), np.concatenate(all_truth)


def _sweep(probs: np.ndarray, truth: np.ndarray, lo=0.30, hi=0.70, step=0.05):
    rows = []
    pos = truth.astype(bool)
    t = lo
    while t <= hi + 1e-9:
        pred = probs >= t
        tp = int(np.sum(pred & pos))
        fp = int(np.sum(pred & ~pos))
        fn = int(np.sum(~pred & pos))
        tn = int(np.sum(~pred & ~pos))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        rows.append(
            {
                "threshold": round(t, 4),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
            }
        )
        t += step
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", type=Path, default=WEIGHTS_DEFAULT)
    ap.add_argument("--images-dir", type=Path, default=IMAGES_DEFAULT)
    ap.add_argument("--masks-dir", type=Path, default=MASKS_DEFAULT)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR_DEFAULT)
    ap.add_argument("--no-png", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "unet_threshold_sweep.csv"
    png_path = args.out_dir / "unet_threshold_sweep.png"

    model, applies_sigmoid = _load_unet(args.weights)
    probs, truth = _collect_probs_and_truth(model, applies_sigmoid, args.images_dir, args.masks_dir)
    print(f"Total pixels: {truth.size:,} | reef pixels: {int(truth.sum()):,} "
          f"({100 * truth.mean():.2f}%)")

    rows = _sweep(probs, truth)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("Wrote", csv_path)

    best = max(rows, key=lambda r: r["f1"])
    print(f"Best F1={best['f1']} at threshold={best['threshold']} "
          f"(P={best['precision']}, R={best['recall']})")

    if not args.no_png:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            th = [r["threshold"] for r in rows]
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(th, [r["precision"] for r in rows], "-o", label="precision")
            ax.plot(th, [r["recall"] for r in rows], "-o", label="recall")
            ax.plot(th, [r["f1"] for r in rows], "-o", label="F1")
            ax.axvline(best["threshold"], ls="--", c="k", lw=1,
                       label=f"best F1 @ {best['threshold']}")
            ax.set_xlabel("threshold")
            ax.set_ylabel("score")
            ax.set_title("Reef U-Net threshold sweep (local labelled patches)")
            ax.legend()
            ax.grid(alpha=0.3)
            plt.tight_layout()
            fig.savefig(png_path, dpi=130)
            print("Wrote", png_path)
        except Exception as e:  # pragma: no cover - plotting optional
            print(f"PNG skipped: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
