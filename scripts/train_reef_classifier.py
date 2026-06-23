#!/usr/bin/env python3
"""
train_reef_classifier.py — Train and evaluate the ResNet-18 reef classifier.

Expects patches already built by build_reef_dataset.py in:
    outputs/reef_habitat/patches/{train,val}/{reef,sand}/*.npy

Saves:
    outputs/reef_habitat/model/reef_cnn_best.pth     — best checkpoint
    outputs/reef_habitat/model/training_log.json      — per-epoch metrics
    outputs/reef_habitat/model/evaluation.json        — final test metrics

Usage
-----
    python scripts/train_reef_classifier.py
    python scripts/train_reef_classifier.py --epochs 50 --batch-size 32 --lr 5e-5
    python scripts/train_reef_classifier.py --eval-only  # skip training, just evaluate
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PATCHES_DIR = Path("outputs/reef_habitat/patches")
MODEL_DIR   = Path("outputs/reef_habitat/model")
TRAIN_DIR   = PATCHES_DIR / "train"
VAL_DIR     = PATCHES_DIR / "val"
BEST_PATH   = MODEL_DIR / "reef_cnn_best.pth"
EVAL_PATH   = MODEL_DIR / "evaluation.json"


def _check_dataset() -> bool:
    """Verify that patch directories are non-empty."""
    for cls in ("reef", "sand"):
        for split in ("train", "val"):
            d = PATCHES_DIR / split / cls
            if not d.exists() or not any(d.glob("*.npy")):
                log.error("Missing patches in %s — run build_reef_dataset.py first", d)
                return False
    return True


def _count_patches() -> dict:
    counts = {}
    for split in ("train", "val"):
        for cls in ("reef", "sand"):
            d = PATCHES_DIR / split / cls
            counts[f"{split}_{cls}"] = len(list(d.glob("*.npy"))) if d.exists() else 0
    return counts


def run_training(args: argparse.Namespace) -> None:
    from src.reef_classifier import ReefClassifier, train, evaluate, load_model

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if not _check_dataset():
        sys.exit(1)

    counts = _count_patches()
    log.info(
        "Dataset: train reef=%d sand=%d  |  val reef=%d sand=%d",
        counts["train_reef"], counts["train_sand"],
        counts["val_reef"],   counts["val_sand"],
    )

    if args.eval_only:
        if not BEST_PATH.exists():
            log.error("No checkpoint at %s — train first (remove --eval-only)", BEST_PATH)
            sys.exit(1)
        log.info("Eval-only mode: loading checkpoint %s", BEST_PATH)
        model = load_model(BEST_PATH)
    else:
        model = ReefClassifier(pretrained=True, dropout=args.dropout)
        n_params = sum(p.numel() for p in model.parameters())
        log.info("Model: ResNet-18 4-channel  (%d params / %.1f MB)", n_params, n_params * 4 / 1e6)

        history = train(
            model,
            train_dir=TRAIN_DIR,
            val_dir=VAL_DIR,
            output_dir=MODEL_DIR,
            n_epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
        )

        # Print training curve summary
        if history:
            best_epoch = max(history, key=lambda h: h["val_auc"])
            print("\n── Training summary ──────────────────────────────────────")
            print(f"  Best epoch  : {best_epoch['epoch']}")
            print(f"  Best val AUC: {best_epoch['val_auc']:.4f}")
            print(f"  Val accuracy: {best_epoch['val_acc']:.4f}")
            print(f"  Checkpoint  : {BEST_PATH}")
            print("──────────────────────────────────────────────────────────")

        # Reload best weights for evaluation
        from src.reef_classifier import load_model
        model = load_model(BEST_PATH)

    # Evaluate on validation set
    log.info("Evaluating on validation set…")
    metrics = evaluate(model, VAL_DIR, threshold=args.threshold)

    with open(EVAL_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n── Evaluation results ────────────────────────────────────────")
    print(f"  AUC       : {metrics['auc']:.4f}")
    print(f"  F1        : {metrics['f1']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  TP={metrics['tp']}  FP={metrics['fp']}  FN={metrics['fn']}  TN={metrics['tn']}")
    print(f"  Saved to  : {EVAL_PATH}")
    print("──────────────────────────────────────────────────────────────")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ReefClassifier (ResNet-18, 4-band S2)")
    parser.add_argument("--epochs",       type=int,   default=30,    help="Max training epochs")
    parser.add_argument("--batch-size",   type=int,   default=16,    help="Mini-batch size")
    parser.add_argument("--lr",           type=float, default=1e-4,  help="Initial learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4,  help="AdamW weight decay")
    parser.add_argument("--patience",     type=int,   default=8,     help="Early stopping patience (epochs)")
    parser.add_argument("--dropout",      type=float, default=0.3,   help="Dropout in classifier head")
    parser.add_argument("--threshold",    type=float, default=0.5,   help="Prediction threshold for binary metrics")
    parser.add_argument("--eval-only",    action="store_true",        help="Skip training, evaluate existing checkpoint")
    args = parser.parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
