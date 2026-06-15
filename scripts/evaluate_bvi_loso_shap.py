#!/usr/bin/env python3
"""
evaluate_bvi_loso_shap.py — honest evaluation of the BVI pairwise model.

Two questions this script answers:

  1. How much does the reported performance drop when we move from a random
     train/test split (the current `train_feature_ranker.py` default) to a
     spatially-grouped CV (Leave-One-Site-Out)?
  2. Which features actually drive the model's preferences, and how confident
     are we in each — given the small sample size?

Data source
-----------
``data/real_pairwise_features.csv`` (206 rows = 103 pairs × 2 sides, 15 sites)
plus ``data/real_pairwise_labels.csv`` (winner per pair). Features actually
present in the CSV are physical/bathymetric — they do NOT match the image-
quality FEATURE_COLS hardcoded in train_feature_ranker.py, which is why the
existing training script silently falls back to synthetic data. This script
uses the real features directly.

Outputs (``outputs/bvi_evaluation/``):
  - metrics_random_vs_loso.json
  - feature_importance.png
  - shap_summary.png
  - shap_bootstrap_ci.png
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import GroupKFold, train_test_split

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _PROJECT_ROOT / "data"
_OUT_DIR = _PROJECT_ROOT / "outputs" / "bvi_evaluation"

# Numeric features actually present in real_pairwise_features.csv.
# Physical / bathymetric — the model learns site-level visibility proxies.
# `nearest_isobath_proximity` is derived from `nearest_isobath_distance_m`
# (see below) to avoid the inf-sentinel artefact of the raw distance column.
NUMERIC_FEATURE_COLS = [
    "kd_b02",                       # Kd490 at B02 — water column attenuation
    "water_trans",                  # two-way Beer-Lambert transmittance
    "image_contrast",               # image contrast (renamed from CSV's
                                    # `contrast` to avoid collision with the
                                    # disabled B02-only "contrast" feature)
    "signal_strength",              # B02 signal level
    "cleanliness",                  # FFT cleanliness
    "bathy_slope_proxy",            # seabed slope proxy
    "contour_density_proxy",        # isobath contour density
    "n_isobaths_aoi",               # # isobaths in AOI
    "nearest_isobath_proximity",    # derived: 1 / (1 + d/100); 0 ⇒ no isobath
]

# Categorical feature one-hot encoded into K-1 indicator columns.
BATHY_ZONE_VALUES = ["very_shallow", "nearshore_mid", "shallow_reef", "offshore", "unknown"]
ZONE_FEATURE_COLS = [f"zone_{v}" for v in BATHY_ZONE_VALUES[:-1]]  # drop 'unknown' as reference

FEATURE_COLS = NUMERIC_FEATURE_COLS + ZONE_FEATURE_COLS

log = logging.getLogger("eval_bvi")


def load_pointwise() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Load real pairwise data and expand to pointwise (winner=1, loser=0).

    Returns
    -------
    X : DataFrame with FEATURE_COLS
    y : np.ndarray of 0/1 floats
    groups : np.ndarray of site names (one per row), for GroupKFold
    """
    feats = pd.read_csv(_DATA_DIR / "real_pairwise_features.csv")
    labels = pd.read_csv(_DATA_DIR / "real_pairwise_labels.csv")

    # Replace inf-based distance with a bounded proximity score in [0, 1].
    # proximity = 1 / (1 + d/100m); d=0 ⇒ 1.0, d→∞ ⇒ 0. No sentinel inflation,
    # monotone, and preserves the physical "closer = more visible" intuition.
    d = feats["nearest_isobath_distance_m"].replace([np.inf, -np.inf], np.nan)
    feats["nearest_isobath_proximity"] = (1.0 / (1.0 + d / 100.0)).fillna(0.0)

    # Rename `contrast` → `image_contrast` (collision with disabled feature).
    feats["image_contrast"] = feats["contrast"]

    # One-hot encode bathy_zone_class (K-1 indicators; 'unknown' is reference).
    for v in BATHY_ZONE_VALUES[:-1]:
        feats[f"zone_{v}"] = (feats["bathy_zone_class"] == v).astype(float)

    by_id = feats.set_index("image_id")

    X_rows, y_rows, group_rows = [], [], []
    for _, lab in labels.iterrows():
        a_id, b_id, winner = lab["image_a"], lab["image_b"], lab["winner"]
        if a_id not in by_id.index or b_id not in by_id.index:
            continue
        feat_a = by_id.loc[a_id]
        feat_b = by_id.loc[b_id]

        # winner gets y=1, loser gets y=0. groups = site (so both samples from
        # the same pair share the same site for the WINNER, but loser may come
        # from a different site — that's the point of GroupKFold here).
        if winner == a_id:
            X_rows.append(feat_a[FEATURE_COLS].values); y_rows.append(1.0); group_rows.append(feat_a["site"])
            X_rows.append(feat_b[FEATURE_COLS].values); y_rows.append(0.0); group_rows.append(feat_b["site"])
        else:
            X_rows.append(feat_b[FEATURE_COLS].values); y_rows.append(1.0); group_rows.append(feat_b["site"])
            X_rows.append(feat_a[FEATURE_COLS].values); y_rows.append(0.0); group_rows.append(feat_a["site"])

    X = pd.DataFrame(X_rows, columns=FEATURE_COLS).astype(float)
    return X, np.asarray(y_rows, dtype=float), np.asarray(group_rows)


def fit_rf(X_train, y_train) -> RandomForestRegressor:
    # Same hyperparams as scripts/train_feature_ranker.py for a fair comparison.
    m = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
    m.fit(X_train, y_train)
    return m


def score(model, X_test, y_test) -> dict:
    pred = model.predict(X_test)
    mse = mean_squared_error(y_test, pred)
    acc = accuracy_score((y_test >= 0.5).astype(int), (pred >= 0.5).astype(int))
    return {"mse": float(mse), "ranking_accuracy": float(acc), "n_test": int(len(y_test))}


def eval_random_split(X, y) -> dict:
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    m = fit_rf(Xtr, ytr)
    out = score(m, Xte, yte)
    out["n_train"] = int(len(Xtr))
    return out


def eval_loso(X, y, groups) -> dict:
    """Leave-One-Site-Out via GroupKFold(n_splits=#sites)."""
    unique_sites = np.unique(groups)
    cv = GroupKFold(n_splits=len(unique_sites))
    fold_metrics = []
    for fold_idx, (tr, te) in enumerate(cv.split(X, y, groups)):
        m = fit_rf(X.iloc[tr], y[tr])
        s = score(m, X.iloc[te], y[te])
        s["fold"] = fold_idx
        s["test_site"] = str(np.unique(groups[te]))
        fold_metrics.append(s)
    mse_arr = np.array([f["mse"] for f in fold_metrics])
    acc_arr = np.array([f["ranking_accuracy"] for f in fold_metrics])
    return {
        "n_folds": int(len(fold_metrics)),
        "mse_mean": float(mse_arr.mean()),
        "mse_std":  float(mse_arr.std()),
        "ranking_accuracy_mean": float(acc_arr.mean()),
        "ranking_accuracy_std":  float(acc_arr.std()),
        "folds": fold_metrics,
    }


def plot_metric_comparison(rand_metrics: dict, loso_metrics: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    labels = ["Random 80/20", "LOSO (GroupKFold)"]
    mse_vals = [rand_metrics["mse"], loso_metrics["mse_mean"]]
    mse_err  = [0,                    loso_metrics["mse_std"]]
    acc_vals = [rand_metrics["ranking_accuracy"], loso_metrics["ranking_accuracy_mean"]]
    acc_err  = [0,                                loso_metrics["ranking_accuracy_std"]]

    axes[0].bar(labels, mse_vals, yerr=mse_err, color=["#4c72b0", "#dd8452"], capsize=8)
    axes[0].set_ylabel("MSE"); axes[0].set_title("Mean Squared Error")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(labels, acc_vals, yerr=acc_err, color=["#4c72b0", "#dd8452"], capsize=8)
    axes[1].set_ylabel("Ranking accuracy"); axes[1].set_title("Pairwise ranking accuracy")
    axes[1].set_ylim(0, 1); axes[1].axhline(0.5, color="grey", linestyle="--", alpha=0.5, label="chance")
    axes[1].grid(axis="y", alpha=0.3); axes[1].legend()

    fig.suptitle("Random split vs Leave-One-Site-Out — honest CV reveals over-optimism")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def shap_with_bootstrap(X: pd.DataFrame, y: np.ndarray, n_boot: int = 100) -> dict:
    """Bootstrap SHAP importances to get confidence intervals.

    For each bootstrap rep: sample with replacement, fit RF, compute mean |SHAP|
    per feature. Across reps, report mean + 5/95% percentiles.
    """
    rng = np.random.default_rng(42)
    n = len(X)
    importances = np.zeros((n_boot, X.shape[1]), dtype=float)

    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        Xb, yb = X.iloc[idx], y[idx]
        m = fit_rf(Xb, yb)
        explainer = shap.TreeExplainer(m)
        sv = explainer.shap_values(Xb, check_additivity=False)
        importances[i] = np.abs(sv).mean(axis=0)

    return {
        "feature": list(X.columns),
        "mean":   importances.mean(axis=0).tolist(),
        "p05":    np.percentile(importances, 5,  axis=0).tolist(),
        "p95":    np.percentile(importances, 95, axis=0).tolist(),
    }


def plot_shap_summary(model: RandomForestRegressor, X: pd.DataFrame, out_path: Path) -> None:
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X, check_additivity=False)
    fig = plt.figure(figsize=(8, 6))
    shap.summary_plot(sv, X, show=False, plot_size=None)
    plt.title("SHAP summary — single fit on full data (descriptive only)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_shap_bootstrap(boot: dict, out_path: Path) -> None:
    df = pd.DataFrame(boot).sort_values("mean", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    y_pos = np.arange(len(df))
    ax.barh(y_pos, df["mean"],
            xerr=[df["mean"] - df["p05"], df["p95"] - df["mean"]],
            color="#4c72b0", ecolor="black", capsize=4, alpha=0.85)
    ax.set_yticks(y_pos); ax.set_yticklabels(df["feature"])
    ax.set_xlabel("Mean |SHAP value|  (bootstrap 5–95% CI)")
    ax.set_title("Feature importance with bootstrap confidence (n=100 reps)\n"
                 "Wide bars = small dataset; treat ordering as suggestive only")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Loading real pairwise data...")
    X, y, groups = load_pointwise()
    log.info("Dataset: %d samples, %d features, %d sites",
             len(X), X.shape[1], len(np.unique(groups)))

    # 1. Random split — replicate the current train_feature_ranker baseline.
    log.info("Evaluating random 80/20 split...")
    rand_metrics = eval_random_split(X, y)
    log.info("  random split → MSE=%.4f, acc=%.2f%%",
             rand_metrics["mse"], rand_metrics["ranking_accuracy"] * 100)

    # 2. Leave-One-Site-Out via GroupKFold.
    log.info("Evaluating Leave-One-Site-Out (GroupKFold)...")
    loso_metrics = eval_loso(X, y, groups)
    log.info("  LOSO → MSE=%.4f±%.4f, acc=%.2f%%±%.2f%%",
             loso_metrics["mse_mean"], loso_metrics["mse_std"],
             loso_metrics["ranking_accuracy_mean"] * 100,
             loso_metrics["ranking_accuracy_std"] * 100)

    # Save metrics + figure.
    (_OUT_DIR / "metrics_random_vs_loso.json").write_text(
        json.dumps({"random_split": rand_metrics, "loso": loso_metrics}, indent=2)
    )
    plot_metric_comparison(rand_metrics, loso_metrics, _OUT_DIR / "metrics_comparison.png")

    # 3. SHAP on a single model fit (descriptive).
    log.info("Computing SHAP values (single fit)...")
    full_model = fit_rf(X, y)
    plot_shap_summary(full_model, X, _OUT_DIR / "shap_summary.png")

    # 4. SHAP with bootstrap CIs (honest, captures small-sample variance).
    log.info("Computing SHAP importances with bootstrap (n=100)...")
    boot = shap_with_bootstrap(X, y, n_boot=100)
    (_OUT_DIR / "shap_bootstrap.json").write_text(json.dumps(boot, indent=2))
    plot_shap_bootstrap(boot, _OUT_DIR / "shap_bootstrap_ci.png")

    log.info("Done. Artifacts saved to %s", _OUT_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
