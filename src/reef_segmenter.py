"""
reef_segmenter.py — U-Net reef-mask inference over a bathymetry (SDB) raster.

Post-SDB stage: takes the depth GeoTIFF produced by the pipeline
(`bathy_*_{date}.tif`, negative metres) and runs the trained ReefUNet
(`models/unet_reef_best.pth`) to produce a per-pixel reef-probability map and
a binary reef mask.

The U-Net was trained on 1-channel bathymetry patches normalized to [0, 1] over
the depth window [-SDB_OPTICAL_LIMIT_M, 0]. Inference MUST use the identical
window — `normalize_depth()` here mirrors `ReefPatchDataset._normalize_depth`
and both default to the same `SDB_OPTICAL_LIMIT_M` constant.

torch is an optional heavy dep; this module degrades gracefully (HAS_TORCH=False,
segment_* raise a clear RuntimeError) so importing it never breaks the pipeline on a
torch-less install. segmentation_models_pytorch is NOT required — the model is pure PyTorch.

Usage:
    from src.reef_segmenter import segment_reef_tif, segment_reef_array
    out = segment_reef_tif("outputs/site/bathy_dgt_lidar_20240905.tif")
    # → writes reef_mask_20240905.tif alongside; returns {"prob","mask","reef_fraction",...}
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from src.constants import SDB_OPTICAL_LIMIT_M

log = logging.getLogger(__name__)

try:
    import torch

    HAS_TORCH = True
except Exception:  # pragma: no cover - exercised only on torch-less installs
    HAS_TORCH = False

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_WEIGHTS = _PROJECT_ROOT / "models" / "unet_reef_best.pth"
_TILE = 256  # inference tile size; new pure-PyTorch model needs multiples of 16

# Lazily-loaded singleton model (keyed by weights path)
_MODEL = None
_MODEL_KEY: str | None = None


def normalize_depth(arr: np.ndarray, min_depth_m: float = -SDB_OPTICAL_LIMIT_M, max_depth_m: float = 0.0) -> np.ndarray:
    """Scale depth (negative metres) to [0,1]; NaN→0. Mirrors the training dataset."""
    arr = np.nan_to_num(arr.astype(np.float32), nan=0.0)
    norm = (arr - min_depth_m) / (max_depth_m - min_depth_m)
    return np.clip(norm, 0.0, 1.0)


def _load_model(weights_path: str | None = None):
    """Load ReefUNet weights once and cache. Raises RuntimeError if torch missing."""
    global _MODEL, _MODEL_KEY
    if not HAS_TORCH:
        raise RuntimeError(
            "PyTorch not installed — reef segmentation unavailable. "
            "Install with: pip install torch segmentation-models-pytorch"
        )
    path = str(weights_path or _DEFAULT_WEIGHTS)
    if _MODEL is not None and path == _MODEL_KEY:
        return _MODEL
    if not Path(path).exists():
        raise RuntimeError(f"U-Net weights not found: {path}")

    from src.ml_unet_model import ReefUNet

    model = ReefUNet(in_channels=1)
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    _MODEL, _MODEL_KEY = model, path
    log.info("ReefUNet loaded from %s", path)
    return model


def segment_reef_array(depth: np.ndarray, threshold: float = 0.5, tile: int = _TILE, weights_path: str | None = None) -> dict:
    """
    Run the U-Net over a 2-D depth array (negative metres).

    Tiles the raster into `tile`×`tile` windows (edge-padded by reflection) so
    memory stays bounded on large rasters, then stitches the probability map.

    Returns {"prob": float32 [0,1] HxW, "mask": uint8 {0,1} HxW, "reef_fraction": float}.
    """
    model = _load_model(weights_path)
    if depth.ndim != 2:
        raise ValueError(f"expected 2-D depth array, got shape {depth.shape}")
    h, w = depth.shape

    # Pad up to whole tiles (reflect avoids fabricating depth=0 land at edges)
    pad_h = (-h) % tile
    pad_w = (-w) % tile
    norm = normalize_depth(depth)
    if pad_h or pad_w:
        norm = np.pad(norm, ((0, pad_h), (0, pad_w)), mode="reflect")

    prob = np.zeros_like(norm, dtype=np.float32)
    with torch.no_grad():
        for r in range(0, norm.shape[0], tile):
            for c in range(0, norm.shape[1], tile):
                patch = norm[r : r + tile, c : c + tile]
                x = torch.from_numpy(patch).float().unsqueeze(0).unsqueeze(0)  # (1,1,T,T)
                probs = model(x)  # model already applies sigmoid internally
                prob[r : r + tile, c : c + tile] = probs[0, 0].cpu().numpy()

    prob = prob[:h, :w]
    mask = (prob >= threshold).astype(np.uint8)
    return {
        "prob": prob,
        "mask": mask,
        "reef_fraction": float(mask.mean()),
    }


def segment_reef_tif(
    depth_tif: str, out_tif: str | None = None, threshold: float = 0.5, weights_path: str | None = None
) -> dict:
    """
    Segment a bathymetry GeoTIFF and write a georeferenced reef-mask GeoTIFF.

    out_tif defaults to `reef_mask_<stem-suffix>.tif` next to the input
    (a leading `bathy_` / `bathy_dgt_lidar_` / `bathy_s2_stumpf_` prefix is
    stripped so the date suffix is preserved).

    Returns the segment_reef_array() dict plus {"out_tif", "src_tif"}.
    """
    import rasterio

    src_path = Path(depth_tif)
    with rasterio.open(str(src_path)) as src:
        depth = src.read(1).astype(np.float32)
        nodata = src.nodata
        profile = src.profile

    if nodata is not None:
        depth = np.where(depth == nodata, np.nan, depth)

    result = segment_reef_array(depth, threshold=threshold, weights_path=weights_path)

    if out_tif is None:
        stem = src_path.stem
        for pref in ("bathy_dgt_lidar_", "bathy_s2_stumpf_", "bathy_"):
            if stem.startswith(pref):
                stem = stem[len(pref) :]
                break
        out_tif = str(src_path.with_name(f"reef_mask_{stem}.tif"))

    profile.update(dtype="uint8", count=1, nodata=0, compress="deflate")
    with rasterio.open(out_tif, "w", **profile) as dst:
        dst.write(result["mask"], 1)
    log.info("Reef mask → %s  (reef_fraction=%.3f)", out_tif, result["reef_fraction"])

    result["out_tif"] = out_tif
    result["src_tif"] = str(src_path)
    return result
