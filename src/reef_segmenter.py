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
# Canonical weights = pure-PyTorch ReefUNet produced by scripts/dgt_train_unet.py
# and notebooks/01 (sigmoid applied inside forward).
_DEFAULT_WEIGHTS = _PROJECT_ROOT / "models" / "unet_reef_best.pth"
# Fallback = vanilla smp.Unet(resnet18) checkpoint (raw logits, sigmoid applied here).
_SMP_FALLBACK_WEIGHTS = _PROJECT_ROOT / "models" / "unet_reef_best_smp_resnet18.pth"
_TILE = 256  # inference tile size; both architectures accept multiples of 32

# Lazily-loaded singleton model (keyed by weights path).
# _MODEL_APPLIES_SIGMOID records whether the loaded architecture already applies
# sigmoid in forward() (ReefUNet) or returns raw logits (smp.Unet) so callers
# get probabilities either way.
_MODEL = None
_MODEL_KEY: str | None = None
_MODEL_APPLIES_SIGMOID: bool = True


def _resolve_weights(weights_path: str | None):
    """Pick the weights file: explicit > canonical ReefUNet > smp fallback."""
    if weights_path:
        return Path(weights_path)
    if _DEFAULT_WEIGHTS.exists():
        return _DEFAULT_WEIGHTS
    if _SMP_FALLBACK_WEIGHTS.exists():
        return _SMP_FALLBACK_WEIGHTS
    return _DEFAULT_WEIGHTS  # non-existent — surfaced as a clear error below


def _looks_like_smp(state: dict) -> bool:
    """smp.Unet state_dicts carry encoder/decoder/segmentation_head keys."""
    return any(
        k.startswith(("model.encoder", "encoder.", "model.decoder", "decoder.", "segmentation_head", "model.segmentation_head"))
        for k in state
    )


def _build_and_load(path: Path):
    """Return (model, applies_sigmoid) by matching architecture to the checkpoint.

    Supports the canonical pure-PyTorch ReefUNet (sigmoid inside) and a vanilla
    smp.Unet(resnet18) checkpoint (raw logits). Handles common state_dict nesting
    and an optional leading ``model.`` prefix from training wrappers.
    """
    state = torch.load(str(path), map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    if _looks_like_smp(state):
        try:
            import segmentation_models_pytorch as smp
        except Exception as e:  # pragma: no cover - torch-less / smp-less install
            raise RuntimeError(
                f"{path.name} is an smp.Unet checkpoint but segmentation_models_pytorch "
                f"is not installed. Install it, or provide the canonical ReefUNet weights "
                f"at {_DEFAULT_WEIGHTS}."
            ) from e
        net = smp.Unet(encoder_name="resnet18", encoder_weights=None, in_channels=1, classes=1, activation=None)
        sd = {(k[len("model.") :] if k.startswith("model.") else k): v for k, v in state.items()}
        net.load_state_dict(sd)
        net.eval()
        log.info("Loaded smp.Unet(resnet18) reef model from %s (logits → sigmoid applied)", path)
        return net, True  # raw logits → sigmoid applied by caller

    from src.ml_unet_model import ReefUNet

    net = ReefUNet(in_channels=1)
    net.load_state_dict(state)
    net.eval()
    log.info("Loaded canonical ReefUNet from %s", path)
    return net, False  # ReefUNet applies sigmoid internally


def normalize_depth(arr: np.ndarray, min_depth_m: float = -SDB_OPTICAL_LIMIT_M, max_depth_m: float = 0.0) -> np.ndarray:
    """Scale depth (negative metres) to [0,1]; NaN→0. Mirrors the training dataset."""
    arr = np.nan_to_num(arr.astype(np.float32), nan=0.0)
    norm = (arr - min_depth_m) / (max_depth_m - min_depth_m)
    return np.clip(norm, 0.0, 1.0)


def _load_model(weights_path: str | None = None):
    """Load reef U-Net weights once and cache. Raises RuntimeError if unavailable.

    Sets module-level _MODEL_APPLIES_SIGMOID so segment_reef_array() knows whether
    to apply sigmoid (smp logits) or not (ReefUNet already does).
    """
    global _MODEL, _MODEL_KEY, _MODEL_APPLIES_SIGMOID
    if not HAS_TORCH:
        raise RuntimeError(
            "PyTorch not installed — reef segmentation unavailable. "
            "Install with: pip install torch segmentation-models-pytorch"
        )
    resolved = _resolve_weights(weights_path)
    path = str(resolved)
    if _MODEL is not None and path == _MODEL_KEY:
        return _MODEL
    if not resolved.exists():
        raise RuntimeError(
            "No reef U-Net weights found. Looked for canonical "
            f"'{_DEFAULT_WEIGHTS.name}' and fallback '{_SMP_FALLBACK_WEIGHTS.name}' "
            f"in {_DEFAULT_WEIGHTS.parent}. Train on the DGT VM "
            "(python -m scripts.dgt_train_unet) or copy the trained "
            f"'{_DEFAULT_WEIGHTS.name}' from the VM into models/."
        )

    model, applies_sigmoid = _build_and_load(resolved)
    _MODEL, _MODEL_KEY, _MODEL_APPLIES_SIGMOID = model, path, applies_sigmoid
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
                out = model(x)
                if _MODEL_APPLIES_SIGMOID:  # smp.Unet returns raw logits
                    out = torch.sigmoid(out)
                prob[r : r + tile, c : c + tile] = out[0, 0].cpu().numpy()

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
