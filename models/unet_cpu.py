"""
unet_cpu.py — Lightweight U-Net for reef segmentation on CPU.

Encoder: MobileNet-V2 (~2.2 M params) — guaranteed available in all SMP versions.
To use MobileNet-V3-Small install timm and set encoder='timm-mobilenetv3_small_100'.

Designed for:
- 1-channel bathymetry input (SDB depth map, negative metres)
- Binary reef / no-reef output
- CPU inference <30 s for 16 patches of 128×128
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn

log = logging.getLogger(__name__)

# Preferred encoder hierarchy (first available wins)
_ENCODER_PREFERENCE = [
    "timm-mobilenetv3_small_100",  # ~1.5 M params, needs timm
    "mobilenet_v2",                # ~2.2 M params, built into SMP
]
_DECODER_CHANNELS = (256, 128, 64, 32, 16)


def _pick_encoder(requested: str | None = None) -> str:
    try:
        import segmentation_models_pytorch as smp  # noqa: F401
        import segmentation_models_pytorch.encoders as enc_reg

        candidates = [requested] if requested else _ENCODER_PREFERENCE
        for name in candidates:
            if name is None:
                continue
            try:
                enc_reg.get_encoder(name, in_channels=1, depth=5, weights=None)
                return name
            except (KeyError, Exception):
                continue
        return "mobilenet_v2"
    except ImportError:
        return "mobilenet_v2"


class ReefUNetCPU(nn.Module):
    """
    U-Net with a lightweight MobileNet encoder — optimised for CPU training.

    Args:
        encoder_name: SMP encoder key. Pass None to auto-select the lightest available.
        in_channels:  Number of input channels (1 for single-band bathymetry).
        classes:      Number of output classes (1 for binary reef mask).
    """

    def __init__(
        self,
        encoder_name: str | None = None,
        in_channels: int = 1,
        classes: int = 1,
    ) -> None:
        super().__init__()
        import segmentation_models_pytorch as smp

        chosen = _pick_encoder(encoder_name)
        log.info("ReefUNetCPU — encoder: %s", chosen)

        self.model = smp.Unet(
            encoder_name=chosen,
            encoder_weights=None,           # train from scratch — ImageNet RGB weights don't suit 1-ch bathy
            in_channels=in_channels,
            classes=classes,
            activation=None,                # raw logits → use BCEWithLogitsLoss
            decoder_channels=_DECODER_CHANNELS,
            decoder_attention_type="scse",  # lightweight channel+spatial attention
        )
        self.encoder_name = chosen

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def load_model(weights_path: str | Path, device: str = "cpu") -> ReefUNetCPU:
    """Load a saved ReefUNetCPU from a .pth checkpoint."""
    path = Path(weights_path)
    if not path.exists():
        raise FileNotFoundError(f"Weights not found: {path}")
    model = ReefUNetCPU()
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    log.info("Loaded ReefUNetCPU from %s  (%d params)", path, model.n_params)
    return model


def benchmark_inference(model: ReefUNetCPU, n_images: int = 16, img_size: int = 128) -> float:
    """Return inference time in seconds for n_images patches of img_size×img_size."""
    import time
    model.eval()
    x = torch.randn(n_images, 1, img_size, img_size)
    with torch.no_grad():
        t0 = time.perf_counter()
        _ = model(x)
        return time.perf_counter() - t0
