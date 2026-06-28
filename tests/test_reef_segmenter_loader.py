"""Tests for reef_segmenter's architecture-aware weight loading.

Pure-logic tests (architecture detection, weights resolution) run everywhere.
A torch-gated test actually loads the local smp checkpoint when present + torch is
installed, so it exercises the real load path in the project .venv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src import reef_segmenter as rs  # noqa: E402


def test_looks_like_smp_detects_smp_checkpoint():
    smp_state = {
        "model.encoder.layer4.1.conv1.weight": 0,
        "model.decoder.blocks.0.conv1.0.weight": 0,
        "model.segmentation_head.0.weight": 0,
    }
    assert rs._looks_like_smp(smp_state) is True


def test_looks_like_smp_rejects_pure_reefunet_checkpoint():
    reefunet_state = {
        "downs.0.net.0.weight": 0,
        "bottleneck.net.0.weight": 0,
        "final.weight": 0,
        "final.bias": 0,
    }
    assert rs._looks_like_smp(reefunet_state) is False


def test_resolve_weights_prefers_explicit(tmp_path):
    p = tmp_path / "custom.pth"
    assert rs._resolve_weights(str(p)) == Path(str(p))


def test_resolve_weights_prefers_canonical_then_fallback(monkeypatch, tmp_path):
    canonical = tmp_path / "unet_reef_best.pth"
    fallback = tmp_path / "unet_reef_best_smp_resnet18.pth"
    monkeypatch.setattr(rs, "_DEFAULT_WEIGHTS", canonical)
    monkeypatch.setattr(rs, "_SMP_FALLBACK_WEIGHTS", fallback)

    # Neither exists -> returns canonical (surfaced as a clear error upstream).
    assert rs._resolve_weights(None) == canonical

    # Only fallback exists -> returns fallback.
    fallback.write_bytes(b"x")
    assert rs._resolve_weights(None) == fallback

    # Canonical exists -> preferred over fallback.
    canonical.write_bytes(b"x")
    assert rs._resolve_weights(None) == canonical


_SMP_CKPT = _PROJECT_ROOT / "models" / "unet_reef_best_smp_resnet18.pth"


@pytest.mark.skipif(not rs.HAS_TORCH, reason="torch not installed")
@pytest.mark.skipif(not _SMP_CKPT.exists(), reason="smp checkpoint not present")
def test_build_and_load_smp_checkpoint_produces_probabilities():
    """The real fallback checkpoint loads and yields a [0,1] probability map."""
    import numpy as np
    import torch

    model, applies_sigmoid = rs._build_and_load(_SMP_CKPT)
    assert applies_sigmoid is True  # vanilla smp.Unet returns logits

    x = torch.zeros(1, 1, 256, 256)
    with torch.no_grad():
        out = model(x)
        if applies_sigmoid:
            out = torch.sigmoid(out)
    arr = out.cpu().numpy()
    assert arr.shape == (1, 1, 256, 256)
    assert np.isfinite(arr).all()
    assert arr.min() >= 0.0 and arr.max() <= 1.0
