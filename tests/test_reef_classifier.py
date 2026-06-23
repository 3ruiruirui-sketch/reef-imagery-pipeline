"""Tests for src/reef_classifier.py — ReefClassifier, dataset, and utilities.

All tests are offline (no network, no ImageNet download).
ReefClassifier is instantiated with pretrained=False throughout.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

pytestmark = pytest.mark.skipif(not _HAS_TORCH, reason="PyTorch not installed")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_dataset(root: Path, n_reef: int = 3, n_sand: int = 5) -> Path:
    """Write synthetic (4, 128, 128) .npy patches into reef/ and sand/ dirs."""
    for cls, n in (("reef", n_reef), ("sand", n_sand)):
        d = root / cls
        d.mkdir(parents=True)
        for i in range(n):
            arr = np.random.rand(4, 128, 128).astype(np.float32)
            np.save(d / f"patch_{i:05d}.npy", arr)
    return root


# ─────────────────────────────────────────────────────────────────────────────
# ReefClassifier — architecture
# ─────────────────────────────────────────────────────────────────────────────

class TestReefClassifierArchitecture:
    def test_instantiates_without_network(self):
        from src.reef_classifier import ReefClassifier
        model = ReefClassifier(pretrained=False)
        assert model is not None

    def test_first_conv_accepts_4_channels(self):
        from src.reef_classifier import ReefClassifier
        model = ReefClassifier(pretrained=False)
        first_conv = model.backbone.conv1
        assert first_conv.in_channels == 4

    def test_forward_output_shape(self):
        from src.reef_classifier import ReefClassifier
        model = ReefClassifier(pretrained=False).eval()
        x = torch.zeros(2, 4, 128, 128)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2,), f"Expected (2,), got {out.shape}"

    def test_output_is_probability(self):
        from src.reef_classifier import ReefClassifier
        model = ReefClassifier(pretrained=False).eval()
        x = torch.rand(4, 4, 128, 128)
        with torch.no_grad():
            probs = model(x)
        assert probs.min() >= 0.0 and probs.max() <= 1.0

    def test_single_sample_forward(self):
        from src.reef_classifier import ReefClassifier
        model = ReefClassifier(pretrained=False).eval()
        x = torch.rand(1, 4, 64, 64)
        with torch.no_grad():
            p = model(x)
        assert p.shape == (1,)
        assert 0.0 <= p.item() <= 1.0

    def test_dropout_disabled_at_eval(self):
        """In eval mode two identical forward passes must give identical output."""
        from src.reef_classifier import ReefClassifier
        model = ReefClassifier(pretrained=False, dropout=0.5).eval()
        x = torch.rand(1, 4, 128, 128)
        with torch.no_grad():
            p1 = model(x).item()
            p2 = model(x).item()
        assert p1 == pytest.approx(p2)


# ─────────────────────────────────────────────────────────────────────────────
# ReefPatchDataset
# ─────────────────────────────────────────────────────────────────────────────

class TestReefPatchDataset:
    def test_len_equals_reef_plus_sand(self, tmp_path):
        from src.reef_classifier import ReefPatchDataset
        _make_dataset(tmp_path, n_reef=3, n_sand=5)
        ds = ReefPatchDataset(tmp_path)
        assert len(ds) == 8

    def test_reef_label_is_1(self, tmp_path):
        from src.reef_classifier import ReefPatchDataset
        _make_dataset(tmp_path, n_reef=2, n_sand=0)
        ds = ReefPatchDataset(tmp_path)
        for i in range(len(ds)):
            _, label = ds[i]
            assert label.item() == 1.0

    def test_sand_label_is_0(self, tmp_path):
        from src.reef_classifier import ReefPatchDataset
        _make_dataset(tmp_path, n_reef=0, n_sand=2)
        ds = ReefPatchDataset(tmp_path)
        for i in range(len(ds)):
            _, label = ds[i]
            assert label.item() == 0.0

    def test_patch_shape(self, tmp_path):
        from src.reef_classifier import ReefPatchDataset
        _make_dataset(tmp_path, n_reef=1, n_sand=1)
        ds = ReefPatchDataset(tmp_path)
        x, _ = ds[0]
        assert x.shape == (4, 128, 128)

    def test_patch_dtype_is_float32(self, tmp_path):
        from src.reef_classifier import ReefPatchDataset
        _make_dataset(tmp_path, n_reef=1, n_sand=0)
        ds = ReefPatchDataset(tmp_path)
        x, _ = ds[0]
        assert x.dtype == torch.float32

    def test_empty_dataset(self, tmp_path):
        from src.reef_classifier import ReefPatchDataset
        _make_dataset(tmp_path, n_reef=0, n_sand=0)
        ds = ReefPatchDataset(tmp_path)
        assert len(ds) == 0


# ─────────────────────────────────────────────────────────────────────────────
# predict_single
# ─────────────────────────────────────────────────────────────────────────────

class TestPredictSingle:
    def test_returns_probability_in_0_1(self):
        from src.reef_classifier import ReefClassifier, predict_single
        model = ReefClassifier(pretrained=False).eval()
        patch = np.random.rand(4, 128, 128).astype(np.float32)
        prob = predict_single(model, patch, device="cpu")
        assert 0.0 <= prob <= 1.0

    def test_returns_float(self):
        from src.reef_classifier import ReefClassifier, predict_single
        model = ReefClassifier(pretrained=False).eval()
        patch = np.zeros((4, 128, 128), dtype=np.float32)
        result = predict_single(model, patch, device="cpu")
        assert isinstance(result, float)

    def test_deterministic_at_eval(self):
        from src.reef_classifier import ReefClassifier, predict_single
        model = ReefClassifier(pretrained=False).eval()
        patch = np.random.rand(4, 64, 64).astype(np.float32)
        p1 = predict_single(model, patch, device="cpu")
        p2 = predict_single(model, patch, device="cpu")
        assert p1 == pytest.approx(p2)


# ─────────────────────────────────────────────────────────────────────────────
# _roc_auc (pure numpy, no torch needed)
# ─────────────────────────────────────────────────────────────────────────────

class TestRocAuc:
    def test_perfect_classifier_returns_1(self):
        from src.reef_classifier import _roc_auc
        labels = np.array([1, 1, 0, 0])
        probs  = np.array([0.9, 0.8, 0.2, 0.1])
        assert _roc_auc(labels, probs) == pytest.approx(1.0)

    def test_random_classifier_near_05(self):
        from src.reef_classifier import _roc_auc
        rng = np.random.default_rng(42)
        labels = rng.integers(0, 2, 200)
        probs  = rng.random(200)
        auc = _roc_auc(labels, probs)
        assert 0.3 < auc < 0.7

    def test_inverted_classifier_returns_0(self):
        from src.reef_classifier import _roc_auc
        labels = np.array([1, 1, 0, 0])
        probs  = np.array([0.1, 0.2, 0.8, 0.9])
        assert _roc_auc(labels, probs) == pytest.approx(0.0)

    def test_no_positives_returns_05(self):
        from src.reef_classifier import _roc_auc
        labels = np.array([0, 0, 0])
        probs  = np.array([0.1, 0.5, 0.9])
        assert _roc_auc(labels, probs) == pytest.approx(0.5)

    def test_no_negatives_returns_05(self):
        from src.reef_classifier import _roc_auc
        labels = np.array([1, 1, 1])
        probs  = np.array([0.1, 0.5, 0.9])
        assert _roc_auc(labels, probs) == pytest.approx(0.5)
