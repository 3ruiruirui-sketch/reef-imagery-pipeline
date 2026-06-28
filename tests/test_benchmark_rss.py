"""Tests for benchmark_cpu._maxrss_to_mb / _peak_rss_mb unit conversion.

``resource.getrusage(RUSAGE_SELF).ru_maxrss`` is in BYTES on macOS/Darwin but
in KILOBYTES on Linux. A logged run once reported 656160 MB for a ~600 MB
process because the units were mishandled. These tests pin both branches.
"""
from __future__ import annotations

import sys
from pathlib import Path

# benchmark_cpu.py lives at the repo root, not under src/.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import benchmark_cpu  # noqa: E402


def test_darwin_treats_maxrss_as_bytes():
    """On Darwin a ~600 MB process reports ru_maxrss in bytes."""
    ru_maxrss_bytes = 600 * 1_000_000  # 600 MB expressed in bytes
    mb = benchmark_cpu._maxrss_to_mb(ru_maxrss_bytes, "darwin")
    assert abs(mb - 600.0) < 1.0
    assert mb < 4000.0


def test_linux_treats_maxrss_as_kilobytes():
    """On Linux a ~600 MB process reports ru_maxrss in kilobytes."""
    ru_maxrss_kb = 600 * 1024  # 600 MB expressed in kibibytes
    mb = benchmark_cpu._maxrss_to_mb(ru_maxrss_kb, "linux")
    assert abs(mb - 600.0) < 1.0
    assert mb < 4000.0


def test_branches_disagree_on_same_raw_value():
    """The two platforms must interpret the same raw value differently —
    proving the conversion is genuinely platform-aware, not a no-op."""
    raw = 600 * 1024  # a plausible Linux KB value
    linux_mb = benchmark_cpu._maxrss_to_mb(raw, "linux")
    darwin_mb = benchmark_cpu._maxrss_to_mb(raw, "darwin")
    assert linux_mb != darwin_mb
    # Same raw treated as bytes on darwin is ~0.6 MB, as KB on linux is 600 MB.
    assert darwin_mb < linux_mb


def test_peak_rss_mb_is_plausible():
    """The live cross-platform call must return a small, finite, positive MB."""
    mb = benchmark_cpu._peak_rss_mb()
    assert isinstance(mb, float)
    assert mb > 0.0
    assert mb < 4000.0  # no test process should ever report multi-GB here
