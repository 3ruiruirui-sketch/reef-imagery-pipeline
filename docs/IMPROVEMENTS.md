# Project Improvement Roadmap

This document captures the highest-impact improvements for the Reef Imagery Pipeline.

## 1. Completed improvements

- Added EMODnet calibration regression tests:
  - `tests/test_stumpf_emodnet_calibration.py`
- Added explicit CI validation for the new EMODnet calibration path:
  - `.github/workflows/ci.yml`
- Added fallback handling and validation for dynamic EMODnet calibration in `scripts/reef_bathy_module.py`.
- Added reusable EMODnet calibration utilities in `src/stumpf_emodnet_calibration.py`.

## 2. Short-term improvements

1. **Documentation cleanup**
   - Ensure install instructions point to `requirements.txt` and `pip install -e .[dev]`.
   - Add a clear quick-start section for the pipeline and bathymetry module.

2. **Repository hygiene**
   - Keep experimental scratch code out of the main tracked workspace or add `scratch/` to `.gitignore` if it should remain local.
   - Remove or archive obsolete large outputs from the repository root.

3. **Test coverage**
   - Add unit tests for `scripts/reef_bathy_module.py` default Stumpf fallback.
   - Add coverage tests for malformed and incomplete EMODnet / Sentinel-2 inputs.

## 3. Medium-term improvements

1. **Modularize script logic**
   - Move shared logic from `scripts/reef_bathy_module.py` into package-safe modules under `src/`.
   - Replace runtime `sys.path.append(...)` imports with proper package exports.

2. **Package development environment**
   - Add `requirements-dev.txt` or `pip-tools` lock files for reproducible installs.
   - Ensure `pyproject.toml` and `requirements.txt` remain in sync.

3. **CI strengthening**
   - Run full `pytest` suite, not only a targeted regression test.
   - Add `ruff` or `flake8` checks as required failures rather than warnings.
   - Add a coverage threshold guard for critical modules.

## 4. Long-term improvements

1. **Data provenance and reproducibility**
   - Document input dataset sources and expected folder layout.
   - Add a reproducible pipeline example with sample data paths.

2. **Validation and benchmarking**
   - Add standardized accuracy tests against IH isobaths and EMODnet ground truth.
   - Record calibration performance over time and log versioned metrics.

3. **Deployment / UX**
   - Add a lightweight CLI wrapper for common workflows.
   - Add a developer-friendly `make` or `invoke` task file for setup and tests.
