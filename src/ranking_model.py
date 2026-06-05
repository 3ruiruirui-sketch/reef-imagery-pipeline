import os
import json
import hashlib
import pickle
import logging
import numpy as np

import math

try:
    from src.drift_monitor import observe as _observe_drift
except ImportError:
    _observe_drift = None  # drift_monitor is optional; inference remains functional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Terrain exposure modifier (Phase 4 integration)
# ---------------------------------------------------------------------------

# Dominant Atlantic swell direction for the Algarve coast (SW = 225°)
_ALGARVE_SWELL_DIR = 225.0


def terrain_exposure_modifier(
    slope_mean: float,
    aspect_mean: float,
    swell_direction: float = _ALGARVE_SWELL_DIR,
) -> float:
    """
    Physics-based BVI score multiplier derived from coastal terrain.

    A south-facing (180°) coast with low slope facing the dominant SW swell
    is moderately exposed; a sheltered north-facing cliff face would score 1.0.

    Returns a value in [0.5, 1.0] — never zeroes out an ML score.

    Args:
        slope_mean: mean terrain slope in degrees within the site buffer
        aspect_mean: mean terrain aspect in degrees (0=N, 90=E, 180=S, 270=W)
        swell_direction: incoming swell azimuth in degrees (default 225 = SW)
    """
    # Angular difference between coast aspect and swell origin
    diff = abs(aspect_mean - swell_direction)
    if diff > 180:
        diff = 360 - diff
    # exposure ∈ [-1, 1]: +1 = coast faces directly into swell, -1 = sheltered
    exposure = math.cos(math.radians(diff))
    # Normalise to [0, 1]: 0 = sheltered, 1 = fully exposed
    exposure_norm = (exposure + 1.0) / 2.0

    # Slope resuspension penalty: 0–15° mapped to [0, 1]
    slope_norm = min(slope_mean / 15.0, 1.0)

    # Combined modifier: exposure contributes 25%, slope 15%
    modifier = 1.0 - 0.25 * exposure_norm - 0.15 * slope_norm
    return max(0.5, min(1.0, modifier))

# Paths
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.join(MODELS_DIR, "feature_ranker_model.pkl")
METADATA_PATH = os.path.join(MODELS_DIR, "feature_ranker_metadata.json")

# Global cache
_RANKER_MODEL = None
_FEATURE_SCHEMA = None
_DISABLED_FEATURES = set()
_SCHEMA_FINGERPRINT = None
_IS_FALLBACK = False


def schema_fingerprint(features_list):
    """Stable hash of the canonical feature schema for quick drift comparison."""
    canonical = "|".join(sorted(features_list))
    return hashlib.md5(canonical.encode()).hexdigest()[:12]


def validate_schema(incoming_features, expected_schema, disabled=None):
    """
    Validate incoming features against the canonical schema.
    
    Returns:
        dict: {
            "ok": bool,          # True if all required features are present
            "missing": set,      # Required features not found in incoming
            "extra": set,        # Incoming features not in schema (may be harmless)
            "deprecated": set,   # Incoming features that are disabled/deprecated
            "type_errors": list, # Features with non-numeric values
        }
    """
    disabled = disabled or set()
    incoming_keys = set(incoming_features.keys())
    expected_keys = set(expected_schema)
    
    missing = expected_keys - incoming_keys
    extra = incoming_keys - expected_keys - disabled
    deprecated = incoming_keys & disabled
    
    # Cheap type check: values that will be passed to model must be numeric
    type_errors = []
    for feat in expected_schema:
        if feat in incoming_features:
            val = incoming_features[feat]
            if not isinstance(val, (int, float, np.integer, np.floating)):
                type_errors.append(f"{feat}: got {type(val).__name__}")
    
    return {
        "ok": len(missing) == 0 and len(type_errors) == 0,
        "missing": missing,
        "extra": extra,
        "deprecated": deprecated,
        "type_errors": type_errors,
    }

def _load_resources():
    """Loads the ML ranker model and metadata schema from disk."""
    global _RANKER_MODEL, _FEATURE_SCHEMA, _DISABLED_FEATURES, _SCHEMA_FINGERPRINT, _IS_FALLBACK
    
    if _RANKER_MODEL is not None and _FEATURE_SCHEMA is not None and not _IS_FALLBACK:
        return
        
    _IS_FALLBACK = True
    
    if os.path.exists(MODEL_PATH) and os.path.exists(METADATA_PATH):
        try:
            with open(MODEL_PATH, 'rb') as f:
                _RANKER_MODEL = pickle.load(f)
            with open(METADATA_PATH, 'r') as f:
                meta = json.load(f)
                _FEATURE_SCHEMA = meta.get("features", [])
                _DISABLED_FEATURES = set(meta.get("disabled_features", {}).keys())
                
            if not _FEATURE_SCHEMA:
                raise ValueError("No features list found in metadata JSON.")
            
            _SCHEMA_FINGERPRINT = schema_fingerprint(_FEATURE_SCHEMA)
            
            # Canonical model path: feature_ranker_model.pkl (schema-driven).
            # Legacy visibility_rf_bathy.pkl is NOT loaded here; it is deprecated
            # for the production ranking flow and retained only for standalone scripts.
            log.info(f"Loaded canonical ML Ranker v{meta.get('model_version', '?')}. "
                     f"Schema ({meta.get('schema_version', '1.x')}) "
                     f"[{_SCHEMA_FINGERPRINT}]: {_FEATURE_SCHEMA}")
            if _DISABLED_FEATURES:
                log.info(f"Disabled features (ignored): {_DISABLED_FEATURES}")
            _IS_FALLBACK = False
        except Exception as e:
            log.error(f"Failed to load ML ranker model or metadata: {e}")
            _RANKER_MODEL = None
            _FEATURE_SCHEMA = None
    else:
        log.warning(f"ML Ranker model or metadata missing. Using FALLBACK heuristic mode.")


def predict_score(features_dict, terrain_features=None):
    """
    Predicts the benthic visibility score using B02-only features.

    Uses a 6-feature schema derived from Sentinel-2 Band 02 (B02) plus
    bathymetry context.  Replaces the legacy multi-band S2 feature set
    (kd_b02, water_trans, signal_strength, cleanliness) with a unified
    B02-only representation: benthic_contrast, snr, fft_clean,
    edge_entropy, dyn_range, signal.

    Args:
        features_dict: spectral/optical features dict
        terrain_features: optional dict with keys slope_mean, aspect_mean,
                          and optionally swell_direction (degrees).
                          When provided, a physics-based terrain exposure
                          modifier is applied as a post-prediction multiplier.

    Returns:
        dict: {
            "score": float,
            "mode": "ML" or "Fallback",
            "features_used": dict,
            "terrain_modifier": float (if terrain_features provided),
        }
    """
    _load_resources()

    cloud_cov = features_dict.get('cloud_cover', 0.0)

    SCHEMA_ORDER = [
        "benthic_contrast",
        "snr",
        "fft_clean",
        "edge_entropy",
        "dyn_range",
        "signal",
    ]

    # Build B02-only features for the unified model
    features = {
        "benthic_contrast": features_dict.get("benthic_contrast",
                       features_dict.get("contrast_benthic_mean",
                       features_dict.get("contrast", 0.1))),
        "snr": features_dict.get("snr",
               features_dict.get("SNR_mean_16m",
               features_dict.get("signal_strength", 0))),
        "fft_clean": features_dict.get("fft_clean",
                     features_dict.get("cleanliness", 5000)),
        "edge_entropy": features_dict.get("edge_entropy",
                     features_dict.get("edge", 5.0)),
        "dyn_range": features_dict.get("dyn_range", 0.008),
        "signal": features_dict.get("signal",
              features_dict.get("raw_mean", 0.12)),
    }

    standard_features = {k: features[k] for k in SCHEMA_ORDER}
    
    # Hard filter
    if cloud_cov > 80.0:
        return {
            "score": 0.0,
            "mode": "HardFilter",
            "features_used": standard_features,
            "reason": "Cloud cover exceeded 80%"
        }

    # Try ML Inference
    if not _IS_FALLBACK and _FEATURE_SCHEMA:
        try:
            # Schema drift detection
            drift = validate_schema(standard_features, _FEATURE_SCHEMA, _DISABLED_FEATURES)
            
            # Critical: always log warnings for type errors and missing features
            if drift["type_errors"]:
                log.warning(f"Schema drift: type mismatch in features: {drift['type_errors']}")
            if not drift["ok"]:
                if drift["missing"]:
                    log.warning(f"Schema drift: missing required features {drift['missing']}. "
                                f"Falling back to heuristic.")
                raise ValueError(f"Schema drift: {drift}")
            
            # Build ordered vector based strictly on canonical schema
            vector = [standard_features[feat_name] for feat_name in _FEATURE_SCHEMA]
                
            import pandas as pd
            feature_df = pd.DataFrame([vector], columns=_FEATURE_SCHEMA)
            score = float(_RANKER_MODEL.predict(feature_df)[0])
            
            if _observe_drift is not None:
                _observe_drift(standard_features, score, _FEATURE_SCHEMA)
            result = {"score": score, "mode": "ML", "features_used": standard_features}
            if terrain_features:
                mod = terrain_exposure_modifier(
                    terrain_features.get("slope_mean", 0.0) or 0.0,
                    terrain_features.get("aspect_mean", 180.0) or 180.0,
                    terrain_features.get("swell_direction", _ALGARVE_SWELL_DIR),
                )
                result["score"] = round(score * mod, 6)
                result["terrain_modifier"] = round(mod, 4)
                result["terrain_features"] = terrain_features
            return result
        except Exception as e:
            log.error(f"ML Inference failed ({e}). Reverting to fallback.")
            
    # Fallback Heuristic (B02-only feature weights)
    benthic_contrast = features["benthic_contrast"]
    snr = features["snr"]
    fft_clean = features["fft_clean"]
    edge_entropy = features["edge_entropy"]
    dyn_range = features["dyn_range"]
    signal = features["signal"]

    # Normalise each feature to 0-1 range
    fft_norm = min(max(fft_clean, 1), 100000) / 100000.0
    edge_norm = min(max(edge_entropy, 0), 10.0) / 10.0
    dyn_norm = min(max(dyn_range, 0), 0.02) / 0.02
    snr_norm = min(max(snr, 0), 200.0) / 200.0
    contrast_norm = min(max(benthic_contrast, 0), 1.5) / 1.5
    signal_norm = min(max(signal, 0), 0.3) / 0.3

    score = (
        0.35 * fft_norm +           # FFT cleanliness
        0.25 * edge_norm +           # Edge entropy
        0.20 * dyn_norm +            # Dynamic range
        0.10 * snr_norm +            # SNR
        0.05 * contrast_norm +       # Benthic contrast
        0.05 * signal_norm           # Signal level
    )

    if fft_clean < 5000:
        score *= 0.1

    if _observe_drift is not None:
        _observe_drift(standard_features, float(score), _FEATURE_SCHEMA)
    result = {
        "score": float(score),
        "mode": "Fallback",
        "features_used": standard_features,
        "reason": "ML Model unavailable or failed",
    }
    if terrain_features:
        mod = terrain_exposure_modifier(
            terrain_features.get("slope_mean", 0.0) or 0.0,
            terrain_features.get("aspect_mean", 180.0) or 180.0,
            terrain_features.get("swell_direction", _ALGARVE_SWELL_DIR),
        )
        result["score"] = round(float(score) * mod, 6)
        result["terrain_modifier"] = round(mod, 4)
        result["terrain_features"] = terrain_features
    return result
