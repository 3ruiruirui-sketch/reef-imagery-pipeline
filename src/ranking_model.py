import os
import json
import hashlib
import pickle
import logging
import numpy as np

from src.drift_monitor import observe as _observe_drift

log = logging.getLogger(__name__)

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
    
    if _RANKER_MODEL is not None and _FEATURE_SCHEMA is not None:
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


def predict_score(features_dict):
    """
    Predicts the benthic visibility score given extracted features.
    
    Returns:
        dict: {
            "score": float,
            "mode": "ML" or "Fallback",
            "features_used": dict
        }
    """
    _load_resources()
    
    # Standardize dictionary mapping (some callers use slightly different names)
    kd = features_dict.get('kd_b02', features_dict.get('kd490_seasonal', 0.08))
    transmittance = features_dict.get('water_transmittance_twoway', features_dict.get('water_trans', 0.5))
    contrast = features_dict.get('contrast_benthic_mean', features_dict.get('contrast', 0.0))
    snr = features_dict.get('SNR_mean_16m', features_dict.get('signal_strength', 15.0))
    cleanliness = features_dict.get('cleanliness', 5000)
    cloud_cov = features_dict.get('cloud_cover', 0.0)
    
    # Bathymetry Features with safe fallback defaults
    def _safe_float(val, default=-1.0):
        try:
            v = float(val)
            if np.isnan(v) or np.isinf(v):
                return default
            return v
        except (TypeError, ValueError):
            return default

    dist_isobath = _safe_float(features_dict.get('nearest_isobath_distance_m'), -1.0)
    depth_isobath = _safe_float(features_dict.get('nearest_isobath_depth_m'), -1.0)
    dist_10m = _safe_float(features_dict.get('dist_to_isobath_10m'), -1.0)
    dist_20m = _safe_float(features_dict.get('dist_to_isobath_20m'), -1.0)
    dist_30m = _safe_float(features_dict.get('dist_to_isobath_30m'), -1.0)
    dist_50m = _safe_float(features_dict.get('dist_to_isobath_50m'), -1.0)
    dist_100m = _safe_float(features_dict.get('dist_to_isobath_100m'), -1.0)
    zone_class_str = features_dict.get('bathymetry_zone_class', 'unknown')
    slope_proxy = _safe_float(features_dict.get('bathy_slope_proxy', features_dict.get('bathymetry_slope_proxy')), -1.0)
    contour_dens = _safe_float(features_dict.get('contour_density_proxy'), -1.0)
    n_isobaths = _safe_float(features_dict.get('n_isobaths_in_aoi'), 0.0)
    
    # Ordinal encode the bathymetry zone
    ZONE_MAP = {
        "unknown": 0,
        "very_shallow": 1,
        "shallow_reef": 2,
        "nearshore_mid": 3,
        "mid_depth": 4,
        "offshore": 5
    }
    zone_encoded = ZONE_MAP.get(str(zone_class_str), 0)
    
    
    standard_features = {
        'kd_b02': kd,
        'water_trans': transmittance,
        'contrast': contrast,
        'signal_strength': snr,
        'cleanliness': cleanliness,
        'nearest_isobath_distance_m': dist_isobath,
        'nearest_isobath_depth_m': depth_isobath,
        'dist_to_isobath_10m': dist_10m,
        'dist_to_isobath_20m': dist_20m,
        'dist_to_isobath_30m': dist_30m,
        'dist_to_isobath_50m': dist_50m,
        'dist_to_isobath_100m': dist_100m,
        'bathymetry_zone_class': zone_encoded,
        'bathy_slope_proxy': slope_proxy,
        'contour_density_proxy': contour_dens,
        'n_isobaths_in_aoi': n_isobaths
    }
    
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
            
            _observe_drift(standard_features, score, _FEATURE_SCHEMA)
            return {
                "score": score,
                "mode": "ML",
                "features_used": standard_features
            }
        except Exception as e:
            log.error(f"ML Inference failed ({e}). Reverting to fallback.")
            
    # Fallback Heuristic
    clarity_score = max(0, 0.045 / max(kd, 0.001))
    edge_score = contrast
    stability_score = np.log10(max(cleanliness, 1))
    semantic_score_proxy = snr / 100.0
    
    score = (
        0.35 * clarity_score + 
        0.25 * edge_score + 
        0.20 * semantic_score_proxy + 
        0.20 * stability_score
    )
    
    if cleanliness < 5000:
        score *= 0.1
    
    _observe_drift(standard_features, float(score), _FEATURE_SCHEMA)
    return {
        "score": float(score),
        "mode": "Fallback",
        "features_used": standard_features,
        "reason": "ML Model unavailable or failed"
    }
