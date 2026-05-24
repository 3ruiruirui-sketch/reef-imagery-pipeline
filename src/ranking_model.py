import os
import json
import pickle
import logging
import numpy as np

log = logging.getLogger(__name__)

# Paths
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.join(MODELS_DIR, "feature_ranker_model.pkl")
METADATA_PATH = os.path.join(MODELS_DIR, "feature_ranker_metadata.json")

# Global cache
_RANKER_MODEL = None
_FEATURE_SCHEMA = None
_IS_FALLBACK = False

def _load_resources():
    """Loads the ML ranker model and metadata schema from disk."""
    global _RANKER_MODEL, _FEATURE_SCHEMA, _IS_FALLBACK
    
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
                
            if not _FEATURE_SCHEMA:
                raise ValueError("No features list found in metadata JSON.")
                
            log.info(f"Loaded ML Ranker model. Schema: {_FEATURE_SCHEMA}")
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
    
    standard_features = {
        'kd_b02': kd,
        'water_trans': transmittance,
        'contrast': contrast,
        'signal_strength': snr,
        'cleanliness': cleanliness
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
            # Schema Validation: Build ordered array based strictly on metadata
            vector = []
            for feat_name in _FEATURE_SCHEMA:
                if feat_name not in standard_features:
                    raise ValueError(f"Missing required feature '{feat_name}' for ML inference.")
                vector.append(standard_features[feat_name])
                
            import pandas as pd
            feature_df = pd.DataFrame([vector], columns=_FEATURE_SCHEMA)
            score = float(_RANKER_MODEL.predict(feature_df)[0])
            
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
        
    return {
        "score": float(score),
        "mode": "Fallback",
        "features_used": standard_features,
        "reason": "ML Model unavailable or failed"
    }
