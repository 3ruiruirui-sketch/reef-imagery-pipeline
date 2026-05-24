import os
import pickle
import logging
import numpy as np

log = logging.getLogger(__name__)

# Path to the trained ML ranking model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "feature_ranker_model.pkl")

# Global cache for the loaded model to avoid reloading on every predict call
_RANKER_MODEL = None
_IS_FALLBACK = False

def load_ranker_model():
    """Loads the ML ranker model from disk. Uses a fallback if missing."""
    global _RANKER_MODEL, _IS_FALLBACK
    if _RANKER_MODEL is not None:
        return _RANKER_MODEL
        
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, 'rb') as f:
                _RANKER_MODEL = pickle.load(f)
            log.info(f"Successfully loaded ML Ranker model from {MODEL_PATH}")
            _IS_FALLBACK = False
        except Exception as e:
            log.error(f"Failed to load ranker model {MODEL_PATH}: {e}")
            _RANKER_MODEL = None
    else:
        log.warning(f"ML Ranker model not found at {MODEL_PATH}. Using fallback heuristic weights.")
        _IS_FALLBACK = True
        
    return _RANKER_MODEL

def predict_score(features_dict):
    """
    Predicts the benthic visibility score for a single image given its extracted features.
    
    If the trained ML model is available, it formats the features as an array and runs predict().
    Otherwise, it uses the fallback heuristic weighting suggested in the architectural plan.
    """
    load_ranker_model()
    
    # 1. Standardize the feature dictionary (handle missing keys gracefully)
    kd = features_dict.get('kd_b02', features_dict.get('kd490_seasonal', 0.08))
    transmittance = features_dict.get('water_transmittance_twoway', features_dict.get('water_trans', 0.5))
    contrast = features_dict.get('contrast_benthic_mean', features_dict.get('contrast', 0.0))
    snr = features_dict.get('SNR_mean_16m', features_dict.get('signal_strength', 15.0))
    cleanliness = features_dict.get('cleanliness', 5000)
    cloud_cov = features_dict.get('cloud_cover', 0.0)
    
    # Hard filter for extreme clouds
    if cloud_cov > 80.0:
        return 0.0

    if not _IS_FALLBACK and hasattr(_RANKER_MODEL, 'predict'):
        # Prepare feature vector for the ML model
        # The expected feature order must match the training script!
        # Example order: [kd, transmittance, contrast, snr, cleanliness]
        feature_vector = np.array([[kd, transmittance, contrast, snr, cleanliness]])
        try:
            score = _RANKER_MODEL.predict(feature_vector)[0]
            return float(score)
        except Exception as e:
            log.error(f"ML Model prediction failed: {e}. Reverting to fallback.")
            
    # 2. Fallback Heuristic Scoring (if no ML model is deployed yet)
    # This aligns with the new proposed BVI_new formula structure
    
    # Clarity proxy (low kd is good)
    clarity_score = max(0, 0.045 / max(kd, 0.001))
    
    # Benthic edge sharpness / contrast
    edge_score = contrast
    
    # Stability (Cleanliness log)
    stability_score = np.log10(max(cleanliness, 1))
    
    # Semantic score placeholder (would ideally come from Siamese CNN)
    semantic_score_proxy = snr / 100.0
    
    # BVI_new = 0.35 * clarity + 0.25 * edge_sharpness + 0.20 * semantic_score + 0.20 * stability
    score = (
        0.35 * clarity_score + 
        0.25 * edge_score + 
        0.20 * semantic_score_proxy + 
        0.20 * stability_score
    )
    
    # Penalty for extreme glint / noise
    if cleanliness < 5000:
        score *= 0.1
        
    return float(score)
