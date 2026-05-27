import os
import pytest
import numpy as np
from unittest.mock import patch, mock_open, MagicMock

from src.ranking_model import predict_score, _load_resources, validate_schema, schema_fingerprint

@pytest.fixture(autouse=True)
def reset_globals():
    import src.ranking_model as rm
    rm._RANKER_MODEL = None
    rm._FEATURE_SCHEMA = None
    rm._DISABLED_FEATURES = set()
    rm._SCHEMA_FINGERPRINT = None
    rm._IS_FALLBACK = False

def test_missing_model_fallback():
    """Test that if the model is missing, the system gracefully reverts to fallback."""
    with patch('os.path.exists', return_value=False):
        features = {
            'kd_b02': 0.045,
            'water_trans': 0.8,
            'contrast': 50.0,
            'signal_strength': 20.0,
            'cleanliness': 10000,
            'cloud_cover': 5.0
        }
        res = predict_score(features)
        
        assert res["mode"] == "Fallback"
        assert res["score"] > 0
        assert "ML Model unavailable" in res["reason"]
        assert res["features_used"]["kd_b02"] == 0.045

def test_extreme_cloud_cover():
    """Test hard filter for clouds > 80%."""
    features = {
        'cloud_cover': 85.0
    }
    res = predict_score(features)
    assert res["score"] == 0.0
    assert res["mode"] == "HardFilter"
    assert "Cloud cover exceeded 80%" in res["reason"]

def test_successful_ml_inference():
    """Test that ML inference works when schema matches."""
    # Mock model and metadata
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([0.95])
    
    mock_meta = '{"features": ["kd_b02", "water_trans"]}'
    
    with patch('os.path.exists', return_value=True), \
         patch('builtins.open', mock_open(read_data=mock_meta)), \
         patch('pickle.load', return_value=mock_model):
         
        features = {
            'kd_b02': 0.045,
            'water_trans': 0.8,
            'contrast': 50.0,
            'signal_strength': 20.0,
            'cleanliness': 10000,
            'cloud_cover': 5.0
        }
        res = predict_score(features)
        
        assert res["mode"] == "ML"
        assert res["score"] == 0.95
        # Ensure predict was called
        assert mock_model.predict.called
        
        # Verify schema order passed to predict
        called_args = mock_model.predict.call_args[0][0]
        # Should be a DataFrame with [kd_b02, water_trans]
        assert called_args.iloc[0, 0] == 0.045
        assert called_args.iloc[0, 1] == 0.8

def test_mismatched_schema():
    """Test that a missing required feature gracefully reverts to fallback."""
    mock_model = MagicMock()
    
    # Require a feature that the pipeline does not provide
    mock_meta = '{"features": ["kd_b02", "non_existent_feature"]}'
    
    with patch('os.path.exists', return_value=True), \
         patch('builtins.open', mock_open(read_data=mock_meta)), \
         patch('pickle.load', return_value=mock_model):
         
        features = {
            'kd_b02': 0.045,
            # Missing 'non_existent_feature'
        }
        res = predict_score(features)
        
        assert res["mode"] == "Fallback"
        assert not mock_model.predict.called

def test_bathymetry_features_present():
    """Test inference successfully maps bathymetry features to the schema."""
    features = {
        'kd_b02': 0.045,
        'water_trans': 0.8,
        'contrast': 50.0,
        'signal_strength': 20.0,
        'cleanliness': 10000,
        'cloud_cover': 5.0,
        'nearest_isobath_distance_m': 150.0,
        'bathymetry_zone_class': 'shallow_reef',
        'bathy_slope_proxy': 2.5
    }
    
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([0.88])
    mock_meta = '{"features": ["kd_b02", "nearest_isobath_distance_m", "bathymetry_zone_class", "bathy_slope_proxy"]}'
    
    with patch('os.path.exists', return_value=True), \
         patch('builtins.open', mock_open(read_data=mock_meta)), \
         patch('pickle.load', return_value=mock_model):
         
        res = predict_score(features)
        
        assert res["mode"] == "ML"
        assert res["score"] == 0.88
        
        called_args = mock_model.predict.call_args[0][0]
        assert called_args.iloc[0, 1] == 150.0
        assert called_args.iloc[0, 2] == 2.0  # Ordinal encoded for 'shallow_reef'
        assert called_args.iloc[0, 3] == 2.5

def test_bathymetry_features_missing():
    """Test inference safely handles np.inf or completely missing bathymetry features."""
    features = {
        'kd_b02': 0.045,
        'water_trans': 0.8,
        'contrast': 50.0,
        'signal_strength': 20.0,
        'cleanliness': 10000,
        'cloud_cover': 5.0,
        'nearest_isobath_distance_m': np.inf, # Simulated missing
        # missing bathymetry_zone_class entirely
    }
    
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([0.77])
    mock_meta = '{"features": ["kd_b02", "nearest_isobath_distance_m", "bathymetry_zone_class"]}'
    
    with patch('os.path.exists', return_value=True), \
         patch('builtins.open', mock_open(read_data=mock_meta)), \
         patch('pickle.load', return_value=mock_model):
         
        res = predict_score(features)
        
        assert res["mode"] == "ML"
        called_args = mock_model.predict.call_args[0][0]
        # inf should become -1.0
        assert called_args.iloc[0, 1] == -1.0
        # missing class should become 0 ('unknown')
        assert called_args.iloc[0, 2] == 0.0

def test_contrast_ignored_in_ml_schema():
    """Test that contrast is safely ignored when not in model schema (v1.2+)."""
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([0.72])
    
    # Schema without contrast (matches v1.2 metadata)
    mock_meta = '{"features": ["kd_b02", "water_trans", "signal_strength", "cleanliness"], "disabled_features": {"contrast": "non-discriminative"}}'
    
    with patch('os.path.exists', return_value=True), \
         patch('builtins.open', mock_open(read_data=mock_meta)), \
         patch('pickle.load', return_value=mock_model):
         
        # Caller still sends contrast (backward compat)
        features = {
            'kd_b02': 0.045,
            'water_trans': 0.35,
            'contrast': 0.8,
            'signal_strength': 75.0,
            'cleanliness': 9000,
            'cloud_cover': 5.0
        }
        res = predict_score(features)
        
        assert res["mode"] == "ML"
        assert res["score"] == 0.72
        
        # Verify contrast is NOT passed to model predict
        called_args = mock_model.predict.call_args[0][0]
        assert list(called_args.columns) == ["kd_b02", "water_trans", "signal_strength", "cleanliness"]
        assert called_args.shape[1] == 4  # Only 4 features, no contrast


def test_legacy_caller_without_contrast():
    """Test that callers who do NOT send contrast still work fine."""
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([0.65])
    
    mock_meta = '{"features": ["kd_b02", "water_trans", "signal_strength", "cleanliness"]}'
    
    with patch('os.path.exists', return_value=True), \
         patch('builtins.open', mock_open(read_data=mock_meta)), \
         patch('pickle.load', return_value=mock_model):
         
        # No contrast provided at all
        features = {
            'kd_b02': 0.050,
            'water_trans': 0.30,
            'signal_strength': 50.0,
            'cleanliness': 7000,
        }
        res = predict_score(features)
        
        assert res["mode"] == "ML"
        assert res["score"] == 0.65
        # contrast should still appear in features_used with default
        assert 'contrast' in res["features_used"]


def test_predict_score_output_contract():
    """Test that predict_score always returns the same stable contract regardless of contrast."""
    with patch('os.path.exists', return_value=False):
        features = {
            'kd_b02': 0.045,
            'water_trans': 0.8,
            'contrast': 0.8,
            'signal_strength': 20.0,
            'cleanliness': 10000,
        }
        res = predict_score(features)
        
        # Contract: always returns score, mode, features_used
        assert "score" in res
        assert "mode" in res
        assert "features_used" in res
        assert isinstance(res["score"], float)
        assert res["mode"] in ("ML", "Fallback", "HardFilter")


def test_deterministic_training():
    """Test that training the ML model with the same fixed seed yields identical feature importances."""
    from sklearn.ensemble import RandomForestRegressor
    import numpy as np
    
    # Dummy data (4 features, no contrast)
    np.random.seed(42)
    X = np.random.rand(100, 4)
    y = np.random.randint(0, 2, 100)
    
    # Train model 1
    model1 = RandomForestRegressor(n_estimators=10, max_depth=3, random_state=42)
    model1.fit(X, y)
    imp1 = model1.feature_importances_
    
    # Train model 2 (identical config)
    model2 = RandomForestRegressor(n_estimators=10, max_depth=3, random_state=42)
    model2.fit(X, y)
    imp2 = model2.feature_importances_
    
    # Verify deterministic behavior
    np.testing.assert_array_equal(imp1, imp2)


def test_training_without_contrast():
    """Test that train_feature_ranker FEATURE_COLS no longer includes contrast."""
    train_ranker = pytest.importorskip(
        "train_feature_ranker",
        reason="scripts/train_feature_ranker.py not installed"
    )
    FEATURE_COLS = train_ranker.FEATURE_COLS
    DISABLED_FEATURES = train_ranker.DISABLED_FEATURES

    assert 'contrast' not in FEATURE_COLS
    assert 'contrast' in DISABLED_FEATURES
    assert len(FEATURE_COLS) == 4


def test_canonical_model_path_loaded():
    """Test that ranking_model.py only loads the canonical model path."""
    import src.ranking_model as rm
    
    # Verify constants point to canonical path
    assert rm.MODEL_PATH.endswith("feature_ranker_model.pkl")
    assert rm.METADATA_PATH.endswith("feature_ranker_metadata.json")
    assert "visibility_rf_bathy" not in rm.MODEL_PATH


def test_metadata_marks_canonical_and_deprecated():
    """Test that metadata explicitly marks canonical status and deprecated models."""
    import json
    meta_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'feature_ranker_metadata.json')

    if not os.path.exists(meta_path):
        pytest.skip("models/feature_ranker_metadata.json not present")

    with open(meta_path, 'r') as f:
        meta = json.load(f)
    
    # Canonical markers
    assert meta.get("canonical") is True
    assert "schema_version" in meta
    
    # Deprecated models documented
    assert "deprecated_models" in meta
    assert "visibility_rf_bathy.pkl" in meta["deprecated_models"]
    
    # Active schema does not include contrast
    assert "contrast" not in meta["features"]
    assert "contrast" in meta.get("disabled_features", {})


def test_legacy_model_not_loaded_by_ranking():
    """Test that predict_score never loads the legacy visibility_rf_bathy model."""
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([0.60])
    
    mock_meta = '{"features": ["kd_b02", "water_trans", "signal_strength", "cleanliness"], "canonical": true, "schema_version": "2.0"}'
    
    with patch('os.path.exists', return_value=True) as mock_exists, \
         patch('builtins.open', mock_open(read_data=mock_meta)), \
         patch('pickle.load', return_value=mock_model):
         
        res = predict_score({'kd_b02': 0.05, 'water_trans': 0.3, 'signal_strength': 40, 'cleanliness': 8000})
        
        assert res["mode"] == "ML"
        # Verify os.path.exists was never called with the legacy path
        for call in mock_exists.call_args_list:
            path_arg = call[0][0] if call[0] else ""
            assert "visibility_rf_bathy" not in path_arg


# =============================================================================
# Schema Drift Detection Tests
# =============================================================================

def test_validate_schema_exact_match():
    """Test that exact schema match returns ok=True with no drift."""
    incoming = {'kd_b02': 0.05, 'water_trans': 0.3, 'signal_strength': 40.0, 'cleanliness': 8000.0}
    schema = ['kd_b02', 'water_trans', 'signal_strength', 'cleanliness']
    
    result = validate_schema(incoming, schema)
    
    assert result["ok"] is True
    assert result["missing"] == set()
    assert result["extra"] == set()
    assert result["deprecated"] == set()
    assert result["type_errors"] == []


def test_validate_schema_missing_required():
    """Test that missing required features are detected."""
    incoming = {'kd_b02': 0.05, 'water_trans': 0.3}
    schema = ['kd_b02', 'water_trans', 'signal_strength', 'cleanliness']
    
    result = validate_schema(incoming, schema)
    
    assert result["ok"] is False
    assert result["missing"] == {'signal_strength', 'cleanliness'}


def test_validate_schema_extra_features():
    """Test that extra features are detected but don't block."""
    incoming = {'kd_b02': 0.05, 'water_trans': 0.3, 'signal_strength': 40.0,
                'cleanliness': 8000.0, 'some_new_feature': 1.0}
    schema = ['kd_b02', 'water_trans', 'signal_strength', 'cleanliness']
    
    result = validate_schema(incoming, schema)
    
    assert result["ok"] is True  # Extra features don't block
    assert 'some_new_feature' in result["extra"]


def test_validate_schema_deprecated_features():
    """Test that deprecated features are classified separately from extras."""
    incoming = {'kd_b02': 0.05, 'water_trans': 0.3, 'signal_strength': 40.0,
                'cleanliness': 8000.0, 'contrast': 0.8}
    schema = ['kd_b02', 'water_trans', 'signal_strength', 'cleanliness']
    disabled = {'contrast'}
    
    result = validate_schema(incoming, schema, disabled=disabled)
    
    assert result["ok"] is True
    assert 'contrast' in result["deprecated"]
    assert 'contrast' not in result["extra"]


def test_validate_schema_type_mismatch():
    """Test that non-numeric feature values are caught."""
    incoming = {'kd_b02': 0.05, 'water_trans': "bad_string",
                'signal_strength': 40.0, 'cleanliness': 8000.0}
    schema = ['kd_b02', 'water_trans', 'signal_strength', 'cleanliness']
    
    result = validate_schema(incoming, schema)
    
    assert result["ok"] is False
    assert any("water_trans" in e for e in result["type_errors"])


def test_schema_fingerprint_stable():
    """Test that fingerprint is stable and order-independent."""
    fp1 = schema_fingerprint(['kd_b02', 'water_trans', 'signal_strength', 'cleanliness'])
    fp2 = schema_fingerprint(['cleanliness', 'signal_strength', 'water_trans', 'kd_b02'])
    fp3 = schema_fingerprint(['kd_b02', 'water_trans', 'signal_strength'])  # Different schema
    
    assert fp1 == fp2  # Order-independent
    assert fp1 != fp3  # Different schemas produce different fingerprints
    assert len(fp1) == 12  # Fixed length


def test_drift_missing_feature_triggers_fallback():
    """Test that missing required feature in ML path triggers fallback."""
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([0.70])
    
    # Schema requires 'nonexistent_required' which won't be in standard_features
    mock_meta = '{"features": ["kd_b02", "water_trans", "nonexistent_required"], "disabled_features": {}}'
    
    with patch('os.path.exists', return_value=True), \
         patch('builtins.open', mock_open(read_data=mock_meta)), \
         patch('pickle.load', return_value=mock_model):
         
        res = predict_score({'kd_b02': 0.05, 'water_trans': 0.3, 'signal_strength': 40, 'cleanliness': 8000})
        
        # Should fall back due to missing required feature
        assert res["mode"] == "Fallback"
        assert not mock_model.predict.called


def test_drift_extra_features_do_not_block():
    """Test that extra features in caller do not block ML inference."""
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([0.85])
    
    mock_meta = '{"features": ["kd_b02", "water_trans"], "disabled_features": {"contrast": "disabled"}}'
    
    with patch('os.path.exists', return_value=True), \
         patch('builtins.open', mock_open(read_data=mock_meta)), \
         patch('pickle.load', return_value=mock_model):
         
        # Caller sends many extra features + deprecated contrast
        res = predict_score({
            'kd_b02': 0.05, 'water_trans': 0.3, 'contrast': 0.8,
            'signal_strength': 40, 'cleanliness': 8000,
            'nearest_isobath_distance_m': 150.0
        })
        
        assert res["mode"] == "ML"
        assert res["score"] == 0.85
