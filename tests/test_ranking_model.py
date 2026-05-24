import pytest
import numpy as np
from unittest.mock import patch, mock_open, MagicMock

from src.ranking_model import predict_score, _load_resources

@pytest.fixture(autouse=True)
def reset_globals():
    import src.ranking_model as rm
    rm._RANKER_MODEL = None
    rm._FEATURE_SCHEMA = None
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

def test_deterministic_training():
    """Test that training the ML model with the same fixed seed yields identical feature importances."""
    from sklearn.ensemble import RandomForestRegressor
    import numpy as np
    
    # Dummy data
    np.random.seed(42)
    X = np.random.rand(100, 5)
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
