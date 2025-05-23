"""
Tests for model implementations
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os
import tempfile
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.ensemble_model import EnsembleModel
from models.bdlstm_model import BidirectionalLSTMModel
from models.catboost_model import CatBoostModel
from utils.config import Config


class TestBidirectionalLSTMModel:
    """Test cases for BidirectionalLSTMModel"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        np.random.seed(42)
        X = np.random.random((100, 10))
        y = np.random.choice([0, 1], 100)
        return X, y
    
    @pytest.fixture
    def config(self):
        """Create test configuration"""
        return Config()
    
    def test_model_initialization(self, config):
        """Test BDLSTM model initialization"""
        input_shape = (10,)
        model = BidirectionalLSTMModel(input_shape, config)
        
        assert model.input_shape == input_shape
        assert model.config == config
        assert model.model is not None
        assert model.history is None
    
    def test_model_architecture(self, config):
        """Test model architecture creation"""
        input_shape = (10,)
        model = BidirectionalLSTMModel(input_shape, config)
        
        # Check model structure
        assert len(model.model.layers) > 5  # Should have multiple layers
        assert model.model.input_shape == (None, 10)
        assert model.model.output_shape == (None, 1)
        
        # Check model compilation
        assert model.model.optimizer is not None
        assert model.model.loss == 'binary_crossentropy'
    
    @patch('tensorflow.keras.models.Model.fit')
    def test_model_training(self, mock_fit, sample_data, config):
        """Test model training"""
        X, y = sample_data
        X_train, X_val = X[:80], X[80:]
        y_train, y_val = y[:80], y[80:]
        
        input_shape = (X.shape[1],)
        model = BidirectionalLSTMModel(input_shape, config)
        
        # Mock training history
        mock_history = MagicMock()
        mock_history.history = {'loss': [0.5, 0.4], 'accuracy': [0.8, 0.9]}
        mock_fit.return_value = mock_history
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=2,
            batch_size=16
        )
        
        # Check training was called
        mock_fit.assert_called_once()
        assert history == mock_history
        assert model.history == mock_history
    
    @patch('tensorflow.keras.models.Model.predict')
    def test_model_prediction(self, mock_predict, sample_data, config):
        """Test model predictions"""
        X, y = sample_data
        input_shape = (X.shape[1],)
        model = BidirectionalLSTMModel(input_shape, config)
        
        # Mock predictions
        mock_predict.return_value = np.array([[0.3], [0.7], [0.4]])
        
        # Test binary predictions
        predictions = model.predict(X[:3])
        expected_predictions = np.array([0, 1, 0])
        
        assert predictions.shape == (3,)
        np.testing.assert_array_equal(predictions, expected_predictions)
        
        # Test probability predictions
        probabilities = model.predict_proba(X[:3])
        
        assert probabilities.shape == (3, 2)
        assert np.allclose(probabilities.sum(axis=1), 1.0)  # Probabilities sum to 1
    
    def test_model_config_retrieval(self, config):
        """Test model configuration retrieval"""
        input_shape = (10,)
        model = BidirectionalLSTMModel(input_shape, config)
        
        model_config = model.get_model_config()
        
        assert 'input_shape' in model_config
        assert 'lstm_units_1' in model_config
        assert 'total_parameters' in model_config
        assert model_config['input_shape'] == input_shape


class TestCatBoostModel:
    """Test cases for CatBoostModel"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        np.random.seed(42)
        X = pd.DataFrame(np.random.random((100, 10)), 
                        columns=[f'feature_{i}' for i in range(10)])
        y = np.random.choice([0, 1], 100)
        return X, y
    
    @pytest.fixture
    def config(self):
        """Create test configuration"""
        return Config()
    
    def test_model_initialization(self, config):
        """Test CatBoost model initialization"""
        model = CatBoostModel(config)
        
        assert model.config == config
        assert model.model is not None
        assert not model.is_fitted
        assert model.feature_names is None
    
    @patch('catboost.CatBoostClassifier.fit')
    def test_model_training(self, mock_fit, sample_data, config):
        """Test model training"""
        X, y = sample_data
        X_train, X_val = X[:80], X[80:]
        y_train, y_val = y[:80], y[80:]
        
        model = CatBoostModel(config)
        
        # Train model
        model.fit(X_train, y_train, validation_data=(X_val, y_val))
        
        # Check training was called
        mock_fit.assert_called_once()
        assert model.is_fitted
        assert model.feature_names == list(X_train.columns)
    
    @patch('catboost.CatBoostClassifier.predict')
    @patch('catboost.CatBoostClassifier.predict_proba')
    def test_model_prediction(self, mock_predict_proba, mock_predict, sample_data, config):
        """Test model predictions"""
        X, y = sample_data
        model = CatBoostModel(config)
        model.is_fitted = True  # Mock fitted state
        
        # Mock predictions
        mock_predict.return_value = np.array([0, 1, 0])
        mock_predict_proba.return_value = np.array([[0.7, 0.3], [0.2, 0.8], [0.6, 0.4]])
        
        # Test binary predictions
        predictions = model.predict(X[:3])
        expected_predictions = np.array([0, 1, 0])
        
        np.testing.assert_array_equal(predictions, expected_predictions)
        
        # Test probability predictions
        probabilities = model.predict_proba(X[:3])
        
        assert probabilities.shape == (3, 2)
        assert np.allclose(probabilities.sum(axis=1), 1.0)
    
    @patch('catboost.CatBoostClassifier.get_feature_importance')
    def test_feature_importance(self, mock_importance, sample_data, config):
        """Test feature importance functionality"""
        X, y = sample_data
        model = CatBoostModel(config)
        model.is_fitted = True
        model.feature_names = list(X.columns)
        
        # Mock feature importance
        mock_importance.return_value = np.array([0.1, 0.2, 0.15, 0.05, 0.3, 
                                               0.08, 0.12, 0.04, 0.06, 0.02])
        
        importance = model.get_feature_importance()
        importance_df = model.get_feature_importance_df()
        
        assert len(importance) == 10
        assert len(importance_df) == 10
        assert 'feature' in importance_df.columns
        assert 'importance' in importance_df.columns
        
        # Check sorting
        assert importance_df['importance'].is_monotonic_decreasing
    
    def test_model_info(self, config):
        """Test model information retrieval"""
        model = CatBoostModel(config)
        
        # Test unfitted model
        info = model.get_model_info()
        assert info['status'] == 'Model not fitted'
        
        # Test fitted model (mock)
        model.is_fitted = True
        model.feature_names = ['feature_0', 'feature_1']
        with patch.object(model.model, 'get_param') as mock_param:
            mock_param.side_effect = lambda x: {'iterations': 1000, 
                                              'learning_rate': 0.1, 
                                              'depth': 6}[x]
            with patch.object(model.model, 'get_best_score', return_value={'validation': 0.95}):
                with patch.object(model.model, 'get_best_iteration', return_value=500):
                    with patch.object(model.model, 'tree_count_', 500):
                        info = model.get_model_info()
        
        assert info['model_type'] == 'CatBoost'
        assert info['is_fitted'] == True
        assert info['num_features'] == 2


class TestEnsembleModel:
    """Test cases for EnsembleModel"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        np.random.seed(42)
        X = pd.DataFrame(np.random.random((100, 10)), 
                        columns=[f'feature_{i}' for i in range(10)])
        y = np.random.choice([0, 1], 100)
        return X, y
    
    @pytest.fixture
    def config(self):
        """Create test configuration"""
        return Config()
    
    def test_ensemble_initialization(self, config):
        """Test ensemble model initialization"""
        model = EnsembleModel(config)
        
        assert model.config == config
        assert model.bdlstm_model is None
        assert model.catboost_model is None
        assert not model.is_fitted
        assert len(model.ensemble_weights) == 2
        assert sum(model.ensemble_weights) == pytest.approx(1.0, rel=1e-6)
    
    @patch('src.models.ensemble_model.BidirectionalLSTMModel')
    @patch('src.models.ensemble_model.CatBoostModel')
    def test_ensemble_training(self, mock_catboost, mock_bdlstm, sample_data, config):
        """Test ensemble model training"""
        X, y = sample_data
        X_train, X_val = X[:80], X[80:]
        y_train, y_val = y[:80], y[80:]
        
        # Mock individual models
        mock_bdlstm_instance = MagicMock()
        mock_catboost_instance = MagicMock()
        mock_bdlstm.return_value = mock_bdlstm_instance
        mock_catboost.return_value = mock_catboost_instance
        
        # Mock fit methods
        mock_history = MagicMock()
        mock_history.history = {'loss': [0.5], 'accuracy': [0.8]}
        mock_bdlstm_instance.fit.return_value = mock_history
        mock_catboost_instance.fit.return_value = mock_catboost_instance
        
        model = EnsembleModel(config)
        
        # Train ensemble
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val))
        
        # Check that both models were trained
        mock_bdlstm_instance.fit.assert_called_once()
        mock_catboost_instance.fit.assert_called_once()
        assert model.is_fitted
    
    def test_ensemble_prediction_not_fitted(self, sample_data, config):
        """Test error handling for unfitted model"""
        X, y = sample_data
        model = EnsembleModel(config)
        
        with pytest.raises(ValueError, match="Model must be fitted"):
            model.predict(X[:5])
        
        with pytest.raises(ValueError, match="Model must be fitted"):
            model.predict_proba(X[:5])
    
    @patch('src.models.ensemble_model.BidirectionalLSTMModel')
    @patch('src.models.ensemble_model.CatBoostModel')
    def test_ensemble_prediction(self, mock_catboost, mock_bdlstm, sample_data, config):
        """Test ensemble predictions"""
        X, y = sample_data
        
        # Mock individual models
        mock_bdlstm_instance = MagicMock()
        mock_catboost_instance = MagicMock()
        mock_bdlstm.return_value = mock_bdlstm_instance
        mock_catboost.return_value = mock_catboost_instance
        
        # Mock predictions
        mock_bdlstm_instance.predict.return_value = np.array([0.3, 0.7, 0.4])
        mock_catboost_instance.predict_proba.return_value = np.array([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4]])
        
        model = EnsembleModel(config)
        model.bdlstm_model = mock_bdlstm_instance
        model.catboost_model = mock_catboost_instance
        model.is_fitted = True
        
        # Test binary predictions
        predictions = model.predict(X[:3])
        
        # Expected ensemble predictions based on weights [0.6, 0.4]
        # Sample 0: 0.6 * 0.3 + 0.4 * 0.2 = 0.26 -> 0
        # Sample 1: 0.6 * 0.7 + 0.4 * 0.7 = 0.70 -> 1
        # Sample 2: 0.6 * 0.4 + 0.4 * 0.4 = 0.40 -> 0
        expected_predictions = np.array([0, 1, 0])
        
        np.testing.assert_array_equal(predictions, expected_predictions)
        
        # Test probability predictions
        probabilities = model.predict_proba(X[:3])
        
        assert probabilities.shape == (3, 2)
        assert np.allclose(probabilities.sum(axis=1), 1.0)
    
    def test_ensemble_save_load(self, config):
        """Test ensemble model save/load functionality"""
        model = EnsembleModel(config)
        
        # Test error for unfitted model
        with pytest.raises(ValueError, match="Model must be fitted"):
            with tempfile.TemporaryDirectory() as temp_dir:
                model.save(temp_dir)
        
        # Mock fitted state for save test
        model.is_fitted = True
        mock_bdlstm = MagicMock()
        mock_catboost = MagicMock()
        mock_bdlstm.model = MagicMock()
        mock_catboost.model = MagicMock()
        model.bdlstm_model = mock_bdlstm
        model.catboost_model = mock_catboost
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock save methods
            with patch('tensorflow.keras.models.Model.save'):
                with patch('joblib.dump'):
                    model.save(temp_dir)
                    # Test passes if no exception is raised
    
    @patch('src.models.ensemble_model.BidirectionalLSTMModel')
    @patch('src.models.ensemble_model.CatBoostModel')
    def test_individual_model_evaluation(self, mock_catboost, mock_bdlstm, sample_data, config):
        """Test individual model evaluation within ensemble"""
        X, y = sample_data
        
        # Mock individual models
        mock_bdlstm_instance = MagicMock()
        mock_catboost_instance = MagicMock()
        mock_bdlstm.return_value = mock_bdlstm_instance
        mock_catboost.return_value = mock_catboost_instance
        
        # Mock predictions
        mock_bdlstm_instance.predict.return_value = np.array([0.4, 0.6, 0.3])
        mock_catboost_instance.predict.return_value = np.array([0, 1, 0])
        
        model = EnsembleModel(config)
        model.bdlstm_model = mock_bdlstm_instance
        model.catboost_model = mock_catboost_instance
        model.is_fitted = True
        
        # Mock ensemble predictions
        with patch.object(model, 'predict', return_value=np.array([0, 1, 0])):
            results = model.evaluate_individual_models(X[:3], y[:3])
        
        assert 'BDLSTM' in results
        assert 'CatBoost' in results
        assert 'Ensemble' in results
        
        for model_name in ['BDLSTM', 'CatBoost', 'Ensemble']:
            assert 'accuracy' in results[model_name]
            assert 'precision' in results[model_name]
            assert 'recall' in results[model_name]
            assert 'f1_score' in results[model_name]


class TestModelIntegration:
    """Integration tests for model components"""
    
    @pytest.fixture
    def sample_data_large(self):
        """Create larger sample dataset for integration testing"""
        np.random.seed(42)
        n_samples = 1000
        n_features = 15
        
        X = pd.DataFrame(
            np.random.random((n_samples, n_features)),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        
        # Create somewhat realistic target based on features
        y = (X['feature_0'] + X['feature_1'] * 2 + 
             np.random.normal(0, 0.1, n_samples) > 1.0).astype(int)
        
        return X, y
    
    def test_end_to_end_pipeline(self, sample_data_large):
        """Test complete model pipeline integration"""
        X, y = sample_data_large
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        
        config = Config()
        
        # Test individual models
        # 1. Test BDLSTM model
        bdlstm_model = BidirectionalLSTMModel((X_train.shape[1],), config)
        
        # Mock training for speed (actual training takes too long for tests)
        with patch.object(bdlstm_model.model, 'fit') as mock_fit:
            mock_history = MagicMock()
            mock_history.history = {'loss': [0.5, 0.4], 'val_loss': [0.6, 0.5]}
            mock_fit.return_value = mock_history
            
            history = bdlstm_model.fit(
                X_train.values, y_train.values,
                validation_data=(X_val.values, y_val.values),
                epochs=2, batch_size=32, verbose=0
            )
            
            assert history is not None
            assert bdlstm_model.history is not None
        
        # 2. Test CatBoost model
        catboost_model = CatBoostModel(config)
        
        with patch.object(catboost_model.model, 'fit'):
            catboost_model.fit(X_train, y_train, validation_data=(X_val, y_val))
            assert catboost_model.is_fitted
            assert catboost_model.feature_names == list(X_train.columns)
        
        # 3. Test Ensemble model
        ensemble_model = EnsembleModel(config)
        
        with patch('src.models.ensemble_model.BidirectionalLSTMModel') as mock_bdlstm:
            with patch('src.models.ensemble_model.CatBoostModel') as mock_catboost:
                # Mock the individual models
                mock_bdlstm_instance = MagicMock()
                mock_catboost_instance = MagicMock()
                mock_bdlstm.return_value = mock_bdlstm_instance
                mock_catboost.return_value = mock_catboost_instance
                
                # Mock training
                mock_history = MagicMock()
                mock_bdlstm_instance.fit.return_value = mock_history
                mock_catboost_instance.fit.return_value = mock_catboost_instance
                
                # Train ensemble
                result = ensemble_model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=2, batch_size=32
                )
                
                assert ensemble_model.is_fitted
                assert result is not None
    
    def test_model_persistence(self, sample_data_large):
        """Test model saving and loading"""
        X, y = sample_data_large
        config = Config()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test BDLSTM save/load
            bdlstm_model = BidirectionalLSTMModel((X.shape[1],), config)
            
            # Mock save/load operations
            with patch.object(bdlstm_model.model, 'save'):
                bdlstm_model.save(os.path.join(temp_dir, 'bdlstm_test.h5'))
            
            with patch('tensorflow.keras.models.load_model') as mock_load:
                mock_model = MagicMock()
                mock_load.return_value = mock_model
                bdlstm_model.load(os.path.join(temp_dir, 'bdlstm_test.h5'))
                assert bdlstm_model.model == mock_model
            
            # Test CatBoost save/load
            catboost_model = CatBoostModel(config)
            catboost_model.is_fitted = True
            catboost_model.feature_names = list(X.columns)
            
            with patch.object(catboost_model.model, 'save_model'):
                with patch('joblib.dump'):
                    catboost_model.save(os.path.join(temp_dir, 'catboost_test.cbm'))
    
    def test_model_configuration_validation(self):
        """Test model configuration validation"""
        config = Config()
        
        # Test valid configuration
        bdlstm_model = BidirectionalLSTMModel((10,), config)
        assert bdlstm_model.config == config
        
        catboost_model = CatBoostModel(config)
        assert catboost_model.config == config
        
        ensemble_model = EnsembleModel(config)
        assert ensemble_model.config == config
        
        # Test ensemble weights validation
        assert len(ensemble_model.ensemble_weights) == 2
        assert abs(sum(ensemble_model.ensemble_weights) - 1.0) < 1e-6
    
    def test_error_handling(self, sample_data_large):
        """Test error handling across models"""
        X, y = sample_data_large
        config = Config()
        
        # Test BDLSTM errors
        bdlstm_model = BidirectionalLSTMModel((X.shape[1],), config)
        
        with pytest.raises(ValueError):
            bdlstm_model.predict(X.values)  # Model not trained
        
        with pytest.raises(ValueError):
            bdlstm_model.save('test.h5')  # Model not trained
        
        # Test CatBoost errors
        catboost_model = CatBoostModel(config)
        
        with pytest.raises(ValueError):
            catboost_model.predict(X)  # Model not fitted
        
        with pytest.raises(ValueError):
            catboost_model.get_feature_importance()  # Model not fitted
        
        # Test Ensemble errors
        ensemble_model = EnsembleModel(config)
        
        with pytest.raises(ValueError):
            ensemble_model.predict(X)  # Model not fitted
        
        with pytest.raises(ValueError):
            ensemble_model.save('test_dir')  # Model not fitted


if __name__ == "__main__":
    pytest.main([__file__])
