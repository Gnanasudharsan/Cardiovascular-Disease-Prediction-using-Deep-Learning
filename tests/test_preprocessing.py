"""
Tests for data preprocessing module
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_preprocessing import DataPreprocessor
from utils.config import Config

class TestDataPreprocessor:
    """Test cases for DataPreprocessor class"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample cardiovascular disease data for testing"""
        np.random.seed(42)
        data = {
            'age': np.random.randint(10000, 25000, 100),
            'gender': np.random.choice([1, 2], 100),
            'height': np.random.randint(150, 200, 100),
            'weight': np.random.randint(50, 120, 100),
            'ap_hi': np.random.randint(80, 200, 100),
            'ap_lo': np.random.randint(50, 120, 100),
            'cholesterol': np.random.choice([1, 2, 3], 100),
            'gluc': np.random.choice([1, 2, 3], 100),
            'smoke': np.random.choice([0, 1], 100),
            'alco': np.random.choice([0, 1], 100),
            'active': np.random.choice([0, 1], 100),
            'target': np.random.choice([0, 1], 100)
        }
        return pd.DataFrame(data)
    
    @pytest.fixture
    def sample_data_with_missing(self):
        """Create sample data with missing values"""
        np.random.seed(42)
        data = {
            'age': np.random.randint(10000, 25000, 100),
            'gender': np.random.choice([1, 2], 100),
            'height': np.random.randint(150, 200, 100),
            'weight': np.random.randint(50, 120, 100),
            'ap_hi': np.random.randint(80, 200, 100),
            'ap_lo': np.random.randint(50, 120, 100),
            'cholesterol': np.random.choice([0, 1, 2, 3], 100),  # 0 represents missing
            'gluc': np.random.choice([1, 2, 3], 100),
            'smoke': np.random.choice([0, 1], 100),
            'alco': np.random.choice([0, 1], 100),
            'active': np.random.choice([0, 1], 100),
            'target': np.random.choice([0, 1], 100)
        }
        df = pd.DataFrame(data)
        
        # Introduce some NaN values
        df.loc[np.random.choice(df.index, 5), 'height'] = np.nan
        df.loc[np.random.choice(df.index, 3), 'weight'] = np.nan
        
        return df
    
    @pytest.fixture
    def config(self):
        """Create test configuration"""
        return Config()
    
    def test_preprocessor_initialization(self, config):
        """Test preprocessor initialization"""
        preprocessor = DataPreprocessor(config)
        assert preprocessor.config == config
        assert not preprocessor.is_fitted
        assert len(preprocessor.label_encoders) == 0
        assert len(preprocessor.imputers) == 0
    
    def test_fit_transform_valid_data(self, sample_data, config):
        """Test fit_transform with valid data"""
        preprocessor = DataPreprocessor(config)
        X, y = preprocessor.fit_transform(sample_data)
        
        # Check output types
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        
        # Check dimensions
        assert len(X) == len(sample_data)
        assert len(y) == len(sample_data)
        assert X.shape[1] > sample_data.shape[1] - 1  # Should have interaction terms
        
        # Check that preprocessor is fitted
        assert preprocessor.is_fitted
    
    def test_handle_missing_cholesterol(self, sample_data_with_missing, config):
        """Test handling of missing cholesterol values (recorded as 0)"""
        preprocessor = DataPreprocessor(config)
        df_processed = preprocessor.handle_missing_values(sample_data_with_missing)
        
        # Check that 0 cholesterol values are replaced
        assert (df_processed['cholesterol'] == 0).sum() == 0
        
        # Check that all cholesterol values are now valid
        assert df_processed['cholesterol'].min() >= 1
        assert df_processed['cholesterol'].max() <= 3
    
    def test_handle_missing_values_general(self, sample_data_with_missing, config):
        """Test general missing value handling"""
        preprocessor = DataPreprocessor(config)
        df_processed = preprocessor.handle_missing_values(sample_data_with_missing)
        
        # Check that no missing values remain
        assert df_processed.isnull().sum().sum() == 0
    
    def test_create_interaction_terms(self, sample_data, config):
        """Test creation of interaction terms"""
        preprocessor = DataPreprocessor(config)
        df_with_interactions = preprocessor.create_interaction_terms(sample_data)
        
        # Check that interaction terms are added
        assert df_with_interactions.shape[1] > sample_data.shape[1]
        
        # Check for specific interaction terms
        interaction_columns = [col for col in df_with_interactions.columns if '_interaction' in col]
        assert len(interaction_columns) > 0
    
    def test_normalize_features(self, sample_data, config):
        """Test feature normalization"""
        preprocessor = DataPreprocessor(config)
        
        # Separate features and target
        X = sample_data.drop('target', axis=1)
        
        # Normalize features
        X_normalized = preprocessor.normalize_features(X)
        
        # Check that features are normalized (approximately mean=0, std=1)
        for col in X_normalized.select_dtypes(include=[np.number]).columns:
            assert abs(X_normalized[col].mean()) < 0.1  # Close to 0
            assert abs(X_normalized[col].std() - 1.0) < 0.1  # Close to 1
    
    def test_train_test_split(self, sample_data, config):
        """Test train-test splitting"""
        preprocessor = DataPreprocessor(config)
        
        X = sample_data.drop('target', axis=1)
        y = sample_data['target']
        
        X_train, X_test, y_train, y_test = preprocessor.train_test_split(X, y, test_size=0.3)
        
        # Check split proportions
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)
        assert len(X_train) + len(X_test) == len(X)
        
        # Check that test size is approximately correct
        test_ratio = len(X_test) / len(X)
        assert abs(test_ratio - 0.3) < 0.05
    
    def test_transform_new_data(self, sample_data, config):
        """Test transforming new data with fitted preprocessor"""
        preprocessor = DataPreprocessor(config)
        
        # Fit on original data
        X, y = preprocessor.fit_transform(sample_data)
        
        # Create new data (same structure)
        new_data = sample_data.iloc[:10].copy()
        
        # Transform new data
        X_new = preprocessor.transform(new_data.drop('target', axis=1))
        
        # Check that transformation works
        assert isinstance(X_new, pd.DataFrame)
        assert X_new.shape[1] == X.shape[1]  # Same number of features
        assert len(X_new) == 10
    
    def test_preprocessing_stats(self, sample_data, config):
        """Test preprocessing statistics generation"""
        preprocessor = DataPreprocessor(config)
        X, y = preprocessor.fit_transform(sample_data)
        
        stats = preprocessor.get_preprocessing_stats(X, y)
        
        # Check stats structure
        assert 'n_samples' in stats
        assert 'n_features' in stats
        assert 'class_distribution' in stats
        assert 'feature_statistics' in stats
        assert 'missing_values' in stats
        
        # Check values
        assert stats['n_samples'] == len(sample_data)
        assert stats['n_features'] == X.shape[1]
    
    def test_error_handling_not_fitted(self, sample_data, config):
        """Test error handling when preprocessor is not fitted"""
        preprocessor = DataPreprocessor(config)
        
        # Should raise error when trying to transform without fitting
        with pytest.raises(ValueError, match="Preprocessor must be fitted before transform"):
            preprocessor.transform(sample_data.drop('target', axis=1))
    
    def test_feature_encoding(self, config):
        """Test categorical feature encoding"""
        preprocessor = DataPreprocessor(config)
        
        # Create data with categorical features
        data = pd.DataFrame({
            'cat_feature': ['A', 'B', 'C', 'A', 'B'],
            'num_feature': [1, 2, 3, 4, 5],
            'target': [0, 1, 0, 1, 0]
        })
        
        encoded_data = preprocessor.encode_categorical_features(data)
        
        # Check that categorical feature is encoded
        assert encoded_data['cat_feature'].dtype in [np.int32, np.int64]
        assert encoded_data['num_feature'].dtype == data['num_feature'].dtype
    
    def test_prepare_target_variable(self, sample_data, config):
        """Test target variable preparation"""
        preprocessor = DataPreprocessor(config)
        
        y = sample_data['target']
        y_prepared = preprocessor.prepare_target_variable(y)
        
        # Check that target is properly prepared
        assert isinstance(y_prepared, pd.Series)
        assert y_prepared.dtype in [np.int32, np.int64]
        assert set(y_prepared.unique()).issubset({0, 1})

class TestDataPreprocessorIntegration:
    """Integration tests for the complete preprocessing pipeline"""
    
    def test_complete_pipeline(self):
        """Test the complete preprocessing pipeline end-to-end"""
        # Create realistic sample data
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'age': np.random.randint(10000, 25000, n_samples),
            'gender': np.random.choice([1, 2], n_samples),
            'height': np.random.randint(150, 200, n_samples),
            'weight': np.random.randint(50, 120, n_samples),
            'ap_hi': np.random.randint(80, 200, n_samples),
            'ap_lo': np.random.randint(50, 120, n_samples),
            'cholesterol': np.random.choice([0, 1, 2, 3], n_samples),  # Include 0s
            'gluc': np.random.choice([1, 2, 3], n_samples),
            'smoke': np.random.choice([0, 1], n_samples),
            'alco': np.random.choice([0, 1], n_samples),
            'active': np.random.choice([0, 1], n_samples),
            'target': np.random.choice([0, 1], n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Add some missing values
        missing_indices = np.random.choice(df.index, 50, replace=False)
        df.loc[missing_indices[:25], 'height'] = np.nan
        df.loc[missing_indices[25:], 'weight'] = np.nan
        
        # Initialize preprocessor
        config = Config()
        preprocessor = DataPreprocessor(config)
        
        # Run complete pipeline
        X, y = preprocessor.fit_transform(df)
        
        # Verify results
        assert preprocessor.is_fitted
        assert X.shape[0] == n_samples
        assert y.shape[0] == n_samples
        assert X.isnull().sum().sum() == 0  # No missing values
        assert y.isnull().sum() == 0  # No missing values in target
        
        # Test train-test split
        X_train, X_test, y_train, y_test = preprocessor.train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        assert len(X_train) == int(n_samples * 0.8)
        assert len(X_test) == n_samples - len(X_train)
        
        # Test statistics generation
        stats = preprocessor.get_preprocessing_stats(X, y)
        assert stats['n_samples'] == n_samples
        assert stats['missing_values'][list(X.columns)[0]] == 0

if __name__ == "__main__":
    pytest.main([__file__])
