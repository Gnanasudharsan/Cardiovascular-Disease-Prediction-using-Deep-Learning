"""
Configuration file for Cardiovascular Disease Prediction
Contains all hyperparameters and settings used in the research
"""

import os
import yaml
from pathlib import Path

class Config:
    """Configuration class for CVD prediction project"""
    
    def __init__(self, config_path=None):
        """
        Initialize configuration
        
        Args:
            config_path: Path to YAML configuration file (optional)
        """
        # Default configuration
        self.set_default_config()
        
        # Load from YAML file if provided
        if config_path and os.path.exists(config_path):
            self.load_from_yaml(config_path)
    
    def set_default_config(self):
        """Set default configuration parameters"""
        
        # Data paths
        self.DATA_PATH = 'data/raw/cardiovascular_disease_dataset.csv'
        self.PROCESSED_DATA_PATH = 'data/processed/'
        self.MODEL_SAVE_PATH = 'models/saved_models/'
        self.RESULTS_PATH = 'results/'
        
        # Data preprocessing
        self.TEST_SIZE = 0.35  # As mentioned in paper (65% train, 35% test)
        self.VALIDATION_SIZE = 0.2  # 20% of training data for validation
        self.RANDOM_STATE = 42
        
        # Feature selection
        self.SHAP_THRESHOLD = 0.1  # Features with Shapley values > 0.1
        self.USE_INTERACTION_TERMS = True
        
        # Model hyperparameters - BDLSTM
        self.LSTM_UNITS_1 = 128
        self.LSTM_UNITS_2 = 64
        self.DENSE_UNITS_1 = 128
        self.DENSE_UNITS_2 = 64
        self.DROPOUT_RATE_1 = 0.3
        self.DROPOUT_RATE_2 = 0.2
        self.LSTM_DROPOUT = 0.2
        self.RECURRENT_DROPOUT = 0.2
        
        # Training parameters
        self.EPOCHS = 100
        self.BATCH_SIZE = 32
        self.LEARNING_RATE = 0.001
        self.EARLY_STOPPING_PATIENCE = 15
        self.REDUCE_LR_PATIENCE = 10
        self.REDUCE_LR_FACTOR = 0.5
        
        # CatBoost parameters
        self.CATBOOST_ITERATIONS = 1000
        self.CATBOOST_LEARNING_RATE = 0.1
        self.CATBOOST_DEPTH = 6
        self.CATBOOST_EARLY_STOPPING = 50
        
        # Ensemble parameters
        self.ENSEMBLE_WEIGHTS = [0.6, 0.4]  # [BDLSTM_weight, CatBoost_weight]
        
        # Cross-validation
        self.CV_FOLDS = 5
        self.CV_SCORING = 'roc_auc'
        
        # Evaluation metrics
        self.CLASSIFICATION_THRESHOLD = 0.5
        self.METRICS = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        # Visualization
        self.FIGURE_SIZE = (10, 6)
        self.DPI = 300
        self.SAVE_PLOTS = True
        
        # Logging
        self.LOG_LEVEL = 'INFO'
        self.LOG_FILE = 'cvd_prediction.log'
        
        # Performance targets (from paper)
        self.TARGET_ACCURACY = 0.94
        self.TARGET_PRECISION = 0.93
        self.TARGET_RECALL = 0.92
        self.TARGET_F1_SCORE = 0.94
        self.TARGET_ROC_AUC = 0.94
        
        # Model comparison baselines
        self.BASELINE_MODELS = {
            'KNN': {
                'n_neighbors': 5,
                'weights': 'uniform'
            },
            'LogisticRegression': {
                'max_iter': 1000,
                'random_state': self.RANDOM_STATE
            },
            'RandomForest': {
                'n_estimators': 100,
                'random_state': self.RANDOM_STATE,
                'max_depth': 10
            },
            'XGBoost': {
                'n_estimators': 100,
                'random_state': self.RANDOM_STATE,
                'max_depth': 6
            },
            'AdaBoost': {
                'n_estimators': 100,
                'random_state': self.RANDOM_STATE
            }
        }
        
        # Hardware configuration
        self.USE_GPU = True
        self.GPU_MEMORY_GROWTH = True
        self.MIXED_PRECISION = False
        
        # Reproducibility
        self.SET_RANDOM_SEEDS = True
        
        # Data validation
        self.VALIDATE_DATA = True
        self.CHECK_DATA_QUALITY = True
        
        # Feature engineering
        self.FEATURE_SCALING = 'standard'  # 'standard', 'minmax', 'robust'
        self.HANDLE_OUTLIERS = True
        self.OUTLIER_METHOD = 'iqr'  # 'iqr', 'zscore', 'isolation_forest'
        
        # Model interpretability
        self.GENERATE_SHAP_PLOTS = True
        self.SHAP_SAMPLE_SIZE = 100  # Number of samples for SHAP analysis
        
        # Experiment tracking
        self.TRACK_EXPERIMENTS = True
        self.EXPERIMENT_NAME = 'cvd_prediction'
        
        # Model deployment
        self.MODEL_VERSION = '1.0.0'
        self.MODEL_DESCRIPTION = 'BDLSTM + CatBoost ensemble for CVD prediction'
    
    def load_from_yaml(self, config_path):
        """
        Load configuration from YAML file
        
        Args:
            config_path: Path to YAML configuration file
        """
        try:
            with open(config_path, 'r') as file:
                config_dict = yaml.safe_load(file)
                
            # Update configuration with values from YAML
            for key, value in config_dict.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                    
        except Exception as e:
            print(f"Error loading configuration from {config_path}: {str(e)}")
            print("Using default configuration...")
    
    def save_to_yaml(self, config_path):
        """
        Save current configuration to YAML file
        
        Args:
            config_path: Path to save YAML configuration file
        """
        config_dict = {
            key: value for key, value in self.__dict__.items()
            if not key.startswith('_')
        }
        
        try:
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, 'w') as file:
                yaml.dump(config_dict, file, default_flow_style=False, indent=2)
                
        except Exception as e:
            print(f"Error saving configuration to {config_path}: {str(e)}")
    
    def get_model_params(self):
        """Get model-specific parameters"""
        return {
            'bdlstm_params': {
                'lstm_units_1': self.LSTM_UNITS_1,
                'lstm_units_2': self.LSTM_UNITS_2,
                'dense_units_1': self.DENSE_UNITS_1,
                'dense_units_2': self.DENSE_UNITS_2,
                'dropout_rate_1': self.DROPOUT_RATE_1,
                'dropout_rate_2': self.DROPOUT_RATE_2,
                'lstm_dropout': self.LSTM_DROPOUT,
                'recurrent_dropout': self.RECURRENT_DROPOUT,
                'learning_rate': self.LEARNING_RATE
            },
            'catboost_params': {
                'iterations': self.CATBOOST_ITERATIONS,
                'learning_rate': self.CATBOOST_LEARNING_RATE,
                'depth': self.CATBOOST_DEPTH,
                'early_stopping_rounds': self.CATBOOST_EARLY_STOPPING,
                'random_seed': self.RANDOM_STATE
            },
            'ensemble_params': {
                'weights': self.ENSEMBLE_WEIGHTS,
                'threshold': self.CLASSIFICATION_THRESHOLD
            }
        }
    
    def get_training_params(self):
        """Get training-specific parameters"""
        return {
            'epochs': self.EPOCHS,
            'batch_size': self.BATCH_SIZE,
            'validation_size': self.VALIDATION_SIZE,
            'early_stopping_patience': self.EARLY_STOPPING_PATIENCE,
            'reduce_lr_patience': self.REDUCE_LR_PATIENCE,
            'reduce_lr_factor': self.REDUCE_LR_FACTOR
        }
    
    def get_data_params(self):
        """Get data processing parameters"""
        return {
            'test_size': self.TEST_SIZE,
            'random_state': self.RANDOM_STATE,
            'shap_threshold': self.SHAP_THRESHOLD,
            'use_interaction_terms': self.USE_INTERACTION_TERMS,
            'feature_scaling': self.FEATURE_SCALING,
            'handle_outliers': self.HANDLE_OUTLIERS,
            'outlier_method': self.OUTLIER_METHOD
        }
    
    def setup_tensorflow(self):
        """Setup TensorFlow configuration"""
        import tensorflow as tf
        
        if self.USE_GPU:
            # Configure GPU memory growth
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, self.GPU_MEMORY_GROWTH)
                except RuntimeError as e:
                    print(f"GPU configuration error: {e}")
        
        # Mixed precision training
        if self.MIXED_PRECISION:
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
        
        # Set random seeds for reproducibility
        if self.SET_RANDOM_SEEDS:
            tf.random.set_seed(self.RANDOM_STATE)
            import numpy as np
            np.random.seed(self.RANDOM_STATE)
            import random
            random.seed(self.RANDOM_STATE)
    
    def validate_config(self):
        """Validate configuration parameters"""
        errors = []
        
        # Check data paths
        if not os.path.exists(os.path.dirname(self.DATA_PATH)):
            errors.append(f"Data directory does not exist: {os.path.dirname(self.DATA_PATH)}")
        
        # Check parameter ranges
        if not 0 < self.TEST_SIZE < 1:
            errors.append("TEST_SIZE must be between 0 and 1")
        
        if not 0 < self.VALIDATION_SIZE < 1:
            errors.append("VALIDATION_SIZE must be between 0 and 1")
        
        if self.LEARNING_RATE <= 0:
            errors.append("LEARNING_RATE must be positive")
        
        if self.EPOCHS <= 0:
            errors.append("EPOCHS must be positive")
        
        if self.BATCH_SIZE <= 0:
            errors.append("BATCH_SIZE must be positive")
        
        # Check ensemble weights
        if len(self.ENSEMBLE_WEIGHTS) != 2:
            errors.append("ENSEMBLE_WEIGHTS must have exactly 2 values")
        
        if abs(sum(self.ENSEMBLE_WEIGHTS) - 1.0) > 1e-6:
            errors.append("ENSEMBLE_WEIGHTS must sum to 1.0")
        
        if errors:
            raise ValueError("Configuration validation failed:\n" + "\n".join(errors))
        
        return True
    
    def __str__(self):
        """String representation of configuration"""
        config_str = "=== CVD Prediction Configuration ===\n"
        config_str += f"Data Path: {self.DATA_PATH}\n"
        config_str += f"Test Size: {self.TEST_SIZE}\n"
        config_str += f"Random State: {self.RANDOM_STATE}\n"
        config_str += f"Epochs: {self.EPOCHS}\n"
        config_str += f"Batch Size: {self.BATCH_SIZE}\n"
        config_str += f"Learning Rate: {self.LEARNING_RATE}\n"
        config_str += f"Ensemble Weights: {self.ENSEMBLE_WEIGHTS}\n"
        config_str += f"Target Accuracy: {self.TARGET_ACCURACY}\n"
        return config_str

# Default configuration instance
DEFAULT_CONFIG = Config()

if __name__ == "__main__":
    # Test configuration
    config = Config()
    
    print("Default Configuration:")
    print(config)
    
    # Validate configuration
    try:
        config.validate_config()
        print("\nConfiguration validation: PASSED")
    except ValueError as e:
        print(f"\nConfiguration validation: FAILED")
        print(f"Errors: {e}")
    
    # Test saving and loading
    config_path = "test_config.yaml"
    config.save_to_yaml(config_path)
    print(f"\nConfiguration saved to {config_path}")
    
    # Load configuration
    new_config = Config(config_path)
    print("\nConfiguration loaded successfully")
    
    # Clean up
    if os.path.exists(config_path):
        os.remove(config_path)
