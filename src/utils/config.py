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
                'iterations': self.CAT
