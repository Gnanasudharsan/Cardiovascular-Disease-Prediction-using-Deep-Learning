"""
Cardiovascular Disease Prediction Package

This package implements deep learning models for cardiovascular disease prediction
using Bidirectional LSTM and CatBoost ensemble approach.


"""

__version__ = "1.0.0"
__author__ = "V Sasikala et al."
__email__ = "sasikala.ece@sairam.edu.in"

from .data_preprocessing import DataPreprocessor
from .feature_selection import FeatureSelector
from .training import ModelTrainer

# Model imports
from .models.ensemble_model import EnsembleModel
from .models.bdlstm_model import BidirectionalLSTMModel
from .models.catboost_model import CatBoostModel

# Utility imports
from .utils.config import Config
from .utils.evaluation_metrics import ModelEvaluator
from .utils.visualization import ResultVisualizer

__all__ = [
    'DataPreprocessor',
    'FeatureSelector', 
    'ModelTrainer',
    'EnsembleModel',
    'BidirectionalLSTMModel',
    'CatBoostModel',
    'Config',
    'ModelEvaluator',
    'ResultVisualizer'
]
