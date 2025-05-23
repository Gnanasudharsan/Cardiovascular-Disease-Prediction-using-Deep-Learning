"""
Ensemble Model: BDLSTM + CatBoost
Implementation of the proposed ensemble model from the research paper
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
import joblib
import logging
import os

logger = logging.getLogger(__name__)

class BidirectionalLSTMModel:
    """
    Bidirectional LSTM model for sequence modeling
    """
    
    def __init__(self, input_shape, config):
        self.input_shape = input_shape
        self.config = config
        self.model = None
        self.build_model()
    
    def build_model(self):
        """Build the Bidirectional LSTM architecture"""
        
        # Input layer
        inputs = Input(shape=self.input_shape)
        
        # Reshape for LSTM (samples, timesteps, features)
        # For tabular data, we treat each feature as a timestep
        reshaped = tf.expand_dims(inputs, axis=1)  # Add time dimension
        
        # Bidirectional LSTM layers
        lstm1 = Bidirectional(
            LSTM(units=128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)
        )(reshaped)
        
        lstm2 = Bidirectional(
            LSTM(units=64, return_sequences=False, dropout=0.2, recurrent_dropout=0.2)
        )(lstm1)
        
        # Dense layers
        dense1 = Dense(128, activation='relu')(lstm2)
        dropout1 = Dropout(0.3)(dense1)
        
        dense2 = Dense(64, activation='relu')(dropout1)
        dropout2 = Dropout(0.2)(dropout2)
        
        # Output layer
        outputs = Dense(1, activation='sigmoid')(dropout2)
        
        # Create model
        self.model = Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=self.config.LEARNING_RATE),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        logger.info("Bidirectional LSTM model built successfully")
        logger.info(f"Model summary:\n{self.model.summary()}")
    
    def fit(self, X_train, y_train, validation_data=None, epochs=100, batch_size=32):
        """Train the BDLSTM model"""
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7
            )
        ]
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict probabilities"""
        proba = self.model.predict(X)
        # Return probabilities for both classes
        return np.column_stack([1 - proba, proba])

class CatBoostModel:
    """
    CatBoost model wrapper
    """
    
    def __init__(self, config):
        self.config = config
        self.model = CatBoostClassifier(
            iterations=1000,
            learning_rate=0.1,
            depth=6,
            eval_metric='AUC',
            random_seed=config.RANDOM_STATE,
            logging_level='Silent'
        )
    
    def fit(self, X_train, y_train, validation_data=None):
        """Train the CatBoost model"""
        
        eval_set = None
        if validation_data is not None:
            eval_set = validation_data
        
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            early_stopping_rounds=50,
            verbose=False
        )
        
        return self.model
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict probabilities"""
        return self.model.predict_proba(X)

class EnsembleModel(BaseEstimator, ClassifierMixin):
    """
    Ensemble model combining BDLSTM and CatBoost
    Implementation of the proposed model from the research paper
    """
    
    def __init__(self, config):
        self.config = config
        self.bdlstm_model = None
        self.catboost_model = None
        self.ensemble_weights = [0.6, 0.4]  # BDLSTM weight, CatBoost weight
        self.is_fitted = False
    
    def fit(self, X_train, y_train, validation_data=None, epochs=100, batch_size=32):
        """
        Train the ensemble model
        
        Args:
            X_train: Training features
            y_train: Training labels
            validation_data: Validation data tuple (X_val, y_val)
            epochs: Number of epochs for BDLSTM
            batch_size: Batch size for BDLSTM
        """
        logger.info("Training ensemble model (BDLSTM + CatBoost)...")
        
        # Initialize models
        input_shape = (X_train.shape[1],)
        self.bdlstm_model = BidirectionalLSTMModel(input_shape, self.config)
        self.catboost_model = CatBoostModel(self.config)
        
        # Train BDLSTM model
        logger.info("Training Bidirectional LSTM...")
        bdlstm_history = self.bdlstm_model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size
        )
        
        # Train CatBoost model
        logger.info("Training CatBoost...")
        self.catboost_model.fit(X_train, y_train, validation_data)
        
        self.is_fitted = True
        logger.info("Ensemble model training completed")
        
        return bdlstm_history
    
    def predict(self, X):
        """
        Make ensemble predictions
        
        Args:
            X: Input features
            
        Returns:
            Binary predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Get predictions from both models
        bdlstm_proba = self.bdlstm_model.predict(X).flatten()
        catboost_proba = self.catboost_model.predict_proba(X)[:, 1]
        
        # Ensemble prediction (weighted average)
        ensemble_proba = (
            self.ensemble_weights[0] * bdlstm_proba +
            self.ensemble_weights[1] * catboost_proba
        )
        
        # Convert to binary predictions
        predictions = (ensemble_proba > 0.5).astype(int)
        return predictions
    
    def predict_proba(self, X):
        """
        Predict class probabilities
        
        Args:
            X: Input features
            
        Returns:
            Class probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Get probabilities from both models
        bdlstm_proba = self.bdlstm_model.predict(X).flatten()
        catboost_proba = self.catboost_model.predict_proba(X)[:, 1]
        
        # Ensemble probability (weighted average)
        ensemble_proba = (
            self.ensemble_weights[0] * bdlstm_proba +
            self.ensemble_weights[1] * catboost_proba
        )
        
        # Return probabilities for both classes
        proba_class_0 = 1 - ensemble_proba
        proba_class_1 = ensemble_proba
        
        return np.column_stack([proba_class_0, proba_class_1])
    
    def get_feature_importance(self):
        """
        Get feature importance from CatBoost model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        return self.catboost_model.model.get_feature_importance()
    
    def save(self, save_dir):
        """
        Save the ensemble model
        
        Args:
            save_dir: Directory to save models
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Save BDLSTM model
        bdlstm_path = os.path.join(save_dir, 'bdlstm_model.h5')
        self.bdlstm_model.model.save(bdlstm_path)
        
        # Save CatBoost model
        catboost_path = os.path.join(save_dir, 'catboost_model.pkl')
        joblib.dump(self.catboost_model.model, catboost_path)
        
        # Save ensemble configuration
        config_path = os.path.join(save_dir, 'ensemble_config.pkl')
        ensemble_config = {
            'ensemble_weights': self.ensemble_weights,
            'input_shape': self.bdlstm_model.input_shape,
            'is_fitted': self.is_fitted
        }
        joblib.dump(ensemble_config, config_path)
        
        logger.info(f"Ensemble model saved to {save_dir}")
    
    def load(self, save_dir):
        """
        Load the ensemble model
        
        Args:
            save_dir: Directory containing saved models
        """
        # Load ensemble configuration
        config_path = os.path.join(save_dir, 'ensemble_config.pkl')
        ensemble_config = joblib.load(config_path)
        
        self.ensemble_weights = ensemble_config['ensemble_weights']
        self.is_fitted = ensemble_config['is_fitted']
        
        # Load BDLSTM model
        bdlstm_path = os.path.join(save_dir, 'bdlstm_model.h5')
        self.bdlstm_model = BidirectionalLSTMModel(
            ensemble_config['input_shape'], 
            self.config
        )
        self.bdlstm_model.model = tf.keras.models.load_model(bdlstm_path)
        
        # Load CatBoost model
        catboost_path = os.path.join(save_dir, 'catboost_model.pkl')
        self.catboost_model = CatBoostModel(self.config)
        self.catboost_model.model = joblib.load(catboost_path)
        
        logger.info(f"Ensemble model loaded from {save_dir}")
    
    def evaluate_individual_models(self, X_test, y_test):
        """
        Evaluate individual models in the ensemble
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary with individual model performances
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        # BDLSTM predictions
        bdlstm_pred = (self.bdlstm_model.predict(X_test).flatten() > 0.5).astype(int)
        
        # CatBoost predictions
        catboost_pred = self.catboost_model.predict(X_test)
        
        # Ensemble predictions
        ensemble_pred = self.predict(X_test)
        
        # Calculate metrics
        results = {
            'BDLSTM': {
                'accuracy': accuracy_score(y_test, bdlstm_pred),
                'precision': precision_score(y_test, bdlstm_pred),
                'recall': recall_score(y_test, bdlstm_pred),
                'f1_score': f1_score(y_test, bdlstm_pred)
            },
            'CatBoost': {
                'accuracy': accuracy_score(y_test, catboost_pred),
                'precision': precision_score(y_test, catboost_pred),
                'recall': recall_score(y_test, catboost_pred),
                'f1_score': f1_score(y_test, catboost_pred)
            },
            'Ensemble': {
                'accuracy': accuracy_score(y_test, ensemble_pred),
                'precision': precision_score(y_test, ensemble_pred),
                'recall': recall_score(y_test, ensemble_pred),
                'f1_score': f1_score(y_test, ensemble_pred)
            }
        }
        
        return results

if __name__ == "__main__":
    # Test the ensemble model
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    import sys
    import os
    
    # Add parent directory to path
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from utils.config import Config
    
    # Generate sample data
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        random_state=42
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # Initialize config and model
    config = Config()
    model = EnsembleModel(config)
    
    # Train model
    print("Training ensemble model...")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32
    )
    
    # Make predictions
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    
    print(f"Predictions shape: {predictions.shape}")
    print(f"Probabilities shape: {probabilities.shape}")
    print(f"Sample predictions: {predictions[:10]}")
    print(f"Sample probabilities: {probabilities[:5]}")
    
    # Evaluate individual models
    results = model.evaluate_individual_models(X_test, y_test)
    print("\nModel Performance:")
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
