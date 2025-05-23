"""
CatBoost Model Implementation
Standalone implementation of the CatBoost component
"""

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
import joblib
import logging
import os

logger = logging.getLogger(__name__)

class CatBoostModel:
    """
    CatBoost model wrapper for cardiovascular disease prediction
    
    This class provides a wrapper around CatBoost with optimized parameters
    for the cardiovascular disease prediction task.
    """
    
    def __init__(self, config):
        """
        Initialize the CatBoost model
        
        Args:
            config: Configuration object with model parameters
        """
        self.config = config
        self.model = None
        self.feature_names = None
        self.feature_importance = None
        self.is_fitted = False
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the CatBoost classifier with optimized parameters"""
        
        logger.info("Initializing CatBoost model...")
        
        self.model = CatBoostClassifier(
            # Core parameters
            iterations=self.config.CATBOOST_ITERATIONS,
            learning_rate=self.config.CATBOOST_LEARNING_RATE,
            depth=self.config.CATBOOST_DEPTH,
            
            # Regularization
            l2_leaf_reg=3.0,
            bagging_temperature=1.0,
            random_strength=1.0,
            
            # Performance optimization
            thread_count=-1,
            used_ram_limit='4gb',
            
            # Evaluation
            eval_metric='AUC',
            custom_loss=['Accuracy', 'Precision', 'Recall', 'F1'],
            
            # Early stopping
            early_stopping_rounds=self.config.CATBOOST_EARLY_STOPPING,
            
            # Reproducibility
            random_seed=self.config.RANDOM_STATE,
            
            # Output control
            logging_level='Silent',
            verbose=False,
            
            # Model type
            objective='Logloss',
            bootstrap_type='Bayesian',
            
            # Feature selection
            max_ctr_complexity=4,
            model_size_reg=0.5,
            
            # Overfitting detection
            od_type='IncToDec',
            od_wait=20
        )
        
        logger.info("CatBoost model initialized with optimized parameters")
    
    def fit(self, X_train, y_train, validation_data=None, categorical_features=None, verbose=False):
        """
        Train the CatBoost model
        
        Args:
            X_train: Training features
            y_train: Training labels
            validation_data: Validation data tuple (X_val, y_val)
            categorical_features: List of categorical feature indices or names
            verbose: Whether to display training progress
            
        Returns:
            Self for method chaining
        """
        logger.info("Starting CatBoost model training...")
        
        # Store feature names
        if hasattr(X_train, 'columns'):
            self.feature_names = list(X_train.columns)
        else:
            self.feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
        
        # Create training pool
        train_pool = Pool(
            data=X_train,
            label=y_train,
            feature_names=self.feature_names,
            cat_features=categorical_features
        )
        
        # Create validation pool if provided
        eval_set = None
        if validation_data is not None:
            X_val, y_val = validation_data
            eval_set = Pool(
                data=X_val,
                label=y_val,
                feature_names=self.feature_names,
                cat_features=categorical_features
            )
        
        # Train the model
        self.model.fit(
            train_pool,
            eval_set=eval_set,
            verbose=verbose,
            plot=False
        )
        
        # Store feature importance
        self.feature_importance = self.model.get_feature_importance()
        self.is_fitted = True
        
        logger.info("CatBoost model training completed")
        logger.info(f"Final training accuracy: {self._get_training_accuracy(X_train, y_train):.4f}")
        
        return self
    
    def predict(self, X):
        """
        Make binary predictions
        
        Args:
            X: Input features
            
        Returns:
            Binary predictions (0 or 1)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        return self.model.predict(X).astype(int)
    
    def predict_proba(self, X):
        """
        Predict class probabilities
        
        Args:
            X: Input features
            
        Returns:
            Class probabilities [P(class=0), P(class=1)]
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        return self.model.predict_proba(X)
    
    def predict_log_proba(self, X):
        """
        Predict log of class probabilities
        
        Args:
            X: Input features
            
        Returns:
            Log probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        return np.log(self.predict_proba(X))
    
    def decision_function(self, X):
        """
        Predict raw decision scores
        
        Args:
            X: Input features
            
        Returns:
            Decision scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        probas = self.predict_proba(X)
        return np.log(probas[:, 1] / probas[:, 0])  # Log odds
    
    def get_feature_importance(self, importance_type='PredictionValuesChange'):
        """
        Get feature importance scores
        
        Args:
            importance_type: Type of importance to calculate
                           Options: 'PredictionValuesChange', 'LossFunctionChange', 'FeatureImportance'
        
        Returns:
            Feature importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        if importance_type == 'PredictionValuesChange':
            importance = self.model.get_feature_importance()
        elif importance_type == 'LossFunctionChange':
            importance = self.model.get_feature_importance(type='LossFunctionChange')
        else:
            importance = self.model.get_feature_importance(type='FeatureImportance')
        
        return importance
    
    def get_feature_importance_df(self, importance_type='PredictionValuesChange'):
        """
        Get feature importance as a DataFrame
        
        Args:
            importance_type: Type of importance to calculate
        
        Returns:
            DataFrame with feature names and importance scores
        """
        importance = self.get_feature_importance(importance_type)
        
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
    
    def plot_feature_importance(self, max_features=20, importance_type='PredictionValuesChange', save_path=None):
        """
        Plot feature importance
        
        Args:
            max_features: Maximum number of features to display
            importance_type: Type of importance to plot
            save_path: Path to save the plot
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before plotting feature importance")
        
        import matplotlib.pyplot as plt
        
        importance_df = self.get_feature_importance_df(importance_type)
        top_features = importance_df.head(max_features)
        
        plt.figure(figsize=(10, max(6, len(top_features) * 0.4)))
        plt.barh(range(len(top_features)), top_features['importance'], color='skyblue')
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel(f'Feature Importance ({importance_type})')
        plt.title(f'Top {max_features} Most Important Features - CatBoost')
        plt.gca().invert_yaxis()
        
        # Add value labels
        for i, importance in enumerate(top_features['importance']):
            plt.text(importance + max(top_features['importance']) * 0.01, i, 
                    f'{importance:.3f}', va='center', ha='left')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
        
        plt.show()
    
    def get_model_info(self):
        """
        Get detailed model information
        
        Returns:
            Dictionary with model information
        """
        if not self.is_fitted:
            return {"status": "Model not fitted"}
        
        return {
            "model_type": "CatBoost",
            "iterations": self.model.get_param('iterations'),
            "learning_rate": self.model.get_param('learning_rate'),
            "depth": self.model.get_param('depth'),
            "num_features": len(self.feature_names),
            "feature_names": self.feature_names,
            "is_fitted": self.is_fitted,
            "best_score": self.model.get_best_score(),
            "best_iteration": self.model.get_best_iteration(),
            "tree_count": self.model.tree_count_
        }
    
    def save(self, filepath):
        """
        Save the trained model
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        # Save the CatBoost model
        self.model.save_model(filepath)
        
        # Save additional metadata
        metadata = {
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance.tolist() if self.feature_importance is not None else None,
            'is_fitted': self.is_fitted,
            'model_info': self.get_model_info()
        }
        
        metadata_path = filepath.replace('.cbm', '_metadata.pkl').replace('.json', '_metadata.pkl')
        if not metadata_path.endswith('_metadata.pkl'):
            metadata_path += '_metadata.pkl'
        
        joblib.dump(metadata, metadata_path)
        
        logger.info(f"CatBoost model saved to {filepath}")
        logger.info(f"Model metadata saved to {metadata_path}")
    
    def load(self, filepath):
        """
        Load a trained model
        
        Args:
            filepath: Path to the saved model
        """
        # Load the CatBoost model
        self.model = CatBoostClassifier()
        self.model.load_model(filepath)
        
        # Load metadata
        metadata_path = filepath.replace('.cbm', '_metadata.pkl').replace('.json', '_metadata.pkl')
        if not metadata_path.endswith('_metadata.pkl'):
            metadata_path += '_metadata.pkl'
        
        try:
            metadata = joblib.load(metadata_path)
            self.feature_names = metadata.get('feature_names', None)
            self.feature_importance = np.array(metadata.get('feature_importance', [])) if metadata.get('feature_importance') else None
            self.is_fitted = metadata.get('is_fitted', True)
        except FileNotFoundError:
            logger.warning(f"Metadata file not found: {metadata_path}")
            self.is_fitted = True
            if hasattr(self.model, 'feature_names_'):
                self.feature_names = self.model.feature_names_
        
        logger.info(f"CatBoost model loaded from {filepath}")
    
    def _get_training_accuracy(self, X, y):
        """Get training accuracy for logging purposes"""
        try:
            predictions = self.predict(X)
            return np.mean(predictions == y)
        except:
            return 0.0
    
    def cross_validate(self, X, y, cv_folds=5, metrics=['AUC', 'Accuracy'], verbose=False):
        """
        Perform cross-validation
        
        Args:
            X: Features
            y: Labels
            cv_folds: Number of cross-validation folds
            metrics: List of metrics to evaluate
            verbose: Whether to display progress
            
        Returns:
            Cross-validation results
        """
        from catboost import cv
        
        logger.info(f"Performing {cv_folds}-fold cross-validation...")
        
        # Create pool
        cv_pool = Pool(
            data=X,
            label=y,
            feature_names=self.feature_names
        )
        
        # Get model parameters
        params = self.model.get_params()
        
        # Perform cross-validation
        cv_results = cv(
            pool=cv_pool,
            params=params,
            fold_count=cv_folds,
            shuffle=True,
            partition_random_seed=self.config.RANDOM_STATE,
            plot=False,
            verbose=verbose
        )
        
        # Extract results
        cv_summary = {}
        for metric in metrics:
            if f'test-{metric}-mean' in cv_results.columns:
                cv_summary[metric] = {
                    'mean': cv_results[f'test-{metric}-mean'].iloc[-1],
                    'std': cv_results[f'test-{metric}-std'].iloc[-1]
                }
        
        logger.info("Cross-validation completed")
        for metric, scores in cv_summary.items():
            logger.info(f"  {metric}: {scores['mean']:.4f} ± {scores['std']:.4f}")
        
        return cv_summary, cv_results
    
    def explain_prediction(self, X, prediction_idx=0):
        """
        Get SHAP-like explanation for a specific prediction
        
        Args:
            X: Input features
            prediction_idx: Index of the prediction to explain
            
        Returns:
            Feature contributions for the prediction
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before explaining predictions")
        
        # Get prediction explanations
        explanations = self.model.get_feature_importance(
            data=Pool(X),
            type='ShapValues'
        )
        
        if prediction_idx < len(explanations):
            explanation = explanations[prediction_idx]
            
            # Create explanation DataFrame
            explanation_df = pd.DataFrame({
                'feature': self.feature_names + ['bias'],
                'contribution': explanation
            })
            
            # Sort by absolute contribution
            explanation_df['abs_contribution'] = np.abs(explanation_df['contribution'])
            explanation_df = explanation_df.sort_values('abs_contribution', ascending=False)
            
            return explanation_df
        else:
            raise IndexError(f"Prediction index {prediction_idx} out of range")

# Example usage and testing
if __name__ == "__main__":
    # Test the CatBoost model
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
    
    # Convert to DataFrame for better feature names
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # Initialize config and model
    config = Config()
    model = CatBoostModel(config)
    
    # Train model
    print("Training CatBoost model...")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        verbose=True
    )
    
    # Make predictions
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    
    print(f"Predictions shape: {predictions.shape}")
    print(f"Probabilities shape: {probabilities.shape}")
    print(f"Sample predictions: {predictions[:10]}")
    print(f"Sample probabilities: {probabilities[:5, 1]}")  # Positive class probabilities
    
    # Get feature importance
    importance_df = model.get_feature_importance_df()
    print(f"\nTop 10 Most Important Features:")
    print(importance_df.head(10))
    
    # Plot feature importance
    model.plot_feature_importance()
    
    # Get model info
    model_info = model.get_model_info()
    print(f"\nModel Information:")
    for key, value in model_info.items():
        if key not in ['feature_names']:  # Skip long lists
            print(f"  {key}: {value}")
    
    # Cross-validation
    cv_results, cv_data = model.cross_validate(X_train, y_train, cv_folds=3)
    print(f"\nCross-validation Results: {cv_results}")
    
    # Explain a prediction
    explanation = model.explain_prediction(X_test.iloc[:1])
    print(f"\nPrediction Explanation for first test sample:")
    print(explanation.head(10))
    
    print("CatBoost model test completed successfully!")
