"""
Training Module for Cardiovascular Disease Prediction
Handles model training, validation, and hyperparameter optimization
"""

import os
import sys
import logging
import time
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report
import warnings

# Add src to path
sys.path.append(os.path.dirname(__file__))

from data_preprocessing import DataPreprocessor
from feature_selection import FeatureSelector
from models.ensemble_model import EnsembleModel
from utils.config import Config
from utils.evaluation_metrics import ModelEvaluator
from utils.visualization import ResultVisualizer

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Comprehensive model training class for CVD prediction
    """
    
    def __init__(self, config=None):
        """
        Initialize the trainer
        
        Args:
            config: Configuration object
        """
        self.config = config if config else Config()
        self.preprocessor = None
        self.feature_selector = None
        self.model = None
        self.training_history = {}
        self.best_params = {}
        
        # Initialize components
        self.evaluator = ModelEvaluator()
        self.visualizer = ResultVisualizer()
        
        # Setup directories
        os.makedirs(self.config.MODEL_SAVE_PATH, exist_ok=True)
        os.makedirs(self.config.RESULTS_PATH, exist_ok=True)
    
    def load_and_preprocess_data(self, data_path=None):
        """
        Load and preprocess the cardiovascular disease dataset
        
        Args:
            data_path: Path to the dataset
        """
        logger.info("Loading and preprocessing data...")
        
        if data_path is None:
            data_path = self.config.DATA_PATH
        
        try:
            # Load data
            df = pd.read_csv(data_path)
            logger.info(f"Dataset loaded: {df.shape}")
            
            # Initialize preprocessor
            self.preprocessor = DataPreprocessor(self.config)
            
            # Preprocess data
            X, y = self.preprocessor.fit_transform(df)
            
            # Split data
            X_train, X_test, y_train, y_test = self.preprocessor.train_test_split(
                X, y, 
                test_size=self.config.TEST_SIZE,
                random_state=self.config.RANDOM_STATE
            )
            
            # Further split for validation
            X_train, X_val, y_train, y_val = self.preprocessor.train_test_split(
                X_train, y_train,
                test_size=self.config.VALIDATION_SIZE,
                random_state=self.config.RANDOM_STATE
            )
            
            self.X_train, self.X_val, self.X_test = X_train, X_val, X_test
            self.y_train, self.y_val, self.y_test = y_train, y_val, y_test
            
            logger.info(f"Data splits - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return False
    
    def perform_feature_selection(self):
        """
        Perform SHAP-based feature selection
        """
        logger.info("Performing feature selection...")
        
        try:
            # Initialize feature selector
            self.feature_selector = FeatureSelector(
                self.config, 
                threshold=self.config.SHAP_THRESHOLD
            )
            
            # Fit and transform
            self.X_train_selected = self.feature_selector.fit_transform(
                self.X_train, self.y_train
            )
            self.X_val_selected = self.feature_selector.transform(self.X_val)
            self.X_test_selected = self.feature_selector.transform(self.X_test)
            
            logger.info(f"Features selected: {len(self.feature_selector.selected_features_)}")
            logger.info(f"Selected features: {self.feature_selector.selected_features_}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error in feature selection: {str(e)}")
            return False
    
    def train_ensemble_model(self, use_hyperparameter_tuning=False):
        """
        Train the BDLSTM + CatBoost ensemble model
        
        Args:
            use_hyperparameter_tuning: Whether to perform hyperparameter optimization
        """
        logger.info("Training ensemble model...")
        
        try:
            # Initialize model
            self.model = EnsembleModel(self.config)
            
            if use_hyperparameter_tuning:
                self._perform_hyperparameter_tuning()
            
            # Train model
            start_time = time.time()
            
            history = self.model.fit(
                self.X_train_selected, self.y_train,
                validation_data=(self.X_val_selected, self.y_val),
                epochs=self.config.EPOCHS,
                batch_size=self.config.BATCH_SIZE
            )
            
            training_time = time.time() - start_time
            
            self.training_history = {
                'training_time': training_time,
                'history': history.history if hasattr(history, 'history') else None,
                'final_epoch': len(history.history['loss']) if hasattr(history, 'history') and 'loss' in history.history else self.config.EPOCHS
            }
            
            logger.info(f"Model training completed in {training_time:.2f} seconds")
            
            return True
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            return False
    
    def _perform_hyperparameter_tuning(self):
        """
        Perform hyperparameter optimization using RandomizedSearchCV
        """
        logger.info("Performing hyperparameter tuning...")
        
        # Define parameter space
        param_space = {
            'bdlstm_params': {
                'lstm_units_1': [64, 128, 256],
                'lstm_units_2': [32, 64, 128],
                'dense_units_1': [64, 128, 256],
                'dense_units_2': [32, 64, 128],
                'dropout_rate_1': [0.2, 0.3, 0.4],
                'dropout_rate_2': [0.1, 0.2, 0.3],
                'learning_rate': [0.001, 0.01, 0.1]
            },
            'catboost_params': {
                'iterations': [500, 1000, 1500],
                'learning_rate': [0.01, 0.1, 0.2],
                'depth': [4, 6, 8]
            },
            'ensemble_weights': [
                [0.5, 0.5],
                [0.6, 0.4],
                [0.7, 0.3],
                [0.4, 0.6]
            ]
        }
        
        # Simple grid search for demonstration
        # In practice, you might want to use more sophisticated optimization
        best_score = -np.inf
        best_config = None
        
        # Try a few different configurations
        configurations = [
            {'ensemble_weights': [0.6, 0.4]},
            {'ensemble_weights': [0.7, 0.3]},
            {'ensemble_weights': [0.5, 0.5]}
        ]
        
        for config in configurations:
            # Update configuration
            temp_config = Config()
            for key, value in config.items():
                setattr(temp_config, key.upper(), value)
            
            # Train model with this configuration
            temp_model = EnsembleModel(temp_config)
            temp_model.fit(
                self.X_train_selected, self.y_train,
                validation_data=(self.X_val_selected, self.y_val),
                epochs=50,  # Reduced epochs for tuning
                batch_size=self.config.BATCH_SIZE
            )
            
            # Evaluate on validation set
            val_pred = temp_model.predict(self.X_val_selected)
            val_score = self.evaluator.calculate_metrics(self.y_val, val_pred)['f1_score']
            
            if val_score > best_score:
                best_score = val_score
                best_config = config
        
        if best_config:
            logger.info(f"Best hyperparameters found: {best_config}")
            logger.info(f"Best validation F1-score: {best_score:.4f}")
            
            # Update configuration with best parameters
            for key, value in best_config.items():
                setattr(self.config, key.upper(), value)
            
            self.best_params = best_config
    
    def evaluate_model(self):
        """
        Comprehensive model evaluation
        """
        logger.info("Evaluating model performance...")
        
        if self.model is None:
            logger.error("Model not trained yet!")
            return None
        
        try:
            # Evaluate on all sets
            train_results = self.evaluator.evaluate_model_performance(
                self.model, self.X_train_selected, self.y_train, "Training"
            )
            
            val_results = self.evaluator.evaluate_model_performance(
                self.model, self.X_val_selected, self.y_val, "Validation"
            )
            
            test_results = self.evaluator.evaluate_model_performance(
                self.model, self.X_test_selected, self.y_test, "Test"
            )
            
            # Compile results
            evaluation_results = {
                'train_metrics': train_results,
                'validation_metrics': val_results,
                'test_metrics': test_results,
                'training_history': self.training_history,
                'best_params': self.best_params,
                'selected_features': self.feature_selector.selected_features_ if self.feature_selector else None
            }
            
            # Log results
            logger.info("=== MODEL PERFORMANCE ===")
            for split, results in [("Train", train_results), ("Validation", val_results), ("Test", test_results)]:
                logger.info(f"\n{split} Results:")
                logger.info(f"  Accuracy: {results.get('accuracy', 0):.4f}")
                logger.info(f"  Precision: {results.get('precision', 0):.4f}")
                logger.info(f"  Recall: {results.get('recall', 0):.4f}")
                logger.info(f"  F1-Score: {results.get('f1_score', 0):.4f}")
                logger.info(f"  ROC-AUC: {results.get('roc_auc', 0):.4f}")
            
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            return None
    
    def compare_with_baselines(self):
        """
        Compare ensemble model with baseline algorithms
        """
        logger.info("Comparing with baseline models...")
        
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.ensemble import AdaBoostClassifier
        from xgboost import XGBClassifier
        
        # Initialize baseline models
        baseline_models = {
            'KNN': KNeighborsClassifier(**self.config.BASELINE_MODELS['KNN']),
            'Logistic Regression': LogisticRegression(**self.config.BASELINE_MODELS['LogisticRegression']),
            'Random Forest': RandomForestClassifier(**self.config.BASELINE_MODELS['RandomForest']),
            'XGBoost': XGBClassifier(**self.config.BASELINE_MODELS['XGBoost']),
            'AdaBoost': AdaBoostClassifier(**self.config.BASELINE_MODELS['AdaBoost'])
        }
        
        comparison_results = []
        
        # Train and evaluate each baseline model
        for name, model in baseline_models.items():
            logger.info(f"Training {name}...")
            
            start_time = time.time()
            model.fit(self.X_train_selected, self.y_train)
            training_time = time.time() - start_time
            
            # Evaluate
            results = self.evaluator.evaluate_model_performance(
                model, self.X_test_selected, self.y_test, name
            )
            results['training_time'] = training_time
            comparison_results.append(results)
        
        # Add ensemble model results
        if self.model:
            ensemble_results = self.evaluator.evaluate_model_performance(
                self.model, self.X_test_selected, self.y_test, "BDLSTM+CatBoost"
            )
            ensemble_results['training_time'] = self.training_history.get('training_time', 0)
            comparison_results.append(ensemble_results)
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame([
            {
                'Model': result['model_name'],
                'Accuracy': result.get('accuracy', 0),
                'Precision': result.get('precision', 0),
                'Recall': result.get('recall', 0),
                'F1-Score': result.get('f1_score', 0),
                'ROC-AUC': result.get('roc_auc', 0),
                'Training Time (s)': result.get('training_time', 0)
            }
            for result in comparison_results
        ])
        
        # Sort by F1-Score
        comparison_df = comparison_df.sort_values('F1-Score', ascending=False)
        
        logger.info("\n=== MODEL COMPARISON ===")
        logger.info(f"\n{comparison_df.to_string(index=False)}")
        
        return comparison_df, comparison_results
    
    def perform_cross_validation(self, cv_folds=5):
        """
        Perform cross-validation analysis
        
        Args:
            cv_folds: Number of cross-validation folds
        """
        logger.info(f"Performing {cv_folds}-fold cross-validation...")
        
        try:
            # Combine training and validation sets for CV
            X_combined = pd.concat([self.X_train_selected, self.X_val_selected])
            y_combined = pd.concat([self.y_train, self.y_val])
            
            # Initialize a simpler version of the model for CV
            from sklearn.ensemble import RandomForestClassifier
            cv_model = RandomForestClassifier(
                n_estimators=100,
                random_state=self.config.RANDOM_STATE
            )
            
            # Perform cross-validation
            cv_scores = cross_val_score(
                cv_model, X_combined, y_combined,
                cv=cv_folds,
                scoring=self.config.CV_SCORING,
                n_jobs=-1
            )
            
            cv_results = {
                'cv_scores': cv_scores.tolist(),
                'mean_cv_score': cv_scores.mean(),
                'std_cv_score': cv_scores.std(),
                'cv_folds': cv_folds
            }
            
            logger.info(f"Cross-validation results:")
            logger.info(f"  Mean {self.config.CV_SCORING}: {cv_scores.mean():.4f}")
            logger.info(f"  Std {self.config.CV_SCORING}: {cv_scores.std():.4f}")
            logger.info(f"  Individual scores: {cv_scores}")
            
            return cv_results
            
        except Exception as e:
            logger.error(f"Error in cross-validation: {str(e)}")
            return None
    
    def save_model_and_results(self, evaluation_results, comparison_results=None):
        """
        Save trained model and all results
        
        Args:
            evaluation_results: Results from model evaluation
            comparison_results: Results from baseline comparison
        """
        logger.info("Saving model and results...")
        
        try:
            # Save model
            if self.model:
                self.model.save(self.config.MODEL_SAVE_PATH)
            
            # Save preprocessor
            import joblib
            joblib.dump(self.preprocessor, 
                       os.path.join(self.config.MODEL_SAVE_PATH, 'preprocessor.pkl'))
            
            # Save feature selector
            if self.feature_selector:
                joblib.dump(self.feature_selector,
                           os.path.join(self.config.MODEL_SAVE_PATH, 'feature_selector.pkl'))
            
            # Save evaluation results
            results_path = os.path.join(self.config.RESULTS_PATH, 'evaluation_results.json')
            with open(results_path, 'w') as f:
                json.dump(evaluation_results, f, indent=2, default=str)
            
            # Save comparison results
            if comparison_results:
                comparison_path = os.path.join(self.config.RESULTS_PATH, 'comparison_results.json')
                with open(comparison_path, 'w') as f:
                    json.dump(comparison_results, f, indent=2, default=str)
            
            # Save configuration
            config_path = os.path.join(self.config.MODEL_SAVE_PATH, 'training_config.json')
            config_dict = {
                key: value for key, value in self.config.__dict__.items()
                if not key.startswith('_')
            }
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=2, default=str)
            
            logger.info(f"Model and results saved to {self.config.MODEL_SAVE_PATH}")
            
        except Exception as e:
            logger.error(f"Error saving model and results: {str(e)}")
    
    def generate_visualizations(self, comparison_df=None):
        """
        Generate all visualizations
        
        Args:
            comparison_df: Model comparison DataFrame
        """
        logger.info("Generating visualizations...")
        
        try:
            figures_path = os.path.join(self.config.RESULTS_PATH, 'figures')
            os.makedirs(figures_path, exist_ok=True)
            
            # Model comparison plots
            if comparison_df is not None:
                self.visualizer.plot_model_comparison(
                    comparison_df,
                    save_path=os.path.join(figures_path, 'model_comparison.png')
                )
            
            # Feature importance plots
            if self.feature_selector and hasattr(self.feature_selector, 'feature_importance_'):
                self.visualizer.plot_feature_importance(
                    self.feature_selector.feature_importance_,
                    self.feature_selector.selected_features_,
                    save_path=os.path.join(figures_path, 'feature_importance.png')
                )
            
            # ROC curves and confusion matrices
            if self.model:
                test_pred_proba = self.model.predict_proba(self.X_test_selected)
                test_pred = self.model.predict(self.X_test_selected)
                
                # ROC curve
                self.visualizer.plot_roc_curve(
                    self.y_test, test_pred_proba,
                    save_path=os.path.join(figures_path, 'roc_curve.png')
                )
                
                # Confusion matrix
                self.visualizer.plot_confusion_matrix(
                    self.y_test, test_pred,
                    save_path=os.path.join(figures_path, 'confusion_matrix.png')
                )
            
            # Training history plots
            if self.training_history.get('history'):
                self.visualizer.plot_training_history(
                    type('History', (), {'history': self.training_history['history']})(),
                    save_path=os.path.join(figures_path, 'training_history.png')
                )
            
            logger.info(f"Visualizations saved to {figures_path}")
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {str(e)}")
    
    def run_complete_training_pipeline(self, data_path=None, use_hyperparameter_tuning=False):
        """
        Run the complete training pipeline
        
        Args:
            data_path: Path to dataset
            use_hyperparameter_tuning: Whether to perform hyperparameter tuning
        """
        logger.info("=== STARTING COMPLETE TRAINING PIPELINE ===")
        
        try:
            # Step 1: Load and preprocess data
            if not self.load_and_preprocess_data(data_path):
                return False
            
            # Step 2: Feature selection
            if not self.perform_feature_selection():
                return False
            
            # Step 3: Train model
            if not self.train_ensemble_model(use_hyperparameter_tuning):
                return False
            
            # Step 4: Evaluate model
            evaluation_results = self.evaluate_model()
            if evaluation_results is None:
                return False
            
            # Step 5: Compare with baselines
            comparison_df, comparison_results = self.compare_with_baselines()
            
            # Step 6: Cross-validation
            cv_results = self.perform_cross_validation()
            if cv_results:
                evaluation_results['cross_validation'] = cv_results
            
            # Step 7: Save everything
            self.save_model_and_results(evaluation_results, comparison_results)
            
            # Step 8: Generate visualizations
            self.generate_visualizations(comparison_df)
            
            logger.info("=== TRAINING PIPELINE COMPLETED SUCCESSFULLY ===")
            
            # Print summary
            test_metrics = evaluation_results['test_metrics']
            logger.info("\n=== FINAL RESULTS SUMMARY ===")
            logger.info(f"Test Accuracy: {test_metrics.get('accuracy', 0):.4f}")
            logger.info(f"Test Precision: {test_metrics.get('precision', 0):.4f}")
            logger.info(f"Test Recall: {test_metrics.get('recall', 0):.4f}")
            logger.info(f"Test F1-Score: {test_metrics.get('f1_score', 0):.4f}")
            logger.info(f"Test ROC-AUC: {test_metrics.get('roc_auc', 0):.4f}")
            logger.info(f"Training Time: {self.training_history.get('training_time', 0):.2f} seconds")
            
            return True
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {str(e)}")
            return False

def main():
    """Main function for running training"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train CVD Prediction Model')
    parser.add_argument('--data', type=str, help='Path to dataset')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--tune', action='store_true', help='Perform hyperparameter tuning')
    
    args = parser.parse_args()
    
    # Initialize trainer
    config = Config(args.config) if args.config else Config()
    trainer = ModelTrainer(config)
    
    # Run training
    success = trainer.run_complete_training_pipeline(
        data_path=args.data,
        use_hyperparameter_tuning=args.tune
    )
    
    if success:
        print("✅ Training completed successfully!")
    else:
        print("❌ Training failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
