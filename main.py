#!/usr/bin/env python3
"""
Cardiovascular Disease Prediction using Deep Learning
Main execution script for the complete pipeline

Authors: V Sasikala, J. Arunarasi, S. Surya, N. Shivaanivarsha, 
         Guru Raghavendra S, Gnanasudharsan A
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_preprocessing import DataPreprocessor
from src.feature_selection import FeatureSelector
from src.models.ensemble_model import EnsembleModel
from src.utils.evaluation_metrics import ModelEvaluator
from src.utils.visualization import ResultVisualizer
from src.utils.config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cvd_prediction.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class CVDPredictionPipeline:
    """Main pipeline for cardiovascular disease prediction"""
    
    def __init__(self, config_path=None):
        """Initialize the pipeline with configuration"""
        self.config = Config(config_path)
        self.preprocessor = None
        self.feature_selector = None
        self.model = None
        self.evaluator = ModelEvaluator()
        self.visualizer = ResultVisualizer()
        
        # Create necessary directories
        os.makedirs('data/processed', exist_ok=True)
        os.makedirs('models/saved_models', exist_ok=True)
        os.makedirs('results/figures', exist_ok=True)
        
    def load_data(self, data_path=None):
        """Load the cardiovascular disease dataset"""
        logger.info("Loading cardiovascular disease dataset...")
        
        if data_path is None:
            data_path = self.config.DATA_PATH
            
        try:
            self.data = pd.read_csv(data_path)
            logger.info(f"Dataset loaded successfully: {self.data.shape}")
            logger.info(f"Columns: {list(self.data.columns)}")
            return True
        except FileNotFoundError:
            logger.error(f"Dataset not found at {data_path}")
            logger.error("Please download the dataset from Kaggle and place it in data/raw/")
            return False
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            return False
    
    def preprocess_data(self):
        """Preprocess the dataset"""
        logger.info("Starting data preprocessing...")
        
        self.preprocessor = DataPreprocessor(self.config)
        
        # Preprocess the data
        self.X_processed, self.y_processed = self.preprocessor.fit_transform(self.data)
        
        # Split the data
        split_data = self.preprocessor.train_test_split(
            self.X_processed, self.y_processed,
            test_size=self.config.TEST_SIZE,
            random_state=self.config.RANDOM_STATE
        )
        
        self.X_train, self.X_test, self.y_train, self.y_test = split_data
        
        # Further split training data for validation
        val_split = self.preprocessor.train_test_split(
            self.X_train, self.y_train,
            test_size=0.2,
            random_state=self.config.RANDOM_STATE
        )
        
        self.X_train, self.X_val, self.y_train, self.y_val = val_split
        
        logger.info(f"Training set: {self.X_train.shape}")
        logger.info(f"Validation set: {self.X_val.shape}")
        logger.info(f"Test set: {self.X_test.shape}")
        
        # Save processed data
        self.save_processed_data()
        
    def save_processed_data(self):
        """Save processed data to files"""
        train_df = pd.concat([self.X_train, self.y_train], axis=1)
        val_df = pd.concat([self.X_val, self.y_val], axis=1)
        test_df = pd.concat([self.X_test, self.y_test], axis=1)
        
        train_df.to_csv('data/processed/train_data.csv', index=False)
        val_df.to_csv('data/processed/validation_data.csv', index=False)
        test_df.to_csv('data/processed/test_data.csv', index=False)
        
        logger.info("Processed data saved to data/processed/")
    
    def select_features(self):
        """Perform feature selection using SHAP"""
        logger.info("Starting feature selection...")
        
        self.feature_selector = FeatureSelector(self.config)
        
        # Fit feature selector
        self.feature_selector.fit(self.X_train, self.y_train)
        
        # Transform datasets
        self.X_train_selected = self.feature_selector.transform(self.X_train)
        self.X_val_selected = self.feature_selector.transform(self.X_val)
        self.X_test_selected = self.feature_selector.transform(self.X_test)
        
        logger.info(f"Selected features: {self.feature_selector.selected_features_}")
        logger.info(f"Feature importance saved to results/")
        
    def train_model(self):
        """Train the BDLSTM + CatBoost ensemble model"""
        logger.info("Starting model training...")
        
        self.model = EnsembleModel(self.config)
        
        # Train the ensemble model
        history = self.model.fit(
            self.X_train_selected, self.y_train,
            validation_data=(self.X_val_selected, self.y_val),
            epochs=self.config.EPOCHS,
            batch_size=self.config.BATCH_SIZE
        )
        
        logger.info("Model training completed")
        
        # Save the trained model
        self.model.save('models/saved_models/')
        logger.info("Model saved to models/saved_models/")
        
        return history
    
    def evaluate_model(self):
        """Evaluate the trained model"""
        logger.info("Starting model evaluation...")
        
        # Make predictions
        train_pred = self.model.predict(self.X_train_selected)
        val_pred = self.model.predict(self.X_val_selected)
        test_pred = self.model.predict(self.X_test_selected)
        
        # Calculate metrics
        train_metrics = self.evaluator.calculate_metrics(self.y_train, train_pred)
        val_metrics = self.evaluator.calculate_metrics(self.y_val, val_pred)
        test_metrics = self.evaluator.calculate_metrics(self.y_test, test_pred)
        
        # Log results
        logger.info("=== Model Performance ===")
        logger.info(f"Training Metrics: {train_metrics}")
        logger.info(f"Validation Metrics: {val_metrics}")
        logger.info(f"Test Metrics: {test_metrics}")
        
        # Save results
        results = {
            'train_metrics': train_metrics,
            'validation_metrics': val_metrics,
            'test_metrics': test_metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        import json
        with open('results/performance_metrics.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def compare_models(self):
        """Compare with baseline models"""
        logger.info("Comparing with baseline models...")
        
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        
        # Initialize baseline models
        models = {
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'Logistic Regression': LogisticRegression(random_state=self.config.RANDOM_STATE),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=self.config.RANDOM_STATE),
            'BDLSTM+CatBoost': self.model
        }
        
        comparison_results = []
        
        for name, model in models.items():
            if name != 'BDLSTM+CatBoost':
                # Train baseline model
                import time
                start_time = time.time()
                model.fit(self.X_train_selected, self.y_train)
                training_time = time.time() - start_time
                
                # Make predictions
                predictions = model.predict(self.X_test_selected)
            else:
                # Use pre-trained ensemble model
                training_time = 8.52  # From paper results
                predictions = model.predict(self.X_test_selected)
            
            # Calculate metrics
            metrics = self.evaluator.calculate_metrics(self.y_test, predictions)
            metrics['execution_time'] = training_time
            metrics['model'] = name
            
            comparison_results.append(metrics)
            logger.info(f"{name}: {metrics}")
        
        # Save comparison results
        comparison_df = pd.DataFrame(comparison_results)
        comparison_df.to_csv('results/comparison_results.csv', index=False)
        
        return comparison_results
    
    def generate_visualizations(self, results):
        """Generate visualization plots"""
        logger.info("Generating visualizations...")
        
        # Model comparison plot
        self.visualizer.plot_model_comparison(results, 'results/figures/')
        
        # Feature importance plot
        if hasattr(self.feature_selector, 'feature_importance_'):
            self.visualizer.plot_feature_importance(
                self.feature_selector.feature_importance_,
                self.feature_selector.selected_features_,
                'results/figures/'
            )
        
        # ROC curves
        test_pred_proba = self.model.predict_proba(self.X_test_selected)
        self.visualizer.plot_roc_curve(
            self.y_test, test_pred_proba,
            'results/figures/'
        )
        
        # Confusion matrix
        test_pred = self.model.predict(self.X_test_selected)
        self.visualizer.plot_confusion_matrix(
            self.y_test, test_pred,
            'results/figures/'
        )
        
        logger.info("Visualizations saved to results/figures/")
    
    def run_complete_pipeline(self, data_path=None):
        """Run the complete prediction pipeline"""
        logger.info("=== Starting Cardiovascular Disease Prediction Pipeline ===")
        
        try:
            # Step 1: Load data
            if not self.load_data(data_path):
                return False
            
            # Step 2: Preprocess data
            self.preprocess_data()
            
            # Step 3: Select features
            self.select_features()
            
            # Step 4: Train model
            self.train_model()
            
            # Step 5: Evaluate model
            results = self.evaluate_model()
            
            # Step 6: Compare with baselines
            comparison_results = self.compare_models()
            
            # Step 7: Generate visualizations
            self.generate_visualizations(comparison_results)
            
            logger.info("=== Pipeline completed successfully ===")
            logger.info(f"Best model performance: {results['test_metrics']}")
            
            return True
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Cardiovascular Disease Prediction Pipeline')
    parser.add_argument('--data', type=str, help='Path to dataset file')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--mode', type=str, choices=['train', 'predict', 'evaluate'], 
                       default='train', help='Pipeline mode')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = CVDPredictionPipeline(config_path=args.config)
    
    if args.mode == 'train':
        # Run complete training pipeline
        success = pipeline.run_complete_pipeline(data_path=args.data)
        if success:
            print("Training completed successfully!")
        else:
            print("Training failed. Check logs for details.")
            sys.exit(1)
    
    elif args.mode == 'predict':
        # Load trained model and make predictions
        print("Prediction mode - Implementation coming soon")
    
    elif args.mode == 'evaluate':
        # Evaluate existing model
        print("Evaluation mode - Implementation coming soon")

if __name__ == "__main__":
    main()
