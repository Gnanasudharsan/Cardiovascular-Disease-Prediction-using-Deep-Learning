"""
Evaluation Metrics Module
Implements all evaluation metrics used in the research paper
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import time
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """
    Comprehensive model evaluation class
    Implements all evaluation metrics from the research paper
    """
    
    def __init__(self):
        self.results_history = []
    
    def calculate_metrics(self, y_true, y_pred, y_pred_proba=None) -> Dict:
        """
        Calculate all evaluation metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)
            
        Returns:
            Dictionary containing all metrics
        """
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='binary')
        metrics['recall'] = recall_score(y_true, y_pred, average='binary')
        metrics['f1_score'] = f1_score(y_true, y_pred, average='binary')
        
        # Specificity (True Negative Rate)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Same as recall
        
        # ROC-AUC if probabilities are provided
        if y_pred_proba is not None:
            if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 1:
                # Multi-class probabilities, use positive class
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
            else:
                # Binary probabilities
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
                
            # Average Precision Score
            if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 1:
                metrics['avg_precision'] = average_precision_score(y_true, y_pred_proba[:, 1])
            else:
                metrics['avg_precision'] = average_precision_score(y_true, y_pred_proba)
        
        # Confusion Matrix components
        metrics['true_positives'] = int(tp)
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        
        # Additional metrics for healthcare
        metrics['positive_predictive_value'] = metrics['precision']  # Same as precision
        metrics['negative_predictive_value'] = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        
        # Diagnostic metrics
        metrics['prevalence'] = (tp + fn) / (tp + tn + fp + fn)
        metrics['detection_rate'] = tp / (tp + tn + fp + fn)
        metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        
        return metrics
    
    def evaluate_model_performance(self, model, X_test, y_test, model_name="Model") -> Dict:
        """
        Comprehensive model evaluation with timing
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            model_name: Name of the model for logging
            
        Returns:
            Dictionary with performance metrics and timing
        """
        logger.info(f"Evaluating {model_name} performance...")
        
        # Time the prediction
        start_time = time.time()
        y_pred = model.predict(X_test)
        prediction_time = time.time() - start_time
        
        # Get probabilities if available
        y_pred_proba = None
        try:
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)
            elif hasattr(model, 'decision_function'):
                # For SVM and similar models
                decision_scores = model.decision_function(X_test)
                # Convert to probabilities using sigmoid
                y_pred_proba = 1 / (1 + np.exp(-decision_scores))
        except Exception as e:
            logger.warning(f"Could not get prediction probabilities: {e}")
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_test, y_pred, y_pred_proba)
        
        # Add timing information
        metrics['prediction_time'] = prediction_time
        metrics['predictions_per_second'] = len(X_test) / prediction_time if prediction_time > 0 else 0
        metrics['model_name'] = model_name
        
        # Store results
        self.results_history.append({
            'model_name': model_name,
            'metrics': metrics,
            'timestamp': time.time()
        })
        
        return metrics
    
    def compare_models(self, results_list: List[Dict]) -> pd.DataFrame:
        """
        Compare multiple model results
        
        Args:
            results_list: List of result dictionaries from evaluate_model_performance
            
        Returns:
            DataFrame with comparison metrics
        """
        comparison_data = []
        
        for result in results_list:
            model_data = {
                'Model': result.get('model_name', 'Unknown'),
                'Accuracy': result.get('accuracy', 0),
                'Precision': result.get('precision', 0),
                'Recall': result.get('recall', 0),
                'F1-Score': result.get('f1_score', 0),
                'ROC-AUC': result.get('roc_auc', 0),
                'Specificity': result.get('specificity', 0),
                'Execution Time (s)': result.get('prediction_time', 0)
            }
            comparison_data.append(model_data)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Sort by F1-Score (primary metric in paper)
        comparison_df = comparison_df.sort_values('F1-Score', ascending=False)
        
        return comparison_df
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names=None, 
                            save_path=None, title="Confusion Matrix"):
        """
        Plot confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Names of classes
            save_path: Path to save plot
            title: Plot title
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        
        if class_names is None:
            class_names = ['Healthy', 'Heart Disease']
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(title)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        # Add percentages
        total = cm.sum()
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j + 0.5, i + 0.7, f'({cm[i,j]/total:.1%})', 
                        ha='center', va='center', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def plot_roc_curve(self, y_true, y_pred_proba, save_path=None, title="ROC Curve"):
        """
        Plot ROC curve
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            save_path: Path to save plot
            title: Plot title
        """
        # Handle different probability formats
        if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 1:
            proba = y_pred_proba[:, 1]  # Use positive class probabilities
        else:
            proba = y_pred_proba
        
        fpr, tpr, thresholds = roc_curve(y_true, proba)
        auc_score = roc_auc_score(y_true, proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curve saved to {save_path}")
        
        plt.show()
        
        return fpr, tpr, auc_score
    
    def plot_precision_recall_curve(self, y_true, y_pred_proba, save_path=None, 
                                   title="Precision-Recall Curve"):
        """
        Plot Precision-Recall curve
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            save_path: Path to save plot
            title: Plot title
        """
        # Handle different probability formats
        if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 1:
            proba = y_pred_proba[:, 1]  # Use positive class probabilities
        else:
            proba = y_pred_proba
        
        precision, recall, thresholds = precision_recall_curve(y_true, proba)
        avg_precision = average_precision_score(y_true, proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, linewidth=2, 
                label=f'PR Curve (AP = {avg_precision:.3f})')
        
        # Add baseline (random classifier)
        baseline = sum(y_true) / len(y_true)
        plt.axhline(y=baseline, color='k', linestyle='--', linewidth=1, 
                   label=f'Random Classifier (AP = {baseline:.3f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Precision-Recall curve saved to {save_path}")
        
        plt.show()
        
        return precision, recall, avg_precision
    
    def plot_model_comparison(self, comparison_df: pd.DataFrame, 
                            save_path=None, title="Model Performance Comparison"):
        """
        Plot model comparison chart
        
        Args:
            comparison_df: DataFrame from compare_models
            save_path: Path to save plot
            title: Plot title
        """
        metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        
        # Filter available metrics
        available_metrics = [m for m in metrics_to_plot if m in comparison_df.columns]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # Plot each metric
        for i, metric in enumerate(available_metrics):
            ax = axes[i]
            bars = ax.bar(comparison_df['Model'], comparison_df[metric], 
                         color=['red' if 'BDLSTM+CatBoost' in model else 'skyblue' 
                               for model in comparison_df['Model']])
            
            ax.set_title(f'{metric} Comparison')
            ax.set_ylabel(metric)
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, comparison_df[metric]):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot execution time in the last subplot
        if 'Execution Time (s)' in comparison_df.columns:
            ax = axes[len(available_metrics)]
            bars = ax.bar(comparison_df['Model'], comparison_df['Execution Time (s)'], 
                         color=['red' if 'BDLSTM+CatBoost' in model else 'lightcoral' 
                               for model in comparison_df['Model']])
            
            ax.set_title('Execution Time Comparison')
            ax.set_ylabel('Time (seconds)')
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, comparison_df['Execution Time (s)']):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       f'{value:.2f}s', ha='center', va='bottom', fontweight='bold')
        
        # Hide unused subplots
        for i in range(len(available_metrics) + 1, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Model comparison plot saved to {save_path}")
        
        plt.show()
    
    def generate_classification_report(self, y_true, y_pred, 
                                     class_names=None, save_path=None):
        """
        Generate detailed classification report
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Names of classes
            save_path: Path to save report
        """
        if class_names is None:
            class_names = ['Healthy', 'Heart Disease']
        
        report = classification_report(y_true, y_pred, 
                                     target_names=class_names, 
                                     digits=4)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write("Classification Report\n")
                f.write("=" * 50 + "\n\n")
                f.write(report)
            logger.info(f"Classification report saved to {save_path}")
        
        return report
    
    def calculate_clinical_metrics(self, y_true, y_pred):
        """
        Calculate clinical significance metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary with clinical metrics
        """
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        clinical_metrics = {
            'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0.0,  # True Positive Rate
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0,  # True Negative Rate
            'positive_predictive_value': tp / (tp + fp) if (tp + fp) > 0 else 0.0,  # Precision
            'negative_predictive_value': tn / (tn + fn) if (tn + fn) > 0 else 0.0,
            'diagnostic_accuracy': (tp + tn) / (tp + tn + fp + fn),
            'positive_likelihood_ratio': (tp / (tp + fn)) / (fp / (fp + tn)) if (fp + tn) > 0 and (tp + fn) > 0 else 0.0,
            'negative_likelihood_ratio': (fn / (tp + fn)) / (tn / (tn + fp)) if (tn + fp) > 0 and (tp + fn) > 0 else 0.0,
            'diagnostic_odds_ratio': (tp * tn) / (fp * fn) if (fp * fn) > 0 else float('inf')
        }
        
        return clinical_metrics

if __name__ == "__main__":
    # Test the evaluator
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    
    # Generate sample data
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train models
    rf_model = RandomForestClassifier(random_state=42)
    lr_model = LogisticRegression(random_state=42)
    
    rf_model.fit(X_train, y_train)
    lr_model.fit(X_train, y_train)
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Evaluate models
    rf_results = evaluator.evaluate_model_performance(rf_model, X_test, y_test, "Random Forest")
    lr_results = evaluator.evaluate_model_performance(lr_model, X_test, y_test, "Logistic Regression")
    
    print("Random Forest Results:")
    for metric, value in rf_results.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")
    
    print("\nLogistic Regression Results:")
    for metric, value in lr_results.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")
    
    # Compare models
    comparison_df = evaluator.compare_models([rf_results, lr_results])
    print("\nModel Comparison:")
    print(comparison_df)
    
    # Plot comparisons
    evaluator.plot_model_comparison(comparison_df)
    
    # ROC curves
    rf_proba = rf_model.predict_proba(X_test)
    evaluator.plot_roc_curve(y_test, rf_proba, title="Random Forest ROC Curve")
