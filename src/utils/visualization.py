"""
Visualization Module for Cardiovascular Disease Prediction
Creates all plots and visualizations from the research paper
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
import os
import logging

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class ResultVisualizer:
    """
    Comprehensive visualization class for CVD prediction results
    """
    
    def __init__(self, figsize=(10, 6), dpi=300):
        self.figsize = figsize
        self.dpi = dpi
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'success': '#F18F01',
            'danger': '#C73E1D',
            'ensemble': '#FF6B35',
            'baseline': '#87CEEB'
        }
    
    def plot_data_distribution(self, df, save_path=None):
        """
        Plot data distribution analysis as shown in the paper
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Target distribution
        target_counts = df['target'].value_counts()
        axes[0, 0].pie(target_counts.values, labels=['Healthy', 'Heart Disease'], 
                      autopct='%1.1f%%', colors=[self.colors['success'], self.colors['danger']])
        axes[0, 0].set_title('Heart Disease Distribution')
        
        # 2. Age distribution by target
        sns.histplot(data=df, x='age', hue='target', bins=20, ax=axes[0, 1])
        axes[0, 1].set_title('Age Distribution by Heart Disease Status')
        axes[0, 1].legend(['Healthy', 'Heart Disease'])
        
        # 3. Chest pain type distribution
        if 'cp' in df.columns:
            cp_counts = df.groupby(['cp', 'target']).size().unstack()
            cp_counts.plot(kind='bar', ax=axes[0, 2], color=[self.colors['success'], self.colors['danger']])
            axes[0, 2].set_title('Chest Pain Type vs Heart Disease')
            axes[0, 2].legend(['Healthy', 'Heart Disease'])
            axes[0, 2].tick_params(axis='x', rotation=45)
        
        # 4. Gender distribution
        if 'sex' in df.columns:
            gender_counts = df.groupby(['sex', 'target']).size().unstack()
            gender_counts.plot(kind='bar', ax=axes[1, 0], color=[self.colors['success'], self.colors['danger']])
            axes[1, 0].set_title('Gender vs Heart Disease')
            axes[1, 0].legend(['Healthy', 'Heart Disease'])
            axes[1, 0].set_xticklabels(['Female', 'Male'], rotation=0)
        
        # 5. Cholesterol distribution
        if 'chol' in df.columns:
            sns.boxplot(data=df, x='target', y='chol', ax=axes[1, 1])
            axes[1, 1].set_title('Cholesterol Levels by Heart Disease Status')
            axes[1, 1].set_xticklabels(['Healthy', 'Heart Disease'])
        
        # 6. Maximum heart rate
        if 'thalach' in df.columns:
            sns.boxplot(data=df, x='target', y='thalach', ax=axes[1, 2])
            axes[1, 2].set_title('Maximum Heart Rate by Heart Disease Status')
            axes[1, 2].set_xticklabels(['Healthy', 'Heart Disease'])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Data distribution plot saved to {save_path}")
        
        plt.show()
    
    def plot_model_comparison(self, comparison_results, save_path=None):
        """
        Create model comparison visualizations as shown in the paper
        """
        # Convert results to DataFrame if needed
        if isinstance(comparison_results, list):
            df = pd.DataFrame(comparison_results)
        else:
            df = comparison_results.copy()
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Performance Metrics Comparison (Bar Chart)
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        x_pos = np.arange(len(df))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            if metric in df.columns:
                axes[0, 0].bar(x_pos + i*width, df[metric], width, 
                              label=metric.capitalize(), alpha=0.8)
        
        axes[0, 0].set_xlabel('Models')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Performance Metrics Comparison')
        axes[0, 0].set_xticks(x_pos + width * 1.5)
        axes[0, 0].set_xticklabels(df['model'] if 'model' in df.columns else df.index)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. ROC-AUC vs Execution Time
        if 'roc_auc' in df.columns and 'execution_time' in df.columns:
            colors = [self.colors['ensemble'] if 'BDLSTM+CatBoost' in str(model) 
                     else self.colors['baseline'] for model in df['model'] if 'model' in df.columns]
            
            scatter = axes[0, 1].scatter(df['execution_time'], df['roc_auc'], 
                                       c=colors, s=100, alpha=0.7)
            
            # Add model labels
            for i, model in enumerate(df['model'] if 'model' in df.columns else df.index):
                axes[0, 1].annotate(model, (df['execution_time'].iloc[i], df['roc_auc'].iloc[i]),
                                  xytext=(5, 5), textcoords='offset points', fontsize=9)
            
            axes[0, 1].set_xlabel('Execution Time (seconds)')
            axes[0, 1].set_ylabel('ROC-AUC Score')
            axes[0, 1].set_title('ROC-AUC vs Execution Time')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Accuracy vs F1-Score
        if 'accuracy' in df.columns and 'f1_score' in df.columns:
            colors = [self.colors['ensemble'] if 'BDLSTM+CatBoost' in str(model) 
                     else self.colors['baseline'] for model in df['model'] if 'model' in df.columns]
            
            axes[1, 0].scatter(df['accuracy'], df['f1_score'], c=colors, s=100, alpha=0.7)
            
            # Add model labels
            for i, model in enumerate(df['model'] if 'model' in df.columns else df.index):
                axes[1, 0].annotate(model, (df['accuracy'].iloc[i], df['f1_score'].iloc[i]),
                                  xytext=(5, 5), textcoords='offset points', fontsize=9)
            
            axes[1, 0].set_xlabel('Accuracy')
            axes[1, 0].set_ylabel('F1-Score')
            axes[1, 0].set_title('Accuracy vs F1-Score')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Performance Summary Radar Chart
        metrics_for_radar = ['accuracy', 'precision', 'recall', 'f1_score']
        available_metrics = [m for m in metrics_for_radar if m in df.columns]
        
        if len(available_metrics) >= 3:
            # Create radar chart data
            angles = np.linspace(0, 2 * np.pi, len(available_metrics), endpoint=False).tolist()
            angles += angles[:1]  # Complete the circle
            
            ax = plt.subplot(2, 2, 4, projection='polar')
            
            for i, (idx, row) in enumerate(df.iterrows()):
                values = [row[metric] for metric in available_metrics]
                values += values[:1]  # Complete the circle
                
                model_name = row['model'] if 'model' in row else f'Model {i+1}'
                color = self.colors['ensemble'] if 'BDLSTM+CatBoost' in str(model_name) else self.colors['baseline']
                
                ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=color)
                ax.fill(angles, values, alpha=0.1, color=color)
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels([m.capitalize() for m in available_metrics])
            ax.set_ylim(0, 1)
            ax.set_title('Performance Radar Chart')
            ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Model comparison plot saved to {save_path}")
        
        plt.show()
    
    def plot_feature_importance(self, feature_importance, feature_names, save_path=None, top_k=15):
        """
        Plot feature importance as shown in the paper
        """
        # Create DataFrame for easier manipulation
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False).head(top_k)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        
        # 1. Horizontal bar chart
        bars = ax1.barh(range(len(importance_df)), importance_df['importance'],
                       color=self.colors['primary'], alpha=0.7)
        ax1.set_yticks(range(len(importance_df)))
        ax1.set_yticklabels(importance_df['feature'])
        ax1.set_xlabel('SHAP Feature Importance')
        ax1.set_title(f'Top {top_k} Most Important Features')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (bar, importance) in enumerate(zip(bars, importance_df['importance'])):
            ax1.text(importance + 0.001, i, f'{importance:.3f}', 
                    va='center', ha='left', fontweight='bold')
        
        # 2. Pie chart for top 10
        top_10 = importance_df.head(10)
        others_sum = importance_df.iloc[10:]['importance'].sum()
        
        pie_data = list(top_10['importance']) + [others_sum] if len(importance_df) > 10 else list(top_10['importance'])
        pie_labels = list(top_10['feature']) + ['Others'] if len(importance_df) > 10 else list(top_10['feature'])
        
        ax2.pie(pie_data, labels=pie_labels, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Feature Importance Distribution')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
        
        plt.show()
    
    def plot_roc_curve(self, y_true, y_pred_proba, save_path=None, title="ROC Curve Analysis"):
        """
        Plot ROC curve with additional analysis
        """
        from sklearn.metrics import roc_curve, auc
        
        # Handle different probability formats
        if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 1:
            proba = y_pred_proba[:, 1]
        else:
            proba = y_pred_proba
        
        fpr, tpr, thresholds = roc_curve(y_true, proba)
        roc_auc = auc(fpr, tpr)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 1. ROC Curve
        ax1.plot(fpr, tpr, color=self.colors['primary'], lw=2, 
                label=f'ROC Curve (AUC = {roc_auc:.3f})')
        ax1.plot([0, 1], [0, 1], color=self.colors['danger'], lw=2, linestyle='--',
                label='Random Classifier (AUC = 0.500)')
        
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax1.legend(loc="lower right")
        ax1.grid(True, alpha=0.3)
        
        # 2. Threshold Analysis
        # Find optimal threshold (Youden's index)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        
        ax2.plot(thresholds, tpr, label='True Positive Rate', color=self.colors['success'])
        ax2.plot(thresholds, fpr, label='False Positive Rate', color=self.colors['danger'])
        ax2.plot(thresholds, tpr - fpr, label="Youden's Index", color=self.colors['primary'])
        
        ax2.axvline(optimal_threshold, color='black', linestyle='--', 
                   label=f'Optimal Threshold = {optimal_threshold:.3f}')
        
        ax2.set_xlabel('Threshold')
        ax2.set_ylabel('Rate')
        ax2.set_title('Threshold Analysis')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"ROC curve analysis saved to {save_path}")
        
        plt.show()
        
        return optimal_threshold
    
    def plot_confusion_matrix(self, y_true, y_pred, save_path=None, 
                            class_names=['Healthy', 'Heart Disease']):
        """
        Plot enhanced confusion matrix with clinical interpretation
        """
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 1. Standard confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names, ax=ax1)
        ax1.set_title('Confusion Matrix (Counts)')
        ax1.set_xlabel('Predicted Label')
        ax1.set_ylabel('True Label')
        
        # 2. Normalized confusion matrix with percentages
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Oranges',
                   xticklabels=class_names, yticklabels=class_names, ax=ax2)
        ax2.set_title('Normalized Confusion Matrix (Percentages)')
        ax2.set_xlabel('Predicted Label')
        ax2.set_ylabel('True Label')
        
        # Add clinical interpretation
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        
        fig.suptitle(f'Clinical Performance: Sensitivity = {sensitivity:.3f}, Specificity = {specificity:.3f}',
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def plot_training_history(self, history, save_path=None):
        """
        Plot training history for deep learning models
        """
        if not hasattr(history, 'history'):
            logger.warning("No training history available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Loss
        if 'loss' in history.history:
            axes[0, 0].plot(history.history['loss'], label='Training Loss', color=self.colors['primary'])
            if 'val_loss' in history.history:
                axes[0, 0].plot(history.history['val_loss'], label='Validation Loss', color=self.colors['danger'])
            axes[0, 0].set_title('Model Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Accuracy
        if 'accuracy' in history.history:
            axes[0, 1].plot(history.history['accuracy'], label='Training Accuracy', color=self.colors['success'])
            if 'val_accuracy' in history.history:
                axes[0, 1].plot(history.history['val_accuracy'], label='Validation Accuracy', color=self.colors['secondary'])
            axes[0, 1].set_title('Model Accuracy')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Precision
        if 'precision' in history.history:
            axes[1, 0].plot(history.history['precision'], label='Training Precision', color=self.colors['primary'])
            if 'val_precision' in history.history:
                axes[1, 0].plot(history.history['val_precision'], label='Validation Precision', color=self.colors['danger'])
            axes[1, 0].set_title('Model Precision')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Recall
        if 'recall' in history.history:
            axes[1, 1].plot(history.history['
