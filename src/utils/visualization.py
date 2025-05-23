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
            axes[1, 1].plot(history.history['recall'], label='Training Recall', color=self.colors['success'])
            if 'val_recall' in history.history:
                axes[1, 1].plot(history.history['val_recall'], label='Validation Recall', color=self.colors['secondary'])
            axes[1, 1].set_title('Model Recall')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Training history plot saved to {save_path}")
        
        plt.show()
    
    def plot_learning_curves(self, train_sizes, train_scores, val_scores, save_path=None):
        """
        Plot learning curves to analyze model performance vs training data size
        """
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        plt.figure(figsize=(12, 8))
        
        plt.plot(train_sizes, train_mean, 'o-', color=self.colors['primary'], 
                label='Training Score', linewidth=2, markersize=8)
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                        alpha=0.1, color=self.colors['primary'])
        
        plt.plot(train_sizes, val_mean, 'o-', color=self.colors['danger'],
                label='Cross-validation Score', linewidth=2, markersize=8)
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                        alpha=0.1, color=self.colors['danger'])
        
        plt.xlabel('Training Set Size')
        plt.ylabel('Score')
        plt.title('Learning Curves')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        
        # Add annotations for final scores
        plt.annotate(f'Final Training: {train_mean[-1]:.3f}±{train_std[-1]:.3f}',
                    xy=(train_sizes[-1], train_mean[-1]), xytext=(10, 10),
                    textcoords='offset points', fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=self.colors['primary'], alpha=0.2))
        
        plt.annotate(f'Final Validation: {val_mean[-1]:.3f}±{val_std[-1]:.3f}',
                    xy=(train_sizes[-1], val_mean[-1]), xytext=(10, -25),
                    textcoords='offset points', fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=self.colors['danger'], alpha=0.2))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Learning curves saved to {save_path}")
        
        plt.show()
    
    def create_interactive_dashboard(self, results_df, save_path=None):
        """
        Create interactive dashboard using Plotly
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Performance Metrics', 'ROC-AUC vs Execution Time',
                           'Precision vs Recall', 'Model Comparison Radar'),
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "polar"}]]
        )
        
        # 1. Performance metrics bar chart
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        for metric in metrics:
            if metric in results_df.columns:
                fig.add_trace(
                    go.Bar(name=metric.capitalize(), 
                          x=results_df['model'] if 'model' in results_df.columns else results_df.index,
                          y=results_df[metric],
                          hovertemplate=f'{metric.capitalize()}: %{{y:.3f}}<extra></extra>'),
                    row=1, col=1
                )
        
        # 2. ROC-AUC vs Execution Time scatter
        if 'roc_auc' in results_df.columns and 'execution_time' in results_df.columns:
            fig.add_trace(
                go.Scatter(x=results_df['execution_time'], y=results_df['roc_auc'],
                          mode='markers+text',
                          text=results_df['model'] if 'model' in results_df.columns else results_df.index,
                          textposition="top center",
                          marker=dict(size=12, color=self.colors['primary']),
                          name='Models',
                          hovertemplate='Model: %{text}<br>Time: %{x:.2f}s<br>ROC-AUC: %{y:.3f}<extra></extra>'),
                row=1, col=2
            )
        
        # 3. Precision vs Recall scatter
        if 'precision' in results_df.columns and 'recall' in results_df.columns:
            fig.add_trace(
                go.Scatter(x=results_df['recall'], y=results_df['precision'],
                          mode='markers+text',
                          text=results_df['model'] if 'model' in results_df.columns else results_df.index,
                          textposition="top center",
                          marker=dict(size=12, color=self.colors['success']),
                          name='Models',
                          hovertemplate='Model: %{text}<br>Recall: %{x:.3f}<br>Precision: %{y:.3f}<extra></extra>'),
                row=2, col=1
            )
        
        # 4. Radar chart for model comparison
        radar_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        available_radar_metrics = [m for m in radar_metrics if m in results_df.columns]
        
        if len(available_radar_metrics) >= 3:
            for idx, (_, row) in enumerate(results_df.iterrows()):
                model_name = row['model'] if 'model' in row else f'Model {idx+1}'
                values = [row[metric] for metric in available_radar_metrics]
                
                fig.add_trace(
                    go.Scatterpolar(r=values + [values[0]],  # Close the polygon
                                  theta=available_radar_metrics + [available_radar_metrics[0]],
                                  fill='toself',
                                  name=model_name,
                                  hovertemplate='%{theta}: %{r:.3f}<extra></extra>'),
                    row=2, col=2
                )
        
        # Update layout
        fig.update_layout(
            title_text="Cardiovascular Disease Prediction - Model Performance Dashboard",
            title_x=0.5,
            height=800,
            showlegend=True,
            hovermode='closest'
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Models", row=1, col=1)
        fig.update_yaxes(title_text="Score", row=1, col=1)
        fig.update_xaxes(title_text="Execution Time (s)", row=1, col=2)
        fig.update_yaxes(title_text="ROC-AUC", row=1, col=2)
        fig.update_xaxes(title_text="Recall", row=2, col=1)
        fig.update_yaxes(title_text="Precision", row=2, col=1)
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Interactive dashboard saved to {save_path}")
        
        fig.show()
        
        return fig
    
    def plot_clinical_decision_analysis(self, y_true, y_pred_proba, save_path=None):
        """
        Plot clinical decision analysis including cost-benefit analysis
        """
        from sklearn.metrics import precision_recall_curve, roc_curve
        
        # Handle different probability formats
        if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 1:
            proba = y_pred_proba[:, 1]
        else:
            proba = y_pred_proba
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Precision-Recall Curve
        precision, recall, pr_thresholds = precision_recall_curve(y_true, proba)
        axes[0, 0].plot(recall, precision, color=self.colors['primary'], linewidth=2)
        axes[0, 0].set_xlabel('Recall (Sensitivity)')
        axes[0, 0].set_ylabel('Precision (PPV)')
        axes[0, 0].set_title('Precision-Recall Curve')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add F1-score contours
        f_scores = np.linspace(0.2, 0.8, num=4)
        for f_score in f_scores:
            x = np.linspace(0.01, 1)
            y = f_score * x / (2 * x - f_score)
            axes[0, 0].plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.5)
            axes[0, 0].annotate(f'F1={f_score:0.1f}', xy=(0.9, y[45] + 0.02))
        
        # 2. Sensitivity vs Specificity
        fpr, tpr, roc_thresholds = roc_curve(y_true, proba)
        specificity = 1 - fpr
        
        axes[0, 1].plot(roc_thresholds, tpr, label='Sensitivity', color=self.colors['success'])
        axes[0, 1].plot(roc_thresholds, specificity, label='Specificity', color=self.colors['danger'])
        axes[0, 1].set_xlabel('Threshold')
        axes[0, 1].set_ylabel('Rate')
        axes[0, 1].set_title('Sensitivity vs Specificity by Threshold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Find optimal threshold (Youden's index)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = roc_thresholds[optimal_idx]
        axes[0, 1].axvline(optimal_threshold, color='black', linestyle='--',
                          label=f'Optimal: {optimal_threshold:.3f}')
        axes[0, 1].legend()
        
        # 3. Cost-Benefit Analysis
        # Simulate different cost scenarios
        cost_fp = 1  # Cost of false positive (unnecessary treatment)
        cost_fn = 10  # Cost of false negative (missed diagnosis)
        
        thresholds = np.linspace(0, 1, 100)
        costs = []
        
        for threshold in thresholds:
            y_pred_thresh = (proba >= threshold).astype(int)
            tn = np.sum((y_true == 0) & (y_pred_thresh == 0))
            fp = np.sum((y_true == 0) & (y_pred_thresh == 1))
            fn = np.sum((y_true == 1) & (y_pred_thresh == 0))
            tp = np.sum((y_true == 1) & (y_pred_thresh == 1))
            
            total_cost = (fp * cost_fp) + (fn * cost_fn)
            costs.append(total_cost)
        
        axes[1, 0].plot(thresholds, costs, color=self.colors['primary'], linewidth=2)
        min_cost_idx = np.argmin(costs)
        min_cost_threshold = thresholds[min_cost_idx]
        
        axes[1, 0].axvline(min_cost_threshold, color='red', linestyle='--',
                          label=f'Min Cost Threshold: {min_cost_threshold:.3f}')
        axes[1, 0].set_xlabel('Decision Threshold')
        axes[1, 0].set_ylabel('Total Cost')
        axes[1, 0].set_title(f'Cost Analysis (FP Cost={cost_fp}, FN Cost={cost_fn})')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Net Benefit Analysis
        # Calculate net benefit for different thresholds
        prevalence = np.mean(y_true)
        net_benefits = []
        
        for threshold in thresholds:
            y_pred_thresh = (proba >= threshold).astype(int)
            tp = np.sum((y_true == 1) & (y_pred_thresh == 1))
            fp = np.sum((y_true == 0) & (y_pred_thresh == 1))
            
            # Net benefit = (TP - FP * pt/(1-pt)) / N
            # where pt is probability threshold
            if threshold > 0 and threshold < 1:
                net_benefit = (tp - fp * (threshold / (1 - threshold))) / len(y_true)
            else:
                net_benefit = 0
            net_benefits.append(net_benefit)
        
        axes[1, 1].plot(thresholds, net_benefits, color=self.colors['primary'], 
                       linewidth=2, label='Model')
        
        # Add "treat all" and "treat none" strategies
        treat_all_benefit = prevalence - (1 - prevalence) * (thresholds / (1 - thresholds + 1e-10))
        treat_none_benefit = np.zeros_like(thresholds)
        
        axes[1, 1].plot(thresholds, treat_all_benefit, '--', color=self.colors['danger'],
                       label='Treat All')
        axes[1, 1].plot(thresholds, treat_none_benefit, '--', color=self.colors['secondary'],
                       label='Treat None')
        
        axes[1, 1].set_xlabel('Threshold Probability')
        axes[1, 1].set_ylabel('Net Benefit')
        axes[1, 1].set_title('Decision Curve Analysis')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Clinical decision analysis saved to {save_path}")
        
        plt.show()
        
        return optimal_threshold, min_cost_threshold
    
    def save_all_figures(self, base_path='results/figures/'):
        """
        Save all generated figures to specified directory
        """
        os.makedirs(base_path, exist_ok=True)
        logger.info(f"All figures will be saved to {base_path}")
        return base_path

def create_publication_plots(data, results, save_dir='results/figures/'):
    """
    Create all plots for publication as shown in the research paper
    """
    visualizer = ResultVisualizer()
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Data distribution plots
    if data is not None:
        visualizer.plot_data_distribution(
            data, 
            save_path=os.path.join(save_dir, 'data_distribution.png')
        )
    
    # 2. Model comparison plots
    if results is not None:
        visualizer.plot_model_comparison(
            results,
            save_path=os.path.join(save_dir, 'model_comparison.png')
        )
    
    logger.info(f"Publication plots created in {save_dir}")

if __name__ == "__main__":
    # Test the visualization module
    import pandas as pd
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    
    # Generate sample data
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create sample dataframe
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    df['age'] = np.random.randint(30, 80, len(df))
    df['sex'] = np.random.randint(0, 2, len(df))
    df['cp'] = np.random.randint(0, 4, len(df))
    df['chol'] = np.random.randint(150, 350, len(df))
    df['thalach'] = np.random.randint(100, 200, len(df))
    
    # Train sample models
    rf_model = RandomForestClassifier(random_state=42)
    lr_model = LogisticRegression(random_state=42)
    rf_model.fit(X_train, y_train)
    lr_model.fit(X_train, y_train)
    
    # Create sample results
    results_data = [
        {'model': 'KNN', 'accuracy': 0.70, 'precision': 0.76, 'recall': 0.78, 
         'f1_score': 0.80, 'roc_auc': 0.67, 'execution_time': 23.44},
        {'model': 'Logistic Regression', 'accuracy': 0.92, 'precision': 0.90, 'recall': 0.90, 
         'f1_score': 0.92, 'roc_auc': 0.88, 'execution_time': 32.59},
        {'model': 'Random Forest', 'accuracy': 0.80, 'precision': 0.84, 'recall': 0.82, 
         'f1_score': 0.90, 'roc_auc': 0.79, 'execution_time': 48.95},
        {'model': 'BDLSTM+CatBoost', 'accuracy': 0.94, 'precision': 0.93, 'recall': 0.92, 
         'f1_score': 0.94, 'roc_auc': 0.94, 'execution_time': 8.52}
    ]
    
    results_df = pd.DataFrame(results_data)
    
    # Initialize visualizer
    visualizer = ResultVisualizer()
    
    # Test plots
    print("Testing visualization module...")
    
    # 1. Data distribution
    visualizer.plot_data_distribution(df)
    
    # 2. Model comparison
    visualizer.plot_model_comparison(results_df)
    
    # 3. Feature importance (sample)
    feature_importance = np.random.random(len(feature_names))
    visualizer.plot_feature_importance(feature_importance, feature_names)
    
    # 4. ROC curve
    rf_proba = rf_model.predict_proba(X_test)
    visualizer.plot_roc_curve(y_test, rf_proba)
    
    # 5. Confusion matrix
    rf_pred = rf_model.predict(X_test)
    visualizer.plot_confusion_matrix(y_test, rf_pred)
    
    # 6. Interactive dashboard
    visualizer.create_interactive_dashboard(results_df)
    
    print("Visualization tests completed successfully!")
