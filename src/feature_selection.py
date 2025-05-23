"""
Feature Selection Module using SHAP values
Implementation based on the research paper methodology
"""

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.base import BaseEstimator, TransformerMixin
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os

logger = logging.getLogger(__name__)

class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Feature selector using SHAP (Shapley Additive Explanations)
    
    Selects features with Shapley values greater than threshold (0.1 from paper)
    """
    
    def __init__(self, config, threshold=0.1):
        """
        Initialize feature selector
        
        Args:
            config: Configuration object
            threshold: SHAP value threshold for feature selection
        """
        self.config = config
        self.threshold = threshold
        self.base_model = None
        self.explainer = None
        self.shap_values_ = None
        self.feature_importance_ = None
        self.selected_features_ = None
        self.selected_indices_ = None
        self.is_fitted = False
    
    def fit(self, X, y):
        """
        Fit the feature selector using gradient boosting and SHAP
        
        Args:
            X: Training features
            y: Training labels
        """
        logger.info("Starting feature selection using SHAP...")
        
        # Train gradient boosting model for SHAP analysis
        self.base_model = GradientBoostingClassifier(
            n_estimators=100,
            random_state=self.config.RANDOM_STATE,
            max_depth=6
        )
        
        self.base_model.fit(X, y)
        logger.info("Gradient boosting model trained for SHAP analysis")
        
        # Create SHAP explainer
        self.explainer = shap.TreeExplainer(self.base_model)
        
        # Calculate SHAP values
        # Use a sample of data for efficiency if dataset is large
        sample_size = min(len(X), 1000)
        X_sample = X.sample(n=sample_size, random_state=self.config.RANDOM_STATE)
        
        self.shap_values_ = self.explainer.shap_values(X_sample)
        
        # Handle multi-class case (take positive class SHAP values)
        if isinstance(self.shap_values_, list):
            shap_values_positive = self.shap_values_[1]  # Positive class
        else:
            shap_values_positive = self.shap_values_
        
        # Calculate mean absolute SHAP values for each feature
        self.feature_importance_ = np.abs(shap_values_positive).mean(axis=0)
        
        # Select features with importance above threshold
        self.selected_indices_ = np.where(self.feature_importance_ >= self.threshold)[0]
        self.selected_features_ = X.columns[self.selected_indices_].tolist()
        
        logger.info(f"Selected {len(self.selected_features_)} features out of {X.shape[1]} total features")
        logger.info(f"Selected features: {self.selected_features_}")
        
        # Save feature importance results
        self._save_feature_importance(X.columns)
        
        self.is_fitted = True
        return self
    
    def transform(self, X):
        """
        Transform data by selecting important features
        
        Args:
            X: Input features
            
        Returns:
            Transformed features with selected columns only
        """
        if not self.is_fitted:
            raise ValueError("FeatureSelector must be fitted before transform")
        
        return X.iloc[:, self.selected_indices_]
    
    def fit_transform(self, X, y):
        """
        Fit and transform in one step
        
        Args:
            X: Training features
            y: Training labels
            
        Returns:
            Transformed features
        """
        return self.fit(X, y).transform(X)
    
    def get_feature_importance_df(self, feature_names):
        """
        Get feature importance as DataFrame
        
        Args:
            feature_names: List of feature names
            
        Returns:
            DataFrame with feature importance scores
        """
        if not self.is_fitted:
            raise ValueError("FeatureSelector must be fitted first")
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.feature_importance_,
            'selected': self.feature_importance_ >= self.threshold
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def plot_feature_importance(self, feature_names, save_path=None, top_k=20):
        """
        Plot feature importance
        
        Args:
            feature_names: List of feature names
            save_path: Path to save plot
            top_k: Number of top features to display
        """
        if not self.is_fitted:
            raise ValueError("FeatureSelector must be fitted first")
        
        importance_df = self.get_feature_importance_df(feature_names)
        top_features = importance_df.head(top_k)
        
        plt.figure(figsize=(12, 8))
        
        # Create color map for selected vs non-selected features
        colors = ['red' if selected else 'gray' for selected in top_features['selected']]
        
        bars = plt.barh(range(len(top_features)), top_features['importance'], color=colors)
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('SHAP Feature Importance')
        plt.title(f'Top {top_k} Features by SHAP Importance\n(Red = Selected, Gray = Not Selected)')
        plt.gca().invert_yaxis()
        
        # Add threshold line
        plt.axvline(x=self.threshold, color='blue', linestyle='--', 
                   label=f'Selection Threshold ({self.threshold})')
        plt.legend()
        
        # Add importance values as text
        for i, (importance, selected) in enumerate(zip(top_features['importance'], top_features['selected'])):
            plt.text(importance + 0.001, i, f'{importance:.3f}', 
                    va='center', ha='left', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
        
        plt.show()
    
    def plot_shap_values(self, X_sample=None, save_path=None):
        """
        Plot SHAP values summary
        
        Args:
            X_sample: Sample data for plotting
            save_path: Path to save plot
        """
        if not self.is_fitted:
            raise ValueError("FeatureSelector must be fitted first")
        
        if X_sample is None:
            # Use the sample used during fitting
            sample_size = min(len(X_sample) if X_sample is not None else 100, 
                            self.shap_values_.shape[0])
            X_display = X_sample.iloc[:sample_size] if X_sample is not None else None
        else:
            X_display = X_sample
        
        # Handle multi-class case
        shap_values_display = (self.shap_values_[1] if isinstance(self.shap_values_, list) 
                             else self.shap_values_)
        
        # Summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values_display, X_display, show=False)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"SHAP summary plot saved to {save_path}")
        
        plt.show()
    
    def plot_shap_waterfall(self, X_sample, instance_idx=0, save_path=None):
        """
        Plot SHAP waterfall for a specific instance
        
        Args:
            X_sample: Sample data
            instance_idx: Index of instance to explain
            save_path: Path to save plot
        """
        if not self.is_fitted:
            raise ValueError("FeatureSelector must be fitted first")
        
        # Get SHAP values for the instance
        instance_shap = self.explainer.shap_values(X_sample.iloc[[instance_idx]])
        
        # Handle multi-class case
        if isinstance(instance_shap, list):
            instance_shap = instance_shap[1]  # Positive class
        
        plt.figure(figsize=(12, 8))
        shap.waterfall_plot(
            shap.Explanation(
                values=instance_shap[0],
                base_values=self.explainer.expected_value[1] if isinstance(self.explainer.expected_value, list) 
                           else self.explainer.expected_value,
                data=X_sample.iloc[instance_idx].values,
                feature_names=X_sample.columns.tolist()
            ),
            show=False
        )
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"SHAP waterfall plot saved to {save_path}")
        
        plt.show()
    
    def _save_feature_importance(self, feature_names):
        """
        Save feature importance results to files
        
        Args:
            feature_names: List of feature names
        """
        # Create results directory
        results_dir = self.config.RESULTS_PATH
        os.makedirs(results_dir, exist_ok=True)
        
        # Save feature importance DataFrame
        importance_df = self.get_feature_importance_df(feature_names)
        importance_path = os.path.join(results_dir, 'feature_importance.csv')
        importance_df.to_csv(importance_path, index=False)
        
        # Save selected features list
        selected_features_path = os.path.join(results_dir, 'selected_features.txt')
        with open(selected_features_path, 'w') as f:
            f.write("Selected Features (SHAP value >= {}):\n".format(self.threshold))
            f.write("=" * 50 + "\n")
            for i, feature in enumerate(self.selected_features_, 1):
                importance = self.feature_importance_[feature_names.get_loc(feature)]
                f.write(f"{i:2d}. {feature:<30} (importance: {importance:.4f})\n")
        
        logger.info(f"Feature importance results saved to {results_dir}")
    
    def get_selected_feature_stats(self):
        """
        Get statistics about selected features
        
        Returns:
            Dictionary with selection statistics
        """
        if not self.is_fitted:
            raise ValueError("FeatureSelector must be fitted first")
        
        total_features = len(self.feature_importance_)
        selected_count = len(self.selected_features_)
        
        stats = {
            'total_features': total_features,
            'selected_features': selected_count,
            'selection_ratio': selected_count / total_features,
            'threshold_used': self.threshold,
            'min_importance': self.feature_importance_.min(),
            'max_importance': self.feature_importance_.max(),
            'mean_importance': self.feature_importance_.mean(),
            'selected_features_list': self.selected_features_,
            'importance_scores': dict(zip(self.selected_features_, 
                                        self.feature_importance_[self.selected_indices_]))
        }
        
        return stats

class MultiMethodFeatureSelector:
    """
    Feature selector using multiple methods for comparison
    """
    
    def __init__(self, config):
        self.config = config
        self.methods = {}
        self.results = {}
    
    def add_method(self, name, selector):
        """Add a feature selection method"""
        self.methods[name] = selector
    
    def fit_all_methods(self, X, y):
        """Fit all feature selection methods"""
        logger.info("Comparing multiple feature selection methods...")
        
        for name, selector in self.methods.items():
            logger.info(f"Fitting {name}...")
            selector.fit(X, y)
            self.results[name] = {
                'selected_features': selector.selected_features_,
                'n_selected': len(selector.selected_features_),
                'selector': selector
            }
        
        return self
    
    def compare_methods(self):
        """Compare results from different methods"""
        comparison_df = pd.DataFrame([
            {
                'method': name,
                'n_selected': result['n_selected'],
                'selected_features': ', '.join(result['selected_features'][:5]) + '...'
            }
            for name, result in self.results.items()
        ])
        
        return comparison_df
    
    def get_consensus_features(self, min_votes=2):
        """Get features selected by multiple methods"""
        if not self.results:
            raise ValueError("Must fit methods first")
        
        # Count votes for each feature
        feature_votes = {}
        for result in self.results.values():
            for feature in result['selected_features']:
                feature_votes[feature] = feature_votes.get(feature, 0) + 1
        
        # Select features with minimum votes
        consensus_features = [
            feature for feature, votes in feature_votes.items()
            if votes >= min_votes
        ]
        
        return consensus_features, feature_votes

if __name__ == "__main__":
    # Test feature selection
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
        n_informative=10,
        n_redundant=5,
        n_classes=2,
        random_state=42
    )
    
    # Convert to DataFrame
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=0.2, random_state=42
    )
    
    # Initialize config and feature selector
    config = Config()
    selector = FeatureSelector(config, threshold=0.05)
    
    # Fit and transform
    print("Fitting feature selector...")
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    
    print(f"Original features: {X_train.shape[1]}")
    print(f"Selected features: {X_train_selected.shape[1]}")
    print(f"Selected feature names: {selector.selected_features_}")
    
    # Get statistics
    stats = selector.get_selected_feature_stats()
    print(f"Selection statistics: {stats}")
    
    # Plot feature importance
    selector.plot_feature_importance(X_train.columns, top_k=15)
