"""
Data Preprocessing Module for Cardiovascular Disease Prediction

This module handles data cleaning, preprocessing, and preparation
for the deep learning models as described in the research paper.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import logging

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Data preprocessing class for cardiovascular disease prediction
    
    Handles:
    - Missing value imputation
    - Feature encoding
    - Data normalization
    - Train-test splitting
    """
    
    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputers = {}
        self.is_fitted = False
        
    def handle_missing_values(self, df):
        """
        Handle missing values in the dataset
        Special handling for cholesterol values (initially recorded as 0)
        """
        df_processed = df.copy()
        
        # Handle cholesterol values recorded as 0
        if 'chol' in df_processed.columns:
            # Separate records by heart disease status
            has_disease = df_processed['target'] == 1
            no_disease = df_processed['target'] == 0
            
            # Calculate mean cholesterol for each group (excluding 0 values)
            disease_chol_mean = df_processed[has_disease & (df_processed['chol'] > 0)]['chol'].mean()
            no_disease_chol_mean = df_processed[no_disease & (df_processed['chol'] > 0)]['chol'].mean()
            
            # Replace 0 values with group means
            df_processed.loc[has_disease & (df_processed['chol'] == 0), 'chol'] = disease_chol_mean
            df_processed.loc[no_disease & (df_processed['chol'] == 0), 'chol'] = no_disease_chol_mean
            
            logger.info(f"Replaced cholesterol 0 values - Disease group mean: {disease_chol_mean:.2f}, "
                       f"No disease group mean: {no_disease_chol_mean:.2f}")
        
        # Handle other missing values
        numeric_columns = df_processed.select_dtypes(include=[np.number]).columns
        categorical_columns = df_processed.select_dtypes(include=['object']).columns
        
        # Impute numeric columns with median
        for col in numeric_columns:
            if df_processed[col].isnull().sum() > 0:
                if col not in self.imputers:
                    self.imputers[col] = SimpleImputer(strategy='median')
                    df_processed[col] = self.imputers[col].fit_transform(
                        df_processed[col].values.reshape(-1, 1)
                    ).flatten()
                else:
                    df_processed[col] = self.imputers[col].transform(
                        df_processed[col].values.reshape(-1, 1)
                    ).flatten()
        
        # Impute categorical columns with mode
        for col in categorical_columns:
            if df_processed[col].isnull().sum() > 0:
                if col not in self.imputers:
                    self.imputers[col] = SimpleImputer(strategy='most_frequent')
                    df_processed[col] = self.imputers[col].fit_transform(
                        df_processed[col].values.reshape(-1, 1)
                    ).flatten()
                else:
                    df_processed[col] = self.imputers[col].transform(
                        df_processed[col].values.reshape(-1, 1)
                    ).flatten()
        
        return df_processed
    
    def encode_categorical_features(self, df):
        """
        Encode categorical features using label encoding
        """
        df_encoded = df.copy()
        categorical_columns = df_encoded.select_dtypes(include=['object']).columns
        
        for col in categorical_columns:
            if col not in ['target']:  # Don't encode target variable yet
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df_encoded[col] = self.label_encoders[col].fit_transform(df_encoded[col])
                else:
                    df_encoded[col] = self.label_encoders[col].transform(df_encoded[col])
        
        return df_encoded
    
    def create_interaction_terms(self, df):
        """
        Create interaction terms between important features
        As mentioned in the paper for enhanced prediction
        """
        df_with_interactions = df.copy()
        
        # Define important feature pairs for interaction
        interaction_pairs = [
            ('age', 'chol'),
            ('trestbps', 'chol'),
            ('age', 'thalach'),
            ('ca', 'thal'),
            ('exang', 'oldpeak')
        ]
        
        for feature1, feature2 in interaction_pairs:
            if feature1 in df_with_interactions.columns and feature2 in df_with_interactions.columns:
                interaction_name = f"{feature1}_{feature2}_interaction"
                df_with_interactions[interaction_name] = (
                    df_with_interactions[feature1] * df_with_interactions[feature2]
                )
        
        logger.info(f"Created {len(interaction_pairs)} interaction terms")
        return df_with_interactions
    
    def normalize_features(self, X):
        """
        Normalize features using StandardScaler
        """
        if not self.is_fitted:
            X_normalized = self.scaler.fit_transform(X)
            self.is_fitted = True
        else:
            X_normalized = self.scaler.transform(X)
        
        return pd.DataFrame(X_normalized, columns=X.columns, index=X.index)
    
    def prepare_target_variable(self, y):
        """
        Prepare target variable for binary classification
        """
        # Ensure target is binary (0, 1)
        y_prepared = y.copy()
        if y_prepared.dtype == 'object':
            if 'target' not in self.label_encoders:
                self.label_encoders['target'] = LabelEncoder()
                y_prepared = self.label_encoders['target'].fit_transform(y_prepared)
            else:
                y_prepared = self.label_encoders['target'].transform(y_prepared)
        
        return pd.Series(y_prepared, index=y.index)
    
    def fit_transform(self, df):
        """
        Fit the preprocessor and transform the data
        
        Args:
            df: Input dataframe
            
        Returns:
            X: Processed features
            y: Processed target variable
        """
        logger.info("Starting data preprocessing...")
        
        # Separate features and target
        if 'target' in df.columns:
            X = df.drop('target', axis=1)
            y = df['target']
        else:
            # Assume last column is target
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]
        
        # Handle missing values
        df_clean = self.handle_missing_values(df)
        X_clean = df_clean.drop('target', axis=1) if 'target' in df_clean.columns else df_clean.iloc[:, :-1]
        y_clean = df_clean['target'] if 'target' in df_clean.columns else df_clean.iloc[:, -1]
        
        # Encode categorical features
        X_encoded = self.encode_categorical_features(X_clean)
        
        # Create interaction terms
        X_interactions = self.create_interaction_terms(X_encoded)
        
        # Normalize features
        X_normalized = self.normalize_features(X_interactions)
        
        # Prepare target variable
        y_prepared = self.prepare_target_variable(y_clean)
        
        logger.info(f"Preprocessing completed - Features: {X_normalized.shape}, Target: {y_prepared.shape}")
        logger.info(f"Feature columns: {list(X_normalized.columns)}")
        
        return X_normalized, y_prepared
    
    def transform(self, df):
        """
        Transform new data using fitted preprocessor
        
        Args:
            df: Input dataframe
            
        Returns:
            X: Processed features
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        # Handle missing values
        df_clean = self.handle_missing_values(df)
        
        # Encode categorical features
        X_encoded = self.encode_categorical_features(df_clean)
        
        # Create interaction terms
        X_interactions = self.create_interaction_terms(X_encoded)
        
        # Normalize features
        X_normalized = self.normalize_features(X_interactions)
        
        return X_normalized
    
    def train_test_split(self, X, y, test_size=0.35, random_state=42):
        """
        Split data into training and testing sets
        
        Args:
            X: Features
            y: Target variable
            test_size: Proportion of test set
            random_state: Random seed
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        return train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state,
            stratify=y  # Maintain class balance
        )
    
    def get_preprocessing_stats(self, X, y):
        """
        Get statistics about the preprocessed data
        """
        stats = {
            'n_samples': len(X),
            'n_features': X.shape[1],
            'class_distribution': y.value_counts().to_dict(),
            'feature_statistics': X.describe().to_dict(),
            'missing_values': X.isnull().sum().to_dict()
        }
        
        return stats

def load_cardiovascular_dataset(file_path):
    """
    Load the cardiovascular disease dataset
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame with the loaded data
    """
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Dataset loaded: {df.shape}")
        
        # Display basic information
        logger.info(f"Columns: {list(df.columns)}")
        logger.info(f"Data types:\n{df.dtypes}")
        logger.info(f"Missing values:\n{df.isnull().sum()}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise

if __name__ == "__main__":
    # Test the preprocessor
    from utils.config import Config
    
    config = Config()
    
    # Load sample data (if available)
    try:
        df = load_cardiovascular_dataset('data/raw/cardiovascular_disease_dataset.csv')
        
        # Initialize preprocessor
        preprocessor = DataPreprocessor(config)
        
        # Preprocess data
        X, y = preprocessor.fit_transform(df)
        
        # Get statistics
        stats = preprocessor.get_preprocessing_stats(X, y)
        print("Preprocessing Statistics:")
        print(f"Samples: {stats['n_samples']}")
        print(f"Features: {stats['n_features']}")
        print(f"Class distribution: {stats['class_distribution']}")
        
    except FileNotFoundError:
        print("Dataset not found. Please download the dataset first.")
    except Exception as e:
        print(f"Error: {str(e)}")
