"""
Bidirectional LSTM Model Implementation
Standalone implementation of the BDLSTM component
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    LSTM, Bidirectional, Dense, Dropout, Input, 
    BatchNormalization, Activation
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l1_l2
import logging

logger = logging.getLogger(__name__)

class BidirectionalLSTMModel:
    """
    Bidirectional LSTM model for cardiovascular disease prediction
    
    This model processes tabular data by treating each feature as a timestep
    in a sequence, allowing the LSTM to capture relationships between features.
    """
    
    def __init__(self, input_shape, config):
        """
        Initialize the Bidirectional LSTM model
        
        Args:
            input_shape: Shape of input features (n_features,)
            config: Configuration object with model parameters
        """
        self.input_shape = input_shape
        self.config = config
        self.model = None
        self.history = None
        self.build_model()
    
    def build_model(self):
        """Build the Bidirectional LSTM architecture"""
        
        logger.info("Building Bidirectional LSTM model...")
        
        # Input layer
        inputs = Input(shape=self.input_shape, name='input_features')
        
        # Reshape for LSTM: (batch_size, timesteps, features)
        # For tabular data, we treat each feature as a timestep
        reshaped = tf.expand_dims(inputs, axis=1)  # Add time dimension
        
        # First Bidirectional LSTM layer
        lstm1 = Bidirectional(
            LSTM(
                units=self.config.LSTM_UNITS_1,
                return_sequences=True,
                dropout=self.config.LSTM_DROPOUT,
                recurrent_dropout=self.config.RECURRENT_DROPOUT,
                kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
                name='lstm_1'
            ),
            name='bidirectional_lstm_1'
        )(reshaped)
        
        # Batch normalization
        lstm1_norm = BatchNormalization(name='batch_norm_1')(lstm1)
        
        # Second Bidirectional LSTM layer
        lstm2 = Bidirectional(
            LSTM(
                units=self.config.LSTM_UNITS_2,
                return_sequences=False,
                dropout=self.config.LSTM_DROPOUT,
                recurrent_dropout=self.config.RECURRENT_DROPOUT,
                kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
                name='lstm_2'
            ),
            name='bidirectional_lstm_2'
        )(lstm1_norm)
        
        # Batch normalization
        lstm2_norm = BatchNormalization(name='batch_norm_2')(lstm2)
        
        # First Dense layer
        dense1 = Dense(
            self.config.DENSE_UNITS_1,
            kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
            name='dense_1'
        )(lstm2_norm)
        dense1_activated = Activation('relu', name='dense_1_activation')(dense1)
        dropout1 = Dropout(self.config.DROPOUT_RATE_1, name='dropout_1')(dense1_activated)
        
        # Second Dense layer
        dense2 = Dense(
            self.config.DENSE_UNITS_2,
            kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
            name='dense_2'
        )(dropout1)
        dense2_activated = Activation('relu', name='dense_2_activation')(dense2)
        dropout2 = Dropout(self.config.DROPOUT_RATE_2, name='dropout_2')(dense2_activated)
        
        # Output layer
        outputs = Dense(
            1, 
            activation='sigmoid',
            name='output'
        )(dropout2)
        
        # Create model
        self.model = Model(inputs=inputs, outputs=outputs, name='BDLSTM_CVD_Predictor')
        
        # Compile model
        optimizer = Adam(
            learning_rate=self.config.LEARNING_RATE,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
        
        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc')
            ]
        )
        
        logger.info("Bidirectional LSTM model built successfully")
        logger.info(f"Model parameters: {self.model.count_params():,}")
        self.print_model_summary()
    
    def print_model_summary(self):
        """Print detailed model summary"""
        logger.info("Model Architecture Summary:")
        self.model.summary(print_fn=logger.info)
    
    def fit(self, X_train, y_train, validation_data=None, epochs=100, batch_size=32, verbose=1):
        """
        Train the BDLSTM model
        
        Args:
            X_train: Training features
            y_train: Training labels
            validation_data: Validation data tuple (X_val, y_val)
            epochs: Number of training epochs
            batch_size: Batch size for training
            verbose: Verbosity level
            
        Returns:
            Training history
        """
        logger.info("Starting BDLSTM model training...")
        
        # Prepare callbacks
        callbacks = self._prepare_callbacks()
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose,
            shuffle=True
        )
        
        logger.info("BDLSTM model training completed")
        return self.history
    
    def _prepare_callbacks(self):
        """Prepare training callbacks"""
        callbacks = []
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=self.config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
        
        # Learning rate reduction
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=self.config.REDUCE_LR_FACTOR,
            patience=self.config.REDUCE_LR_PATIENCE,
            min_lr=1e-8,
            verbose=1
        )
        callbacks.append(reduce_lr)
        
        # Model checkpoint (save best model)
        checkpoint = ModelCheckpoint(
            filepath='models/saved_models/best_bdlstm_model.h5',
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
        callbacks.append(checkpoint)
        
        return callbacks
    
    def predict(self, X):
        """
        Make predictions
        
        Args:
            X: Input features
            
        Returns:
            Binary predictions
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        probabilities = self.model.predict(X)
        return (probabilities > 0.5).astype(int).flatten()
    
    def predict_proba(self, X):
        """
        Predict class probabilities
        
        Args:
            X: Input features
            
        Returns:
            Class probabilities
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        pos_proba = self.model.predict(X).flatten()
        neg_proba = 1 - pos_proba
        
        return np.column_stack([neg_proba, pos_proba])
    
    def evaluate(self, X_test, y_test, verbose=0):
        """
        Evaluate model performance
        
        Args:
            X_test: Test features
            y_test: Test labels
            verbose: Verbosity level
            
        Returns:
            Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
        
        return self.model.evaluate(X_test, y_test, verbose=verbose)
    
    def save(self, filepath):
        """
        Save the trained model
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model must be trained before saving")
        
        self.model.save(filepath)
        logger.info(f"BDLSTM model saved to {filepath}")
    
    def load(self, filepath):
        """
        Load a trained model
        
        Args:
            filepath: Path to the saved model
        """
        self.model = tf.keras.models.load_model(filepath)
        logger.info(f"BDLSTM model loaded from {filepath}")
    
    def get_model_config(self):
        """Get model configuration dictionary"""
        return {
            'input_shape': self.input_shape,
            'lstm_units_1': self.config.LSTM_UNITS_1,
            'lstm_units_2': self.config.LSTM_UNITS_2,
            'dense_units_1': self.config.DENSE_UNITS_1,
            'dense_units_2': self.config.DENSE_UNITS_2,
            'dropout_rate_1': self.config.DROPOUT_RATE_1,
            'dropout_rate_2': self.config.DROPOUT_RATE_2,
            'lstm_dropout': self.config.LSTM_DROPOUT,
            'recurrent_dropout': self.config.RECURRENT_DROPOUT,
            'learning_rate': self.config.LEARNING_RATE,
            'total_parameters': self.model.count_params() if self.model else 0
        }
    
    def get_layer_outputs(self, X, layer_names=None):
        """
        Get outputs from intermediate layers for analysis
        
        Args:
            X: Input data
            layer_names: List of layer names to extract outputs from
            
        Returns:
            Dictionary of layer outputs
        """
        if self.model is None:
            raise ValueError("Model must be built before extracting layer outputs")
        
        if layer_names is None:
            layer_names = ['bidirectional_lstm_1', 'bidirectional_lstm_2', 'dense_1', 'dense_2']
        
        # Create intermediate models
        intermediate_outputs = {}
        
        for layer_name in layer_names:
            try:
                layer = self.model.get_layer(layer_name)
                intermediate_model = Model(inputs=self.model.input, outputs=layer.output)
                intermediate_outputs[layer_name] = intermediate_model.predict(X)
            except ValueError:
                logger.warning(f"Layer '{layer_name}' not found in model")
        
        return intermediate_outputs
    
    def plot_training_history(self, save_path=None):
        """
        Plot training history
        
        Args:
            save_path: Path to save the plot
        """
        if self.history is None:
            logger.warning("No training history available")
            return
        
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot training & validation loss
        axes[0, 0].plot(self.history.history['loss'], label='Training Loss')
        axes[0, 0].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot training & validation accuracy
        axes[0, 1].plot(self.history.history['accuracy'], label='Training Accuracy')
        axes[0, 1].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 1].set_title('Model Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Plot training & validation precision
        if 'precision' in self.history.history:
            axes[1, 0].plot(self.history.history['precision'], label='Training Precision')
            axes[1, 0].plot(self.history.history['val_precision'], label='Validation Precision')
            axes[1, 0].set_title('Model Precision')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Plot training & validation recall
        if 'recall' in self.history.history:
            axes[1, 1].plot(self.history.history['recall'], label='Training Recall')
            axes[1, 1].plot(self.history.history['val_recall'], label='Validation Recall')
            axes[1, 1].set_title('Model Recall')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history plot saved to {save_path}")
        
        plt.show()

# Example usage and testing
if __name__ == "__main__":
    # Test the BDLSTM model
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
    input_shape = (X_train.shape[1],)
    
    model = BidirectionalLSTMModel(input_shape, config)
    
    # Train model
    print("Training BDLSTM model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        verbose=1
    )
    
    # Make predictions
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    
    print(f"Predictions shape: {predictions.shape}")
    print(f"Probabilities shape: {probabilities.shape}")
    print(f"Sample predictions: {predictions[:10]}")
    
    # Evaluate model
    test_metrics = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test metrics: {test_metrics}")
    
    # Get model configuration
    model_config = model.get_model_config()
    print(f"Model configuration: {model_config}")
    
    print("BDLSTM model test completed successfully!")
