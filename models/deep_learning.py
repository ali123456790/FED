"""
Deep learning models for IoT device security.
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Any
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input

logger = logging.getLogger(__name__)

class LSTMModel:
    """LSTM model for IoT device security."""
    
    def __init__(self, config: Dict):
        """
        Initialize the LSTM model.
        
        Args:
            config: Model configuration
        """
        self.config = config
        self.hyperparameters = config["hyperparameters"]["lstm"]
        
        # Build the model
        self.model = self._build_model()
        
        logger.info("LSTM model initialized")
    
    def _build_model(self) -> tf.keras.Model:
        """
        Build the LSTM model.
        
        Returns:
            Compiled Keras model
        """
        # Get hyperparameters
        hidden_size = self.hyperparameters["hidden_size"]
        num_layers = self.hyperparameters["num_layers"]
        dropout = self.hyperparameters["dropout"]
        
        # Build the model
        model = Sequential()
        
        # Reshape the input for LSTM
        model.add(tf.keras.layers.Reshape((-1, 1), input_shape=(None,)))
        
        # Add LSTM layers
        for i in range(num_layers):
            return_sequences = i < num_layers - 1
            if i == 0:
                model.add(LSTM(hidden_size, return_sequences=return_sequences))
            else:
                model.add(LSTM(hidden_size, return_sequences=return_sequences))
            model.add(Dropout(dropout))
        
        # Add output layer
        model.add(Dense(1, activation='sigmoid'))
        
        # Compile the model
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """
        Train the model.
        
        Args:
            X: Training features
            y: Training labels
            **kwargs: Additional arguments
        """
        # Get training parameters
        epochs = kwargs.get('epochs', 5)
        batch_size = kwargs.get('batch_size', 32)
        verbose = kwargs.get('verbose', 1)
        
        # Train the model
        self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose
        )
        
        logger.info("LSTM model trained")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features to predict
            
        Returns:
            Predicted labels
        """
        return (self.model.predict(X) > 0.5).astype(int)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Tuple[float, float]:
        """
        Evaluate the model.
        
        Args:
            X: Test features
            y: Test labels
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (loss, accuracy)
        """
        verbose = kwargs.get('verbose', 1)
        
        # Evaluate the model
        loss, accuracy = self.model.evaluate(X, y, verbose=verbose)
        
        logger.info(f"LSTM evaluation: loss={loss:.4f}, accuracy={accuracy:.4f}")
        
        return loss, accuracy
    
    def get_weights(self) -> List[np.ndarray]:
        """
        Get model weights.
        
        Returns:
            List of model weights
        """
        return self.model.get_weights()
    
    def set_weights(self, weights: List[np.ndarray]) -> None:
        """
        Set model weights.
        
        Args:
            weights: List of model weights
        """
        self.model.set_weights(weights)

class BiLSTMModel:
    """Bidirectional LSTM model for IoT device security."""
    
    def __init__(self, config: Dict):
        """
        Initialize the Bidirectional LSTM model.
        
        Args:
            config: Model configuration
        """
        self.config = config
        self.hyperparameters = config["hyperparameters"]["bilstm"]
        
        # Build the model
        self.model = self._build_model()
        
        logger.info("Bidirectional LSTM model initialized")
    
    def _build_model(self) -> tf.keras.Model:
        """
        Build the Bidirectional LSTM model.
        
        Returns:
            Compiled Keras model
        """
        # Get hyperparameters
        hidden_size = self.hyperparameters["hidden_size"]
        num_layers = self.hyperparameters["num_layers"]
        dropout = self.hyperparameters["dropout"]
        
        # Build the model
        model = Sequential()
        
        # Reshape the input for LSTM
        model.add(tf.keras.layers.Reshape((-1, 1), input_shape=(None,)))
        
        # Add Bidirectional LSTM layers
        for i in range(num_layers):
            return_sequences = i < num_layers - 1
            if i == 0:
                model.add(Bidirectional(LSTM(hidden_size, return_sequences=return_sequences)))
            else:
                model.add(Bidirectional(LSTM(hidden_size, return_sequences=return_sequences)))
            model.add(Dropout(dropout))
        
        # Add output layer
        model.add(Dense(1, activation='sigmoid'))
        
        # Compile the model
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """
        Train the model.
        
        Args:
            X: Training features
            y: Training labels
            **kwargs: Additional arguments
        """
        # Get training parameters
        epochs = kwargs.get('epochs', 5)
        batch_size = kwargs.get('batch_size', 32)
        verbose = kwargs.get('verbose', 1)
        
        # Train the model
        self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose
        )
        
        logger.info("Bidirectional LSTM model trained")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features to predict
            
        Returns:
            Predicted labels
        """
        return (self.model.predict(X) > 0.5).astype(int)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Tuple[float, float]:
        """
        Evaluate the model.
        
        Args:
            X: Test features
            y: Test labels
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (loss, accuracy)
        """
        verbose = kwargs.get('verbose', 1)
        
        # Evaluate the model
        loss, accuracy = self.model.evaluate(X, y, verbose=verbose)
        
        logger.info(f"Bidirectional LSTM evaluation: loss={loss:.4f}, accuracy={accuracy:.4f}")
        
        return loss, accuracy
    
    def get_weights(self) -> List[np.ndarray]:
        """
        Get model weights.
        
        Returns:
            List of model weights
        """
        return self.model.get_weights()
    
    def set_weights(self, weights: List[np.ndarray]) -> None:
        """
        Set model weights.
        
        Args:
            weights: List of model weights
        """
        self.model.set_weights(weights)

