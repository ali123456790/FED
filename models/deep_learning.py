"""
Deep learning models for IoT device security.

This module implements TensorFlow/Keras based models such as LSTM,
BiLSTM, CNN, and MLP for anomaly detection in IoT devices.
"""

import logging
import numpy as np
import os
import time
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LSTM, Bidirectional, Conv1D, 
    MaxPooling1D, Flatten, BatchNormalization, 
    GlobalAveragePooling1D, Reshape, TimeDistributed
)
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, TensorBoard, 
    ReduceLROnPlateau, LearningRateScheduler
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import AUC, Precision, Recall
from tensorflow.keras.regularizers import l1_l2
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)

logger = logging.getLogger(__name__)

class BaseDeepLearningModel:
    """Base class for deep learning models."""
    
    def __init__(self, config: Dict):
        """
        Initialize the base deep learning model.
        
        Args:
            config: Model configuration
        """
        self.config = config
        self.model = None
        self.model_path = None
        self.history = None
        self.is_fitted = False
        
        # Set up hardware acceleration if available
        if tf.config.list_physical_devices('GPU'):
            logger.info("Using GPU for training")
        else:
            logger.info("No GPU available, using CPU for training")
        
    def build_model(self, input_shape: Tuple[int, ...], num_classes: int) -> None:
        """
        Build the model architecture.
        
        Args:
            input_shape: Shape of the input data
            num_classes: Number of classes
        """
        raise NotImplementedError("Subclasses must implement build_model method")
    
    def compile_model(self, learning_rate: float = 0.001, metrics: List[str] = None) -> None:
        """
        Compile the model.
        
        Args:
            learning_rate: Learning rate for the optimizer
            metrics: List of metrics to track
        """
        if self.model is None:
            raise ValueError("Model must be built before compilation")
        
        # Default metrics if none provided
        if metrics is None:
            metrics = ['accuracy', AUC(), Precision(), Recall()]
        
        # Get optimizer from config or use default
        optimizer_name = self.config.get("optimizer", "adam").lower()
        
        if optimizer_name == "adam":
            optimizer = Adam(learning_rate=learning_rate)
        elif optimizer_name == "rmsprop":
            optimizer = RMSprop(learning_rate=learning_rate)
        else:
            logger.warning(f"Unknown optimizer: {optimizer_name}, using Adam")
            optimizer = Adam(learning_rate=learning_rate)
        
        # Compile the model
        num_classes = self.config.get("default_num_classes", 2)
        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy' if num_classes == 2 else 'categorical_crossentropy',
            metrics=metrics
        )
        
        logger.info(f"Model compiled with {optimizer_name} optimizer and learning rate {learning_rate}")
    
    def fit(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        **kwargs
    ) -> 'BaseDeepLearningModel':
        """
        Train the model.
        
        Args:
            X: Training features
            y: Training labels
            **kwargs: Additional arguments including:
                - validation_data: Tuple of (X_val, y_val)
                - batch_size: Batch size for training
                - epochs: Number of epochs to train
                - callbacks: List of callbacks
                - verbose: Verbosity level
                
        Returns:
            Self for method chaining
        """
        if self.model is None:
            # Infer input shape and number of classes
            input_shape = X.shape[1:]
            if len(input_shape) == 0:  # 1D array
                input_shape = (X.shape[1], 1)
                X = X.reshape((-1, X.shape[1], 1))
                
            # Determine number of classes
            if len(y.shape) == 1 or y.shape[1] == 1:
                num_classes = len(np.unique(y))
                # Convert to categorical if more than 2 classes
                if num_classes > 2:
                    y = to_categorical(y)
            else:
                num_classes = y.shape[1]
                
            # Build and compile the model
            self.build_model(input_shape, num_classes)
            self.compile_model(
                learning_rate=kwargs.get('learning_rate', self.config.get('learning_rate', 0.001)),
                metrics=kwargs.get('metrics', None)
            )
            
        # Extract training parameters
        validation_data = kwargs.get('validation_data', None)
        batch_size = kwargs.get('batch_size', self.config.get('batch_size', 32))
        epochs = kwargs.get('epochs', self.config.get('local_epochs', 5))
        callbacks = kwargs.get('callbacks', self._get_default_callbacks())
        verbose = kwargs.get('verbose', 1)
        
        # Check if validation data should be reshaped
        if validation_data is not None and len(validation_data) == 2:
            X_val, y_val = validation_data
            if len(X_val.shape) < len(X.shape):
                X_val = X_val.reshape((-1, X_val.shape[1], 1))
                validation_data = (X_val, y_val)
        
        # Train the model
        start_time = time.time()
        
        self.history = self.model.fit(
            X, y,
            validation_data=validation_data,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            verbose=verbose
        )
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        self.is_fitted = True
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features to predict
            
        Returns:
            Predicted labels
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model must be fitted before making predictions")
        
        # Reshape input if needed
        if len(X.shape) < len(self.model.input_shape):
            X = X.reshape((-1, X.shape[1], 1))
        
        # Get raw predictions
        y_pred_proba = self.model.predict(X)
        
        # Convert to binary predictions for binary classification
        if y_pred_proba.shape[1] == 1:
            y_pred = (y_pred_proba > 0.5).astype(int).reshape(-1)
        else:
            # Multi-class: use argmax
            y_pred = np.argmax(y_pred_proba, axis=1)
        
        return y_pred
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Features to predict
            
        Returns:
            Class probabilities
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model must be fitted before predicting probabilities")
        
        # Reshape input if needed
        if len(X.shape) < len(self.model.input_shape):
            X = X.reshape((-1, X.shape[1], 1))
        
        # Get probability predictions
        y_pred_proba = self.model.predict(X)
        
        # For binary classification with single output
        if y_pred_proba.shape[1] == 1:
            # Convert to 2-column format ([prob_0, prob_1])
            y_pred_proba_2col = np.zeros((len(y_pred_proba), 2))
            y_pred_proba_2col[:, 1] = y_pred_proba.flatten()
            y_pred_proba_2col[:, 0] = 1 - y_pred_proba.flatten()
            return y_pred_proba_2col
        
        # Multi-class: already in correct format
        return y_pred_proba
    
    def evaluate(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        **kwargs
    ) -> Dict[str, float]:
        """
        Evaluate the model.
        
        Args:
            X: Test features
            y: Test labels
            **kwargs: Additional arguments
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model must be fitted before evaluation")
        
        # Reshape input if needed
        if len(X.shape) < len(self.model.input_shape):
            X = X.reshape((-1, X.shape[1], 1))
        
        # Get raw model evaluation
        verbose = kwargs.get('verbose', 0)
        model_metrics = self.model.evaluate(X, y, verbose=verbose)
        metrics_dict = {name: float(value) for name, value in zip(self.model.metrics_names, model_metrics)}
        
        # Get predictions
        y_pred = self.predict(X)
        
        # For multi-class, convert one-hot encoded labels to class indices
        y_true = y
        if len(y.shape) > 1 and y.shape[1] > 1:
            y_true = np.argmax(y, axis=1)
        
        # Calculate sklearn metrics - ensure they're all floats
        sk_metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, average='weighted')),
            "recall": float(recall_score(y_true, y_pred, average='weighted')),
            "f1": float(f1_score(y_true, y_pred, average='weighted'))
        }
        
        # Merge metrics
        metrics_dict.update(sk_metrics)
        
        # Calculate AUC for binary classification
        if len(np.unique(y_true)) == 2:
            try:
                y_pred_proba = self.predict_proba(X)
                metrics_dict["auc"] = float(roc_auc_score(y_true, y_pred_proba[:, 1]))
            except Exception as e:
                logger.warning(f"Could not calculate AUC: {e}")
        
        # Convert confusion matrix to string representation
        cm = confusion_matrix(y_true, y_pred)
        metrics_dict["confusion_matrix"] = str(cm)
        
        # Log classification report
        if kwargs.get('log_report', True):
            report = classification_report(y_true, y_pred)
            logger.info(f"Classification report:\n{report}")
        
        return metrics_dict
    
    def save(self, path: str) -> None:
        """
        Save the model to disk.
        
        Args:
            path: Path to save the model
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model must be fitted before saving")
        
        # Make sure the directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save the model
        self.model.save(path)
        self.model_path = path
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str) -> 'BaseDeepLearningModel':
        """
        Load the model from disk.
        
        Args:
            path: Path to load the model from
            
        Returns:
            Self for method chaining
        """
        # Check if the file exists
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        # Load the model
        self.model = load_model(path)
        self.model_path = path
        self.is_fitted = True
        logger.info(f"Model loaded from {path}")
        
        return self
    
    def get_weights(self) -> List[np.ndarray]:
        """
        Get model weights for federated learning.
        
        Returns:
            List of model weights as numpy arrays
        """
        if self.model is None:
            logger.warning("Underlying Keras model is not built; attempting to build with default settings.")

            # Retrieve default input shape and number of classes from config
            input_shape = self.config.get("default_input_shape")
            if input_shape is None:
                raise RuntimeError("Model is not initialized and no default_input_shape provided in config.")

            num_classes = self.config.get("default_num_classes", 2)

            # Build and compile the model
            self.build_model(input_shape, num_classes)
            self.compile_model(learning_rate=self.config.get("learning_rate", 0.001))

        return self.model.get_weights()
    
    def set_weights(self, weights: List[np.ndarray]) -> None:
        """
        Set model weights for federated learning.
        
        Args:
            weights: List of model weights
        """
        if self.model is None:
            raise RuntimeError("Model must be initialized before setting weights")
        
        try:
            self.model.set_weights(weights)
            self.is_fitted = True
        except ValueError as e:
            # More detailed error message to help with debugging
            weight_shapes = [w.shape for w in weights]
            model_shapes = [w.shape for w in self.model.get_weights()]
            
            logger.error(f"Failed to set weights: {e}")
            logger.error(f"Received weight shapes: {weight_shapes}")
            logger.error(f"Model expected shapes: {model_shapes}")
            raise ValueError(f"Weight shapes mismatch. Received: {weight_shapes}, Expected: {model_shapes}") from e
    
    def _get_default_callbacks(self) -> List[tf.keras.callbacks.Callback]:
        """
        Get default training callbacks.
        
        Returns:
            List of callbacks
        """
        callbacks = []
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        callbacks.append(early_stopping)
        
        # Reduce learning rate on plateau
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
        callbacks.append(reduce_lr)
        
        return callbacks

class LSTMModel(BaseDeepLearningModel):
    """LSTM model for IoT device security."""
    
    def __init__(self, config: Dict):
        """
        Initialize the LSTM model.
        
        Args:
            config: Model configuration
        """
        super().__init__(config)
        
        # Get hyperparameters
        self.hyperparameters = config["hyperparameters"]["lstm"]
        logger.info("LSTM model initialized")
    
    def build_model(self, input_shape: Tuple[int, ...], num_classes: int) -> None:
        """
        Build the LSTM model architecture.
        
        Args:
            input_shape: Shape of the input data
            num_classes: Number of classes
        """
        # Get hyperparameters
        hidden_size = self.hyperparameters.get("hidden_size", 128)
        num_layers = self.hyperparameters.get("num_layers", 2)
        dropout = self.hyperparameters.get("dropout", 0.2)
        recurrent_dropout = self.hyperparameters.get("recurrent_dropout", 0.0)
        l1_reg = self.hyperparameters.get("l1_reg", 0.0)
        l2_reg = self.hyperparameters.get("l2_reg", 0.0)
        
        # Input shape should be (timesteps, features)
        # For a standard feature vector, reshape to (features, 1)
        if len(input_shape) == 1:
            input_shape = (input_shape[0], 1)
        
        logger.info(f"Building LSTM model with input shape {input_shape} and {num_classes} classes")
        
        # Build model
        model = Sequential()
        
        # Add LSTM layers
        for i in range(num_layers):
            return_sequences = i < num_layers - 1
            
            # First layer needs input_shape
            if i == 0:
                model.add(LSTM(
                    hidden_size,
                    input_shape=input_shape,
                    return_sequences=return_sequences,
                    dropout=dropout,
                    recurrent_dropout=recurrent_dropout,
                    kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)
                ))
            else:
                model.add(LSTM(
                    hidden_size,
                    return_sequences=return_sequences,
                    dropout=dropout,
                    recurrent_dropout=recurrent_dropout,
                    kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)
                ))
            
            # Add batch normalization
            if self.hyperparameters.get("batch_normalization", False):
                model.add(BatchNormalization())
        
        # Add fully connected layers
        fc_layers = self.hyperparameters.get("fc_layers", [])
        for fc_size in fc_layers:
            model.add(Dense(
                fc_size,
                activation='relu',
                kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)
            ))
            model.add(Dropout(dropout))
        
        # Add output layer
        if num_classes == 2:
            model.add(Dense(1, activation='sigmoid'))
        else:
            model.add(Dense(num_classes, activation='softmax'))
        
        self.model = model
        logger.info(f"LSTM model built with {model.count_params()} parameters")
        
        # Log model summary
        model.summary(print_fn=logger.info)

class BiLSTMModel(BaseDeepLearningModel):
    """Bidirectional LSTM model for IoT device security."""
    
    def __init__(self, config: Dict):
        """
        Initialize the BiLSTM model.
        
        Args:
            config: Model configuration
        """
        super().__init__(config)
        
        # Get hyperparameters
        self.hyperparameters = config["hyperparameters"]["bilstm"]
        logger.info("Bidirectional LSTM model initialized")
    
    def build_model(self, input_shape: Tuple[int, ...], num_classes: int) -> None:
        """
        Build the Bidirectional LSTM model architecture.
        
        Args:
            input_shape: Shape of the input data
            num_classes: Number of classes
        """
        # Get hyperparameters
        hidden_size = self.hyperparameters.get("hidden_size", 128)
        num_layers = self.hyperparameters.get("num_layers", 2)
        dropout = self.hyperparameters.get("dropout", 0.2)
        recurrent_dropout = self.hyperparameters.get("recurrent_dropout", 0.0)
        l1_reg = self.hyperparameters.get("l1_reg", 0.0)
        l2_reg = self.hyperparameters.get("l2_reg", 0.0)
        
        # Input shape should be (timesteps, features)
        # For a standard feature vector, reshape to (features, 1)
        if len(input_shape) == 1:
            input_shape = (input_shape[0], 1)
        
        logger.info(f"Building BiLSTM model with input shape {input_shape} and {num_classes} classes")
        
        # Build model
        model = Sequential()
        
        # Add Bidirectional LSTM layers
        for i in range(num_layers):
            return_sequences = i < num_layers - 1
            
            # First layer needs input_shape
            if i == 0:
                model.add(Bidirectional(LSTM(
                    hidden_size,
                    return_sequences=return_sequences,
                    dropout=dropout,
                    recurrent_dropout=recurrent_dropout,
                    kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)
                ), input_shape=input_shape))
            else:
                model.add(Bidirectional(LSTM(
                    hidden_size,
                    return_sequences=return_sequences,
                    dropout=dropout,
                    recurrent_dropout=recurrent_dropout,
                    kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)
                )))
            
            # Add batch normalization
            if self.hyperparameters.get("batch_normalization", False):
                model.add(BatchNormalization())
        
        # Add fully connected layers
        fc_layers = self.hyperparameters.get("fc_layers", [])
        for fc_size in fc_layers:
            model.add(Dense(
                fc_size,
                activation='relu',
                kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)
            ))
            model.add(Dropout(dropout))
        
        # Add output layer
        if num_classes == 2:
            model.add(Dense(1, activation='sigmoid'))
        else:
            model.add(Dense(num_classes, activation='softmax'))
        
        self.model = model
        logger.info(f"BiLSTM model built with {model.count_params()} parameters")
        
        # Log model summary
        model.summary(print_fn=logger.info)

class CNNModel(BaseDeepLearningModel):
    """Convolutional Neural Network model for IoT device security."""
    
    def __init__(self, config: Dict):
        """
        Initialize the CNN model.
        
        Args:
            config: Model configuration
        """
        super().__init__(config)
        
        # Get hyperparameters
        self.hyperparameters = config["hyperparameters"]["cnn"]
        logger.info("CNN model initialized")
    
    def build_model(self, input_shape: Tuple[int, ...], num_classes: int) -> None:
        """
        Build the CNN model architecture.
        
        Args:
            input_shape: Shape of the input data
            num_classes: Number of classes
        """
        # Get hyperparameters
        filters = self.hyperparameters.get("filters", [64, 128, 128])
        kernel_sizes = self.hyperparameters.get("kernel_sizes", [3, 3, 3])
        pool_sizes = self.hyperparameters.get("pool_sizes", [2, 2, 2])
        dropout = self.hyperparameters.get("dropout", 0.5)
        l1_reg = self.hyperparameters.get("l1_reg", 0.0)
        l2_reg = self.hyperparameters.get("l2_reg", 0.0)
        
        # Input shape should be (timesteps, features)
        # For a standard feature vector, reshape to (features, 1)
        if len(input_shape) == 1:
            input_shape = (input_shape[0], 1)
        
        logger.info(f"Building CNN model with input shape {input_shape} and {num_classes} classes")
        
        # Build model
        model = Sequential()
        
        # First convolutional layer
        model.add(Conv1D(
            filters=filters[0],
            kernel_size=kernel_sizes[0],
            activation='relu',
            input_shape=input_shape,
            padding='same',
            kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)
        ))
        model.add(MaxPooling1D(pool_size=pool_sizes[0]))
        model.add(Dropout(dropout))
        
        # Add additional convolutional layers
        for i in range(1, len(filters)):
            model.add(Conv1D(
                filters=filters[i],
                kernel_size=kernel_sizes[i if i < len(kernel_sizes) else -1],
                activation='relu',
                padding='same',
                kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)
            ))
            model.add(MaxPooling1D(pool_size=pool_sizes[i if i < len(pool_sizes) else -1]))
            model.add(Dropout(dropout))
        
        # Add global pooling
        if self.hyperparameters.get("global_pooling", True):
            model.add(GlobalAveragePooling1D())
        else:
            model.add(Flatten())
        
        # Add fully connected layers
        fc_layers = self.hyperparameters.get("fc_layers", [128])
        for fc_size in fc_layers:
            model.add(Dense(
                fc_size,
                activation='relu',
                kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)
            ))
            model.add(Dropout(dropout))
        
        # Add output layer
        if num_classes == 2:
            model.add(Dense(1, activation='sigmoid'))
        else:
            model.add(Dense(num_classes, activation='softmax'))
        
        self.model = model
        logger.info(f"CNN model built with {model.count_params()} parameters")
        
        # Log model summary
        model.summary(print_fn=logger.info)

class MLPModel(BaseDeepLearningModel):
    """Multi-Layer Perceptron model for IoT device security."""
    
    def __init__(self, config: Dict):
        """
        Initialize the MLP model.
        
        Args:
            config: Model configuration
        """
        super().__init__(config)
        
        # Get hyperparameters
        self.hyperparameters = config["hyperparameters"]["mlp"]
        logger.info("MLP model initialized")
    
    def build_model(self, input_shape: Tuple[int, ...], num_classes: int) -> None:
        """
        Build the MLP model architecture.
        
        Args:
            input_shape: Shape of the input data
            num_classes: Number of classes
        """
        # Get hyperparameters
        hidden_layers = self.hyperparameters.get("hidden_layers", [128, 64])
        dropout = self.hyperparameters.get("dropout", 0.5)
        activation = self.hyperparameters.get("activation", "relu")
        l1_reg = self.hyperparameters.get("l1_reg", 0.0)
        l2_reg = self.hyperparameters.get("l2_reg", 0.0)
        batch_norm = self.hyperparameters.get("batch_normalization", False)
        
        # For MLP, input shape should be flattened
        if len(input_shape) > 1:
            flattened_dim = np.prod(input_shape)
            input_shape = (flattened_dim,)
        
        logger.info(f"Building MLP model with input shape {input_shape} and {num_classes} classes")
        
        # Build model
        model = Sequential()
        
        # Add input layer with flattening if needed
        model.add(Dense(
            hidden_layers[0],
            input_shape=input_shape,
            activation=activation,
            kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)
        ))
        model.add(Dropout(dropout))
        
        # Add batch normalization
        if batch_norm:
            model.add(BatchNormalization())
        
        # Add hidden layers
        for units in hidden_layers[1:]:
            model.add(Dense(
                units,
                activation=activation,
                kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)
            ))
            model.add(Dropout(dropout))
            
            # Add batch normalization
            if batch_norm:
                model.add(BatchNormalization())
        
        # Add output layer
        if num_classes == 2:
            model.add(Dense(1, activation='sigmoid'))
        else:
            model.add(Dense(num_classes, activation='softmax'))
        
        self.model = model
        logger.info(f"MLP model built with {model.count_params()} parameters")
        
        # Log model summary
        model.summary(print_fn=logger.info)

def create_callbacks(
    model_name: str,
    checkpoint_dir: str = "./checkpoints",
    tensorboard_dir: str = "./logs",
    patience: int = 10,
    monitor: str = "val_loss",
    mode: str = "min",
    verbose: int = 1
) -> List[tf.keras.callbacks.Callback]:
    """
    Create training callbacks.
    
    Args:
        model_name: Name of the model
        checkpoint_dir: Directory to save checkpoints
        tensorboard_dir: Directory to save TensorBoard logs
        patience: Patience for early stopping
        monitor: Metric to monitor
        mode: Mode for monitoring ('min' or 'max')
        verbose: Verbosity level
        
    Returns:
        List of callbacks
    """
    callbacks = []
    
    # Create directories
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    
    # Model checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_best.h5")
    checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor=monitor,
        verbose=verbose,
        save_best_only=True,
        mode=mode
    )
    callbacks.append(checkpoint)
    
    # Early stopping
    early_stopping = EarlyStopping(
        monitor=monitor,
        patience=patience,
        verbose=verbose,
        restore_best_weights=True,
        mode=mode
    )
    callbacks.append(early_stopping)
    
    # TensorBoard
    tensorboard = TensorBoard(
        log_dir=os.path.join(tensorboard_dir, model_name),
        histogram_freq=1,
        write_graph=True,
        update_freq='epoch'
    )
    callbacks.append(tensorboard)
    
    # Reduce learning rate on plateau
    reduce_lr = ReduceLROnPlateau(
        monitor=monitor,
        factor=0.5,
        patience=patience // 2,
        verbose=verbose,
        min_lr=1e-6,
        mode=mode
    )
    callbacks.append(reduce_lr)
    
    return callbacks

def preprocess_data_for_deep_learning(
    X_train: np.ndarray, 
    X_test: np.ndarray, 
    y_train: np.ndarray, 
    y_test: np.ndarray,
    model_type: str = "lstm",
    num_classes: int = 2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Preprocess data for deep learning models.
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        model_type: Type of model ('lstm', 'bilstm', 'cnn', 'mlp')
        num_classes: Number of classes
        
    Returns:
        Tuple of (X_train_processed, X_test_processed, y_train_processed, y_test_processed)
    """
    # Reshape input data based on model type
    if model_type in ['lstm', 'bilstm', 'cnn']:
        # Check if reshaping is needed
        if len(X_train.shape) == 2:
            # Reshape to (samples, timesteps, features)
            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    elif model_type == 'mlp':
        # Flatten input for MLP
        if len(X_train.shape) > 2:
            X_train = X_train.reshape((X_train.shape[0], -1))
            X_test = X_test.reshape((X_test.shape[0], -1))
    
    # Convert labels to categorical for multi-class classification
    if num_classes > 2:
        if len(y_train.shape) == 1 or y_train.shape[1] == 1:
            y_train = to_categorical(y_train, num_classes=num_classes)
            y_test = to_categorical(y_test, num_classes=num_classes)
    else:
        # For binary classification, ensure labels are in the right shape
        if len(y_train.shape) > 1 and y_train.shape[1] > 1:
            y_train = np.argmax(y_train, axis=1)
            y_test = np.argmax(y_test, axis=1)
    
    return X_train, X_test, y_train, y_test