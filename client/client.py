"""
Flower client implementation for federated learning.
"""

import flwr as fl
import logging
import os
import numpy as np
import tensorflow as tf
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import re
from flwr.client import NumPyClient, start_client

from .device_manager import DeviceManager
from .utils import load_data, preprocess_data, save_local_model
from models.model_factory import create_model
from security.differential_privacy import DifferentialPrivacy

logger = logging.getLogger(__name__)

class FlowerClient(fl.client.NumPyClient):
    """Flower client implementation for federated learning."""
    
    def __init__(self, config: Dict, client_id: str):
        """
        Initialize the client with configuration.
        
        Args:
            config: Configuration dictionary with all settings
            client_id: Unique identifier for this client
        """
        self.config = config
        self.client_id = client_id
        self.client_config = config["client"]
        self.model_config = config["model"]
        self.data_config = config["data"]
        self.security_config = config["security"]
        
        # Set device-specific random seed for reproducibility
        match = re.search(r'\d+$', client_id)
        if match:
            seed = int(match.group()) + 42
        else:
            raise ValueError(f"Invalid client_id format: '{client_id}'. Must end with a numeric value.")
        np.random.seed(seed)
        tf.random.set_seed(seed)
        
        # Initialize device manager
        self.device_manager = DeviceManager(
            client_id=client_id,
            heterogeneity_enabled=self.client_config["device_heterogeneity"],
            resource_monitoring=self.client_config["resource_monitoring"]
        )
        
        # Initialize data variables
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.x_val = None
        self.y_val = None
        
        # Load and preprocess data
        self._load_and_preprocess_data()
        
        # Create model
        self.model = create_model(self.model_config)
        
        # Create differential privacy module if enabled
        self.dp = None
        if self.security_config["differential_privacy"]["enabled"]:
            self.dp = DifferentialPrivacy(self.security_config)
        
        # Track training history
        self.training_history = []
        
        logger.info(f"Client {client_id} initialized with device: {self.device_manager.device_info['system']} - {self.device_manager.device_info['machine']}")
    
    def _load_and_preprocess_data(self) -> None:
        """Load and preprocess data for this client."""
        # Ensure the data directory exists
        # Use the processed directory for cached preprocessed data
        data_path = Path("data/processed")
        data_path.mkdir(parents=True, exist_ok=True)
        
        # Check if we have cached preprocessed data for this client
        cache_file = data_path / f"preprocessed_{self.client_id}.npz"
        
        if os.path.exists(cache_file):
            logger.info(f"Loading preprocessed data from cache for client {self.client_id}")
            data = np.load(cache_file)
            self.x_train = data['x_train']
            self.y_train = data['y_train']
            self.x_test = data['x_test']
            self.y_test = data['y_test']
            if 'x_val' in data and 'y_val' in data:
                self.x_val = data['x_val']
                self.y_val = data['y_val']
        else:
            logger.info(f"Loading and preprocessing data for client {self.client_id}")
            # Load raw data
            raw_data = load_data(
                dataset=self.data_config["dataset"],
                path=str(data_path),
                client_id=self.client_id
            )
            
            # Preprocess data
            from data.preprocessing import split_train_test_validation, normalize_data
            
            # Split the data
            X = raw_data["X"]
            y = raw_data["y"]
            
            X_train, X_test, X_val, y_train, y_test, y_val = split_train_test_validation(
                X, y,
                test_size=self.data_config["test_size"],
                validation_size=self.data_config["validation_size"]
            )
            
            # Apply feature selection if enabled
            if self.data_config["feature_selection"]:
                from data.feature_selection import select_features
                X_train, selected_indices = select_features(
                    X_train, y_train,
                    method="kbest",
                    n_features=self.data_config["num_features"]
                )
                X_test = X_test[:, selected_indices]
                if X_val is not None:
                    X_val = X_val[:, selected_indices]
            
            # Normalize data
            X_train, scaler = normalize_data(X_train)
            X_test, _ = normalize_data(X_test, scaler)
            if X_val is not None:
                X_val, _ = normalize_data(X_val, scaler)
            
            # Save the preprocessed data
            self.x_train, self.y_train = X_train, y_train
            self.x_test, self.y_test = X_test, y_test
            if X_val is not None and y_val is not None:
                self.x_val, self.y_val = X_val, y_val
                np.savez(
                    cache_file, 
                    x_train=self.x_train, y_train=self.y_train,
                    x_test=self.x_test, y_test=self.y_test,
                    x_val=self.x_val, y_val=self.y_val
                )
            else:
                np.savez(
                    cache_file, 
                    x_train=self.x_train, y_train=self.y_train,
                    x_test=self.x_test, y_test=self.y_test
                )
        
        logger.info(f"Data loaded: {self.x_train.shape[0]} training samples, {self.x_test.shape[0]} test samples")
    
    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """
        Get model parameters.
        
        Args:
            config: Configuration for getting parameters
            
        Returns:
            List of model parameters as numpy arrays
        """
        # If the model instance itself is not created, create it using the model factory
        if self.model is None:
            from models.model_factory import create_model
            self.model = create_model(self.model_config)

        # If the underlying Keras model is not built yet, build it using training data
        if self.model.model is None:
            if self.x_train is None or self.y_train is None:
                raise RuntimeError("Training data not loaded; cannot build model")

            # Infer input shape from training data
            input_shape = self.x_train.shape[1:]

            # Infer number of classes from training labels
            unique_labels = np.unique(self.y_train)
            num_classes = len(unique_labels)

            # Build and compile the model using the configuration
            self.model.build_model(input_shape, num_classes)
            self.model.compile_model(learning_rate=self.model.config.get('learning_rate', 0.001))

        return self.model.get_weights()
    
    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        """
        Train the model on the local dataset.
        
        Args:
            parameters: Model parameters from the server
            config: Configuration for the training process
            
        Returns:
            Tuple of (updated_parameters, num_samples, metrics)
        """
        # Debug logging for parameters
        logger.debug(f"Received parameters: {len(parameters)} weight arrays")
        for i, param in enumerate(parameters):
            logger.debug(f"Parameter {i} shape: {param.shape}")
        
        # Update local model with global parameters
        try:
            self.model.set_weights(parameters)
        except Exception as e:
            logger.error(f"Error setting weights: {e}")
            # If setting weights fails, log more details
            if self.model is None:
                logger.error("Model is None before setting weights")
            else:
                logger.error(f"Model summary: {self.model.summary()}")
                logger.error(f"Model input shape: {self.model.input_shape}")
                logger.error(f"Model output shape: {self.model.output_shape}")
            raise
        
        # Check device resources
        resources = self.device_manager.check_resources()
        if not self.device_manager.can_train(resources):
            logger.warning(f"Client {self.client_id} cannot train due to resource constraints")
            return parameters, 0, {
                "status": "resource_constrained",
                "cpu_percent": resources.get("cpu_percent", -1),
                "memory_percent": resources.get("memory_percent", -1),
                "battery_percent": resources.get("battery_percent", -1)
            }
        
        # Adjust training parameters based on device capabilities
        adjusted_config = self.device_manager.adjust_training_parameters(self.client_config)
        
        epochs = config.get("local_epochs", adjusted_config["local_epochs"])
        batch_size = config.get("batch_size", adjusted_config["batch_size"])
        learning_rate = config.get("learning_rate", adjusted_config["learning_rate"])
        
        # Set learning rate if model supports it
        if hasattr(self.model, 'optimizer') and hasattr(self.model.optimizer, 'learning_rate'):
            self.model.optimizer.learning_rate.assign(learning_rate)
        
        # Define validation data if available
        validation_data = None
        if self.x_val is not None and self.y_val is not None:
            validation_data = (self.x_val, self.y_val)
        
        # Train the model
        start_time = time.time()
        history = self.model.fit(
            self.x_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            validation_data=validation_data
        )
        training_time = time.time() - start_time
        
        # Store training metrics
        metrics = {
            "status": "success",
            "training_time": training_time,
            "final_loss": float(history.history["loss"][-1]),
            "device_type": self.device_manager.device_info["machine"],
            "battery_status": resources.get("battery_percent", -1)
        }
        
        if "accuracy" in history.history:
            metrics["final_accuracy"] = float(history.history["accuracy"][-1])
        
        if validation_data is not None and "val_loss" in history.history:
            metrics["val_loss"] = float(history.history["val_loss"][-1])
            if "val_accuracy" in history.history:
                metrics["val_accuracy"] = float(history.history["val_accuracy"][-1])
        
        # Apply differential privacy if enabled
        parameters_to_return = self.model.get_weights()
        
        if self.security_config["differential_privacy"]["enabled"] and self.dp is not None:
            logger.info(f"Applying differential privacy to model updates for client {self.client_id}")
            
            # Get the original parameters for comparison
            original_params = parameters
            
            # Calculate the updates (difference between original and trained)
            updates = [trained - original for trained, original in zip(parameters_to_return, original_params)]
            
            # Apply differential privacy to the updates
            privatized_updates = self.dp.apply_dp_to_gradients(updates)
            
            # Add the privatized updates back to the original parameters
            parameters_to_return = [original + privatized for original, privatized in zip(original_params, privatized_updates)]
            
            # Compute privacy budget
            epsilon, delta = self.dp.compute_privacy_budget(
                num_samples=len(self.x_train),
                batch_size=batch_size,
                epochs=epochs
            )
            metrics["privacy_epsilon"] = epsilon
            metrics["privacy_delta"] = delta
        
        # Save local model if needed
        save_local_model(self.model, self.client_id)
        
        return parameters_to_return, len(self.x_train), metrics
    
    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:
        """
        Evaluate the model on the local test dataset.
        
        Args:
            parameters: Model parameters from the server
            config: Configuration for the evaluation process
            
        Returns:
            Tuple of (loss, num_samples, metrics)
        """
        # Update local model with global parameters
        self.model.set_weights(parameters)
        
        # Evaluate the model
        results = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        
        # Handle TensorFlow and non-TensorFlow models
        if isinstance(results, list):
            loss = results[0]
            metrics = {self.model.metrics_names[i]: results[i] for i in range(len(results))}
        else:
            loss = results
            metrics = {"loss": loss}
        
        # Get predictions for more detailed metrics
        y_pred = self.model.predict(self.x_test)
        
        # Calculate additional metrics if binary classification
        if len(np.unique(self.y_test)) == 2:
            from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
            
            # Ensure y_pred is in the correct format (binary)
            if y_pred.ndim > 1 and y_pred.shape[1] > 1:
                # Multi-class output, take the argmax
                y_pred_binary = np.argmax(y_pred, axis=1)
            else:
                # Binary output, threshold at 0.5
                y_pred_binary = (y_pred > 0.5).astype(int).flatten()
            
            try:
                metrics["precision"] = float(precision_score(self.y_test, y_pred_binary))
                metrics["recall"] = float(recall_score(self.y_test, y_pred_binary))
                metrics["f1"] = float(f1_score(self.y_test, y_pred_binary))
                
                # ROC AUC if we have probability predictions
                if hasattr(self.model, 'predict_proba'):
                    y_pred_proba = self.model.predict_proba(self.x_test)[:, 1]
                    metrics["auc"] = float(roc_auc_score(self.y_test, y_pred_proba))
                elif y_pred.ndim > 1 and y_pred.shape[1] > 1:
                    # For NN models that output class probabilities directly
                    metrics["auc"] = float(roc_auc_score(self.y_test, y_pred[:, 1]))
                else:
                    # For NN models with sigmoid output
                    metrics["auc"] = float(roc_auc_score(self.y_test, y_pred.flatten()))
            except Exception as e:
                logger.warning(f"Error calculating metrics: {e}")
        
        # Add device info to metrics
        metrics["device_type"] = self.device_manager.device_info["machine"]
        resources = self.device_manager.check_resources()
        metrics["cpu_percent"] = resources.get("cpu_percent", -1)
        metrics["memory_percent"] = resources.get("memory_percent", -1)
        
        return float(loss), len(self.x_test), metrics
    
    def start(self):
        """Start the Flower client."""
        # Log client details before starting
        logger.info(f"Starting Flower client {self.client_id}...")
        logger.info(f"Device details: {self.device_manager.device_info['system']} {self.device_manager.device_info['release']} on {self.device_manager.device_info['machine']}")
        logger.info(f"CPU: {self.device_manager.device_info['cpu_count']} cores, Memory: {self.device_manager.device_info['memory_total'] / (1024**3):.1f} GB")
        
        # Create secure connection if encryption is enabled
        secure_grpc = False
        if self.security_config["encryption"]["enabled"] and self.security_config["encryption"]["type"] == "tls":
            secure_grpc = True
            logger.info("TLS encryption enabled for client-server communication")
        
        # Start client
        server_ip = self.config['server']['address']
        if server_ip == "0.0.0.0":
            server_ip = "127.0.0.1"
        server_address = f"{server_ip}:{self.config['server']['port']}"
        logger.info(f"Connecting to server at {server_address}")
        
        try:
            start_client(
                server_address=server_address,
                client=FlowerClientWrapper(client=self).to_client(),
                root_certificates=None if not secure_grpc else self._get_root_certificates()
            )
            logger.info(f"Client {self.client_id} successfully completed federated learning")
        except Exception as e:
            logger.error(f"Error in client {self.client_id}: {e}")
    
    def _get_root_certificates(self) -> Optional[bytes]:
        """
        Get root certificates for TLS connection.
        
        Returns:
            Root certificates as bytes, or None if not found
        """
        # Try to load certificates from configured path
        cert_path = os.environ.get("FL_CERT_PATH", "./certificates/ca.crt")
        try:
            with open(cert_path, "rb") as f:
                return f.read()
        except FileNotFoundError:
            logger.warning(f"Certificate file not found at {cert_path}, using insecure connection")
            return None

import time  # Added for timing measurements

class FlowerClientWrapper(NumPyClient):
    def __init__(self, client):
        self.client = client

    def get_parameters(self, config):
        return self.client.get_parameters(config)

    def fit(self, parameters, config):
        return self.client.fit(parameters, config)

    def evaluate(self, parameters, config):
        return self.client.evaluate(parameters, config)

    def to_client(self):
        # The correct way to convert a NumPyClient to Client
        # Since fl.client.NumPyClientWrapper doesn't exist, use the to_client method
        # which is part of the NumPyClient class API
        return super().to_client()