"""
Model factory for creating machine learning models in FIDS.

This module implements a factory pattern for creating models based on configuration.
It supports both traditional ML models and deep learning models.
"""

import logging
from typing import Dict, Any, List, Optional
import os
import yaml

logger = logging.getLogger(__name__)

def create_model(config: Dict) -> Any:
    """
    Create a model based on configuration.
    
    Args:
        config: Model configuration dictionary containing type, name, and hyperparameters
        
    Returns:
        Model instance
    """
    model_type = config["type"]
    model_name = config["name"]
    
    logger.info(f"Creating model: type={model_type}, name={model_name}")
    
    # Ensure hyperparameters exist in config
    if "hyperparameters" not in config:
        config["hyperparameters"] = {}
    
    # Add default hyperparameters for each model type if missing
    if model_type == "deep_learning":
        if model_name not in config["hyperparameters"]:
            if model_name == "lstm":
                config["hyperparameters"]["lstm"] = {
                    "hidden_size": 128,
                    "num_layers": 2,
                    "dropout": 0.2
                }
            elif model_name == "bilstm":
                config["hyperparameters"]["bilstm"] = {
                    "hidden_size": 128,
                    "num_layers": 2,
                    "dropout": 0.2
                }
            elif model_name == "cnn":
                config["hyperparameters"]["cnn"] = {
                    "filters": [32, 64, 128],
                    "kernel_size": 3,
                    "pool_size": 2,
                    "dropout": 0.2
                }
            elif model_name == "mlp":
                config["hyperparameters"]["mlp"] = {
                    "hidden_sizes": [128, 64],
                    "dropout": 0.2
                }
    elif model_type == "traditional":
        if model_name not in config["hyperparameters"]:
            if model_name == "random_forest":
                config["hyperparameters"]["random_forest"] = {
                    "n_estimators": 100,
                    "max_depth": 10
                }
            elif model_name == "naive_bayes":
                config["hyperparameters"]["naive_bayes"] = {
                    "var_smoothing": 1e-9
                }
            elif model_name == "logistic_regression":
                config["hyperparameters"]["logistic_regression"] = {
                    "C": 1.0,
                    "max_iter": 100
                }
            elif model_name == "svm":
                config["hyperparameters"]["svm"] = {
                    "C": 1.0,
                    "kernel": "rbf"
                }
    
    # Add default input shape and number of classes for deep learning models
    if model_type == "deep_learning":
        if "default_input_shape" not in config:
            # Default to 10 features for time series data
            config["default_input_shape"] = (10,)
        if "default_num_classes" not in config:
            # Default to binary classification
            config["default_num_classes"] = 2
    
    if model_type == "traditional":
        if model_name == "random_forest":
            from .traditional import RandomForestModel
            return RandomForestModel(config)
        elif model_name == "naive_bayes":
            from .traditional import NaiveBayesModel
            return NaiveBayesModel(config)
        elif model_name == "logistic_regression":
            from .traditional import LogisticRegressionModel
            return LogisticRegressionModel(config)
        elif model_name == "svm":
            from .traditional import SVMModel
            return SVMModel(config)
        else:
            raise ValueError(f"Unknown traditional model: {model_name}")
    elif model_type == "deep_learning":
        if model_name == "lstm":
            from .deep_learning import LSTMModel
            return LSTMModel(config)
        elif model_name == "bilstm":
            from .deep_learning import BiLSTMModel
            return BiLSTMModel(config)
        elif model_name == "cnn":
            from .deep_learning import CNNModel
            return CNNModel(config)
        elif model_name == "mlp":
            from .deep_learning import MLPModel
            return MLPModel(config)
        else:
            raise ValueError(f"Unknown deep learning model: {model_name}")
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def list_available_models() -> Dict[str, List[str]]:
    """
    List all available models.
    
    Returns:
        Dictionary mapping model types to lists of model names
    """
    return {
        "traditional": ["random_forest", "naive_bayes", "logistic_regression", "svm"],
        "deep_learning": ["lstm", "bilstm", "cnn", "mlp"]
    }

def load_model_config(config_path: str) -> Dict:
    """
    Load model configuration from YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def save_model(model: Any, path: str) -> None:
    """
    Save a model to disk.
    
    Args:
        model: Model to save
        path: Path to save the model
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Save the model (using the model's save method)
    if hasattr(model, 'save'):
        model.save(path)
    else:
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(model, f)
    
    logger.info(f"Model saved to {path}")

def load_model(path: str, config: Optional[Dict] = None) -> Any:
    """
    Load a model from disk.
    
    Args:
        path: Path to load the model from
        config: Optional configuration to initialize the model
        
    Returns:
        Loaded model
    """
    # Check if the file exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    
    # Determine the file extension
    _, ext = os.path.splitext(path)
    
    if ext == '.h5':
        # TensorFlow/Keras model
        import tensorflow as tf
        model = tf.keras.models.load_model(path)
    elif ext == '.pt' or ext == '.pth':
        # PyTorch model
        import torch
        model = torch.load(path)
    else:
        # Assume pickle file for scikit-learn models
        import pickle
        with open(path, 'rb') as f:
            model = pickle.load(f)
    
    logger.info(f"Model loaded from {path}")
    return model