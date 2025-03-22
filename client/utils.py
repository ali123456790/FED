"""
Utility functions for the client.
"""

import os
import logging
import numpy as np
from typing import Dict, Tuple, Any
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif

logger = logging.getLogger(__name__)

def load_data(dataset: str, path: str, client_id: str) -> Dict:
    """
    Load data for a specific client.
    
    Args:
        dataset: Name of the dataset
        path: Path to the dataset
        client_id: ID of the client
        
    Returns:
        Dictionary with loaded data
    """
    logger.info(f"Loading {dataset} data for client {client_id}")
    
    # This is a placeholder for actual data loading
    # In a real implementation, you would load data from files or databases
    
    # Simulate loading data
    if dataset == "n_baiot":
        # Simulate N-BaIoT dataset
        # In a real implementation, you would load the actual dataset
        num_samples = 1000
        num_features = 115  # N-BaIoT has 115 features
        
        # Generate random data for demonstration
        X = np.random.rand(num_samples, num_features)
        y = np.random.randint(0, 2, num_samples)  # Binary classification
        
        return {"X": X, "y": y}
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

def preprocess_data(
    data: Dict,
    test_size: float = 0.2,
    validation_size: float = 0.1,
    feature_selection: bool = True,
    num_features: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Preprocess data for training and evaluation.
    
    Args:
        data: Dictionary with raw data
        test_size: Proportion of data to use for testing
        validation_size: Proportion of data to use for validation
        feature_selection: Whether to perform feature selection
        num_features: Number of features to select if feature_selection is True
        
    Returns:
        Tuple of (x_train, y_train, x_test, y_test)
    """
    X = data["X"]
    y = data["y"]
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    # Perform feature selection if enabled
    if feature_selection:
        selector = SelectKBest(f_classif, k=num_features)
        X_train = selector.fit_transform(X_train, y_train)
        X_test = selector.transform(X_test)
    
    logger.info(f"Preprocessed data: X_train shape={X_train.shape}, X_test shape={X_test.shape}")
    
    return X_train, y_train, X_test, y_test

def save_local_model(model: Any, client_id: str, path: str = "./models/local") -> None:
    """
    Save a local model to disk.
    
    Args:
        model: The model to save
        client_id: ID of the client
        path: Path to save the model to
    """
    import pickle
    
    os.makedirs(path, exist_ok=True)
    model_path = os.path.join(path, f"model_client_{client_id}.pkl")
    
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    
    logger.info(f"Local model saved to {model_path}")

def load_local_model(client_id: str, path: str = "./models/local") -> Any:
    """
    Load a local model from disk.
    
    Args:
        client_id: ID of the client
        path: Path to load the model from
        
    Returns:
        The loaded model, or None if not found
    """
    import pickle
    
    model_path = os.path.join(path, f"model_client_{client_id}.pkl")
    
    if not os.path.exists(model_path):
        logger.warning(f"Local model not found: {model_path}")
        return None
    
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    logger.info(f"Local model loaded from {model_path}")
    return model

def load_n_baiot_data(client_id: str, cache: bool = True):
    """
    Load N-BaIoT dataset for a specific client.
    
    Args:
        client_id: Client identifier
        cache: Whether to use cached preprocessed data
        
    Returns:
        Training and test data (X_train, y_train, X_test, y_test)
    """
    # Try to load preprocessed data from cache
    cache_file = os.path.join("data", "processed", f"preprocessed_{client_id}.npz")
    
    if cache and os.path.exists(cache_file):
        logger.info(f"Loading preprocessed data from {cache_file}")
        data = np.load(cache_file)
        return data['X_train'], data['y_train'], data['X_test'], data['y_test']
    
    # Simulate loading data
    num_samples = 1000
    num_features = 115  # N-BaIoT has 115 features
    
    # Generate random data for demonstration
    X = np.random.rand(num_samples, num_features)
    y = np.random.randint(0, 2, num_samples)  # Binary classification
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Save preprocessed data to cache
    if cache:
        os.makedirs(os.path.dirname(os.path.join("data", "processed")), exist_ok=True)
        np.savez_compressed(os.path.join("data", "processed", f"preprocessed_{client_id}.npz"),
                          X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
        
    return X_train, y_train, X_test, y_test

