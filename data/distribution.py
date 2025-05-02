"""
Data distribution utilities for IID and non-IID partitioning.
"""

import logging
import numpy as np
from typing import Dict, List, Tuple
import random

logger = logging.getLogger(__name__)

def distribute_iid(
    X: np.ndarray,
    y: np.ndarray,
    num_clients: int
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Distribute data in an IID (Independent and Identically Distributed) manner.
    
    Args:
        X: Features
        y: Labels
        num_clients: Number of clients to distribute data to
        
    Returns:
        List of (X, y) tuples for each client
    """
    # Shuffle data
    indices = np.random.permutation(len(X))
    X_shuffled = X[indices]
    y_shuffled = y[indices]
    
    # Split data equally among clients
    client_data = []
    samples_per_client = len(X) // num_clients
    
    for i in range(num_clients):
        start_idx = i * samples_per_client
        end_idx = (i + 1) * samples_per_client if i < num_clients - 1 else len(X)
        
        client_data.append((
            X_shuffled[start_idx:end_idx],
            y_shuffled[start_idx:end_idx]
        ))
    
    logger.info(f"Distributed data in IID manner to {num_clients} clients")
    
    return client_data

def distribute_non_iid_label_skew(
    X: np.ndarray,
    y: np.ndarray,
    num_clients: int,
    num_classes: int,
    alpha: float = 0.5
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Distribute data in a non-IID manner with label distribution skew.
    Uses Dirichlet distribution to allocate different proportions of classes to clients.
    
    Args:
        X: Features
        y: Labels
        num_clients: Number of clients to distribute data to
        num_classes: Number of classes in the dataset
        alpha: Concentration parameter for Dirichlet distribution (lower means more skew)
        
    Returns:
        List of (X, y) tuples for each client
    """
    # Group data by class
    class_indices = [np.where(y == c)[0] for c in range(num_classes)]
    
    # Distribute class data using Dirichlet distribution
    client_data = [[] for _ in range(num_clients)]
    
    for c in range(num_classes):
        # Sample from Dirichlet distribution
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        
        # Calculate number of samples per client for this class
        num_samples_per_client = (proportions * len(class_indices[c])).astype(int)
        
        # Adjust to ensure all samples are allocated
        num_samples_per_client[-1] = len(class_indices[c]) - np.sum(num_samples_per_client[:-1])
        
        # Shuffle indices for this class
        indices_for_class = class_indices[c].copy()
        np.random.shuffle(indices_for_class)
        
        # Distribute data
        start_idx = 0
        for i in range(num_clients):
            end_idx = start_idx + num_samples_per_client[i]
            client_data[i].extend(indices_for_class[start_idx:end_idx])
            start_idx = end_idx
    
    # Convert to numpy arrays and shuffle
    client_data_arrays = []
    for indices in client_data:
        if len(indices) == 0:
            # Handle empty client
            client_data_arrays.append((np.array([]), np.array([])))
            continue
        
        # Shuffle indices
        np.random.shuffle(indices)
        
        # Extract data
        client_data_arrays.append((X[indices], y[indices]))
    
    logger.info(f"Distributed data in non-IID manner (label skew) to {num_clients} clients")
    
    return client_data_arrays

def distribute_non_iid_quantity_skew(
    X: np.ndarray,
    y: np.ndarray,
    num_clients: int,
    beta: float = 0.5
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Distribute data in a non-IID manner with quantity skew.
    Uses Beta distribution to allocate different amounts of data to clients.
    
    Args:
        X: Features
        y: Labels
        num_clients: Number of clients to distribute data to
        beta: Parameter for Beta distribution (lower means more skew)
        
    Returns:
        List of (X, y) tuples for each client
    """
    # Shuffle data
    indices = np.random.permutation(len(X))
    X_shuffled = X[indices]
    y_shuffled = y[indices]
    
    # Sample from Beta distribution
    proportions = np.random.beta(beta, beta, size=num_clients)
    proportions = proportions / np.sum(proportions)
    
    # Calculate number of samples per client
    num_samples_per_client = (proportions * len(X)).astype(int)
    
    # Adjust to ensure all samples are allocated
    num_samples_per_client[-1] = len(X) - np.sum(num_samples_per_client[:-1])
    
    # Distribute data
    client_data = []
    start_idx = 0
    for i in range(num_clients):
        end_idx = start_idx + num_samples_per_client[i]
        client_data.append((
            X_shuffled[start_idx:end_idx],
            y_shuffled[start_idx:end_idx]
        ))
        start_idx = end_idx
    
    logger.info(f"Distributed data in non-IID manner (quantity skew) to {num_clients} clients")
    
    return client_data

def distribute_data(
    X: np.ndarray,
    y: np.ndarray,
    num_clients: int,
    distribution_type: str = "iid",
    **kwargs
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Distribute data to clients.
    
    Args:
        X: Features
        y: Labels
        num_clients: Number of clients to distribute data to
        distribution_type: Type of distribution (iid, non_iid_label, or non_iid_quantity)
        **kwargs: Additional arguments for the distribution method
        
    Returns:
        List of (X, y) tuples for each client
    """
    if distribution_type == "iid":
        return distribute_iid(X, y, num_clients)
    elif distribution_type == "non_iid_label":
        num_classes = len(np.unique(y))
        return distribute_non_iid_label_skew(X, y, num_clients, num_classes, **kwargs)
    elif distribution_type == "non_iid_quantity":
        return distribute_non_iid_quantity_skew(X, y, num_clients, **kwargs)
    else:
        raise ValueError(f"Unknown distribution type: {distribution_type}")

