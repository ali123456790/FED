"""
Utility functions for the server.
"""

import os
import pickle
from typing import Any, Dict
import logging

logger = logging.getLogger(__name__)

def save_model(model: Any, path: str) -> None:
    """
    Save a model to disk.
    
    Args:
        model: The model to save
        path: Path to save the model to
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Model saved to {path}")

def load_model(path: str) -> Any:
    """
    Load a model from disk.
    
    Args:
        path: Path to load the model from
        
    Returns:
        The loaded model
    """
    if not os.path.exists(path):
        logger.error(f"Model file not found: {path}")
        return None
    
    with open(path, "rb") as f:
        model = pickle.load(f)
    logger.info(f"Model loaded from {path}")
    return model

def get_model_size(model: Any) -> int:
    """
    Get the size of a model in bytes.
    
    Args:
        model: The model to get the size of
        
    Returns:
        Size of the model in bytes
    """
    # Serialize the model to get its size
    with open("temp_model.pkl", "wb") as f:
        pickle.dump(model, f)
    
    size = os.path.getsize("temp_model.pkl")
    os.remove("temp_model.pkl")
    return size

def monitor_resources() -> Dict[str, float]:
    """
    Monitor server resources.
    
    Returns:
        Dictionary with resource usage information
    """
    import psutil
    
    # Get CPU usage
    cpu_percent = psutil.cpu_percent(interval=1)
    
    # Get memory usage
    memory = psutil.virtual_memory()
    memory_percent = memory.percent
    
    return {
        "cpu_percent": cpu_percent,
        "memory_percent": memory_percent,
    }

