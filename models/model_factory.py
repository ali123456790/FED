"""
Model factory for creating models.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def create_model(config: Dict) -> Any:
    """
    Create a model based on configuration.
    
    Args:
        config: Model configuration
        
    Returns:
        Model instance
    """
    model_type = config["type"]
    model_name = config["name"]
    
    logger.info(f"Creating model: type={model_type}, name={model_name}")
    
    if model_type == "traditional":
        if model_name == "random_forest":
            from .traditional import RandomForestModel
            return RandomForestModel(config)
        elif model_name == "naive_bayes":
            from .traditional import NaiveBayesModel
            return NaiveBayesModel(config)
        else:
            raise ValueError(f"Unknown traditional model: {model_name}")
    elif model_type == "deep_learning":
        if model_name == "lstm":
            from .deep_learning import LSTMModel
            return LSTMModel(config)
        elif model_name == "bilstm":
            from .deep_learning import BiLSTMModel
            return BiLSTMModel(config)
        else:
            raise ValueError(f"Unknown deep learning model: {model_name}")
    else:
        raise ValueError(f"Unknown model type: {model_type}")

