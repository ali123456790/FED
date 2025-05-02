"""
Client wrapper module for Flower federated learning.

This module provides wrapper classes to adapt the NumPyClient interface to
the Flower Client interface with proper parameter handling.
"""

import logging
import flwr as fl
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
from flwr.common import (
    EvaluateIns, EvaluateRes, FitIns, FitRes, Parameters, Scalar,
    parameters_to_ndarrays, ndarrays_to_parameters
)
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, roc_auc_score

logger = logging.getLogger(__name__)

class FlowerClientWrapper(fl.client.NumPyClient):
    """Wrapper for NumPyClient that ensures proper parameter handling."""
    
    def __init__(self, client):
        """
        Initialize the wrapper with a client instance.
        
        Args:
            client: The underlying client implementation
        """
        self.client = client
    
    def get_parameters(self, config):
        """Get parameters from client and convert to Flower Parameters format."""
        return self.client.get_parameters(config)
    
    def fit(self, parameters, config):
        """Train model and convert parameters between formats."""
        try:
            # Convert parameters to numpy arrays if needed
            if hasattr(parameters, 'tensors'):
                parameters_as_ndarrays = parameters_to_ndarrays(parameters)
            else:
                parameters_as_ndarrays = parameters
                
            # Call client's fit method
            updated_parameters, num_examples, metrics = self.client.fit(
                parameters_as_ndarrays, config
            )
            
            return updated_parameters, num_examples, metrics
            
        except Exception as e:
            logger.error(f"Error in fit: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return [], 0, {"error": str(e)}
    
    def evaluate(self, parameters, config):
        """Evaluate model on the locally held test set."""
        try:
            # Convert parameters to numpy arrays if needed
            if hasattr(parameters, 'tensors'):
                parameters_as_ndarrays = parameters_to_ndarrays(parameters)
            else:
                parameters_as_ndarrays = parameters
                
            loss, num_examples, metrics = self.client.evaluate(
                parameters_as_ndarrays, config
            )
            
            # Ensure metrics contains only serializable values
            clean_metrics = {}
            for k, v in metrics.items():
                if isinstance(v, (bool, int, float, str)):
                    clean_metrics[k] = v
                elif isinstance(v, list):
                    # For confusion matrix and other list-based metrics,
                    # convert to string representation to ensure serializability
                    try:
                        clean_metrics[k] = str(v)
                    except:
                        logger.warning(f"Could not serialize list metric {k}")
                else:
                    try:
                        if hasattr(v, 'item'):
                            clean_metrics[k] = v.item()
                        else:
                            clean_metrics[k] = float(v)
                    except:
                        logger.warning(f"Dropping metric {k} with non-serializable value of type {type(v)}")
            
            return float(loss), num_examples, clean_metrics
            
        except Exception as e:
            logger.error(f"Error in evaluate: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return 0.0, 0, {"error": str(e)}
    
    def to_client(self):
        """
        Convert NumPyClient to Client.
        
        Returns:
            Flower Client instance
        """
        # Use NumPyClientWrapper for Flower 1.0.0 compatibility
        return fl.client.NumPyClientWrapper(self)


class FlowerClientWrapperDebug(fl.client.Client):
    """Debug wrapper for Client with additional logging."""
    
    def __init__(self, numpy_client):
        """
        Initialize the debug wrapper.
        
        Args:
            numpy_client: The NumPyClient to wrap
        """
        self.numpy_client = numpy_client
        self.logger = logging.getLogger("client.wrapper.debug")
    
    def get_parameters(self, ins):
        """Get parameters with debug logging."""
        self.logger.debug("Debug wrapper: get_parameters() called")
        return self.numpy_client.get_parameters(ins)
    
    def fit(self, ins):
        """
        Fit with additional debug logging.
        
        Args:
            ins: Fit instructions
            
        Returns:
            Fit results
        """
        self.logger.debug(f"Debug wrapper: fit() called for round {ins.config.get('server_round', 0)}")
        try:
            result = self.numpy_client.fit(ins)
            self.logger.debug(f"Debug wrapper: fit() completed successfully")
            return result
        except Exception as e:
            self.logger.error(f"Debug wrapper: Error in fit(): {e}")
            raise
    
    def evaluate(self, ins):
        """
        Evaluate with additional debug logging.
        
        Args:
            ins: Evaluate instructions
            
        Returns:
            Evaluation results
        """
        self.logger.debug(f"Debug wrapper: evaluate() called for round {ins.config.get('server_round', 0)}")
        try:
            result = self.numpy_client.evaluate(ins)
            self.logger.debug(f"Debug wrapper: evaluate() completed successfully")
            return result
        except Exception as e:
            self.logger.error(f"Debug wrapper: Error in evaluate(): {e}")
            raise 