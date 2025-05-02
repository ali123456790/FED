"""
Differential Privacy implementation for federated learning.

This module provides mechanisms for applying differential privacy to model updates
using TensorFlow Privacy and custom noise mechanisms.
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import tensorflow as tf
import tensorflow_privacy as tfp
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy
import math

logger = logging.getLogger(__name__)

class DifferentialPrivacy:
    """Differential Privacy mechanisms for federated learning."""
    
    def __init__(self, config: Dict):
        """
        Initialize DP with configuration.
        
        Args:
            config: Configuration dictionary containing DP parameters
        """
        self.config = config.get("security", {}).get("differential_privacy", {})
        self.enabled = self.config.get("enabled", True)  # Enable by default
        
        # DP parameters
        self.noise_multiplier = self.config.get("noise_multiplier", 1.0)
        self.l2_norm_clip = self.config.get("l2_norm_clip", 1.0)
        self.target_epsilon = self.config.get("target_epsilon", 10.0)
        self.target_delta = self.config.get("target_delta", 1e-5)
        
        # Mechanism type: 'gaussian' or 'laplace'
        self.mechanism = self.config.get("mechanism", "gaussian")
        
        # Track privacy budget
        self.cumulative_steps = 0
        self.privacy_spent = 0.0
        
        # Per-client privacy tracking
        self.client_privacy_spent = {}
        
        if self.enabled:
            logger.info(f"Differential Privacy enabled with {self.mechanism} mechanism")
            logger.info(f"DP parameters: noise={self.noise_multiplier}, clip={self.l2_norm_clip}, "
                       f"target_ε={self.target_epsilon}, target_δ={self.target_delta}")
        else:
            logger.info("Differential Privacy is disabled")
    
    def apply_dp_to_gradients(self, gradients: List[np.ndarray], client_id: str) -> List[np.ndarray]:
        """
        Apply differential privacy to gradients.
        
        Args:
            gradients: List of gradient arrays
            client_id: Client identifier for tracking privacy budget
            
        Returns:
            Privacy-preserving gradients
        """
        if not self.enabled:
            return gradients
        
        try:
            # Convert gradients to tensors
            gradient_tensors = [tf.convert_to_tensor(g) for g in gradients]
            
            # Compute global norm for clipping
            global_norm = tf.linalg.global_norm(gradient_tensors)
            
            # Clip gradients
            factor = tf.minimum(self.l2_norm_clip / global_norm, 1.0)
            clipped_gradients = [g * factor for g in gradient_tensors]
            
            # Apply noise based on mechanism
            if self.mechanism == "gaussian":
                noised_gradients = self._apply_gaussian_noise(clipped_gradients)
            else:  # laplace
                noised_gradients = self._apply_laplace_noise(clipped_gradients)
            
            # Convert back to numpy arrays
            result = [g.numpy() for g in noised_gradients]
            
            # Update privacy tracking
            self.cumulative_steps += 1
            self._update_privacy_spent(client_id)
            
            return result
            
        except Exception as e:
            logger.error(f"Error applying DP to gradients: {e}")
            return gradients
    
    def _apply_gaussian_noise(self, tensors: List[tf.Tensor]) -> List[tf.Tensor]:
        """Apply Gaussian noise to tensors."""
        noised_tensors = []
        for tensor in tensors:
            noise_stddev = self.l2_norm_clip * self.noise_multiplier
            noise = tf.random.normal(shape=tensor.shape, 
                                   mean=0.0, 
                                   stddev=noise_stddev,
                                   dtype=tensor.dtype)
            noised_tensors.append(tensor + noise)
        return noised_tensors
    
    def _apply_laplace_noise(self, tensors: List[tf.Tensor]) -> List[tf.Tensor]:
        """Apply Laplace noise to tensors."""
        noised_tensors = []
        for tensor in tensors:
            # Scale parameter for Laplace noise
            scale = self.l2_norm_clip * self.noise_multiplier / math.sqrt(2)
            
            # Generate Laplace noise using inverse CDF method
            uniform = tf.random.uniform(shape=tensor.shape,
                                      minval=0.0,
                                      maxval=1.0,
                                      dtype=tensor.dtype)
            noise = -scale * tf.sign(uniform - 0.5) * tf.math.log(1 - 2 * tf.abs(uniform - 0.5))
            noised_tensors.append(tensor + noise)
        return noised_tensors
    
    def compute_privacy_budget(self, num_samples: int, batch_size: int, epochs: int) -> Tuple[float, float]:
        """
        Compute the privacy budget (epsilon, delta) for given training parameters.
        
        Args:
            num_samples: Number of training samples
            batch_size: Batch size used in training
            epochs: Number of training epochs
            
        Returns:
            Tuple of (epsilon, delta) values
        """
        if not self.enabled:
            return float('inf'), 1.0
        
        try:
            # Calculate number of steps
            steps = (num_samples * epochs) // batch_size
            
            # Use TF Privacy's compute_dp_sgd_privacy
            eps, delta = compute_dp_sgd_privacy.compute_dp_sgd_privacy(
                n=num_samples,
                batch_size=batch_size,
                noise_multiplier=self.noise_multiplier,
                epochs=epochs,
                delta=self.target_delta
            )
            
            return float(eps), float(delta)
            
        except Exception as e:
            logger.error(f"Error computing privacy budget: {e}")
            return float('inf'), 1.0
    
    def _update_privacy_spent(self, client_id: str) -> None:
        """Update privacy budget tracking for a client."""
        try:
            if client_id not in self.client_privacy_spent:
                self.client_privacy_spent[client_id] = {
                    'steps': 0,
                    'epsilon': 0.0,
                    'delta': self.target_delta
                }
            
            self.client_privacy_spent[client_id]['steps'] += 1
            
            # Compute current privacy spent
            eps, _ = self.compute_privacy_budget(
                num_samples=1000,  # Use a default value
                batch_size=32,     # Use a default value
                epochs=self.client_privacy_spent[client_id]['steps']
            )
            
            self.client_privacy_spent[client_id]['epsilon'] = eps
            
        except Exception as e:
            logger.error(f"Error updating privacy spent: {e}")
    
    def get_privacy_per_client(self, client_id: str) -> Dict:
        """
        Get privacy metrics for a specific client.
        
        Args:
            client_id: Client identifier
            
        Returns:
            Dictionary with privacy metrics
        """
        if not self.enabled:
            return {"enabled": False}
        
        return {
            "enabled": True,
            "mechanism": self.mechanism,
            "noise_multiplier": self.noise_multiplier,
            "l2_norm_clip": self.l2_norm_clip,
            "steps": self.client_privacy_spent.get(client_id, {}).get('steps', 0),
            "epsilon": self.client_privacy_spent.get(client_id, {}).get('epsilon', 0.0),
            "delta": self.target_delta
        }
    
    def apply_dp_to_parameters(self, parameters: List[np.ndarray], client_id: str) -> List[np.ndarray]:
        """
        Apply differential privacy to model parameters.
        
        Args:
            parameters: List of parameter arrays
            client_id: Client identifier for tracking privacy budget
            
        Returns:
            Privacy-preserving parameters
        """
        if not self.enabled:
            return parameters
        
        try:
            # Convert parameters to tensors
            param_tensors = [tf.convert_to_tensor(p) for p in parameters]
            
            # Compute global norm for clipping
            global_norm = tf.linalg.global_norm(param_tensors)
            
            # Clip parameters
            factor = tf.minimum(self.l2_norm_clip / global_norm, 1.0)
            clipped_params = [p * factor for p in param_tensors]
            
            # Apply noise based on mechanism
            if self.mechanism == "gaussian":
                noised_params = self._apply_gaussian_noise(clipped_params)
            else:  # laplace
                noised_params = self._apply_laplace_noise(clipped_params)
            
            # Convert back to numpy arrays
            result = [p.numpy() for p in noised_params]
            
            # Update privacy tracking
            self.cumulative_steps += 1
            self._update_privacy_spent(client_id)
            
            return result
            
        except Exception as e:
            logger.error(f"Error applying DP to parameters: {e}")
            return parameters
    
    def verify_privacy_guarantee(self, epsilon: float, delta: float) -> bool:
        """
        Verify if privacy guarantees are met.
        
        Args:
            epsilon: Privacy budget epsilon
            delta: Privacy budget delta
            
        Returns:
            True if privacy guarantees are met, False otherwise
        """
        return epsilon <= self.target_epsilon and delta <= self.target_delta