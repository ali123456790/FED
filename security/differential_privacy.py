"""
Differential privacy mechanisms using TensorFlow Privacy.

This module implements differential privacy techniques to protect user privacy
during federated learning by adding calibrated noise to model updates.
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
import tensorflow as tf
import tensorflow_privacy as tfp
from tensorflow_privacy.privacy.analysis.rdp_accountant import compute_rdp, get_privacy_spent

logger = logging.getLogger(__name__)

class DifferentialPrivacy:
    """Differential privacy mechanisms for federated learning."""
    
    def __init__(self, config: Dict):
        """
        Initialize differential privacy with configuration.
        
        Args:
            config: Configuration dictionary with differential privacy settings
        """
        self.config = config
        self.dp_config = config.get("differential_privacy", {})
        
        # Extract parameters
        self.enabled = self.dp_config.get("enabled", False)
        self.noise_multiplier = self.dp_config.get("noise_multiplier", 1.0)
        self.l2_norm_clip = self.dp_config.get("l2_norm_clip", 1.0)
        self.microbatches = self.dp_config.get("microbatches", 1)
        self.target_delta = self.dp_config.get("target_delta", 1e-5)
        
        # Privacy budget tracking
        self.privacy_spent = 0.0
        self.cumulative_steps = 0
        
        # Track clients' participation for personalized privacy accounting
        self.client_participation = {}
        
        if self.enabled:
            logger.info(f"Differential privacy initialized with noise_multiplier={self.noise_multiplier}, "
                        f"l2_norm_clip={self.l2_norm_clip}")
        else:
            logger.info("Differential privacy is disabled")
    
    def apply_dp_to_gradients(self, gradients: List[np.ndarray], client_id: Optional[str] = None) -> List[np.ndarray]:
        """
        Apply differential privacy to gradients.
        
        Args:
            gradients: List of gradients
            client_id: Optional client identifier for personalized privacy accounting
            
        Returns:
            List of privatized gradients
        """
        if not self.enabled:
            return gradients
        
        try:
            # Convert to TensorFlow tensors
            tf_gradients = [tf.convert_to_tensor(grad, dtype=tf.float32) for grad in gradients]
            
            # Clip gradients
            tf_gradients_clipped, _ = tf.clip_by_global_norm(tf_gradients, self.l2_norm_clip)
            
            # Add noise
            tf_gradients_privatized = []
            for grad in tf_gradients_clipped:
                if grad is not None and grad.shape.num_elements() > 0:
                    noise = tf.random.normal(
                        shape=grad.shape, 
                        mean=0.0, 
                        stddev=self.noise_multiplier * self.l2_norm_clip,
                        dtype=tf.float32
                    )
                    tf_gradients_privatized.append(grad + noise)
                else:
                    tf_gradients_privatized.append(grad)
            
            # Convert back to numpy arrays
            gradients_privatized = [grad.numpy() if grad is not None else None for grad in tf_gradients_privatized]
            
            # Update client participation (for personalized privacy accounting)
            if client_id:
                if client_id not in self.client_participation:
                    self.client_participation[client_id] = 0
                self.client_participation[client_id] += 1
            
            # Update steps (for privacy budget calculation)
            self.cumulative_steps += 1
            
            return gradients_privatized
        
        except Exception as e:
            logger.error(f"Error applying differential privacy to gradients: {e}")
            # Return original gradients in case of error
            return gradients
    
    def apply_dp_to_model(self, model: Any) -> Any:
        """
        Apply differential privacy to a model by replacing the optimizer.
        
        Args:
            model: Model to privatize
            
        Returns:
            Privatized model with DP optimizer
        """
        if not self.enabled:
            return model
        
        try:
            # For TensorFlow models
            if hasattr(model, 'optimizer'):
                # Get current optimizer parameters
                learning_rate = model.optimizer.learning_rate
                
                # Create DP optimizer
                if isinstance(model.optimizer, tf.keras.optimizers.Adam):
                    dp_optimizer = tfp.DPKerasAdamOptimizer(
                        l2_norm_clip=self.l2_norm_clip,
                        noise_multiplier=self.noise_multiplier,
                        num_microbatches=self.microbatches,
                        learning_rate=learning_rate
                    )
                elif isinstance(model.optimizer, tf.keras.optimizers.SGD):
                    dp_optimizer = tfp.DPKerasSGDOptimizer(
                        l2_norm_clip=self.l2_norm_clip,
                        noise_multiplier=self.noise_multiplier,
                        num_microbatches=self.microbatches,
                        learning_rate=learning_rate
                    )
                else:
                    # Default to DP-SGD for unsupported optimizers
                    logger.warning(f"Unsupported optimizer type: {type(model.optimizer)}, using DP-SGD instead")
                    dp_optimizer = tfp.DPKerasSGDOptimizer(
                        l2_norm_clip=self.l2_norm_clip,
                        noise_multiplier=self.noise_multiplier,
                        num_microbatches=self.microbatches,
                        learning_rate=learning_rate
                    )
                
                # Compile model with DP optimizer
                model.compile(
                    optimizer=dp_optimizer,
                    loss=model.loss,
                    metrics=model.metrics
                )
                
                logger.info(f"Applied differential privacy to model with {type(dp_optimizer).__name__}")
            
            return model
        
        except Exception as e:
            logger.error(f"Error applying differential privacy to model: {e}")
            # Return original model in case of error
            return model
    
    def compute_privacy_budget(
        self, 
        num_samples: int, 
        batch_size: int, 
        epochs: int, 
        delta: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Compute privacy budget (epsilon, delta) using RDP accountant.
        
        Args:
            num_samples: Number of training samples
            batch_size: Batch size
            epochs: Number of training epochs
            delta: Target delta (if None, use self.target_delta)
            
        Returns:
            Tuple of (epsilon, delta)
        """
        if not self.enabled:
            return (float('inf'), 1.0)
        
        try:
            # Set delta if not provided
            if delta is None:
                delta = self.target_delta
            
            # Calculate sampling probability q
            q = batch_size / num_samples
            
            # Calculate number of steps
            steps = epochs * (num_samples // batch_size)
            
            # Add to cumulative steps for tracking
            self.cumulative_steps += steps
            
            # Compute RDP for a range of orders
            orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
            rdp = compute_rdp(
                q=q,
                noise_multiplier=self.noise_multiplier,
                steps=steps,
                orders=orders
            )
            
            # Convert to (epsilon, delta)
            epsilon, _ = get_privacy_spent(orders, rdp, target_delta=delta)
            
            # Update privacy spent
            self.privacy_spent = epsilon
            
            logger.info(f"Privacy budget: epsilon={epsilon:.4f}, delta={delta}")
            
            return epsilon, delta
        
        except Exception as e:
            logger.error(f"Error computing privacy budget: {e}")
            # Return infinity (no privacy guarantee) in case of error
            return float('inf'), 1.0
    
    def get_current_privacy_spent(self, delta: Optional[float] = None) -> Tuple[float, float]:
        """
        Get current privacy spent so far.
        
        Args:
            delta: Target delta (if None, use self.target_delta)
            
        Returns:
            Tuple of (epsilon, delta)
        """
        if not self.enabled:
            return (float('inf'), 1.0)
        
        if delta is None:
            delta = self.target_delta
        
        return self.privacy_spent, delta
    
    def get_privacy_per_client(self, client_id: str) -> Dict[str, Union[float, int]]:
        """
        Get privacy metrics for a specific client.
        
        Args:
            client_id: Client identifier
            
        Returns:
            Dictionary with privacy metrics for the client
        """
        if not self.enabled or client_id not in self.client_participation:
            return {
                "enabled": self.enabled,
                "participations": 0,
                "approximate_epsilon": float('inf')
            }
        
        # Get number of times client participated
        participations = self.client_participation.get(client_id, 0)
        
        # Calculate approximate epsilon for this client (simplified)
        # In a real system, you would use a more sophisticated accounting method
        approximate_epsilon = self.privacy_spent * (participations / max(1, self.cumulative_steps))
        
        return {
            "enabled": self.enabled,
            "participations": participations,
            "approximate_epsilon": approximate_epsilon,
            "noise_multiplier": self.noise_multiplier,
            "l2_norm_clip": self.l2_norm_clip
        }
    
    def adjust_noise_for_round(self, server_round: int, total_rounds: int) -> None:
        """
        Adjust noise multiplier based on the current round.
        
        This allows for adaptive privacy protection - more noise in early rounds,
        less noise in later rounds for better convergence.
        
        Args:
            server_round: Current server round
            total_rounds: Total number of rounds
        """
        if not self.enabled:
            return
        
        # Example adaptive strategy: linearly decrease noise over rounds
        if self.dp_config.get("adaptive_noise", False):
            initial_noise = self.dp_config.get("initial_noise_multiplier", self.noise_multiplier)
            final_noise = self.dp_config.get("final_noise_multiplier", self.noise_multiplier * 0.5)
            
            if server_round < total_rounds:
                # Linear interpolation
                progress = server_round / total_rounds
                new_noise = initial_noise - (progress * (initial_noise - final_noise))
                
                # Update noise multiplier
                self.noise_multiplier = new_noise
                
                logger.info(f"Adjusted noise multiplier to {self.noise_multiplier:.4f} for round {server_round}")