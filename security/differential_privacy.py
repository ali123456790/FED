"""
Differential privacy mechanisms using TensorFlow Privacy.
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Any
import tensorflow as tf
import tensorflow_privacy as tfp

logger = logging.getLogger(__name__)

class DifferentialPrivacy:
    """Differential privacy mechanisms for federated learning."""
    
    def __init__(self, config: Dict):
        """
        Initialize differential privacy with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.dp_config = config["differential_privacy"]
        
        # Extract parameters
        self.noise_multiplier = self.dp_config["noise_multiplier"]
        self.l2_norm_clip = self.dp_config["l2_norm_clip"]
        
        logger.info(f"Differential privacy initialized with noise_multiplier={self.noise_multiplier}, l2_norm_clip={self.l2_norm_clip}")
    
    def apply_dp_to_gradients(self, gradients: List[np.ndarray]) -> List[np.ndarray]:
        """
        Apply differential privacy to gradients.
        
        Args:
            gradients: List of gradients
            
        Returns:
            List of privatized gradients
        """
        # Convert to TensorFlow tensors
        tf_gradients = [tf.convert_to_tensor(grad) for grad in gradients]
        
        # Clip gradients
        tf_gradients_clipped, _ = tf.clip_by_global_norm(tf_gradients, self.l2_norm_clip)
        
        # Add noise
        tf_gradients_privatized = []
        for grad in tf_gradients_clipped:
            noise = tf.random.normal(shape=grad.shape, mean=0.0, stddev=self.noise_multiplier * self.l2_norm_clip)
            tf_gradients_privatized.append(grad + noise)
        
        # Convert back to numpy arrays
        gradients_privatized = [grad.numpy() for grad in tf_gradients_privatized]
        
        return gradients_privatized
    
    def apply_dp_to_model(self, model: Any) -> Any:
        """
        Apply differential privacy to a model.
        
        Args:
            model: Model to privatize
            
        Returns:
            Privatized model
        """
        # This is a placeholder for actual implementation
        # In a real implementation, you would use TensorFlow Privacy to create a DP optimizer
        
        # For TensorFlow models
        if hasattr(model, 'optimizer'):
            # Create DP optimizer
            optimizer = tfp.DPKerasAdamOptimizer(
                l2_norm_clip=self.l2_norm_clip,
                noise_multiplier=self.noise_multiplier,
                num_microbatches=1,
                learning_rate=model.optimizer.learning_rate
            )
            
            # Compile model with DP optimizer
            model.compile(
                optimizer=optimizer,
                loss=model.loss,
                metrics=model.metrics
            )
        
        return model
    
    def compute_privacy_budget(self, num_samples: int, batch_size: int, epochs: int, delta: float = 1e-5) -> Tuple[float, float]:
        """
        Compute privacy budget (epsilon, delta).
        
        Args:
            num_samples: Number of training samples
            batch_size: Batch size
            epochs: Number of training epochs
            delta: Target delta
            
        Returns:
            Tuple of (epsilon, delta)
        """
        # Calculate number of steps
        steps = epochs * (num_samples // batch_size)
        
        # Compute epsilon
        from tensorflow_privacy.privacy.analysis.rdp_accountant import compute_rdp
        from tensorflow_privacy.privacy.analysis.rdp_accountant import get_privacy_spent
        
        # Compute RDP
        orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
        rdp = compute_rdp(
            q=batch_size / num_samples,
            noise_multiplier=self.noise_multiplier,
            steps=steps,
            orders=orders
        )
        
        # Convert to (epsilon, delta)
        epsilon, _ = get_privacy_spent(orders, rdp, target_delta=delta)
        
        logger.info(f"Privacy budget: epsilon={epsilon:.4f}, delta={delta}")
        
        return epsilon, delta

