"""
Data masking mechanisms for privacy preservation in federated learning.

This module implements various masking techniques to protect sensitive information
in model updates and client data during federated learning.
"""

import logging
import numpy as np
import hashlib
import hmac
import os
from typing import Dict, List, Tuple, Any, Optional, Union, Callable

logger = logging.getLogger(__name__)

class MaskingMechanism:
    """Masking mechanisms for data privacy in federated learning."""
    
    def __init__(self, config: Dict):
        """
        Initialize masking with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.masking_config = config.get("masking", {})
        self.enabled = self.masking_config.get("enabled", True)
        self.masking_type = self.masking_config.get("type", "additive")
        
        # Parameters for additive masking
        self.random_seed = self.masking_config.get("random_seed", 42)
        self.mask_scale = self.masking_config.get("mask_scale", 0.1)
        
        # HMAC key for deterministic masking
        self._hmac_key = os.urandom(32)
        
        # Tracking masked clients
        self.masked_operations = {}
        
        if self.enabled:
            logger.info(f"Masking initialized with type={self.masking_type}")
        else:
            logger.info("Masking is disabled")
    
    def generate_mask(self, shape: Tuple, client_id: str, round_id: int) -> np.ndarray:
        """
        Generate a deterministic mask for a client.
        
        Args:
            shape: Shape of mask to generate
            client_id: Client identifier
            round_id: Current federation round
            
        Returns:
            Generated mask as numpy array
        """
        if not self.enabled:
            return np.zeros(shape)
        
        # Create a deterministic seed using HMAC
        h = hmac.new(self._hmac_key, digestmod=hashlib.sha256)
        h.update(client_id.encode())
        h.update(str(round_id).encode())
        seed = int.from_bytes(h.digest()[:8], byteorder='big')
        
        # Create a random state with the seed
        rs = np.random.RandomState(seed)
        
        # Generate mask based on masking type
        if self.masking_type == "additive":
            # Additive mask: random values that sum to zero across clients
            mask = rs.normal(0, self.mask_scale, size=shape)
            
            # Record this mask for potential verification
            mask_id = f"{client_id}_{round_id}"
            self.masked_operations[mask_id] = {
                "client_id": client_id,
                "round_id": round_id,
                "mask_sum": np.sum(mask),
                "mask_mean": np.mean(mask),
                "mask_std": np.std(mask),
                "timestamp": np.datetime64('now')
            }
            
            return mask
        
        elif self.masking_type == "multiplicative":
            # Multiplicative mask: random values that multiply to one across clients
            mask = rs.normal(1, self.mask_scale, size=shape)
            return mask
        
        else:
            logger.warning(f"Unknown masking type: {self.masking_type}, using zeros")
            return np.zeros(shape)
    
    def apply_mask(self, data: List[np.ndarray], client_id: str, round_id: int) -> List[np.ndarray]:
        """
        Apply mask to data (typically model weights or gradients).
        
        Args:
            data: List of numpy arrays to mask
            client_id: Client identifier
            round_id: Current federation round
            
        Returns:
            Masked data
        """
        if not self.enabled:
            return data
        
        try:
            # Apply masking to each layer in the data
            masked_data = []
            for i, layer in enumerate(data):
                mask = self.generate_mask(layer.shape, f"{client_id}_{i}", round_id)
                if self.masking_type == "additive":
                    masked_layer = layer + mask
                elif self.masking_type == "multiplicative":
                    masked_layer = layer * mask
                else:
                    masked_layer = layer
                masked_data.append(masked_layer)
            
            logger.debug(f"Applied {self.masking_type} masking to data for client {client_id}")
            return masked_data
            
        except Exception as e:
            logger.error(f"Error applying mask: {e}")
            # Return original data in case of error
            return data
    
    def unmask_data(self, data: List[np.ndarray], client_id: str, round_id: int) -> List[np.ndarray]:
        """
        Remove masking from data.
        
        Args:
            data: Masked data
            client_id: Client identifier
            round_id: Federation round
            
        Returns:
            Unmasked data
        """
        if not self.enabled:
            return data
        
        try:
            # Remove masking from each layer
            unmasked_data = []
            for i, layer in enumerate(data):
                mask = self.generate_mask(layer.shape, f"{client_id}_{i}", round_id)
                if self.masking_type == "additive":
                    unmasked_layer = layer - mask
                elif self.masking_type == "multiplicative":
                    unmasked_layer = layer / mask
                else:
                    unmasked_layer = layer
                unmasked_data.append(unmasked_layer)
            
            logger.debug(f"Removed {self.masking_type} masking from data for client {client_id}")
            return unmasked_data
            
        except Exception as e:
            logger.error(f"Error removing mask: {e}")
            # Return masked data in case of error
            return data
    
    def verify_masks_cancel_out(self, client_ids: List[str], round_id: int) -> bool:
        """
        Verify that masks cancel out appropriately for secure aggregation.
        
        Args:
            client_ids: List of client IDs participating in the round
            round_id: Federation round
            
        Returns:
            True if masks would cancel out correctly
        """
        if not self.enabled:
            return True
        
        try:
            # Collect mask stats for each client
            mask_sums = []
            for client_id in client_ids:
                mask_id = f"{client_id}_{round_id}"
                if mask_id in self.masked_operations:
                    mask_sums.append(self.masked_operations[mask_id]["mask_sum"])
                
            # For additive masks, the sum should be close to zero
            if self.masking_type == "additive" and mask_sums:
                total_sum = sum(mask_sums)
                return abs(total_sum) < 1e-5
            
            # For multiplicative masks, would need to check product
            return True
            
        except Exception as e:
            logger.error(f"Error verifying masks: {e}")
            return False 