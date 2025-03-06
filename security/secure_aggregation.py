"""
Secure aggregation mechanisms for federated learning.

This module implements secure aggregation protocols to protect
the privacy of client model updates during aggregation, including
secure sum and robust aggregation methods.
"""

import logging
import numpy as np
import hmac
import hashlib
import os
import time
from typing import Dict, List, Tuple, Any, Optional, Union, Callable

logger = logging.getLogger(__name__)

class SecureAggregation:
    """Secure aggregation mechanisms for federated learning."""
    
    def __init__(self, config: Dict):
        """
        Initialize secure aggregation with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.secure_agg_config = config.get("secure_aggregation", {})
        self.enabled = self.secure_agg_config.get("enabled", False)
        self.secure_agg_type = self.secure_agg_config.get("type", "secure_sum")
        
        # Parameters for robust aggregation
        self.outlier_threshold = self.secure_agg_config.get("outlier_threshold", 1.5)
        self.min_clients_for_aggregation = self.secure_agg_config.get("min_clients", 3)
        
        # For secure sum protocol
        self._secret_keys = {}  # Maps client_id to secret key for secure sum
        self._seed = os.urandom(32)  # Random seed for secure operations
        
        # For adaptive aggregation
        self.client_trust_scores = {}  # Maps client_id to trust score
        self.aggregation_history = []  # Tracks aggregation metrics for analysis
        
        if self.enabled:
            logger.info(f"Secure aggregation initialized with type={self.secure_agg_type}")
        else:
            logger.info("Secure aggregation is disabled")
    
    def aggregate(
        self, 
        updates: List[List[np.ndarray]], 
        weights: Optional[List[float]] = None,
        client_ids: Optional[List[str]] = None
    ) -> List[np.ndarray]:
        """
        Securely aggregate model updates.
        
        Args:
            updates: List of model updates from clients
            weights: Weights for each update (if None, equal weights are used)
            client_ids: Optional client identifiers for tracking
            
        Returns:
            Aggregated model update
        """
        if not self.enabled or not updates:
            # Fall back to regular weighted average if disabled or no updates
            return self._weighted_average(updates, weights)
        
        try:
            # Choose appropriate aggregation method
            if self.secure_agg_type == "secure_sum":
                return self._secure_sum(updates, weights, client_ids)
            elif self.secure_agg_type == "robust_aggregation":
                return self._robust_aggregation(updates, weights, client_ids)
            else:
                logger.warning(f"Unknown secure aggregation type: {self.secure_agg_type}, falling back to weighted average")
                return self._weighted_average(updates, weights)
        except Exception as e:
            logger.error(f"Error in secure aggregation: {e}, falling back to weighted average")
            return self._weighted_average(updates, weights)
    
    def _weighted_average(
        self, 
        updates: List[List[np.ndarray]], 
        weights: Optional[List[float]] = None
    ) -> List[np.ndarray]:
        """
        Compute weighted average of model updates.
        
        Args:
            updates: List of model updates from clients
            weights: Weights for each update (if None, equal weights are used)
            
        Returns:
            Weighted average of updates
        """
        # Ensure updates is not empty
        if not updates:
            raise ValueError("No updates to aggregate")
        
        # If weights are not provided, use equal weights
        if weights is None:
            weights = [1.0 / len(updates)] * len(updates)
        else:
            # Normalize weights to sum to 1
            weights = np.array(weights) / np.sum(weights)
        
        # Compute weighted average
        aggregated_update = []
        for i in range(len(updates[0])):
            # Initialize with zeros of the right shape
            layer_update = np.zeros_like(updates[0][i], dtype=np.float32)
            
            # Sum weighted updates
            for j, update in enumerate(updates):
                if update[i] is not None:  # Handle potential None values
                    layer_update += weights[j] * update[i]
            
            aggregated_update.append(layer_update)
        
        return aggregated_update
    
    def _secure_sum(
        self, 
        updates: List[List[np.ndarray]], 
        weights: Optional[List[float]] = None,
        client_ids: Optional[List[str]] = None
    ) -> List[np.ndarray]:
        """
        Secure sum protocol for aggregating model updates.
        
        This is a simplified implementation of secure sum. In a real implementation, 
        each client would add random noise before sending updates, and then the noise 
        would be removed during aggregation.
        
        Args:
            updates: List of model updates from clients
            weights: Weights for each update (if None, equal weights are used)
            client_ids: Optional client identifiers for secure operations
            
        Returns:
            Aggregated model update
        """
        # Use regular weighted average as base implementation
        aggregated_update = self._weighted_average(updates, weights)
        
        # In a real secure sum implementation, clients would add noise using a seed
        # derived from their identity and the round number, and then the server
        # would subtract this noise during aggregation
        
        # For demonstration purposes, we'll use HMAC to verify update integrity
        if client_ids and len(client_ids) == len(updates):
            # Generate or retrieve secret keys for each client
            for client_id in client_ids:
                if client_id not in self._secret_keys:
                    # Generate a unique secret key for this client
                    self._secret_keys[client_id] = os.urandom(32)
            
            # Verify integrity of updates (in a real system, this would be done before aggregation)
            for i, client_id in enumerate(client_ids):
                # Compute HMAC of the update using client's secret key
                h = hmac.new(self._secret_keys[client_id], digestmod=hashlib.sha256)
                
                # Add each layer to the HMAC
                for layer in updates[i]:
                    if layer is not None:
                        h.update(layer.tobytes())
                
                # Store the HMAC digest for verification (in a real system, the client would send this)
                update_hmac = h.digest()
                
                # In a real system, we would verify this HMAC against what the client sends
                # If verification fails, we might exclude the client's update
                
                # For this example, we're just logging that we computed the HMAC
                logger.debug(f"Computed HMAC for client {client_id}: {update_hmac.hex()[:8]}...")
        
        # Track aggregation for analysis
        self._track_aggregation("secure_sum", len(updates), weights)
        
        return aggregated_update
    
    def _robust_aggregation(
        self, 
        updates: List[List[np.ndarray]], 
        weights: Optional[List[float]] = None,
        client_ids: Optional[List[str]] = None
    ) -> List[np.ndarray]:
        """
        Robust aggregation protocol for aggregating model updates,
        filtering potential poisoning attacks.
        
        Args:
            updates: List of model updates from clients
            weights: Weights for each update (if None, equal weights are used)
            client_ids: Optional client identifiers for tracking
            
        Returns:
            Aggregated model update
        """
        # Ensure we have enough updates for robust aggregation
        if len(updates) < self.min_clients_for_aggregation:
            logger.warning(f"Not enough clients for robust aggregation. Using weighted average instead.")
            return self._weighted_average(updates, weights)
        
        # If weights are not provided, use equal weights
        if weights is None:
            weights = [1.0 / len(updates)] * len(updates)
        else:
            # Normalize weights to sum to 1
            weights = np.array(weights) / np.sum(weights)
        
        # Detect and remove outliers
        filtered_updates = []
        filtered_weights = []
        filtered_client_ids = []
        excluded_clients = []
        
        # Calculate trust scores if client_ids are provided
        if client_ids and len(client_ids) == len(updates):
            # Initialize trust scores for new clients
            for client_id in client_ids:
                if client_id not in self.client_trust_scores:
                    self.client_trust_scores[client_id] = 1.0  # Initial trust score
        
        # Calculate norm of each update for outlier detection
        update_norms = []
        for i, update in enumerate(updates):
            # Calculate Frobenius norm across all layers
            norm_squared = 0
            for layer in update:
                if layer is not None:
                    norm_squared += np.sum(np.square(layer))
            norm = np.sqrt(norm_squared)
            update_norms.append(norm)
        
        # Calculate statistical properties for outlier detection
        median_norm = np.median(update_norms)
        q1 = np.percentile(update_norms, 25)
        q3 = np.percentile(update_norms, 75)
        iqr = q3 - q1
        
        # Define bounds for outlier detection
        lower_bound = q1 - self.outlier_threshold * iqr
        upper_bound = q3 + self.outlier_threshold * iqr
        
        # Filter updates based on outlier detection
        for i, update in enumerate(updates):
            norm = update_norms[i]
            
            # Check if update is an outlier
            is_outlier = norm < lower_bound or norm > upper_bound
            
            # If client trust scores are available, use them for additional filtering
            client_id = client_ids[i] if client_ids and i < len(client_ids) else None
            trust_score = self.client_trust_scores.get(client_id, 1.0) if client_id else 1.0
            
            # Adjust outlier decision based on trust score
            if is_outlier and trust_score > 0.8:
                # Give benefit of doubt to highly trusted clients
                is_outlier = False
                logger.debug(f"Giving benefit of doubt to trusted client {client_id} with trust score {trust_score:.2f}")
            elif not is_outlier and trust_score < 0.3:
                # Be more suspicious of low-trust clients
                is_outlier = True
                logger.debug(f"Excluding low-trust client {client_id} with trust score {trust_score:.2f}")
            
            if not is_outlier:
                filtered_updates.append(update)
                filtered_weights.append(weights[i])
                if client_ids and i < len(client_ids):
                    filtered_client_ids.append(client_ids[i])
            else:
                # Track excluded clients
                if client_ids and i < len(client_ids):
                    excluded_clients.append(client_ids[i])
                    # Reduce trust score for excluded client
                    if client_ids[i] in self.client_trust_scores:
                        self.client_trust_scores[client_ids[i]] *= 0.9  # Decay trust
        
        # If all updates are outliers, use the original updates
        if len(filtered_updates) == 0:
            logger.warning("All updates detected as outliers. Using original updates.")
            filtered_updates = updates
            filtered_weights = weights
            filtered_client_ids = client_ids if client_ids else []
        else:
            # Normalize weights to sum to 1
            filtered_weights = np.array(filtered_weights) / np.sum(filtered_weights)
            logger.info(f"Robust aggregation excluded {len(excluded_clients)} of {len(updates)} clients")
        
        # Update trust scores for included clients
        if filtered_client_ids:
            for client_id in filtered_client_ids:
                if client_id in self.client_trust_scores:
                    # Increase trust score for included client (with ceiling at 1.0)
                    self.client_trust_scores[client_id] = min(1.0, self.client_trust_scores[client_id] * 1.05)
        
        # Calculate aggregated update from filtered updates
        aggregated_update = []
        for i in range(len(filtered_updates[0])):
            # Initialize with zeros of the right shape
            layer_update = np.zeros_like(filtered_updates[0][i], dtype=np.float32)
            
            # Sum weighted updates
            for j, update in enumerate(filtered_updates):
                if update[i] is not None:  # Handle potential None values
                    layer_update += filtered_weights[j] * update[i]
            
            aggregated_update.append(layer_update)
        
        # Track aggregation for analysis
        self._track_aggregation("robust", len(updates), weights, len(filtered_updates))
        
        return aggregated_update
    
    def _cosine_similarity(self, update1: List[np.ndarray], update2: List[np.ndarray]) -> float:
        """
        Calculate cosine similarity between two updates.
        
        Args:
            update1: First update
            update2: Second update
            
        Returns:
            Cosine similarity value between -1 and 1
        """
        # Flatten both updates
        flat1 = np.concatenate([layer.flatten() for layer in update1 if layer is not None])
        flat2 = np.concatenate([layer.flatten() for layer in update2 if layer is not None])
        
        # Calculate dot product
        dot_product = np.dot(flat1, flat2)
        
        # Calculate magnitudes
        mag1 = np.sqrt(np.dot(flat1, flat1))
        mag2 = np.sqrt(np.dot(flat2, flat2))
        
        # Calculate cosine similarity
        if mag1 * mag2 > 0:
            return dot_product / (mag1 * mag2)
        else:
            return 0.0
    
    def _track_aggregation(
        self, 
        method: str, 
        total_clients: int, 
        weights: Optional[List[float]] = None,
        included_clients: Optional[int] = None
    ) -> None:
        """
        Track aggregation metrics for analysis.
        
        Args:
            method: Aggregation method used
            total_clients: Total number of clients
            weights: Weights used for aggregation
            included_clients: Number of clients included after filtering
        """
        self.aggregation_history.append({
            "timestamp": time.time(),
            "method": method,
            "total_clients": total_clients,
            "included_clients": included_clients if included_clients is not None else total_clients,
            "weight_distribution": weights.tolist() if weights is not None else None,
            "exclusion_rate": (total_clients - (included_clients if included_clients is not None else total_clients)) / total_clients if total_clients > 0 else 0
        })
        
        # Keep history size manageable
        if len(self.aggregation_history) > 100:
            self.aggregation_history = self.aggregation_history[-100:]
    
    def get_aggregation_metrics(self) -> Dict:
        """
        Get metrics about the secure aggregation process.
        
        Returns:
            Dictionary with aggregation metrics
        """
        if not self.aggregation_history:
            return {
                "enabled": self.enabled,
                "method": self.secure_agg_type,
                "total_aggregations": 0
            }
        
        # Calculate metrics
        total_aggregations = len(self.aggregation_history)
        total_clients = sum(agg["total_clients"] for agg in self.aggregation_history)
        included_clients = sum(agg["included_clients"] for agg in self.aggregation_history)
        avg_exclusion_rate = sum(agg["exclusion_rate"] for agg in self.aggregation_history) / total_aggregations
        
        return {
            "enabled": self.enabled,
            "method": self.secure_agg_type,
            "total_aggregations": total_aggregations,
            "avg_clients_per_round": total_clients / total_aggregations,
            "avg_included_clients": included_clients / total_aggregations,
            "avg_exclusion_rate": avg_exclusion_rate,
            "client_trust_scores": {k: round(v, 2) for k, v in self.client_trust_scores.items()}
        }
    
    def reset_client_trust_scores(self) -> None:
        """Reset all client trust scores to initial value."""
        self.client_trust_scores = {client_id: 1.0 for client_id in self.client_trust_scores}
        logger.info("Reset all client trust scores to initial value")