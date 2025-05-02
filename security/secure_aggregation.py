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
        self.secure_agg_config = config.get("security", {}).get("secure_aggregation", {})
        self.enabled = self.secure_agg_config.get("enabled", True)  # Enable by default
        self.secure_agg_type = self.secure_agg_config.get("type", "trimmed_mean")  # Use trimmed mean by default
        
        # Parameters for robust aggregation
        self.outlier_threshold = self.secure_agg_config.get("outlier_threshold", 1.5)
        self.min_clients_for_aggregation = self.secure_agg_config.get("min_clients", 3)
        self.trimming_percentage = self.secure_agg_config.get("trimming_percentage", 0.2)  # For trimmed mean
        
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
        if not updates:
            raise ValueError("No updates to aggregate")
        
        if not self.enabled:
            # Fall back to regular weighted average if disabled
            return self._weighted_average(updates, weights)
        
        try:
            # Choose appropriate aggregation method
            if self.secure_agg_type == "secure_sum":
                return self._secure_sum(updates, weights, client_ids)
            elif self.secure_agg_type == "robust_aggregation":
                return self._robust_aggregation(updates, weights, client_ids)
            elif self.secure_agg_type == "trimmed_mean":
                return self._trimmed_mean(updates, weights, client_ids)
            elif self.secure_agg_type == "median":
                return self._median(updates, client_ids)
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

    def _trimmed_mean(
        self,
        updates: List[List[np.ndarray]],
        weights: Optional[List[float]] = None,
        client_ids: Optional[List[str]] = None
    ) -> List[np.ndarray]:
        """
        Implement Trimmed Mean aggregation - trim a percentage of the highest and 
        lowest values for each parameter before averaging.
        
        Args:
            updates: List of model updates from clients
            weights: Weights for each update (if None, equal weights are used)
            client_ids: Optional client identifiers for tracking
            
        Returns:
            Aggregated model update using trimmed mean
        """
        logger.info(f"Applying trimmed mean aggregation with {self.trimming_percentage*100}% trimming")
        
        # Ensure we have enough updates for meaningful trimming
        if len(updates) < self.min_clients_for_aggregation:
            logger.warning(f"Not enough clients for trimmed mean. Using weighted average instead.")
            return self._weighted_average(updates, weights)
        
        # If weights are provided, use them to replicate updates before trimming
        # This is a weighted trimmed mean approach
        effective_updates = updates
        if weights is not None:
            # Normalize weights to sum to 1 and convert to integer counts
            normalized_weights = np.array(weights) / np.sum(weights)
            total_replicas = max(100, len(updates) * 10)  # Use at least 100 replicas for granularity
            counts = (normalized_weights * total_replicas).astype(int)
            
            # Ensure we have at least one replica per update
            for i in range(len(counts)):
                if counts[i] == 0:
                    counts[i] = 1
            
            # Create replicated updates based on weights
            effective_updates = []
            for i, update in enumerate(updates):
                for _ in range(counts[i]):
                    effective_updates.append(update)
        
        # Calculate trim count based on percentage
        n_updates = len(effective_updates)
        trim_count = int(n_updates * self.trimming_percentage)
        
        # Prepare the result container
        result = []
        
        # Process each layer separately
        for i in range(len(updates[0])):
            # Extract this layer from all updates
            layer_updates = [update[i] for update in effective_updates]
            
            # Get the shape
            layer_shape = layer_updates[0].shape
            
            # Flatten the arrays for element-wise operations
            flattened_updates = [layer.flatten() for layer in layer_updates]
            
            # Stack the flattened arrays for sorting
            stacked_updates = np.stack(flattened_updates)
            
            # Sort the values for each parameter position
            sorted_values = np.sort(stacked_updates, axis=0)
            
            # Compute trimmed mean - remove trim_count elements from each end
            if trim_count > 0:
                trimmed_values = sorted_values[trim_count:-trim_count]
            else:
                trimmed_values = sorted_values
            
            # Compute mean of the remaining values
            mean_values = np.mean(trimmed_values, axis=0)
            
            # Reshape back to original shape
            layer_result = mean_values.reshape(layer_shape)
            
            # Add to result
            result.append(layer_result)
        
        # Track aggregation metrics
        self._track_aggregation("trimmed_mean", len(updates), weights, n_updates - 2 * trim_count)
        
        return result

    def _median(
        self,
        updates: List[List[np.ndarray]],
        client_ids: Optional[List[str]] = None
    ) -> List[np.ndarray]:
        """
        Implement Median aggregation - use the median value for each parameter
        across all client updates.
        
        Args:
            updates: List of model updates from clients
            client_ids: Optional client identifiers for tracking
            
        Returns:
            Aggregated model update using coordinate-wise median
        """
        logger.info("Applying coordinate-wise median aggregation")
        
        # Ensure we have enough updates for a meaningful median
        if len(updates) < self.min_clients_for_aggregation:
            logger.warning(f"Not enough clients for median. Using weighted average instead.")
            return self._weighted_average(updates)
        
        # Prepare the result container
        result = []
        
        # Process each layer separately
        for i in range(len(updates[0])):
            # Extract this layer from all updates
            layer_updates = [update[i] for update in updates]
            
            # Get the shape
            layer_shape = layer_updates[0].shape
            
            # Flatten the arrays for element-wise operations
            flattened_updates = [layer.flatten() for layer in layer_updates]
            
            # Stack the flattened arrays 
            stacked_updates = np.stack(flattened_updates)
            
            # Compute median for each parameter position
            median_values = np.median(stacked_updates, axis=0)
            
            # Reshape back to original shape
            layer_result = median_values.reshape(layer_shape)
            
            # Add to result
            result.append(layer_result)
        
        # Track aggregation metrics
        self._track_aggregation("median", len(updates), None, len(updates))
        
        return result

def clipping(
    weights: np.ndarray, 
    clip_threshold: float = 3.0
) -> np.ndarray:
    """
    Apply clipping to weights to remove extreme outliers.
    
    Args:
        weights: Numpy array of weights
        clip_threshold: Threshold for clipping in terms of std deviations from mean
        
    Returns:
        Clipped weights
    """
    mean = np.mean(weights)
    std = np.std(weights)
    
    # Calculate clipping bounds
    lower_bound = mean - clip_threshold * std
    upper_bound = mean + clip_threshold * std
    
    # Apply clipping
    clipped_weights = np.clip(weights, lower_bound, upper_bound)
    
    # Report percentage of values clipped
    total_values = weights.size
    clipped_values = np.sum((weights < lower_bound) | (weights > upper_bound))
    clipping_percentage = (clipped_values / total_values) * 100 if total_values > 0 else 0
    
    logger.info(f"Clipped {clipped_values}/{total_values} values ({clipping_percentage:.2f}%)")
    
    return clipped_weights

class RobustAggregation:
    """
    Robust aggregation methods for federated learning with Paillier encryption.
    
    This class provides methods optimized for handling arrays of decrypted values
    from Paillier homomorphic encryption, which might be flattened model updates.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize robust aggregation with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.secure_agg_config = config.get("security", {}).get("secure_aggregation", {})
        self.enabled = self.secure_agg_config.get("enabled", True)
        self.secure_agg_type = self.secure_agg_config.get("type", "trimmed_mean")
        
        # Parameters for robust aggregation
        self.outlier_threshold = self.secure_agg_config.get("outlier_threshold", 1.5)
        self.min_clients_for_aggregation = self.secure_agg_config.get("min_clients", 3)
        self.trimming_percentage = self.secure_agg_config.get("trimming_percentage", 0.2)
        
        # For adaptive aggregation
        self.client_trust_scores = {}
        self.aggregation_history = []
        
        if self.enabled:
            logger.info(f"Robust aggregation initialized with type={self.secure_agg_type}")
        else:
            logger.info("Robust aggregation is disabled")
    
    def aggregate_flatten_weights(
        self, 
        updates: List[np.ndarray]
    ) -> np.ndarray:
        """
        Robustly aggregate flattened weight arrays.
        
        Args:
            updates: List of numpy arrays representing model updates
            
        Returns:
            Robustly aggregated numpy array
        """
        if not updates:
            raise ValueError("No updates to aggregate")
        
        if not self.enabled:
            return np.mean(updates, axis=0)
        
        try:
            # Choose appropriate aggregation method
            if self.secure_agg_type == "trimmed_mean":
                return self._trimmed_mean_flattened(updates)
            elif self.secure_agg_type == "median":
                return self._median_flattened(updates)
            elif self.secure_agg_type == "krum":
                return self._krum_flattened(updates)
            else:
                logger.warning(f"Unknown robust aggregation type: {self.secure_agg_type}, falling back to mean")
                return np.mean(updates, axis=0)
        except Exception as e:
            logger.error(f"Error in robust aggregation: {e}, falling back to mean")
            return np.mean(updates, axis=0)
    
    def _trimmed_mean_flattened(
        self,
        updates: List[np.ndarray]
    ) -> np.ndarray:
        """
        Compute trimmed mean of flattened updates.
        
        Args:
            updates: List of numpy arrays representing model updates
            
        Returns:
            Trimmed mean as numpy array
        """
        # Ensure we have enough updates for trimmed mean
        if len(updates) < 3:  # Need at least 3 for meaningful trimming
            logger.warning("Not enough updates for trimmed mean, using regular mean.")
            return np.mean(updates, axis=0)
        
        # Convert to numpy array for easier processing
        updates_array = np.array(updates)
        
        # Calculate how many updates to trim from each end
        trim_count = int(len(updates) * self.trimming_percentage)
        if trim_count == 0 and len(updates) > 3:
            trim_count = 1  # Trim at least one if we have more than 3 updates
        
        logger.info(f"Trimming {trim_count} updates from each end of {len(updates)} total updates")
        
        # Sort along first axis (the list of updates)
        sorted_indices = np.argsort(updates_array, axis=0)
        
        # Create a mask for values to keep
        mask = np.ones_like(sorted_indices, dtype=bool)
        
        # Mark values to trim
        for i in range(trim_count):
            # Mark smallest values
            min_indices = sorted_indices[i]
            mask[i, np.arange(len(min_indices)), min_indices] = False
            
            # Mark largest values
            max_indices = sorted_indices[-(i+1)]
            mask[-(i+1), np.arange(len(max_indices)), max_indices] = False
        
        # Apply mask and compute mean of remaining values
        trimmed_updates = updates_array[mask].reshape(len(updates) - 2*trim_count, -1)
        
        return np.mean(trimmed_updates, axis=0)
    
    def _median_flattened(
        self,
        updates: List[np.ndarray]
    ) -> np.ndarray:
        """
        Compute coordinate-wise median of flattened updates.
        
        Args:
            updates: List of numpy arrays representing model updates
            
        Returns:
            Median as numpy array
        """
        # Convert to numpy array for easier processing
        updates_array = np.array(updates)
        
        # Apply median along the first axis (the list of updates)
        return np.median(updates_array, axis=0)
    
    def _krum_flattened(
        self,
        updates: List[np.ndarray],
        num_byzantine: Optional[int] = None
    ) -> np.ndarray:
        """
        Implement Krum algorithm for byzantine-robust aggregation.
        
        Krum selects the update that is closest to its neighbors in Euclidean distance.
        
        Args:
            updates: List of numpy arrays representing model updates
            num_byzantine: Number of potentially byzantine clients (defaults to n/4)
            
        Returns:
            Selected update as numpy array
        """
        num_updates = len(updates)
        
        if num_updates < 4:  # Need at least 4 for meaningful Krum
            logger.warning("Not enough updates for Krum, using mean.")
            return np.mean(updates, axis=0)
        
        # Default: assume up to n/4 byzantine clients
        if num_byzantine is None:
            num_byzantine = max(1, num_updates // 4)
        
        # Krum requires n >= 2f + 3 where f is num_byzantine
        if num_updates < 2 * num_byzantine + 3:
            logger.warning(f"Not enough honest clients for Krum with {num_byzantine} Byzantine. Using mean.")
            return np.mean(updates, axis=0)
        
        # Calculate pairwise squared distances
        distances = np.zeros((num_updates, num_updates))
        
        for i in range(num_updates):
            for j in range(i+1, num_updates):
                # Calculate squared Euclidean distance
                dist = np.sum((updates[i] - updates[j]) ** 2)
                distances[i, j] = dist
                distances[j, i] = dist
        
        # For each update, calculate the sum of distances to the closest n-f-2 updates
        scores = np.zeros(num_updates)
        
        for i in range(num_updates):
            # Sort distances for this update
            sorted_dists = np.sort(distances[i])
            # Sum the closest n-f-2 distances (excluding distance to self which is 0)
            scores[i] = np.sum(sorted_dists[1:num_updates - num_byzantine - 1])
        
        # Select the update with the minimum score
        best_idx = np.argmin(scores)
        
        logger.info(f"Krum selected update {best_idx} out of {num_updates} with {num_byzantine} assumed Byzantine")
        
        return updates[best_idx]