"""
Threat model and defenses for federated learning.

This module implements threat detection and prevention mechanisms
for protecting federated learning against various attacks, including
poisoning attacks, adversarial examples, and free-riding.
"""

import logging
import numpy as np
import time
from typing import Dict, List, Tuple, Any, Optional, Union
from collections import defaultdict
import pickle
import hashlib

logger = logging.getLogger(__name__)

class ThreatModel:
    """Threat model and defenses for federated learning."""
    
    def __init__(self, config: Dict):
        """
        Initialize threat model with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.threat_model_config = config.get("threat_model", {})
        self.enabled = self.threat_model_config.get("enabled", True)
        self.poisoning_defense = self.threat_model_config.get("poisoning_defense", True)
        self.adversarial_defense = self.threat_model_config.get("adversarial_defense", True)
        self.freerider_detection = self.threat_model_config.get("freerider_detection", True)
        
        # Parameters for poisoning detection
        self.poisoning_threshold = self.threat_model_config.get("poisoning_threshold", 0.8)
        self.poisoning_window = self.threat_model_config.get("poisoning_window", 5)
        
        # Parameters for adversarial example detection
        self.adversarial_threshold = self.threat_model_config.get("adversarial_threshold", 0.9)
        
        # Parameters for free-rider detection
        self.freerider_threshold = self.threat_model_config.get("freerider_threshold", 0.05)
        self.freerider_similarity_threshold = self.threat_model_config.get("freerider_similarity_threshold", 0.9)
        
        # Metrics for tracking threats
        self.client_metrics = defaultdict(list)  # Track metrics for each client
        self.flagged_clients = set()  # Set of flagged client IDs
        self.threat_detections = []  # List of threat detections
        
        # Store reference model updates for comparison
        self.global_update_history = []  # History of global model updates
        self.client_update_hashes = {}  # Maps client_id to hashes of updates
        
        if self.enabled:
            defense_methods = []
            if self.poisoning_defense:
                defense_methods.append("poisoning defense")
            if self.adversarial_defense:
                defense_methods.append("adversarial defense")
            if self.freerider_detection:
                defense_methods.append("free-rider detection")
            
            logger.info(f"Threat model initialized with {', '.join(defense_methods)}")
        else:
            logger.info("Threat model is disabled")
    
    def detect_poisoning(
        self, 
        updates: List[List[np.ndarray]],
        client_ids: List[str],
        global_model: Optional[List[np.ndarray]] = None
    ) -> List[bool]:
        """
        Detect poisoning attacks in client updates.
        
        Args:
            updates: List of model updates from clients
            client_ids: List of client IDs corresponding to updates
            global_model: Current global model parameters
            
        Returns:
            List of booleans indicating whether each update is poisoned
        """
        if not self.enabled or not self.poisoning_defense:
            return [False] * len(updates)
        
        if len(updates) == 0:
            return []
        
        try:
            # Initialize result
            is_poisoned = [False] * len(updates)
            
            # Calculate the mean and standard deviation of each parameter across clients
            param_stats = []
            for layer_idx in range(len(updates[0])):
                layer_updates = []
                for client_update in updates:
                    if client_update[layer_idx] is not None:
                        layer_updates.append(client_update[layer_idx])
                
                if layer_updates:
                    # Stack along a new axis to calculate stats
                    layer_stack = np.stack(layer_updates)
                    layer_mean = np.mean(layer_stack, axis=0)
                    layer_std = np.std(layer_stack, axis=0)
                    param_stats.append((layer_mean, layer_std))
                else:
                    param_stats.append((None, None))
            
            # Calculate z-scores for each client update
            for client_idx, (client_update, client_id) in enumerate(zip(updates, client_ids)):
                # Calculate the deviation from mean for this client
                z_scores = []
                for layer_idx, layer in enumerate(client_update):
                    if layer is not None and param_stats[layer_idx][0] is not None:
                        # Calculate z-score
                        layer_mean, layer_std = param_stats[layer_idx]
                        # Add small epsilon to avoid division by zero
                        eps = 1e-8
                        z_score = np.abs((layer - layer_mean) / (layer_std + eps))
                        
                        # Calculate mean z-score for this layer
                        mean_z_score = np.mean(z_score)
                        z_scores.append(mean_z_score)
                
                # Calculate overall z-score if we have values
                if z_scores:
                    overall_z_score = np.mean(z_scores)
                    
                    # Track metrics for this client
                    self.client_metrics[client_id].append({
                        "timestamp": time.time(),
                        "z_score": float(overall_z_score),
                        "type": "poisoning"
                    })
                    
                    # Keep metrics history manageable
                    if len(self.client_metrics[client_id]) > 20:
                        self.client_metrics[client_id] = self.client_metrics[client_id][-20:]
                    
                    # Check if z-score exceeds threshold
                    if overall_z_score > self.poisoning_threshold:
                        is_poisoned[client_idx] = True
                        
                        # Flag client
                        self.flagged_clients.add(client_id)
                        
                        # Record detection
                        self.threat_detections.append({
                            "timestamp": time.time(),
                            "client_id": client_id,
                            "type": "poisoning",
                            "z_score": float(overall_z_score),
                            "threshold": self.poisoning_threshold
                        })
                        
                        logger.warning(f"Potential poisoning attack detected from client {client_id} "
                                      f"with z-score {overall_z_score:.4f} (threshold: {self.poisoning_threshold})")
            
            return is_poisoned
            
        except Exception as e:
            logger.error(f"Error detecting poisoning attacks: {e}")
            return [False] * len(updates)
    
    def detect_freeriding(
        self, 
        updates: List[List[np.ndarray]],
        client_ids: List[str],
        global_model: Optional[List[np.ndarray]] = None
    ) -> List[bool]:
        """
        Detect free-riding clients (clients not doing real work).
        
        Args:
            updates: List of model updates from clients
            client_ids: List of client IDs corresponding to updates
            global_model: Current global model parameters
            
        Returns:
            List of booleans indicating whether each client is free-riding
        """
        if not self.enabled or not self.freerider_detection:
            return [False] * len(updates)
        
        if len(updates) == 0:
            return []
        
        try:
            # Initialize result
            is_freeriding = [False] * len(updates)
            
            # Store current global model in history
            if global_model is not None:
                self.global_update_history.append(global_model)
                # Keep history manageable
                if len(self.global_update_history) > 5:
                    self.global_update_history = self.global_update_history[-5:]
            
            # For each client, check update magnitude and similarity to global updates
            for client_idx, (client_update, client_id) in enumerate(zip(updates, client_ids)):
                # Calculate update magnitude
                update_magnitude = 0.0
                for layer in client_update:
                    if layer is not None:
                        update_magnitude += np.sum(np.square(layer))
                update_magnitude = np.sqrt(update_magnitude)
                
                # Check update magnitude - very small updates might indicate free-riding
                if update_magnitude < self.freerider_threshold:
                    is_freeriding[client_idx] = True
                    self.flagged_clients.add(client_id)
                    
                    self.threat_detections.append({
                        "timestamp": time.time(),
                        "client_id": client_id,
                        "type": "freeriding-small",
                        "magnitude": float(update_magnitude),
                        "threshold": self.freerider_threshold
                    })
                    
                    logger.warning(f"Potential free-riding detected from client {client_id} "
                                  f"with small update magnitude {update_magnitude:.4f} "
                                  f"(threshold: {self.freerider_threshold})")
                
                # Check similarity to previous global updates
                if self.global_update_history:
                    for prev_global_update in self.global_update_history:
                        similarity = self._calculate_similarity(client_update, prev_global_update)
                        
                        if similarity > self.freerider_similarity_threshold:
                            is_freeriding[client_idx] = True
                            self.flagged_clients.add(client_id)
                            
                            self.threat_detections.append({
                                "timestamp": time.time(),
                                "client_id": client_id,
                                "type": "freeriding-similarity",
                                "similarity": float(similarity),
                                "threshold": self.freerider_similarity_threshold
                            })
                            
                            logger.warning(f"Potential free-riding detected from client {client_id} "
                                          f"with high similarity to global model {similarity:.4f} "
                                          f"(threshold: {self.freerider_similarity_threshold})")
                            
                            # Skip checking other global updates
                            break
                
                # Generate hash of update to track duplicates across rounds
                update_hash = self._hash_update(client_update)
                
                # Check if client has submitted same update before
                if client_id in self.client_update_hashes:
                    if update_hash in self.client_update_hashes[client_id]:
                        is_freeriding[client_idx] = True
                        self.flagged_clients.add(client_id)
                        
                        self.threat_detections.append({
                            "timestamp": time.time(),
                            "client_id": client_id,
                            "type": "freeriding-duplicate",
                            "hash": update_hash[:8]  # First 8 chars for brevity
                        })
                        
                        logger.warning(f"Potential free-riding detected from client {client_id} "
                                      f"with duplicate update (hash: {update_hash[:8]})")
                
                # Store hash for future rounds
                if client_id not in self.client_update_hashes:
                    self.client_update_hashes[client_id] = []
                self.client_update_hashes[client_id].append(update_hash)
                
                # Keep hash history manageable
                if len(self.client_update_hashes[client_id]) > 5:
                    self.client_update_hashes[client_id] = self.client_update_hashes[client_id][-5:]
            
            return is_freeriding
            
        except Exception as e:
            logger.error(f"Error detecting free-riding: {e}")
            return [False] * len(updates)
    
    def detect_adversarial_examples(
        self, 
        examples: List[np.ndarray],
        labels: List[int],
        model: Any,
        client_id: str
    ) -> List[bool]:
        """
        Detect adversarial examples in evaluation data.
        
        Args:
            examples: List of examples to evaluate
            labels: List of labels for the examples
            model: Model to use for detection
            client_id: ID of the client submitting the examples
            
        Returns:
            List of booleans indicating whether each example is adversarial
        """
        if not self.enabled or not self.adversarial_defense:
            return [False] * len(examples)
        
        if len(examples) == 0:
            return []
        
        try:
            # Initialize result
            is_adversarial = [False] * len(examples)
            
            # Skip if model can't predict
            if not hasattr(model, 'predict'):
                return is_adversarial
            
            # Get predictions
            predictions = model.predict(np.array(examples))
            
            # For each example, check if it's adversarial
            for i, (example, label, prediction) in enumerate(zip(examples, labels, predictions)):
                # For binary classification
                if len(prediction.shape) == 1 and prediction.shape[0] == 1:
                    prediction_binary = (prediction > 0.5).astype(int)
                    confidence = max(prediction, 1 - prediction)
                # For multi-class classification
                elif len(prediction.shape) == 1 and prediction.shape[0] > 1:
                    prediction_binary = np.argmax(prediction)
                    confidence = prediction[prediction_binary]
                else:
                    # Skip if prediction format is unknown
                    continue
                
                # Check confidence - high confidence but wrong prediction might be adversarial
                if prediction_binary != label and confidence > self.adversarial_threshold:
                    is_adversarial[i] = True
                    
                    # Only log once per batch to avoid flooding
                    if i == 0:
                        self.threat_detections.append({
                            "timestamp": time.time(),
                            "client_id": client_id,
                            "type": "adversarial",
                            "confidence": float(confidence),
                            "threshold": self.adversarial_threshold
                        })
                        
                        logger.warning(f"Potential adversarial example detected from client {client_id} "
                                      f"with confidence {confidence:.4f} (threshold: {self.adversarial_threshold})")
            
            return is_adversarial
            
        except Exception as e:
            logger.error(f"Error detecting adversarial examples: {e}")
            return [False] * len(examples)
    
    def _calculate_similarity(self, update1: List[np.ndarray], update2: List[np.ndarray]) -> float:
        """
        Calculate cosine similarity between two updates.
        
        Args:
            update1: First update
            update2: Second update
            
        Returns:
            Cosine similarity value between -1 and 1
        """
        # Convert updates to flat vectors
        flat_update1 = []
        flat_update2 = []
        
        for layer1, layer2 in zip(update1, update2):
            if layer1 is not None and layer2 is not None:
                flat_update1.append(layer1.flatten())
                flat_update2.append(layer2.flatten())
        
        # Concatenate all layers
        if flat_update1 and flat_update2:
            flat_update1 = np.concatenate(flat_update1)
            flat_update2 = np.concatenate(flat_update2)
            
            # Calculate dot product
            dot_product = np.dot(flat_update1, flat_update2)
            
            # Calculate magnitudes
            magnitude1 = np.linalg.norm(flat_update1)
            magnitude2 = np.linalg.norm(flat_update2)
            
            # Calculate cosine similarity
            if magnitude1 > 0 and magnitude2 > 0:
                return dot_product / (magnitude1 * magnitude2)
        
        return 0.0
    
    def _hash_update(self, update: List[np.ndarray]) -> str:
        """
        Create a hash of a model update for duplicate detection.
        
        Args:
            update: List of model update layers
            
        Returns:
            String hash of the update
        """
        # Serialize update
        serialized = pickle.dumps([layer.tobytes() if layer is not None else None for layer in update])
        
        # Create hash
        return hashlib.sha256(serialized).hexdigest()
    
    def get_flagged_clients(self) -> List[str]:
        """
        Get list of clients that have been flagged for threats.
        
        Returns:
            List of flagged client IDs
        """
        return list(self.flagged_clients)
    
    def get_client_threat_score(self, client_id: str) -> float:
        """
        Get threat score for a client based on historical metrics.
        
        Args:
            client_id: Client ID
            
        Returns:
            Threat score between 0 and 1 (higher is more suspicious)
        """
        # Return 0 if no metrics for client or client not flagged
        if client_id not in self.client_metrics:
            return 0.0
        
        # Higher score if client has been flagged
        flagged_score = 0.5 if client_id in self.flagged_clients else 0.0
        
        # Calculate average z-score for poisoning metrics
        poisoning_metrics = [m for m in self.client_metrics[client_id] if m["type"] == "poisoning"]
        if poisoning_metrics:
            avg_z_score = sum(m["z_score"] for m in poisoning_metrics) / len(poisoning_metrics)
            # Normalize to 0-0.5 range (will be added to flagged_score)
            poisoning_score = min(0.5, avg_z_score / (2 * self.poisoning_threshold))
        else:
            poisoning_score = 0.0
        
        # Calculate overall threat score
        threat_score = min(1.0, flagged_score + poisoning_score)
        
        return threat_score
    
    def get_threat_metrics(self) -> Dict:
        """
        Get metrics about detected threats.
        
        Returns:
            Dictionary with threat metrics
        """
        return {
            "enabled": self.enabled,
            "total_flagged_clients": len(self.flagged_clients),
            "recent_detections": self.threat_detections[-10:] if self.threat_detections else [],
            "detection_types": {
                "poisoning": sum(1 for d in self.threat_detections if d["type"] == "poisoning"),
                "freeriding": sum(1 for d in self.threat_detections 
                                if d["type"].startswith("freeriding")),
                "adversarial": sum(1 for d in self.threat_detections if d["type"] == "adversarial")
            }
        }
    
    def reset_flagged_clients(self) -> None:
        """Reset the list of flagged clients."""
        self.flagged_clients = set()
        logger.info("Reset flagged clients list")