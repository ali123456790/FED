"""
Security manager for coordinating security components in federated learning.

This module provides a centralized manager for coordinating the various
security components of the FIDS system, including differential privacy,
encryption, secure aggregation, and threat detection.
"""

import logging
import os
import time
import json
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
from pathlib import Path

from .differential_privacy import DifferentialPrivacy
from .encryption import Encryption
from .secure_aggregation import SecureAggregation
from .threat_model import ThreatModel

logger = logging.getLogger(__name__)

class SecurityManager:
    """
    Central manager for all security components in the FIDS system.
    
    This class coordinates the various security components, providing
    a unified interface for the rest of the system and ensuring that
    security mechanisms are applied consistently.
    """
    
    def __init__(self, config: Dict, metrics_dir: Optional[str] = None):
        """
        Initialize the security manager with configuration.
        
        Args:
            config: Configuration dictionary
            metrics_dir: Directory to store security metrics
        """
        self.config = config
        self.security_config = config.get("security", {})
        self.metrics_dir = metrics_dir or "./security_metrics"
        
        # Create metrics directory
        Path(self.metrics_dir).mkdir(exist_ok=True, parents=True)
        
        # Initialize security components
        self.differential_privacy = DifferentialPrivacy(config)
        self.encryption = Encryption(config)
        self.secure_aggregation = SecureAggregation(config)
        self.threat_model = ThreatModel(config)
        
        # Track banned clients
        self.banned_clients = set()
        
        # Metrics tracking
        self.security_events = []
        self.client_security_metrics = {}
        
        logger.info("Security manager initialized")
    
    def process_client_update(
        self, 
        client_id: str, 
        update: List[np.ndarray], 
        num_samples: int, 
        metadata: Optional[Dict] = None
    ) -> Tuple[List[np.ndarray], bool, Dict]:
        """
        Process a model update from a client, applying security measures.
        
        Args:
            client_id: Client identifier
            update: Model update from client
            num_samples: Number of samples used for training
            metadata: Additional metadata about the update
            
        Returns:
            Tuple of (processed_update, is_accepted, security_metrics)
        """
        # Start timing
        start_time = time.time()
        
        # Initialize security metrics
        security_metrics = {
            "client_id": client_id,
            "timestamp": time.time(),
            "processed": False,
            "dp_applied": False,
            "encryption_status": "none",
            "threats_detected": False,
            "processing_time": 0
        }
        
        # Check if client is banned
        if client_id in self.banned_clients:
            logger.warning(f"Rejected update from banned client {client_id}")
            security_metrics["rejected_reason"] = "banned"
            
            # Log security event
            self._log_security_event("update_rejected", client_id, "Client is banned")
            
            return update, False, security_metrics
        
        # Process the update through each security component
        try:
            processed_update = update
            
            # Apply differential privacy to the update
            if self.differential_privacy.enabled:
                processed_update = self.differential_privacy.apply_dp_to_gradients(processed_update, client_id)
                security_metrics["dp_applied"] = True
                
                # Calculate privacy budget
                if metadata and "batch_size" in metadata and "epochs" in metadata:
                    epsilon, delta = self.differential_privacy.compute_privacy_budget(
                        num_samples=num_samples,
                        batch_size=metadata["batch_size"],
                        epochs=metadata["epochs"]
                    )
                    security_metrics["dp_epsilon"] = float(epsilon)
                    security_metrics["dp_delta"] = float(delta)
            
            # Track client security metrics
            self._update_client_security_metrics(client_id, security_metrics)
            
            # Update processing status
            security_metrics["processed"] = True
            security_metrics["processing_time"] = time.time() - start_time
            
            return processed_update, True, security_metrics
            
        except Exception as e:
            logger.error(f"Error processing client update: {e}")
            security_metrics["error"] = str(e)
            security_metrics["processing_time"] = time.time() - start_time
            
            # Log security event
            self._log_security_event("update_processing_error", client_id, str(e))
            
            return update, False, security_metrics
    
    def aggregate_updates(
        self, 
        updates: List[List[np.ndarray]], 
        client_ids: List[str],
        weights: Optional[List[float]] = None,
        global_model: Optional[List[np.ndarray]] = None,
        round_num: Optional[int] = None
    ) -> Tuple[List[np.ndarray], Dict]:
        """
        Securely aggregate model updates from multiple clients.
        
        Args:
            updates: List of model updates from clients
            client_ids: List of client IDs corresponding to updates
            weights: Weights for each update (if None, equal weights are used)
            global_model: Current global model parameters
            round_num: Current round number
            
        Returns:
            Tuple of (aggregated_update, security_metrics)
        """
        # Start timing
        start_time = time.time()
        
        # Initialize security metrics
        security_metrics = {
            "timestamp": time.time(),
            "round": round_num,
            "num_clients": len(updates),
            "secure_aggregation_applied": False,
            "threats_detected": False,
            "num_threats": 0,
            "processing_time": 0
        }
        
        try:
            # Detect potential poisoning attacks
            poisoning_detected = self.threat_model.detect_poisoning(updates, client_ids, global_model)
            
            # Detect free-riders
            freeriding_detected = self.threat_model.detect_freeriding(updates, client_ids, global_model)
            
            # Combine threat detections and filter updates
            threat_detected = [p or f for p, f in zip(poisoning_detected, freeriding_detected)]
            security_metrics["num_threats"] = sum(threat_detected)
            security_metrics["threats_detected"] = any(threat_detected)
            
            if security_metrics["threats_detected"]:
                # Log security event
                threatened_clients = [client_ids[i] for i, is_threat in enumerate(threat_detected) if is_threat]
                self._log_security_event(
                    "threats_detected",
                    ",".join(threatened_clients[:5]) + (f"... and {len(threatened_clients)-5} more" if len(threatened_clients) > 5 else ""),
                    f"Detected {security_metrics['num_threats']} threats during aggregation"
                )
                
                # Filter out updates from malicious clients
                filtered_updates = []
                filtered_client_ids = []
                filtered_weights = []
                
                for i, (update, client_id, is_threat) in enumerate(zip(updates, client_ids, threat_detected)):
                    if not is_threat:
                        filtered_updates.append(update)
                        filtered_client_ids.append(client_id)
                        if weights is not None:
                            filtered_weights.append(weights[i])
                
                # Update metrics
                security_metrics["num_clients_after_filtering"] = len(filtered_updates)
                security_metrics["filtering_rate"] = (len(updates) - len(filtered_updates)) / len(updates) if len(updates) > 0 else 0
                
                # If no updates left after filtering, use all updates
                if not filtered_updates:
                    logger.warning("No updates left after filtering threats, using all updates")
                    filtered_updates = updates
                    filtered_client_ids = client_ids
                    filtered_weights = weights
                    security_metrics["fallback_to_all"] = True
                
                # Use filtered updates
                updates_for_aggregation = filtered_updates
                client_ids_for_aggregation = filtered_client_ids
                weights_for_aggregation = filtered_weights
            else:
                # Use all updates
                updates_for_aggregation = updates
                client_ids_for_aggregation = client_ids
                weights_for_aggregation = weights
            
            # Apply secure aggregation
            if self.secure_aggregation.enabled:
                aggregated_update = self.secure_aggregation.aggregate(
                    updates_for_aggregation, 
                    weights_for_aggregation,
                    client_ids_for_aggregation
                )
                security_metrics["secure_aggregation_applied"] = True
                
                # Get aggregation metrics
                agg_metrics = self.secure_aggregation.get_aggregation_metrics()
                security_metrics["aggregation_method"] = agg_metrics["method"]
                security_metrics["exclusion_rate"] = agg_metrics.get("avg_exclusion_rate", 0)
            else:
                # Fall back to simple weighted average
                aggregated_update = self._weighted_average(updates_for_aggregation, weights_for_aggregation)
            
            # Finalize metrics
            security_metrics["processing_time"] = time.time() - start_time
            
            # Save security metrics periodically
            if round_num is not None and round_num % 5 == 0:
                self._save_security_metrics()
            
            return aggregated_update, security_metrics
            
        except Exception as e:
            logger.error(f"Error in secure aggregation: {e}")
            security_metrics["error"] = str(e)
            security_metrics["processing_time"] = time.time() - start_time
            
            # Log security event
            self._log_security_event("aggregation_error", "", str(e))
            
            # Fall back to simple weighted average
            return self._weighted_average(updates, weights), security_metrics
    
    def _weighted_average(
        self, 
        updates: List[List[np.ndarray]], 
        weights: Optional[List[float]] = None
    ) -> List[np.ndarray]:
        """Simple weighted average implementation for fallback."""
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
            layer_update = np.zeros_like(updates[0][i])
            for j, update in enumerate(updates):
                if update[i] is not None:
                    layer_update += weights[j] * update[i]
            aggregated_update.append(layer_update)
        
        return aggregated_update
    
    def process_model_for_client(
        self, 
        client_id: str, 
        model: Any
    ) -> Tuple[Any, Dict]:
        """
        Process a model before sending to a client, applying security measures.
        
        Args:
            client_id: Client identifier
            model: Model to process
            
        Returns:
            Tuple of (processed_model, security_metrics)
        """
        # Start timing
        start_time = time.time()
        
        # Initialize security metrics
        security_metrics = {
            "client_id": client_id,
            "timestamp": time.time(),
            "processed": False,
            "processing_time": 0
        }
        
        # Check if client is banned
        if client_id in self.banned_clients:
            logger.warning(f"Refused to send model to banned client {client_id}")
            security_metrics["rejected_reason"] = "banned"
            
            # Log security event
            self._log_security_event("model_distribution_blocked", client_id, "Client is banned")
            
            return model, security_metrics
        
        try:
            # No security processing needed for model distribution in current implementation
            # But the framework is in place for future extensions
            
            # Finalize metrics
            security_metrics["processed"] = True
            security_metrics["processing_time"] = time.time() - start_time
            
            return model, security_metrics
            
        except Exception as e:
            logger.error(f"Error processing model for client: {e}")
            security_metrics["error"] = str(e)
            security_metrics["processing_time"] = time.time() - start_time
            
            # Log security event
            self._log_security_event("model_processing_error", client_id, str(e))
            
            return model, security_metrics
    
    def process_evaluation_data(
        self, 
        client_id: str, 
        examples: List[np.ndarray],
        labels: List[int],
        model: Any
    ) -> Tuple[List[np.ndarray], List[int], Dict]:
        """
        Process evaluation data from a client, applying security measures.
        
        Args:
            client_id: Client identifier
            examples: Examples for evaluation
            labels: Labels for the examples
            model: Model for detecting adversarial examples
            
        Returns:
            Tuple of (processed_examples, processed_labels, security_metrics)
        """
        # Start timing
        start_time = time.time()
        
        # Initialize security metrics
        security_metrics = {
            "client_id": client_id,
            "timestamp": time.time(),
            "processed": False,
            "adversarial_examples_detected": False,
            "num_adversarial": 0,
            "processing_time": 0
        }
        
        # Check if client is banned
        if client_id in self.banned_clients:
            logger.warning(f"Rejected evaluation data from banned client {client_id}")
            security_metrics["rejected_reason"] = "banned"
            
            # Log security event
            self._log_security_event("evaluation_rejected", client_id, "Client is banned")
            
            return examples, labels, security_metrics
        
        try:
            # Detect adversarial examples
            if self.threat_model.enabled and self.threat_model.adversarial_defense:
                adversarial_detected = self.threat_model.detect_adversarial_examples(
                    examples, labels, model, client_id
                )
                
                security_metrics["adversarial_examples_detected"] = any(adversarial_detected)
                security_metrics["num_adversarial"] = sum(adversarial_detected)
                
                if security_metrics["adversarial_examples_detected"]:
                    # Log security event
                    self._log_security_event(
                        "adversarial_examples_detected",
                        client_id,
                        f"Detected {security_metrics['num_adversarial']} adversarial examples"
                    )
                    
                    # Filter out adversarial examples
                    filtered_examples = []
                    filtered_labels = []
                    
                    for example, label, is_adversarial in zip(examples, labels, adversarial_detected):
                        if not is_adversarial:
                            filtered_examples.append(example)
                            filtered_labels.append(label)
                    
                    # Update metrics
                    security_metrics["num_examples_after_filtering"] = len(filtered_examples)
                    security_metrics["filtering_rate"] = (len(examples) - len(filtered_examples)) / len(examples) if len(examples) > 0 else 0
                    
                    # If no examples left after filtering, use all examples
                    if not filtered_examples:
                        logger.warning("No examples left after filtering adversarial examples, using all examples")
                        filtered_examples = examples
                        filtered_labels = labels
                        security_metrics["fallback_to_all"] = True
                    
                    # Use filtered examples
                    examples = filtered_examples
                    labels = filtered_labels
            
            # Finalize metrics
            security_metrics["processed"] = True
            security_metrics["processing_time"] = time.time() - start_time
            
            return examples, labels, security_metrics
            
        except Exception as e:
            logger.error(f"Error processing evaluation data: {e}")
            security_metrics["error"] = str(e)
            security_metrics["processing_time"] = time.time() - start_time
            
            # Log security event
            self._log_security_event("evaluation_processing_error", client_id, str(e))
            
            return examples, labels, security_metrics
    
    def analyze_client_security(self, client_id: str) -> Dict:
        """
        Generate a security report for a client.
        
        Args:
            client_id: Client identifier
            
        Returns:
            Dictionary with client security metrics
        """
        report = {
            "client_id": client_id,
            "timestamp": time.time(),
            "banned": client_id in self.banned_clients,
            "threat_score": self.threat_model.get_client_threat_score(client_id),
            "privacy_metrics": self.differential_privacy.get_privacy_per_client(client_id)
        }
        
        # Add historical metrics if available
        if client_id in self.client_security_metrics:
            report["historical_metrics"] = self.client_security_metrics[client_id]
        
        return report
    
    def ban_client(self, client_id: str, reason: str) -> None:
        """
        Ban a client from participating in federated learning.
        
        Args:
            client_id: Client identifier
            reason: Reason for banning
        """
        self.banned_clients.add(client_id)
        
        # Log security event
        self._log_security_event("client_banned", client_id, reason)
        
        logger.warning(f"Banned client {client_id}: {reason}")
    
    def unban_client(self, client_id: str) -> None:
        """
        Unban a previously banned client.
        
        Args:
            client_id: Client identifier
        """
        if client_id in self.banned_clients:
            self.banned_clients.remove(client_id)
            
            # Log security event
            self._log_security_event("client_unbanned", client_id, "")
            
            logger.info(f"Unbanned client {client_id}")
    
    def get_certificates_dir(self) -> str:
        """
        Get the path to TLS certificates.
        
        Returns:
            Path to certificates directory
        """
        return self.encryption.cert_path
    
    def generate_certificates(self, common_name: str = "localhost") -> bool:
        """
        Generate TLS certificates for secure communication.
        
        Args:
            common_name: Common name for the certificate
            
        Returns:
            True if certificates were generated successfully, False otherwise
        """
        success = self.encryption.generate_certificates(common_name=common_name)
        
        if success:
            # Log security event
            self._log_security_event("certificates_generated", "", f"Generated certificates for {common_name}")
        
        return success
    
    def get_security_metrics(self) -> Dict:
        """
        Get overall security metrics for the system.
        
        Returns:
            Dictionary with security metrics
        """
        metrics = {
            "timestamp": time.time(),
            "differential_privacy": {
                "enabled": self.differential_privacy.enabled,
                "noise_multiplier": self.differential_privacy.noise_multiplier,
                "l2_norm_clip": self.differential_privacy.l2_norm_clip,
                "cumulative_steps": self.differential_privacy.cumulative_steps,
                "privacy_spent": self.differential_privacy.privacy_spent
            },
            "encryption": {
                "enabled": self.encryption.enabled,
                "type": self.encryption.encryption_type
            },
            "secure_aggregation": self.secure_aggregation.get_aggregation_metrics(),
            "threat_model": self.threat_model.get_threat_metrics(),
            "banned_clients": len(self.banned_clients),
            "security_events": len(self.security_events),
            "recent_events": self.security_events[-10:] if self.security_events else []
        }
        
        return metrics
    
    def _log_security_event(self, event_type: str, client_id: str, details: str) -> None:
        """
        Log a security event for analysis.
        
        Args:
            event_type: Type of security event
            client_id: Client identifier (if applicable)
            details: Additional details about the event
        """
        event = {
            "timestamp": time.time(),
            "event_type": event_type,
            "client_id": client_id,
            "details": details
        }
        
        self.security_events.append(event)
        
        # Keep events list manageable
        if len(self.security_events) > 1000:
            self.security_events = self.security_events[-1000:]
        
        # Log to system logger
        logger.info(f"Security event: {event_type} for {client_id or 'system'} - {details}")
    
    def _update_client_security_metrics(self, client_id: str, metrics: Dict) -> None:
        """
        Update security metrics for a client.
        
        Args:
            client_id: Client identifier
            metrics: New metrics to add
        """
        if client_id not in self.client_security_metrics:
            self.client_security_metrics[client_id] = []
        
        self.client_security_metrics[client_id].append(metrics)
        
        # Keep metrics history manageable
        if len(self.client_security_metrics[client_id]) > 20:
            self.client_security_metrics[client_id] = self.client_security_metrics[client_id][-20:]
    
    def _save_security_metrics(self) -> None:
        """Save security metrics to disk."""
        try:
            # Create metrics directory if it doesn't exist
            os.makedirs(self.metrics_dir, exist_ok=True)
            
            # Get current metrics
            metrics = self.get_security_metrics()
            
            # Save to file
            metrics_file = os.path.join(self.metrics_dir, "security_metrics.json")
            with open(metrics_file, "w") as f:
                json.dump(metrics, f, indent=2)
            
            # Save events to file
            events_file = os.path.join(self.metrics_dir, "security_events.json")
            with open(events_file, "w") as f:
                json.dump(self.security_events, f, indent=2)
            
            logger.debug(f"Saved security metrics to {metrics_file}")
            
        except Exception as e:
            logger.error(f"Error saving security metrics: {e}")