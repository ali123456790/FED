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
import hmac
import hashlib
import pickle
import base64
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
from pathlib import Path

from .differential_privacy import DifferentialPrivacy
from .encryption import Encryption
from .secure_aggregation import SecureAggregation
from .threat_model import ThreatModel
from .encoding import FixedPointEncoder

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
        
        # Configure secure aggregation options from config
        secure_agg_config = self.security_config.get("secure_aggregation", {})
        self.secure_aggregation_enabled = secure_agg_config.get("enabled", True)
        self.secure_aggregation_type = secure_agg_config.get("type", "trimmed_mean")
        
        # Track banned clients
        self.banned_clients = set()
        
        # Metrics tracking
        self.security_events = []
        self.client_security_metrics = {}
        
        # HMAC key management
        self.auth_config = self.security_config.get("authentication", {})
        self.hmac_keys = {}
        self.key_rotation_interval = self.auth_config.get("key_rotation_interval", 3600)  # 1 hour default
        self.last_rotation = time.time()
        
        # Setup specific keys for different communication paths
        self.client_edge_key = None  # K_CE
        self.edge_server_key = None  # K_ES
        
        # Initialize HMAC keys
        self._initialize_hmac_keys()
        
        logger.info("Security manager initialized")
        if self.secure_aggregation_enabled:
            logger.info(f"Secure aggregation enabled with method: {self.secure_aggregation_type}")
    
    def _initialize_hmac_keys(self):
        """Initialize HMAC keys for different communication paths."""
        # Get key management strategy from config
        hmac_config = self.auth_config.get("hmac_keys", {})
        key_management = hmac_config.get("key_management", "dynamic")
        
        # Get key paths in case we need to load from files
        key_paths = hmac_config.get("key_paths", {})
        ce_key_path = key_paths.get("client_edge_key_path", "./keys/k_ce.key")
        es_key_path = key_paths.get("edge_server_key_path", "./keys/k_es.key")
        
        # Create directory for key files if it doesn't exist
        keys_dir = os.path.dirname(ce_key_path)
        if keys_dir and keys_dir != ".":
            os.makedirs(keys_dir, exist_ok=True)
        
        # Initialize Client-Edge key (K_CE)
        self.client_edge_key = self._load_or_generate_key(
            key_type="client_edge",
            config_key=hmac_config.get("client_edge_key", ""),
            key_path=ce_key_path,
            key_management=key_management
        )
        
        # Initialize Edge-Server key (K_ES)
        self.edge_server_key = self._load_or_generate_key(
            key_type="edge_server",
            config_key=hmac_config.get("edge_server_key", ""),
            key_path=es_key_path,
            key_management=key_management
        )
        
        # For backward compatibility, also generate general rotation keys
        self._generate_initial_hmac_keys()
        
        logger.info("HMAC keys initialized for Client-Edge and Edge-Server communication")
    
    def _load_or_generate_key(self, key_type: str, config_key: str, key_path: str, key_management: str) -> bytes:
        """
        Load or generate an HMAC key based on the specified strategy.
        
        Args:
            key_type: Type of key (client_edge or edge_server)
            config_key: Key value from config
            key_path: Path to key file
            key_management: Key management strategy (config, file, dynamic)
            
        Returns:
            HMAC key as bytes
        """
        if key_management == "config" and config_key:
            # Use key directly from config (convert from hex or base64 if needed)
            try:
                if len(config_key) == 64:  # Hex-encoded 32-byte key
                    key = bytes.fromhex(config_key)
                else:  # Assume base64-encoded
                    key = base64.b64decode(config_key)
                logger.info(f"Loaded {key_type} key from config")
                return key
            except Exception as e:
                logger.error(f"Error loading {key_type} key from config: {e}")
                # Fall back to generating a random key
        
        elif key_management == "file":
            # Try to load key from file
            try:
                if os.path.exists(key_path):
                    with open(key_path, "rb") as f:
                        key = f.read()
                    logger.info(f"Loaded {key_type} key from {key_path}")
                    return key
            except Exception as e:
                logger.error(f"Error loading {key_type} key from {key_path}: {e}")
                # Fall back to generating a random key
        
        # Generate a random key (default fallback)
        key = os.urandom(32)  # 256-bit key
        
        # Save the key to file for future use if using file management
        if key_management == "file":
            try:
                with open(key_path, "wb") as f:
                    f.write(key)
                logger.info(f"Generated and saved {key_type} key to {key_path}")
            except Exception as e:
                logger.error(f"Error saving {key_type} key to {key_path}: {e}")
        
        logger.info(f"Generated random {key_type} key")
        return key
    
    def _generate_initial_hmac_keys(self):
        """Generate initial HMAC keys for backward compatibility."""
        self.current_key_id = os.urandom(16).hex()
        self.hmac_keys[self.current_key_id] = {
            'key': os.urandom(32),
            'created_at': time.time()
        }
    
    def rotate_hmac_keys(self):
        """Rotate HMAC keys if needed (for backward compatibility)."""
        current_time = time.time()
        if current_time - self.last_rotation >= self.key_rotation_interval:
            # Generate new key
            new_key_id = os.urandom(16).hex()
            self.hmac_keys[new_key_id] = {
                'key': os.urandom(32),
                'created_at': current_time
            }
            
            # Remove old keys (keep last 2 for verification)
            current_keys = sorted(self.hmac_keys.items(), key=lambda x: x[1]['created_at'], reverse=True)
            self.hmac_keys = dict(current_keys[:2])
            
            # Update current key
            self.current_key_id = new_key_id
            self.last_rotation = current_time
            
            # We could also rotate the client_edge_key and edge_server_key here
            # but for simplicity, we'll leave them unchanged for now
            logger.info("Rotated HMAC keys")
    
    def get_current_hmac_key(self) -> bytes:
        """Get current HMAC key (backward compatibility)."""
        self.rotate_hmac_keys()  # Check and rotate if needed
        return self.hmac_keys[self.current_key_id]['key']
    
    def get_client_edge_key(self) -> bytes:
        """
        Get the HMAC key for Client-Edge communication (K_CE).
        
        Returns:
            HMAC key for Client-Edge communication
        """
        return self.client_edge_key
    
    def get_edge_server_key(self) -> bytes:
        """
        Get the HMAC key for Edge-Server communication (K_ES).
        
        Returns:
            HMAC key for Edge-Server communication
        """
        return self.edge_server_key
    
    def verify_hmac(self, message: bytes, signature: bytes, key_id: Optional[str] = None, 
                    communication_path: str = "general") -> bool:
        """
        Verify HMAC signature using the appropriate key for the specified communication path.
        
        Args:
            message: Message to verify
            signature: HMAC signature
            key_id: Key identifier (for backward compatibility)
            communication_path: Communication path ("client_edge", "edge_server", or "general")
            
        Returns:
            True if signature is valid, False otherwise
        """
        # Determine which key to use based on communication path
        if communication_path == "client_edge":
            key = self.client_edge_key
            expected_hmac = hmac.new(key, message, hashlib.sha256).digest()
            return hmac.compare_digest(expected_hmac, signature)
        elif communication_path == "edge_server":
            key = self.edge_server_key
            expected_hmac = hmac.new(key, message, hashlib.sha256).digest()
            return hmac.compare_digest(expected_hmac, signature)
        else:
            # Backward compatibility mode - try with key_id or all keys
            if key_id and key_id in self.hmac_keys:
                key = self.hmac_keys[key_id]['key']
                expected_hmac = hmac.new(key, message, hashlib.sha256).digest()
                return hmac.compare_digest(expected_hmac, signature)
            else:
                # Try all recent keys
                for key_data in self.hmac_keys.values():
                    expected_hmac = hmac.new(key_data['key'], message, hashlib.sha256).digest()
                    if hmac.compare_digest(expected_hmac, signature):
                        return True
                # Also try with client_edge_key and edge_server_key as fallback
                for key in [self.client_edge_key, self.edge_server_key]:
                    expected_hmac = hmac.new(key, message, hashlib.sha256).digest()
                    if hmac.compare_digest(expected_hmac, signature):
                        return True
                return False
    
    def process_client_update(
        self, 
        client_id: str, 
        update: List[np.ndarray], 
        num_samples: int, 
        metadata: Optional[Dict] = None
    ) -> Tuple[List[np.ndarray], bool, Dict]:
        """
        Process a model update from a client, applying security measures.
        Following paper order: Verify HMAC -> Decrypt -> Apply DP (if needed).
        
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
            "processing_time": 0,
            "hmac_verified": False,
            "paillier_encrypted": False
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
            # Start with the received update
            processed_update = update
            
            # STEP 1: Verify HMAC first, as per paper flow
            if metadata and "hmac_digest" in metadata:
                hmac_verified = self.verify_hmac(
                    message=pickle.dumps(processed_update),
                    signature=metadata["hmac_digest"],
                    key_id=metadata.get("hmac_key_id"),
                    communication_path=metadata.get("communication_path", "general")
                )
                
                security_metrics["hmac_verified"] = hmac_verified
                
                if not hmac_verified:
                    logger.warning(f"HMAC verification failed for client {client_id}")
                    security_metrics["rejected_reason"] = "hmac_verification_failed"
                    return update, False, security_metrics
                
                logger.info(f"HMAC verification successful for client {client_id}")
            else:
                logger.warning(f"No HMAC digest provided for client {client_id}")
                # Continue processing but note the lack of authentication
                security_metrics["hmac_verified"] = False
            
            # STEP 2: Handle different encryption types
            if metadata and metadata.get("is_encrypted", False):
                encryption_type = metadata.get("encryption_type", "custom")
                
                if encryption_type == "paillier":
                    # For Paillier, we don't decrypt here - we'll keep it encrypted for edge aggregation
                    # Check if the update is correctly formatted for Paillier
                    if not all(isinstance(layer, dict) and "encrypted_values" in layer for layer in processed_update):
                        logger.warning(f"Invalid Paillier encrypted format from client {client_id}")
                        security_metrics["rejected_reason"] = "invalid_paillier_format"
                        return update, False, security_metrics
                    
                    # Mark as Paillier encrypted
                    security_metrics["paillier_encrypted"] = True
                    security_metrics["encryption_status"] = "paillier_encrypted"
                    logger.info(f"Received Paillier encrypted update from client {client_id}")
                
                elif encryption_type == "custom" or encryption_type == "hybrid":
                    # DEPRECATED: This code path handles the old RSA/AES hybrid encryption
                    # and should be phased out in favor of Paillier homomorphic encryption
                    logger.warning(f"Client {client_id} is using deprecated RSA/AES encryption. Should migrate to Paillier PHE.")
                    
                    # Extract encryption information from metadata for RSA/AES hybrid encryption
                    encrypted_aes_key = metadata.get("encrypted_aes_key")
                    encryption_iv = metadata.get("encryption_iv")
                    
                    if not all([encrypted_aes_key, encryption_iv]):
                        logger.warning(f"Missing encryption parameters from client {client_id}")
                        security_metrics["rejected_reason"] = "missing_encryption_params"
                        return update, False, security_metrics
                    
                    try:
                        # For hybrid encryption, we need to decrypt the update
                        for i, layer in enumerate(processed_update):
                            if isinstance(layer, bytes):
                                # Decrypt each layer
                                decrypted_layer = self.encryption.decrypt_update_hybrid(
                                    layer, encrypted_aes_key, encryption_iv
                                )
                                processed_update[i] = pickle.loads(decrypted_layer)
                        
                        security_metrics["encryption_status"] = "decrypted"
                        logger.info(f"Successfully decrypted update from client {client_id}")
                    except Exception as e:
                        logger.error(f"Error decrypting update from client {client_id}: {e}")
                        security_metrics["rejected_reason"] = "decryption_error"
                        return update, False, security_metrics
                else:
                    logger.warning(f"Unsupported encryption type '{encryption_type}' from client {client_id}")
                    security_metrics["rejected_reason"] = "unsupported_encryption"
                    return update, False, security_metrics
                
            # STEP 3: Apply differential privacy if enabled and not Paillier encrypted
            # For Paillier, DP is applied by the client before encryption
            dp_config = self.security_config.get("differential_privacy", {})
            if dp_config.get("enabled", False) and not security_metrics.get("paillier_encrypted", False):
                try:
                    processed_update = self.differential_privacy.apply_noise(processed_update, num_samples)
                    security_metrics["dp_applied"] = True
                    logger.info(f"Applied differential privacy to update from client {client_id}")
                except Exception as e:
                    logger.error(f"Error applying differential privacy to update from client {client_id}: {e}")
            
            # STEP 4: Check for threats if not Paillier encrypted
            # For Paillier, we can't inspect the update content until decryption
            if self.security_config.get("threat_model", {}).get("enabled", False) and not security_metrics.get("paillier_encrypted", False):
                threat_result = self.threat_model.analyze_update(
                    client_id, processed_update, metadata
                )
                
                if threat_result.get("is_malicious", False):
                    logger.warning(f"Threat detected in update from client {client_id}: {threat_result.get('threat_type')}")
                    security_metrics["threats_detected"] = True
                    security_metrics["threat_details"] = threat_result
                    
                    if threat_result.get("action", "") == "reject":
                        security_metrics["rejected_reason"] = "threat_detected"
                        return update, False, security_metrics
            
            # End timing
            security_metrics["processing_time"] = time.time() - start_time
            security_metrics["processed"] = True
            
            # Update client security metrics
            self._update_client_security_metrics(client_id, security_metrics)
            
            return processed_update, True, security_metrics
            
        except Exception as e:
            logger.error(f"Error processing update from client {client_id}: {e}", exc_info=True)
            security_metrics["processing_time"] = time.time() - start_time
            security_metrics["processed"] = False
            security_metrics["error"] = str(e)
            
            return update, False, security_metrics
    
    def aggregate_updates(
        self, 
        updates: List[List[np.ndarray]], 
        client_ids: List[str],
        weights: Optional[List[float]] = None,
        global_model: Optional[List[np.ndarray]] = None,
        round_num: Optional[int] = None,
        metadata: Optional[List[Dict]] = None
    ) -> Tuple[List[np.ndarray], Dict]:
        """
        Aggregate model updates securely.
        
        Args:
            updates: List of model updates from clients
            client_ids: List of client identifiers
            weights: Optional weights for weighted aggregation
            global_model: Optional current global model
            round_num: Optional round number
            metadata: Optional list of update metadata
            
        Returns:
            Tuple of (aggregated_update, aggregation_metrics)
        """
        if not updates:
            logger.warning("No updates to aggregate")
            return [], {"error": "no_updates"}
            
        start_time = time.time()
        
        # Initialize metrics
        aggregation_metrics = {
            "timestamp": time.time(),
            "num_updates": len(updates),
            "client_ids": client_ids,
            "aggregation_strategy": self.secure_aggregation_type,
            "aggregation_time": 0,
            "edge_aggregated": False
        }
        
        try:
            # Check if updates are Paillier encrypted
            is_paillier_encrypted = metadata and all(
                m.get("is_encrypted", False) and m.get("encryption_type", "") == "paillier" 
                for m in metadata if m is not None
            )
            
            # For Paillier encrypted updates, we need special handling
            if is_paillier_encrypted:
                logger.info("Aggregating Paillier encrypted updates")
                aggregation_metrics["edge_aggregated"] = True
                
                # Two cases:
                # 1. We're on an edge node - just homomorphically add the encrypted updates
                if not hasattr(self.encryption, "_paillier_private_key") or self.encryption._paillier_private_key is None:
                    logger.info("Edge node aggregating encrypted updates homomorphically")
                    # We're on an edge node without private key, so homomorphically add
                    aggregated_update = self.encryption.homomorphic_add_weights(updates)
                    aggregation_metrics["aggregation_type"] = "homomorphic_addition"
                    
                # 2. We're on the server - decrypt and then apply secure aggregation
                else:
                    logger.info("Server decrypting and aggregating updates")
                    # We're on the server with private key, so decrypt first
                    decrypted_updates = []
                    
                    # For a single aggregated update from edge
                    if len(updates) == 1:
                        decrypted_update = self.encryption.decrypt_weights_paillier(updates[0])
                        
                        # If weights are provided (e.g., client count from edge), apply scaling
                        if weights and len(weights) == 1 and weights[0] > 1:
                            # Scale back the update based on client count
                            client_count = weights[0]
                            decrypted_update = [layer / client_count for layer in decrypted_update]
                            
                        # Return the single decrypted update
                        aggregation_metrics["aggregation_type"] = "paillier_decryption"
                        aggregation_metrics["aggregation_time"] = time.time() - start_time
                        return decrypted_update, aggregation_metrics
                    
                    # For multiple encrypted updates (server aggregating from multiple edges)
                    for i, update in enumerate(updates):
                        decrypted_update = self.encryption.decrypt_weights_paillier(update)
                        decrypted_updates.append(decrypted_update)
                    
                    # Apply secure aggregation on decrypted updates
                    return self._apply_secure_aggregation(
                        decrypted_updates, client_ids, weights, 
                        global_model, aggregation_metrics, start_time
                    )
            
            # For non-Paillier updates, use standard secure aggregation
            return self._apply_secure_aggregation(
                updates, client_ids, weights, global_model, 
                aggregation_metrics, start_time
            )
            
        except Exception as e:
            logger.error(f"Error in secure aggregation: {e}", exc_info=True)
            aggregation_metrics["error"] = str(e)
            aggregation_metrics["aggregation_time"] = time.time() - start_time
            
            # Return empty update on error or current global model if available
            if global_model:
                return global_model, aggregation_metrics
            else:
                return [], aggregation_metrics
                
    def _apply_secure_aggregation(
        self,
        updates: List[List[np.ndarray]],
        client_ids: List[str],
        weights: Optional[List[float]],
        global_model: Optional[List[np.ndarray]],
        aggregation_metrics: Dict,
        start_time: float
    ) -> Tuple[List[np.ndarray], Dict]:
        """Apply the appropriate secure aggregation method."""
        if self.secure_aggregation_enabled:
            # Use secure aggregation
            aggregated_update, agg_metrics = self.secure_aggregation.aggregate(
                updates=updates,
                weights=weights,
                client_ids=client_ids,
                method=self.secure_aggregation_type,
                global_model=global_model
            )
            
            # Update metrics
            aggregation_metrics.update(agg_metrics)
            
        else:
            # Use simple weighted average
            aggregated_update = self._weighted_average(updates, weights)
            aggregation_metrics["aggregation_type"] = "weighted_average"
            
        aggregation_metrics["aggregation_time"] = time.time() - start_time
        return aggregated_update, aggregation_metrics
    
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