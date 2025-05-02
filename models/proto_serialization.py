"""
Protocol Buffer serialization utilities for model updates.

This module provides functions to serialize and deserialize model updates and other
messages using Protocol Buffers, replacing the previous ID-based approach.
"""

import logging
import numpy as np
import time
from typing import Dict, List, Tuple, Any, Optional, Union
import hmac
import hashlib
import os
import base64
from pathlib import Path
from phe import paillier

try:
    import model_messages_pb2
    from model_messages_pb2 import ClientUpdateProto, EdgeAggregateProto
except ImportError:
    # Generate Python code from .proto file if it doesn't exist
    import subprocess
    import sys
    
    proto_file = Path(__file__).parent / "model_messages.proto"
    output_dir = Path(__file__).parent
    
    if not proto_file.exists():
        logging.error(f"Proto file not found: {proto_file}")
        raise ImportError("Protocol buffer definition file not found")
    
    try:
        # Try to compile the proto file
        subprocess.check_call([
            "protoc",
            f"--proto_path={output_dir.parent}",
            f"--python_out={output_dir.parent}",
            proto_file
        ])
        logging.info(f"Generated Python code from {proto_file}")
        
        # Add the parent directory to the path to find the generated module
        sys.path.append(str(output_dir.parent))
        import model_messages_pb2
        from model_messages_pb2 import ClientUpdateProto, EdgeAggregateProto
    except Exception as e:
        logging.error(f"Failed to generate Python code from proto file: {e}")
        raise ImportError("Failed to generate Protocol Buffer code")

logger = logging.getLogger(__name__)

class ProtoSerializer:
    """Protocol Buffer serialization for model updates and messages."""
    
    def __init__(self, hmac_key: Optional[bytes] = None, 
                client_edge_key: Optional[bytes] = None,
                edge_server_key: Optional[bytes] = None):
        """
        Initialize the serializer with HMAC keys for authentication.
        
        Args:
            hmac_key: Generic HMAC key for backward compatibility
            client_edge_key: K_CE - Key for Client-Edge communication
            edge_server_key: K_ES - Key for Edge-Server communication
        """
        self.hmac_key = hmac_key or os.urandom(32)
        
        # Initialize specific keys for different communication paths
        self.client_edge_key = client_edge_key or self.hmac_key  # K_CE - Use general key as fallback
        self.edge_server_key = edge_server_key or self.hmac_key  # K_ES - Use general key as fallback
    
    def serialize_model_weights(self, weights: List[np.ndarray], client_id: str, 
                               round_id: int, metrics: Dict[str, float], 
                               training_time: float, samples: int,
                               encrypted_data: Optional[Dict] = None) -> bytes:
        """
        Serialize model weights using Protocol Buffers.
        Following paper security flow: DP -> Encrypt -> HMAC
        
        Note: For new code, prefer using serialize_client_update_phe() with Paillier 
        encryption rather than this method with hybrid RSA/AES encryption.
        This method is retained for backward compatibility.
        
        Args:
            weights: List of model weights as numpy arrays
            client_id: Client identifier
            round_id: Federation round number
            metrics: Training metrics
            training_time: Time taken for training
            samples: Number of training samples
            encrypted_data: DEPRECATED - Optional dict containing encryption info for
                            hybrid RSA/AES encryption: {"encrypted_aes_key": bytes, "encryption_iv": bytes}
            
        Returns:
            Serialized model update as bytes
        """
        try:
            if encrypted_data is not None:
                logger.warning("Using deprecated hybrid RSA/AES encryption. Consider migrating to Paillier homomorphic encryption.")
            
            # Create a model update message
            update = model_messages_pb2.ModelUpdate()
            update.client_id = client_id
            update.round_id = round_id
            update.timestamp = int(time.time())
            update.training_time = training_time
            update.training_samples = samples
            
            # Add all metrics
            for key, value in metrics.items():
                update.metrics[key] = float(value)
            
            # Handle encryption if present
            if encrypted_data is not None and "encrypted_aes_key" in encrypted_data:
                update.is_encrypted = True
                update.encrypted_aes_key = encrypted_data["encrypted_aes_key"]
                update.encryption_iv = encrypted_data["encryption_iv"]
            else:
                update.is_encrypted = False
            
            # Add all layers
            for layer in weights:
                layer_data = update.layers.add()
                layer_data.shape.extend(layer.shape)
                layer_data.dtype = str(layer.dtype)
                layer_data.data = layer.tobytes()
            
            # Generate HMAC digest after everything else (including encryption)
            # This follows the paper's security flow: DP -> Encrypt -> HMAC
            h = hmac.new(self.hmac_key, digestmod=hashlib.sha256)
            h.update(client_id.encode())
            h.update(str(round_id).encode())
            
            # Add weights data to HMAC (these should be the encrypted weights if encryption was used)
            for layer in weights:
                h.update(layer.tobytes())
            
            # Add encryption data to HMAC if present
            if update.is_encrypted:
                h.update(update.encrypted_aes_key)
                h.update(update.encryption_iv)
            
            update.hmac_digest = h.digest()
            
            # Serialize the message
            return update.SerializeToString()
            
        except Exception as e:
            logger.error(f"Error serializing model weights: {e}")
            # Fallback to pickle in case of error
            import pickle
            return pickle.dumps(weights)
    
    def serialize_client_update_phe(self, client_id: str, round_id: int, num_examples: int, 
                                   paillier_ciphertexts_list: List[str], 
                                   device_type: str = "unknown",
                                   training_time: float = 0.0,
                                   metadata: Optional[Dict] = None) -> bytes:
        """
        Serializes client update with Paillier ciphertexts.
        Uses the Client-Edge key (K_CE) for HMAC authentication.
        
        Args:
            client_id: Client identifier
            round_id: Federation round number
            num_examples: Number of training samples
            paillier_ciphertexts_list: List of encrypted values as strings
            device_type: Type of device (e.g., "mobile", "desktop")
            training_time: Time taken for training in seconds
            metadata: Additional metadata to include
            
        Returns:
            Serialized ClientUpdateProto message as bytes
        """
        try:
            # Create the ClientUpdateProto message
            proto = ClientUpdateProto()
            proto.client_id = client_id
            proto.round_id = round_id
            proto.training_samples = num_examples
            proto.device_type = device_type
            proto.timestamp = int(time.time())
            proto.training_time = training_time
            
            # Populate the repeated field with ciphertexts
            proto.paillier_ciphertexts.extend(paillier_ciphertexts_list)
            
            # Add any additional metadata
            if metadata:
                for key, value in metadata.items():
                    if isinstance(value, (int, float)):
                        proto.metrics[key] = float(value)
                    elif isinstance(value, str):
                        proto.metrics[key] = float(0.0)  # Placeholder for string metrics
                        logger.warning(f"String metric '{key}' converted to 0.0 (metrics only support float values)")
            
            # First serialize without HMAC for HMAC calculation
            serialized_data_for_hmac = proto.SerializeToString()
            
            # Calculate HMAC using Client-Edge key (K_CE)
            h = hmac.new(self.client_edge_key, digestmod=hashlib.sha256)
            h.update(client_id.encode())
            h.update(str(round_id).encode())
            h.update(serialized_data_for_hmac)
            
            # Add HMAC to the proto
            proto.hmac_digest = h.digest()
            
            # Add a field indicating which key was used (for verification)
            proto.communication_path = "client_edge"
            
            # Final serialization with HMAC included
            return proto.SerializeToString()
            
        except Exception as e:
            logger.error(f"Error serializing PHE client update: {e}")
            # In case of error, return an empty ClientUpdateProto
            error_proto = ClientUpdateProto()
            error_proto.client_id = client_id
            error_proto.round_id = round_id
            error_proto.timestamp = int(time.time())
            error_proto.metrics["error"] = 1.0
            return error_proto.SerializeToString()
    
    def deserialize_client_update_phe(self, serialized_data: bytes) -> Tuple[List[str], Dict, bool]:
        """
        Deserialize a ClientUpdateProto message.
        Verifies HMAC using the Client-Edge key (K_CE).
        
        Args:
            serialized_data: Serialized ClientUpdateProto message
            
        Returns:
            Tuple of (paillier_ciphertexts, metadata, authentication_verified)
        """
        try:
            # Parse the message
            update = ClientUpdateProto()
            update.ParseFromString(serialized_data)
            
            # Extract metadata
            metadata = {
                "client_id": update.client_id,
                "round_id": update.round_id,
                "timestamp": update.timestamp,
                "training_time": update.training_time,
                "training_samples": update.training_samples,
                "device_type": update.device_type,
                "metrics": dict(update.metrics),
                "communication_path": "client_edge"  # Add communication path to metadata
            }
            
            # Extract ciphertexts
            paillier_ciphertexts = list(update.paillier_ciphertexts)
            
            # Verify HMAC
            received_hmac = update.hmac_digest
            metadata["hmac_digest"] = received_hmac
            
            # Clear HMAC for verification
            update.hmac_digest = b""
            
            # Recompute HMAC using Client-Edge key (K_CE)
            h = hmac.new(self.client_edge_key, digestmod=hashlib.sha256)
            h.update(update.client_id.encode())
            h.update(str(update.round_id).encode())
            h.update(update.SerializeToString())
            
            computed_hmac = h.digest()
            hmac_verified = (received_hmac == computed_hmac)
            
            if not hmac_verified:
                logger.warning(f"HMAC verification failed for client {update.client_id}")
            
            return paillier_ciphertexts, metadata, hmac_verified
            
        except Exception as e:
            logger.error(f"Error deserializing PHE client update: {e}")
            return [], {"error": str(e)}, False

    def deserialize_model_weights(self, serialized_data: bytes) -> Tuple[List[np.ndarray], Dict, bool]:
        """
        Deserialize model weights from Protocol Buffer format.
        Following paper security flow for verification: Verify HMAC -> Decrypt -> Process
        
        Args:
            serialized_data: Serialized model update
            
        Returns:
            Tuple of (weights, metadata, authentication_verified)
        """
        try:
            # Parse the message
            update = model_messages_pb2.ModelUpdate()
            update.ParseFromString(serialized_data)
            
            # Extract metadata
            metadata = {
                "client_id": update.client_id,
                "round_id": update.round_id,
                "timestamp": update.timestamp,
                "training_time": update.training_time,
                "training_samples": update.training_samples,
                "metrics": dict(update.metrics)
            }
            
            # Add encryption information to metadata if present
            if update.is_encrypted:
                metadata["is_encrypted"] = True
                metadata["encrypted_aes_key"] = update.encrypted_aes_key
                metadata["encryption_iv"] = update.encryption_iv
            else:
                metadata["is_encrypted"] = False
                
            # Extract weights
            weights = []
            for layer_data in update.layers:
                shape = tuple(layer_data.shape)
                dtype = np.dtype(layer_data.dtype)
                layer = np.frombuffer(layer_data.data, dtype=dtype).reshape(shape)
                weights.append(layer)
            
            # STEP 1: Verify HMAC first (on the potentially encrypted weights)
            # This follows the paper's security flow: Verify HMAC -> Decrypt -> Process
            received_hmac = update.hmac_digest
            metadata["hmac_digest"] = received_hmac
            
            # Recompute HMAC
            h = hmac.new(self.hmac_key, digestmod=hashlib.sha256)
            h.update(update.client_id.encode())
            h.update(str(update.round_id).encode())
            
            # Add weights data to HMAC (these are still encrypted if encryption was used)
            for layer in weights:
                h.update(layer.tobytes())
            
            # Add encryption data to HMAC if present
            if update.is_encrypted:
                h.update(update.encrypted_aes_key)
                h.update(update.encryption_iv)
            
            computed_hmac = h.digest()
            
            # Verify HMAC
            hmac_verified = (received_hmac == computed_hmac)
            
            if not hmac_verified:
                logger.warning(f"HMAC verification failed for client {update.client_id}")
            
            return weights, metadata, hmac_verified
            
        except Exception as e:
            logger.error(f"Error deserializing model weights: {e}")
            
            # Try to fallback to pickle
            try:
                import pickle
                weights = pickle.loads(serialized_data)
                if isinstance(weights, list) and all(isinstance(w, np.ndarray) for w in weights):
                    return weights, {"deserialization_error": str(e), "fallback": "pickle"}, False
                else:
                    return [], {"deserialization_error": str(e)}, False
            except:
                return [], {"deserialization_error": str(e)}, False
    
    def create_client_identity(self, client_id: str, device_id: str, 
                              round_id: int, public_key: bytes,
                              signature: bytes) -> bytes:
        """
        Create a client identity message for authentication.
        
        Args:
            client_id: Client identifier
            device_id: Device identifier
            round_id: Federation round number
            public_key: Client's public key for encryption
            signature: Signature for authentication
            
        Returns:
            Serialized client identity message
        """
        try:
            identity = model_messages_pb2.ClientIdentity()
            identity.client_id = client_id
            identity.device_id = device_id
            identity.round_id = round_id
            identity.timestamp = int(time.time())
            identity.public_key = public_key
            identity.signature = signature
            
            return identity.SerializeToString()
        except Exception as e:
            logger.error(f"Error creating client identity: {e}")
            # Return minimal identity
            identity = model_messages_pb2.ClientIdentity()
            identity.client_id = client_id
            identity.timestamp = int(time.time())
            return identity.SerializeToString()
    
    def verify_client_identity(self, serialized_identity: bytes) -> Tuple[Dict, bool]:
        """
        Verify a client identity message.
        
        Args:
            serialized_identity: Serialized client identity message
            
        Returns:
            Tuple of (metadata, verification_result)
        """
        try:
            identity = model_messages_pb2.ClientIdentity()
            identity.ParseFromString(serialized_identity)
            
            # Extract metadata
            metadata = {
                "client_id": identity.client_id,
                "device_id": identity.device_id,
                "round_id": identity.round_id,
                "timestamp": identity.timestamp,
                "public_key": identity.public_key
            }
            
            # In a real implementation, verification would be performed here
            # For now, we just return the parsed data
            return metadata, True
        except Exception as e:
            logger.error(f"Error verifying client identity: {e}")
            return {"error": str(e)}, False
    
    def serialize_edge_aggregate_phe(self, edge_id: str, round_id: int, client_count: int,
                                    aggregated_paillier_ciphertexts: List[str],
                                    metrics: Optional[Dict] = None,
                                    total_examples: int = 0) -> bytes:
        """
        Serializes edge node aggregation with homomorphically added Paillier ciphertexts.
        Uses the Edge-Server key (K_ES) for HMAC authentication.
        
        Args:
            edge_id: Edge node identifier
            round_id: Federation round number
            client_count: Number of clients contributing to the aggregation
            aggregated_paillier_ciphertexts: List of homomorphically added ciphertexts as strings
                or list of paillier.EncryptedNumber objects
            metrics: Additional metrics to include
            total_examples: Total number of examples across all clients (optional)
            
        Returns:
            Serialized EdgeAggregateProto message as bytes
        """
        try:
            # Create the EdgeAggregateProto message
            proto = EdgeAggregateProto()
            proto.edge_id = edge_id
            proto.round_id = round_id
            proto.client_count = client_count
            proto.timestamp = int(time.time())
            
            if total_examples > 0:
                proto.total_examples = total_examples
            
            # Handle the case where we might get EncryptedNumber objects instead of strings
            cipher_strings = []
            
            # Check if we received EncryptedNumber objects
            if aggregated_paillier_ciphertexts and isinstance(aggregated_paillier_ciphertexts[0], paillier.EncryptedNumber):
                # Convert EncryptedNumber objects to string representations
                cipher_strings = [str(ct.ciphertext(be_secure=False)) for ct in aggregated_paillier_ciphertexts]
            else:
                # Already string representations
                cipher_strings = aggregated_paillier_ciphertexts
            
            # Populate the repeated field with aggregated ciphertexts
            proto.aggregated_paillier_ciphertexts.extend(cipher_strings)
            
            # Add any additional metrics
            if metrics:
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        proto.metrics[key] = float(value)
                    elif isinstance(value, str):
                        proto.metrics[key] = float(0.0)  # Placeholder for string metrics
                        logger.warning(f"String metric '{key}' converted to 0.0 (metrics only support float values)")
            
            # First serialize without HMAC for HMAC calculation
            serialized_data_for_hmac = proto.SerializeToString()
            
            # Calculate HMAC using Edge-Server key (K_ES)
            h = hmac.new(self.edge_server_key, digestmod=hashlib.sha256)
            h.update(edge_id.encode())
            h.update(str(round_id).encode())
            h.update(serialized_data_for_hmac)
            
            # Add HMAC to the proto
            proto.hmac_digest = h.digest()
            
            # Add a field indicating which key was used (for verification)
            proto.communication_path = "edge_server"
            
            # Final serialization with HMAC included
            return proto.SerializeToString()
            
        except Exception as e:
            logger.error(f"Error serializing PHE edge aggregation: {e}")
            # In case of error, return an empty EdgeAggregateProto
            error_proto = EdgeAggregateProto()
            error_proto.edge_id = edge_id
            error_proto.round_id = round_id
            error_proto.timestamp = int(time.time())
            error_proto.metrics["error"] = 1.0
            return error_proto.SerializeToString()

def serialize_edge_aggregate_phe(edge_id: str, round_num: int, num_clients: int, total_examples: int, 
                               aggregated_ciphertexts: List[paillier.EncryptedNumber], 
                               metadata: Dict = None, hmac_key: Optional[bytes] = None,
                               communication_path: str = "edge_server") -> bytes:
    """
    Serializes edge aggregate with Paillier ciphertexts and prepares for HMAC.
    
    This standalone function can be used without creating a ProtoSerializer instance.
    Uses the Edge-Server key (K_ES) for HMAC authentication.
    
    Args:
        edge_id: Edge node identifier
        round_num: Federation round number
        num_clients: Number of clients that contributed to the aggregation
        total_examples: Total number of training examples across all clients
        aggregated_ciphertexts: List of homomorphically added Paillier EncryptedNumber objects
        metadata: Additional metadata to include in the serialization
        hmac_key: Key for HMAC authentication (K_ES - Edge-Server key)
        communication_path: Communication path ("edge_server" by default)
        
    Returns:
        Serialized EdgeAggregateProto message as bytes
    """
    # Create the EdgeAggregateProto message
    proto = EdgeAggregateProto()
    proto.edge_id = edge_id
    proto.round_id = round_num
    proto.client_count = num_clients
    proto.timestamp = int(time.time())
    proto.total_examples = total_examples
    proto.communication_path = communication_path
    
    # Convert Paillier objects to strings
    agg_cipher_strs = [str(ct.ciphertext(be_secure=False)) for ct in aggregated_ciphertexts]
    proto.aggregated_paillier_ciphertexts.extend(agg_cipher_strs)
    
    # Add metadata
    if metadata:
        for key, value in metadata.items():
            if isinstance(value, (int, float)):
                proto.metrics[key] = float(value)
            elif isinstance(value, str):
                # Protocol buffers only support float values for metrics
                logger.warning(f"String metric '{key}' cannot be added directly")
    
    # Use a random HMAC key if none provided
    if hmac_key is None:
        hmac_key = os.urandom(32)
    
    # First serialize without HMAC for HMAC calculation
    serialized_data_for_hmac = proto.SerializeToString()
    
    # Calculate HMAC
    h = hmac.new(hmac_key, digestmod=hashlib.sha256)
    h.update(edge_id.encode())
    h.update(str(round_num).encode())
    h.update(serialized_data_for_hmac)
    
    # Add HMAC to the proto
    proto.hmac_digest = h.digest()
    
    # Final serialization with HMAC included
    return proto.SerializeToString() 