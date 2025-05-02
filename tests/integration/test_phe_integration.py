"""
Integration tests for Paillier Homomorphic Encryption in Federated Learning.

These tests verify the end-to-end flow of homomorphic encryption in the federated 
learning process, from client-side encryption to edge aggregation to server decryption.
"""

import unittest
import numpy as np
import sys
import os
import logging
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.ERROR)

# Add the project root to the path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import PHE library
from phe import paillier

# Import our modules
from security.encoding import FixedPointEncoder
from client.device_manager import DeviceManager
from edge.edge_aggregator import EdgeAggregator
from server.server import Server
from server.aggregation import FederatedAggregator

class MockClient:
    """Mock client for testing."""
    
    def __init__(self, client_id, model_weights, public_key):
        self.client_id = client_id
        self.model_weights = model_weights
        self.public_key = public_key
        self.encoder = FixedPointEncoder(precision_bits=32)
    
    def encrypt_weights(self) -> List[List[str]]:
        """Encrypt weights using the public key."""
        encrypted_weights_list = []
        
        for layer in self.model_weights:
            layer_encrypted = []
            flat_layer = layer.flatten()
            
            for value in flat_layer:
                # Encode the float to an integer
                encoded = self.encoder.encode(value)
                
                # Encrypt the encoded value
                encrypted = self.public_key.encrypt(encoded)
                
                # Convert to string representation
                layer_encrypted.append(str(encrypted.ciphertext()))
            
            encrypted_weights_list.append(layer_encrypted)
        
        return encrypted_weights_list

class MockEdge:
    """Mock edge for testing."""
    
    def __init__(self, public_key):
        self.public_key = public_key
    
    def aggregate_encrypted_weights(self, encrypted_weights_list):
        """Aggregate encrypted weights from multiple clients."""
        if not encrypted_weights_list:
            return None
        
        # Assuming all clients have the same model structure
        num_layers = len(encrypted_weights_list[0])
        aggregated_weights = []
        
        for layer_idx in range(num_layers):
            # Get all client encrypted values for this layer
            layer_encrypted_values = []
            for client_weights in encrypted_weights_list:
                layer_encrypted_values.append(client_weights[layer_idx])
            
            # Convert string representations back to EncryptedNumber objects
            layer_ciphers = []
            for client_idx, client_layer in enumerate(layer_encrypted_values):
                client_ciphers = []
                for value_str in client_layer:
                    ciphertext = int(value_str)
                    encrypted_value = paillier.EncryptedNumber(self.public_key, ciphertext)
                    client_ciphers.append(encrypted_value)
                layer_ciphers.append(client_ciphers)
            
            # Perform homomorphic addition
            if not layer_ciphers:
                continue
                
            # Initialize with the first client's values
            aggregated_layer = layer_ciphers[0].copy()
            
            # Add all other clients' values
            for client_idx in range(1, len(layer_ciphers)):
                for i in range(len(aggregated_layer)):
                    aggregated_layer[i] += layer_ciphers[client_idx][i]
            
            # Convert back to strings for transmission
            aggregated_layer_str = [str(val.ciphertext()) for val in aggregated_layer]
            aggregated_weights.append(aggregated_layer_str)
        
        return aggregated_weights

class MockServer:
    """Mock server for testing."""
    
    def __init__(self, private_key, original_shapes):
        self.private_key = private_key
        self.original_shapes = original_shapes
        self.encoder = FixedPointEncoder(precision_bits=32)
    
    def decrypt_weights(self, encrypted_weights_list):
        """Decrypt the aggregated encrypted weights."""
        if not encrypted_weights_list:
            return None
        
        decrypted_weights = []
        
        for layer_idx, layer_str_values in enumerate(encrypted_weights_list):
            # Convert string representations back to EncryptedNumber objects
            layer_encrypted = []
            for value_str in layer_str_values:
                ciphertext = int(value_str)
                encrypted_value = paillier.EncryptedNumber(self.private_key.public_key, ciphertext)
                layer_encrypted.append(encrypted_value)
            
            # Decrypt all values in the layer
            layer_decrypted = np.array([self.private_key.decrypt(val) for val in layer_encrypted])
            
            # Decode the values back to floats
            layer_decoded = self.encoder.decode(layer_decrypted)
            
            # Reshape to original tensor shape
            original_shape = self.original_shapes[layer_idx]
            layer_reshaped = layer_decoded.reshape(original_shape)
            
            decrypted_weights.append(layer_reshaped)
        
        return decrypted_weights


class TestPHEIntegration(unittest.TestCase):
    """Test the integration of Paillier homomorphic encryption across components."""
    
    def setUp(self):
        """Set up test environment."""
        # Generate keypair
        self.public_key, self.private_key = paillier.generate_paillier_keypair(n_length=1024)
        
        # Create mock model weights for 3 clients
        # Simple model with 2 layers
        self.client1_weights = [
            np.array([[0.1, 0.2], [0.3, 0.4]]),  # 2x2 weight matrix
            np.array([0.5, 0.6])                  # bias vector
        ]
        
        self.client2_weights = [
            np.array([[0.2, 0.3], [0.4, 0.5]]),
            np.array([0.6, 0.7])
        ]
        
        self.client3_weights = [
            np.array([[0.3, 0.4], [0.5, 0.6]]),
            np.array([0.7, 0.8])
        ]
        
        # Store original shapes for reshaping after decryption
        self.original_shapes = [w.shape for w in self.client1_weights]
        
        # Create mock clients, edge, and server
        self.client1 = MockClient("client1", self.client1_weights, self.public_key)
        self.client2 = MockClient("client2", self.client2_weights, self.public_key)
        self.client3 = MockClient("client3", self.client3_weights, self.public_key)
        
        self.edge = MockEdge(self.public_key)
        self.server = MockServer(self.private_key, self.original_shapes)
    
    def test_end_to_end_phe_workflow(self):
        """Test the complete PHE workflow from client encryption to server decryption."""
        # 1. Client-side: Encrypt model weights
        encrypted_weights1 = self.client1.encrypt_weights()
        encrypted_weights2 = self.client2.encrypt_weights()
        encrypted_weights3 = self.client3.encrypt_weights()
        
        all_encrypted_weights = [encrypted_weights1, encrypted_weights2, encrypted_weights3]
        
        # 2. Edge-side: Aggregate encrypted weights
        aggregated_encrypted_weights = self.edge.aggregate_encrypted_weights(all_encrypted_weights)
        
        # 3. Server-side: Decrypt aggregated weights
        decrypted_weights = self.server.decrypt_weights(aggregated_encrypted_weights)
        
        # 4. Verify correctness by comparing with direct sum
        expected_weights = []
        for i in range(len(self.client1_weights)):
            expected = self.client1_weights[i] + self.client2_weights[i] + self.client3_weights[i]
            expected_weights.append(expected)
        
        # Check results
        for i in range(len(expected_weights)):
            np.testing.assert_allclose(
                decrypted_weights[i], 
                expected_weights[i],
                rtol=1e-4, atol=1e-4,
                err_msg=f"Layer {i} does not match expected sum"
            )
    
    def test_fedavg_with_phe(self):
        """Test Federal averaging with homomorphic encryption."""
        # 1. Client-side: Encrypt model weights
        encrypted_weights1 = self.client1.encrypt_weights()
        encrypted_weights2 = self.client2.encrypt_weights()
        encrypted_weights3 = self.client3.encrypt_weights()
        
        all_encrypted_weights = [encrypted_weights1, encrypted_weights2, encrypted_weights3]
        
        # 2. Edge-side: Aggregate encrypted weights
        aggregated_encrypted_weights = self.edge.aggregate_encrypted_weights(all_encrypted_weights)
        
        # 3. Server-side: Decrypt aggregated weights
        decrypted_sum = self.server.decrypt_weights(aggregated_encrypted_weights)
        
        # 4. Server-side: Perform FedAvg (divide by number of clients)
        num_clients = 3
        fedavg_weights = [layer / num_clients for layer in decrypted_sum]
        
        # 5. Verify correctness by comparing with direct average
        expected_avg = []
        for i in range(len(self.client1_weights)):
            expected = (self.client1_weights[i] + self.client2_weights[i] + self.client3_weights[i]) / num_clients
            expected_avg.append(expected)
        
        # Check results
        for i in range(len(expected_avg)):
            np.testing.assert_allclose(
                fedavg_weights[i], 
                expected_avg[i],
                rtol=1e-4, atol=1e-4,
                err_msg=f"Layer {i} average does not match expected"
            )
    
    def test_weighted_aggregation_with_phe(self):
        """Test weighted aggregation with homomorphic encryption."""
        # Define client weights for weighted averaging
        client_weights = {"client1": 0.5, "client2": 0.3, "client3": 0.2}
        
        # 1. Client-side: Encrypt model weights
        encrypted_weights1 = self.client1.encrypt_weights()
        encrypted_weights2 = self.client2.encrypt_weights()
        encrypted_weights3 = self.client3.encrypt_weights()
        
        # Apply client weights before aggregation (multiply each client's weights by their importance)
        # Note: With PHE, we can multiply encrypted values by scalars
        
        # Custom edge aggregation with weights
        weighted_aggregated = []
        
        # Process each layer
        for layer_idx in range(len(encrypted_weights1)):
            # Convert all clients' encrypted values for this layer
            client1_values = [paillier.EncryptedNumber(self.public_key, int(val)) * client_weights["client1"] 
                             for val in encrypted_weights1[layer_idx]]
            
            client2_values = [paillier.EncryptedNumber(self.public_key, int(val)) * client_weights["client2"] 
                             for val in encrypted_weights2[layer_idx]]
            
            client3_values = [paillier.EncryptedNumber(self.public_key, int(val)) * client_weights["client3"] 
                             for val in encrypted_weights3[layer_idx]]
            
            # Perform homomorphic addition
            aggregated_layer = []
            for i in range(len(client1_values)):
                sum_val = client1_values[i] + client2_values[i] + client3_values[i]
                aggregated_layer.append(str(sum_val.ciphertext()))
            
            weighted_aggregated.append(aggregated_layer)
        
        # Server-side: Decrypt weighted aggregated results
        decrypted_weighted = self.server.decrypt_weights(weighted_aggregated)
        
        # Calculate expected weighted average directly
        expected_weighted = []
        for i in range(len(self.client1_weights)):
            expected = (
                self.client1_weights[i] * client_weights["client1"] +
                self.client2_weights[i] * client_weights["client2"] +
                self.client3_weights[i] * client_weights["client3"]
            )
            expected_weighted.append(expected)
        
        # Check results
        for i in range(len(expected_weighted)):
            np.testing.assert_allclose(
                decrypted_weighted[i], 
                expected_weighted[i],
                rtol=1e-4, atol=1e-4,
                err_msg=f"Layer {i} weighted average does not match expected"
            )

if __name__ == '__main__':
    unittest.main() 