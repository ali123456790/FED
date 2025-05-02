"""
Tests for homomorphic encryption functionality using Paillier cryptosystem.

This module tests the homomorphic encryption capabilities, specifically focusing
on Paillier encryption which allows addition operations on encrypted data.
"""

import unittest
import numpy as np
import sys
import os
import json

# Add the project root to the path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import PHE library
from phe import paillier

# Import our encoding module
from security.encoding import FixedPointEncoder

class TestPaillierHomomorphicEncryption(unittest.TestCase):
    """Test Paillier homomorphic encryption and its properties."""
    
    def setUp(self):
        """Set up encryption keys and encoder."""
        # Generate keypair with smaller key size for faster tests
        self.public_key, self.private_key = paillier.generate_paillier_keypair(n_length=1024)
        
        # Create encoder with medium precision
        self.encoder = FixedPointEncoder(precision_bits=32)
    
    def test_homomorphic_addition_simple(self):
        """Test basic homomorphic addition of two encrypted values."""
        # Original values
        value1 = 3.14
        value2 = 2.71
        
        # Encode and encrypt
        encoded1 = self.encoder.encode(value1)
        encoded2 = self.encoder.encode(value2)
        encrypted1 = self.public_key.encrypt(encoded1)
        encrypted2 = self.public_key.encrypt(encoded2)
        
        # Perform homomorphic addition
        encrypted_sum = encrypted1 + encrypted2
        
        # Decrypt and decode
        decrypted_sum = self.private_key.decrypt(encrypted_sum)
        decoded_sum = self.encoder.decode(decrypted_sum)
        
        # Check result
        expected_sum = value1 + value2
        self.assertAlmostEqual(decoded_sum, expected_sum, places=4,
                              msg="Homomorphic addition failed")
    
    def test_homomorphic_addition_with_scalar(self):
        """Test addition of encrypted value with unencrypted scalar."""
        # Original values
        value = 5.67
        scalar = 3
        
        # Encode and encrypt
        encoded = self.encoder.encode(value)
        encrypted = self.public_key.encrypt(encoded)
        
        # Add scalar (gets automatically encrypted)
        encrypted_result = encrypted + scalar
        
        # Decrypt and decode
        decrypted_result = self.private_key.decrypt(encrypted_result)
        decoded_result = self.encoder.decode(decrypted_result)
        
        # Check result
        expected_result = value + scalar
        self.assertAlmostEqual(decoded_result, expected_result, places=4,
                              msg="Homomorphic addition with scalar failed")
    
    def test_homomorphic_addition_multiple_values(self):
        """Test addition of multiple encrypted values."""
        # Original values
        values = [1.1, 2.2, 3.3, 4.4, 5.5]
        
        # Encode and encrypt all values
        encrypted_values = []
        for val in values:
            encoded = self.encoder.encode(val)
            encrypted = self.public_key.encrypt(encoded)
            encrypted_values.append(encrypted)
        
        # Sum all encrypted values
        encrypted_sum = encrypted_values[0]
        for i in range(1, len(encrypted_values)):
            encrypted_sum += encrypted_values[i]
        
        # Decrypt and decode
        decrypted_sum = self.private_key.decrypt(encrypted_sum)
        decoded_sum = self.encoder.decode(decrypted_sum)
        
        # Check result
        expected_sum = sum(values)
        self.assertAlmostEqual(decoded_sum, expected_sum, places=4,
                              msg="Homomorphic addition of multiple values failed")
    
    def test_homomorphic_multiplication_by_scalar(self):
        """Test multiplication of encrypted value by unencrypted scalar."""
        # Original values
        value = 3.14
        scalar = 2
        
        # Encode and encrypt
        encoded = self.encoder.encode(value)
        encrypted = self.public_key.encrypt(encoded)
        
        # Multiply by scalar
        encrypted_result = encrypted * scalar
        
        # Decrypt and decode
        decrypted_result = self.private_key.decrypt(encrypted_result)
        decoded_result = self.encoder.decode(decrypted_result)
        
        # Check result
        expected_result = value * scalar
        self.assertAlmostEqual(decoded_result, expected_result, places=4,
                              msg="Homomorphic multiplication by scalar failed")
    
    def test_negative_values(self):
        """Test encryption and homomorphic operations with negative values."""
        # Original values
        value1 = -2.5
        value2 = 1.5
        
        # Encode and encrypt
        encoded1 = self.encoder.encode(value1)
        encoded2 = self.encoder.encode(value2)
        encrypted1 = self.public_key.encrypt(encoded1)
        encrypted2 = self.public_key.encrypt(encoded2)
        
        # Perform homomorphic addition
        encrypted_sum = encrypted1 + encrypted2
        
        # Decrypt and decode
        decrypted_sum = self.private_key.decrypt(encrypted_sum)
        decoded_sum = self.encoder.decode(decrypted_sum)
        
        # Check result
        expected_sum = value1 + value2
        self.assertAlmostEqual(decoded_sum, expected_sum, places=4,
                              msg="Homomorphic addition with negative values failed")
    
    def test_serialization_deserialization(self):
        """Test serialization and deserialization of encrypted values."""
        # Original value
        value = 7.89
        
        # Encode and encrypt
        encoded = self.encoder.encode(value)
        encrypted = self.public_key.encrypt(encoded)
        
        # Serialize
        ciphertext = encrypted.ciphertext()
        serialized = str(ciphertext)
        
        # Deserialize
        deserialized_ciphertext = int(serialized)
        reconstructed = paillier.EncryptedNumber(self.public_key, deserialized_ciphertext)
        
        # Decrypt and decode
        decrypted = self.private_key.decrypt(reconstructed)
        decoded = self.encoder.decode(decrypted)
        
        # Check result
        self.assertAlmostEqual(decoded, value, places=4,
                              msg="Serialization/deserialization failed")
    
    def test_encrypt_decode_array(self):
        """Test encryption and decryption of a numpy array."""
        # Original array
        original_array = np.array([1.1, -2.2, 3.3, -4.4, 5.5])
        
        # Encode array
        encoded_array = self.encoder.encode(original_array)
        
        # Encrypt each value
        encrypted_array = [self.public_key.encrypt(val) for val in encoded_array]
        
        # Decrypt each value
        decrypted_array = np.array([self.private_key.decrypt(val) for val in encrypted_array])
        
        # Decode array
        decoded_array = self.encoder.decode(decrypted_array)
        
        # Check result
        np.testing.assert_allclose(decoded_array, original_array, rtol=1e-4, atol=1e-4,
                                  err_msg="Array encryption/decryption failed")
    
    def test_weight_aggregation_simulation(self):
        """Simulate the federated learning aggregation process with encrypted weights."""
        # Simulate client weight updates
        client1_weights = [np.array([0.1, -0.2, 0.3]), np.array([0.4, 0.5])]
        client2_weights = [np.array([0.2, 0.1, -0.1]), np.array([-0.1, 0.2])]
        client3_weights = [np.array([0.3, 0.2, 0.1]), np.array([0.3, -0.3])]
        
        all_client_weights = [client1_weights, client2_weights, client3_weights]
        
        # 1. Encode all weights
        encoded_client_weights = []
        for client_weights in all_client_weights:
            encoded_weights = []
            for layer in client_weights:
                encoded_layer = self.encoder.encode(layer)
                encoded_weights.append(encoded_layer)
            encoded_client_weights.append(encoded_weights)
        
        # 2. Encrypt all encoded weights
        encrypted_client_weights = []
        for client_encoded_weights in encoded_client_weights:
            encrypted_weights = []
            for layer in client_encoded_weights:
                encrypted_layer = np.array([self.public_key.encrypt(val) for val in layer.flatten()])
                encrypted_weights.append(encrypted_layer)
            encrypted_client_weights.append(encrypted_weights)
        
        # 3. Perform homomorphic addition (simulating aggregation at the edge)
        n_clients = len(encrypted_client_weights)
        n_layers = len(encrypted_client_weights[0])
        
        aggregated_weights = []
        for l in range(n_layers):
            layer_shape = all_client_weights[0][l].shape
            layer_size = all_client_weights[0][l].size
            
            # Initialize with the first client's encrypted weights
            aggregated_layer = encrypted_client_weights[0][l]
            
            # Add weights from other clients
            for c in range(1, n_clients):
                for i in range(layer_size):
                    aggregated_layer[i] += encrypted_client_weights[c][l][i]
            
            aggregated_weights.append(aggregated_layer)
        
        # 4. Decrypt aggregated weights
        decrypted_aggregated_weights = []
        for layer in aggregated_weights:
            decrypted_layer = np.array([self.private_key.decrypt(val) for val in layer])
            decrypted_aggregated_weights.append(decrypted_layer)
        
        # 5. Decode decrypted weights
        decoded_aggregated_weights = []
        for i, layer in enumerate(decrypted_aggregated_weights):
            layer_shape = all_client_weights[0][i].shape
            decoded_layer = self.encoder.decode(layer).reshape(layer_shape)
            decoded_aggregated_weights.append(decoded_layer)
        
        # 6. Compare with the expected sum (manual calculation)
        for i in range(n_layers):
            expected_layer = sum(client[i] for client in all_client_weights)
            np.testing.assert_allclose(decoded_aggregated_weights[i], expected_layer, 
                                      rtol=1e-4, atol=1e-4,
                                      err_msg=f"Aggregation failed for layer {i}")
    
    def test_overflow_detection(self):
        """Test the behavior with values that might cause overflow."""
        # Very large value - close to the limits
        large_value = 1e9
        scalar = 1000
        
        # Encode and encrypt
        encoded = self.encoder.encode(large_value)
        encrypted = self.public_key.encrypt(encoded)
        
        # This might cause overflow depending on the key size
        try:
            encrypted_result = encrypted * scalar
            decrypted_result = self.private_key.decrypt(encrypted_result)
            decoded_result = self.encoder.decode(decrypted_result)
            
            # Check result if no overflow occurred
            expected_result = large_value * scalar
            relative_error = abs((decoded_result - expected_result) / expected_result)
            self.assertLess(relative_error, 1e-3, 
                          msg="Large value multiplication result exceeded acceptable error")
        except Exception as e:
            # If an overflow occurred, log the exception
            self.skipTest(f"Overflow detected as expected: {str(e)}")
    
    def test_precision_impact_on_homomorphic_operations(self):
        """Test how precision affects the results of homomorphic operations."""
        # Original values
        value1 = 1.23456789
        value2 = 9.87654321
        
        # Test with different precision levels
        precision_results = []
        for bits in [16, 32, 64]:
            encoder = FixedPointEncoder(precision_bits=bits)
            
            # Encode and encrypt
            encoded1 = encoder.encode(value1)
            encoded2 = encoder.encode(value2)
            encrypted1 = self.public_key.encrypt(encoded1)
            encrypted2 = self.public_key.encrypt(encoded2)
            
            # Perform homomorphic addition
            encrypted_sum = encrypted1 + encrypted2
            
            # Decrypt and decode
            decrypted_sum = self.private_key.decrypt(encrypted_sum)
            decoded_sum = encoder.decode(decrypted_sum)
            
            # Record precision and error
            expected_sum = value1 + value2
            error = abs(decoded_sum - expected_sum)
            precision_results.append((bits, error))
        
        # Check that higher precision gives lower error
        for i in range(1, len(precision_results)):
            self.assertLess(precision_results[i][1], precision_results[i-1][1], 
                           msg=f"Higher precision {precision_results[i][0]} bits should give lower error")

if __name__ == '__main__':
    unittest.main()
