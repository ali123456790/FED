"""
Tests for the fixed-point encoding and decoding functionality.

This module tests the FixedPointEncoder class which is used to convert
floating point values to integers for Paillier homomorphic encryption.
"""

import unittest
import numpy as np
import sys
import os

# Add the project root to the path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import directly from the encoding module, bypassing security/__init__.py
from security.encoding import FixedPointEncoder, encode_float, decode_integer

class TestFixedPointEncoding(unittest.TestCase):
    """Test fixed-point encoding and decoding for Paillier encryption."""
    
    def setUp(self):
        """Set up encoder instances with different precision levels."""
        self.encoder_low = FixedPointEncoder(precision_bits=16)
        self.encoder_medium = FixedPointEncoder(precision_bits=32)
        self.encoder_high = FixedPointEncoder(precision_bits=64)
    
    def test_encode_decode_single_value(self):
        """Test encoding and decoding a single float value."""
        original_value = 3.14159
        
        # Test with different precision levels
        for encoder in [self.encoder_low, self.encoder_medium, self.encoder_high]:
            encoded = encoder.encode(original_value)
            decoded = encoder.decode(encoded)
            
            # Check that the decoded value is close to the original
            self.assertAlmostEqual(original_value, decoded, places=4,
                                  msg=f"Encoding/decoding failed with precision_bits={encoder.precision_bits}")
    
    def test_encode_decode_array(self):
        """Test encoding and decoding numpy arrays."""
        original_array = np.array([1.23, -4.56, 7.89, 0.0, 100.001])
        
        # Test with different precision levels
        for encoder in [self.encoder_low, self.encoder_medium, self.encoder_high]:
            encoded_array = encoder.encode(original_array)
            decoded_array = encoder.decode(encoded_array)
            
            # Check that each decoded value is close to its original
            np.testing.assert_allclose(original_array, decoded_array, rtol=1e-4, atol=1e-4,
                                      err_msg=f"Array encoding/decoding failed with precision_bits={encoder.precision_bits}")
    
    def test_encode_decode_matrix(self):
        """Test encoding and decoding multi-dimensional arrays."""
        original_matrix = np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6], [7.7, 8.8, 9.9]])
        
        for encoder in [self.encoder_low, self.encoder_medium, self.encoder_high]:
            encoded_matrix = encoder.encode(original_matrix)
            decoded_matrix = encoder.decode(encoded_matrix)
            
            # Check that the shape is preserved
            self.assertEqual(original_matrix.shape, encoded_matrix.shape)
            self.assertEqual(original_matrix.shape, decoded_matrix.shape)
            
            # Check that values are close
            np.testing.assert_allclose(original_matrix, decoded_matrix, rtol=1e-4, atol=1e-4,
                                      err_msg=f"Matrix encoding/decoding failed with precision_bits={encoder.precision_bits}")
    
    def test_encode_large_values(self):
        """Test encoding and decoding large values."""
        large_values = np.array([1e6, 1e9, 1e12])
        
        for encoder in [self.encoder_medium, self.encoder_high]: # Skip low precision for large values
            encoded = encoder.encode(large_values)
            decoded = encoder.decode(encoded)
            
            # Check relative error for large values
            relative_error = np.abs((large_values - decoded) / large_values)
            self.assertTrue(np.all(relative_error < 1e-4),
                          msg=f"Large value encoding failed with precision_bits={encoder.precision_bits}")
    
    def test_encode_small_values(self):
        """Test encoding and decoding very small values."""
        small_values = np.array([1e-6, 1e-9, 1e-12])
        
        for encoder in [self.encoder_medium, self.encoder_high]: # Skip low precision for small values
            encoded = encoder.encode(small_values)
            decoded = encoder.decode(encoded)
            
            # For very small values, check absolute error
            absolute_error = np.abs(small_values - decoded)
            self.assertTrue(np.all(absolute_error < 1e-11),
                          msg=f"Small value encoding failed with precision_bits={encoder.precision_bits}")
    
    def test_encode_weights(self):
        """Test encoding a list of weight arrays like in neural networks."""
        # Simulate neural network weights
        weights = [
            np.random.randn(10, 5),  # Layer 1 weights
            np.random.randn(5),      # Layer 1 bias
            np.random.randn(5, 3),   # Layer 2 weights
            np.random.randn(3)       # Layer 2 bias
        ]
        
        for encoder in [self.encoder_low, self.encoder_medium, self.encoder_high]:
            encoded_weights = encoder.encode_weights(weights)
            decoded_weights = encoder.decode_weights(encoded_weights)
            
            # Check that we have the same number of layers
            self.assertEqual(len(weights), len(decoded_weights))
            
            # Check each layer's shape and values
            for orig_layer, decoded_layer in zip(weights, decoded_weights):
                self.assertEqual(orig_layer.shape, decoded_layer.shape)
                np.testing.assert_allclose(orig_layer, decoded_layer, rtol=1e-3, atol=1e-3)
    
    def test_encode_decode_edge_cases(self):
        """Test encoding and decoding edge cases."""
        edge_cases = np.array([0.0, -0.0, np.inf, -np.inf, np.nan])
        
        for encoder in [self.encoder_low, self.encoder_medium, self.encoder_high]:
            # Handle special values
            with self.assertRaises(ValueError):
                encoder.encode(edge_cases[2:]) # inf and nan should raise ValueError
            
            # Zero should encode/decode correctly
            zero_encoded = encoder.encode(0.0)
            zero_decoded = encoder.decode(zero_encoded)
            self.assertEqual(0.0, zero_decoded)
    
    def test_standalone_functions(self):
        """Test the standalone encode_float and decode_integer functions."""
        original_value = 123.456
        precision_bits = 32
        
        # Encode using standalone function
        encoded = encode_float(original_value, precision_bits)
        
        # Decode using standalone function
        decoded = decode_integer(encoded, precision_bits)
        
        # Check value
        self.assertAlmostEqual(original_value, decoded, places=4)
    
    def test_precision_impact(self):
        """Test how different precision levels affect accuracy."""
        test_value = 1.23456789
        
        # Test increasing precision
        results = []
        for bits in [8, 16, 32, 64]:
            encoder = FixedPointEncoder(precision_bits=bits)
            encoded = encoder.encode(test_value)
            decoded = encoder.decode(encoded)
            results.append((bits, abs(test_value - decoded)))
        
        # Check that higher precision gives lower error
        for i in range(1, len(results)):
            self.assertLess(results[i][1], results[i-1][1], 
                           msg=f"Higher precision {results[i][0]} bits should give lower error than {results[i-1][0]} bits")

if __name__ == '__main__':
    unittest.main() 