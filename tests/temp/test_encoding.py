"""
Simplified tests for the fixed-point encoding functionality.
"""

import unittest
import numpy as np
import sys
import os

# Add the temp directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import directly from our local copy
from encoding import FixedPointEncoder, encode_float, decode_integer

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

if __name__ == '__main__':
    unittest.main() 