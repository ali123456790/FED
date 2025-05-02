"""
Simple test script for the encoding functionality.
"""

import numpy as np
from encoding import FixedPointEncoder, encode_float, decode_integer

# Test single value encoding/decoding
encoder = FixedPointEncoder(precision_bits=32)
original_value = 3.14159
encoded = encoder.encode(original_value)
decoded = encoder.decode(encoded)

print("==== Testing single value encoding/decoding ====")
print(f"Original: {original_value}")
print(f"Encoded: {encoded}")
print(f"Decoded: {decoded}")
print(f"Difference: {abs(original_value - decoded)}")
print(f"Test passed: {abs(original_value - decoded) < 1e-6}")
print("")

# Test array encoding/decoding
print("==== Testing array encoding/decoding ====")
original_array = np.array([1.23, -4.56, 7.89, 0.0, 100.001])
print(f"Original array: {original_array}")

try:
    # Try to encode the array
    encoded_array = encoder.encode(original_array)
    print(f"Encoded array: {encoded_array}")
    
    # Try to decode the array
    decoded_array = encoder.decode(encoded_array)
    print(f"Decoded array: {decoded_array}")
    
    # Check differences
    differences = np.abs(original_array - decoded_array)
    print(f"Max difference: {np.max(differences)}")
    print(f"Test passed: {np.all(differences < 1e-6)}")
except Exception as e:
    print(f"Error: {e}")
    
    # Try element-wise encoding/decoding
    print("\nFalling back to element-wise encoding/decoding:")
    encoded_list = [encoder.encode(float(val)) for val in original_array]
    print(f"Encoded list: {encoded_list}")
    
    decoded_list = [encoder.decode(val) for val in encoded_list]
    print(f"Decoded list: {decoded_list}")
    
    # Convert back to numpy array for comparison
    decoded_array = np.array(decoded_list)
    differences = np.abs(original_array - decoded_array)
    print(f"Max difference: {np.max(differences)}")
    print(f"Test passed: {np.all(differences < 1e-6)}")

# Test standalone functions
print("\n==== Testing standalone functions ====")
original_value = 123.456
precision_bits = 32
encoded = encode_float(original_value, precision_bits)
decoded = decode_integer(encoded, precision_bits)
print(f"Original: {original_value}")
print(f"Encoded: {encoded}")
print(f"Decoded: {decoded}")
print(f"Difference: {abs(original_value - decoded)}")
print(f"Test passed: {abs(original_value - decoded) < 1e-6}") 