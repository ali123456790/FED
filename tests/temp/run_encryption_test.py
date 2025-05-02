"""
Simple test script for Paillier homomorphic encryption.
"""

import numpy as np
from phe import paillier
from encoding import FixedPointEncoder

# Create encoder
encoder = FixedPointEncoder(precision_bits=32)

# Generate a keypair
print("Generating Paillier keypair...")
public_key, private_key = paillier.generate_paillier_keypair(n_length=1024)
print("Keypair generated.")

# Test basic encryption/decryption
print("\n==== Testing basic encryption/decryption ====")
original_value = 3.14159
encoded = encoder.encode(original_value)
encrypted = public_key.encrypt(encoded)
decrypted = private_key.decrypt(encrypted)
decoded = encoder.decode(decrypted)

print(f"Original: {original_value}")
print(f"Encoded: {encoded}")
print(f"Encrypted (showing ciphertext): {encrypted.ciphertext()}")
print(f"Decrypted: {decrypted}")
print(f"Decoded: {decoded}")
print(f"Difference: {abs(original_value - decoded)}")
print(f"Test passed: {abs(original_value - decoded) < 1e-6}")

# Test homomorphic addition
print("\n==== Testing homomorphic addition ====")
value1 = 3.14
value2 = 2.71

# Encode and encrypt values
encoded1 = encoder.encode(value1)
encoded2 = encoder.encode(value2)
encrypted1 = public_key.encrypt(encoded1)
encrypted2 = public_key.encrypt(encoded2)

# Perform homomorphic addition
encrypted_sum = encrypted1 + encrypted2

# Decrypt and decode
decrypted_sum = private_key.decrypt(encrypted_sum)
decoded_sum = encoder.decode(decrypted_sum)

# Check result
expected_sum = value1 + value2
print(f"Value 1: {value1}")
print(f"Value 2: {value2}")
print(f"Expected sum: {expected_sum}")
print(f"Homomorphic sum: {decoded_sum}")
print(f"Difference: {abs(expected_sum - decoded_sum)}")
print(f"Test passed: {abs(expected_sum - decoded_sum) < 1e-6}")

# Test homomorphic multiplication by scalar
print("\n==== Testing multiplication by scalar ====")
value = 5.0
scalar = 3

# Encode and encrypt
encoded = encoder.encode(value)
encrypted = public_key.encrypt(encoded)

# Multiply by scalar
encrypted_product = encrypted * scalar

# Decrypt and decode
decrypted_product = private_key.decrypt(encrypted_product)
decoded_product = encoder.decode(decrypted_product)

# Check result
expected_product = value * scalar
print(f"Value: {value}")
print(f"Scalar: {scalar}")
print(f"Expected product: {expected_product}")
print(f"Homomorphic product: {decoded_product}")
print(f"Difference: {abs(expected_product - decoded_product)}")
print(f"Test passed: {abs(expected_product - decoded_product) < 1e-6}")

# Test with small array
print("\n==== Testing with small array ====")
original_array = np.array([1.23, -4.56, 7.89])

# Encode array
encoded_array = encoder.encode(original_array)
print(f"Original array: {original_array}")
print(f"Encoded array: {encoded_array}")

# Encrypt each value in the array
encrypted_array = [public_key.encrypt(val) for val in encoded_array]
print(f"Encrypted array (length): {len(encrypted_array)}")

# Sum the encrypted values (homomorphic addition)
encrypted_sum = encrypted_array[0]
for i in range(1, len(encrypted_array)):
    encrypted_sum += encrypted_array[i]

# Decrypt and decode the sum
decrypted_sum = private_key.decrypt(encrypted_sum)
decoded_sum = encoder.decode(decrypted_sum)

# Compare with direct sum
expected_sum = np.sum(original_array)
print(f"Expected sum: {expected_sum}")
print(f"Homomorphic sum: {decoded_sum}")
print(f"Difference: {abs(expected_sum - decoded_sum)}")
print(f"Test passed: {abs(expected_sum - decoded_sum) < 1e-6}")

# Test serialization
print("\n==== Testing serialization ====")
value = 7.89
encoded = encoder.encode(value)
encrypted = public_key.encrypt(encoded)

# Serialize encrypted value
serialized = str(encrypted.ciphertext())
print(f"Serialized ciphertext: {serialized[:50]}... (length: {len(serialized)})")

# Deserialize
deserialized_ciphertext = int(serialized)
reconstructed = paillier.EncryptedNumber(public_key, deserialized_ciphertext)

# Decrypt and decode
decrypted = private_key.decrypt(reconstructed)
decoded = encoder.decode(decrypted)

print(f"Original value: {value}")
print(f"Decoded after serialization: {decoded}")
print(f"Difference: {abs(value - decoded)}")
print(f"Test passed: {abs(value - decoded) < 1e-6}")

print("\nAll tests completed.") 