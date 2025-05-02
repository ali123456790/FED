"""
Performance tests for Paillier homomorphic encryption.

This module benchmarks the performance of homomorphic encryption operations
to evaluate computational overhead and scalability.
"""

import unittest
import numpy as np
import time
import sys
import os
import logging
from typing import List, Tuple

# Configure logging
logging.basicConfig(level=logging.ERROR)

# Add the project root to the path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import PHE library
from phe import paillier

# Import our encoding module
from security.encoding import FixedPointEncoder

class TestEncryptionPerformance(unittest.TestCase):
    """Benchmark the performance of Paillier homomorphic encryption operations."""
    
    def setUp(self):
        """Set up encryption keys and encoder."""
        # Generate keypair with different key sizes for benchmarking
        start_time = time.time()
        self.public_key_1024, self.private_key_1024 = paillier.generate_paillier_keypair(n_length=1024)
        self.key_gen_time_1024 = time.time() - start_time
        
        start_time = time.time()
        self.public_key_2048, self.private_key_2048 = paillier.generate_paillier_keypair(n_length=2048)
        self.key_gen_time_2048 = time.time() - start_time
        
        # Create encoders with different precision levels
        self.encoder_32 = FixedPointEncoder(precision_bits=32)
        self.encoder_64 = FixedPointEncoder(precision_bits=64)
        
        # Test data for benchmarks
        self.small_tensor = np.random.randn(10, 10)  # 100 values
        self.medium_tensor = np.random.randn(32, 32)  # 1,024 values
        self.large_tensor = np.random.randn(100, 100)  # 10,000 values
    
    def test_key_generation_time(self):
        """Test the time required to generate Paillier keypairs of different sizes."""
        print(f"\nKey generation time (1024 bits): {self.key_gen_time_1024:.4f} seconds")
        print(f"Key generation time (2048 bits): {self.key_gen_time_2048:.4f} seconds")
        
        # The 2048-bit key should take significantly longer to generate
        self.assertGreater(self.key_gen_time_2048, self.key_gen_time_1024)
    
    def _benchmark_encryption(self, tensor: np.ndarray, public_key, encoder, 
                              key_size: int, precision_bits: int) -> Tuple[float, float, float]:
        """Benchmark encryption, operation, and decryption times."""
        # Flatten the tensor for easier processing
        flat_tensor = tensor.flatten()
        n_elements = flat_tensor.size
        
        # 1. Measure encoding + encryption time
        start_time = time.time()
        encoded_values = encoder.encode(flat_tensor)
        encrypted_values = [public_key.encrypt(val) for val in encoded_values]
        encryption_time = time.time() - start_time
        
        # 2. Measure homomorphic addition time
        # Sum all the values in pairs to benchmark homomorphic operations
        start_time = time.time()
        result = encrypted_values[0]
        for i in range(1, len(encrypted_values)):
            result += encrypted_values[i]
        operation_time = time.time() - start_time
        
        # 3. Measure decryption + decoding time
        start_time = time.time()
        decrypted_value = self.private_key_1024.decrypt(result) if key_size == 1024 else self.private_key_2048.decrypt(result)
        decoded_value = encoder.decode(decrypted_value)
        decryption_time = time.time() - start_time
        
        return encryption_time, operation_time, decryption_time
    
    def test_encryption_performance_by_tensor_size(self):
        """Test encryption performance with different tensor sizes."""
        results = []
        
        # Test with 1024-bit key and 32-bit precision
        for tensor_name, tensor in [
            ("Small (10x10)", self.small_tensor), 
            ("Medium (32x32)", self.medium_tensor), 
            ("Large (100x100)", self.large_tensor)
        ]:
            encryption_time, operation_time, decryption_time = self._benchmark_encryption(
                tensor, self.public_key_1024, self.encoder_32, 1024, 32
            )
            results.append((tensor_name, tensor.size, encryption_time, operation_time, decryption_time))
        
        print("\nEncryption Performance by Tensor Size (1024-bit key, 32-bit precision):")
        print(f"{'Tensor':<15} {'Size':<10} {'Encryption (s)':<15} {'Addition (s)':<15} {'Decryption (s)':<15}")
        print("-" * 70)
        for tensor_name, size, enc_time, op_time, dec_time in results:
            print(f"{tensor_name:<15} {size:<10} {enc_time:<15.4f} {op_time:<15.4f} {dec_time:<15.4f}")
        
        # Verify that larger tensors take longer to process
        self.assertLess(results[0][2], results[2][2], "Larger tensors should take longer to encrypt")
    
    def test_encryption_performance_by_key_size(self):
        """Test encryption performance with different key sizes."""
        results = []
        
        # Test with medium tensor (32x32) and 32-bit precision
        for key_name, public_key, key_size in [
            ("1024-bit", self.public_key_1024, 1024),
            ("2048-bit", self.public_key_2048, 2048)
        ]:
            encryption_time, operation_time, decryption_time = self._benchmark_encryption(
                self.medium_tensor, public_key, self.encoder_32, key_size, 32
            )
            results.append((key_name, encryption_time, operation_time, decryption_time))
        
        print("\nEncryption Performance by Key Size (32x32 tensor, 32-bit precision):")
        print(f"{'Key Size':<10} {'Encryption (s)':<15} {'Addition (s)':<15} {'Decryption (s)':<15}")
        print("-" * 55)
        for key_name, enc_time, op_time, dec_time in results:
            print(f"{key_name:<10} {enc_time:<15.4f} {op_time:<15.4f} {dec_time:<15.4f}")
        
        # Verify that larger keys take longer to process
        self.assertLess(results[0][1], results[1][1], "Larger keys should take longer to encrypt")
        self.assertLess(results[0][3], results[1][3], "Larger keys should take longer to decrypt")
    
    def test_encryption_performance_by_precision(self):
        """Test encryption performance with different precision levels."""
        results = []
        
        # Test with medium tensor (32x32) and 1024-bit key
        for precision_name, encoder, precision_bits in [
            ("32-bit", self.encoder_32, 32),
            ("64-bit", self.encoder_64, 64)
        ]:
            encryption_time, operation_time, decryption_time = self._benchmark_encryption(
                self.medium_tensor, self.public_key_1024, encoder, 1024, precision_bits
            )
            results.append((precision_name, encryption_time, operation_time, decryption_time))
        
        print("\nEncryption Performance by Precision (32x32 tensor, 1024-bit key):")
        print(f"{'Precision':<10} {'Encryption (s)':<15} {'Addition (s)':<15} {'Decryption (s)':<15}")
        print("-" * 55)
        for precision_name, enc_time, op_time, dec_time in results:
            print(f"{precision_name:<10} {enc_time:<15.4f} {op_time:<15.4f} {dec_time:<15.4f}")
    
    def test_scaling_with_number_of_clients(self):
        """Test how performance scales with the number of clients in federated learning."""
        # This simulates Edge aggregation of encrypted client updates
        client_counts = [5, 10, 20]
        results = []
        
        # Use small tensor for faster tests
        tensor_size = 5 * 5  # 25 values
        small_test_tensor = np.random.randn(5, 5)
        
        for num_clients in client_counts:
            # Generate encrypted values for each "client"
            start_time = time.time()
            all_client_encrypted = []
            for _ in range(num_clients):
                # Simulate a client encrypting their weights
                encoded = self.encoder_32.encode(small_test_tensor.flatten())
                encrypted = [self.public_key_1024.encrypt(val) for val in encoded]
                all_client_encrypted.append(encrypted)
            encryption_time = time.time() - start_time
            
            # Simulate aggregation at the edge
            start_time = time.time()
            aggregated = all_client_encrypted[0].copy()
            for client_idx in range(1, num_clients):
                for i in range(len(aggregated)):
                    aggregated[i] += all_client_encrypted[client_idx][i]
            aggregation_time = time.time() - start_time
            
            # Decrypt at the server
            start_time = time.time()
            decrypted = [self.private_key_1024.decrypt(val) for val in aggregated]
            decoded = self.encoder_32.decode(np.array(decrypted))
            decryption_time = time.time() - start_time
            
            results.append((num_clients, encryption_time, aggregation_time, decryption_time))
        
        print("\nPerformance Scaling with Number of Clients (5x5 tensor, 1024-bit key):")
        print(f"{'# Clients':<10} {'Encryption (s)':<15} {'Aggregation (s)':<15} {'Decryption (s)':<15}")
        print("-" * 55)
        for num_clients, enc_time, agg_time, dec_time in results:
            print(f"{num_clients:<10} {enc_time:<15.4f} {agg_time:<15.4f} {dec_time:<15.4f}")
        
        # Verify that aggregation time increases with more clients
        self.assertLess(results[0][2], results[-1][2], "Aggregation time should increase with more clients")
    
    def test_compare_with_plaintext_operations(self):
        """Compare homomorphic operations with regular plaintext operations."""
        # Small tensor for quick testing
        tensor = np.random.randn(10, 10)
        flat_tensor = tensor.flatten()
        
        # 1. Time for plaintext operations
        start_time = time.time()
        # Just sum all values
        plaintext_sum = np.sum(flat_tensor)
        plaintext_time = time.time() - start_time
        
        # 2. Time for homomorphic operations with 1024-bit key
        # Encode and encrypt
        start_time = time.time()
        encoded = self.encoder_32.encode(flat_tensor)
        encrypted = [self.public_key_1024.encrypt(val) for val in encoded]
        
        # Sum all encrypted values
        encrypted_sum = encrypted[0]
        for i in range(1, len(encrypted)):
            encrypted_sum += encrypted[i]
        
        # Decrypt and decode
        decrypted_sum = self.private_key_1024.decrypt(encrypted_sum)
        homomorphic_sum = self.encoder_32.decode(decrypted_sum)
        homomorphic_time = time.time() - start_time
        
        print(f"\nPlaintext vs. Homomorphic Operations (10x10 tensor):")
        print(f"Plaintext sum time: {plaintext_time:.6f} seconds")
        print(f"Homomorphic sum time: {homomorphic_time:.6f} seconds")
        print(f"Overhead factor: {homomorphic_time/plaintext_time:.2f}x")
        
        # Verify correctness
        self.assertAlmostEqual(plaintext_sum, homomorphic_sum, places=3, 
                              msg="Homomorphic sum should equal plaintext sum")
        
        # Homomorphic operations should be significantly slower
        self.assertGreater(homomorphic_time, plaintext_time * 10, 
                          "Homomorphic operations should be at least 10x slower than plaintext")

if __name__ == '__main__':
    unittest.main() 