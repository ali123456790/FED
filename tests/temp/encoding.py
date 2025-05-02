"""
Fixed-point encoding utilities for homomorphic encryption.

This module provides utilities for encoding floating-point values to integers
for use with Paillier homomorphic encryption, which operates only on integers.
"""

import logging
import numpy as np
import math
from typing import List, Tuple, Dict, Any, Union

logger = logging.getLogger(__name__)

# Simple function-based encoding/decoding

def encode_float(value: float, precision_bits: int) -> int:
    """Encodes a float to an integer using fixed-point representation."""
    if not isinstance(precision_bits, int) or precision_bits <= 0:
        raise ValueError("precision_bits must be a positive integer")
    factor = 1 << precision_bits # 2**precision_bits
    encoded = int(round(value * factor))
    # Optional: Add checks for overflow based on Paillier key size if needed
    # logger.debug(f"Encoding {value} with {precision_bits} bits -> {encoded}")
    return encoded

def decode_integer(encoded_value: int, precision_bits: int) -> float:
    """Decodes an integer back to a float using fixed-point representation."""
    if not isinstance(precision_bits, int) or precision_bits <= 0:
        raise ValueError("precision_bits must be a positive integer")
    factor = 1 << precision_bits
    value = float(encoded_value) / factor
    # logger.debug(f"Decoding {encoded_value} with {precision_bits} bits -> {value}")
    return value

class FixedPointEncoder:
    """Encode floating point values to integers for homomorphic encryption."""
    
    def __init__(self, precision_bits: int = 64):
        """
        Initialize the encoder with a specific precision.
        
        Args:
            precision_bits: Number of bits to use for encoding precision. 
                           Higher values provide better precision but increase
                           the risk of overflow during homomorphic operations.
        """
        self.precision_bits = precision_bits
        self.scaling_factor = 2**precision_bits
        logger.info(f"Initialized FixedPointEncoder with precision_bits={precision_bits}, scaling_factor={self.scaling_factor}")
    
    def encode(self, value: Union[float, np.ndarray]) -> Union[int, np.ndarray]:
        """
        Encode a floating point value or array to integer(s).
        
        Args:
            value: Float or numpy array of floats to encode
            
        Returns:
            Integer or numpy array of integers
        """
        if isinstance(value, np.ndarray):
            # For arrays, ensure we're working with int64 to avoid overflow
            result = np.round(value * self.scaling_factor).astype(np.int64)
            return result
        else:
            # For single values
            return int(round(value * self.scaling_factor))
    
    def decode(self, encoded_value: Union[int, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Decode an integer or array of integers back to float(s).
        
        Args:
            encoded_value: Integer or numpy array of integers to decode
            
        Returns:
            Float or numpy array of floats
        """
        if isinstance(encoded_value, np.ndarray):
            # For arrays, convert from int64 to float
            return encoded_value.astype(np.float64) / self.scaling_factor
        else:
            # For single values
            return float(encoded_value) / self.scaling_factor
    
    def encode_weights(self, weights: List[np.ndarray]) -> List[np.ndarray]:
        """
        Encode a list of weight arrays (model parameters).
        
        Args:
            weights: List of numpy arrays representing model weights
            
        Returns:
            List of numpy arrays with encoded integer values
        """
        return [self.encode(w) for w in weights]
    
    def decode_weights(self, encoded_weights: List[np.ndarray]) -> List[np.ndarray]:
        """
        Decode a list of encoded weight arrays back to float arrays.
        
        Args:
            encoded_weights: List of numpy arrays with encoded integer values
            
        Returns:
            List of numpy arrays with floating point values
        """
        return [self.decode(w) for w in encoded_weights] 