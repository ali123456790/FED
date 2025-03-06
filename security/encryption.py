"""
Encryption mechanisms for secure communication.
"""

import logging
import os
from typing import Dict, Any, Tuple
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

logger = logging.getLogger(__name__)

class Encryption:
    """Encryption mechanisms for secure communication."""
    
    def __init__(self, config: Dict):
        """
        Initialize encryption with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.encryption_config = config["encryption"]
        self.encryption_type = self.encryption_config["type"]
        
        # Initialize keys
        if self.encryption_type == "tls":
            # TLS/SSL is handled by the communication layer
            pass
        elif self.encryption_type == "custom":
            # Generate RSA key pair
            self.private_key, self.public_key = self._generate_rsa_key_pair()
        else:
            raise ValueError(f"Unknown encryption type: {self.encryption_type}")
        
        logger.info(f"Encryption initialized with type={self.encryption_type}")
    
    def _generate_rsa_key_pair(self) -> Tuple[rsa.RSAPrivateKey, rsa.RSAPublicKey]:
        """
        Generate RSA key pair.
        
        Returns:
            Tuple of (private_key, public_key)
        """
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        public_key = private_key.public_key()
        
        return private_key, public_key
    
    def encrypt_data(self, data: bytes, public_key: Any = None) -> bytes:
        """
        Encrypt data.
        
        Args:
            data: Data to encrypt
            public_key: Public key to use for encryption (if None, use self.public_key)
            
        Returns:
            Encrypted data
        """
        if self.encryption_type == "tls":
            # TLS/SSL is handled by the communication layer
            return data
        elif self.encryption_type == "custom":
            # Use RSA for key exchange and AES for data encryption
            
            # Generate AES key and IV
            aes_key = os.urandom(32)  # 256-bit key
            iv = os.urandom(16)  # 128-bit IV
            
            # Encrypt data with AES
            cipher = Cipher(algorithms.AES(aes_key), modes.CBC(iv))
            encryptor = cipher.encryptor()
            
            # Pad data to multiple of block size
            block_size = 16
            padding_length = block_size - (len(data) % block_size)
            padded_data = data + bytes([padding_length]) * padding_length
            
            # Encrypt data
            encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
            
            # Encrypt AES key with RSA
            if public_key is None:
                public_key = self.public_key
            
            encrypted_key = public_key.encrypt(
                aes_key,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            
            # Combine encrypted key, IV, and encrypted data
            return encrypted_key + iv + encrypted_data
        else:
            raise ValueError(f"Unknown encryption type: {self.encryption_type}")
    
    def decrypt_data(self, encrypted_data: bytes) -> bytes:
        """
        Decrypt data.
        
        Args:
            encrypted_data: Encrypted data
            
        Returns:
            Decrypted data
        """
        if self.encryption_type == "tls":
            # TLS/SSL is handled by the communication layer
            return encrypted_data
        elif self.encryption_type == "custom":
            # Extract encrypted key, IV, and encrypted data
            encrypted_key = encrypted_data[:256]  # 2048-bit RSA key
            iv = encrypted_data[256:272]  # 128-bit IV
            encrypted_data = encrypted_data[272:]
            
            # Decrypt AES key with RSA
            aes_key = self.private_key.decrypt(
                encrypted_key,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            
            # Decrypt data with AES
            cipher = Cipher(algorithms.AES(aes_key), modes.CBC(iv))
            decryptor = cipher.decryptor()
            padded_data = decryptor.update(encrypted_data) + decryptor.finalize()
            
            # Remove padding
            padding_length = padded_data[-1]
            data = padded_data[:-padding_length]
            
            return data
        else:
            raise ValueError(f"Unknown encryption type: {self.encryption_type}")
    
    def serialize_public_key(self) -> bytes:
        """
        Serialize public key.
        
        Returns:
            Serialized public key
        """
        if self.encryption_type == "tls":
            # TLS/SSL is handled by the communication layer
            return b""
        elif self.encryption_type == "custom":
            return self.public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
        else:
            raise ValueError(f"Unknown encryption type: {self.encryption_type}")
    
    def deserialize_public_key(self, serialized_key: bytes) -> Any:
        """
        Deserialize public key.
        
        Args:
            serialized_key: Serialized public key
            
        Returns:
            Deserialized public key
        """
        if self.encryption_type == "tls":
            # TLS/SSL is handled by the communication layer
            return None
        elif self.encryption_type == "custom":
            return serialization.load_pem_public_key(serialized_key)
        else:
            raise ValueError(f"Unknown encryption type: {self.encryption_type}")

