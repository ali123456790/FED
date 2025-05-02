"""
Encryption mechanisms for secure communication in federated learning.

This module provides encryption utilities for protecting data in transit
and at rest during the federated learning process, supporting both TLS,
custom encryption methods, and Paillier homomorphic encryption.
"""

import logging
import os
import json
import base64
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union, List
from datetime import datetime, timedelta
import numpy as np

# Cryptographic libraries
from cryptography.hazmat.primitives.asymmetric import rsa, padding, ec
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.fernet import Fernet
from cryptography.x509 import load_pem_x509_certificate
from cryptography import x509
from cryptography.x509.oid import NameOID

# For Paillier homomorphic encryption
import phe
from phe import paillier

# For handling certificates and TLS
import ssl

# Import encoder for Paillier
from .encoding import FixedPointEncoder

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
        self.encryption_config = config.get("encryption", {})
        self.enabled = self.encryption_config.get("enabled", False)
        self.encryption_type = self.encryption_config.get("type", "tls")
        
        # Paths for certificates and keys
        self.cert_path = self.encryption_config.get("cert_path", "./certificates")
        Path(self.cert_path).mkdir(exist_ok=True, parents=True)
        
        # Key storage
        self._private_key = None
        self._public_key = None
        self._symmetric_key = None
        self._fernet = None
        
        # Paillier specific variables
        self._paillier_public_key = None
        self._paillier_private_key = None
        self._encoder = None
        
        # Initialize keys and certificates
        if self.enabled:
            if self.encryption_type == "tls":
                # TLS/SSL is handled by the communication layer
                self._check_certificates()
            elif self.encryption_type == "custom":
                # Generate or load RSA key pair
                self._init_custom_encryption()
            elif self.encryption_type == "paillier":
                # Initialize Paillier homomorphic encryption
                self._init_paillier_encryption()
            else:
                raise ValueError(f"Unknown encryption type: {self.encryption_type}")
            
            logger.info(f"Encryption initialized with type={self.encryption_type}")
        else:
            logger.info("Encryption is disabled")

    def _check_certificates(self) -> bool:
        """
        Check if TLS certificates exist and are valid.
        
        Returns:
            True if certificates are valid, False otherwise
        """
        # Certificate files
        ca_cert = os.path.join(self.cert_path, "ca.crt")
        server_cert = os.path.join(self.cert_path, "server.crt")
        server_key = os.path.join(self.cert_path, "server.key")
        
        # Check if files exist
        if not all(os.path.exists(f) for f in [ca_cert, server_cert, server_key]):
            logger.warning("TLS certificates not found, secure communication will not be possible")
            return False
        
        try:
            # Check certificate validity
            with open(server_cert, "rb") as f:
                cert_data = f.read()
                cert = load_pem_x509_certificate(cert_data)
                
                # Check expiration
                if datetime.utcnow() > cert.not_valid_after:
                    logger.warning("Server certificate has expired")
                    return False
                
                # Check if it's valid soon
                if datetime.utcnow() + timedelta(days=30) > cert.not_valid_after:
                    logger.warning("Server certificate will expire soon")
            
            logger.info("TLS certificates are valid")
            return True
            
        except Exception as e:
            logger.error(f"Error checking certificates: {e}")
            return False
    
    def _init_custom_encryption(self) -> None:
        """Initialize custom encryption with RSA and AES."""
        # This method is retained for backward compatibility but marked as deprecated
        logger.warning("RSA-based custom encryption is deprecated in favor of Paillier homomorphic encryption")
        try:
            # Try to load existing keys
            private_key_path = os.path.join(self.cert_path, "private_key.pem")
            public_key_path = os.path.join(self.cert_path, "public_key.pem")
            
            if os.path.exists(private_key_path) and os.path.exists(public_key_path):
                # Load existing keys
                with open(private_key_path, "rb") as f:
                    self._private_key = serialization.load_pem_private_key(
                        f.read(),
                        password=None
                    )
                
                with open(public_key_path, "rb") as f:
                    self._public_key = serialization.load_pem_public_key(
                        f.read()
                    )
                
                logger.info("Loaded existing encryption keys")
            else:
                # Generate new keys
                self._private_key, self._public_key = self._generate_rsa_key_pair()
                
                # Save keys
                with open(private_key_path, "wb") as f:
                    f.write(self._private_key.private_bytes(
                        encoding=serialization.Encoding.PEM,
                        format=serialization.PrivateFormat.PKCS8,
                        encryption_algorithm=serialization.NoEncryption()
                    ))
                
                with open(public_key_path, "wb") as f:
                    f.write(self._public_key.public_bytes(
                        encoding=serialization.Encoding.PEM,
                        format=serialization.PublicFormat.SubjectPublicKeyInfo
                    ))
                
                logger.info("Generated and saved new encryption keys")
            
            # Generate symmetric key for faster encryption of large data
            self._symmetric_key = os.urandom(32)  # 256-bit key
            self._fernet = Fernet(base64.urlsafe_b64encode(self._symmetric_key))
            
        except Exception as e:
            logger.error(f"Error initializing custom encryption: {e}")
            raise
    
    def _generate_rsa_key_pair(self) -> Tuple[rsa.RSAPrivateKey, rsa.RSAPublicKey]:
        """
        Generate RSA key pair.
        
        Note: This method is deprecated in favor of Paillier homomorphic encryption.
        
        Returns:
            Tuple of (private_key, public_key)
        """
        logger.warning("RSA key generation is deprecated in favor of Paillier homomorphic encryption")
        key_size = self.encryption_config.get("rsa_key_size", 2048)
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=key_size
        )
        public_key = private_key.public_key()
        
        return private_key, public_key
    
    def encrypt_data(self, data: Union[bytes, str, Dict], public_key: Any = None) -> bytes:
        """
        Encrypt data using the appropriate encryption method.
        
        Args:
            data: Data to encrypt (bytes, string, or dictionary)
            public_key: Public key to use for encryption (if None, use self.public_key)
            
        Returns:
            Encrypted data as bytes
        """
        if not self.enabled:
            # Return plaintext if encryption is disabled
            if isinstance(data, str):
                return data.encode()
            elif isinstance(data, dict):
                return json.dumps(data).encode()
            return data
        
        try:
            # Convert data to bytes if needed
            if isinstance(data, str):
                data_bytes = data.encode()
            elif isinstance(data, dict):
                data_bytes = json.dumps(data).encode()
            else:
                data_bytes = data
            
            if self.encryption_type == "tls":
                # TLS/SSL is handled by the communication layer
                return data_bytes
            elif self.encryption_type == "custom":
                # For small data, use RSA directly
                if len(data_bytes) < 200:
                    if public_key is None:
                        public_key = self._public_key
                    
                    return public_key.encrypt(
                        data_bytes,
                        padding.OAEP(
                            mgf=padding.MGF1(algorithm=hashes.SHA256()),
                            algorithm=hashes.SHA256(),
                            label=None
                        )
                    )
                else:
                    # For larger data, use hybrid encryption (RSA + AES)
                    # First, encrypt the data with AES
                    if self._fernet:
                        encrypted_data = self._fernet.encrypt(data_bytes)
                        
                        # Then encrypt the AES key with RSA
                        if public_key is None:
                            public_key = self._public_key
                        
                        encrypted_key = public_key.encrypt(
                            self._symmetric_key,
                            padding.OAEP(
                                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                                algorithm=hashes.SHA256(),
                                label=None
                            )
                        )
                        
                        # Combine encrypted key and data
                        key_size = len(encrypted_key).to_bytes(4, byteorder='big')
                        return key_size + encrypted_key + encrypted_data
                    else:
                        raise ValueError("Symmetric encryption not initialized")
            else:
                raise ValueError(f"Unknown encryption type: {self.encryption_type}")
                
        except Exception as e:
            logger.error(f"Error encrypting data: {e}")
            # Return plaintext in case of error (with warning)
            logger.warning("Returning unencrypted data due to encryption error")
            if isinstance(data, str):
                return data.encode()
            elif isinstance(data, dict):
                return json.dumps(data).encode()
            return data
    
    def decrypt_data(self, encrypted_data: bytes) -> bytes:
        """
        Decrypt data using the appropriate encryption method.
        
        Args:
            encrypted_data: Encrypted data
            
        Returns:
            Decrypted data as bytes
        """
        if not self.enabled:
            # Return as-is if encryption is disabled
            return encrypted_data
        
        try:
            if self.encryption_type == "tls":
                # TLS/SSL is handled by the communication layer
                return encrypted_data
            elif self.encryption_type == "custom":
                # Determine if this is direct RSA or hybrid encryption
                if len(encrypted_data) <= 256:  # RSA-2048 output size
                    # Direct RSA decryption
                    return self._private_key.decrypt(
                        encrypted_data,
                        padding.OAEP(
                            mgf=padding.MGF1(algorithm=hashes.SHA256()),
                            algorithm=hashes.SHA256(),
                            label=None
                        )
                    )
                else:
                    # Hybrid decryption (RSA + AES)
                    # Extract key size, encrypted key, and encrypted data
                    key_size = int.from_bytes(encrypted_data[:4], byteorder='big')
                    encrypted_key = encrypted_data[4:4+key_size]
                    encrypted_data = encrypted_data[4+key_size:]
                    
                    # Decrypt the symmetric key
                    key = self._private_key.decrypt(
                        encrypted_key,
                        padding.OAEP(
                            mgf=padding.MGF1(algorithm=hashes.SHA256()),
                            algorithm=hashes.SHA256(),
                            label=None
                        )
                    )
                    
                    # Decrypt the data with the symmetric key
                    f = Fernet(base64.urlsafe_b64encode(key))
                    return f.decrypt(encrypted_data)
            else:
                raise ValueError(f"Unknown encryption type: {self.encryption_type}")
                
        except Exception as e:
            logger.error(f"Error decrypting data: {e}")
            # Return encrypted data in case of error (with warning)
            logger.warning("Returning encrypted data due to decryption error")
            return encrypted_data
    
    def encrypt_model_weights(self, weights: List[np.ndarray], public_key: Any = None) -> bytes:
        """
        Encrypt model weights for secure transport.
        
        Args:
            weights: List of model weights as numpy arrays
            public_key: Public key to use for encryption (if None, use self.public_key)
            
        Returns:
            Encrypted weights as bytes
        """
        if not self.enabled:
            # Return serialized weights if encryption is disabled
            import pickle
            return pickle.dumps(weights)
        
        try:
            # Serialize weights
            import pickle
            serialized_weights = pickle.dumps(weights)
            
            # Encrypt the serialized weights
            return self.encrypt_data(serialized_weights, public_key)
            
        except Exception as e:
            logger.error(f"Error encrypting weights: {e}")
            # Return serialized weights in case of error (with warning)
            logger.warning("Returning unencrypted weights due to encryption error")
            import pickle
            return pickle.dumps(weights)
    
    def decrypt_model_weights(self, encrypted_data: bytes) -> List[np.ndarray]:
        """
        Decrypt model weights received from clients.
        
        Args:
            encrypted_data: Encrypted model weights
            
        Returns:
            Decrypted model weights as list of numpy arrays
        """
        if not self.enabled:
            # Return as-is if encryption is disabled
            import pickle
            return pickle.loads(encrypted_data)
        
        try:
            # Decrypt data
            decrypted_data = self.decrypt_data(encrypted_data)
            
            # Deserialize weights
            import pickle
            return pickle.loads(decrypted_data)
            
        except Exception as e:
            logger.error(f"Error decrypting weights: {e}")
            # Try to deserialize as-is in case of error
            try:
                import pickle
                return pickle.loads(encrypted_data)
            except:
                logger.error("Could not deserialize weights after decryption error")
                return []
    
    def serialize_public_key(self) -> bytes:
        """
        Serialize public key for distribution to clients.
        
        Returns:
            Serialized public key as bytes
        """
        if not self.enabled or self.encryption_type == "tls":
            # No key serialization needed for TLS
            return b""
        
        try:
            return self._public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
        except Exception as e:
            logger.error(f"Error serializing public key: {e}")
            return b""
    
    def deserialize_public_key(self, serialized_key: bytes) -> Any:
        """
        Deserialize public key received from server or clients.
        
        Args:
            serialized_key: Serialized public key
            
        Returns:
            Deserialized public key
        """
        if not self.enabled or self.encryption_type == "tls":
            # No key deserialization needed for TLS
            return None
        
        try:
            return serialization.load_pem_public_key(serialized_key)
        except Exception as e:
            logger.error(f"Error deserializing public key: {e}")
            return None
    
    def get_ssl_context(self, is_server: bool = False) -> Optional[ssl.SSLContext]:
        """
        Get SSL context for TLS connections.
        
        Args:
            is_server: Whether this is for a server (True) or client (False)
            
        Returns:
            SSL context for secure communication, or None if not available
        """
        if not self.enabled or self.encryption_type != "tls":
            return None
        
        try:
            if is_server:
                # Server-side SSL context
                server_cert = os.path.join(self.cert_path, "server.crt")
                server_key = os.path.join(self.cert_path, "server.key")
                ca_cert = os.path.join(self.cert_path, "ca.crt")
                
                if not all(os.path.exists(f) for f in [server_cert, server_key, ca_cert]):
                    logger.warning("Server certificates not found, TLS not available")
                    return None
                
                context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
                context.load_cert_chain(certfile=server_cert, keyfile=server_key)
                context.load_verify_locations(cafile=ca_cert)
                context.verify_mode = ssl.CERT_REQUIRED
                
                return context
            else:
                # Client-side SSL context
                client_cert = os.path.join(self.cert_path, "client.crt")
                client_key = os.path.join(self.cert_path, "client.key")
                ca_cert = os.path.join(self.cert_path, "ca.crt")
                
                if not all(os.path.exists(f) for f in [client_cert, client_key, ca_cert]):
                    if os.path.exists(ca_cert):
                        # If only CA cert exists, create context with server verification only
                        context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
                        context.load_verify_locations(cafile=ca_cert)
                        return context
                    else:
                        logger.warning("Client certificates not found, TLS not available")
                        return None
                
                context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
                context.load_cert_chain(certfile=client_cert, keyfile=client_key)
                context.load_verify_locations(cafile=ca_cert)
                
                return context
                
        except Exception as e:
            logger.error(f"Error creating SSL context: {e}")
            return None
    
    def generate_certificates(
        self, 
        common_name: str = "localhost", 
        organization: str = "FIDS", 
        validity_days: int = 365
    ) -> bool:
        """
        Generate self-signed certificates for development/testing.
        
        Args:
            common_name: Common name for the certificate (hostname)
            organization: Organization name
            validity_days: Certificate validity in days
            
        Returns:
            True if certificates were generated successfully, False otherwise
        """
        try:
            # Create certificate directory
            os.makedirs(self.cert_path, exist_ok=True)
            
            # Generate CA key and certificate
            ca_key = ec.generate_private_key(ec.SECP384R1())
            ca_subject = x509.Name([
                x509.NameAttribute(NameOID.COMMON_NAME, f"FIDS CA"),
                x509.NameAttribute(NameOID.ORGANIZATION_NAME, organization)
            ])
            
            ca_cert = x509.CertificateBuilder().subject_name(
                ca_subject
            ).issuer_name(
                ca_subject
            ).public_key(
                ca_key.public_key()
            ).serial_number(
                x509.random_serial_number()
            ).not_valid_before(
                datetime.utcnow()
            ).not_valid_after(
                datetime.utcnow() + timedelta(days=validity_days * 2)  # CA valid for twice as long
            ).add_extension(
                x509.BasicConstraints(ca=True, path_length=None), critical=True
            ).sign(ca_key, hashes.SHA256())
            
            # Write CA certificate
            with open(os.path.join(self.cert_path, "ca.crt"), "wb") as f:
                f.write(ca_cert.public_bytes(serialization.Encoding.PEM))
            
            # Write CA key
            with open(os.path.join(self.cert_path, "ca.key"), "wb") as f:
                f.write(ca_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption()
                ))
            
            # Generate server key and certificate
            server_key = ec.generate_private_key(ec.SECP384R1())
            server_subject = x509.Name([
                x509.NameAttribute(NameOID.COMMON_NAME, common_name),
                x509.NameAttribute(NameOID.ORGANIZATION_NAME, organization)
            ])
            
            server_cert = x509.CertificateBuilder().subject_name(
                server_subject
            ).issuer_name(
                ca_subject
            ).public_key(
                server_key.public_key()
            ).serial_number(
                x509.random_serial_number()
            ).not_valid_before(
                datetime.utcnow()
            ).not_valid_after(
                datetime.utcnow() + timedelta(days=validity_days)
            ).add_extension(
                x509.SubjectAlternativeName([
                    x509.DNSName(common_name),
                    x509.DNSName("localhost")
                ]),
                critical=False
            ).sign(ca_key, hashes.SHA256())
            
            # Write server certificate
            with open(os.path.join(self.cert_path, "server.crt"), "wb") as f:
                f.write(server_cert.public_bytes(serialization.Encoding.PEM))
            
            # Write server key
            with open(os.path.join(self.cert_path, "server.key"), "wb") as f:
                f.write(server_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption()
                ))
            
            # Generate client key and certificate
            client_key = ec.generate_private_key(ec.SECP384R1())
            client_subject = x509.Name([
                x509.NameAttribute(NameOID.COMMON_NAME, f"client"),
                x509.NameAttribute(NameOID.ORGANIZATION_NAME, organization)
            ])
            
            client_cert = x509.CertificateBuilder().subject_name(
                client_subject
            ).issuer_name(
                ca_subject
            ).public_key(
                client_key.public_key()
            ).serial_number(
                x509.random_serial_number()
            ).not_valid_before(
                datetime.utcnow()
            ).not_valid_after(
                datetime.utcnow() + timedelta(days=validity_days)
            ).sign(ca_key, hashes.SHA256())
            
            # Write client certificate
            with open(os.path.join(self.cert_path, "client.crt"), "wb") as f:
                f.write(client_cert.public_bytes(serialization.Encoding.PEM))
            
            # Write client key
            with open(os.path.join(self.cert_path, "client.key"), "wb") as f:
                f.write(client_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption()
                ))
            
            logger.info(f"Generated certificates in {self.cert_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error generating certificates: {e}")
            return False

    def encrypt_update_hybrid(self, update_bytes: bytes, recipient_public_key: Any = None) -> Tuple[bytes, bytes, bytes]:
        """
        Encrypt model update using hybrid encryption (RSA + AES).
        
        DEPRECATED: This method is deprecated in favor of Paillier homomorphic encryption.
        
        Args:
            update_bytes: Serialized model update as bytes
            recipient_public_key: Recipient's public key (if None, use self.public_key)
            
        Returns:
            Tuple of (encrypted_update, encrypted_aes_key, iv)
        """
        logger.warning("Hybrid RSA/AES encryption is deprecated in favor of Paillier homomorphic encryption")
        
        # Generate random AES key and IV
        aes_key = os.urandom(32)  # 256-bit key
        iv = os.urandom(16)  # 128-bit IV
        
        # Encrypt update with AES
        cipher = Cipher(algorithms.AES(aes_key), modes.CBC(iv))
        encryptor = cipher.encryptor()
        
        # Pad update_bytes to multiple of 16 bytes (AES block size)
        pad_length = 16 - (len(update_bytes) % 16)
        padded_update = update_bytes + bytes([pad_length] * pad_length)
        
        # Encrypt with AES
        encrypted_update = encryptor.update(padded_update) + encryptor.finalize()
        
        # Encrypt AES key with RSA
        if recipient_public_key is None:
            recipient_public_key = self._public_key
        
        encrypted_aes_key = recipient_public_key.encrypt(
            aes_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        return encrypted_update, encrypted_aes_key, iv

    def decrypt_update_hybrid(self, encrypted_update: bytes, encrypted_aes_key: bytes, iv: bytes) -> bytes:
        """
        Decrypt model update using hybrid encryption (RSA + AES).
        
        DEPRECATED: This method is deprecated in favor of Paillier homomorphic encryption.
        
        Args:
            encrypted_update: Encrypted model update
            encrypted_aes_key: Encrypted AES key
            iv: Initialization vector for AES
            
        Returns:
            Decrypted update as bytes
        """
        logger.warning("Hybrid RSA/AES decryption is deprecated in favor of Paillier homomorphic encryption")
        
        # Decrypt AES key with RSA
        aes_key = self._private_key.decrypt(
            encrypted_aes_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        # Decrypt update with AES
        cipher = Cipher(algorithms.AES(aes_key), modes.CBC(iv))
        decryptor = cipher.decryptor()
        padded_update = decryptor.update(encrypted_update) + decryptor.finalize()
        
        # Remove padding
        pad_length = padded_update[-1]
        update_bytes = padded_update[:-pad_length]
        
        return update_bytes

    def _init_paillier_encryption(self) -> None:
        """Initialize Paillier homomorphic encryption."""
        try:
            # Get Paillier configuration
            paillier_config = self.encryption_config.get("paillier", {})
            key_size = paillier_config.get("key_size", 2048)
            precision_bits = paillier_config.get("encoding_precision_bits", 64)
            
            # Initialize fixed-point encoder
            self._encoder = FixedPointEncoder(precision_bits=precision_bits)
            
            # Try to load existing Paillier keys
            paillier_private_key_path = os.path.join(self.cert_path, "paillier_private_key.json")
            paillier_public_key_path = os.path.join(self.cert_path, "paillier_public_key.json")
            
            if os.path.exists(paillier_private_key_path) and os.path.exists(paillier_public_key_path):
                # Load existing keys
                logger.info("Loading existing Paillier keys")
                with open(paillier_private_key_path, "r") as f:
                    private_key_dict = json.load(f)
                    self._paillier_private_key = paillier.PaillierPrivateKey(
                        p=int(private_key_dict["p"]),
                        q=int(private_key_dict["q"])
                    )
                
                with open(paillier_public_key_path, "r") as f:
                    public_key_dict = json.load(f)
                    self._paillier_public_key = paillier.PaillierPublicKey(n=int(public_key_dict["n"]))
            else:
                # Generate new keys
                logger.info(f"Generating new Paillier key pair with key_size={key_size}")
                self._paillier_public_key, self._paillier_private_key = paillier.generate_paillier_keypair(n_length=key_size)
                
                # Save keys
                with open(paillier_private_key_path, "w") as f:
                    json.dump({
                        "p": str(self._paillier_private_key.p),
                        "q": str(self._paillier_private_key.q)
                    }, f)
                
                with open(paillier_public_key_path, "w") as f:
                    json.dump({
                        "n": str(self._paillier_public_key.n)
                    }, f)
                
                logger.info("Generated and saved new Paillier keys")
            
        except Exception as e:
            logger.error(f"Error initializing Paillier encryption: {e}")
            raise

    def encrypt_paillier(self, value: Union[int, float, np.ndarray], public_key=None) -> Any:
        """
        Encrypt a value using Paillier homomorphic encryption.
        
        Args:
            value: Value to encrypt (integer, float, or numpy array)
            public_key: Paillier public key to use (if None, use self._paillier_public_key)
            
        Returns:
            Encrypted value(s) as PaillierEncryptedNumber or list of PaillierEncryptedNumbers
        """
        if not self.enabled or self.encryption_type != "paillier":
            logger.warning("Paillier encryption not enabled or not selected")
            return value
            
        if public_key is None:
            public_key = self._paillier_public_key
            
        try:
            # If it's a numpy array, encrypt each value
            if isinstance(value, np.ndarray):
                # First, ensure values are encoded as integers
                encoded_values = self._encoder.encode(value)
                
                # Flatten array for encryption
                flat_array = encoded_values.flatten()
                
                # Encrypt each value
                encrypted_values = [public_key.encrypt(int(val)) for val in flat_array]
                
                # Return with original shape information for later reconstruction
                return {
                    "encrypted_values": encrypted_values,
                    "original_shape": value.shape
                }
            else:
                # For single values, encode and encrypt
                encoded_value = self._encoder.encode(value)
                return public_key.encrypt(encoded_value)
        except Exception as e:
            logger.error(f"Error in Paillier encryption: {e}")
            raise
            
    def decrypt_paillier(self, encrypted_value: Any) -> Union[float, np.ndarray]:
        """
        Decrypt a Paillier-encrypted value.
        
        Args:
            encrypted_value: PaillierEncryptedNumber, or dictionary with encrypted values and shape
            
        Returns:
            Decrypted value as float or numpy array
        """
        if not self.enabled or self.encryption_type != "paillier":
            logger.warning("Paillier encryption not enabled or not selected")
            return encrypted_value
            
        try:
            # If it's a dictionary with encrypted values and shape, reconstruct the array
            if isinstance(encrypted_value, dict) and "encrypted_values" in encrypted_value:
                # Decrypt each value
                decrypted_values = [self._paillier_private_key.decrypt(val) for val in encrypted_value["encrypted_values"]]
                
                # Reshape to original shape
                decrypted_array = np.array(decrypted_values).reshape(encrypted_value["original_shape"])
                
                # Decode back to floating point
                return self._encoder.decode(decrypted_array)
            else:
                # For single values
                decrypted_value = self._paillier_private_key.decrypt(encrypted_value)
                return self._encoder.decode(decrypted_value)
        except Exception as e:
            logger.error(f"Error in Paillier decryption: {e}")
            raise
            
    def encrypt_weights_paillier(self, weights: List[np.ndarray]) -> List[Dict]:
        """
        Encrypt model weights using Paillier homomorphic encryption.
        
        Args:
            weights: List of numpy arrays representing model weights
            
        Returns:
            List of dictionaries with encrypted weights and shape information
        """
        if not self.enabled or self.encryption_type != "paillier":
            logger.warning("Paillier encryption not enabled or not selected")
            return weights
            
        return [self.encrypt_paillier(w) for w in weights]
        
    def decrypt_weights_paillier(self, encrypted_weights: List[Dict]) -> List[np.ndarray]:
        """
        Decrypt model weights encrypted with Paillier.
        
        Args:
            encrypted_weights: List of dictionaries with encrypted weights and shape information
            
        Returns:
            List of numpy arrays with decrypted weights
        """
        if not self.enabled or self.encryption_type != "paillier":
            logger.warning("Paillier encryption not enabled or not selected")
            return encrypted_weights
            
        return [self.decrypt_paillier(ew) for ew in encrypted_weights]
        
    def homomorphic_add(self, encrypted_values: List[Any]) -> Any:
        """
        Perform homomorphic addition on Paillier-encrypted values.
        
        Args:
            encrypted_values: List of Paillier-encrypted values to add
            
        Returns:
            Sum of encrypted values (still encrypted)
        """
        if not encrypted_values:
            return None
            
        # For single values
        if not isinstance(encrypted_values[0], dict):
            result = encrypted_values[0]
            for val in encrypted_values[1:]:
                result += val
            return result
            
        # For arrays
        if "encrypted_values" in encrypted_values[0]:
            # Get the shape from the first array
            original_shape = encrypted_values[0]["original_shape"]
            
            # Check if all arrays have the same shape
            if not all(ev["original_shape"] == original_shape for ev in encrypted_values):
                raise ValueError("Cannot add encrypted arrays with different shapes")
                
            # Get the number of elements in each array
            num_elements = np.prod(original_shape)
            
            # Initialize result list
            result_values = []
            
            # Add corresponding elements from each array
            for i in range(num_elements):
                # Start with the first array's element
                element_sum = encrypted_values[0]["encrypted_values"][i]
                
                # Add elements from other arrays
                for j in range(1, len(encrypted_values)):
                    element_sum += encrypted_values[j]["encrypted_values"][i]
                    
                result_values.append(element_sum)
                
            # Return with original shape
            return {
                "encrypted_values": result_values,
                "original_shape": original_shape
            }
        
        raise ValueError("Invalid encrypted value format for homomorphic addition")
        
    def homomorphic_add_weights(self, encrypted_weights_list: List[List[Dict]]) -> List[Dict]:
        """
        Perform homomorphic addition on multiple sets of encrypted model weights.
        
        Args:
            encrypted_weights_list: List of encrypted weight lists (from multiple clients)
            
        Returns:
            List of dictionaries with summed encrypted weights
        """
        if not encrypted_weights_list:
            return []
            
        # Get the number of weight arrays (layers) from the first client
        num_layers = len(encrypted_weights_list[0])
        
        # Check if all clients have the same number of layers
        if not all(len(ew) == num_layers for ew in encrypted_weights_list):
            raise ValueError("Cannot add encrypted weights with different numbers of layers")
            
        # Initialize result list
        result_weights = []
        
        # For each layer
        for layer_idx in range(num_layers):
            # Get encrypted weights for this layer from all clients
            layer_weights = [ew[layer_idx] for ew in encrypted_weights_list]
            
            # Add them homomorphically
            summed_layer = self.homomorphic_add(layer_weights)
            
            result_weights.append(summed_layer)
            
        return result_weights
        
    def serialize_paillier_public_key(self) -> str:
        """
        Serialize Paillier public key to string for transmission.
        
        Returns:
            JSON string representation of public key
        """
        if self._paillier_public_key is None:
            raise ValueError("Paillier public key not initialized")
            
        return json.dumps({"n": str(self._paillier_public_key.n)})
        
    def deserialize_paillier_public_key(self, serialized_key: str) -> paillier.PaillierPublicKey:
        """
        Deserialize Paillier public key from string.
        
        Args:
            serialized_key: JSON string representation of public key
            
        Returns:
            PaillierPublicKey object
        """
        key_dict = json.loads(serialized_key)
        return paillier.PaillierPublicKey(n=int(key_dict["n"]))
        
    def serialize_encrypted_value(self, encrypted_value: Any) -> Dict:
        """
        Serialize a Paillier-encrypted value for transmission.
        
        Args:
            encrypted_value: PaillierEncryptedNumber or dictionary with encrypted values
            
        Returns:
            Dictionary representation that can be JSON serialized
        """
        if isinstance(encrypted_value, dict) and "encrypted_values" in encrypted_value:
            # For arrays
            return {
                "encrypted_values": [
                    {"value": str(val.ciphertext()), "exponent": val.exponent}
                    for val in encrypted_value["encrypted_values"]
                ],
                "original_shape": encrypted_value["original_shape"]
            }
        elif hasattr(encrypted_value, "ciphertext"):
            # For single values
            return {
                "value": str(encrypted_value.ciphertext()),
                "exponent": encrypted_value.exponent
            }
        else:
            raise ValueError("Unknown encrypted value type for serialization")
            
    def deserialize_encrypted_value(self, serialized_value: Dict, public_key=None) -> Any:
        """
        Deserialize a Paillier-encrypted value.
        
        Args:
            serialized_value: Dictionary representation of encrypted value
            public_key: Paillier public key to use (if None, use self._paillier_public_key)
            
        Returns:
            PaillierEncryptedNumber or dictionary with encrypted values
        """
        if public_key is None:
            public_key = self._paillier_public_key
            
        if "encrypted_values" in serialized_value:
            # For arrays
            return {
                "encrypted_values": [
                    paillier.EncryptedNumber(
                        public_key,
                        int(val["value"]),
                        int(val["exponent"])
                    )
                    for val in serialized_value["encrypted_values"]
                ],
                "original_shape": tuple(serialized_value["original_shape"])
            }
        elif "value" in serialized_value:
            # For single values
            return paillier.EncryptedNumber(
                public_key,
                int(serialized_value["value"]),
                int(serialized_value["exponent"])
            )
        else:
            raise ValueError("Unknown serialized value format")