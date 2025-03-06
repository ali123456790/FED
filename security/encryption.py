"""
Encryption mechanisms for secure communication in federated learning.

This module provides encryption utilities for protecting data in transit
and at rest during the federated learning process, supporting both TLS
and custom encryption methods.
"""

import logging
import os
import json
import base64
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union, List
from datetime import datetime, timedelta

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

# For handling certificates and TLS
import ssl

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
        
        # Initialize keys and certificates
        if self.enabled:
            if self.encryption_type == "tls":
                # TLS/SSL is handled by the communication layer
                self._check_certificates()
            elif self.encryption_type == "custom":
                # Generate or load RSA key pair
                self._init_custom_encryption()
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
        
        Returns:
            Tuple of (private_key, public_key)
        """
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
    
    def decrypt_model_weights(self, encrypted_weights: bytes) -> List[np.ndarray]:
        """
        Decrypt model weights received from clients.
        
        Args:
            encrypted_weights: Encrypted weights
            
        Returns:
            List of decrypted model weights as numpy arrays
        """
        if not self.enabled:
            # Return deserialized weights if encryption is disabled
            import pickle
            return pickle.loads(encrypted_weights)
        
        try:
            # Decrypt the weights
            decrypted_data = self.decrypt_data(encrypted_weights)
            
            # Deserialize the weights
            import pickle
            return pickle.loads(decrypted_data)
            
        except Exception as e:
            logger.error(f"Error decrypting weights: {e}")
            # Try to deserialize without decryption as fallback
            try:
                import pickle
                return pickle.loads(encrypted_weights)
            except:
                logger.error("Failed to deserialize weights after decryption error")
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