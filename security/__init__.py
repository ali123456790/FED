"""
Security module for FIDS (Federated IoT Device Security).

This module provides security mechanisms for federated learning including:
- Differential privacy for training
- Encryption for secure communication (including Paillier homomorphic encryption)
- Fixed-point encoding for Paillier encryption
- Secure aggregation for model updates
- Threat detection for poisoning and adversarial attacks
"""

from .differential_privacy import DifferentialPrivacy
from .encryption import Encryption
from .secure_aggregation import SecureAggregation
from .threat_model import ThreatModel
from .security_manager import SecurityManager
from .encoding import FixedPointEncoder

__all__ = [
    'DifferentialPrivacy',
    'Encryption',
    'SecureAggregation',
    'ThreatModel',
    'SecurityManager',
    'FixedPointEncoder'
]