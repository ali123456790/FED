"""
Edge module for FIDS.

This module contains components for implementing edge/fog nodes in the 
federated learning system for IoT device security.
"""

from .edge_aggregator import EdgeAggregator, EdgeToServerClient
from .utils import (
    aggregate_weights, 
    assign_clients, 
    monitor_edge_resources, 
    calculate_network_latency,
    NetworkTopology
)
from . import edge_aggregator_registry

__all__ = [
    'EdgeAggregator',
    'EdgeToServerClient',
    'aggregate_weights',
    'assign_clients',
    'monitor_edge_resources',
    'calculate_network_latency',
    'NetworkTopology',
    'edge_aggregator_registry'
]