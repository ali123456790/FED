"""
Registry for storing and retrieving edge aggregator instances.

This module maintains a global registry of edge aggregator instances,
allowing them to be accessed by their IDs from different parts of the code.
"""

import logging
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)

# Global registry of edge aggregators
_edge_aggregators: Dict[str, Any] = {}

def register(edge_id: str, edge_aggregator: Any) -> None:
    """
    Register an edge aggregator instance.
    
    Args:
        edge_id: The unique ID of the edge aggregator
        edge_aggregator: The edge aggregator instance to register
    """
    global _edge_aggregators
    _edge_aggregators[edge_id] = edge_aggregator
    logger.debug(f"Registered edge aggregator with ID: {edge_id}")

def get(edge_id: str) -> Optional[Any]:
    """
    Get an edge aggregator instance by its ID.
    
    Args:
        edge_id: The unique ID of the edge aggregator
    
    Returns:
        The edge aggregator instance, or None if not found
    """
    return _edge_aggregators.get(edge_id)

def unregister(edge_id: str) -> None:
    """
    Unregister an edge aggregator instance.
    
    Args:
        edge_id: The unique ID of the edge aggregator to unregister
    """
    if edge_id in _edge_aggregators:
        del _edge_aggregators[edge_id]
        logger.debug(f"Unregistered edge aggregator with ID: {edge_id}")

def get_all() -> Dict[str, Any]:
    """
    Get all registered edge aggregator instances.
    
    Returns:
        A dictionary mapping edge IDs to edge aggregator instances
    """
    return _edge_aggregators.copy() 