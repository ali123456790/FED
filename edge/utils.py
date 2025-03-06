"""
Utility functions for edge computing.
"""

import logging
import numpy as np
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

def aggregate_weights(weights: List[List[np.ndarray]], num_samples: List[int]) -> List[np.ndarray]:
    """
    Aggregate model weights using weighted average.
    
    Args:
        weights: List of model weights from clients
        num_samples: Number of samples for each client
        
    Returns:
        Aggregated weights
    """
    # Ensure weights and num_samples have the same length
    if len(weights) != len(num_samples):
        raise ValueError("Length of weights and num_samples must be the same")
    
    # Calculate the total number of samples
    total_samples = sum(num_samples)
    
    # If there are no samples, return the first set of weights
    if total_samples == 0:
        return weights[0]
    
    # Calculate the weighted average
    weighted_weights = []
    for i in range(len(weights[0])):
        layer_weights = np.zeros_like(weights[0][i])
        for j in range(len(weights)):
            layer_weights += (num_samples[j] / total_samples) * weights[j][i]
        weighted_weights.append(layer_weights)
    
    return weighted_weights

def assign_clients(edge_id: str, assignment_strategy: str, config: Dict) -> List[str]:
    """
    Assign clients to an edge node.
    
    Args:
        edge_id: Unique identifier for the edge node
        assignment_strategy: Strategy for assigning clients
        config: Configuration dictionary
        
    Returns:
        List of client IDs assigned to this edge node
    """
    # This is a placeholder for actual client assignment
    # In a real implementation, you would assign clients based on proximity, resources, etc.
    
    if assignment_strategy == "proximity":
        # Assign clients based on proximity (e.g., network latency)
        # This is a placeholder for actual implementation
        return [f"client_{i}" for i in range(3)]
    elif assignment_strategy == "resource":
        # Assign clients based on resource availability
        # This is a placeholder for actual implementation
        return [f"client_{i}" for i in range(2, 5)]
    elif assignment_strategy == "random":
        # Assign clients randomly
        # This is a placeholder for actual implementation
        import random
        num_clients = 10  # Total number of clients
        num_edge_nodes = config["edge"]["nodes"]
        clients_per_edge = num_clients // num_edge_nodes
        
        # Generate random client IDs
        return [f"client_{i}" for i in range(clients_per_edge)]
    else:
        raise ValueError(f"Unknown assignment strategy: {assignment_strategy}")

def monitor_edge_resources() -> Dict:
    """
    Monitor edge node resources.
    
    Returns:
        Dictionary with resource usage information
    """
    import psutil
    
    # Get CPU usage
    cpu_percent = psutil.cpu_percent(interval=1)
    
    # Get memory usage
    memory = psutil.virtual_memory()
    memory_percent = memory.percent
    
    # Get disk usage
    disk = psutil.disk_usage('/')
    disk_percent = disk.percent
    
    return {
        "cpu_percent": cpu_percent,
        "memory_percent": memory_percent,
        "disk_percent": disk_percent,
    }

def calculate_network_latency(client_id: str) -> float:
    """
    Calculate network latency to a client.
    
    Args:
        client_id: ID of the client
        
    Returns:
        Network latency in milliseconds
    """
    # This is a placeholder for actual network latency calculation
    # In a real implementation, you would ping the client or use other methods
    
    # Simulate network latency
    import random
    return random.uniform(10, 100)  # Latency between 10ms and 100ms

