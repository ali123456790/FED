"""
Utility functions for edge computing.
"""

import logging
import numpy as np
import time
import psutil
import socket
import json
import os
import threading
import random
from typing import Dict, List, Tuple, Optional, Any, Set
import itertools

logger = logging.getLogger(__name__)

class NetworkTopology:
    """Class representing the network topology for edge nodes and clients."""
    
    def __init__(self, edge_id: str):
        """
        Initialize the network topology.
        
        Args:
            edge_id: Unique identifier for the edge node
        """
        self.edge_id = edge_id
        self.clients: Dict[str, Dict] = {}
        self.edges: Dict[str, Dict] = {}
        self.latency_matrix: Dict[str, Dict[str, float]] = {}
        self.topology_file = os.path.join("data", "network_topology.json")
        
        # Load topology if available
        self._load_topology()
    
    def _load_topology(self) -> None:
        """Load network topology from file if available."""
        try:
            if os.path.exists(self.topology_file):
                with open(self.topology_file, 'r') as f:
                    topology = json.load(f)
                
                if "clients" in topology:
                    self.clients = topology["clients"]
                
                if "edges" in topology:
                    self.edges = topology["edges"]
                
                if "latency_matrix" in topology:
                    self.latency_matrix = topology["latency_matrix"]
                
                logger.info(f"Loaded network topology with {len(self.clients)} clients and {len(self.edges)} edges")
        except Exception as e:
            logger.warning(f"Error loading network topology: {e}")
    
    def save_topology(self) -> None:
        """Save network topology to file."""
        try:
            os.makedirs(os.path.dirname(self.topology_file), exist_ok=True)
            
            topology = {
                "clients": self.clients,
                "edges": self.edges,
                "latency_matrix": self.latency_matrix
            }
            
            with open(self.topology_file, 'w') as f:
                json.dump(topology, f, indent=2)
            
            logger.info(f"Saved network topology to {self.topology_file}")
        except Exception as e:
            logger.warning(f"Error saving network topology: {e}")
    
    def add_client(
        self, 
        client_id: str, 
        location: Optional[Tuple[float, float]] = None,
        device_type: str = "unknown"
    ) -> None:
        """
        Add a client to the network topology.
        
        Args:
            client_id: Unique identifier for the client
            location: Optional tuple of (latitude, longitude)
            device_type: Type of device
        """
        if location is None:
            # Generate random location if not provided
            location = (random.uniform(-90, 90), random.uniform(-180, 180))
        
        self.clients[client_id] = {
            "location": location,
            "device_type": device_type,
            "added_at": time.time()
        }
    
    def add_edge(
        self, 
        edge_id: str, 
        location: Optional[Tuple[float, float]] = None,
        capacity: int = 10
    ) -> None:
        """
        Add an edge node to the network topology.
        
        Args:
            edge_id: Unique identifier for the edge node
            location: Optional tuple of (latitude, longitude)
            capacity: Maximum number of clients the edge can handle
        """
        if location is None:
            # Generate random location if not provided
            location = (random.uniform(-90, 90), random.uniform(-180, 180))
        
        self.edges[edge_id] = {
            "location": location,
            "capacity": capacity,
            "added_at": time.time()
        }
    
    def update_latency(self, source_id: str, target_id: str, latency: float) -> None:
        """
        Update the latency between two nodes in the network.
        
        Args:
            source_id: Source node ID
            target_id: Target node ID
            latency: Latency in milliseconds
        """
        if source_id not in self.latency_matrix:
            self.latency_matrix[source_id] = {}
        
        self.latency_matrix[source_id][target_id] = latency
    
    def get_latency(self, source_id: str, target_id: str) -> float:
        """
        Get the latency between two nodes in the network.
        
        Args:
            source_id: Source node ID
            target_id: Target node ID
            
        Returns:
            Latency in milliseconds, or -1 if unknown
        """
        try:
            return self.latency_matrix.get(source_id, {}).get(target_id, -1)
        except Exception:
            return -1
    
    def get_closest_edge(self, client_id: str) -> Optional[str]:
        """
        Get the closest edge node to a client.
        
        Args:
            client_id: Client ID
            
        Returns:
            ID of the closest edge node, or None if no edge nodes
        """
        if not self.edges:
            return None
        
        best_edge = None
        best_latency = float('inf')
        
        for edge_id in self.edges:
            latency = self.get_latency(client_id, edge_id)
            if latency >= 0 and latency < best_latency:
                best_latency = latency
                best_edge = edge_id
        
        return best_edge
    
    def generate_synthetic_topology(
        self, 
        num_clients: int = 20, 
        num_edges: int = 3
    ) -> None:
        """
        Generate a synthetic network topology for testing.
        
        Args:
            num_clients: Number of clients to generate
            num_edges: Number of edge nodes to generate
        """
        # Clear existing topology
        self.clients = {}
        self.edges = {}
        self.latency_matrix = {}
        
        # Generate edge nodes
        for i in range(num_edges):
            edge_id = f"edge_{i}"
            location = (random.uniform(-90, 90), random.uniform(-180, 180))
            self.add_edge(edge_id, location, capacity=num_clients // num_edges * 2)
        
        # Generate clients
        device_types = ["thermostat", "camera", "doorbell", "monitor", "webcam"]
        for i in range(num_clients):
            client_id = f"client_{i}"
            device_type = random.choice(device_types)
            
            # Generate location near an edge node
            edge_id = f"edge_{i % num_edges}"
            edge_location = self.edges[edge_id]["location"]
            
            # Add some random offset (within ~50km)
            location = (
                edge_location[0] + random.uniform(-0.5, 0.5),
                edge_location[1] + random.uniform(-0.5, 0.5)
            )
            
            self.add_client(client_id, location, device_type)
        
        # Generate synthetic latencies
        for client_id, edge_id in itertools.product(self.clients, self.edges):
            # Calculate distance-based latency
            client_location = self.clients[client_id]["location"]
            edge_location = self.edges[edge_id]["location"]
            
            # Simple Euclidean distance (not accurate for geo, but fine for testing)
            distance = ((client_location[0] - edge_location[0])**2 + 
                       (client_location[1] - edge_location[1])**2)**0.5
            
            # Convert distance to latency (1 degree ~ 111km, assume 0.5ms per km + base latency)
            latency = 10 + distance * 111 * 0.5
            
            # Add some random variation
            latency *= random.uniform(0.8, 1.2)
            
            self.update_latency(client_id, edge_id, latency)
            self.update_latency(edge_id, client_id, latency)
        
        # Generate edge-to-edge latencies
        for edge1, edge2 in itertools.combinations(self.edges, 2):
            edge1_location = self.edges[edge1]["location"]
            edge2_location = self.edges[edge2]["location"]
            
            # Simple Euclidean distance
            distance = ((edge1_location[0] - edge2_location[0])**2 + 
                       (edge1_location[1] - edge2_location[1])**2)**0.5
            
            # Convert distance to latency (edge-to-edge is faster)
            latency = 5 + distance * 111 * 0.3
            
            # Add some random variation
            latency *= random.uniform(0.9, 1.1)
            
            self.update_latency(edge1, edge2, latency)
            self.update_latency(edge2, edge1, latency)
        
        # Save the generated topology
        self.save_topology()
        
        logger.info(f"Generated synthetic network topology with {num_clients} clients and {num_edges} edge nodes")

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
        raise ValueError(f"Length of weights ({len(weights)}) and num_samples ({len(num_samples)}) must be the same")
    
    # If no weights, return empty list
    if not weights:
        return []
    
    # Calculate the total number of samples
    total_samples = sum(num_samples)
    
    # If there are no samples, return the first set of weights
    if total_samples == 0:
        logger.warning("Total sample count is 0, returning first set of weights")
        return weights[0]
    
    # Calculate the weighted average
    weighted_weights = []
    
    # Check if all weight lists have the same structure
    if not all(len(w) == len(weights[0]) for w in weights):
        raise ValueError("All weight lists must have the same length")
    
    for i in range(len(weights[0])):
        # Check if shapes match for this layer
        shapes = [w[i].shape for w in weights]
        if not all(s == shapes[0] for s in shapes):
            raise ValueError(f"Shape mismatch for layer {i}: {shapes}")
        
        # Create a zero array with the right shape
        layer_weights = np.zeros_like(weights[0][i])
        
        # Compute weighted average
        for j in range(len(weights)):
            layer_weights += (num_samples[j] / total_samples) * weights[j][i]
        
        weighted_weights.append(layer_weights)
    
    logger.debug(f"Aggregated weights from {len(weights)} clients with {total_samples} samples total")
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
    logger.info(f"Assigning clients to edge {edge_id} using strategy: {assignment_strategy}")
    
    # Load or generate network topology
    topology = NetworkTopology(edge_id)
    
    # If topology is empty, generate synthetic one
    if not topology.clients or not topology.edges:
        num_clients = config.get("num_clients", 20)
        num_edges = config.get("edge", {}).get("nodes", 3)
        topology.generate_synthetic_topology(num_clients, num_edges)
    
    # Check if this edge node exists in the topology
    if edge_id not in topology.edges:
        edge_num = int(edge_id.split('_')[-1]) if '_' in edge_id else 0
        if f"edge_{edge_num}" in topology.edges:
            edge_id = f"edge_{edge_num}"
        else:
            # Add this edge to the topology
            topology.add_edge(edge_id)
            topology.save_topology()
    
    # Get all available clients
    all_clients = list(topology.clients.keys())
    
    # If no clients in topology, create default client list
    if not all_clients:
        all_clients = [f"client_{i}" for i in range(config.get("num_clients", 20))]
    
    # Determine the assignment based on strategy
    if assignment_strategy == "proximity":
        # Assign based on network proximity (latency)
        assigned_clients = []
        
        for client_id in all_clients:
            # Get the closest edge for this client
            closest_edge = topology.get_closest_edge(client_id)
            
            # If this is the closest edge, assign the client
            if closest_edge == edge_id:
                assigned_clients.append(client_id)
            else:
                # If latency information doesn't exist, use a heuristic
                client_to_this_edge = topology.get_latency(client_id, edge_id)
                if client_to_this_edge < 0:  # No latency info
                    # Assign based on client ID and edge ID hash
                    # This ensures consistent assignment without latency info
                    client_num = int(client_id.split('_')[-1]) if '_' in client_id else 0
                    edge_num = int(edge_id.split('_')[-1]) if '_' in edge_id else 0
                    
                    if client_num % config.get("edge", {}).get("nodes", 3) == edge_num:
                        assigned_clients.append(client_id)
        
        return assigned_clients
    
    elif assignment_strategy == "resource":
        # Assign based on resource availability
        edge_capacity = topology.edges.get(edge_id, {}).get("capacity", 10)
        
        # Calculate client resource scores
        client_scores = {}
        for client_id in all_clients:
            # For now, use a random score as placeholder
            # In a real implementation, you would query client resources
            client_scores[client_id] = random.random()
        
        # Sort clients by score (higher score = better resources)
        sorted_clients = sorted(client_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Get edge number from edge_id
        edge_num = int(edge_id.split('_')[-1]) if '_' in edge_id else 0
        num_edges = config.get("edge", {}).get("nodes", 3)
        
        # Assign clients in a round-robin fashion to balance load
        assigned_clients = []
        for i, (client_id, _) in enumerate(sorted_clients):
            if i % num_edges == edge_num:
                assigned_clients.append(client_id)
                if len(assigned_clients) >= edge_capacity:
                    break
        
        return assigned_clients
    
    elif assignment_strategy == "random":
        # Assign clients randomly
        import random
        random.seed(int(edge_id.split('_')[-1]) if '_' in edge_id else 0)
        
        num_clients = len(all_clients)
        num_edges = config.get("edge", {}).get("nodes", 3)
        clients_per_edge = num_clients // num_edges
        
        # Randomly shuffle clients
        shuffled_clients = random.sample(all_clients, len(all_clients))
        
        # Get index for this edge
        edge_index = int(edge_id.split('_')[-1]) if '_' in edge_id else 0
        
        # Assign a slice of clients to this edge
        start_idx = edge_index * clients_per_edge
        end_idx = start_idx + clients_per_edge
        if edge_index == num_edges - 1:
            # Last edge gets any remaining clients
            end_idx = num_clients
        
        return shuffled_clients[start_idx:end_idx]
    
    elif assignment_strategy == "static":
        # Use static assignment defined in config
        edge_assignments = config.get("edge", {}).get("client_assignments", {})
        
        if edge_id in edge_assignments:
            return edge_assignments[edge_id]
        else:
            # Fall back to default assignment
            edge_num = int(edge_id.split('_')[-1]) if '_' in edge_id else 0
            return [client_id for client_id in all_clients 
                   if int(client_id.split('_')[-1]) % config.get("edge", {}).get("nodes", 3) == edge_num]
    
    else:
        raise ValueError(f"Unknown assignment strategy: {assignment_strategy}")

def monitor_edge_resources() -> Dict:
    """
    Monitor edge node resources.
    
    Returns:
        Dictionary with resource usage information
    """
    try:
        # Get CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.5)
        
        # Get memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_available = memory.available
        
        # Get disk usage
        disk = psutil.disk_usage('/')
        disk_percent = disk.percent
        disk_free = disk.free
        
        # Get network I/O stats
        net_io = psutil.net_io_counters()
        
        return {
            "timestamp": time.time(),
            "cpu_percent": cpu_percent,
            "memory_percent": memory_percent,
            "memory_available": memory_available,
            "disk_percent": disk_percent,
            "disk_free": disk_free,
            "net_bytes_sent": net_io.bytes_sent,
            "net_bytes_recv": net_io.bytes_recv,
        }
    except Exception as e:
        logger.error(f"Error monitoring resources: {e}")
        return {
            "timestamp": time.time(),
            "error": str(e)
        }

def calculate_network_latency(client_id: str, num_packets: int = 10, timeout: float = 2.0) -> float:
    """
    Calculate network latency to a client.
    
    Args:
        client_id: ID of the client
        num_packets: Number of packets to send for measurement
        timeout: Timeout for each packet in seconds
        
    Returns:
        Network latency in milliseconds, or -1 if not available
    """
    # Extract hostname from client_id or use default
    try:
        # In a real implementation, you would have a mapping from client_id to hostname
        # For now, we'll assume client_id contains hostname information
        if client_id.startswith("client_"):
            hostname = f"client{client_id[7:]}.local"
        else:
            hostname = "localhost"
        
        # Try to ping the client
        import subprocess
        import platform
        
        if platform.system().lower() == "windows":
            cmd = ["ping", "-n", str(num_packets), "-w", str(int(timeout * 1000)), hostname]
        else:  # Linux or MacOS
            cmd = ["ping", "-c", str(num_packets), "-W", str(int(timeout)), hostname]
        
        # Run the ping command
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Parse the output
        if result.returncode == 0:
            output = result.stdout
            
            # Extract average latency
            if platform.system().lower() == "windows":
                avg_line = [line for line in output.split("\n") if "Average" in line]
                if avg_line:
                    parts = avg_line[0].split("=")
                    if len(parts) >= 2:
                        return float(parts[1].strip().replace("ms", ""))
            else:
                avg_line = [line for line in output.split("\n") if "avg" in line]
                if avg_line:
                    parts = avg_line[0].split("=")[1].split("/")
                    if len(parts) >= 2:
                        return float(parts[1])  # Average is the second value
        
        # If we can't ping, simulate latency based on client_id
        client_num = int(client_id.split('_')[-1]) if '_' in client_id else 0
        simulated_latency = 20 + (client_num % 10) * 5  # 20-65ms
        return simulated_latency
    
    except Exception as e:
        logger.debug(f"Error calculating network latency to {client_id}: {e}")
        return -1  # Indicate failure

# Example usage function for demonstration
def example_usage():
    """Example usage of the edge module functions."""
    # Create configuration
    config = {
        "server": {
            "address": "localhost",
            "port": 8080,
            "rounds": 10
        },
        "edge": {
            "enabled": True,
            "nodes": 3,
            "aggregation_strategy": "weighted_average",
            "client_assignment": "proximity",
            "base_port": 8090
        },
        "security": {
            "encryption": {
                "enabled": True,
                "type": "tls"
            }
        }
    }
    
    # Create network topology
    topology = NetworkTopology("edge_0")
    topology.generate_synthetic_topology(num_clients=20, num_edges=3)
    
    # Create edge aggregator
    edge = EdgeAggregator(config, "edge_0")
    
    # Start edge server (non-blocking)
    edge.start(blocking=False)
    
    # Wait for a bit
    time.sleep(2)
    
    # Get client status
    client_status = edge.get_client_status()
    print(f"Client status: {len(client_status)} clients")
    
    # Stop edge server
    edge.stop()
    
    print("Edge module example completed")

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run example
    example_usage()