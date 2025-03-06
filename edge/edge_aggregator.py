"""
Edge/fog node aggregator for federated learning.
"""

import logging
import flwr as fl
from typing import Dict, List, Optional, Tuple, Union
from flwr.common import Parameters, Scalar
from flwr.server.client_proxy import ClientProxy
import numpy as np
from .utils import aggregate_weights, assign_clients

logger = logging.getLogger(__name__)

class EdgeAggregator:
    """Edge/fog node aggregator for federated learning."""
    
    def __init__(self, config: Dict, edge_id: str):
        """
        Initialize the edge aggregator.
        
        Args:
            config: Configuration dictionary
            edge_id: Unique identifier for the edge node
        """
        self.config = config
        self.edge_id = edge_id
        self.edge_config = config["edge"]
        self.server_config = config["server"]
        
        # List of clients assigned to this edge node
        self.clients = []
        
        logger.info(f"Edge aggregator {edge_id} initialized")
    
    def start(self):
        """Start the edge aggregator."""
        logger.info(f"Starting edge aggregator {self.edge_id}...")
        
        # Assign clients to this edge node
        self.clients = assign_clients(
            edge_id=self.edge_id,
            assignment_strategy=self.edge_config["client_assignment"],
            config=self.config
        )
        
        # Start the edge server
        self._start_edge_server()
    
    def _start_edge_server(self):
        """Start the edge server to communicate with clients."""
        # Define strategy for the edge server
        strategy = fl.server.strategy.FedAvg(
            min_fit_clients=2,
            min_evaluate_clients=2,
            min_available_clients=2,
        )
        
        # Start server
        fl.server.start_server(
            server_address=f"0.0.0.0:{self._get_edge_port()}",
            config=fl.server.ServerConfig(num_rounds=self.server_config["rounds"]),
            strategy=strategy,
        )
    
    def _get_edge_port(self) -> int:
        """Get the port for this edge node."""
        # Base port for edge nodes
        base_port = 8090
        # Calculate port based on edge ID
        # Assuming edge_id is a string that can be converted to an integer
        edge_num = int(self.edge_id.split('_')[-1])
        return base_port + edge_num
    
    def aggregate(self, results: List[Tuple[ClientProxy, fl.common.FitRes]]) -> Parameters:
        """
        Aggregate model updates from clients.
        
        Args:
            results: List of tuples containing client proxies and fit results
            
        Returns:
            Aggregated model parameters
        """
        # Extract weights from results
        weights = [r.parameters.tensors for _, r in results]
        # Extract number of samples from each client
        num_samples = [r.num_samples for _, r in results]
        
        # Aggregate weights
        aggregated_weights = aggregate_weights(weights, num_samples)
        
        return fl.common.ndarrays_to_parameters(aggregated_weights)
    
    def forward_to_server(self, aggregated_parameters: Parameters):
        """
        Forward aggregated parameters to the central server.
        
        Args:
            aggregated_parameters: Aggregated model parameters
        """
        # Connect to the central server as a client
        client = EdgeToServerClient(self.edge_id, aggregated_parameters)
        
        # Start client
        fl.client.start_numpy_client(
            server_address=f"{self.server_config['address']}:{self.server_config['port']}",
            client=client
        )

class EdgeToServerClient(fl.client.NumPyClient):
    """Client that connects the edge node to the central server."""
    
    def __init__(self, edge_id: str, aggregated_parameters: Parameters):
        """
        Initialize the edge-to-server client.
        
        Args:
            edge_id: Unique identifier for the edge node
            aggregated_parameters: Aggregated model parameters from clients
        """
        self.edge_id = edge_id
        self.aggregated_parameters = aggregated_parameters
    
    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """Get model parameters."""
        return fl.common.parameters_to_ndarrays(self.aggregated_parameters)
    
    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        """
        No actual training happens here, just return the aggregated parameters.
        
        Args:
            parameters: Global model parameters
            config: Configuration for training
            
        Returns:
            Tuple of (parameters, num_samples, metrics)
        """
        # Return the aggregated parameters from the edge node
        return fl.common.parameters_to_ndarrays(self.aggregated_parameters), 0, {}
    
    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:
        """
        Evaluate the model.
        
        Args:
            parameters: Global model parameters
            config: Configuration for evaluation
            
        Returns:
            Tuple of (loss, num_samples, metrics)
        """
        # No actual evaluation happens here
        return 0.0, 0, {}

