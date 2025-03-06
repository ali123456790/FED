"""
Edge/fog node aggregator for federated learning.

This module implements the edge node functionality for aggregating updates from
nearby clients before forwarding them to the central server, reducing communication
overhead and improving efficiency.
"""

import logging
import os
import json
import time
import threading
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import flwr as fl
from flwr.common import Parameters, Scalar, FitRes, EvaluateRes, parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server.client_proxy import ClientProxy
import ssl

from .utils import aggregate_weights, assign_clients, monitor_edge_resources, calculate_network_latency, NetworkTopology

logger = logging.getLogger(__name__)

class EdgeAggregator:
    """Edge/fog node aggregator for federated learning."""
    
    def __init__(
        self, 
        config: Dict, 
        edge_id: str,
        certificates_dir: Optional[str] = None,
        custom_aggregation_fn: Optional[Callable] = None
    ):
        """
        Initialize the edge aggregator.
        
        Args:
            config: Configuration dictionary
            edge_id: Unique identifier for the edge node
            certificates_dir: Directory containing TLS certificates (if None, uses insecure connection)
            custom_aggregation_fn: Optional custom function for aggregating model updates
        """
        self.config = config
        self.edge_id = edge_id
        self.edge_config = config.get("edge", {})
        self.server_config = config.get("server", {})
        self.security_config = config.get("security", {})
        self.certificates_dir = certificates_dir
        self.custom_aggregation_fn = custom_aggregation_fn
        
        # List of clients assigned to this edge node
        self.clients = []
        
        # Track online status of each client
        self.client_status = {}
        
        # Edge server instance
        self.edge_server = None
        
        # Network topology for client assignment
        self.network_topology = NetworkTopology(self.edge_id)
        
        # Cached global model parameters
        self.latest_global_parameters = None
        
        # Round tracking
        self.current_round = 0
        self.max_rounds = self.server_config.get("rounds", 10)
        
        # Stats tracking
        self.aggregation_stats = []
        
        # Thread control
        self.stop_edge = threading.Event()
        self.edge_thread = None
        
        # Create directory for storing aggregated models
        self.models_dir = os.path.join("models", "edge", self.edge_id)
        os.makedirs(self.models_dir, exist_ok=True)
        
        logger.info(f"Edge aggregator {edge_id} initialized")
    
    def start(self, blocking: bool = True) -> None:
        """
        Start the edge aggregator.
        
        Args:
            blocking: Whether to run in blocking mode or in a separate thread
        """
        logger.info(f"Starting edge aggregator {self.edge_id}...")
        
        # Assign clients to this edge node
        self._assign_clients()
        
        if blocking:
            # Start the edge server directly
            self._start_edge_server()
        else:
            # Start in a separate thread
            self.stop_edge.clear()
            self.edge_thread = threading.Thread(
                target=self._start_edge_server,
                daemon=True
            )
            self.edge_thread.start()
            logger.info(f"Edge aggregator {self.edge_id} started in background thread")
    
    def stop(self, timeout: float = 5.0) -> None:
        """
        Stop the edge aggregator.
        
        Args:
            timeout: Timeout in seconds for stopping the thread
        """
        if self.edge_thread and self.edge_thread.is_alive():
            logger.info(f"Stopping edge aggregator {self.edge_id}...")
            self.stop_edge.set()
            self.edge_thread.join(timeout=timeout)
            if self.edge_thread.is_alive():
                logger.warning(f"Edge aggregator {self.edge_id} thread did not stop within {timeout} seconds")
            else:
                logger.info(f"Edge aggregator {self.edge_id} stopped successfully")
    
    def _assign_clients(self) -> None:
        """Assign clients to this edge node."""
        assignment_strategy = self.edge_config.get("client_assignment", "proximity")
        
        try:
            self.clients = assign_clients(
                edge_id=self.edge_id,
                assignment_strategy=assignment_strategy,
                config=self.config
            )
            
            # Initialize client status
            for client_id in self.clients:
                self.client_status[client_id] = {
                    "online": False,
                    "last_seen": None,
                    "resource_info": None,
                    "latency": None
                }
            
            logger.info(f"Assigned {len(self.clients)} clients to edge node {self.edge_id}: {self.clients}")
        except Exception as e:
            logger.error(f"Error assigning clients to edge node {self.edge_id}: {e}")
            # Fall back to a default assignment
            self.clients = [f"client_{i}" for i in range(int(self.edge_id.split('_')[-1]) * 5, 
                                                         int(self.edge_id.split('_')[-1]) * 5 + 5)]
            logger.info(f"Using fallback client assignment: {self.clients}")
    
    def _start_edge_server(self) -> None:
        """Start the edge server to communicate with clients."""
        try:
            # Define custom strategy for the edge server
            strategy = self._create_edge_strategy()
            
            # Get port for this edge node
            edge_port = self._get_edge_port()
            
            # Check if TLS should be used
            secure_mode = self._should_use_tls()
            
            # Prepare server config
            server_config = fl.server.ServerConfig(num_rounds=self.max_rounds)
            
            # Log startup information
            logger.info(f"Edge server {self.edge_id} starting on port {edge_port} with "
                        f"{'secure' if secure_mode else 'insecure'} connection")
            
            # Set server address
            server_address = f"0.0.0.0:{edge_port}"
            
            # Start server with or without TLS
            if secure_mode and self.certificates_dir:
                # Load certificates
                server_cert = os.path.join(self.certificates_dir, f"{self.edge_id}.crt")
                server_key = os.path.join(self.certificates_dir, f"{self.edge_id}.key")
                root_cert = os.path.join(self.certificates_dir, "ca.crt")
                
                if not all(os.path.exists(cert) for cert in [server_cert, server_key, root_cert]):
                    logger.warning("TLS certificates not found, falling back to insecure connection")
                    self.edge_server = fl.server.start_server(
                        server_address=server_address,
                        config=server_config,
                        strategy=strategy,
                    )
                else:
                    # Create SSL context
                    ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
                    ssl_context.load_cert_chain(certfile=server_cert, keyfile=server_key)
                    ssl_context.load_verify_locations(cafile=root_cert)
                    ssl_context.verify_mode = ssl.CERT_REQUIRED
                    
                    self.edge_server = fl.server.start_server(
                        server_address=server_address,
                        config=server_config,
                        strategy=strategy,
                        certificates=(root_cert, server_cert, server_key)
                    )
            else:
                self.edge_server = fl.server.start_server(
                    server_address=server_address,
                    config=server_config,
                    strategy=strategy,
                )
            
            logger.info(f"Edge server {self.edge_id} stopped")
        except Exception as e:
            logger.error(f"Error starting edge server {self.edge_id}: {e}")
            raise
    
    def _create_edge_strategy(self) -> fl.server.strategy.Strategy:
        """
        Create a custom strategy for the edge server.
        
        Returns:
            Flower strategy for the edge server
        """
        # Extract configuration parameters
        min_clients = max(2, self.edge_config.get("min_clients", 2))
        min_available_clients = max(2, self.edge_config.get("min_available_clients", 2))
        fraction_fit = self.edge_config.get("fraction_fit", 1.0)
        fraction_evaluate = self.edge_config.get("fraction_evaluate", 1.0)
        
        # Adjust min clients if we have fewer assigned clients
        if len(self.clients) < min_clients:
            min_clients = max(1, len(self.clients))
            min_available_clients = max(1, min_clients)
        
        # Get aggregation strategy
        agg_strategy = self.edge_config.get("aggregation_strategy", "weighted_average")
        
        # Create a custom strategy with our aggregation logic
        class EdgeStrategy(fl.server.strategy.FedAvg):
            def __init__(self, edge_agg, **kwargs):
                super().__init__(**kwargs)
                self.edge_agg = edge_agg
            
            def aggregate_fit(
                self,
                server_round: int,
                results: List[Tuple[ClientProxy, FitRes]],
                failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
            ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
                # Log client results
                self.edge_agg._log_client_results(server_round, results, failures)
                
                # Call parent method to get default aggregation
                parameters, metrics = super().aggregate_fit(server_round, results, failures)
                
                # Save aggregated model for this round
                if parameters is not None:
                    self.edge_agg._save_aggregated_model(parameters, server_round)
                
                # Add edge-specific metrics
                if metrics is None:
                    metrics = {}
                
                metrics["edge_id"] = self.edge_agg.edge_id
                metrics["num_clients"] = len(results)
                metrics["num_failures"] = len(failures)
                metrics["edge_resources"] = monitor_edge_resources()
                
                # Forward to central server if needed
                if self.edge_agg._should_forward_to_server(server_round):
                    try:
                        self.edge_agg.forward_to_server(parameters, server_round, metrics)
                    except Exception as e:
                        logger.error(f"Error forwarding to server: {e}")
                
                return parameters, metrics
            
            def aggregate_evaluate(
                self,
                server_round: int,
                results: List[Tuple[ClientProxy, EvaluateRes]],
                failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
            ) -> Tuple[Optional[float], Dict[str, Scalar]]:
                # Log evaluation results
                self.edge_agg._log_evaluation_results(server_round, results, failures)
                
                # Call parent method
                loss, metrics = super().aggregate_evaluate(server_round, results, failures)
                
                # Add edge-specific metrics
                if metrics is None:
                    metrics = {}
                
                metrics["edge_id"] = self.edge_agg.edge_id
                metrics["num_eval_clients"] = len(results)
                metrics["num_eval_failures"] = len(failures)
                
                return loss, metrics
        
        # Create the strategy instance
        return EdgeStrategy(
            edge_agg=self,
            min_fit_clients=min_clients,
            min_evaluate_clients=min_clients,
            min_available_clients=min_available_clients,
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            evaluate_metrics_aggregation_fn=self._aggregate_evaluation_metrics
        )
    
    def _get_edge_port(self) -> int:
        """
        Get the port for this edge node.
        
        Returns:
            Port number for this edge node
        """
        try:
            # Base port for edge nodes
            base_port = self.edge_config.get("base_port", 8090)
            
            # Calculate port based on edge ID
            # Assuming edge_id is a string like "edge_0", "edge_1", etc.
            edge_num = int(self.edge_id.split('_')[-1])
            return base_port + edge_num
        except Exception as e:
            logger.error(f"Error getting port for edge node {self.edge_id}: {e}")
            # Fall back to a random port in the range 8090-8099
            import random
            return random.randint(8090, 8099)
    
    def _should_use_tls(self) -> bool:
        """
        Determine if TLS should be used for connections.
        
        Returns:
            True if TLS should be used, False otherwise
        """
        encryption_enabled = self.security_config.get("encryption", {}).get("enabled", False)
        encryption_type = self.security_config.get("encryption", {}).get("type", "tls")
        return encryption_enabled and encryption_type == "tls" and self.certificates_dir is not None
    
    def _log_client_results(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]
    ) -> None:
        """
        Log client training results.
        
        Args:
            server_round: Current round number
            results: List of successful client results
            failures: List of client failures
        """
        self.current_round = server_round
        
        # Update client status for successful clients
        for client_proxy, fit_res in results:
            client_id = client_proxy.cid
            if client_id in self.client_status:
                self.client_status[client_id]["online"] = True
                self.client_status[client_id]["last_seen"] = time.time()
                
                # Extract client metrics if available
                if hasattr(fit_res, "metrics") and fit_res.metrics:
                    # Update resource info if available
                    resource_keys = ["cpu_percent", "memory_percent", "battery_percent"]
                    resource_info = {k: fit_res.metrics.get(k, -1) for k in resource_keys if k in fit_res.metrics}
                    if resource_info:
                        self.client_status[client_id]["resource_info"] = resource_info
        
        # Update client status for failed clients
        for failure in failures:
            if isinstance(failure, tuple) and len(failure) >= 1:
                client_proxy = failure[0]
                client_id = client_proxy.cid
                if client_id in self.client_status:
                    # Mark as offline but don't update last_seen
                    self.client_status[client_id]["online"] = False
        
        # Log summary
        logger.info(f"Round {server_round}: {len(results)} successful clients, {len(failures)} failures")
        
        # Add to aggregation stats
        stats = {
            "round": server_round,
            "timestamp": time.time(),
            "num_clients": len(results),
            "num_failures": len(failures),
            "client_ids": [client_proxy.cid for client_proxy, _ in results],
            "edge_resources": monitor_edge_resources()
        }
        self.aggregation_stats.append(stats)
        
        # Save stats to disk periodically
        if server_round % 5 == 0 or server_round == self.max_rounds:
            self._save_aggregation_stats()
    
    def _log_evaluation_results(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]]
    ) -> None:
        """
        Log client evaluation results.
        
        Args:
            server_round: Current round number
            results: List of successful client evaluation results
            failures: List of client evaluation failures
        """
        # Track evaluation metrics
        metrics = []
        for client_proxy, eval_res in results:
            client_id = client_proxy.cid
            if eval_res and hasattr(eval_res, "metrics") and eval_res.metrics:
                metrics.append((client_id, eval_res.metrics))
        
        # Log summary of evaluation
        avg_loss = np.mean([res.loss for _, res in results]) if results else float('nan')
        logger.info(f"Round {server_round} evaluation: {len(results)} clients, avg_loss={avg_loss:.4f}")
        
        # Add to aggregation stats if we have an entry for this round
        for stats in self.aggregation_stats:
            if stats["round"] == server_round:
                stats["evaluation"] = {
                    "avg_loss": avg_loss,
                    "num_eval_clients": len(results),
                    "num_eval_failures": len(failures),
                    "client_metrics": metrics
                }
                break
    
    def _save_aggregation_stats(self) -> None:
        """Save aggregation statistics to disk."""
        stats_file = os.path.join(self.models_dir, f"aggregation_stats.json")
        try:
            with open(stats_file, 'w') as f:
                json.dump(self.aggregation_stats, f, indent=2)
            logger.debug(f"Saved aggregation stats to {stats_file}")
        except Exception as e:
            logger.error(f"Error saving aggregation stats: {e}")
    
    def _save_aggregated_model(self, parameters: Parameters, round_num: int) -> None:
        """
        Save the aggregated model for the current round.
        
        Args:
            parameters: Aggregated model parameters
            round_num: Current round number
        """
        # Convert parameters to NumPy arrays
        weights = parameters_to_ndarrays(parameters)
        
        # Save weights to disk
        model_file = os.path.join(self.models_dir, f"model_round_{round_num}.npz")
        try:
            np.savez_compressed(model_file, *weights)
            logger.debug(f"Saved aggregated model for round {round_num} to {model_file}")
        except Exception as e:
            logger.error(f"Error saving aggregated model: {e}")
        
        # Update latest global parameters
        self.latest_global_parameters = parameters
    
    def _should_forward_to_server(self, round_num: int) -> bool:
        """
        Determine if the aggregated model should be forwarded to the central server.
        
        Args:
            round_num: Current round number
            
        Returns:
            True if the model should be forwarded, False otherwise
        """
        # Check if edge mode is enabled
        if not self.edge_config.get("enabled", True):
            return False
        
        # Always forward on the final round
        if round_num >= self.max_rounds:
            return True
        
        # Forward every N rounds based on configuration
        forward_frequency = self.edge_config.get("forward_frequency", 1)
        return round_num % forward_frequency == 0
    
    def _aggregate_evaluation_metrics(
        self, 
        metrics: List[Tuple[int, Dict[str, Scalar]]]
    ) -> Dict[str, Scalar]:
        """
        Aggregate evaluation metrics from multiple clients.
        
        Args:
            metrics: List of tuples containing number of examples and metrics dictionaries
            
        Returns:
            Aggregated metrics dictionary
        """
        if not metrics:
            return {}
        
        # Extract all metric names
        metric_names = set()
        for _, client_metrics in metrics:
            metric_names.update(client_metrics.keys())
        
        # Initialize aggregated metrics
        aggregated = {}
        
        # Aggregate each metric separately
        for metric_name in metric_names:
            # Skip non-numeric metrics
            valid_values = [
                (num_examples, float(client_metrics[metric_name]))
                for num_examples, client_metrics in metrics
                if metric_name in client_metrics and isinstance(client_metrics[metric_name], (int, float))
            ]
            
            if valid_values:
                # Calculate weighted average
                total_examples = sum(num_examples for num_examples, _ in valid_values)
                weighted_average = sum(num_examples * value for num_examples, value in valid_values) / total_examples
                aggregated[metric_name] = float(weighted_average)
                
                # Also calculate min, max
                min_value = min(value for _, value in valid_values)
                max_value = max(value for _, value in valid_values)
                aggregated[f"{metric_name}_min"] = float(min_value)
                aggregated[f"{metric_name}_max"] = float(max_value)
        
        # Add number of clients
        aggregated["num_clients"] = len(metrics)
        
        return aggregated
    
    def forward_to_server(
        self, 
        aggregated_parameters: Parameters, 
        round_num: int,
        metrics: Dict[str, Scalar]
    ) -> None:
        """
        Forward aggregated parameters to the central server.
        
        Args:
            aggregated_parameters: Aggregated model parameters
            round_num: Current round number
            metrics: Metrics to send to the server
        """
        if aggregated_parameters is None:
            logger.warning(f"Round {round_num}: No aggregated parameters to forward to server")
            return
        
        logger.info(f"Round {round_num}: Forwarding aggregated parameters to central server")
        
        # Get server address
        server_address = f"{self.server_config.get('address', 'localhost')}:{self.server_config.get('port', 8080)}"
        
        # Create client to server connection
        try:
            # Determine if TLS should be used
            use_tls = self._should_use_tls()
            root_certificates = None
            
            if use_tls and self.certificates_dir:
                # Load root certificates
                root_cert_path = os.path.join(self.certificates_dir, "ca.crt")
                if os.path.exists(root_cert_path):
                    with open(root_cert_path, 'rb') as f:
                        root_certificates = f.read()
            
            # Create client with edge node information
            client = EdgeToServerClient(
                edge_id=self.edge_id,
                aggregated_parameters=aggregated_parameters,
                num_clients=len(self.clients),
                round_num=round_num,
                metrics=metrics
            )
            
            # Start the client
            fl.client.start_client(
                server_address=server_address,
                client=client,
                root_certificates=root_certificates
            )
            
            logger.info(f"Successfully forwarded round {round_num} parameters to central server")
        except Exception as e:
            logger.error(f"Error forwarding parameters to central server: {e}")
            raise
    
    def get_client_status(self) -> Dict[str, Dict]:
        """
        Get status information for all clients.
        
        Returns:
            Dictionary mapping client IDs to status information
        """
        return self.client_status
    
    def get_aggregation_stats(self) -> List[Dict]:
        """
        Get aggregation statistics.
        
        Returns:
            List of dictionaries containing aggregation statistics
        """
        return self.aggregation_stats

class EdgeToServerClient(fl.client.Client):
    """Client that connects the edge node to the central server."""
    
    def __init__(
        self, 
        edge_id: str, 
        aggregated_parameters: Parameters,
        num_clients: int = 0,
        round_num: int = 0,
        metrics: Optional[Dict[str, Scalar]] = None
    ):
        """
        Initialize the edge-to-server client.
        
        Args:
            edge_id: Unique identifier for the edge node
            aggregated_parameters: Aggregated model parameters from clients
            num_clients: Number of clients that contributed to the aggregation
            round_num: Current round number
            metrics: Additional metrics to send to the server
        """
        self.edge_id = edge_id
        self.aggregated_parameters = aggregated_parameters
        self.num_clients = num_clients
        self.round_num = round_num
        self.metrics = metrics if metrics is not None else {}
        
        logger.debug(f"EdgeToServerClient initialized for edge {edge_id}, round {round_num}")
    
    def get_parameters(self, config: Dict[str, Scalar]) -> Parameters:
        """
        Get model parameters to send to the server.
        
        Args:
            config: Configuration from the server
            
        Returns:
            Aggregated model parameters
        """
        logger.debug(f"Server requested parameters from edge {self.edge_id}")
        return self.aggregated_parameters
    
    def fit(
        self, 
        parameters: Parameters, 
        config: Dict[str, Scalar]
    ) -> Tuple[Parameters, int, Dict[str, Scalar]]:
        """
        No actual training happens here, just return the aggregated parameters.
        
        Args:
            parameters: Global model parameters from the server
            config: Configuration for training
            
        Returns:
            Tuple of (aggregated_parameters, num_samples, metrics)
        """
        logger.info(f"Edge {self.edge_id} received fit request from server for round {config.get('server_round', 0)}")
        
        # Store the global parameters from the server if needed
        # This could be used to update the edge node's model in the future
        
        # Prepare metrics
        metrics = {
            "edge_id": self.edge_id,
            "num_clients": self.num_clients,
            "round": self.round_num,
            "edge_timestamp": time.time()
        }
        
        # Add custom metrics
        metrics.update(self.metrics)
        
        # Return the aggregated parameters
        return self.aggregated_parameters, self.num_clients, metrics
    
    def evaluate(
        self, 
        parameters: Parameters, 
        config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        """
        Evaluate the model on behalf of the edge node.
        
        Args:
            parameters: Global model parameters
            config: Configuration for evaluation
            
        Returns:
            Tuple of (loss, num_samples, metrics)
        """
        logger.info(f"Edge {self.edge_id} received evaluate request from server")
        
        # No actual evaluation happens here
        # Just return aggregated metrics if available
        loss = 0.0
        if "loss" in self.metrics:
            loss = float(self.metrics["loss"])
        
        metrics = {
            "edge_id": self.edge_id,
            "num_clients": self.num_clients,
            "evaluation_timestamp": time.time()
        }
        
        # Add evaluation metrics if available
        if "evaluation" in self.metrics:
            metrics.update(self.metrics["evaluation"])
        
        return loss, self.num_clients, metrics