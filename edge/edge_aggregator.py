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
import pickle
from phe import paillier

from .utils import aggregate_weights, assign_clients, monitor_edge_resources, calculate_network_latency, NetworkTopology
from security.encryption import Encryption
from models.model_messages_pb2 import ClientUpdateProto, EdgeAggregateProto
from models.proto_serialization import ProtoSerializer, serialize_edge_aggregate_phe

logger = logging.getLogger(__name__)

class EdgeAggregator:
    """Edge/fog node aggregator for federated learning."""
    
    def __init__(
        self, 
        config: Dict, 
        edge_id: str,
        certificates_dir: Optional[str] = None,
        custom_aggregation_fn: Optional[Callable] = None,
        server_address: str = "localhost",
        server_port: int = 8080,
        security_manager: Optional[Any] = None
    ):
        """
        Initialize the edge aggregator.
        
        Args:
            config: Configuration dictionary
            edge_id: Unique identifier for the edge node
            certificates_dir: Directory containing TLS certificates (if None, uses insecure connection)
            custom_aggregation_fn: Optional custom function for aggregating model updates
            server_address: Address of the central server
            server_port: Port of the central server
            security_manager: Security manager instance (optional)
        """
        self.config = config
        self.edge_id = edge_id
        self.edge_config = config.get("edge", {})
        self.server_config = config.get("server", {})
        self.security_config = config.get("security", {})
        self.certificates_dir = certificates_dir
        self.custom_aggregation_fn = custom_aggregation_fn
        self.server_address = server_address
        self.server_port = server_port
        self.security_manager = security_manager
        
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
        
        # Initialize encryption module (for Paillier homomorphic encryption)
        self.encryption = Encryption(config=config or {})
        
        # Paillier public key to be received from server
        self.paillier_public_key = None
        
        # Register this edge aggregator in the registry for retrieval later
        from . import edge_aggregator_registry
        edge_aggregator_registry.register(self.edge_id, self)
        
        logger.info(f"Edge aggregator {edge_id} initialized")
        
        # Check if Paillier encryption is enabled
        if (self.security_config.get("encryption", {}).get("enabled", False) and 
            self.security_config.get("encryption", {}).get("type", "") == "paillier"):
            logger.info("Paillier homomorphic encryption is enabled")
            
        # Security components
        self.encryption = Encryption(config=config or {})
        self.security_manager = security_manager
        
        # Initialize serializer
        if security_manager:
            try:
                # Use both Client-Edge (K_CE) and Edge-Server (K_ES) keys with serializer
                self.serializer = ProtoSerializer(
                    hmac_key=os.urandom(32),
                    client_edge_key=security_manager.get_client_edge_key(),
                    edge_server_key=security_manager.get_edge_server_key()
                )
                logger.info("Initialized Protocol Buffer serializer with communication-specific keys")
            except Exception as e:
                logger.warning(f"Error initializing ProtoSerializer with specific keys: {e}")
                self.serializer = ProtoSerializer(hmac_key=os.urandom(32))
        else:
            # When no security manager is available, use random keys
            self.serializer = ProtoSerializer(hmac_key=os.urandom(32))
        
        # Keys for Edge-Server communication
        self._edge_server_key = os.urandom(32)  # Default random key
        if security_manager:
            try:
                self._edge_server_key = security_manager.get_edge_server_key()
                logger.info("Using Edge-Server key (K_ES) from security manager")
            except Exception as e:
                logger.warning(f"Unable to get Edge-Server key: {e}")
        
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
                
        # Unregister from the registry
        from . import edge_aggregator_registry
        edge_aggregator_registry.unregister(self.edge_id)
    
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
        Create a custom Flower strategy for the edge server.
        
        Returns:
            Flower server strategy
        """
        strategy_config = {
            "fraction_fit": 1.0,  # All available clients participate
            "min_fit_clients": 2,  # At least 2 clients are needed
            "min_available_clients": 2,  # Wait for at least 2 clients
            "min_eval_clients": 2,  # At least 2 clients for evaluation
        }
        
        class EdgeStrategy(fl.server.strategy.FedAvg):
            """Custom aggregation strategy for edge nodes."""
            
            def __init__(self, edge_agg, **kwargs):
                super().__init__(**kwargs)
                self.edge_aggregator = edge_agg
                self.edge_id = edge_agg.edge_id
                self.serializer = ProtoSerializer(hmac_key=os.urandom(32))
            
            def aggregate_fit(
                self,
                server_round: int,
                results: List[Tuple[ClientProxy, FitRes]],
                failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
            ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
                """Aggregate model updates from clients."""
                if len(results) == 0:
                    return None, {}
                
                # Log client results
                self.edge_aggregator._log_client_results(server_round, results, failures)
                
                # Update round number
                self.edge_aggregator.current_round = server_round
                
                # Check if Paillier encryption is enabled
                use_paillier = (
                    self.edge_aggregator.security_config and 
                    self.edge_aggregator.security_config.get("encryption", {}).get("enabled", False) and
                    self.edge_aggregator.security_config.get("encryption", {}).get("type", "") == "paillier"
                )
                
                if use_paillier and self.edge_aggregator.paillier_public_key:
                    # Use homomorphic addition for encrypted client updates
                    return self.aggregate_encrypted_updates(server_round, results, failures)
                
                # Default aggregation for unencrypted updates
                weights_results = [
                    (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples) 
                    for _, fit_res in results
                ]
                
                weights = [weights for weights, _ in weights_results]
                examples = [num_examples for _, num_examples in weights_results]
                
                # Use custom aggregation function if provided
                if self.edge_aggregator.custom_aggregation_fn:
                    aggregated_weights = self.edge_aggregator.custom_aggregation_fn(weights, examples)
                else:
                    # Default weighted average
                    aggregated_weights = aggregate_weights(weights, examples)
                
                # Convert back to Parameters
                aggregated_parameters = ndarrays_to_parameters(aggregated_weights)
                
                # Save the aggregated model
                self.edge_aggregator._save_aggregated_model(aggregated_parameters, server_round)
                
                # Get metrics from results
                metrics = {}
                for _, fit_res in results:
                    for key, value in fit_res.metrics.items():
                        if key not in metrics:
                            metrics[key] = []
                        metrics[key].append(value)
                
                # Average metrics
                for key in metrics:
                    metrics[key] = float(sum(metrics[key]) / len(metrics[key]))
                
                # Add edge node metadata
                metrics["edge_id"] = self.edge_id
                metrics["edge_clients"] = len(results)
                metrics["edge_failures"] = len(failures)
                
                # Decide whether to forward to server
                if self.edge_aggregator._should_forward_to_server(server_round):
                    try:
                        self.edge_aggregator.forward_to_server(
                            aggregated_parameters, server_round, metrics
                        )
                    except Exception as e:
                        logging.error(f"Failed to forward to server: {e}")
                
                # Record aggregation stats
                self.edge_aggregator.aggregation_stats.append({
                    "round": server_round,
                    "clients": len(results),
                    "failures": len(failures),
                    "timestamp": time.time(),
                    "metrics": metrics
                })
                
                # Save stats periodically
                if server_round % 5 == 0 or server_round == self.edge_aggregator.max_rounds:
                    self.edge_aggregator._save_aggregation_stats()
                
                return aggregated_parameters, metrics
            
            def aggregate_encrypted_updates(
                self,
                server_round: int,
                results: List[Tuple[ClientProxy, FitRes]],
                failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
            ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
                """Aggregate homomorphically encrypted client updates using Paillier."""
                logger.info(f"Performing homomorphic aggregation for {len(results)} encrypted client updates.")
                
                # Start tracking time for homomorphic operations
                start_time = time.time()
                
                all_client_cipher_lists = []
                total_examples = 0
                client_ids = []
                metadata_list = []
                
                for client_proxy, fit_res in results:
                    try:
                        # 1. Deserialize FitRes bytes to get the actual ClientUpdateProto bytes
                        client_proto_bytes = fit_res.parameters.tensors[0]  # Adjust based on how you pack it in FitRes
                        
                        # 2. Verify HMAC (requires the Client-Edge key K_CE)
                        client_ciphertexts, metadata, hmac_verified = self.serializer.deserialize_client_update_phe(
                            client_proto_bytes
                        )
                        
                        # Skip if HMAC verification fails
                        if not hmac_verified:
                            logger.warning(f"HMAC verification failed for client {metadata.get('client_id', 'unknown')}")
                            continue
                        
                        # 3. Reconstruct Paillier objects
                        client_cipher_objs = []
                        try:
                            for cts in client_ciphertexts:
                                enc_number = paillier.EncryptedNumber(
                                    self.edge_aggregator.paillier_public_key, 
                                    int(cts)
                                )
                                client_cipher_objs.append(enc_number)
                            
                            all_client_cipher_lists.append(client_cipher_objs)
                            client_ids.append(metadata.get("client_id", "unknown"))
                            total_examples += metadata.get("training_samples", fit_res.num_examples)
                            metadata_list.append(metadata)
                            
                            logger.info(f"Successfully processed encrypted update from client {metadata.get('client_id', 'unknown')}")
                        except Exception as e:
                            logger.error(f"Error reconstructing Paillier objects: {e}")
                            continue
                        
                    except Exception as e:
                        logger.error(f"Error processing encrypted client update: {e}")
                        continue
                
                # Check if any valid results were received
                if not all_client_cipher_lists:
                    logger.warning("No valid client results received for homomorphic aggregation.")
                    return None, {"error": "no_valid_encrypted_updates"}
                
                try:
                    # 4. Perform Homomorphic Addition
                    num_weights = len(all_client_cipher_lists[0])  # Assumes all clients send same size update
                    aggregated_ciphertexts = []
                    
                    for i in range(num_weights):
                        # Start sum for this weight index with the first client's value
                        current_sum = all_client_cipher_lists[0][i]
                        
                        # Add subsequent clients (starting from second client)
                        for client_idx in range(1, len(all_client_cipher_lists)):
                            try:
                                current_sum += all_client_cipher_lists[client_idx][i]
                            except Exception as e:
                                logger.error(f"Homomorphic addition failed for weight {i}, client {client_idx}: {e}")
                        
                        # Store the aggregated ciphertext
                        aggregated_ciphertexts.append(current_sum)
                    
                    logger.info(f"Homomorphically aggregated {num_weights} encrypted values from {len(all_client_cipher_lists)} clients.")
                    
                    # 5. Convert aggregated ciphertexts to string representation for serialization
                    aggregated_ciphertext_strs = [str(ct.ciphertext(be_secure=False)) for ct in aggregated_ciphertexts]
                    
                    # 6. Prepare metadata for the serialized EdgeAggregateProto
                    edge_metrics = {
                        "aggregation_time": time.time() - start_time,
                        "client_count": len(all_client_cipher_lists),
                        "total_examples": total_examples,
                        "edge_id": self.edge_id,
                        "round": server_round,
                        "aggregation_method": "paillier_homomorphic"
                    }
                    
                    # 7. Serialize the aggregated result using Edge-Server key (K_ES)
                    serialized_edge_aggregate = self.serializer.serialize_edge_aggregate_phe(
                        edge_id=self.edge_id,
                        round_id=server_round,
                        client_count=len(all_client_cipher_lists),
                        aggregated_paillier_ciphertexts=aggregated_ciphertext_strs,
                        metrics=edge_metrics,
                        total_examples=total_examples
                    )
                    
                    # 8. Pack into Parameters for Flower's expected format
                    aggregated_parameters = Parameters(
                        tensors=[serialized_edge_aggregate],
                        tensor_type="bytes"
                    )
                    
                    # Save the aggregated model information (though we can't save the actual model since it's encrypted)
                    self.edge_aggregator._save_encrypted_stats(edge_metrics, server_round)
                    
                    # 9. Decide whether to forward to server
                    if self.edge_aggregator._should_forward_to_server(server_round):
                        try:
                            self.edge_aggregator.forward_to_server(
                                aggregated_parameters, server_round, edge_metrics
                            )
                        except Exception as e:
                            logging.error(f"Failed to forward encrypted aggregate to server: {e}")
                    
                    # 10. Update aggregation stats
                    self.edge_aggregator.aggregation_stats.append({
                        "round": server_round,
                        "clients": len(all_client_cipher_lists),
                        "failures": len(failures),
                        "timestamp": time.time(),
                        "metrics": edge_metrics,
                        "encryption": "paillier"
                    })
                    
                    # Save stats periodically
                    if server_round % 5 == 0 or server_round == self.edge_aggregator.max_rounds:
                        self.edge_aggregator._save_aggregation_stats()
                    
                    return aggregated_parameters, edge_metrics
                    
                except Exception as e:
                    logger.error(f"Error during homomorphic aggregation: {e}")
                    return None, {"error": str(e)}
            
            def aggregate_evaluate(
                self,
                server_round: int,
                results: List[Tuple[ClientProxy, EvaluateRes]],
                failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
            ) -> Tuple[Optional[float], Dict[str, Scalar]]:
                """Aggregate evaluation results from clients."""
                # Log evaluation results
                self.edge_aggregator._log_evaluation_results(server_round, results, failures)
                
                # Call parent method
                loss, metrics = super().aggregate_evaluate(server_round, results, failures)
                
                # Add edge-specific metrics
                if metrics is None:
                    metrics = {}
                
                metrics["edge_id"] = self.edge_id
                metrics["num_eval_clients"] = len(results)
                metrics["num_eval_failures"] = len(failures)
                
                return loss, metrics
        
        # Create and return the strategy
        return EdgeStrategy(edge_agg=self, **strategy_config)
    
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
        server_address = f"{self.server_address}:{self.server_port}"
        
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

    def _save_encrypted_stats(self, metrics: Dict[str, Scalar], round_num: int) -> None:
        """
        Save statistics about encrypted aggregation.
        
        Args:
            metrics: Metrics from the aggregation
            round_num: The current round number
        """
        try:
            stats_dir = os.path.join(self.models_dir, "encrypted_stats")
            os.makedirs(stats_dir, exist_ok=True)
            
            stats_file = os.path.join(stats_dir, f"round_{round_num}_stats.json")
            
            with open(stats_file, 'w') as f:
                json.dump({
                    "edge_id": self.edge_id,
                    "round": round_num,
                    "timestamp": time.time(),
                    "metrics": metrics
                }, f, indent=2)
                
            logger.debug(f"Saved encrypted aggregation stats for round {round_num}")
        except Exception as e:
            logger.error(f"Failed to save encrypted stats: {e}")

    def set_paillier_public_key(self, public_key: paillier.PaillierPublicKey) -> None:
        """
        Set the Paillier public key received from the server.
        
        Args:
            public_key: The Paillier public key
        """
        self.paillier_public_key = public_key
        logger.info(f"Edge {self.edge_id} received Paillier public key from server")

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
        
        # Check if Paillier public key is included in config and pass it to the EdgeAggregator
        if "paillier_n" in config:
            from edge import edge_aggregator_registry
            edge_agg = edge_aggregator_registry.get(self.edge_id)
            if edge_agg:
                n_str = config["paillier_n"]
                edge_agg.paillier_public_key = paillier.PaillierPublicKey(n=int(n_str))
                precision_bits = config.get("paillier_precision_bits", 64)
                logger.info(f"Edge {self.edge_id} received and configured Paillier public key (n={n_str[:20]}...) and precision={precision_bits}")
                
                # Update the encryption module with the key
                if hasattr(edge_agg, 'encryption') and edge_agg.encryption:
                    edge_agg.encryption._paillier_public_key = edge_agg.paillier_public_key
                    if hasattr(edge_agg.encryption, '_encoder') and edge_agg.encryption._encoder:
                        edge_agg.encryption._encoder.precision_bits = precision_bits
        
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