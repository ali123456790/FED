"""
Flower server implementation for federated learning in IoT Device Security.

This module implements a federated learning server for the FIDS system,
handling coordination of federated learning rounds, model aggregation,
and communication with edge nodes and clients.
"""

import os
import time
import json
import logging
import threading
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import flwr as fl
from flwr.common import Metrics, Parameters, Scalar, FitRes, EvaluateRes
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy
from flwr.server.history import History
from flwr.server.strategy import Strategy
from phe import paillier

from .aggregation import get_aggregation_strategy
from .utils import save_model, load_model, monitor_resources, get_model_size
from models.model_factory import create_model
from security.masking import MaskingMechanism
from models.proto_serialization import ProtoSerializer

logger = logging.getLogger(__name__)

class FlowerServer:
    """
    Flower server implementation for federated learning with enhanced
    features for IoT device security applications.
    """
    
    def __init__(
        self, 
        config: Dict,
        model_path: Optional[str] = None,
        certificates_dir: Optional[str] = None,
        metrics_dir: Optional[str] = None
    ):
        """
        Initialize the federated learning server.
        
        Args:
            config: Configuration dictionary with server settings
            model_path: Path to initial global model (if None, a new model is created)
            certificates_dir: Directory containing TLS certificates
            metrics_dir: Directory to save metrics and logs
        """
        self.config = config
        self.server_config = config["server"]
        self.model_config = config["model"]
        self.security_config = config["security"]
        self.data_config = config["data"]
        self.evaluation_config = config["evaluation"]
        
        # Paths
        self.model_path = model_path
        self.certificates_dir = certificates_dir
        self.metrics_dir = metrics_dir or "./metrics"
        
        # Initialize paths
        os.makedirs(self.metrics_dir, exist_ok=True)
        
        # Initialize model
        self.model = self._initialize_model()
        
        # Initialize masking mechanism for unmasking client updates
        self.masking = MaskingMechanism(config)
        
        # Initialize proto serializer for handling serialized updates
        self.serializer = ProtoSerializer(
            hmac_key=os.urandom(32)  # In production should be derived securely
        )
        
        # Client identity verification
        self.verified_clients = {}
        self.client_hmac_keys = {}
        self.client_device_info = {}
        
        # Initialize Paillier homomorphic encryption if enabled
        if self.security_config.get('encryption', {}).get('enabled', False) and self.security_config.get('encryption', {}).get('type', '') == 'paillier':
            key_size = self.security_config.get('encryption', {}).get('paillier', {}).get('key_size', 2048)
            logger.info(f"Generating Paillier keypair with size {key_size}...")
            self.paillier_public_key, self.paillier_secret_key = paillier.generate_paillier_keypair(n_length=key_size)
            logger.info("Paillier keypair generated.")
        else:
            self.paillier_public_key = None
            self.paillier_secret_key = None
        
        # Create aggregation strategy
        self.aggregation_strategy = get_aggregation_strategy(
            strategy_name=self.server_config["aggregation_strategy"],
            security_config=self.security_config,
            config=config,
            server_instance=self  # Pass the server instance for strategies that need it (like Paillier)
        )
        
        # Store model shape for Paillier decryption and reshaping
        self.model_shape = self._get_model_shape()
        
        # Server instance
        self.server = None
        self.history = None
        
        # Metrics tracking
        self.round_metrics = []
        self.resource_usage = []
        
        # Monitoring
        self.enable_monitoring = True
        self.monitoring_thread = None
        self.stop_monitoring = threading.Event()
        
        logger.info(f"FIDS Server initialized with {self.server_config['aggregation_strategy']} aggregation strategy")
        logger.info(f"Using model: {self.model_config['type']}/{self.model_config['name']}")
    
    def _initialize_model(self) -> Any:
        """
        Initialize the global model.
        
        Returns:
            Global model instance
        """
        if self.model_path and os.path.exists(self.model_path):
            logger.info(f"Loading global model from {self.model_path}")
            return load_model(self.model_path)
        else:
            logger.info(f"Creating new global model: {self.model_config['type']}/{self.model_config['name']}")
            return create_model(self.model_config)
    
    def start(self, blocking: bool = True) -> None:
        """
        Start the Flower server.
        
        Args:
            blocking: Whether to run in blocking mode or in a separate thread
        """
        logger.info("Starting FIDS Flower server...")
        
        # Define server address and port
        server_address = f"{self.server_config['address']}:{self.server_config['port']}"
        
        # Configure server
        server_config = fl.server.ServerConfig(
            num_rounds=self.server_config["rounds"]
        )
        
        # Add custom handlers to strategy
        strategy = self._customize_strategy(self.aggregation_strategy)
        
        # Start resource monitoring
        if self.enable_monitoring:
            self._start_monitoring()
        
        try:
            # Determine if secure communication should be used
            if (self.security_config["encryption"]["enabled"] and 
                self.security_config["encryption"]["type"] == "tls" and 
                self.certificates_dir):
                
                # Load certificates
                server_cert = os.path.join(self.certificates_dir, "server.crt")
                server_key = os.path.join(self.certificates_dir, "server.key")
                ca_cert = os.path.join(self.certificates_dir, "ca.crt")
                
                if not all(os.path.exists(f) for f in [server_cert, server_key, ca_cert]):
                    logger.warning("TLS certificates not found, falling back to insecure mode")
                    certificates = None
                else:
                    certificates = (ca_cert, server_cert, server_key)
                    logger.info("Using TLS for secure communication")
            else:
                certificates = None
                logger.info("Starting server in insecure mode")
            
            # Start server
            if blocking:
                # In Flower 1.0.0, start_server returns a History object, not a tuple
                self.history = fl.server.start_server(
                    server_address=server_address,
                    config=server_config,
                    strategy=strategy,
                    certificates=certificates
                )
                
                # Save final metrics
                self._save_metrics_and_model()
            else:
                # Start in a new thread
                self.server_thread = threading.Thread(
                    target=self._start_server_thread,
                    args=(server_address, server_config, strategy, certificates),
                    daemon=True
                )
                self.server_thread.start()
                logger.info("Server started in background thread")
                
        except Exception as e:
            logger.error(f"Error starting server: {e}", exc_info=True)
            self.stop_monitoring.set()
            raise
    
    def _start_server_thread(self, 
                           server_address: str, 
                           server_config: fl.server.ServerConfig,
                           strategy: Strategy, 
                           certificates: Optional[Tuple[str, str, str]]) -> None:
        """
        Start the server in a separate thread.
        
        Args:
            server_address: Server address and port
            server_config: Server configuration
            strategy: Aggregation strategy
            certificates: TLS certificates
        """
        try:
            # In Flower 1.0.0, start_server returns a History object, not a tuple
            self.history = fl.server.start_server(
                server_address=server_address,
                config=server_config,
                strategy=strategy,
                certificates=certificates
            )
            
            # Save final metrics
            self._save_metrics_and_model()
        except Exception as e:
            logger.error(f"Error in server thread: {e}", exc_info=True)
        finally:
            # Stop monitoring
            self.stop_monitoring.set()
    
    def stop(self, timeout: float = 10.0) -> None:
        """
        Stop the server gracefully.
        
        Args:
            timeout: Timeout for stopping threads in seconds
        """
        logger.info("Stopping FIDS server...")
        
        # Stop monitoring
        self.stop_monitoring.set()
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=timeout)
        
        # Stop server if running in non-blocking mode
        if hasattr(self, 'server_thread') and self.server_thread.is_alive():
            # Let's try to give the server a chance to finish gracefully
            self.server_thread.join(timeout=timeout)
            if self.server_thread.is_alive():
                logger.warning(f"Server thread did not stop within {timeout} seconds")
        
        # Save final metrics and model
        self._save_metrics_and_model()
        
        logger.info("FIDS server stopped")
    
    def _customize_strategy(self, strategy: Strategy) -> Strategy:
        """
        Customize the provided strategy with additional callbacks and functionality.
        
        Args:
            strategy: Base aggregation strategy
            
        Returns:
            Enhanced strategy
        """
        # Store original methods to call them from our custom methods
        original_aggregate_fit = strategy.aggregate_fit
        original_aggregate_evaluate = strategy.aggregate_evaluate
        original_evaluate = strategy.evaluate
        original_configure_fit = strategy.configure_fit
        
        # Customize strategy methods
        def custom_configure_fit(
            server_round: int,
            parameters: Parameters,
            client_manager: fl.server.client_manager.ClientManager
        ) -> List[Tuple[fl.server.client_proxy.ClientProxy, Dict]]:
            """Add Paillier public key to client configurations."""
            # Call original configure_fit to get clients and configs
            client_configs = original_configure_fit(server_round, parameters, client_manager)
            
            # Add Paillier public key if available
            if hasattr(self, 'paillier_public_key') and self.paillier_public_key is not None:
                for i, (client, config) in enumerate(client_configs):
                    # Add Paillier public key modulus
                    config["paillier_n"] = str(self.paillier_public_key.n)
                    client_configs[i] = (client, config)
                    
                    logger.debug(f"Added Paillier public key to config for client {client.cid}")
            
            return client_configs
        
        def custom_aggregate_fit(
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
        ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
            """Custom aggregation with authentication and masking handling."""
            logger.info(f"Round {server_round}: Aggregating updates from {len(results)} clients")
            logger.info(f"Round {server_round}: {len(failures)} clients failed to send updates")
            
            # Process and authenticate each client update
            processed_results = []
            authenticated_client_ids = []
            
            for client_proxy, fit_res in results:
                client_id = client_proxy.cid
                
                try:
                    # In a real implementation, we would deserialize protobuf message here
                    # For now, we'll assume the parameters are already in numpy format
                    
                    # Extract metrics indicating authentication
                    metrics = fit_res.metrics
                    is_authenticated = metrics.get("hmac_authenticated", False)
                    is_masked = metrics.get("masked", False)
                    device_id = metrics.get("device_id", "unknown")
                    
                    # Store client device info
                    if device_id != "unknown":
                        self.client_device_info[client_id] = {
                            "device_id": device_id,
                            "device_category": metrics.get("device_category", "unknown"),
                            "last_seen": time.time()
                        }
                    
                    # In a real implementation, we would unmask the parameters if masked
                    if is_masked and self.masking.enabled:
                        # For now, just log this - in reality we would get masked parameters
                        # through protobuf and unmask them
                        logger.info(f"Client {client_id} used masking - would unmask here")
                    
                    # If authenticated, add to the list of authenticated clients
                    if is_authenticated:
                        logger.info(f"Client {client_id} authentication verified")
                        authenticated_client_ids.append(client_id)
                        processed_results.append((client_proxy, fit_res))
                    else:
                        logger.warning(f"Client {client_id} failed authentication, ignoring update")
                
                except Exception as e:
                    logger.error(f"Error processing update from client {client_id}: {e}")
            
            # Track metrics about authentication for this round
            auth_metrics = {
                "authenticated_clients": len(authenticated_client_ids),
                "unauthenticated_clients": len(results) - len(authenticated_client_ids),
                "authentication_rate": len(authenticated_client_ids) / len(results) if results else 0
            }
            
            # Save round metrics for authentication
            self._save_round_metrics(
                server_round,
                "authentication",
                auth_metrics,
                results,
                failures
            )
            
            # If no clients were authenticated, return empty results
            if not processed_results:
                logger.warning(f"Round {server_round}: No authenticated clients, skipping aggregation")
                return None, {}
            
            # Call the original aggregation method with authenticated results
            parameters, metrics = original_aggregate_fit(server_round, processed_results, failures)
            
            # Add authentication metrics to the results
            metrics.update(auth_metrics)
                
            # Save metrics for this round
            self._save_round_metrics(
                server_round,
                "fit",
                metrics,
                processed_results,
                failures
            )
            
            return parameters, metrics
        
        def custom_aggregate_evaluate(
            server_round: int,
            results: List[Tuple[ClientProxy, EvaluateRes]],
            failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
        ) -> Tuple[Optional[float], Dict[str, Scalar]]:
            """Custom evaluate aggregation with authentication handling."""
            # Here, simply filter out unauthenticated clients
            processed_results = []
            for client_proxy, eval_res in results:
                client_id = client_proxy.cid
                metrics = eval_res.metrics
                
                # Check if this client was authenticated
                if client_id in self.verified_clients:
                    processed_results.append((client_proxy, eval_res))
                else:
                    logger.warning(f"Ignoring evaluation from unauthenticated client {client_id}")
            
            # Call original method with authenticated results
            loss, metrics = original_aggregate_evaluate(server_round, processed_results, failures)
            
            # Save metrics for this round
            self._save_round_metrics(
                server_round,
                "evaluate",
                metrics,
                processed_results,
                failures
            )
            
            return loss, metrics
        
        def custom_evaluate(
            server_round: int, parameters: Parameters
        ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
            """Custom central evaluation with additional metrics."""
            eval_result = original_evaluate(server_round, parameters)
            
            if eval_result is not None:
                loss, metrics = eval_result
                
                # Add server resource usage metrics
                if self.enable_monitoring:
                    resource_metrics = monitor_resources()
                    metrics.update({f"server_{k}": v for k, v in resource_metrics.items()})
                
                return loss, metrics
            
            return eval_result
        
        # Replace methods in strategy
        strategy.configure_fit = custom_configure_fit
        strategy.aggregate_fit = custom_aggregate_fit
        strategy.aggregate_evaluate = custom_aggregate_evaluate
        strategy.evaluate = custom_evaluate
        
        return strategy
    
    def _start_monitoring(self) -> None:
        """Start a background thread for resource monitoring."""
        self.stop_monitoring.clear()
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_task,
            daemon=True
        )
        self.monitoring_thread.start()
        logger.debug("Resource monitoring started")
    
    def _monitoring_task(self) -> None:
        """Background task for resource monitoring."""
        interval = 60  # Check every 60 seconds
        
        while not self.stop_monitoring.is_set():
            try:
                # Collect resource usage
                resources = monitor_resources()
                resources["timestamp"] = time.time()
                
                # Store resource usage
                self.resource_usage.append(resources)
                
                # Save to disk occasionally
                if len(self.resource_usage) % 10 == 0:
                    self._save_resource_metrics()
                
            except Exception as e:
                logger.error(f"Error in monitoring task: {e}")
            
            # Wait for interval or until stopped
            for _ in range(interval):
                if self.stop_monitoring.is_set():
                    break
                time.sleep(1)
    
    def _save_round_metrics(
        self, 
        server_round: int, 
        phase: str,
        metrics: Dict[str, Scalar],
        results: List[Tuple[ClientProxy, Any]],
        failures: List[Any]
    ) -> None:
        """
        Save metrics for a training/evaluation round.
        
        Args:
            server_round: Current server round
            phase: 'fit' or 'evaluate'
            metrics: Aggregated metrics
            results: List of successful client results
            failures: List of client failures
        """
        # Create metrics entry
        failed_clients = []
        
        # Process failures properly, handling both tuple failures and exception failures
        for failure in failures:
            if isinstance(failure, tuple) and len(failure) >= 1:
                # Handle tuple failures (client, _)
                failed_clients.append(failure[0].cid)
            elif hasattr(failure, '__str__'):
                # Handle exceptions and other non-tuple failures
                logger.warning(f"Non-tuple failure in round {server_round}: {str(failure)}")
        
        clients_list = []
        for res in results:
            if isinstance(res, tuple) and len(res) >= 1:
                try:
                    clients_list.append(res[0].cid)
                except Exception as e:
                    logger.warning(f"Error extracting client cid: {e} from result: {res}")
            else:
                logger.warning(f"Ignoring invalid result entry (not a tuple): {res}")
        
        metrics_entry = {
            "round": server_round,
            "phase": phase,
            "timestamp": time.time(),
            "metrics": metrics,
            "clients": clients_list,
            "failed_clients": failed_clients
        }
        
        # Add to metrics list
        self.round_metrics.append(metrics_entry)
        
        # Save to disk
        self._save_metrics()
    
    def _save_metrics(self) -> None:
        """Save round metrics to disk."""
        metrics_file = os.path.join(self.metrics_dir, "round_metrics.json")
        try:
            with open(metrics_file, 'w') as f:
                json.dump(self.round_metrics, f, indent=2)
            logger.debug(f"Round metrics saved to {metrics_file}")
        except Exception as e:
            logger.error(f"Error saving round metrics: {e}")
    
    def _save_resource_metrics(self) -> None:
        """Save resource metrics to disk."""
        resource_file = os.path.join(self.metrics_dir, "resource_metrics.json")
        try:
            with open(resource_file, 'w') as f:
                json.dump(self.resource_usage, f, indent=2)
            logger.debug(f"Resource metrics saved to {resource_file}")
        except Exception as e:
            logger.error(f"Error saving resource metrics: {e}")
    
    def _save_metrics_and_model(self) -> None:
        """Save all metrics and the final model."""
        try:
            # Save metrics
            self._save_metrics()
            self._save_resource_metrics()
            
            # Save final history if available
            if self.history:
                history_file = os.path.join(self.metrics_dir, "training_history.json")
                with open(history_file, 'w') as f:
                    # Convert history to serializable format
                    history_dict = {
                        "metrics_distributed": self.history.metrics_distributed,
                        "metrics_centralized": self.history.metrics_centralized,
                        "losses_distributed": self.history.losses_distributed,
                        "losses_centralized": self.history.losses_centralized,
                    }
                    json.dump(history_dict, f, indent=2)
                logger.info(f"Training history saved to {history_file}")
            
            # Save final model if available
            if self.server and hasattr(self.server, "parameters"):
                final_model_path = os.path.join(self.metrics_dir, "final_model.bin")
                save_model(self.server.parameters, final_model_path)
                logger.info(f"Final model saved to {final_model_path}")
        
        except Exception as e:
            logger.error(f"Error saving final metrics and model: {e}")
    
    def get_metrics(self) -> Dict:
        """
        Get server metrics and statistics.
        
        Returns:
            Dictionary with server metrics
        """
        return {
            "round_metrics": self.round_metrics,
            "resource_usage": self.resource_usage[-10:],  # Return last 10 resource measurements
            "server_config": self.server_config,
            "rounds_completed": len(self.round_metrics) // 2,  # Approximate
            "clients_seen": len(set(
                client for metrics in self.round_metrics 
                for client in metrics.get("clients", [])
            )),
            "latest_metrics": self.round_metrics[-1]["metrics"] if self.round_metrics else {}
        }

    # Health check endpoint for Kubernetes/Docker
    def health_check(self) -> Dict[str, str]:
        """
        Health check endpoint for deployment environments.
        
        Returns:
            Dictionary with health status
        """
        is_running = (
            hasattr(self, 'server_thread') and 
            self.server_thread.is_alive()
        ) if hasattr(self, 'server_thread') else True
        
        return {
            "status": "healthy" if is_running else "unhealthy",
            "timestamp": str(time.time()),
            "version": "1.0.0",  # Add version info
            "monitoring": "active" if (self.monitoring_thread and self.monitoring_thread.is_alive()) else "inactive"
        }

    def _get_model_shape(self) -> List[Tuple]:
        """
        Get the shapes of all layers in the model.
        
        Returns:
            List of tuples representing shapes of model layers
        """
        try:
            # Get model weights
            model_weights = self.model.get_weights()
            # Extract shapes
            return [layer.shape for layer in model_weights]
        except Exception as e:
            logger.error(f"Error getting model shape: {e}")
            return []

def run_standalone_server():
    """Run the server in standalone mode."""
    import argparse
    import yaml
    from utils.logging_utils import setup_logging
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="FIDS Federated Learning Server")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--model", type=str, default=None, help="Path to initial model")
    parser.add_argument("--certs", type=str, default=None, help="Path to certificates directory")
    parser.add_argument("--metrics", type=str, default="./metrics", help="Path to metrics directory")
    parser.add_argument("--port", type=int, default=None, help="Server port (overrides config)")
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # Override port if specified
    if args.port:
        config["server"]["port"] = args.port
    
    # Set up logging
    setup_logging(config["logging"])
    
    # Create and start server
    server = FlowerServer(
        config=config,
        model_path=args.model,
        certificates_dir=args.certs,
        metrics_dir=args.metrics
    )
    
    try:
        # Handle graceful shutdown
        import signal
        
        def signal_handler(sig, frame):
            logger.info("Shutdown signal received, stopping server...")
            server.stop()
            import sys
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Start server
        server.start(blocking=True)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, stopping server...")
        server.stop()
    except Exception as e:
        logger.error(f"Error running server: {e}", exc_info=True)
        server.stop()
        raise

if __name__ == "__main__":
    run_standalone_server()

def aggregate_metrics(metrics_list):
    """Aggregate metrics from multiple clients."""
    if not metrics_list:
        return {}
        
    # Initialize aggregated metrics
    aggregated = {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "accuracy": 0.0,
        "confusion_matrix": None,
        "num_clients": len(metrics_list)
    }
    
    # Add AUC-ROC if present
    if "auc_roc" in metrics_list[0]:
        aggregated["auc_roc"] = 0.0
    
    # Sum up all metrics
    for metrics in metrics_list:
        aggregated["precision"] += metrics["precision"]
        aggregated["recall"] += metrics["recall"]
        aggregated["f1"] += metrics["f1"]
        aggregated["accuracy"] += metrics["accuracy"]
        
        if "auc_roc" in metrics:
            aggregated["auc_roc"] += metrics["auc_roc"]
            
        # Sum confusion matrices
        if aggregated["confusion_matrix"] is None:
            aggregated["confusion_matrix"] = np.array(metrics["confusion_matrix"])
        else:
            aggregated["confusion_matrix"] += np.array(metrics["confusion_matrix"])
    
    # Calculate averages
    n = len(metrics_list)
    aggregated["precision"] /= n
    aggregated["recall"] /= n
    aggregated["f1"] /= n
    aggregated["accuracy"] /= n
    if "auc_roc" in aggregated:
        aggregated["auc_roc"] /= n
        
    # Convert confusion matrix back to list
    aggregated["confusion_matrix"] = aggregated["confusion_matrix"].tolist()
    
    # Log aggregated metrics
    print("\nAggregated Evaluation Metrics:")
    print(f"Number of Clients: {n}")
    print(f"Average Precision: {aggregated['precision']:.4f}")
    print(f"Average Recall: {aggregated['recall']:.4f}")
    print(f"Average F1 Score: {aggregated['f1']:.4f}")
    print(f"Average Accuracy: {aggregated['accuracy']:.4f}")
    if "auc_roc" in aggregated:
        print(f"Average AUC-ROC: {aggregated['auc_roc']:.4f}")
    print("\nAggregate Confusion Matrix:")
    print(np.array(aggregated["confusion_matrix"]))
    
    return aggregated

def aggregate_evaluate(self, server_round, results, failures):
    """Aggregate evaluation results from multiple clients."""
    if not results:
        return None
    
    # Aggregate metrics from all clients
    metrics_list = [r.metrics for r in results]
    aggregated_metrics = aggregate_metrics(metrics_list)
    
    # Calculate weighted average loss
    loss_aggregated = weighted_average([r.loss for r in results], [r.num_examples for r in results])
    
    return loss_aggregated, aggregated_metrics