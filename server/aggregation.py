"""
Aggregation strategies for federated learning.

This module provides different aggregation strategies for the FIDS federated learning system.
It includes implementations of FedAvg, FedProx, and FedOpt strategies, with additional
security enhancements and customizations for IoT device security applications.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable, cast
import flwr as fl
from flwr.common import (
    Metrics, Parameters, Scalar, FitRes, EvaluateRes,
    parameters_to_ndarrays, ndarrays_to_parameters
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import (
    Strategy, FedAvg, FedOpt, FedAdagrad, FedAdam, FedYogi
)
from phe import paillier
from security.encoding import decode_integer
from models.model_messages_pb2 import EdgeAggregateProto
from security.secure_aggregation import SecureAggregation, RobustAggregation, clipping

logger = logging.getLogger(__name__)

def get_aggregation_strategy(
    strategy_name: str, 
    security_config: Dict,
    config: Dict,
    server_instance: Any = None
) -> fl.server.strategy.Strategy:
    """
    Factory function to create an aggregation strategy.
    
    Args:
        strategy_name: Name of the strategy to use
        security_config: Security configuration
        config: Full configuration dictionary
        server_instance: Optional server instance for strategies that need it
        
    Returns:
        A Flower strategy
    """
    logger.info(f"Creating aggregation strategy: {strategy_name}")
    
    # Extract server config
    server_config = config["server"]
    
    # Common strategy parameters
    common_params = {
        "fraction_fit": server_config.get("fraction_fit", 1.0),
        "fraction_evaluate": server_config.get("fraction_evaluate", 1.0),
        "min_fit_clients": server_config.get("min_clients", 2),
        "min_evaluate_clients": server_config.get("min_clients", 2),
        "min_available_clients": server_config.get("min_available_clients", 2),
        "on_fit_config_fn": get_fit_config_fn(config),
        "on_evaluate_config_fn": get_evaluate_config_fn(config),
        "evaluate_metrics_aggregation_fn": weighted_average_metrics,
        "accept_failures": True  # Accept some failures to improve resilience
    }
    
    # Check if Paillier encryption is enabled
    paillier_enabled = (security_config.get("encryption", {}).get("enabled", False) and 
                       security_config.get("encryption", {}).get("type", "") == "paillier")
    
    if paillier_enabled and server_instance:
        # If Paillier is enabled and we have the server instance, use the Paillier strategy
        logger.info("Using Paillier homomorphic encryption strategy")
        return create_paillier_fedavg_strategy(security_config, server_instance, **common_params)
    
    # Regular strategies without Paillier encryption
    if strategy_name.lower() == "fedavg":
        return create_fedavg_strategy(security_config, **common_params)
    elif strategy_name.lower() == "fedprox":
        return create_fedprox_strategy(security_config, config, **common_params)
    elif strategy_name.lower() == "fedopt" or strategy_name.lower() == "fedadam":
        return create_fedopt_strategy(security_config, config, **common_params)
    elif strategy_name.lower() == "fedadagrad":
        return create_fedadagrad_strategy(security_config, **common_params)
    elif strategy_name.lower() == "fedyogi":
        return create_fedyogi_strategy(security_config, **common_params)
    else:
        logger.warning(f"Unknown aggregation strategy: {strategy_name}, using FedAvg")
        return create_fedavg_strategy(security_config, **common_params)

def create_fedavg_strategy(
    security_config: Dict, 
    **kwargs
) -> fl.server.strategy.Strategy:
    """
    Create a FedAvg strategy with security enhancements if enabled.
    
    Args:
        security_config: Security configuration
        **kwargs: Additional parameters for the strategy
        
    Returns:
        FedAvg strategy
    """
    # Create base strategy
    strategy = FedAvg(**kwargs)
    
    # Apply security enhancements if enabled
    if security_config["secure_aggregation"]["enabled"]:
        return enhance_with_secure_aggregation(strategy, security_config)
    
    return strategy

def create_fedprox_strategy(
    security_config: Dict,
    config: Dict,
    **kwargs
) -> fl.server.strategy.Strategy:
    """
    Create a FedProx strategy with security enhancements if enabled.
    
    Args:
        security_config: Security configuration
        config: Full configuration dictionary
        **kwargs: Additional parameters for the strategy
        
    Returns:
        FedProx strategy
    """
    # Since Flower doesn't have a native FedProx implementation,
    # we extend FedAvg with proximal term in the client training
    
    # Get the base on_fit_config_fn
    base_fit_config_fn = kwargs.get("on_fit_config_fn", lambda _: {})
    
    # Get the proximal term mu from config
    mu = config.get("fedprox", {}).get("mu", 0.01)
    
    # Override the fit config function to include proximal term
    def fedprox_fit_config_fn(server_round: int) -> Dict[str, Scalar]:
        """Return fit configuration with proximal term."""
        base_config = base_fit_config_fn(server_round)
        base_config["proximal_mu"] = mu
        return base_config
    
    # Update kwargs with new fit config function
    kwargs["on_fit_config_fn"] = fedprox_fit_config_fn
    
    # Create strategy
    strategy = FedAvg(**kwargs)
    
    # Apply security enhancements if enabled
    if security_config["secure_aggregation"]["enabled"]:
        return enhance_with_secure_aggregation(strategy, security_config)
    
    return strategy

def create_fedopt_strategy(
    security_config: Dict,
    config: Dict,
    **kwargs
) -> fl.server.strategy.Strategy:
    """
    Create a FedOpt (FedAdam) strategy with security enhancements if enabled.
    
    Args:
        security_config: Security configuration
        config: Full configuration dictionary
        **kwargs: Additional parameters for the strategy
        
    Returns:
        FedOpt strategy
    """
    # Get FedOpt parameters from config
    fedopt_config = config.get("fedopt", {})
    tau = fedopt_config.get("tau", 0.1)
    eta = fedopt_config.get("eta", 0.01)
    eta_l = fedopt_config.get("eta_l", 0.0)
    beta_1 = fedopt_config.get("beta_1", 0.9)
    beta_2 = fedopt_config.get("beta_2", 0.99)
    
    # Create FedAdam strategy
    strategy = FedAdam(
        tau=tau,
        eta=eta,
        eta_l=eta_l,
        beta_1=beta_1,
        beta_2=beta_2,
        **kwargs
    )
    
    # Apply security enhancements if enabled
    if security_config["secure_aggregation"]["enabled"]:
        return enhance_with_secure_aggregation(strategy, security_config)
    
    return strategy

def create_fedadagrad_strategy(
    security_config: Dict,
    **kwargs
) -> fl.server.strategy.Strategy:
    """
    Create a FedAdagrad strategy with security enhancements if enabled.
    
    Args:
        security_config: Security configuration
        **kwargs: Additional parameters for the strategy
        
    Returns:
        FedAdagrad strategy
    """
    # Create FedAdagrad strategy
    strategy = FedAdagrad(
        eta=0.1,  # Server-side learning rate
        eta_l=0.01,  # Client-side learning rate
        tau=0.1,  # Controls the importance of the initial point
        **kwargs
    )
    
    # Apply security enhancements if enabled
    if security_config["secure_aggregation"]["enabled"]:
        return enhance_with_secure_aggregation(strategy, security_config)
    
    return strategy

def create_fedyogi_strategy(
    security_config: Dict,
    **kwargs
) -> fl.server.strategy.Strategy:
    """
    Create a FedYogi strategy with security enhancements if enabled.
    
    Args:
        security_config: Security configuration
        **kwargs: Additional parameters for the strategy
        
    Returns:
        FedYogi strategy
    """
    # Create FedYogi strategy
    strategy = FedYogi(
        eta=0.1,  # Server-side learning rate
        eta_l=0.01,  # Client-side learning rate
        tau=0.1,  # Controls the importance of the initial point
        beta_1=0.9,
        beta_2=0.99,
        **kwargs
    )
    
    # Apply security enhancements if enabled
    if security_config["secure_aggregation"]["enabled"]:
        return enhance_with_secure_aggregation(strategy, security_config)
    
    return strategy

def enhance_with_secure_aggregation(
    strategy: Strategy,
    security_config: Dict
) -> Strategy:
    """
    Enhance a strategy with secure aggregation.
    
    Args:
        strategy: Base strategy to enhance
        security_config: Security configuration
        
    Returns:
        Enhanced strategy with secure aggregation
    """
    # Save original aggregate_fit and aggregate_evaluate methods
    original_aggregate_fit = strategy.aggregate_fit
    original_aggregate_evaluate = strategy.aggregate_evaluate
    
    # Determine secure aggregation type
    secure_agg_type = security_config["secure_aggregation"]["type"]
    logger.info(f"Enhancing strategy with {secure_agg_type} secure aggregation")
    
    # Create secure aggregation wrapper
    if secure_agg_type == "secure_sum":
        from security.secure_aggregation import SecureAggregation
        secure_agg = SecureAggregation(security_config)
        
        def secure_aggregate_fit(
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]
        ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
            """Securely aggregate model updates."""
            # Get weights and num_samples from results
            weights_results = [
                (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
                for _, fit_res in results
            ]
            
            # Don't apply secure aggregation if no results
            if not weights_results:
                return original_aggregate_fit(server_round, results, failures)
            
            # Extract weights and num_samples
            weights = [w for w, _ in weights_results]
            num_samples = [n for _, n in weights_results]
            
            try:
                # Apply secure aggregation
                aggregated_weights = secure_agg.aggregate(weights, num_samples)
                
                # Convert back to parameters
                parameters = ndarrays_to_parameters(aggregated_weights)
                
                # Call original method to get metrics
                _, metrics = original_aggregate_fit(server_round, results, failures)
                
                # Add secure aggregation metrics
                if metrics is None:
                    metrics = {}
                metrics["secure_aggregation"] = True
                metrics["secure_aggregation_type"] = secure_agg_type
                
                return parameters, metrics
                
            except Exception as e:
                logger.error(f"Error in secure aggregation: {e}, falling back to regular aggregation")
                return original_aggregate_fit(server_round, results, failures)
        
        # Replace aggregate_fit with secure version
        strategy.aggregate_fit = secure_aggregate_fit
        
        # Do the same for evaluate
        def secure_aggregate_evaluate(
            server_round: int,
            results: List[Tuple[ClientProxy, EvaluateRes]],
            failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]]
        ) -> Tuple[Optional[float], Dict[str, Scalar]]:
            """Securely aggregate evaluation results."""
            try:
                return original_aggregate_evaluate(server_round, results, failures)
            except Exception as e:
                logger.error(f"Error in secure evaluation aggregation: {e}")
                return None, {"error": "Secure aggregation failed"}
        
        strategy.aggregate_evaluate = secure_aggregate_evaluate
    
    elif secure_agg_type == "robust_aggregation":
        # Implement robust aggregation to defend against poisoning attacks
        from security.secure_aggregation import SecureAggregation
        secure_agg = SecureAggregation(security_config)
        
        def robust_aggregate_fit(
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]
        ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
            """Robustly aggregate model updates, filtering potential poisoning attacks."""
            # Get weights and num_samples from results
            weights_results = [
                (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
                for _, fit_res in results
            ]
            
            # Don't apply robust aggregation if no results
            if not weights_results:
                return original_aggregate_fit(server_round, results, failures)
            
            # Extract weights and num_samples
            weights = [w for w, _ in weights_results]
            num_samples = [n for _, n in weights_results]
            
            try:
                # Apply robust aggregation
                aggregated_weights = secure_agg.aggregate(weights, num_samples)
                
                # Convert back to parameters
                parameters = ndarrays_to_parameters(aggregated_weights)
                
                # Call original method to get metrics
                _, metrics = original_aggregate_fit(server_round, results, failures)
                
                # Add robust aggregation metrics
                if metrics is None:
                    metrics = {}
                metrics["secure_aggregation"] = True
                metrics["secure_aggregation_type"] = secure_agg_type
                
                return parameters, metrics
                
            except Exception as e:
                logger.error(f"Error in robust aggregation: {e}, falling back to regular aggregation")
                return original_aggregate_fit(server_round, results, failures)
        
        # Replace aggregate_fit with robust version
        strategy.aggregate_fit = robust_aggregate_fit
    
    else:
        logger.warning(f"Unknown secure aggregation type: {secure_agg_type}, no enhancement applied")
    
    return strategy

def get_fit_config_fn(config: Dict) -> Callable[[int], Dict[str, Scalar]]:
    """
    Create a function that returns the configuration for client training.
    
    Args:
        config: Full configuration dictionary
        
    Returns:
        Function that returns training configuration
    """
    # Extract client config
    client_config = config.get("client", {})
    security_config = config.get("security", {})
    
    # Check if Paillier encryption is enabled
    paillier_enabled = (security_config.get("encryption", {}).get("enabled", False) and 
                       security_config.get("encryption", {}).get("type", "") == "paillier")
    
    # Get Paillier configuration if enabled
    paillier_precision_bits = None
    if paillier_enabled:
        paillier_precision_bits = security_config.get("encryption", {}).get("paillier", {}).get("encoding_precision_bits", 64)
    
    def fit_config_fn(server_round: int) -> Dict[str, Scalar]:
        """Return training configuration dict for each round."""
        # Default configuration
        fit_config = {
            "server_round": server_round,
            "local_epochs": client_config.get("local_epochs", 1),
            "batch_size": client_config.get("batch_size", 32),
            "learning_rate": client_config.get("learning_rate", 0.01),
            "optimizer": client_config.get("optimizer", "adam")
        }
        
        # Add resource monitoring config
        fit_config["resource_monitoring"] = client_config.get("resource_monitoring", True)
        
        # Learning rate scheduling (decrease learning rate for later rounds)
        if server_round > 10:
            fit_config["learning_rate"] = fit_config["learning_rate"] * 0.5
        if server_round > 20:
            fit_config["learning_rate"] = fit_config["learning_rate"] * 0.5
        
        # Add Paillier configuration if enabled
        if paillier_enabled and paillier_precision_bits is not None:
            fit_config["paillier_enabled"] = True
            fit_config["paillier_precision_bits"] = paillier_precision_bits
        
        return fit_config
    
    return fit_config_fn

def get_evaluate_config_fn(config: Dict) -> Callable[[int], Dict[str, Scalar]]:
    """
    Create a function that returns the configuration for client evaluation.
    
    Args:
        config: Full configuration dictionary
        
    Returns:
        Function that returns evaluation configuration
    """
    # Extract evaluation config
    evaluation_config = config.get("evaluation", {})
    
    def evaluate_config_fn(server_round: int) -> Dict[str, Scalar]:
        """Return evaluation configuration dict for each round."""
        # Default configuration
        eval_config = {
            "server_round": server_round,
            "batch_size": config.get("client", {}).get("batch_size", 32)
        }
        
        # Add metrics to compute
        eval_config["metrics"] = evaluation_config.get("metrics", ["accuracy", "precision", "recall", "f1"])
        
        return eval_config
    
    return evaluate_config_fn

def weighted_average_metrics(metrics: List[Tuple[int, Dict[str, Scalar]]]) -> Dict[str, Scalar]:
    """
    Aggregate evaluation metrics weighted by number of examples.
    
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

class CustomFedAvg(FedAvg):
    """
    Custom FedAvg implementation with additional features.
    
    This class extends Flower's FedAvg strategy with:
    - Client selection based on device capabilities
    - Support for dropping slow clients
    - Adaptive aggregation weights
    - Additional metrics and monitoring
    """
    
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[Callable[[Parameters], Optional[Tuple[float, Dict[str, Scalar]]]]] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[Callable[[List[Tuple[ClientProxy, FitRes]]], Dict[str, Scalar]]] = None,
        evaluate_metrics_aggregation_fn: Optional[Callable[[List[Tuple[int, Dict[str, Scalar]]]], Dict[str, Scalar]]] = None,
        # Custom parameters
        drop_slow_clients: bool = False,
        slow_client_threshold: float = 2.0,  # Multiple of median training time
        prioritize_by_resources: bool = True,
        security_config: Optional[Dict] = None
    ) -> None:
        """Initialize CustomFedAvg strategy."""
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn
        )
        
        # Custom attributes
        self.drop_slow_clients = drop_slow_clients
        self.slow_client_threshold = slow_client_threshold
        self.prioritize_by_resources = prioritize_by_resources
        self.security_config = security_config
        
        # Tracking client performance
        self.client_training_times: Dict[str, List[float]] = {}
        self.client_resources: Dict[str, Dict] = {}
    
    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, Dict]]:
        """Configure the next round of training."""
        # Get random sample of clients
        sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
        
        # If prioritizing by resources, select clients based on their capabilities
        if self.prioritize_by_resources and self.client_resources:
            # Get all available clients
            available_clients = client_manager.all().values()
            
            # Sort clients by CPU and memory resources if available
            sorted_clients = []
            for client in available_clients:
                if client.cid in self.client_resources:
                    # Calculate a resource score (higher is better)
                    resources = self.client_resources[client.cid]
                    if "cpu_percent" in resources and "memory_percent" in resources:
                        # Lower percentages mean more available resources
                        resource_score = 100 - (resources["cpu_percent"] + resources["memory_percent"]) / 2
                    else:
                        resource_score = 50  # Default score if no resource info
                    
                    # Penalize slow clients if drop_slow_clients is enabled
                    if self.drop_slow_clients and client.cid in self.client_training_times:
                        training_times = self.client_training_times[client.cid]
                        if training_times:
                            avg_time = sum(training_times) / len(training_times)
                            
                            # Get median time across all clients
                            all_times = [t for times in self.client_training_times.values() for t in times]
                            if all_times:
                                median_time = sorted(all_times)[len(all_times) // 2]
                                
                                # If client is slow, reduce its score
                                if avg_time > median_time * self.slow_client_threshold:
                                    resource_score *= 0.5  # Penalize slow clients
                    
                    sorted_clients.append((client, resource_score))
                else:
                    # If no resource info, use a default score
                    sorted_clients.append((client, 50))
            
            # Sort by resource score (higher is better)
            sorted_clients.sort(key=lambda x: x[1], reverse=True)
            
            # Select top clients
            clients = [client for client, _ in sorted_clients[:sample_size]]
            
            # Ensure we have enough clients
            if len(clients) < min_num_clients:
                # Fallback to random selection for remaining clients
                remaining = client_manager.sample(
                    num_clients=min_num_clients - len(clients),
                    exclude=[client.cid for client in clients]
                )
                clients.extend(remaining)
        else:
            # Use default random selection
            clients = client_manager.sample(
                num_clients=sample_size, 
                min_num_clients=min_num_clients
            )
        
        # Create and return client/config pairs
        config = {}
        if self.on_fit_config_fn is not None:
            config = self.on_fit_config_fn(server_round)
        
        return [(client, config) for client in clients]
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate model updates from clients."""
        # Update client training time tracking
        for client, fit_res in results:
            if "training_time" in fit_res.metrics:
                training_time = fit_res.metrics["training_time"]
                if client.cid not in self.client_training_times:
                    self.client_training_times[client.cid] = []
                self.client_training_times[client.cid].append(training_time)
            
            # Update resource info
            resource_keys = ["cpu_percent", "memory_percent", "battery_percent", "device_type"]
            resource_info = {k: fit_res.metrics.get(k, -1) for k in resource_keys if k in fit_res.metrics}
            if resource_info:
                self.client_resources[client.cid] = resource_info
        
        # Call parent method
        return super().aggregate_fit(server_round, results, failures)

def create_paillier_fedavg_strategy(
    security_config: Dict,
    server_instance: Any,
    **kwargs
) -> fl.server.strategy.Strategy:
    """
    Create a FedAvg strategy with Paillier homomorphic encryption support.
    
    Args:
        security_config: Security configuration
        server_instance: The server instance containing Paillier keys
        **kwargs: Additional parameters for the strategy
        
    Returns:
        FedAvg strategy with Paillier support
    """
    class PaillierFedAvg(FedAvg):
        """
        FedAvg strategy with Paillier homomorphic encryption support.
        
        This strategy extends the Flower FedAvg strategy to handle Paillier 
        homomorphically encrypted updates from edges/clients, decrypt them,
        and apply robust aggregation.
        """
        
        def __init__(
            self,
            *,
            fraction_fit: float = 1.0,
            fraction_evaluate: float = 1.0,
            min_fit_clients: int = 2,
            min_evaluate_clients: int = 2,
            min_available_clients: int = 2,
            evaluate_fn: Optional[
                Callable[[NDArrays], Optional[Tuple[float, Dict[str, Scalar]]]]
            ] = None,
            on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
            on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
            accept_failures: bool = True,
            initial_parameters: Optional[Parameters] = None,
            fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
            evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
            server_instance: Optional[Any] = None,
            config: Optional[Dict] = None,
            security_manager: Optional[Any] = None,
        ):
            """
            Initialize the PaillierFedAvg strategy.
            
            Args:
                fraction_fit: Fraction of clients to use for training, float between 0 and 1
                fraction_evaluate: Fraction of clients to use for evaluation, float between 0 and 1
                min_fit_clients: Minimum number of clients to use for training
                min_evaluate_clients: Minimum number of clients to use for evaluation
                min_available_clients: Minimum number of available clients required for training
                evaluate_fn: Optional function to evaluate the model
                on_fit_config_fn: Optional function to configure fit
                on_evaluate_config_fn: Optional function to configure evaluation
                accept_failures: Whether to accept client failures
                initial_parameters: Initial global model parameters
                fit_metrics_aggregation_fn: Function to aggregate training metrics
                evaluate_metrics_aggregation_fn: Function to aggregate evaluation metrics
                server_instance: Server instance for access to Paillier keys
                config: Additional configuration parameters
                security_manager: Security manager instance for access to HMAC keys
            """
            super().__init__(
                fraction_fit=fraction_fit,
                fraction_evaluate=fraction_evaluate,
                min_fit_clients=min_fit_clients,
                min_evaluate_clients=min_evaluate_clients,
                min_available_clients=min_available_clients,
                evaluate_fn=evaluate_fn,
                on_fit_config_fn=on_fit_config_fn,
                on_evaluate_config_fn=on_evaluate_config_fn,
                accept_failures=accept_failures,
                initial_parameters=initial_parameters,
                fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
                evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
            )
            
            # Store server reference for access to Paillier keys
            self.server = server_instance
            self.config = config or {}
            self.security_manager = security_manager
            
            # For accessing Edge-Server key (K_ES)
            self.edge_server_key = None
            if security_manager:
                try:
                    self.edge_server_key = security_manager.get_edge_server_key()
                    logger.info("Using Edge-Server key (K_ES) from security manager")
                except Exception as e:
                    logger.warning(f"Unable to get Edge-Server key: {e}")
            
            logger.info("Initialized PaillierFedAvg strategy with homomorphic encryption support")

        def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
        ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
            """
            Aggregate fit results from multiple clients/edges.
            
            This method handles the decryption and aggregation of Paillier homomorphically
            encrypted updates from edge nodes.
            
            Args:
                server_round: The current round of federated learning
                results: Results from successful clients/edges
                failures: Failures from unsuccessful clients/edges
                
            Returns:
                Tuple of (aggregated_parameters, metrics)
            """
            if not results:
                return None, {}
            
            if failures:
                logger.warning(f"Failures during fit round {server_round}: {failures}")
            
            # Check if server has the Paillier keys
            if not hasattr(self.server, "paillier_secret_key") or self.server.paillier_secret_key is None:
                logger.error("Server does not have Paillier secret key for decryption")
                return None, {}
            
            # Start processing results from edges
            aggregated_data_from_edges = []  # Store (decrypted_float_sum_list, num_examples) from each edge
            
            for _, fit_res in results:
                # 1. Deserialize FitRes bytes to get EdgeAggregateProto bytes
                try:
                    edge_proto_bytes = fit_res.parameters.tensors[0]  # Adjust as needed
                    
                    # 2. Verify HMAC (requires K_ES key)
                    edge_agg = EdgeAggregateProto()
                    edge_agg.ParseFromString(edge_proto_bytes)
                    
                    # Get the HMAC and communication path
                    received_hmac = edge_agg.hmac_digest
                    edge_id = edge_agg.edge_id
                    communication_path = getattr(edge_agg, "communication_path", "edge_server")
                    
                    # Clear HMAC for verification
                    original_hmac = edge_agg.hmac_digest
                    edge_agg.hmac_digest = b""
                    serialized_data_for_hmac = edge_agg.SerializeToString()
                    
                    # Verify HMAC using Edge-Server key (K_ES)
                    hmac_verified = False
                    if self.security_manager:
                        hmac_verified = self.security_manager.verify_hmac(
                            message=serialized_data_for_hmac,
                            signature=original_hmac,
                            communication_path=communication_path
                        )
                    
                    if not hmac_verified and self.edge_server_key:
                        # Fallback to direct verification using edge_server_key
                        import hmac as hmac_lib
                        import hashlib
                        h = hmac_lib.new(self.edge_server_key, digestmod=hashlib.sha256)
                        h.update(edge_id.encode())
                        h.update(str(edge_agg.round_id).encode())
                        h.update(serialized_data_for_hmac)
                        hmac_verified = hmac_lib.compare_digest(h.digest(), original_hmac)
                    
                    if not hmac_verified:
                        logger.warning(f"HMAC verification failed for edge {edge_id}")
                        continue
                    
                    # 4. Decrypt and Decode
                    agg_cipher_strs = edge_agg.aggregated_paillier_ciphertexts
                    num_examples_edge = edge_agg.total_examples
                    precision_bits = self.config.get("security", {}).get("encryption", {}).get("paillier", {}).get("encoding_precision_bits", 64)
                    
                    decrypted_int_sum_list = []
                    for agg_cs in agg_cipher_strs:
                        try:
                            ciphertext_obj = paillier.EncryptedNumber(
                                self.server.paillier_public_key, 
                                int(agg_cs)
                            )
                            decrypted_int_sum = self.server.paillier_secret_key.decrypt(ciphertext_obj)
                            decrypted_int_sum_list.append(decrypted_int_sum)
                        except Exception as e:
                            logger.error(f"Decryption failed for ciphertext {agg_cs[:20]}...: {e}")
                            # Handle error: skip edge, use zero, stop?
                            decrypted_int_sum_list = None  # Signal failure
                            break
                    
                    if decrypted_int_sum_list is not None:
                        # Decode the integer sums to float sums
                        decrypted_float_sum_list = [decode_integer(val, precision_bits) for val in decrypted_int_sum_list]
                        aggregated_data_from_edges.append((decrypted_float_sum_list, num_examples_edge))
                        logger.info(f"Successfully decrypted update from edge {edge_id}")
                    else:
                        logger.warning(f"Skipping results from edge {edge_id} due to decryption failure.")
                
                except Exception as e:
                    logger.error(f"Error processing edge update: {e}")
            
            # Check if any valid results were received
            if not aggregated_data_from_edges:
                logger.warning("No valid aggregated data received from any edge/client.")
                return None, {}  # Return empty aggregate
            
            # Aggregate decrypted sums and total examples
            num_weights = len(aggregated_data_from_edges[0][0])  # Get size from first result
            final_float_sum = np.zeros(num_weights, dtype=float)
            total_examples_overall = 0
            
            for float_sum_list, num_examples in aggregated_data_from_edges:
                if len(float_sum_list) == num_weights:  # Basic sanity check
                    final_float_sum += np.array(float_sum_list)
                    total_examples_overall += num_examples
                else:
                    logger.warning(f"Mismatch in weights length from an edge result. Expected {num_weights}, got {len(float_sum_list)}. Skipping.")
            
            if total_examples_overall == 0:
                logger.warning("Total examples from valid results is zero.")
                return None, {}
            
            # Calculate weighted average (this IS FedAvg essentially)
            average_update_np = final_float_sum / total_examples_overall
            
            # Apply Robust Aggregation
            # This is a simplified approach as robust aggregation is typically applied
            # before averaging, but with PHE we can only apply it after decryption
            from security.secure_aggregation import clipping
            final_update_np = clipping(average_update_np, clip_threshold=3.0)  # 3 std deviations
            
            # Reshape final_update_np back to the model's layer structure if we have shape info
            if hasattr(self.server, 'model_shape') and self.server.model_shape:
                model_shape = self.server.model_shape
                reshaped_weights = []
                idx = 0
                for shape in model_shape:
                    size = np.prod(shape)
                    layer_weights = final_update_np[idx:idx+size].reshape(shape)
                    reshaped_weights.append(layer_weights)
                    idx += size
                
                # Convert to Parameters object
                parameters = ndarrays_to_parameters(reshaped_weights)
                
                # Return with metrics
                metrics = {
                    "aggregation_type": "paillier_homomorphic",
                    "num_edges": len(aggregated_data_from_edges),
                    "total_examples": total_examples_overall,
                    "robust_method": "clipping"
                }
                
                return parameters, metrics
            else:
                logger.error("Missing model shape information for reshaping")
                return None, {"error": "missing_model_shape"}