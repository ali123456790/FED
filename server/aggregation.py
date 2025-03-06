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

logger = logging.getLogger(__name__)

def get_aggregation_strategy(
    strategy_name: str, 
    security_config: Dict,
    config: Dict
) -> fl.server.strategy.Strategy:
    """
    Factory function to create an aggregation strategy.
    
    Args:
        strategy_name: Name of the strategy to use
        security_config: Security configuration
        config: Full configuration dictionary
        
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
    
    # Create appropriate strategy
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