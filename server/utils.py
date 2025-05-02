"""
Utility functions for the FIDS server.

This module provides utility functions for the federated learning server,
including model saving/loading, resource monitoring, and logging functionality.
"""

import os
import pickle
import json
import time
import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import psutil
from flwr.common import Parameters, parameters_to_ndarrays, ndarrays_to_parameters

logger = logging.getLogger(__name__)

def save_model(model: Any, path: str) -> None:
    """
    Save a model to disk.
    
    Args:
        model: The model to save (can be Parameters, tensors, or an actual model)
        path: Path to save the model to
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model based on type
        if isinstance(model, Parameters):
            # Handle Flower Parameters type
            weights = parameters_to_ndarrays(model)
            np.savez_compressed(path, *weights)
            logger.info(f"Saved model parameters to {path}")
        
        elif isinstance(model, list) and all(isinstance(x, np.ndarray) for x in model):
            # Handle list of NumPy arrays
            np.savez_compressed(path, *model)
            logger.info(f"Saved model weights to {path}")
        
        elif hasattr(model, 'save'):
            # Handle models with save method (like Keras)
            model.save(path)
            logger.info(f"Saved model to {path} using model's save method")
        
        else:
            # Default: use pickle
            with open(path, "wb") as f:
                pickle.dump(model, f)
            logger.info(f"Saved model to {path} using pickle")
            
    except Exception as e:
        logger.error(f"Error saving model to {path}: {e}")
        raise

def load_model(path: str) -> Any:
    """
    Load a model from disk.
    
    Args:
        path: Path to load the model from
        
    Returns:
        The loaded model
    """
    try:
        if not os.path.exists(path):
            logger.error(f"Model file not found: {path}")
            return None
        
        # Determine file type
        if path.endswith(".npz"):
            # Load NumPy arrays
            weights = []
            with np.load(path, allow_pickle=True) as data:
                for key in data.files:
                    weights.append(data[key])
            
            # Convert to Parameters if needed
            return ndarrays_to_parameters(weights)
        
        elif path.endswith((".h5", ".keras")):
            # Load Keras model
            from tensorflow import keras
            return keras.models.load_model(path)
        
        elif path.endswith(".pt"):
            # Load PyTorch model
            import torch
            return torch.load(path)
        
        else:
            # Default: use pickle
            with open(path, "rb") as f:
                model = pickle.load(f)
            logger.info(f"Loaded model from {path}")
            return model
    
    except Exception as e:
        logger.error(f"Error loading model from {path}: {e}")
        raise

def get_model_size(model: Any) -> int:
    """
    Get the size of a model in bytes.
    
    Args:
        model: The model to get the size of (Parameters, list of arrays, or model)
        
    Returns:
        Size of the model in bytes
    """
    try:
        if isinstance(model, Parameters):
            # Handle Flower Parameters type
            weights = parameters_to_ndarrays(model)
            return sum(w.nbytes for w in weights)
        
        elif isinstance(model, list) and all(isinstance(x, np.ndarray) for x in model):
            # Handle list of NumPy arrays
            return sum(w.nbytes for w in model)
        
        else:
            # Serialize the model to get its size
            with tempfile.NamedTemporaryFile(delete=True) as tmp:
                pickle.dump(model, tmp)
                tmp.flush()
                return os.path.getsize(tmp.name)
    
    except Exception as e:
        logger.error(f"Error getting model size: {e}")
        return -1

def monitor_resources() -> Dict[str, float]:
    """
    Monitor server resources.
    
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
        memory_used = memory.used
        
        # Get disk usage
        disk = psutil.disk_usage('/')
        disk_percent = disk.percent
        disk_free = disk.free
        
        # Get network I/O stats
        net_io = psutil.net_io_counters()
        net_sent = net_io.bytes_sent
        net_recv = net_io.bytes_recv
        
        return {
            "cpu_percent": cpu_percent,
            "memory_percent": memory_percent,
            "memory_available_mb": memory_available / (1024 * 1024),
            "memory_used_mb": memory_used / (1024 * 1024),
            "disk_percent": disk_percent,
            "disk_free_gb": disk_free / (1024 * 1024 * 1024),
            "net_sent_mb": net_sent / (1024 * 1024),
            "net_recv_mb": net_recv / (1024 * 1024)
        }
    except Exception as e:
        logger.error(f"Error monitoring resources: {e}")
        return {"error": str(e)}

def initialize_metrics_directory(metrics_dir: str) -> None:
    """
    Initialize the metrics directory structure.
    
    Args:
        metrics_dir: Directory to store metrics
    """
    try:
        # Create main directory
        os.makedirs(metrics_dir, exist_ok=True)
        
        # Create subdirectories
        os.makedirs(os.path.join(metrics_dir, "rounds"), exist_ok=True)
        os.makedirs(os.path.join(metrics_dir, "models"), exist_ok=True)
        os.makedirs(os.path.join(metrics_dir, "clients"), exist_ok=True)
        os.makedirs(os.path.join(metrics_dir, "resources"), exist_ok=True)
        
        logger.info(f"Initialized metrics directory at {metrics_dir}")
    except Exception as e:
        logger.error(f"Error initializing metrics directory: {e}")
        raise

def save_round_metrics(metrics_dir: str, round_num: int, metrics: Dict[str, Any]) -> None:
    """
    Save metrics for a specific round.
    
    Args:
        metrics_dir: Directory to store metrics
        round_num: Round number
        metrics: Metrics to save
    """
    try:
        # Create rounds directory
        rounds_dir = os.path.join(metrics_dir, "rounds")
        os.makedirs(rounds_dir, exist_ok=True)
        
        # Add timestamp if not present
        if "timestamp" not in metrics:
            metrics["timestamp"] = time.time()
        
        # Save metrics
        metrics_file = os.path.join(rounds_dir, f"round_{round_num}.json")
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.debug(f"Saved round {round_num} metrics to {metrics_file}")
    except Exception as e:
        logger.error(f"Error saving round metrics: {e}")

def load_round_metrics(metrics_dir: str, round_num: int) -> Optional[Dict[str, Any]]:
    """
    Load metrics for a specific round.
    
    Args:
        metrics_dir: Directory with stored metrics
        round_num: Round number
        
    Returns:
        Dictionary with round metrics or None if not found
    """
    try:
        metrics_file = os.path.join(metrics_dir, "rounds", f"round_{round_num}.json")
        if not os.path.exists(metrics_file):
            return None
        
        with open(metrics_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading round metrics: {e}")
        return None

def save_client_metrics(metrics_dir: str, client_id: str, metrics: Dict[str, Any]) -> None:
    """
    Save metrics for a specific client.
    
    Args:
        metrics_dir: Directory to store metrics
        client_id: Client identifier
        metrics: Metrics to save
    """
    try:
        # Create clients directory
        clients_dir = os.path.join(metrics_dir, "clients")
        os.makedirs(clients_dir, exist_ok=True)
        
        # Create client directory
        client_dir = os.path.join(clients_dir, client_id)
        os.makedirs(client_dir, exist_ok=True)
        
        # Add timestamp if not present
        if "timestamp" not in metrics:
            metrics["timestamp"] = time.time()
        
        # Save metrics with timestamp in filename
        timestamp = int(metrics["timestamp"])
        metrics_file = os.path.join(client_dir, f"metrics_{timestamp}.json")
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Also update a "latest" file
        latest_file = os.path.join(client_dir, "latest.json")
        with open(latest_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.debug(f"Saved client {client_id} metrics to {metrics_file}")
    except Exception as e:
        logger.error(f"Error saving client metrics: {e}")

def get_client_history(metrics_dir: str, client_id: str) -> List[Dict[str, Any]]:
    """
    Get historical metrics for a specific client.
    
    Args:
        metrics_dir: Directory with stored metrics
        client_id: Client identifier
        
    Returns:
        List of metrics dictionaries in chronological order
    """
    try:
        client_dir = os.path.join(metrics_dir, "clients", client_id)
        if not os.path.exists(client_dir):
            return []
        
        # Get all metrics files
        metrics_files = [
            f for f in os.listdir(client_dir) 
            if f.startswith("metrics_") and f.endswith(".json")
        ]
        
        # Sort by timestamp
        metrics_files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
        
        # Load all metrics
        metrics_list = []
        for metrics_file in metrics_files:
            with open(os.path.join(client_dir, metrics_file), 'r') as f:
                metrics_list.append(json.load(f))
        
        return metrics_list
    except Exception as e:
        logger.error(f"Error getting client history: {e}")
        return []

def generate_summary_report(metrics_dir: str) -> Dict[str, Any]:
    """
    Generate a summary report of the federated learning process.
    
    Args:
        metrics_dir: Directory with stored metrics
        
    Returns:
        Dictionary with summary information
    """
    try:
        # Initialize summary
        summary = {
            "generated_at": time.time(),
            "rounds": {},
            "clients": {},
            "performance": {},
            "resources": {}
        }
        
        # Get all round metrics
        rounds_dir = os.path.join(metrics_dir, "rounds")
        if os.path.exists(rounds_dir):
            round_files = [
                f for f in os.listdir(rounds_dir) 
                if f.startswith("round_") and f.endswith(".json")
            ]
            
            # Extract round numbers and sort
            round_nums = sorted([int(f.split("_")[1].split(".")[0]) for f in round_files])
            
            # Process round metrics
            accuracy_trend = []
            loss_trend = []
            participation_trend = []
            
            for round_num in round_nums:
                metrics = load_round_metrics(metrics_dir, round_num)
                if metrics:
                    # Store round metrics
                    summary["rounds"][str(round_num)] = metrics
                    
                    # Extract key metrics for trends
                    if "accuracy" in metrics:
                        accuracy_trend.append((round_num, metrics["accuracy"]))
                    if "loss" in metrics:
                        loss_trend.append((round_num, metrics["loss"]))
                    if "num_clients" in metrics:
                        participation_trend.append((round_num, metrics["num_clients"]))
            
            # Store trends
            summary["performance"]["accuracy_trend"] = accuracy_trend
            summary["performance"]["loss_trend"] = loss_trend
            summary["performance"]["participation_trend"] = participation_trend
        
        # Get client information
        clients_dir = os.path.join(metrics_dir, "clients")
        if os.path.exists(clients_dir):
            # Get all client directories
            client_ids = [
                d for d in os.listdir(clients_dir) 
                if os.path.isdir(os.path.join(clients_dir, d))
            ]
            
            # Process client metrics
            for client_id in client_ids:
                # Get latest metrics
                latest_file = os.path.join(clients_dir, client_id, "latest.json")
                if os.path.exists(latest_file):
                    with open(latest_file, 'r') as f:
                        latest_metrics = json.load(f)
                    
                    # Store client info
                    summary["clients"][client_id] = {
                        "last_seen": latest_metrics.get("timestamp", 0),
                        "device_type": latest_metrics.get("device_type", "unknown"),
                        "last_resources": {
                            k: v for k, v in latest_metrics.items() 
                            if k in ["cpu_percent", "memory_percent", "battery_percent"]
                        }
                    }
                    
                    # Get participation history
                    history = get_client_history(metrics_dir, client_id)
                    if history:
                        summary["clients"][client_id]["participation_count"] = len(history)
                        
                        # Calculate average training time
                        training_times = [
                            h.get("training_time", 0) for h in history
                            if "training_time" in h
                        ]
                        if training_times:
                            summary["clients"][client_id]["avg_training_time"] = (
                                sum(training_times) / len(training_times)
                            )
        
        # Get resource usage information
        resources_file = os.path.join(metrics_dir, "resources", "resource_metrics.json")
        if os.path.exists(resources_file):
            with open(resources_file, 'r') as f:
                resources = json.load(f)
            
            # Get latest resource metrics
            if resources:
                summary["resources"]["latest"] = resources[-1]
                
                # Calculate average resource usage
                cpu_usage = [r.get("cpu_percent", 0) for r in resources if "cpu_percent" in r]
                memory_usage = [r.get("memory_percent", 0) for r in resources if "memory_percent" in r]
                
                if cpu_usage:
                    summary["resources"]["avg_cpu_percent"] = sum(cpu_usage) / len(cpu_usage)
                if memory_usage:
                    summary["resources"]["avg_memory_percent"] = sum(memory_usage) / len(memory_usage)
        
        return summary
    except Exception as e:
        logger.error(f"Error generating summary report: {e}")
        return {"error": str(e)}

def get_client_metrics_from_fit_res(fit_res: Any) -> Dict[str, Any]:
    """
    Extract client metrics from a FitRes object.
    
    Args:
        fit_res: FitRes object from client
        
    Returns:
        Dictionary with metrics
    """
    try:
        metrics = {}
        
        # Extract standard metrics
        if hasattr(fit_res, "metrics") and fit_res.metrics:
            metrics.update(fit_res.metrics)
        
        # Add num_examples
        if hasattr(fit_res, "num_examples"):
            metrics["num_examples"] = fit_res.num_examples
        
        return metrics
    except Exception as e:
        logger.error(f"Error extracting client metrics: {e}")
        return {}

def create_server_state_snapshot(
    server: Any, 
    metrics_dir: str,
    include_model: bool = True
) -> None:
    """
    Create a snapshot of the server state for checkpointing.
    
    Args:
        server: The server instance
        metrics_dir: Directory to save the snapshot
        include_model: Whether to include the model in the snapshot
    """
    try:
        # Create checkpoint directory
        checkpoint_dir = os.path.join(metrics_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Create timestamp
        timestamp = int(time.time())
        
        # Create state dictionary
        state = {
            "timestamp": timestamp,
            "current_round": getattr(server, "current_round", 0),
            "metrics": getattr(server, "round_metrics", []),
            "resources": getattr(server, "resource_usage", [])[-10:],  # Last 10 entries
        }
        
        # Save state
        state_file = os.path.join(checkpoint_dir, f"server_state_{timestamp}.json")
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)
        
        # Save model if requested
        if include_model and hasattr(server, "server") and hasattr(server.server, "parameters"):
            model_file = os.path.join(checkpoint_dir, f"model_{timestamp}.bin")
            save_model(server.server.parameters, model_file)
        
        # Update latest checkpoint file
        latest_file = os.path.join(checkpoint_dir, "latest_checkpoint.txt")
        with open(latest_file, 'w') as f:
            f.write(str(timestamp))
        
        logger.info(f"Created server state snapshot at {state_file}")
    
    except Exception as e:
        logger.error(f"Error creating server state snapshot: {e}")

def restore_server_state_from_snapshot(
    server: Any,
    metrics_dir: str,
    timestamp: Optional[int] = None
) -> bool:
    """
    Restore server state from a snapshot.
    
    Args:
        server: The server instance
        metrics_dir: Directory with snapshots
        timestamp: Specific snapshot timestamp to restore (if None, uses latest)
        
    Returns:
        True if restoration was successful, False otherwise
    """
    try:
        # Get checkpoint directory
        checkpoint_dir = os.path.join(metrics_dir, "checkpoints")
        if not os.path.exists(checkpoint_dir):
            logger.error(f"Checkpoint directory not found: {checkpoint_dir}")
            return False
        
        # Get snapshot timestamp
        if timestamp is None:
            # Use latest
            latest_file = os.path.join(checkpoint_dir, "latest_checkpoint.txt")
            if not os.path.exists(latest_file):
                logger.error("No latest checkpoint file found")
                return False
            
            with open(latest_file, 'r') as f:
                timestamp = int(f.read().strip())
        
        # Check if state file exists
        state_file = os.path.join(checkpoint_dir, f"server_state_{timestamp}.json")
        if not os.path.exists(state_file):
            logger.error(f"State file not found: {state_file}")
            return False
        
        # Load state
        with open(state_file, 'r') as f:
            state = json.load(f)
        
        # Restore state
        if hasattr(server, "current_round"):
            server.current_round = state.get("current_round", 0)
        
        if hasattr(server, "round_metrics"):
            server.round_metrics = state.get("metrics", [])
        
        if hasattr(server, "resource_usage"):
            server.resource_usage = state.get("resources", [])
        
        # Restore model if it exists
        model_file = os.path.join(checkpoint_dir, f"model_{timestamp}.bin")
        if os.path.exists(model_file) and hasattr(server, "server") and hasattr(server.server, "parameters"):
            model = load_model(model_file)
            if model is not None:
                server.server.parameters = model
        
        logger.info(f"Restored server state from snapshot at {state_file}")
        return True
    
    except Exception as e:
        logger.error(f"Error restoring server state: {e}")
        return False

def get_available_snapshots(metrics_dir: str) -> List[Dict[str, Any]]:
    """
    Get a list of available server state snapshots.
    
    Args:
        metrics_dir: Directory with snapshots
        
    Returns:
        List of dictionaries with snapshot information
    """
    try:
        # Get checkpoint directory
        checkpoint_dir = os.path.join(metrics_dir, "checkpoints")
        if not os.path.exists(checkpoint_dir):
            return []
        
        # Get state files
        state_files = [
            f for f in os.listdir(checkpoint_dir) 
            if f.startswith("server_state_") and f.endswith(".json")
        ]
        
        # Extract timestamps and sort
        snapshots = []
        for state_file in state_files:
            timestamp = int(state_file.split("_")[2].split(".")[0])
            
            # Get state info
            with open(os.path.join(checkpoint_dir, state_file), 'r') as f:
                state = json.load(f)
            
            # Check if model exists
            model_exists = os.path.exists(os.path.join(checkpoint_dir, f"model_{timestamp}.bin"))
            
            snapshots.append({
                "timestamp": timestamp,
                "datetime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp)),
                "round": state.get("current_round", 0),
                "model_available": model_exists,
                "metrics_count": len(state.get("metrics", []))
            })
        
        # Sort by timestamp (descending)
        snapshots.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return snapshots
    
    except Exception as e:
        logger.error(f"Error getting available snapshots: {e}")
        return []

def export_model_for_deployment(
    model: Any,
    export_dir: str,
    model_format: str = "saved_model",
    model_name: str = "fids_model"
) -> str:
    """
    Export a model for deployment.
    
    Args:
        model: The model to export
        export_dir: Directory to export the model to
        model_format: Format to export the model in ("saved_model", "tflite", "onnx")
        model_name: Name for the exported model
        
    Returns:
        Path to the exported model
    """
    try:
        # Create export directory
        os.makedirs(export_dir, exist_ok=True)
        
        # Determine model type and export accordingly
        if model_format == "saved_model":
            # For TensorFlow models
            if hasattr(model, "save"):
                export_path = os.path.join(export_dir, model_name)
                model.save(export_path)
                logger.info(f"Exported model to {export_path}")
                return export_path
            
            # For scikit-learn models
            else:
                export_path = os.path.join(export_dir, f"{model_name}.pkl")
                with open(export_path, 'wb') as f:
                    pickle.dump(model, f)
                logger.info(f"Exported model to {export_path}")
                return export_path
        
        elif model_format == "tflite":
            # For TensorFlow models
            if hasattr(model, "save"):
                import tensorflow as tf
                
                # Create a converter
                saved_model_path = os.path.join(export_dir, f"{model_name}_saved")
                model.save(saved_model_path)
                
                converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
                tflite_model = converter.convert()
                
                # Save the model
                export_path = os.path.join(export_dir, f"{model_name}.tflite")
                with open(export_path, 'wb') as f:
                    f.write(tflite_model)
                
                logger.info(f"Exported TFLite model to {export_path}")
                return export_path
            else:
                logger.error("Cannot export to TFLite: model is not a TensorFlow model")
                return ""
        
        elif model_format == "onnx":
            # For TensorFlow models
            if hasattr(model, "save"):
                import tf2onnx
                import tensorflow as tf
                
                # Create a saved model
                saved_model_path = os.path.join(export_dir, f"{model_name}_saved")
                model.save(saved_model_path)
                
                # Convert to ONNX
                export_path = os.path.join(export_dir, f"{model_name}.onnx")
                
                # Get model signature
                tf_model = tf.saved_model.load(saved_model_path)
                model_proto, _ = tf2onnx.convert.from_tensorflow(
                    tf_model, 
                    input_signature=None, 
                    opset=13, 
                    output_path=export_path
                )
                
                logger.info(f"Exported ONNX model to {export_path}")
                return export_path
            else:
                logger.error("Cannot export to ONNX: model is not a TensorFlow model")
                return ""
        
        else:
            logger.error(f"Unknown model format: {model_format}")
            return ""
    
    except Exception as e:
        logger.error(f"Error exporting model: {e}")
        return ""