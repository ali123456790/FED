"""
Device heterogeneity handling for federated learning clients.
"""

import logging
import platform
from typing import Dict, Optional, List, Any, Tuple
import psutil
import os
import json
import time
import numpy as np
from pathlib import Path
import subprocess
import threading
import hmac
import hashlib
import uuid

# Import new modules for masking and serialization
from security.masking import MaskingMechanism
from models.proto_serialization import ProtoSerializer
from security.differential_privacy import DifferentialPrivacy

# Import new modules for Paillier encryption
from security.encoding import FixedPointEncoder

logger = logging.getLogger(__name__)

class DeviceManager:
    """Manage device heterogeneity for federated learning clients."""
    
    def __init__(self, client_id: str, heterogeneity_enabled: bool = True, 
                 resource_monitoring: bool = True, monitor_interval: int = 60,
                 resource_history_size: int = 24, config: Optional[Dict] = None,
                 security_manager: Optional[Any] = None):
        """
        Initialize the device manager.
        
        Args:
            client_id: Client identifier
            heterogeneity_enabled: Whether to adapt to device heterogeneity
            resource_monitoring: Whether to monitor device resources
            monitor_interval: Interval for resource monitoring in seconds
            resource_history_size: Number of resource records to keep
            config: Configuration dictionary with device settings
            security_manager: Security manager instance (optional)
        """
        self.client_id = client_id
        self.device_id = self._generate_device_id()
        self.device_info = self._get_device_info()
        self.device_category = self._categorize_device()
        
        # Random HMAC key for client authentication
        self._hmac_key = os.urandom(32)
        
        # If security manager is provided, get the Client-Edge key (K_CE) from it
        if security_manager:
            try:
                self._hmac_key = security_manager.get_client_edge_key()
                logger.info("Using Client-Edge key (K_CE) from security manager for HMAC")
            except Exception as e:
                logger.warning(f"Unable to get Client-Edge key from security manager: {e}")
                logger.info("Using random HMAC key as fallback")
        
        # Configuration
        self.config = config or {}
        self.heterogeneity_enabled = heterogeneity_enabled
        self.resource_monitoring = resource_monitoring
        self.monitor_interval = monitor_interval
        
        # Set up paths
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.profile_dir = os.path.join(base_dir, "device_profiles")
        os.makedirs(self.profile_dir, exist_ok=True)
        self.profile_path = os.path.join(self.profile_dir, f"{client_id}.json")
        
        # Load or create device profile
        self.device_profile = self._load_or_create_profile()
        
        # Initialize security components
        self.masking = None
        if config and "security" in config:
            from security.masking import MaskingMechanism
            self.masking = MaskingMechanism(config)
            
            # Initialize encryption
            from security.encryption import Encryption
            self.encryption = Encryption(config)
            
            # Get server's public key for encryption if available
            self.server_public_key = None
            if self.encryption.enabled and hasattr(self.encryption, "_public_key"):
                # For development, use the encryption class's public key
                # In production, this would be fetched securely from the server
                self.server_public_key = self.encryption._public_key
        else:
            self.encryption = None
            self.server_public_key = None
        
        # Initialize serializer
        from models.proto_serialization import ProtoSerializer
        if security_manager:
            try:
                self.serializer = ProtoSerializer(
                    hmac_key=self._hmac_key,
                    client_edge_key=security_manager.get_client_edge_key()
                )
                logger.info("Initialized Protocol Buffer serializer with Client-Edge key")
            except Exception as e:
                logger.warning(f"Error initializing ProtoSerializer with Client-Edge key: {e}")
                self.serializer = ProtoSerializer(hmac_key=self._hmac_key)
        else:
            self.serializer = ProtoSerializer(hmac_key=self._hmac_key)
        
        # Track current round and updates
        self.current_round_id = 0
        self.current_updates = {}
        
        # Start resource monitoring if enabled
        self.monitoring_thread = None
        self.resource_history = []
        self.resource_history_size = resource_history_size
        
        if self.resource_monitoring:
            self._start_background_monitoring()
        
        logger.info(f"Device manager initialized for client {client_id} ({self.device_category})")
    
    def __del__(self):
        """Clean up resources when the object is destroyed."""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.stop_monitoring.set()
            self.monitoring_thread.join(timeout=2.0)
    
    def _generate_device_id(self) -> str:
        """
        Generate a unique hardware identifier for this device.
        
        Returns:
            Device hardware identifier
        """
        # Get unique hardware information
        try:
            # Collect hardware identifiers that are stable across reboots
            identifiers = []
            
            # Add hostname
            identifiers.append(platform.node())
            
            # Add CPU info
            if hasattr(platform, 'processor'):
                identifiers.append(platform.processor())
            
            # Add machine type
            identifiers.append(platform.machine())
            
            # Try to get disk serial number or disk id on different platforms
            try:
                if platform.system() == 'Linux':
                    # Use the first disk's UUID
                    disk_info = subprocess.check_output(['lsblk', '-d', '-n', '-o', 'UUID'])
                    if disk_info:
                        identifiers.append(disk_info.decode().strip())
                elif platform.system() == 'Darwin':  # macOS
                    # Use disk serial
                    disk_info = subprocess.check_output(['diskutil', 'info', 'disk0'])
                    for line in disk_info.decode().split('\n'):
                        if 'Disk / Partition UUID' in line:
                            identifiers.append(line.split(':')[1].strip())
                            break
                elif platform.system() == 'Windows':
                    # Use disk serial
                    disk_info = subprocess.check_output(['wmic', 'diskdrive', 'get', 'SerialNumber'])
                    identifiers.append(disk_info.decode().strip())
            except Exception as e:
                logger.debug(f"Could not get disk identifier: {e}")
            
            # Create a string of all identifiers
            id_string = "".join(identifiers)
            
            # Generate a hash of the identifiers
            device_hash = hashlib.sha256(id_string.encode()).hexdigest()
            
            # Use the first 16 characters of the hash
            return device_hash[:16]
            
        except Exception as e:
            logger.warning(f"Could not generate device ID from hardware: {e}")
            # Fallback to a random UUID stored in a file
            id_file = Path(".device_id")
            if id_file.exists():
                return id_file.read_text().strip()
            else:
                device_id = str(uuid.uuid4())
                id_file.write_text(device_id)
                return device_id
    
    def _get_device_info(self) -> Dict:
        """
        Get detailed information about the device.
        
        Returns:
            Dictionary with device information
        """
        device_info = {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": psutil.cpu_count(logical=True),
            "cpu_physical_count": psutil.cpu_count(logical=False) or 1,  # Default to 1 if None
            "memory_total": psutil.virtual_memory().total,
            "platform": platform.platform(),
            "python_version": platform.python_version(),
        }
        
        # Get CPU frequency if available
        try:
            cpu_freq = psutil.cpu_freq()
            if cpu_freq:
                device_info["cpu_freq_max"] = cpu_freq.max
                device_info["cpu_freq_min"] = cpu_freq.min
                device_info["cpu_freq_current"] = cpu_freq.current
        except Exception as e:
            logger.debug(f"Could not get CPU frequency: {e}")
        
        # Try to get GPU information
        device_info["gpu_available"] = False
        device_info["gpu_info"] = []
        
        # Check for NVIDIA GPUs
        try:
            if os.path.exists("/usr/bin/nvidia-smi"):
                gpu_info = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader"],
                    universal_newlines=True
                )
                if gpu_info:
                    device_info["gpu_available"] = True
                    device_info["gpu_info"] = [line.strip() for line in gpu_info.strip().split('\n')]
        except Exception as e:
            logger.debug(f"Could not get NVIDIA GPU info: {e}")
        
        # Check for TensorFlow GPU support
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                device_info["tensorflow_gpu"] = True
                device_info["tensorflow_gpu_count"] = len(gpus)
                device_info["gpu_available"] = True
            else:
                device_info["tensorflow_gpu"] = False
                device_info["tensorflow_gpu_count"] = 0
        except Exception as e:
            logger.debug(f"Could not get TensorFlow GPU info: {e}")
            device_info["tensorflow_gpu"] = False
            device_info["tensorflow_gpu_count"] = 0
        
        # Check for PyTorch GPU support
        try:
            import torch
            device_info["pytorch_gpu"] = torch.cuda.is_available()
            device_info["pytorch_gpu_count"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
            if torch.cuda.is_available():
                device_info["gpu_available"] = True
                device_info["pytorch_gpu_name"] = torch.cuda.get_device_name(0)
        except Exception as e:
            logger.debug(f"Could not get PyTorch GPU info: {e}")
            device_info["pytorch_gpu"] = False
            device_info["pytorch_gpu_count"] = 0
        
        # Add network information
        try:
            net_io = psutil.net_io_counters()
            device_info["net_bytes_sent"] = net_io.bytes_sent
            device_info["net_bytes_recv"] = net_io.bytes_recv
        except Exception as e:
            logger.debug(f"Could not get network info: {e}")
        
        # Add disk information
        try:
            disk = psutil.disk_usage('/')
            device_info["disk_total"] = disk.total
            device_info["disk_free"] = disk.free
            device_info["disk_percent"] = disk.percent
        except Exception as e:
            logger.debug(f"Could not get disk info: {e}")
        
        return device_info
    
    def _categorize_device(self) -> str:
        """
        Categorize the device based on its capabilities.
        
        Returns:
            Device category as a string
        """
        cpu_count = self.device_info.get("cpu_physical_count", 1)
        memory_gb = self.device_info.get("memory_total", 0) / (1024**3)
        gpu_available = self.device_info.get("gpu_available", False)
        
        # Categorize based on available resources
        if gpu_available:
            if memory_gb >= 8:
                return "high_end_gpu"
            else:
                return "mid_range_gpu"
        elif cpu_count >= 8 and memory_gb >= 16:
            return "high_end_cpu"
        elif cpu_count >= 4 and memory_gb >= 8:
            return "mid_range_cpu"
        elif memory_gb >= 4:
            return "low_end"
        else:
            return "constrained"
    
    def _load_or_create_profile(self) -> Dict:
        """
        Load existing device profile or create a new one.
        
        Returns:
            Device profile dictionary
        """
        if self.profile_path.exists():
            try:
                with open(self.profile_path, 'r') as f:
                    profile = json.load(f)
                
                # Update some fields in the profile
                profile["last_seen"] = time.time()
                profile["device_info"] = self.device_info
                profile["device_category"] = self.device_category
                
                # Save updated profile
                with open(self.profile_path, 'w') as f:
                    json.dump(profile, f, indent=2)
                
                logger.info(f"Loaded existing profile for client {self.client_id}")
                return profile
            except Exception as e:
                logger.warning(f"Could not load profile for client {self.client_id}: {e}")
        
        # Create new profile
        profile = {
            "client_id": self.client_id,
            "device_info": self.device_info,
            "device_category": self.device_category,
            "created_at": time.time(),
            "last_seen": time.time(),
            "training_count": 0,
            "average_training_time": 0,
            "max_batch_size": self._estimate_max_batch_size(),
            "performance_benchmark": self._run_performance_benchmark(),
            "resource_history": []
        }
        
        # Save profile
        with open(self.profile_path, 'w') as f:
            json.dump(profile, f, indent=2)
        
        logger.info(f"Created new profile for client {self.client_id}")
        return profile
    
    def _estimate_max_batch_size(self) -> int:
        """
        Estimate maximum batch size based on device memory.
        
        Returns:
            Estimated maximum batch size
        """
        memory_gb = self.device_info.get("memory_total", 0) / (1024**3)
        
        # Rough heuristic for batch size based on memory
        # Assuming each sample in a batch needs about 100KB for processing
        # and we want to use at most 50% of the memory
        if memory_gb <= 1:
            return 16
        elif memory_gb <= 4:
            return 32
        elif memory_gb <= 8:
            return 64
        elif memory_gb <= 16:
            return 128
        else:
            return 256
    
    def _run_performance_benchmark(self) -> Dict:
        """
        Run a simple performance benchmark to measure device capabilities.
        
        Returns:
            Dictionary with benchmark results
        """
        import time
        import numpy as np
        
        benchmark_results = {}
        
        # CPU benchmark: matrix multiplication
        try:
            start_time = time.time()
            size = 1000
            matrix_a = np.random.random((size, size))
            matrix_b = np.random.random((size, size))
            _ = np.dot(matrix_a, matrix_b)
            cpu_time = time.time() - start_time
            benchmark_results["matrix_mul_1000x1000"] = cpu_time
        except Exception as e:
            logger.warning(f"CPU benchmark failed: {e}")
            benchmark_results["matrix_mul_1000x1000"] = -1
        
        # Memory benchmark: allocation and copy
        try:
            start_time = time.time()
            size = 1000000  # 1M elements
            arr_a = np.random.random(size)
            arr_b = np.copy(arr_a)
            memory_time = time.time() - start_time
            benchmark_results["memory_copy_1M"] = memory_time
        except Exception as e:
            logger.warning(f"Memory benchmark failed: {e}")
            benchmark_results["memory_copy_1M"] = -1
        
        # GPU benchmark if available
        if self.device_info.get("gpu_available", False):
            try:
                # Try TensorFlow first
                if self.device_info.get("tensorflow_gpu", False):
                    import tensorflow as tf
                    with tf.device('/GPU:0'):
                        start_time = time.time()
                        a = tf.random.normal([1000, 1000])
                        b = tf.random.normal([1000, 1000])
                        c = tf.matmul(a, b)
                        # Force execution
                        _ = c.numpy()
                        gpu_time = time.time() - start_time
                        benchmark_results["gpu_tf_matmul_1000x1000"] = gpu_time
                # Try PyTorch if TensorFlow not available
                elif self.device_info.get("pytorch_gpu", False):
                    import torch
                    if torch.cuda.is_available():
                        start_time = time.time()
                        a = torch.randn(1000, 1000, device="cuda")
                        b = torch.randn(1000, 1000, device="cuda")
                        c = torch.matmul(a, b)
                        # Force execution
                        _ = c.cpu()
                        gpu_time = time.time() - start_time
                        benchmark_results["gpu_torch_matmul_1000x1000"] = gpu_time
            except Exception as e:
                logger.warning(f"GPU benchmark failed: {e}")
                benchmark_results["gpu_matmul_1000x1000"] = -1
        
        return benchmark_results
    
    def check_resources(self) -> Dict:
        """
        Check available resources on the device.
        
        Returns:
            Dictionary with resource information
        """
        if not self.resource_monitoring:
            return {"status": "monitoring_disabled"}
        
        # Get CPU usage
        try:
            cpu_percent = psutil.cpu_percent(interval=0.5)
        except Exception as e:
            logger.warning(f"Error getting CPU usage: {e}")
            cpu_percent = -1
        
        # Get memory usage
        try:
            memory = psutil.virtual_memory()
            memory_available = memory.available
            memory_percent = memory.percent
        except Exception as e:
            logger.warning(f"Error getting memory usage: {e}")
            memory_available = -1
            memory_percent = -1
        
        # Get disk usage
        try:
            disk = psutil.disk_usage('/')
            disk_free = disk.free
            disk_percent = disk.percent
        except Exception as e:
            logger.warning(f"Error getting disk usage: {e}")
            disk_free = -1
            disk_percent = -1
        
        # Get battery info if available
        battery_percent = -1
        battery_power_plugged = True
        
        try:
            if hasattr(psutil, "sensors_battery"):
                battery = psutil.sensors_battery()
                if battery:
                    battery_percent = battery.percent
                    battery_power_plugged = battery.power_plugged
        except Exception as e:
            logger.warning(f"Error getting battery info: {e}")
        
        # Get network activity
        net_sent, net_recv = 0, 0
        try:
            net_io = psutil.net_io_counters()
            net_sent = net_io.bytes_sent
            net_recv = net_io.bytes_recv
        except Exception as e:
            logger.warning(f"Error getting network activity: {e}")
        
        # Compile resource information
        resources = {
            "timestamp": time.time(),
            "cpu_percent": cpu_percent,
            "memory_available": memory_available,
            "memory_percent": memory_percent,
            "disk_free": disk_free,
            "disk_percent": disk_percent,
            "battery_percent": battery_percent,
            "battery_power_plugged": battery_power_plugged,
            "net_sent": net_sent,
            "net_recv": net_recv
        }
        
        # Add to history
        self.resource_history.append(resources)
        
        # Keep history size limited
        if len(self.resource_history) > self.resource_history_size:
            self.resource_history = self.resource_history[-self.resource_history_size:]
        
        # Update profile if it exists
        if hasattr(self, 'device_profile') and self.device_profile:
            self.device_profile["resource_history"] = self.resource_history[-5:]  # Keep last 5 records in profile
            self.device_profile["last_resource_check"] = resources
            
            try:
                with open(self.profile_path, 'w') as f:
                    json.dump(self.device_profile, f, indent=2)
            except Exception as e:
                logger.debug(f"Could not update device profile: {e}")
        
        return resources
    
    def _start_background_monitoring(self) -> None:
        """Start background resource monitoring thread."""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            return  # Already running
        
        self.stop_monitoring.clear()
        self.monitoring_thread = threading.Thread(
            target=self._background_monitoring_task,
            daemon=True
        )
        self.monitoring_thread.start()
        logger.debug("Started background resource monitoring")
    
    def _background_monitoring_task(self) -> None:
        """Background task for resource monitoring."""
        while not self.stop_monitoring.is_set():
            try:
                self.check_resources()
            except Exception as e:
                logger.error(f"Error in background resource monitoring: {e}")
            
            # Sleep for the specified interval
            for _ in range(self.monitor_interval):
                if self.stop_monitoring.is_set():
                    break
                time.sleep(1)  # Check for stop signal every second
    
    def can_train(self, resources: Optional[Dict] = None) -> bool:
        """
        Determine if the device can train based on available resources.
        
        Args:
            resources: Resource information (if None, will be fetched)
            
        Returns:
            True if the device can train, False otherwise
        """
        if not self.heterogeneity_enabled:
            return True
        
        if resources is None:
            resources = self.check_resources()
        
        # If resource monitoring is disabled, allow training
        if resources.get("status") == "monitoring_disabled":
            return True
        
        # Define thresholds for training
        cpu_threshold = 90     # CPU usage below 90%
        memory_threshold = 90  # Memory usage below 90%
        disk_threshold = 95    # Disk usage below 95%
        battery_threshold = 20 # Battery above 20% if not plugged in
        
        # Get latest resource values
        cpu_percent = resources.get("cpu_percent", 0)
        memory_percent = resources.get("memory_percent", 0)
        disk_percent = resources.get("disk_percent", 0)
        battery_percent = resources.get("battery_percent", 100)
        battery_plugged = resources.get("battery_power_plugged", True)
        
        # Flag to indicate if detail logs were written
        detail_logged = False
        
        # Check CPU usage
        if cpu_percent > cpu_threshold:
            logger.warning(f"CPU usage too high for training: {cpu_percent}% > {cpu_threshold}%")
            detail_logged = True
            return False
        
        # Check memory usage
        if memory_percent > memory_threshold:
            logger.warning(f"Memory usage too high for training: {memory_percent}% > {memory_threshold}%")
            detail_logged = True
            return False
        
        # Check disk usage
        if disk_percent > disk_threshold:
            logger.warning(f"Disk usage too high for training: {disk_percent}% > {disk_threshold}%")
            detail_logged = True
            return False
        
        # Check battery if available and not plugged in
        if battery_percent != -1 and not battery_plugged:
            if battery_percent < battery_threshold:
                logger.warning(f"Battery too low for training: {battery_percent}% < {battery_threshold}%")
                detail_logged = True
                return False
        
        # Log resource status if none of the above triggered
        if not detail_logged:
            logger.info(f"Device can train: CPU {cpu_percent}%, Memory {memory_percent}%, Disk {disk_percent}%{', Battery ' + str(battery_percent) + '%' if battery_percent != -1 else ''}")
        
        return True
    
    def adjust_training_parameters(self, config: Dict) -> Dict:
        """
        Adjust training parameters based on device capabilities.
        
        Args:
            config: Training configuration
            
        Returns:
            Adjusted training configuration
        """
        if not self.heterogeneity_enabled:
            return config
        
        # Make a copy of the config
        adjusted_config = config.copy()
        
        # Get device resources
        resources = self.check_resources()
        
        # Get device category
        category = self.device_category
        
        # Adjust batch size based on device category and available memory
        memory_available_gb = resources.get("memory_available", 0) / (1024 ** 3)
        
        # Start with category-based batch size
        if category == "constrained":
            adjusted_config["batch_size"] = min(adjusted_config.get("batch_size", 32), 8)
            adjusted_config["local_epochs"] = min(adjusted_config.get("local_epochs", 5), 1)
        elif category == "low_end":
            adjusted_config["batch_size"] = min(adjusted_config.get("batch_size", 32), 16)
            adjusted_config["local_epochs"] = min(adjusted_config.get("local_epochs", 5), 2)
        elif category == "mid_range_cpu":
            adjusted_config["batch_size"] = min(adjusted_config.get("batch_size", 32), 32)
            # Keep default epochs
        elif category == "high_end_cpu" or category == "mid_range_gpu":
            adjusted_config["batch_size"] = min(adjusted_config.get("batch_size", 32), 64)
            # Can potentially increase epochs
            adjusted_config["local_epochs"] = adjusted_config.get("local_epochs", 5)
        elif category == "high_end_gpu":
            # Can use larger batch size
            adjusted_config["batch_size"] = min(adjusted_config.get("batch_size", 32), 128)
            # Can potentially increase epochs
            adjusted_config["local_epochs"] = adjusted_config.get("local_epochs", 5)
        
        # Further adjust based on current memory availability
        if memory_available_gb < 0.5:  # Less than 512MB available
            adjusted_config["batch_size"] = min(adjusted_config["batch_size"], 4)
        elif memory_available_gb < 1.0:  # Less than 1GB available
            adjusted_config["batch_size"] = min(adjusted_config["batch_size"], 8)
        
        # Adjust epochs based on CPU and battery
        cpu_percent = resources.get("cpu_percent", 0)
        battery_percent = resources.get("battery_percent", 100)
        battery_plugged = resources.get("battery_power_plugged", True)
        
        if cpu_percent > 70 or (not battery_plugged and battery_percent < 50):
            adjusted_config["local_epochs"] = max(1, adjusted_config.get("local_epochs", 5) // 2)
        
        # Adjust learning rate based on batch size changes
        original_batch_size = config.get("batch_size", 32)
        new_batch_size = adjusted_config.get("batch_size", 32)
        
        if new_batch_size != original_batch_size:
            # Scale learning rate linearly with batch size
            original_lr = config.get("learning_rate", 0.001)
            adjusted_config["learning_rate"] = original_lr * (new_batch_size / original_batch_size)
        
        # Log adjustments
        if adjusted_config != config:
            logger.info(f"Adjusted training parameters for {category} device: {adjusted_config}")
        
        return adjusted_config
    
    def update_training_stats(self, training_time: float, batch_size: int, epochs: int) -> None:
        """
        Update training statistics in the device profile.
        
        Args:
            training_time: Time taken for training in seconds
            batch_size: Batch size used
            epochs: Number of epochs
        """
        if not hasattr(self, 'device_profile') or not self.device_profile:
            return
        
        try:
            # Update profile
            self.device_profile["training_count"] += 1
            
            # Update average training time (moving average)
            count = self.device_profile["training_count"]
            old_avg = self.device_profile["average_training_time"]
            
            if count == 1:
                self.device_profile["average_training_time"] = training_time
            else:
                self.device_profile["average_training_time"] = old_avg * (count - 1) / count + training_time / count
            
            # Add training record
            if "training_history" not in self.device_profile:
                self.device_profile["training_history"] = []
            
            training_record = {
                "timestamp": time.time(),
                "training_time": training_time,
                "batch_size": batch_size,
                "epochs": epochs,
                "resources": self.check_resources()
            }
            
            self.device_profile["training_history"].append(training_record)
            
            # Keep history limited
            if len(self.device_profile["training_history"]) > 10:  # Keep last 10 records
                self.device_profile["training_history"] = self.device_profile["training_history"][-10:]
            
            # Save updated profile
            with open(self.profile_path, 'w') as f:
                json.dump(self.device_profile, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Error updating training stats: {e}")
    
    def get_device_report(self) -> Dict:
        """
        Generate a comprehensive device report.
        
        Returns:
            Dictionary with device report
        """
        # Get current resources
        resources = self.check_resources()
        
        # Prepare report
        report = {
            "client_id": self.client_id,
            "device_category": self.device_category,
            "device_info": self.device_info,
            "current_resources": resources,
            "can_train": self.can_train(resources),
            "training_stats": {
                "count": self.device_profile.get("training_count", 0),
                "average_time": self.device_profile.get("average_training_time", 0)
            },
            "performance_benchmark": self.device_profile.get("performance_benchmark", {}),
            "recommended_config": self.adjust_training_parameters({
                "batch_size": 32,
                "local_epochs": 5,
                "learning_rate": 0.001
            })
        }
        
        return report
    
    def process_model_updates(self, weights: List[np.ndarray], 
                             round_id: int, 
                             metrics: Dict[str, float],
                             training_time: float,
                             samples: int) -> bytes:
        """
        Process model updates before sending to server.
        
        This method applies:
        1. Differential privacy (optional)
        2. Fixed-point encoding and Paillier encryption (if enabled)
        3. Masking (optional)
        4. HMAC authentication
        
        Args:
            weights: Model weights after training
            round_id: Current round ID
            metrics: Training metrics
            training_time: Training time in seconds
            samples: Number of training samples
            
        Returns:
            Serialized bytes ready to be sent to server
        """
        logger.info(f"Processing updates for client {self.client_id} in round {round_id}")
        
        # Store current round ID
        self.current_round_id = round_id
        
        # Store original weights for later comparison if needed
        original_weights = [w.copy() for w in weights]
        processed_weights = weights
        
        # 1. Apply differential privacy if enabled
        if self.config and "security" in self.config and self.config["security"].get("differential_privacy", {}).get("enabled", False):
            try:
                from security.differential_privacy import DifferentialPrivacy
                dp = DifferentialPrivacy(self.config)
                processed_weights = dp.apply_noise(processed_weights, samples)
                logger.info(f"Applied differential privacy to model updates (samples: {samples})")
                metrics["dp_applied"] = True
            except Exception as e:
                logger.error(f"Error applying differential privacy: {e}")
                metrics["dp_applied"] = False
        
        # 2. Check if Paillier encryption is enabled
        use_paillier = (
            self.config and 
            "security" in self.config and 
            self.config["security"].get("encryption", {}).get("enabled", False) and
            self.config["security"].get("encryption", {}).get("type", "") == "paillier"
        )
        
        paillier_ciphertexts_list = []
        if use_paillier:
            try:
                # First encode the weights to fixed-point integers
                logger.info("Encoding weights to fixed-point integers for Paillier encryption")
                precision_bits = self.config["security"]["encryption"].get("paillier", {}).get("encoding_precision_bits", 64)
                encoder = FixedPointEncoder(precision_bits=precision_bits)
                encoded_weights = encoder.encode_weights(processed_weights)
                
                # Then encrypt using the server's public key
                logger.info("Applying Paillier encryption to encoded model updates")
                
                # Ensure the server's public key is available
                if not hasattr(self.encryption, "_paillier_public_key") or self.encryption._paillier_public_key is None:
                    logger.warning("Server's Paillier public key not available. Requesting key from server.")
                    # In a real implementation, we would request the key here
                    raise ValueError("Paillier encryption enabled but public key not available")
                
                # Encrypt the weights
                encrypted_weights = self.encryption.encrypt_weights_paillier(encoded_weights)
                processed_weights = encrypted_weights
                
                # Flatten and convert to string representation for protocol buffer
                for layer in encrypted_weights:
                    flat_layer = layer.flatten()
                    for value in flat_layer:
                        # Convert each encrypted value to string representation
                        ciphertext_str = str(value.ciphertext(be_secure=False))
                        paillier_ciphertexts_list.append(ciphertext_str)
                
                logger.info(f"Converted {len(paillier_ciphertexts_list)} encrypted values to string representation")
                
                # Use the new serialization method for PHE
                metrics["is_encrypted"] = True
                metrics["encryption_type"] = "paillier"
                metrics["precision_bits"] = precision_bits
                
                # Create metadata for the serialized update
                metadata = {
                    "client_id": self.client_id,
                    "device_id": self.device_id,
                    "device_category": self.device_category,
                    "training_time": training_time,
                    **metrics
                }
                
                # Use the new serialization method specifically for PHE
                return self.serializer.serialize_client_update_phe(
                    client_id=self.client_id,
                    round_id=round_id,
                    num_examples=samples,
                    paillier_ciphertexts_list=paillier_ciphertexts_list,
                    device_type=self.device_category,
                    training_time=training_time,
                    metadata=metadata
                )
                
            except Exception as e:
                logger.error(f"Error applying Paillier encryption: {e}")
                metrics["encryption_error"] = str(e)
        
        # 3. Apply masking if enabled and not Paillier encrypted
        # Paillier encryption provides stronger privacy than masking
        elif self.masking and self.masking.enabled and not use_paillier:
            try:
                processed_weights = self.masking.apply_mask(processed_weights)
                logger.info(f"Applied {self.masking.mask_type} masking to model updates")
                metrics["masked"] = True
                metrics["mask_type"] = self.masking.mask_type
            except Exception as e:
                logger.error(f"Error applying mask: {e}")
                metrics["masked"] = False
        
        # Store current updates
        self.current_updates = {
            "round_id": round_id,
            "original": original_weights,
            "processed": processed_weights,
            "metrics": metrics
        }
        
        # 4. Generate HMAC for integrity verification
        try:
            hmac_digest = self._generate_hmac(processed_weights, round_id)
            metadata = {
                "client_id": self.client_id,
                "round_id": round_id,
                "timestamp": time.time(),
                "hmac_digest": hmac_digest,
                "hmac_key_id": self.client_id,  # In practice, this would be a key ID
                "device_id": self.device_id,
                "device_category": self.device_category,
                **metrics
            }
            logger.info(f"Generated HMAC for model updates (round {round_id})")
        except Exception as e:
            logger.error(f"Error generating HMAC: {e}")
            metadata = {
                "client_id": self.client_id,
                "round_id": round_id,
                "timestamp": time.time(),
                "device_id": self.device_id,
                "device_category": self.device_category,
                **metrics
            }
        
        # 5. Serialize updates (using the standard method for non-PHE updates)
        try:
            # Use protocol buffer serialization if available
            serialized_update = self.serializer.serialize_model_weights(
                processed_weights, self.client_id, round_id, metrics, training_time, samples
            )
            logger.info(f"Serialized model updates ({len(serialized_update)} bytes)")
            return serialized_update
        except Exception as e:
            logger.error(f"Error serializing update: {e}, falling back to pickle")
            # Fall back to pickle
            import pickle
            return pickle.dumps({
                "weights": processed_weights,
                "metadata": metadata
            })
    
    def _generate_hmac(self, weights: List[np.ndarray], round_id: int) -> bytes:
        """
        Generate HMAC for model updates.
        Uses the Client-Edge key (K_CE) for HMAC authentication.
        
        Args:
            weights: Model weight updates
            round_id: Current round number
            
        Returns:
            HMAC digest
        """
        h = hmac.new(self._hmac_key, digestmod=hashlib.sha256)
        
        # Add client_id and round_id to HMAC
        h.update(self.client_id.encode())
        h.update(str(round_id).encode())
        
        # Add model weights to HMAC
        for layer in weights:
            h.update(layer.tobytes())
        
        return h.digest()
    
    def create_identity_message(self, round_id: int, public_key: bytes = b'') -> bytes:
        """
        Create identity message for authentication.
        
        Args:
            round_id: Federation round
            public_key: Optional public key for encryption
            
        Returns:
            Serialized identity message
        """
        # In a real implementation, we would create a proper signature
        # For now, we'll just use HMAC as a simplified signature
        signature = hmac.new(
            self._hmac_key, 
            msg=f"{self.client_id}{self.device_id}{round_id}".encode(),
            digestmod=hashlib.sha256
        ).digest()
        
        # Create and serialize identity message
        identity_message = self.serializer.create_client_identity(
            self.client_id,
            self.device_id,
            round_id,
            public_key,
            signature
        )
        
        return identity_message