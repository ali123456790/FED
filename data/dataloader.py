"""
N-BaIoT dataset loading module for FIDS.

This module handles loading, preprocessing, and partitioning of the N-BaIoT dataset
for federated learning. The N-BaIoT dataset contains network traffic data from IoT devices
with both benign and malicious samples.

The dataset is available at: https://www.kaggle.com/datasets/mkashifn/nbaiot-dataset
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union, Optional
from pathlib import Path
import glob
import re
from sklearn.preprocessing import StandardScaler, LabelEncoder

logger = logging.getLogger(__name__)

# Constants for the dataset
BENIGN_LABEL = 0
MALICIOUS_LABEL = 1

# Attack types in the dataset
ATTACK_TYPES = [
    "gafgyt_attacks/combo", 
    "gafgyt_attacks/junk", 
    "gafgyt_attacks/scan", 
    "gafgyt_attacks/tcp", 
    "gafgyt_attacks/udp", 
    "mirai_attacks/ack", 
    "mirai_attacks/scan", 
    "mirai_attacks/syn", 
    "mirai_attacks/udp", 
    "mirai_attacks/udpplain"
]

# IoT devices in the dataset
IOT_DEVICES = [
    "Danmini_Doorbell", 
    "Ecobee_Thermostat", 
    "Ennio_Doorbell", 
    "Philips_B120N10_Baby_Monitor", 
    "Provision_PT_737E_Security_Camera", 
    "Provision_PT_838_Security_Camera", 
    "Samsung_SNH_1011_N_Webcam", 
    "SimpleHome_XCS7_1002_WHT_Security_Camera", 
    "SimpleHome_XCS7_1003_WHT_Security_Camera"
]

class NBaIoTDataLoader:
    """
    Loader for the N-BaIoT dataset, handling preprocessing and partitioning
    for federated learning.
    """
    
    def __init__(self, data_path: str, cache_dir: Optional[str] = None):
        """
        Initialize the N-BaIoT data loader.
        
        Args:
            data_path: Path to the extracted N-BaIoT dataset
            cache_dir: Directory to cache preprocessed data (if None, no caching)
        """
        self.data_path = Path(data_path)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        if self.cache_dir:
            self.cache_dir.mkdir(exist_ok=True, parents=True)
            
        # Verify dataset path
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path not found: {self.data_path}")
        
        # Create a mapping of device names to client IDs
        self.device_to_client_map = {device: f"client_{i}" for i, device in enumerate(IOT_DEVICES)}
        self.client_to_device_map = {v: k for k, v in self.device_to_client_map.items()}
        
        logger.info(f"Initialized N-BaIoT data loader with path: {self.data_path}")
    
    def load_dataset(self, use_cache: bool = True) -> Dict[str, Dict]:
        """
        Load the entire N-BaIoT dataset.
        
        Args:
            use_cache: Whether to use cached data if available
            
        Returns:
            Dictionary mapping device names to their data
        """
        cache_file = self.cache_dir / "nbaiot_full_dataset.npz" if self.cache_dir else None
        
        # Try to load from cache if enabled
        if use_cache and cache_file and cache_file.exists():
            logger.info(f"Loading dataset from cache: {cache_file}")
            try:
                data = np.load(cache_file, allow_pickle=True)
                dataset = data['dataset'].item()  # Convert to dictionary
                logger.info(f"Loaded dataset from cache with {len(dataset)} devices")
                return dataset
            except Exception as e:
                logger.warning(f"Failed to load dataset from cache: {e}")
                
        # Load fresh dataset
        dataset = {}
        
        # Load data for each device
        for device in IOT_DEVICES:
            logger.info(f"Loading data for device: {device}")
            device_data = self._load_device_data(device)
            dataset[device] = device_data
            
        # Cache the dataset if enabled
        if use_cache and cache_file:
            logger.info(f"Caching dataset to: {cache_file}")
            try:
                np.savez_compressed(cache_file, dataset=dataset)
            except Exception as e:
                logger.warning(f"Failed to cache dataset: {e}")
                
        return dataset
    
    def _load_device_data(self, device: str) -> Dict:
        """
        Load data for a specific IoT device.
        
        Args:
            device: Name of the IoT device
            
        Returns:
            Dictionary with device data
        """
        device_dir = self.data_path / device
        
        if not device_dir.exists():
            raise FileNotFoundError(f"Device directory not found: {device_dir}")
            
        # Load benign data
        benign_file = next(device_dir.glob("benign_traffic.csv"), None)
        if not benign_file:
            raise FileNotFoundError(f"Benign traffic file not found for device: {device}")
            
        benign_data = pd.read_csv(benign_file)
        benign_data['label'] = BENIGN_LABEL
        
        # Initialize combined data with benign data
        all_data = benign_data.copy()
        
        # Load attack data for each attack type
        attack_data_combined = []
        
        for attack in ATTACK_TYPES:
            attack_pattern = f"*{attack.replace('/', '_')}*.csv"
            attack_files = list(device_dir.glob(attack_pattern))
            
            if not attack_files:
                logger.warning(f"No files found for attack {attack} in device {device}")
                continue
                
            # Load and combine all files for this attack type
            for attack_file in attack_files:
                try:
                    attack_data = pd.read_csv(attack_file)
                    attack_data['label'] = MALICIOUS_LABEL
                    attack_data['attack_type'] = attack
                    attack_data_combined.append(attack_data)
                except Exception as e:
                    logger.warning(f"Error loading attack file {attack_file}: {e}")
        
        # Combine all attack data
        if attack_data_combined:
            attack_df = pd.concat(attack_data_combined, ignore_index=True)
            
            # Combine benign and attack data
            all_data = pd.concat([all_data, attack_df], ignore_index=True)
            
        # Extract features and labels
        X = all_data.drop(['label', 'attack_type'] if 'attack_type' in all_data.columns else ['label'], axis=1)
        y = all_data['label'].values
        
        # Get feature names
        feature_names = X.columns.tolist()
        
        # Convert to numpy arrays
        X = X.values
        
        # Count statistics
        benign_count = np.sum(y == BENIGN_LABEL)
        malicious_count = np.sum(y == MALICIOUS_LABEL)
        
        logger.info(f"Device {device}: {X.shape[0]} samples, {X.shape[1]} features, "
                   f"{benign_count} benign, {malicious_count} malicious")
        
        return {
            "X": X,
            "y": y,
            "feature_names": feature_names,
            "benign_count": benign_count,
            "malicious_count": malicious_count,
            "device": device
        }
    
    def load_data_for_client(self, client_id: str, use_cache: bool = True) -> Dict:
        """
        Load data for a specific client.
        
        Args:
            client_id: ID of the client
            use_cache: Whether to use cached data if available
            
        Returns:
            Dictionary with client data
        """
        # Get device name from client ID
        device = self.client_to_device_map.get(client_id)
        
        if device is None:
            raise ValueError(f"Unknown client ID: {client_id}")
            
        cache_file = self.cache_dir / f"{client_id}_data.npz" if self.cache_dir else None
        
        # Try to load from cache if enabled
        if use_cache and cache_file and cache_file.exists():
            logger.info(f"Loading client data from cache: {cache_file}")
            try:
                data = np.load(cache_file, allow_pickle=True)
                client_data = {
                    "X": data['X'],
                    "y": data['y'],
                    "feature_names": data['feature_names'].tolist() if data['feature_names'].ndim > 0 else data['feature_names'].item(),
                    "benign_count": data['benign_count'].item(),
                    "malicious_count": data['malicious_count'].item(),
                    "device": data['device'].item()
                }
                logger.info(f"Loaded client data from cache with {client_data['X'].shape[0]} samples")
                return client_data
            except Exception as e:
                logger.warning(f"Failed to load client data from cache: {e}")
        
        # Load device data directly
        client_data = self._load_device_data(device)
        
        # Cache the data if enabled
        if use_cache and cache_file:
            logger.info(f"Caching client data to: {cache_file}")
            try:
                np.savez_compressed(
                    cache_file,
                    X=client_data["X"],
                    y=client_data["y"],
                    feature_names=client_data["feature_names"],
                    benign_count=client_data["benign_count"],
                    malicious_count=client_data["malicious_count"],
                    device=device
                )
            except Exception as e:
                logger.warning(f"Failed to cache client data: {e}")
        
        return client_data
    
    def create_federated_data_partition(
        self, 
        distribution_type: str = "iid",
        num_clients: Optional[int] = None,
        balance_ratio: bool = True,
        min_samples_per_client: int = 1000,
        alpha: float = 0.5,  # Dirichlet parameter for non-IID distribution
        use_real_devices: bool = True,
        seed: int = 42
    ) -> Dict[str, Dict]:
        """
        Create a federated learning data partition.
        
        Args:
            distribution_type: Type of distribution ('iid', 'non_iid_label', 'non_iid_quantity')
            num_clients: Number of clients to create (if None, use actual number of devices)
            balance_ratio: Whether to balance benign/malicious ratio across clients
            min_samples_per_client: Minimum number of samples per client
            alpha: Parameter for Dirichlet distribution in non-IID case
            use_real_devices: Whether to use real device names for client mapping
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary mapping client IDs to their data
        """
        np.random.seed(seed)
        
        # Load the full dataset
        full_dataset = self.load_dataset()
        
        # Set number of clients
        if num_clients is None:
            num_clients = len(IOT_DEVICES) if use_real_devices else 10
            
        # Create client mapping
        if use_real_devices and num_clients <= len(IOT_DEVICES):
            client_map = {f"client_{i}": device for i, device in enumerate(IOT_DEVICES[:num_clients])}
        else:
            client_map = {f"client_{i}": f"synthetic_device_{i}" for i in range(num_clients)}
        
        # Prepare data for distribution
        all_X = []
        all_y = []
        all_devices = []
        
        for device, data in full_dataset.items():
            X = data["X"]
            y = data["y"]
            
            # Add to combined dataset
            all_X.append(X)
            all_y.append(y)
            all_devices.extend([device] * len(X))
            
        # Combine all data
        X_combined = np.vstack(all_X)
        y_combined = np.concatenate(all_y)
        devices_array = np.array(all_devices)
        
        # Get feature names (assuming same features for all devices)
        feature_names = full_dataset[IOT_DEVICES[0]]["feature_names"]
        
        logger.info(f"Combined dataset: {X_combined.shape[0]} samples, {X_combined.shape[1]} features")
        
        # Distribute data to clients
        client_data = {}
        
        if distribution_type == "iid":
            # IID distribution
            logger.info(f"Creating IID distribution for {num_clients} clients")
            
            # Shuffle indices
            indices = np.random.permutation(len(X_combined))
            X_shuffled = X_combined[indices]
            y_shuffled = y_combined[indices]
            devices_shuffled = devices_array[indices]
            
            # Calculate samples per client
            samples_per_client = len(X_combined) // num_clients
            samples_per_client = max(samples_per_client, min_samples_per_client)
            
            # Distribute data
            for i, client_id in enumerate(sorted(client_map.keys())):
                start_idx = i * samples_per_client
                end_idx = min((i + 1) * samples_per_client, len(X_combined))
                
                # If we've run out of data, wrap around
                if start_idx >= len(X_combined):
                    logger.warning(f"Not enough data for client {client_id}, recycling data")
                    start_idx = start_idx % len(X_combined)
                    end_idx = min(start_idx + samples_per_client, len(X_combined))
                
                client_X = X_shuffled[start_idx:end_idx]
                client_y = y_shuffled[start_idx:end_idx]
                client_devices = devices_shuffled[start_idx:end_idx]
                
                benign_count = np.sum(client_y == BENIGN_LABEL)
                malicious_count = np.sum(client_y == MALICIOUS_LABEL)
                
                client_data[client_id] = {
                    "X": client_X,
                    "y": client_y,
                    "feature_names": feature_names,
                    "benign_count": benign_count,
                    "malicious_count": malicious_count,
                    "device": client_map[client_id],
                    "original_devices": list(set(client_devices))
                }
                
                logger.info(f"Client {client_id}: {client_X.shape[0]} samples, "
                           f"{benign_count} benign, {malicious_count} malicious")
                
        elif distribution_type == "non_iid_label":
            # Non-IID distribution with label skew
            logger.info(f"Creating non-IID label distribution for {num_clients} clients with alpha={alpha}")
            
            # Group data by label
            benign_indices = np.where(y_combined == BENIGN_LABEL)[0]
            malicious_indices = np.where(y_combined == MALICIOUS_LABEL)[0]
            
            # Shuffle indices within each label
            np.random.shuffle(benign_indices)
            np.random.shuffle(malicious_indices)
            
            # Sample Dirichlet distribution for each class
            benign_proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            malicious_proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            
            # Calculate number of samples per client for each class
            benign_samples_per_client = (benign_proportions * len(benign_indices)).astype(int)
            malicious_samples_per_client = (malicious_proportions * len(malicious_indices)).astype(int)
            
            # Adjust to ensure all samples are allocated
            benign_samples_per_client[-1] = len(benign_indices) - np.sum(benign_samples_per_client[:-1])
            malicious_samples_per_client[-1] = len(malicious_indices) - np.sum(malicious_samples_per_client[:-1])
            
            # Distribute data
            benign_start_idx = 0
            malicious_start_idx = 0
            
            for i, client_id in enumerate(sorted(client_map.keys())):
                # Get indices for this client
                benign_end_idx = benign_start_idx + benign_samples_per_client[i]
                malicious_end_idx = malicious_start_idx + malicious_samples_per_client[i]
                
                client_benign_indices = benign_indices[benign_start_idx:benign_end_idx]
                client_malicious_indices = malicious_indices[malicious_start_idx:malicious_end_idx]
                
                # Combine indices
                client_indices = np.concatenate([client_benign_indices, client_malicious_indices])
                np.random.shuffle(client_indices)  # Shuffle combined indices
                
                # Extract data
                client_X = X_combined[client_indices]
                client_y = y_combined[client_indices]
                client_devices = devices_array[client_indices]
                
                benign_count = np.sum(client_y == BENIGN_LABEL)
                malicious_count = np.sum(client_y == MALICIOUS_LABEL)
                
                client_data[client_id] = {
                    "X": client_X,
                    "y": client_y,
                    "feature_names": feature_names,
                    "benign_count": benign_count,
                    "malicious_count": malicious_count,
                    "device": client_map[client_id],
                    "original_devices": list(set(client_devices))
                }
                
                logger.info(f"Client {client_id}: {client_X.shape[0]} samples, "
                           f"{benign_count} benign ({benign_count/len(client_y)*100:.1f}%), "
                           f"{malicious_count} malicious ({malicious_count/len(client_y)*100:.1f}%)")
                
                # Update start indices
                benign_start_idx = benign_end_idx
                malicious_start_idx = malicious_end_idx
                
        elif distribution_type == "non_iid_quantity":
            # Non-IID distribution with quantity skew
            logger.info(f"Creating non-IID quantity distribution for {num_clients} clients with alpha={alpha}")
            
            # Sample from Beta distribution for client data sizes
            proportions = np.random.beta(alpha, alpha, size=num_clients)
            proportions = proportions / np.sum(proportions)
            
            # Calculate number of samples per client
            samples_per_client = (proportions * len(X_combined)).astype(int)
            
            # Adjust to ensure all samples are allocated
            samples_per_client[-1] = len(X_combined) - np.sum(samples_per_client[:-1])
            
            # Shuffle data
            indices = np.random.permutation(len(X_combined))
            X_shuffled = X_combined[indices]
            y_shuffled = y_combined[indices]
            devices_shuffled = devices_array[indices]
            
            # Distribute data
            start_idx = 0
            
            for i, client_id in enumerate(sorted(client_map.keys())):
                end_idx = start_idx + samples_per_client[i]
                
                # Ensure we don't go out of bounds
                end_idx = min(end_idx, len(X_combined))
                
                client_X = X_shuffled[start_idx:end_idx]
                client_y = y_shuffled[start_idx:end_idx]
                client_devices = devices_shuffled[start_idx:end_idx]
                
                benign_count = np.sum(client_y == BENIGN_LABEL)
                malicious_count = np.sum(client_y == MALICIOUS_LABEL)
                
                client_data[client_id] = {
                    "X": client_X,
                    "y": client_y,
                    "feature_names": feature_names,
                    "benign_count": benign_count,
                    "malicious_count": malicious_count,
                    "device": client_map[client_id],
                    "original_devices": list(set(client_devices))
                }
                
                logger.info(f"Client {client_id}: {client_X.shape[0]} samples, "
                           f"{benign_count} benign, {malicious_count} malicious")
                
                # Update start index
                start_idx = end_idx
                
        elif distribution_type == "by_device":
            # Each real device becomes a client with its own data
            logger.info("Creating distribution by device (each device is a client)")
            
            for i, device in enumerate(IOT_DEVICES):
                client_id = f"client_{i}"
                
                device_data = full_dataset[device]
                
                client_data[client_id] = {
                    "X": device_data["X"],
                    "y": device_data["y"],
                    "feature_names": device_data["feature_names"],
                    "benign_count": device_data["benign_count"],
                    "malicious_count": device_data["malicious_count"],
                    "device": device,
                    "original_devices": [device]
                }
                
                logger.info(f"Client {client_id} (device {device}): {device_data['X'].shape[0]} samples, "
                           f"{device_data['benign_count']} benign, {device_data['malicious_count']} malicious")
        
        else:
            raise ValueError(f"Unknown distribution type: {distribution_type}")
        
        return client_data

    def preprocess_client_data(
        self,
        client_data: Dict,
        normalize: bool = True,
        feature_selection: bool = False,
        num_features: int = 10,
        test_size: float = 0.2,
        validation_size: float = 0.1,
        balance_classes: bool = False,
        random_state: int = 42
    ) -> Dict:
        """
        Preprocess client data for training.
        
        Args:
            client_data: Client data dictionary
            normalize: Whether to normalize features
            feature_selection: Whether to perform feature selection
            num_features: Number of features to select if feature_selection is True
            test_size: Proportion of data to use for testing
            validation_size: Proportion of training data to use for validation
            balance_classes: Whether to balance classes in the training set
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary with preprocessed data
        """
        from sklearn.model_selection import train_test_split
        
        np.random.seed(random_state)
        
        X = client_data["X"]
        y = client_data["y"]
        feature_names = client_data["feature_names"]
        
        # Split data into train, validation, and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Further split training data to create validation set if needed
        X_val = None
        y_val = None
        
        if validation_size > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, 
                test_size=validation_size / (1 - test_size),
                random_state=random_state,
                stratify=y_train
            )
        
        # Balance classes in training set if requested
        if balance_classes:
            from sklearn.utils import resample
            
            # Find minority and majority classes
            benign_indices = np.where(y_train == BENIGN_LABEL)[0]
            malicious_indices = np.where(y_train == MALICIOUS_LABEL)[0]
            
            if len(benign_indices) < len(malicious_indices):
                minority_indices = benign_indices
                majority_indices = malicious_indices
                minority_label = BENIGN_LABEL
                majority_label = MALICIOUS_LABEL
            else:
                minority_indices = malicious_indices
                majority_indices = benign_indices
                minority_label = MALICIOUS_LABEL
                majority_label = BENIGN_LABEL
            
            # Upsample minority class
            minority_upsampled_indices = resample(
                minority_indices,
                replace=True,
                n_samples=len(majority_indices),
                random_state=random_state
            )
            
            # Combine majority class with upsampled minority class
            resampled_indices = np.concatenate([majority_indices, minority_upsampled_indices])
            np.random.shuffle(resampled_indices)
            
            X_train = X_train[resampled_indices]
            y_train = y_train[resampled_indices]
            
            logger.info(f"Balanced training set: {len(X_train)} samples, "
                       f"{np.sum(y_train == BENIGN_LABEL)} benign, "
                       f"{np.sum(y_train == MALICIOUS_LABEL)} malicious")
        
        # Normalize features
        if normalize:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            if X_val is not None:
                X_val = scaler.transform(X_val)
        
        # Perform feature selection if requested
        selected_indices = None
        
        if feature_selection and num_features > 0 and num_features < X_train.shape[1]:
            from sklearn.feature_selection import SelectKBest, f_classif
            
            selector = SelectKBest(f_classif, k=num_features)
            X_train_selected = selector.fit_transform(X_train, y_train)
            
            # Get indices of selected features
            selected_indices = selector.get_support(indices=True)
            
            # Transform test and validation sets
            X_test = X_test[:, selected_indices]
            if X_val is not None:
                X_val = X_val[:, selected_indices]
                
            # Update feature names
            selected_feature_names = [feature_names[i] for i in selected_indices]
            
            # Replace X_train with selected features
            X_train = X_train_selected
            
            logger.info(f"Selected {num_features} features: {selected_feature_names}")
        else:
            selected_feature_names = feature_names
        
        # Prepare result dictionary
        result = {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
            "selected_feature_names": selected_feature_names,
            "selected_indices": selected_indices,
            "benign_train": np.sum(y_train == BENIGN_LABEL),
            "malicious_train": np.sum(y_train == MALICIOUS_LABEL),
            "benign_test": np.sum(y_test == BENIGN_LABEL),
            "malicious_test": np.sum(y_test == MALICIOUS_LABEL)
        }
        
        if X_val is not None:
            result["X_val"] = X_val
            result["y_val"] = y_val
            result["benign_val"] = np.sum(y_val == BENIGN_LABEL)
            result["malicious_val"] = np.sum(y_val == MALICIOUS_LABEL)
        
        logger.info(f"Preprocessed data: {X_train.shape[0]} training samples, "
                   f"{X_test.shape[0]} test samples"
                   f"{f', {X_val.shape[0]} validation samples' if X_val is not None else ''}")
        
        return result
    
    def save_client_data(self, client_id: str, data: Dict, path: Optional[str] = None) -> str:
        """
        Save preprocessed client data to disk.
        
        Args:
            client_id: ID of the client
            data: Preprocessed client data
            path: Directory to save the data (if None, use cache_dir)
            
        Returns:
            Path to the saved file
        """
        save_dir = Path(path) if path else self.cache_dir
        
        if save_dir is None:
            raise ValueError("No save directory specified and no cache directory set")
            
        save_dir.mkdir(exist_ok=True, parents=True)
        
        save_path = save_dir / f"{client_id}_preprocessed.npz"
        
        # Save the data
        np.savez_compressed(
            save_path,
            X_train=data["X_train"],
            y_train=data["y_train"],
            X_test=data["X_test"],
            y_test=data["y_test"],
            X_val=data.get("X_val"),
            y_val=data.get("y_val"),
            selected_feature_names=data["selected_feature_names"],
            selected_indices=data["selected_indices"],
            benign_train=data["benign_train"],
            malicious_train=data["malicious_train"],
            benign_test=data["benign_test"],
            malicious_test=data["malicious_test"],
            benign_val=data.get("benign_val"),
            malicious_val=data.get("malicious_val")
        )
        
        logger.info(f"Saved preprocessed data for client {client_id} to {save_path}")
        
        return str(save_path)
    
    def load_preprocessed_client_data(self, client_id: str, path: Optional[str] = None) -> Dict:
        """
        Load preprocessed client data from disk.
        
        Args:
            client_id: ID of the client
            path: Directory to load the data from (if None, use cache_dir)
            
        Returns:
            Dictionary with preprocessed client data
        """
        load_dir = Path(path) if path else self.cache_dir
        
        if load_dir is None:
            raise ValueError("No load directory specified and no cache directory set")
            
        load_path = load_dir / f"{client_id}_preprocessed.npz"
        
        if not load_path.exists():
            raise FileNotFoundError(f"Preprocessed data file not found: {load_path}")
            
        # Load the data
        try:
            data = np.load(load_path, allow_pickle=True)
            
            # Create result dictionary
            result = {
                "X_train": data["X_train"],
                "y_train": data["y_train"],
                "X_test": data["X_test"],
                "y_test": data["y_test"],
                "selected_feature_names": data["selected_feature_names"].tolist() if data["selected_feature_names"].ndim > 0 else data["selected_feature_names"].item(),
                "selected_indices": data["selected_indices"].tolist() if data["selected_indices"] is not None and data["selected_indices"].ndim > 0 else data["selected_indices"].item() if data["selected_indices"] is not None else None,
                "benign_train": data["benign_train"].item(),
                "malicious_train": data["malicious_train"].item(),
                "benign_test": data["benign_test"].item(),
                "malicious_test": data["malicious_test"].item()
            }
            
            # Add validation data if available
            if "X_val" in data and data["X_val"] is not None:
                result["X_val"] = data["X_val"]
                result["y_val"] = data["y_val"]
                result["benign_val"] = data["benign_val"].item()
                result["malicious_val"] = data["malicious_val"].item()
                
            logger.info(f"Loaded preprocessed data for client {client_id}: "
                       f"{result['X_train'].shape[0]} training samples, "
                       f"{result['X_test'].shape[0]} test samples"
                       f"{f', {result[\"X_val\"].shape[0]} validation samples' if 'X_val' in result else ''}")
                
            return result
        except Exception as e:
            raise IOError(f"Error loading preprocessed data for client {client_id}: {e}")
        
    def get_client_ids(self) -> List[str]:
        """
        Get a list of all client IDs mapped to devices.
        
        Returns:
            List of client IDs
        """
        return list(self.device_to_client_map.values())
    
    def get_dataset_statistics(self) -> Dict:
        """
        Get statistics about the full dataset.
        
        Returns:
            Dictionary with dataset statistics
        """
        # Load the full dataset
        dataset = self.load_dataset()
        
        total_samples = 0
        total_benign = 0
        total_malicious = 0
        devices_stats = {}
        
        for device, data in dataset.items():
            benign_count = data["benign_count"]
            malicious_count = data["malicious_count"]
            total_samples += len(data["X"])
            total_benign += benign_count
            total_malicious += malicious_count
            
            devices_stats[device] = {
                "samples": len(data["X"]),
                "benign": benign_count,
                "malicious": malicious_count,
                "benign_ratio": benign_count / len(data["X"]),
                "malicious_ratio": malicious_count / len(data["X"])
            }
        
        return {
            "total_samples": total_samples,
            "total_benign": total_benign,
            "total_malicious": total_malicious,
            "benign_ratio": total_benign / total_samples,
            "malicious_ratio": total_malicious / total_samples,
            "num_devices": len(dataset),
            "devices": devices_stats,
            "feature_count": dataset[IOT_DEVICES[0]]["X"].shape[1],
            "feature_names": dataset[IOT_DEVICES[0]]["feature_names"]
        }