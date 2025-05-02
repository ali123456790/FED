"""
NSL-KDD dataset loading module for FIDS.

This module handles loading, preprocessing, and partitioning of the NSL-KDD dataset
for federated learning. The NSL-KDD dataset contains network traffic data with both
benign and attack samples for intrusion detection systems.

The dataset is available at: https://www.unb.ca/cic/datasets/nsl.html
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union, Optional
from pathlib import Path
import glob
import re
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

from .nslkdd_constants import (
    BENIGN_LABEL,
    MALICIOUS_LABEL,
    ATTACK_CLASSES,
    ATTACK_TYPES,
    FEATURE_NAMES,
    PROTOCOL_TYPES,
    SERVICE_TYPES,
    FLAG_TYPES
)

logger = logging.getLogger(__name__)

class NSLKDDDataLoader:
    """
    Loader for the NSL-KDD dataset, handling preprocessing and partitioning
    for federated learning.
    """
    
    def __init__(self, data_path: str, cache_dir: Optional[str] = None):
        """
        Initialize the NSL-KDD data loader.
        
        Args:
            data_path: Path to the extracted NSL-KDD dataset
            cache_dir: Directory to cache preprocessed data (if None, no caching)
        """
        self.data_path = Path(data_path)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        if self.cache_dir:
            self.cache_dir.mkdir(exist_ok=True, parents=True)
            
        # Verify dataset path
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path not found: {self.data_path}")
        
        # Create simulated clients based on attack types
        # We'll create one client per attack type plus one for normal traffic
        attack_groups = list(ATTACK_CLASSES.keys())
        self.clients = []
        for i, attack_group in enumerate(attack_groups):
            self.clients.append(f"client_{i}_{attack_group}")
        
        # Create a mapping of attack types to client IDs
        self.attack_to_client_map = {attack: f"client_{i}_{attack}" for i, attack in enumerate(attack_groups)}
        self.client_to_attack_map = {v: k for k, v in self.attack_to_client_map.items()}
        
        logger.info(f"Initialized NSL-KDD data loader with path: {self.data_path}")
    
    def load_dataset(self, use_cache: bool = True) -> Dict[str, Dict]:
        """
        Load the entire NSL-KDD dataset.
        
        Args:
            use_cache: Whether to use cached data if available
            
        Returns:
            Dictionary mapping attack types to their data
        """
        cache_file = self.cache_dir / "nslkdd_full_dataset.npz" if self.cache_dir else None
        
        # Try to load from cache if enabled
        if use_cache and cache_file and cache_file.exists():
            logger.info(f"Loading dataset from cache: {cache_file}")
            try:
                data = np.load(cache_file, allow_pickle=True)
                dataset = data['dataset'].item()  # Convert to dictionary
                logger.info(f"Loaded dataset from cache with {len(dataset)} attack types")
                return dataset
            except Exception as e:
                logger.warning(f"Failed to load dataset from cache: {e}")
                
        # Load fresh dataset
        dataset = {}
        
        # Find the training data file
        train_file = self._find_dataset_file("KDDTrain+")
        if not train_file:
            raise FileNotFoundError("KDDTrain+ file not found in the data directory")
        
        # Find the test data file
        test_file = self._find_dataset_file("KDDTest+")
        if not test_file:
            raise FileNotFoundError("KDDTest+ file not found in the data directory")
        
        # Load and combine training and test data
        train_data = self._load_data_file(train_file)
        test_data = self._load_data_file(test_file)
        
        all_data = pd.concat([train_data, test_data], ignore_index=True)
        logger.info(f"Loaded combined dataset with {len(all_data)} samples")
        
        # Group data by attack class
        for attack_class in ATTACK_CLASSES.keys():
            if attack_class == 'normal':
                # For normal traffic
                class_data = all_data[all_data['class'] == 'normal']
                class_data['label'] = BENIGN_LABEL
            else:
                # For attack traffic, filter based on attack types
                attack_list = ATTACK_TYPES[attack_class]
                class_data = all_data[all_data['class'].isin(attack_list)]
                class_data['label'] = MALICIOUS_LABEL
            
            if len(class_data) == 0:
                logger.warning(f"No data found for attack class: {attack_class}")
                continue
            
            # Prepare X and y
            X = class_data.drop(['class', 'difficulty', 'label'], axis=1, errors='ignore')
            y = class_data['label'].values
            
            # Get feature names
            feature_names = X.columns.tolist()
            
            # Convert to numpy arrays
            X = X.values
            
            # Count statistics
            benign_count = np.sum(y == BENIGN_LABEL)
            malicious_count = np.sum(y == MALICIOUS_LABEL)
            
            logger.info(f"Attack class {attack_class}: {X.shape[0]} samples, {X.shape[1]} features, "
                       f"{benign_count} benign, {malicious_count} malicious")
            
            dataset[attack_class] = {
                "X": X,
                "y": y,
                "feature_names": feature_names,
                "benign_count": benign_count,
                "malicious_count": malicious_count,
                "attack_class": attack_class
            }
            
        # Cache the dataset if enabled
        if use_cache and cache_file:
            logger.info(f"Caching dataset to: {cache_file}")
            try:
                np.savez_compressed(cache_file, dataset=dataset)
            except Exception as e:
                logger.warning(f"Failed to cache dataset: {e}")
                
        return dataset
    
    def _find_dataset_file(self, file_prefix: str) -> Optional[Path]:
        """
        Find a dataset file with the given prefix.
        
        Args:
            file_prefix: Prefix of the file to find
            
        Returns:
            Path to the file if found, None otherwise
        """
        # Look for csv files first
        csv_files = list(self.data_path.glob(f"{file_prefix}*.csv"))
        if csv_files:
            return csv_files[0]
        
        # Look for txt files next
        txt_files = list(self.data_path.glob(f"{file_prefix}*.txt"))
        if txt_files:
            return txt_files[0]
        
        return None
    
    def _load_data_file(self, file_path: Path) -> pd.DataFrame:
        """
        Load data from a file.
        
        Args:
            file_path: Path to the data file
            
        Returns:
            DataFrame with the loaded data
        """
        # Check file extension
        if file_path.suffix.lower() == '.csv':
            # Try to load as CSV
            try:
                data = pd.read_csv(file_path)
                # If the data has no column names, assign them
                if data.shape[1] == len(FEATURE_NAMES):
                    data.columns = FEATURE_NAMES
                return data
            except Exception as e:
                logger.warning(f"Error loading CSV file: {e}")
                # Fall back to loading without header
                try:
                    data = pd.read_csv(file_path, header=None)
                    # Assign column names
                    if data.shape[1] == len(FEATURE_NAMES):
                        data.columns = FEATURE_NAMES
                    return data
                except Exception as e2:
                    logger.error(f"Error loading CSV file without header: {e2}")
                    raise
        elif file_path.suffix.lower() == '.txt':
            # Try to load as txt with comma separator
            try:
                data = pd.read_csv(file_path, header=None, sep=',')
                # Assign column names
                if data.shape[1] == len(FEATURE_NAMES):
                    data.columns = FEATURE_NAMES
                return data
            except Exception as e:
                logger.warning(f"Error loading TXT file with comma separator: {e}")
                # Try with different separators
                for sep in ['\t', ' ']:
                    try:
                        data = pd.read_csv(file_path, header=None, sep=sep)
                        if data.shape[1] == len(FEATURE_NAMES):
                            data.columns = FEATURE_NAMES
                        return data
                    except:
                        pass
                logger.error(f"Failed to load TXT file with any separator")
                raise
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
    def load_data_for_client(self, client_id: str, use_cache: bool = True) -> Dict:
        """
        Load data for a specific client.
        
        Args:
            client_id: ID of the client
            use_cache: Whether to use cached data if available
            
        Returns:
            Dictionary with client data
        """
        # Get attack type from client ID
        attack_class = self.client_to_attack_map.get(client_id)
        
        if attack_class is None:
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
                    "attack_class": data['attack_class'].item()
                }
                logger.info(f"Loaded client data from cache with {client_data['X'].shape[0]} samples")
                return client_data
            except Exception as e:
                logger.warning(f"Failed to load client data from cache: {e}")
        
        # Load full dataset
        full_dataset = self.load_dataset(use_cache=use_cache)
        
        # Get data for the specific attack class
        if attack_class not in full_dataset:
            raise ValueError(f"No data found for attack class: {attack_class}")
            
        client_data = full_dataset[attack_class]
        
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
                    attack_class=attack_class
                )
            except Exception as e:
                logger.warning(f"Failed to cache client data: {e}")
        
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
        encode_categorical: bool = True,
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
            encode_categorical: Whether to one-hot encode categorical features
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary with preprocessed data
        """
        from sklearn.model_selection import train_test_split
        
        np.random.seed(random_state)
        
        X = client_data["X"]
        y = client_data["y"]
        feature_names = client_data["feature_names"]
        
        # Handle categorical features first
        if encode_categorical:
            X, feature_names = self._encode_categorical_features(X, feature_names)
            
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
    
    def _encode_categorical_features(self, X: np.ndarray, feature_names: List[str]) -> Tuple[np.ndarray, List[str]]:
        """
        Encode categorical features in the dataset.
        
        Args:
            X: Feature matrix
            feature_names: Names of the features
            
        Returns:
            Tuple of (encoded_features, new_feature_names)
        """
        # Identify categorical feature indices
        categorical_indices = []
        for i, name in enumerate(feature_names):
            if name in ['protocol_type', 'service', 'flag']:
                categorical_indices.append(i)
        
        if not categorical_indices:
            return X, feature_names
        
        # One-hot encode each categorical feature
        encoded_features = []
        new_feature_names = []
        
        # Process non-categorical features first
        for i in range(X.shape[1]):
            if i not in categorical_indices:
                encoded_features.append(X[:, i].reshape(-1, 1))
                new_feature_names.append(feature_names[i])
        
        # Process categorical features
        for i in categorical_indices:
            feature_name = feature_names[i]
            
            # Get the possible values for this feature
            if feature_name == 'protocol_type':
                categories = PROTOCOL_TYPES
            elif feature_name == 'service':
                categories = SERVICE_TYPES
            elif feature_name == 'flag':
                categories = FLAG_TYPES
            else:
                # If we don't know the categories, use OneHotEncoder to determine them
                encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
                encoded = encoder.fit_transform(X[:, i].reshape(-1, 1))
                categories = encoder.categories_[0]
            
            # Create one-hot encoder for this feature
            encoder = OneHotEncoder(sparse=False, handle_unknown='ignore', categories=[categories])
            encoded = encoder.fit_transform(X[:, i].reshape(-1, 1))
            
            # Add encoded feature columns
            encoded_features.append(encoded)
            
            # Add encoded feature names
            for category in encoder.categories_[0]:
                new_feature_names.append(f"{feature_name}_{category}")
        
        # Concatenate all features
        X_encoded = np.hstack(encoded_features)
        
        return X_encoded, new_feature_names
        
    def create_federated_data_partition(
        self,
        num_clients: int = None,
        client_ids: List[str] = None,
        include_normal: bool = True,
        include_attacks: bool = True,
        preprocessing_params: Dict = None,
        use_cache: bool = True,
        random_state: int = 42
    ) -> Dict[str, Dict]:
        """
        Create a federated data partition for training.
        
        Args:
            num_clients: Number of clients to create (if client_ids is None)
            client_ids: List of specific client IDs to use (takes precedence over num_clients)
            include_normal: Whether to include normal traffic
            include_attacks: Whether to include attack traffic
            preprocessing_params: Parameters for preprocessing data
            use_cache: Whether to use cached data if available
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary mapping client IDs to their preprocessed data
        """
        np.random.seed(random_state)
        
        # Set default preprocessing parameters if not provided
        if preprocessing_params is None:
            preprocessing_params = {
                "normalize": True,
                "feature_selection": False,
                "num_features": 10,
                "test_size": 0.2,
                "validation_size": 0.1,
                "balance_classes": True,
                "encode_categorical": True,
                "random_state": random_state
            }
        
        # Choose clients based on parameters
        all_clients = self.clients
        
        # Filter clients based on attack/normal
        if not include_normal:
            all_clients = [c for c in all_clients if 'normal' not in c]
        if not include_attacks:
            all_clients = [c for c in all_clients if 'normal' in c]
        
        # Select clients
        if client_ids is not None:
            # Validate client IDs
            invalid_clients = [c for c in client_ids if c not in self.clients]
            if invalid_clients:
                raise ValueError(f"Invalid client IDs: {invalid_clients}")
            selected_clients = client_ids
        elif num_clients is not None:
            if num_clients > len(all_clients):
                logger.warning(f"Requested {num_clients} clients, but only {len(all_clients)} are available")
                num_clients = len(all_clients)
            selected_clients = np.random.choice(all_clients, num_clients, replace=False).tolist()
        else:
            selected_clients = all_clients
        
        logger.info(f"Creating federated data partition for {len(selected_clients)} clients: {selected_clients}")
        
        # Load and preprocess data for each client
        federated_data = {}
        
        for client_id in selected_clients:
            # Create cache key for this partition
            partition_key = f"{client_id}_{'_'.join([f'{k}_{v}' for k, v in preprocessing_params.items() if not isinstance(v, dict)])}"
            partition_cache_file = self.cache_dir / f"{partition_key}_partition.npz" if self.cache_dir else None
            
            # Try to load from cache if enabled
            if use_cache and partition_cache_file and partition_cache_file.exists():
                logger.info(f"Loading partition data from cache: {partition_cache_file}")
                try:
                    data = np.load(partition_cache_file, allow_pickle=True)
                    client_partition = {k: data[k] for k in data.files}
                    # Convert arrays with object dtype to their actual types
                    if "selected_feature_names" in client_partition:
                        client_partition["selected_feature_names"] = client_partition["selected_feature_names"].tolist()
                    if "selected_indices" in client_partition and client_partition["selected_indices"].dtype == np.dtype('O'):
                        client_partition["selected_indices"] = client_partition["selected_indices"].item()
                    
                    logger.info(f"Loaded partition data from cache for client {client_id}")
                    federated_data[client_id] = client_partition
                    continue
                except Exception as e:
                    logger.warning(f"Failed to load partition data from cache: {e}")
            
            # Load client data
            client_data = self.load_data_for_client(client_id, use_cache=use_cache)
            
            # Preprocess client data
            client_partition = self.preprocess_client_data(client_data, **preprocessing_params)
            
            # Cache the processed partition if enabled
            if use_cache and partition_cache_file:
                logger.info(f"Caching partition data to: {partition_cache_file}")
                try:
                    np.savez_compressed(partition_cache_file, **client_partition)
                except Exception as e:
                    logger.warning(f"Failed to cache partition data: {e}")
            
            federated_data[client_id] = client_partition
        
        logger.info(f"Created federated data partition for {len(federated_data)} clients")
        
        return federated_data
    
    def get_all_clients(self) -> List[str]:
        """
        Get all client IDs.
        
        Returns:
            List of client IDs
        """
        return self.clients
    
    def get_client_info(self) -> Dict[str, str]:
        """
        Get mapping of client IDs to attack types.
        
        Returns:
            Dictionary mapping client IDs to attack types
        """
        return self.client_to_attack_map 