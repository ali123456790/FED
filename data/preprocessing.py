"""
Data preprocessing utilities.
"""

import logging
import numpy as np
from typing import Dict, Tuple, List, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

def normalize_data(X: np.ndarray, scaler: Optional[StandardScaler] = None) -> Tuple[np.ndarray, StandardScaler]:
    """
    Normalize data using StandardScaler.
    
    Args:
        X: Data to normalize
        scaler: Scaler to use (if None, a new one will be created)
        
    Returns:
        Tuple of (normalized_data, scaler)
    """
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(X)
    
    X_normalized = scaler.transform(X)
    
    return X_normalized, scaler

def scale_data(X: np.ndarray, scaler: Optional[MinMaxScaler] = None) -> Tuple[np.ndarray, MinMaxScaler]:
    """
    Scale data to [0, 1] using MinMaxScaler.
    
    Args:
        X: Data to scale
        scaler: Scaler to use (if None, a new one will be created)
        
    Returns:
        Tuple of (scaled_data, scaler)
    """
    if scaler is None:
        scaler = MinMaxScaler()
        scaler.fit(X)
    
    X_scaled = scaler.transform(X)
    
    return X_scaled, scaler

def split_train_test_validation(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    validation_size: float = 0.1,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train, test, and validation sets.
    
    Args:
        X: Features
        y: Labels
        test_size: Proportion of data to use for testing
        validation_size: Proportion of data to use for validation
        random_state: Random state for reproducibility
        
    Returns:
        Tuple of (X_train, X_test, X_val, y_train, y_test, y_val)
    """
    # First, split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Then, split train into train and validation
    if validation_size > 0:
        # Calculate validation size relative to the training set
        val_size_relative = validation_size / (1 - test_size)
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size_relative, random_state=random_state
        )
        
        return X_train, X_test, X_val, y_train, y_test, y_val
    else:
        return X_train, X_test, None, y_train, y_test, None

def handle_missing_values(X: np.ndarray, strategy: str = "mean") -> np.ndarray:
    """
    Handle missing values in the data.
    
    Args:
        X: Data with missing values
        strategy: Strategy for handling missing values (mean, median, mode, or constant)
        
    Returns:
        Data with missing values handled
    """
    from sklearn.impute import SimpleImputer
    
    # Create imputer
    if strategy == "mean":
        imputer = SimpleImputer(strategy="mean")
    elif strategy == "median":
        imputer = SimpleImputer(strategy="median")
    elif strategy == "mode":
        imputer = SimpleImputer(strategy="most_frequent")
    elif strategy == "constant":
        imputer = SimpleImputer(strategy="constant", fill_value=0)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # Fit and transform
    X_imputed = imputer.fit_transform(X)
    
    return X_imputed

def remove_outliers(X: np.ndarray, y: np.ndarray, method: str = "iqr") -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove outliers from the data.
    
    Args:
        X: Features
        y: Labels
        method: Method for outlier detection (iqr or zscore)
        
    Returns:
        Tuple of (X_cleaned, y_cleaned)
    """
    if method == "iqr":
        # IQR method
        q1 = np.percentile(X, 25, axis=0)
        q3 = np.percentile(X, 75, axis=0)
        iqr = q3 - q1
        
        # Define bounds
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Find outliers
        outlier_mask = np.any((X < lower_bound) | (X > upper_bound), axis=1)
    elif method == "zscore":
        # Z-score method
        from scipy import stats
        z_scores = np.abs(stats.zscore(X))
        outlier_mask = np.any(z_scores > 3, axis=1)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Remove outliers
    X_cleaned = X[~outlier_mask]
    y_cleaned = y[~outlier_mask]
    
    logger.info(f"Removed {np.sum(outlier_mask)} outliers using {method} method")
    
    return X_cleaned, y_cleaned

