"""
Feature selection utilities.
"""

import logging
import numpy as np
from typing import Dict, Tuple, List, Optional
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger(__name__)

def select_features_kbest(
    X: np.ndarray,
    y: np.ndarray,
    k: int = 10,
    score_func: str = "f_classif"
) -> Tuple[np.ndarray, List[int]]:
    """
    Select k best features using univariate statistical tests.
    
    Args:
        X: Features
        y: Labels
        k: Number of features to select
        score_func: Scoring function to use (f_classif or mutual_info_classif)
        
    Returns:
        Tuple of (X_selected, selected_indices)
    """
    # Choose scoring function
    if score_func == "f_classif":
        score_function = f_classif
    elif score_func == "mutual_info_classif":
        score_function = mutual_info_classif
    else:
        raise ValueError(f"Unknown score function: {score_func}")
    
    # Create selector
    selector = SelectKBest(score_function, k=k)
    
    # Fit and transform
    X_selected = selector.fit_transform(X, y)
    
    # Get selected feature indices
    selected_indices = np.where(selector.get_support())[0]
    
    logger.info(f"Selected {k} features using {score_func}")
    
    return X_selected, selected_indices

def select_features_rfe(
    X: np.ndarray,
    y: np.ndarray,
    n_features_to_select: int = 10
) -> Tuple[np.ndarray, List[int]]:
    """
    Select features using Recursive Feature Elimination.
    
    Args:
        X: Features
        y: Labels
        n_features_to_select: Number of features to select
        
    Returns:
        Tuple of (X_selected, selected_indices)
    """
    # Create estimator
    estimator = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Create selector
    selector = RFE(estimator, n_features_to_select=n_features_to_select)
    
    # Fit and transform
    X_selected = selector.fit_transform(X, y)
    
    # Get selected feature indices
    selected_indices = np.where(selector.get_support())[0]
    
    logger.info(f"Selected {n_features_to_select} features using RFE")
    
    return X_selected, selected_indices

def select_features_importance(
    X: np.ndarray,
    y: np.ndarray,
    n_features_to_select: int = 10
) -> Tuple[np.ndarray, List[int]]:
    """
    Select features using feature importance from a Random Forest.
    
    Args:
        X: Features
        y: Labels
        n_features_to_select: Number of features to select
        
    Returns:
        Tuple of (X_selected, selected_indices)
    """
    # Create and fit estimator
    estimator = RandomForestClassifier(n_estimators=100, random_state=42)
    estimator.fit(X, y)
    
    # Get feature importances
    importances = estimator.feature_importances_
    
    # Get indices of top features
    selected_indices = np.argsort(importances)[::-1][:n_features_to_select]
    
    # Select features
    X_selected = X[:, selected_indices]
    
    logger.info(f"Selected {n_features_to_select} features using feature importance")
    
    return X_selected, selected_indices

def select_features(
    X: np.ndarray,
    y: np.ndarray,
    method: str = "kbest",
    n_features: int = 10,
    **kwargs
) -> Tuple[np.ndarray, List[int]]:
    """
    Select features using the specified method.
    
    Args:
        X: Features
        y: Labels
        method: Method for feature selection (kbest, rfe, or importance)
        n_features: Number of features to select
        **kwargs: Additional arguments for the feature selection method
        
    Returns:
        Tuple of (X_selected, selected_indices)
    """
    if method == "kbest":
        return select_features_kbest(X, y, k=n_features, **kwargs)
    elif method == "rfe":
        return select_features_rfe(X, y, n_features_to_select=n_features)
    elif method == "importance":
        return select_features_importance(X, y, n_features_to_select=n_features)
    else:
        raise ValueError(f"Unknown feature selection method: {method}")

