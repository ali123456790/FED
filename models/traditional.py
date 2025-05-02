"""
Traditional machine learning models for IoT device security.

This module implements scikit-learn based models such as Random Forest,
Naive Bayes, Logistic Regression, and SVM for anomaly detection in IoT devices.
"""

import logging
import numpy as np
import os
import pickle
from typing import Dict, List, Tuple, Any, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, log_loss
from sklearn.model_selection import GridSearchCV
import joblib

logger = logging.getLogger(__name__)

class BaseTraditionalModel:
    """Base class for traditional machine learning models."""
    
    def __init__(self, config: Dict):
        """
        Initialize the base traditional model.
        
        Args:
            config: Model configuration
        """
        self.config = config
        self.model = None
        self.model_path = None
        self.is_fitted = False
        
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'BaseTraditionalModel':
        """
        Train the model.
        
        Args:
            X: Training features
            y: Training labels
            **kwargs: Additional arguments
            
        Returns:
            Self for method chaining
        """
        raise NotImplementedError("Subclasses must implement fit method")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features to predict
            
        Returns:
            Predicted labels
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Features to predict
            
        Returns:
            Class probabilities
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predicting probabilities")
            
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            # If the model doesn't support probability predictions,
            # convert binary predictions to pseudo-probabilities
            preds = self.predict(X)
            proba = np.zeros((len(preds), 2))
            proba[:, 1] = preds
            proba[:, 0] = 1 - preds
            return proba
    
    def evaluate(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, float]:
        """
        Evaluate the model.
        
        Args:
            X: Test features
            y: Test labels
            **kwargs: Additional arguments
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before evaluation")
            
        # Get predictions
        y_pred = self.predict(X)
        
        # Get probability predictions if available
        try:
            y_pred_proba = self.predict_proba(X)
            has_proba = True
        except (AttributeError, RuntimeError):
            has_proba = False
        
        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, average='weighted'),
            "recall": recall_score(y, y_pred, average='weighted'),
            "f1": f1_score(y, y_pred, average='weighted')
        }
        
        # Calculate AUC and log loss if probability predictions are available
        if has_proba and y_pred_proba.shape[1] >= 2:
            # For binary classification
            if len(np.unique(y)) == 2:
                metrics["auc"] = roc_auc_score(y, y_pred_proba[:, 1])
                metrics["log_loss"] = log_loss(y, y_pred_proba)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y, y_pred)
        metrics["confusion_matrix"] = cm
        
        logger.info(f"Evaluation metrics: {metrics}")
        
        return metrics
    
    def save(self, path: str) -> None:
        """
        Save the model to disk.
        
        Args:
            path: Path to save the model
        """
        # Make sure the directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save the model
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
            
        self.model_path = path
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str) -> 'BaseTraditionalModel':
        """
        Load the model from disk.
        
        Args:
            path: Path to load the model from
            
        Returns:
            Self for method chaining
        """
        # Check if the file exists
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
            
        # Load the model
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
            
        self.model_path = path
        self.is_fitted = True
        logger.info(f"Model loaded from {path}")
        
        return self
    
    def get_weights(self) -> List[np.ndarray]:
        """
        Get model weights for federated learning.
        
        Returns:
            List of model weights as numpy arrays
        """
        # Serialize the model
        serialized_model = pickle.dumps(self.model)
        
        # Convert to numpy array
        weights = [np.frombuffer(serialized_model, dtype=np.uint8)]
        
        return weights
    
    def set_weights(self, weights: List[np.ndarray]) -> None:
        """
        Set model weights for federated learning.
        
        Args:
            weights: List of model weights
        """
        if len(weights) != 1:
            raise ValueError(f"Expected 1 weight array, got {len(weights)}")
            
        # Convert numpy array back to bytes
        serialized_model = weights[0].tobytes()
        
        # Deserialize the model
        self.model = pickle.loads(serialized_model)
        self.is_fitted = True

class RandomForestModel(BaseTraditionalModel):
    """Random Forest model for IoT device security."""
    
    def __init__(self, config: Dict):
        """
        Initialize the Random Forest model.
        
        Args:
            config: Model configuration
        """
        super().__init__(config)
        self.hyperparameters = config["hyperparameters"]["random_forest"]
        
        # Create the model
        self.model = RandomForestClassifier(
            n_estimators=self.hyperparameters.get("n_estimators", 100),
            max_depth=self.hyperparameters.get("max_depth", None),
            min_samples_split=self.hyperparameters.get("min_samples_split", 2),
            min_samples_leaf=self.hyperparameters.get("min_samples_leaf", 1),
            max_features=self.hyperparameters.get("max_features", "sqrt"),
            bootstrap=self.hyperparameters.get("bootstrap", True),
            random_state=self.hyperparameters.get("random_state", 42),
            n_jobs=self.hyperparameters.get("n_jobs", -1),
            class_weight=self.hyperparameters.get("class_weight", None)
        )
        
        logger.info("Random Forest model initialized")
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'RandomForestModel':
        """
        Train the Random Forest model.
        
        Args:
            X: Training features
            y: Training labels
            **kwargs: Additional arguments including:
                - feature_names: Names of features
                - validation_data: Tuple of (X_val, y_val)
                
        Returns:
            Self for method chaining
        """
        logger.info(f"Training Random Forest model with {X.shape[0]} samples and {X.shape[1]} features")
        
        # Extract additional parameters
        feature_names = kwargs.get('feature_names', None)
        validation_data = kwargs.get('validation_data', None)
        sample_weight = kwargs.get('sample_weight', None)
        
        # Set feature names if provided
        if feature_names is not None and hasattr(self.model, 'feature_names_in_'):
            self.model.feature_names_in_ = feature_names
        
        # Enable warm start if continuing training
        if self.is_fitted and kwargs.get('warm_start', False):
            self.model.warm_start = True
        
        # Train the model
        start_time = kwargs.get('start_time', None)
        if start_time:
            import time
            training_start = time.time()
            
        self.model.fit(X, y, sample_weight=sample_weight)
        self.is_fitted = True
        
        if start_time:
            training_time = time.time() - training_start
            logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Evaluate on validation data if provided
        if validation_data is not None:
            X_val, y_val = validation_data
            val_metrics = self.evaluate(X_val, y_val)
            logger.info(f"Validation metrics: {val_metrics}")
        
        return self
    
    def feature_importance(self, feature_names: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Get feature importance from the Random Forest model.
        
        Args:
            feature_names: Names of features (if None, use indices)
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before getting feature importance")
            
        # Get feature importance
        importances = self.model.feature_importances_
        
        # Map to feature names
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importances))]
            
        # Create dictionary of feature importances
        importance_dict = {name: float(importance) for name, importance in zip(feature_names, importances)}
        
        # Sort by importance (descending)
        importance_dict = {k: v for k, v in sorted(importance_dict.items(), key=lambda item: item[1], reverse=True)}
        
        return importance_dict

class NaiveBayesModel(BaseTraditionalModel):
    """Gaussian Naive Bayes model for IoT device security."""
    
    def __init__(self, config: Dict):
        """
        Initialize the Naive Bayes model.
        
        Args:
            config: Model configuration
        """
        super().__init__(config)
        self.hyperparameters = config["hyperparameters"]["naive_bayes"]
        
        # Create the model
        self.model = GaussianNB(
            var_smoothing=self.hyperparameters.get("var_smoothing", 1e-9),
            priors=self.hyperparameters.get("priors", None)
        )
        
        logger.info("Naive Bayes model initialized")
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'NaiveBayesModel':
        """
        Train the Naive Bayes model.
        
        Args:
            X: Training features
            y: Training labels
            **kwargs: Additional arguments including:
                - validation_data: Tuple of (X_val, y_val)
                
        Returns:
            Self for method chaining
        """
        logger.info(f"Training Naive Bayes model with {X.shape[0]} samples and {X.shape[1]} features")
        
        # Extract additional parameters
        validation_data = kwargs.get('validation_data', None)
        sample_weight = kwargs.get('sample_weight', None)
        
        # Train the model
        start_time = kwargs.get('start_time', None)
        if start_time:
            import time
            training_start = time.time()
            
        self.model.fit(X, y, sample_weight=sample_weight)
        self.is_fitted = True
        
        if start_time:
            training_time = time.time() - training_start
            logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Evaluate on validation data if provided
        if validation_data is not None:
            X_val, y_val = validation_data
            val_metrics = self.evaluate(X_val, y_val)
            logger.info(f"Validation metrics: {val_metrics}")
        
        return self

class LogisticRegressionModel(BaseTraditionalModel):
    """Logistic Regression model for IoT device security."""
    
    def __init__(self, config: Dict):
        """
        Initialize the Logistic Regression model.
        
        Args:
            config: Model configuration
        """
        super().__init__(config)
        self.hyperparameters = config["hyperparameters"]["logistic_regression"]
        
        # Create the model
        self.model = LogisticRegression(
            penalty=self.hyperparameters.get("penalty", "l2"),
            C=self.hyperparameters.get("C", 1.0),
            solver=self.hyperparameters.get("solver", "lbfgs"),
            max_iter=self.hyperparameters.get("max_iter", 1000),
            multi_class=self.hyperparameters.get("multi_class", "auto"),
            class_weight=self.hyperparameters.get("class_weight", None),
            random_state=self.hyperparameters.get("random_state", 42),
            n_jobs=self.hyperparameters.get("n_jobs", -1)
        )
        
        logger.info("Logistic Regression model initialized")
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'LogisticRegressionModel':
        """
        Train the Logistic Regression model.
        
        Args:
            X: Training features
            y: Training labels
            **kwargs: Additional arguments including:
                - validation_data: Tuple of (X_val, y_val)
                
        Returns:
            Self for method chaining
        """
        logger.info(f"Training Logistic Regression model with {X.shape[0]} samples and {X.shape[1]} features")
        
        # Extract additional parameters
        validation_data = kwargs.get('validation_data', None)
        sample_weight = kwargs.get('sample_weight', None)
        
        # Train the model
        start_time = kwargs.get('start_time', None)
        if start_time:
            import time
            training_start = time.time()
            
        self.model.fit(X, y, sample_weight=sample_weight)
        self.is_fitted = True
        
        if start_time:
            training_time = time.time() - training_start
            logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Evaluate on validation data if provided
        if validation_data is not None:
            X_val, y_val = validation_data
            val_metrics = self.evaluate(X_val, y_val)
            logger.info(f"Validation metrics: {val_metrics}")
        
        return self
    
    def get_coefficients(self, feature_names: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
        """
        Get coefficients from the Logistic Regression model.
        
        Args:
            feature_names: Names of features (if None, use indices)
            
        Returns:
            Dictionary mapping class names to dictionaries of feature coefficients
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before getting coefficients")
            
        # Get coefficients and intercept
        coefficients = self.model.coef_
        intercept = self.model.intercept_
        
        # Get class names
        class_names = [str(c) for c in self.model.classes_]
        
        # Map to feature names
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(coefficients.shape[1])]
            
        # Create coefficient dictionary
        coef_dict = {}
        
        # For binary classification with only one coefficient array
        if len(class_names) == 2 and coefficients.shape[0] == 1:
            # Positive class (coefficients given)
            pos_class = class_names[1]
            coef_dict[pos_class] = {
                "intercept": float(intercept[0]),
                "coefficients": {name: float(coef) for name, coef in zip(feature_names, coefficients[0])}
            }
            
            # Negative class (negative coefficients)
            neg_class = class_names[0]
            coef_dict[neg_class] = {
                "intercept": float(-intercept[0]),
                "coefficients": {name: float(-coef) for name, coef in zip(feature_names, coefficients[0])}
            }
        else:
            # Multi-class
            for i, class_name in enumerate(class_names):
                coef_dict[class_name] = {
                    "intercept": float(intercept[i]),
                    "coefficients": {name: float(coef) for name, coef in zip(feature_names, coefficients[i])}
                }
                
        return coef_dict

class SVMModel(BaseTraditionalModel):
    """Support Vector Machine model for IoT device security."""
    
    def __init__(self, config: Dict):
        """
        Initialize the SVM model.
        
        Args:
            config: Model configuration
        """
        super().__init__(config)
        self.hyperparameters = config["hyperparameters"]["svm"]
        
        # Create the model
        self.model = SVC(
            C=self.hyperparameters.get("C", 1.0),
            kernel=self.hyperparameters.get("kernel", "rbf"),
            degree=self.hyperparameters.get("degree", 3),
            gamma=self.hyperparameters.get("gamma", "scale"),
            coef0=self.hyperparameters.get("coef0", 0.0),
            shrinking=self.hyperparameters.get("shrinking", True),
            probability=self.hyperparameters.get("probability", True),
            tol=self.hyperparameters.get("tol", 1e-3),
            class_weight=self.hyperparameters.get("class_weight", None),
            random_state=self.hyperparameters.get("random_state", 42),
            max_iter=self.hyperparameters.get("max_iter", -1)
        )
        
        logger.info("SVM model initialized")
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'SVMModel':
        """
        Train the SVM model.
        
        Args:
            X: Training features
            y: Training labels
            **kwargs: Additional arguments including:
                - validation_data: Tuple of (X_val, y_val)
                
        Returns:
            Self for method chaining
        """
        logger.info(f"Training SVM model with {X.shape[0]} samples and {X.shape[1]} features")
        
        # Extract additional parameters
        validation_data = kwargs.get('validation_data', None)
        sample_weight = kwargs.get('sample_weight', None)
        
        # Train the model
        start_time = kwargs.get('start_time', None)
        if start_time:
            import time
            training_start = time.time()
            
        self.model.fit(X, y, sample_weight=sample_weight)
        self.is_fitted = True
        
        if start_time:
            training_time = time.time() - training_start
            logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Evaluate on validation data if provided
        if validation_data is not None:
            X_val, y_val = validation_data
            val_metrics = self.evaluate(X_val, y_val)
            logger.info(f"Validation metrics: {val_metrics}")
        
        return self

def find_best_model(
    X_train: np.ndarray, 
    y_train: np.ndarray,
    X_val: np.ndarray, 
    y_val: np.ndarray,
    model_types: List[str] = ["random_forest", "naive_bayes", "logistic_regression", "svm"],
    param_grids: Optional[Dict] = None,
    metric: str = "f1",
    n_jobs: int = -1,
    cv: int = 3,
    verbose: int = 1
) -> Tuple[str, Dict, Any]:
    """
    Find the best model using grid search.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        model_types: List of model types to try
        param_grids: Dictionary of parameter grids for each model type
        metric: Metric to optimize
        n_jobs: Number of jobs to run in parallel
        cv: Number of cross-validation folds
        verbose: Verbosity level
        
    Returns:
        Tuple of (best_model_type, best_params, best_model)
    """
    logger.info(f"Finding best model from {model_types}")
    
    # Define default parameter grids if not provided
    if param_grids is None:
        param_grids = {
            "random_forest": {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 10, 20, 30],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4]
            },
            "naive_bayes": {
                "var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6]
            },
            "logistic_regression": {
                "C": [0.1, 1.0, 10.0],
                "penalty": ["l2"],
                "solver": ["lbfgs", "liblinear"]
            },
            "svm": {
                "C": [0.1, 1.0, 10.0],
                "kernel": ["linear", "rbf"],
                "gamma": ["scale", "auto", 0.1, 0.01]
            }
        }
    
    # Define models
    models = {
        "random_forest": RandomForestClassifier(random_state=42),
        "naive_bayes": GaussianNB(),
        "logistic_regression": LogisticRegression(random_state=42, max_iter=1000),
        "svm": SVC(probability=True, random_state=42)
    }
    
    best_score = -1
    best_model_type = None
    best_params = None
    best_model = None
    
    # Try each model type
    for model_type in model_types:
        if model_type not in models:
            logger.warning(f"Unknown model type: {model_type}, skipping")
            continue
            
        logger.info(f"Grid searching for {model_type}")
        
        # Get model and parameter grid
        model = models[model_type]
        param_grid = param_grids.get(model_type, {})
        
        # Perform grid search
        grid_search = GridSearchCV(
            model,
            param_grid,
            scoring=metric,
            cv=cv,
            n_jobs=n_jobs,
            verbose=verbose
        )
        
        grid_search.fit(X_train, y_train)
        
        # Evaluate on validation set
        val_score = grid_search.score(X_val, y_val)
        
        logger.info(f"{model_type} - Best {metric}: {grid_search.best_score_:.4f} (validation: {val_score:.4f})")
        logger.info(f"{model_type} - Best parameters: {grid_search.best_params_}")
        
        # Update best model if this one is better
        if val_score > best_score:
            best_score = val_score
            best_model_type = model_type
            best_params = grid_search.best_params_
            best_model = grid_search.best_estimator_
    
    logger.info(f"Best model: {best_model_type} with {metric}={best_score:.4f}")
    
    return best_model_type, best_params, best_model