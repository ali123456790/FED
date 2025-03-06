"""
Traditional machine learning models for IoT device security.
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

logger = logging.getLogger(__name__)

class RandomForestModel:
    """Random Forest model for IoT device security."""
    
    def __init__(self, config: Dict):
        """
        Initialize the Random Forest model.
        
        Args:
            config: Model configuration
        """
        self.config = config
        self.hyperparameters = config["hyperparameters"]["random_forest"]
        self.model = RandomForestClassifier(
            n_estimators=self.hyperparameters["n_estimators"],
            max_depth=self.hyperparameters["max_depth"],
            random_state=42
        )
        
        logger.info("Random Forest model initialized")
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """
        Train the model.
        
        Args:
            X: Training features
            y: Training labels
            **kwargs: Additional arguments
        """
        self.model.fit(X, y)
        logger.info("Random Forest model trained")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features to predict
            
        Returns:
            Predicted labels
        """
        return self.model.predict(X)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Tuple[float, float]:
        """
        Evaluate the model.
        
        Args:
            X: Test features
            y: Test labels
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (loss, accuracy)
        """
        from sklearn.metrics import accuracy_score, log_loss
        
        y_pred = self.model.predict(X)
        y_pred_proba = self.model.predict_proba(X)
        
        accuracy = accuracy_score(y, y_pred)
        loss = log_loss(y, y_pred_proba)
        
        logger.info(f"Random Forest evaluation: loss={loss:.4f}, accuracy={accuracy:.4f}")
        
        return loss, accuracy
    
    def get_weights(self) -> List[np.ndarray]:
        """
        Get model weights.
        
        Returns:
            List of model weights
        """
        # Extract model parameters
        # For Random Forest, we can serialize the entire model
        import pickle
        serialized_model = pickle.dumps(self.model)
        return [np.frombuffer(serialized_model, dtype=np.uint8)]
    
    def set_weights(self, weights: List[np.ndarray]) -> None:
        """
        Set model weights.
        
        Args:
            weights: List of model weights
        """
        # Deserialize the model
        import pickle
        serialized_model = weights[0].tobytes()
        self.model = pickle.loads(serialized_model)

class NaiveBayesModel:
    """Naive Bayes model for IoT device security."""
    
    def __init__(self, config: Dict):
        """
        Initialize the Naive Bayes model.
        
        Args:
            config: Model configuration
        """
        self.config = config
        self.hyperparameters = config["hyperparameters"]["naive_bayes"]
        self.model = GaussianNB(
            var_smoothing=self.hyperparameters["var_smoothing"]
        )
        
        logger.info("Naive Bayes model initialized")
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """
        Train the model.
        
        Args:
            X: Training features
            y: Training labels
            **kwargs: Additional arguments
        """
        self.model.fit(X, y)
        logger.info("Naive Bayes model trained")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features to predict
            
        Returns:
            Predicted labels
        """
        return self.model.predict(X)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Tuple[float, float]:
        """
        Evaluate the model.
        
        Args:
            X: Test features
            y: Test labels
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (loss, accuracy)
        """
        from sklearn.metrics import accuracy_score, log_loss
        
        y_pred = self.model.predict(X)
        y_pred_proba = self.model.predict_proba(X)
        
        accuracy = accuracy_score(y, y_pred)
        loss = log_loss(y, y_pred_proba)
        
        logger.info(f"Naive Bayes evaluation: loss={loss:.4f}, accuracy={accuracy:.4f}")
        
        return loss, accuracy
    
    def get_weights(self) -> List[np.ndarray]:
        """
        Get model weights.
        
        Returns:
            List of model weights
        """
        # Extract model parameters
        # For Naive Bayes, we can serialize the entire model
        import pickle
        serialized_model = pickle.dumps(self.model)
        return [np.frombuffer(serialized_model, dtype=np.uint8)]
    
    def set_weights(self, weights: List[np.ndarray]) -> None:
        """
        Set model weights.
        
        Args:
            weights: List of model weights
        """
        # Deserialize the model
        import pickle
        serialized_model = weights[0].tobytes()
        self.model = pickle.loads(serialized_model)

