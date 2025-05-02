"""
Models module for FIDS.

This module contains implementations of various machine learning and deep learning
models for IoT device security in federated learning settings.
"""

from .model_factory import create_model, list_available_models, save_model, load_model
from .traditional import RandomForestModel, NaiveBayesModel, LogisticRegressionModel, SVMModel
from .deep_learning import LSTMModel, BiLSTMModel, CNNModel, MLPModel

__all__ = [
    'create_model',
    'list_available_models',
    'save_model',
    'load_model',
    'RandomForestModel',
    'NaiveBayesModel',
    'LogisticRegressionModel',
    'SVMModel',
    'LSTMModel',
    'BiLSTMModel',
    'CNNModel',
    'MLPModel'
]