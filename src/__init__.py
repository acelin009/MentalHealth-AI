"""
Mental Health AI - Source Code Package

This package contains all the core functionality for the mental health prediction project.
"""

from src.config import (
    BASE_DIR, DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, 
    MODELS_DIR, RESULTS_DIR, NOTEBOOKS_DIR, DASHBOARD_DIR  # CHANGED: MODEL_DIR -> MODELS_DIR
)
from src.data_loader import load_ml_dataset, load_engineered_dataset, DataLoader
from src.preprocessor import build_preprocessor, separate_features_target, get_feature_types
from src.train import train_logistic, train_tree, train_forest, train_with_cv, save_model, load_model
from src.evaluate import evaluate_model, plot_confusion_matrix, plot_feature_importance, compare_models
from src.predict import MentalHealthPredictor, create_prediction_input
from src.explain import compute_shap_values, plot_shap_summary, explain_prediction
from src.data_cleaner import DataCleaner

__version__ = "1.0.0"
__all__ = [
    'BASE_DIR', 'DATA_DIR', 'RAW_DATA_DIR', 'PROCESSED_DATA_DIR', 
    'MODELS_DIR', 'RESULTS_DIR', 'NOTEBOOKS_DIR', 'DASHBOARD_DIR',  # CHANGED
    'load_ml_dataset', 'load_engineered_dataset', 'DataLoader',
    'build_preprocessor', 'separate_features_target', 'get_feature_types',
    'train_logistic', 'train_tree', 'train_forest', 'train_with_cv', 'save_model', 'load_model',
    'evaluate_model', 'plot_confusion_matrix', 'plot_feature_importance', 'compare_models',
    'MentalHealthPredictor', 'create_prediction_input',
    'compute_shap_values', 'plot_shap_summary', 'explain_prediction',
    'DataCleaner'
]