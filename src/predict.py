import pandas as pd
import numpy as np
import joblib
import streamlit as st
from src.config import MODELS_DIR


class MentalHealthPredictor:
    """
    A class for making predictions with trained mental health models.
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the predictor with a trained model.
        
        Args:
            model_path: Path to the saved model file
        """
        if model_path is None:
            model_path = MODELS_DIR / "best_model.pkl"
        elif isinstance(model_path, str):
            model_path = Path(model_path)
        
        self.model_path = model_path
        self.model = None
        self.feature_names = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model from disk."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        self.model = joblib.load(self.model_path)
        return self.model
    
    def predict(self, data):
        """
        Make predictions on new data.
        
        Args:
            data: DataFrame or array-like input data
        
        Returns:
            Array of predictions
        """
        if self.model is None:
            self.load_model()
        
        # Handle single instance
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        elif isinstance(data, list) and not isinstance(data, pd.DataFrame):
            data = pd.DataFrame([data])
        
        return self.model.predict(data)
    
    def predict_proba(self, data):
        """
        Get prediction probabilities.
        
        Args:
            data: DataFrame or array-like input data
        
        Returns:
            Array of prediction probabilities
        """
        if self.model is None:
            self.load_model()
        
        try:
            return self.model.predict_proba(data)
        except AttributeError:
            raise AttributeError("Model does not support probability predictions")
    
    def predict_single(self, data_dict):
        """
        Make a prediction on a single record.
        
        Args:
            data_dict: Dictionary of feature values
        
        Returns:
            Single prediction
        """
        df = pd.DataFrame([data_dict])
        return self.predict(df)[0]
    
    def get_feature_importance(self, top_n=None):
        """
        Get feature importance from the model.
        
        Args:
            top_n: Number of top features to return
        
        Returns:
            DataFrame of feature importances
        """
        if self.model is None:
            self.load_model()
        
        classifier = self.model.named_steps.get('classifier', self.model)
        
        if not hasattr(classifier, 'feature_importances_'):
            return None
        
        importances = classifier.feature_importances_
        
        # Try to get feature names
        if 'preprocessor' in self.model.named_steps:
            preprocessor = self.model.named_steps['preprocessor']
            try:
                feature_names = preprocessor.get_feature_names_out()
            except:
                feature_names = [f"feature_{i}" for i in range(len(importances))]
        else:
            feature_names = [f"feature_{i}" for i in range(len(importances))]
        
        # Ensure lengths match
        if len(feature_names) != len(importances):
            feature_names = [f"feature_{i}" for i in range(len(importances))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        if top_n:
            importance_df = importance_df.head(top_n)
        
        return importance_df
    
    def reload_model(self):
        """Reload the model from disk."""
        self.model = None
        return self.load_model()


def create_prediction_input(feature_values):
    """
    Create a properly formatted input for prediction.
    
    Args:
        feature_values: Dictionary of feature name to value
    
    Returns:
        DataFrame ready for prediction
    """
    return pd.DataFrame([feature_values])


# Example usage for Streamlit
def get_predictor(model_path=None):
    """Singleton pattern for Streamlit app."""
    if 'predictor' not in st.session_state:
        st.session_state.predictor = MentalHealthPredictor(model_path)
    return st.session_state.predictor


# Add missing import
from pathlib import Path