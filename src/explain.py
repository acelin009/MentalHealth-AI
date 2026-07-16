import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from src.config import RESULTS_DIR


def compute_shap_values(model, X_sample, background=None):
    """
    Compute SHAP values for model explanations.
    
    Args:
        model: Trained model (must be a pipeline with preprocessor)
        X_sample: Sample data for explanation
        background: Background data for SHAP explainer
    
    Returns:
        SHAP explainer and values
    """
    # Extract the classifier from pipeline
    if hasattr(model, 'named_steps'):
        classifier = model.named_steps.get('classifier', model)
        preprocessor = model.named_steps.get('preprocessor', None)
    else:
        classifier = model
        preprocessor = None
    
    # If background is not provided, use X_sample as background
    if background is None:
        background = X_sample
    
    # Transform data if preprocessor is available
    if preprocessor is not None:
        X_background_transformed = preprocessor.transform(background)
        X_sample_transformed = preprocessor.transform(X_sample)
        try:
            feature_names = preprocessor.get_feature_names_out()
        except:
            feature_names = [f"feature_{i}" for i in range(X_sample_transformed.shape[1])]
    else:
        X_background_transformed = background
        X_sample_transformed = X_sample
        feature_names = X_sample.columns.tolist() if hasattr(X_sample, 'columns') else [f"feature_{i}" for i in range(X_sample.shape[1])]
    
    # Create SHAP explainer
    try:
        if hasattr(classifier, 'predict_proba'):
            # For classification models
            explainer = shap.TreeExplainer(classifier, X_background_transformed, feature_names=feature_names)
            shap_values = explainer.shap_values(X_sample_transformed)
        else:
            # Fallback
            explainer = shap.KernelExplainer(classifier.predict, X_background_transformed, feature_names=feature_names)
            shap_values = explainer.shap_values(X_sample_transformed)
    except Exception as e:
        print(f"Error computing SHAP values: {e}")
        return None, None, feature_names
    
    return explainer, shap_values, feature_names


def plot_shap_summary(shap_values, feature_names=None, X_sample=None, save=True, title="SHAP Summary Plot"):
    """
    Plot SHAP summary plot.
    
    Args:
        shap_values: SHAP values from compute_shap_values
        feature_names: Feature names
        X_sample: Sample data
        save: Whether to save the plot
        title: Title of the plot
    """
    if shap_values is None:
        print("No SHAP values to plot")
        return
    
    plt.figure(figsize=(12, 8))
    
    # Handle binary classification
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Use positive class
    
    # Create summary plot
    if X_sample is not None and feature_names is not None:
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
    elif feature_names is not None:
        shap.summary_plot(shap_values, feature_names=feature_names, show=False)
    else:
        shap.summary_plot(shap_values, show=False)
    
    plt.title(title)
    
    if save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(RESULTS_DIR / 'shap_summary.png', dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_shap_waterfall(shap_values, feature_names=None, X_sample=None, index=0, save=True):
    """
    Plot SHAP waterfall plot for a single prediction.
    
    Args:
        shap_values: SHAP values from compute_shap_values
        feature_names: Feature names
        X_sample: Sample data
        index: Index of sample to explain
        save: Whether to save the plot
    """
    if shap_values is None:
        print("No SHAP values to plot")
        return
    
    plt.figure(figsize=(12, 6))
    
    # Handle binary classification
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    # Ensure shap_values is a 2D array
    if len(shap_values.shape) == 1:
        shap_values = shap_values.reshape(1, -1)
    
    # Get the expected value (base value)
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(shap_values.shape[1])]
    
    try:
        shap.waterfall_plot(shap_values[index], feature_names=feature_names, show=False)
    except Exception as e:
        print(f"Error creating waterfall plot: {e}")
        print("Falling back to bar plot...")
        shap.bar_plot(shap_values[index], feature_names=feature_names, show=False)
    
    if save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(RESULTS_DIR / 'shap_waterfall.png', dpi=300, bbox_inches='tight')
    
    plt.show()


def compute_permutation_importance(model, X_test, y_test, scoring='f1', n_repeats=10, save=True):
    """
    Compute permutation importance for model features.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        scoring: Scoring metric
        n_repeats: Number of permutation repeats
        save: Whether to save the plot
    
    Returns:
        DataFrame with permutation importance scores
    """
    from sklearn.inspection import permutation_importance
    
    # Extract classifier if using pipeline
    if hasattr(model, 'named_steps'):
        classifier = model.named_steps.get('classifier', model)
        preprocessor = model.named_steps.get('preprocessor', None)
        
        # Transform X_test if preprocessor exists
        if preprocessor is not None:
            X_test_transformed = preprocessor.transform(X_test)
        else:
            X_test_transformed = X_test
    else:
        classifier = model
        X_test_transformed = X_test
    
    # Compute permutation importance
    result = permutation_importance(
        classifier, X_test_transformed, y_test,
        n_repeats=n_repeats, random_state=42, scoring=scoring
    )
    
    # Get feature names
    if hasattr(model, 'named_steps') and 'preprocessor' in model.named_steps:
        try:
            feature_names = model.named_steps['preprocessor'].get_feature_names_out()
        except:
            feature_names = [f"feature_{i}" for i in range(len(result.importances_mean))]
    else:
        feature_names = [f"feature_{i}" for i in range(len(result.importances_mean))]
    
    # Create importance dataframe
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': result.importances_mean,
        'std': result.importances_std
    }).sort_values('importance', ascending=False)
    
    # Plot
    plt.figure(figsize=(10, max(6, len(importance_df) * 0.3)))
    top_n = min(20, len(importance_df))
    top_features = importance_df.head(top_n)
    plt.barh(top_features['feature'], top_features['importance'])
    plt.xlabel(f'Permutation Importance ({scoring})')
    plt.title(f'Top {top_n} Features by Permutation Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    if save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(RESULTS_DIR / 'permutation_importance.png', dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return importance_df


def explain_prediction(model, data_point, feature_names=None, shap_values=None):
    """
    Generate a natural language explanation for a single prediction.
    
    Args:
        model: Trained model
        data_point: Single data point to explain
        feature_names: Feature names
        shap_values: Pre-computed SHAP values (optional)
    
    Returns:
        Dictionary with explanation components
    """
    # Ensure data_point is a DataFrame
    if isinstance(data_point, dict):
        data_point = pd.DataFrame([data_point])
    elif not isinstance(data_point, pd.DataFrame):
        data_point = pd.DataFrame([data_point])
    
    # Get prediction
    prediction = model.predict(data_point)[0]
    
    # Get prediction probability
    try:
        proba = model.predict_proba(data_point)[0]
        proba_positive = proba[1] if len(proba) > 1 else proba[0]
    except:
        proba_positive = None
    
    # Get SHAP values if not provided
    if shap_values is None:
        _, shap_values, _ = compute_shap_values(model, data_point)
    
    # Extract top contributing features
    if shap_values is not None:
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use positive class
        
        # For a single data point
        if len(shap_values.shape) == 1:
            shap_vals = shap_values
        else:
            shap_vals = shap_values[0]
        
        # Get feature names
        if feature_names is None:
            if hasattr(model, 'named_steps') and 'preprocessor' in model.named_steps:
                try:
                    feature_names = model.named_steps['preprocessor'].get_feature_names_out()
                except:
                    feature_names = [f"feature_{i}" for i in range(len(shap_vals))]
            else:
                feature_names = [f"feature_{i}" for i in range(len(shap_vals))]
        
        # Sort by absolute SHAP value
        feature_contributions = sorted(
            zip(feature_names, shap_vals),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:5]
        
        top_features = [
            {'feature': name, 'impact': float(value)}
            for name, value in feature_contributions
        ]
    else:
        top_features = []
    
    # Create explanation
    explanation = {
        'prediction': str(prediction),
        'probability': proba_positive,
        'top_features': top_features,
        'confidence': proba_positive if proba_positive is not None else None
    }
    
    return explanation