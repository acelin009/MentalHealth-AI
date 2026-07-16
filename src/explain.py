"""
Model explanation utilities for SHAP 0.52.0+
Compatible with scikit-learn pipelines and modern SHAP API
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from src.config import RESULTS_DIR


def compute_shap_values(model, X_sample, background=None, nsamples=100):
    """
    Compute SHAP values for model explanations using SHAP 0.52.0+ API.
    
    Args:
        model: Trained model (must be a pipeline with preprocessor)
        X_sample: Sample data for explanation (DataFrame)
        background: Background data for SHAP explainer (DataFrame, optional)
        nsamples: Number of samples for KernelExplainer (default: 100)
    
    Returns:
        Dictionary containing:
        - shap_values: SHAP Explanation object
        - feature_names: List of feature names
        - expected_value: Base/expected value
        - X_transformed: Transformed feature matrix
        - explainer: SHAP explainer object
    """
    # Extract classifier and preprocessor from pipeline
    if hasattr(model, 'named_steps'):
        classifier = model.named_steps.get('classifier', model)
        preprocessor = model.named_steps.get('preprocessor', None)
    else:
        classifier = model
        preprocessor = None
    
    # Convert to DataFrame if needed
    if isinstance(X_sample, (np.ndarray, list)):
        X_sample = pd.DataFrame(X_sample)
    
    # If background is not provided, use X_sample as background
    if background is None:
        background = X_sample
    elif isinstance(background, (np.ndarray, list)):
        background = pd.DataFrame(background)
    
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
    
    # Create SHAP explainer based on model type
    try:
        # Try TreeExplainer first (for tree-based models)
        if hasattr(classifier, 'tree_') or hasattr(classifier, 'estimators_'):
            # For RandomForest, XGBoost, etc.
            explainer = shap.TreeExplainer(
                classifier,
                X_background_transformed,
                feature_names=feature_names,
                model_output='probability' if hasattr(classifier, 'predict_proba') else 'raw'
            )
            shap_values = explainer.shap_values(X_sample_transformed)
            
            # For binary classification, TreeExplainer returns a list of arrays
            # The positive class is at index 1
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            
            # Create Explanation object
            explanation = shap.Explanation(
                values=shap_values,
                base_values=explainer.expected_value if not isinstance(explainer.expected_value, list) else explainer.expected_value[1],
                data=X_sample_transformed,
                feature_names=feature_names
            )
            
            return {
                'shap_values': explanation,
                'feature_names': feature_names,
                'expected_value': explanation.base_values if hasattr(explanation, 'base_values') else explainer.expected_value,
                'X_transformed': X_sample_transformed,
                'explainer': explainer
            }
        
        # For Linear models (LogisticRegression, etc.)
        elif hasattr(classifier, 'coef_'):
            explainer = shap.LinearExplainer(
                classifier,
                X_background_transformed,
                feature_names=feature_names
            )
            shap_values = explainer.shap_values(X_sample_transformed)
            
            explanation = shap.Explanation(
                values=shap_values,
                base_values=explainer.expected_value,
                data=X_sample_transformed,
                feature_names=feature_names
            )
            
            return {
                'shap_values': explanation,
                'feature_names': feature_names,
                'expected_value': explainer.expected_value,
                'X_transformed': X_sample_transformed,
                'explainer': explainer
            }
        
        # Fallback to KernelExplainer
        else:
            # For KernelExplainer, we need a prediction function
            if hasattr(classifier, 'predict_proba'):
                def predict_fn(x):
                    return classifier.predict_proba(x)[:, 1]  # Positive class probability
            else:
                def predict_fn(x):
                    return classifier.predict(x)
            
            # Sample background if too large
            if len(X_background_transformed) > 100:
                idx = np.random.choice(len(X_background_transformed), 100, replace=False)
                X_background_transformed = X_background_transformed[idx]
            
            explainer = shap.KernelExplainer(
                predict_fn,
                X_background_transformed,
                feature_names=feature_names
            )
            
            shap_values = explainer.shap_values(X_sample_transformed, nsamples=nsamples)
            
            explanation = shap.Explanation(
                values=shap_values,
                base_values=explainer.expected_value,
                data=X_sample_transformed,
                feature_names=feature_names
            )
            
            return {
                'shap_values': explanation,
                'feature_names': feature_names,
                'expected_value': explainer.expected_value,
                'X_transformed': X_sample_transformed,
                'explainer': explainer
            }
            
    except Exception as e:
        print(f"Error computing SHAP values: {e}")
        print("Falling back to KernelExplainer...")
        
        # Fallback to KernelExplainer
        if hasattr(classifier, 'predict_proba'):
            def predict_fn(x):
                return classifier.predict_proba(x)[:, 1]
        else:
            def predict_fn(x):
                return classifier.predict(x)
        
        # Sample background if too large
        if len(X_background_transformed) > 100:
            idx = np.random.choice(len(X_background_transformed), 100, replace=False)
            X_background_transformed = X_background_transformed[idx]
        
        explainer = shap.KernelExplainer(
            predict_fn,
            X_background_transformed,
            feature_names=feature_names
        )
        
        shap_values = explainer.shap_values(X_sample_transformed, nsamples=nsamples)
        
        explanation = shap.Explanation(
            values=shap_values,
            base_values=explainer.expected_value,
            data=X_sample_transformed,
            feature_names=feature_names
        )
        
        return {
            'shap_values': explanation,
            'feature_names': feature_names,
            'expected_value': explainer.expected_value,
            'X_transformed': X_sample_transformed,
            'explainer': explainer
        }


def plot_shap_summary(shap_result, X_sample=None, feature_names=None, 
                      plot_type='bar', max_display=20, save=True, 
                      title="SHAP Summary Plot"):
    """
    Plot SHAP summary plot using SHAP 0.52.0+ API.
    
    Args:
        shap_result: Output from compute_shap_values() (dictionary or Explanation object)
        X_sample: Sample data (optional)
        feature_names: Feature names (optional)
        plot_type: 'bar' or 'dot' (default: 'bar')
        max_display: Maximum number of features to display
        save: Whether to save the plot
        title: Title of the plot
    """
    # Extract shap_values from dictionary if needed
    if isinstance(shap_result, dict):
        shap_values = shap_result['shap_values']
        if feature_names is None:
            feature_names = shap_result.get('feature_names')
        if X_sample is None:
            X_sample = shap_result.get('X_transformed')
    else:
        shap_values = shap_result
    
    if shap_values is None:
        print("No SHAP values to plot")
        return
    
    # If shap_values is a list (old API), convert to Explanation
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Use positive class
        # Convert to Explanation object
        shap_values = shap.Explanation(
            values=shap_values,
            base_values=0,  # Placeholder
            data=X_sample if X_sample is not None else np.zeros_like(shap_values),
            feature_names=feature_names
        )
    
    # Create summary plot
    plt.figure(figsize=(12, max(6, max_display * 0.4)))
    
    if plot_type == 'bar':
        shap.plots.bar(shap_values, max_display=max_display, show=False)
    else:
        shap.plots.beeswarm(shap_values, max_display=max_display, show=False)
    
    plt.title(title)
    
    if save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(RESULTS_DIR / f'shap_{plot_type}_plot.png', dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_shap_waterfall(shap_result, index=0, feature_names=None, 
                        max_display=15, save=True):
    """
    Plot SHAP waterfall plot for a single prediction using SHAP 0.52.0+ API.
    
    Args:
        shap_result: Output from compute_shap_values() (dictionary or Explanation object)
        index: Index of sample to explain
        feature_names: Feature names (optional)
        max_display: Maximum number of features to display
        save: Whether to save the plot
    """
    # Extract shap_values from dictionary if needed
    if isinstance(shap_result, dict):
        shap_values = shap_result['shap_values']
        if feature_names is None:
            feature_names = shap_result.get('feature_names')
    else:
        shap_values = shap_result
    
    if shap_values is None:
        print("No SHAP values to plot")
        return
    
    # If shap_values is a list (old API), convert to Explanation
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Use positive class
        shap_values = shap.Explanation(
            values=shap_values,
            base_values=0,
            data=np.zeros_like(shap_values),
            feature_names=feature_names
        )
    
    # Create waterfall plot
    plt.figure(figsize=(12, max(6, max_display * 0.4)))
    
    try:
        shap.plots.waterfall(shap_values[index], max_display=max_display, show=False)
    except Exception as e:
        print(f"Error creating waterfall plot: {e}")
        print("Falling back to bar plot...")
        # Fallback to bar plot for the individual prediction
        shap.plots.bar(shap_values[index], max_display=max_display, show=False)
    
    if save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(RESULTS_DIR / 'shap_waterfall.png', dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_shap_force(shap_result, index=0, save=True, **kwargs):
    """
    Plot SHAP force plot for a single prediction.
    
    Args:
        shap_result: Output from compute_shap_values()
        index: Index of sample to explain
        save: Whether to save the plot
        **kwargs: Additional arguments passed to shap.plots.force
    """
    # Extract shap_values from dictionary if needed
    if isinstance(shap_result, dict):
        shap_values = shap_result['shap_values']
        expected_value = shap_result.get('expected_value')
        X_transformed = shap_result.get('X_transformed')
    else:
        shap_values = shap_result
        expected_value = None
        X_transformed = None
    
    if shap_values is None:
        print("No SHAP values to plot")
        return
    
    # If shap_values is a list (old API), convert
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    # Create force plot
    if hasattr(shap_values, '__len__') and len(shap_values.shape) == 2:
        # Multiple samples
        shap_values_single = shap_values[index]
    else:
        shap_values_single = shap_values
    
    plt.figure(figsize=(15, 4))
    
    try:
        shap.force_plot(
            expected_value if expected_value is not None else 0,
            shap_values_single,
            X_transformed[index] if X_transformed is not None else None,
            **kwargs
        )
    except Exception as e:
        print(f"Error creating force plot: {e}")
        print("Using waterfall plot instead...")
        shap.plots.waterfall(shap_values[index])
    
    if save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(RESULTS_DIR / 'shap_force.png', dpi=300, bbox_inches='tight')
    
    plt.show()


def compute_permutation_importance(model, X_test, y_test, scoring='f1', 
                                   n_repeats=10, save=True):
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
    plt.barh(top_features['feature'], top_features['importance'],
             xerr=top_features['std'] if 'std' in top_features else None)
    plt.xlabel(f'Permutation Importance ({scoring})')
    plt.title(f'Top {top_n} Features by Permutation Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    if save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(RESULTS_DIR / 'permutation_importance.png', dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return importance_df


def explain_prediction(model, data_point, shap_result=None, top_n=5):
    """
    Generate a natural language explanation for a single prediction.
    
    Args:
        model: Trained model
        data_point: Single data point to explain (dict, list, or DataFrame)
        shap_result: Pre-computed SHAP result from compute_shap_values() (optional)
        top_n: Number of top features to include in explanation
    
    Returns:
        Dictionary with explanation components
    """
    # Ensure data_point is a DataFrame
    if isinstance(data_point, dict):
        data_point = pd.DataFrame([data_point])
    elif isinstance(data_point, list):
        data_point = pd.DataFrame([data_point])
    elif not isinstance(data_point, pd.DataFrame):
        data_point = pd.DataFrame([data_point])
    
    # Get prediction
    prediction = model.predict(data_point)[0]
    
    # Get prediction probability
    try:
        proba = model.predict_proba(data_point)[0]
        # For binary classification, get probability of positive class
        if len(proba) == 2:
            proba_positive = proba[1]
            proba_negative = proba[0]
        else:
            proba_positive = proba[0]
            proba_negative = None
    except:
        proba_positive = None
        proba_negative = None
    
    # Get SHAP values if not provided
    top_features = []
    if shap_result is None:
        # Compute SHAP values
        shap_dict = compute_shap_values(model, data_point)
        shap_values = shap_dict.get('shap_values')
        feature_names = shap_dict.get('feature_names')
        
        if shap_values is not None and feature_names is not None:
            # Extract SHAP values for this prediction
            if isinstance(shap_values, shap.Explanation):
                shap_vals = shap_values.values
            elif isinstance(shap_values, list):
                shap_vals = shap_values[1]  # Positive class
            else:
                shap_vals = shap_values
            
            if len(shap_vals.shape) == 2:
                shap_vals = shap_vals[0]  # First sample
            
            # Sort by absolute SHAP value
            feature_contributions = sorted(
                zip(feature_names, shap_vals),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:top_n]
            
            top_features = [
                {'feature': name, 'impact': float(value), 
                 'direction': 'increases' if value > 0 else 'decreases'}
                for name, value in feature_contributions
            ]
    else:
        # Use provided SHAP result
        if isinstance(shap_result, dict):
            shap_values = shap_result.get('shap_values')
            feature_names = shap_result.get('feature_names')
            
            if shap_values is not None and feature_names is not None:
                if isinstance(shap_values, shap.Explanation):
                    shap_vals = shap_values.values
                elif isinstance(shap_values, list):
                    shap_vals = shap_values[1]
                else:
                    shap_vals = shap_values
                
                if len(shap_vals.shape) == 2:
                    shap_vals = shap_vals[0]
                
                feature_contributions = sorted(
                    zip(feature_names, shap_vals),
                    key=lambda x: abs(x[1]),
                    reverse=True
                )[:top_n]
                
                top_features = [
                    {'feature': name, 'impact': float(value),
                     'direction': 'increases' if value > 0 else 'decreases'}
                    for name, value in feature_contributions
                ]
    
    # Create natural language explanation
    explanation_text = f"The model predicts: {prediction}"
    if proba_positive is not None:
        explanation_text += f" (confidence: {proba_positive:.1%})"
    
    if top_features:
        explanation_text += "\n\nTop contributing factors:"
        for i, feat in enumerate(top_features, 1):
            direction = "increases" if feat['direction'] == 'increases' else "decreases"
            explanation_text += f"\n{i}. {feat['feature']}: {direction} likelihood by {abs(feat['impact']):.3f}"
    
    # Create explanation dictionary
    explanation = {
        'prediction': str(prediction),
        'probability': proba_positive,
        'probability_negative': proba_negative,
        'top_features': top_features,
        'confidence': proba_positive if proba_positive is not None else None,
        'explanation_text': explanation_text
    }
    
    return explanation


def compare_explanations(model, X_sample, feature_names=None, top_n=10, save=True):
    """
    Create a comprehensive comparison of different explanation methods.
    
    Args:
        model: Trained model
        X_sample: Sample data
        feature_names: Feature names (optional)
        top_n: Number of top features to display
        save: Whether to save plots
    
    Returns:
        Dictionary with all explanation results
    """
    results = {}
    
    # 1. Feature Importance (if available)
    if hasattr(model, 'named_steps'):
        classifier = model.named_steps.get('classifier', model)
    else:
        classifier = model
    
    if hasattr(classifier, 'feature_importances_'):
        importances = classifier.feature_importances_
        
        if feature_names is None and hasattr(model, 'named_steps'):
            preprocessor = model.named_steps.get('preprocessor')
            if preprocessor is not None:
                try:
                    feature_names = preprocessor.get_feature_names_out()
                except:
                    feature_names = [f"feature_{i}" for i in range(len(importances))]
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importances))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        results['feature_importance'] = importance_df.head(top_n)
    
    # 2. SHAP values
    try:
        shap_dict = compute_shap_values(model, X_sample)
        results['shap'] = shap_dict
        
        # Plot SHAP summary
        plot_shap_summary(shap_dict, save=save, title="SHAP Summary - All Features")
        
        # Plot SHAP waterfall for first sample
        plot_shap_waterfall(shap_dict, index=0, save=save)
        
    except Exception as e:
        print(f"SHAP computation failed: {e}")
        results['shap'] = None
    
    # 3. Permutation Importance
    try:
        # Use the same X_sample for permutation importance
        # Note: We need y_test for this, so this is just a placeholder
        # You should call this separately with your test data
        pass
    except Exception as e:
        print(f"Permutation importance computation failed: {e}")
    
    return results