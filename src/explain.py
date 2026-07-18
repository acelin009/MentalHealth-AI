"""
Model explanation utilities
Focuses on permutation importance and model interpretability
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from src.config import RESULTS_DIR


def compute_permutation_importance(model, X, y, n_repeats=10, random_state=42, scoring='f1'):
    """
    Compute permutation feature importance for a trained model.
    
    Parameters
    ----------
    model : estimator or Pipeline
        Trained machine learning model.
    X : pandas.DataFrame
        Input features.
    y : pandas.Series
        Target values.
    n_repeats : int, default=10
        Number of times each feature is shuffled.
    random_state : int, default=42
        Random seed for reproducibility.
    scoring : str, default='f1'
        Scoring metric to use for importance calculation.
    
    Returns
    -------
    pandas.DataFrame
        Feature importance scores sorted from highest to lowest.
    """
    # Extract classifier if using pipeline
    if hasattr(model, 'named_steps'):
        classifier = model.named_steps.get('classifier', model)
        preprocessor = model.named_steps.get('preprocessor', None)
        
        # Transform X if preprocessor exists
        if preprocessor is not None:
            X_transformed = preprocessor.transform(X)
            try:
                feature_names = preprocessor.get_feature_names_out()
            except:
                feature_names = [f"feature_{i}" for i in range(X_transformed.shape[1])]
        else:
            X_transformed = X
            feature_names = X.columns.tolist() if hasattr(X, 'columns') else [f"feature_{i}" for i in range(X.shape[1])]
    else:
        classifier = model
        X_transformed = X
        feature_names = X.columns.tolist() if hasattr(X, 'columns') else [f"feature_{i}" for i in range(X.shape[1])]
    
    # Compute permutation importance
    result = permutation_importance(
        classifier,
        X_transformed,
        y,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring=scoring
    )
    
    # Create importance dataframe
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': result.importances_mean,
        'std': result.importances_std
    }).sort_values('importance', ascending=False).reset_index(drop=True)
    
    return importance_df


def plot_permutation_importance(importance_df, top_n=20, save=True, title="Permutation Feature Importance"):
    """
    Plot permutation feature importance.
    
    Parameters
    ----------
    importance_df : pandas.DataFrame
        DataFrame from compute_permutation_importance()
    top_n : int, default=20
        Number of top features to display
    save : bool, default=True
        Whether to save the plot
    title : str, default="Permutation Feature Importance"
        Plot title
    
    Returns
    -------
    matplotlib.figure.Figure
        The figure object
    """
    # Get top N features
    top_features = importance_df.head(top_n)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
    
    # Horizontal bar chart with error bars
    ax.barh(top_features['feature'], top_features['importance'],
            xerr=top_features['std'] if 'std' in top_features else None,
            color='steelblue', edgecolor='navy', alpha=0.8)
    
    ax.set_xlabel('Permutation Importance')
    ax.set_title(title)
    ax.invert_yaxis()  # Highest importance at top
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(top_features['importance']):
        ax.text(v + v * 0.01, i, f'{v:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    
    if save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(RESULTS_DIR / 'permutation_importance.png', dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return fig


def plot_feature_importance_comparison(model, X, y, n_repeats=10, random_state=42, top_n=20, save=True):
    """
    Compare built-in feature importance (if available) with permutation importance.
    
    Parameters
    ----------
    model : estimator or Pipeline
        Trained machine learning model.
    X : pandas.DataFrame
        Input features.
    y : pandas.Series
        Target values.
    n_repeats : int, default=10
        Number of times each feature is shuffled.
    random_state : int, default=42
        Random seed for reproducibility.
    top_n : int, default=20
        Number of top features to display
    save : bool, default=True
        Whether to save the plot
    
    Returns
    -------
    tuple
        (builtin_importance_df, permutation_importance_df)
    """
    # Get built-in feature importance if available
    builtin_importance = None
    if hasattr(model, 'named_steps'):
        classifier = model.named_steps.get('classifier', model)
        preprocessor = model.named_steps.get('preprocessor', None)
        
        if hasattr(classifier, 'feature_importances_'):
            importances = classifier.feature_importances_
            
            if preprocessor is not None:
                try:
                    feature_names = preprocessor.get_feature_names_out()
                except:
                    feature_names = [f"feature_{i}" for i in range(len(importances))]
            else:
                feature_names = X.columns.tolist() if hasattr(X, 'columns') else [f"feature_{i}" for i in range(len(importances))]
            
            builtin_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False).reset_index(drop=True)
    
    # Compute permutation importance
    perm_importance = compute_permutation_importance(
        model, X, y, n_repeats=n_repeats, random_state=random_state
    )
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 2 if builtin_importance is not None else 1, 
                             figsize=(14 if builtin_importance is not None else 8, 
                                      max(6, top_n * 0.3)))
    
    if builtin_importance is not None:
        ax1 = axes[0]
        # Plot built-in importance
        top_builtin = builtin_importance.head(top_n)
        ax1.barh(top_builtin['feature'], top_builtin['importance'],
                 color='darkgreen', edgecolor='green', alpha=0.8)
        ax1.set_xlabel('Built-in Feature Importance')
        ax1.set_title('Built-in Feature Importance')
        ax1.invert_yaxis()
        ax1.grid(axis='x', alpha=0.3)
        
        ax2 = axes[1]
        # Plot permutation importance
        top_perm = perm_importance.head(top_n)
        ax2.barh(top_perm['feature'], top_perm['importance'],
                 xerr=top_perm['std'] if 'std' in top_perm else None,
                 color='steelblue', edgecolor='navy', alpha=0.8)
        ax2.set_xlabel('Permutation Importance')
        ax2.set_title('Permutation Importance')
        ax2.invert_yaxis()
        ax2.grid(axis='x', alpha=0.3)
        
        plt.suptitle('Feature Importance Comparison', fontsize=14, fontweight='bold')
    else:
        ax = axes if not isinstance(axes, list) else axes[0]
        # Only permutation importance
        top_perm = perm_importance.head(top_n)
        ax.barh(top_perm['feature'], top_perm['importance'],
                xerr=top_perm['std'] if 'std' in top_perm else None,
                color='steelblue', edgecolor='navy', alpha=0.8)
        ax.set_xlabel('Permutation Importance')
        ax.set_title('Permutation Feature Importance')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(RESULTS_DIR / 'feature_importance_comparison.png', dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return builtin_importance, perm_importance


def get_top_features(importance_df, top_n=10):
    """
    Get the top N most important features.
    
    Parameters
    ----------
    importance_df : pandas.DataFrame
        DataFrame from compute_permutation_importance()
    top_n : int, default=10
        Number of top features to return
    
    Returns
    -------
    pandas.DataFrame
        Top N features
    """
    return importance_df.head(top_n).copy()