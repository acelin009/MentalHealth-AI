import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import RESULTS_DIR


def evaluate_model(model, X_test, y_test, pos_label=None):
    """
    Evaluate a trained model on test data.
    
    Args:
        model: Trained model pipeline
        X_test: Test features
        y_test: Test target
        pos_label: Positive class label (auto-detect if None)
    
    Returns:
        Dictionary of evaluation metrics
    """
    y_pred = model.predict(X_test)
    
    # Auto-detect positive label if not provided
    if pos_label is None:
        unique_labels = np.unique(y_test)
        if len(unique_labels) == 2:
            # For binary classification, assume the "positive" class is the less frequent one
            # or the one that makes sense for the problem
            counts = pd.Series(y_test).value_counts()
            pos_label = counts.index[0]  # Most frequent class
            # You might want to adjust this logic based on your specific problem
    
    # For probability predictions (if available)
    auc_roc = None
    try:
        y_pred_proba = model.predict_proba(X_test)
        # Get probability of positive class
        if y_pred_proba.shape[1] == 2:
            # Handle binary classification
            if pos_label is not None:
                # Find the index of the positive class
                classes = model.classes_ if hasattr(model, 'classes_') else np.unique(y_test)
                pos_idx = np.where(classes == pos_label)[0]
                if len(pos_idx) > 0:
                    y_pred_proba_pos = y_pred_proba[:, pos_idx[0]]
                    y_test_binary = (y_test == pos_label).astype(int)
                    auc_roc = roc_auc_score(y_test_binary, y_pred_proba_pos)
    except Exception:
        pass
    
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
        "Recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
        "F1": f1_score(y_test, y_pred, average='weighted', zero_division=0),
        "Confusion_Matrix": confusion_matrix(y_test, y_pred),
        "Classification_Report": classification_report(y_test, y_pred, zero_division=0)
    }
    
    if auc_roc is not None:
        metrics["AUC_ROC"] = auc_roc
    
    return metrics


def plot_confusion_matrix(model, X_test, y_test, save=True, title="Confusion Matrix"):
    """
    Plot and optionally save confusion matrix.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        save: Whether to save the plot
        title: Title of the plot
    """
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=np.unique(y_test), 
                yticklabels=np.unique(y_test))
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    if save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(RESULTS_DIR / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_feature_importance(model, feature_names=None, top_n=20, save=True):
    """
    Plot feature importance for tree-based models.
    
    Args:
        model: Trained model (must have feature_importances_)
        feature_names: List of feature names or None to auto-extract
        top_n: Number of top features to display
        save: Whether to save the plot
    
    Returns:
        DataFrame of feature importances or None if not available
    """
    # Extract the classifier from pipeline
    if hasattr(model, 'named_steps'):
        classifier = model.named_steps['classifier']
    else:
        classifier = model
    
    if not hasattr(classifier, 'feature_importances_'):
        print("Model does not have feature_importances_ attribute")
        return None
    
    # Get feature importances
    importances = classifier.feature_importances_
    
    # Get feature names
    if feature_names is None:
        # Try to get from preprocessor
        if hasattr(model, 'named_steps') and 'preprocessor' in model.named_steps:
            preprocessor = model.named_steps['preprocessor']
            try:
                feature_names = preprocessor.get_feature_names_out()
            except:
                feature_names = [f"feature_{i}" for i in range(len(importances))]
        else:
            feature_names = [f"feature_{i}" for i in range(len(importances))]
    
    # Ensure lengths match
    if len(feature_names) != len(importances):
        feature_names = [f"feature_{i}" for i in range(len(importances))]
    
    # Create dataframe of importances
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # Plot top N features
    plt.figure(figsize=(10, max(6, top_n * 0.3)))
    top_features = importance_df.head(top_n)
    plt.barh(top_features['feature'], top_features['importance'])
    plt.xlabel('Feature Importance')
    plt.title(f'Top {top_n} Most Important Features')
    plt.gca().invert_yaxis()
    
    if save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(RESULTS_DIR / 'feature_importance.png', dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return importance_df


def compare_models(model_results, save=True):
    """
    Compare multiple models by their metrics.
    
    Args:
        model_results: Dictionary mapping model names to their metric dictionaries
        save: Whether to save the plot
    
    Returns:
        DataFrame with model comparisons
    """
    comparison_data = []
    
    for model_name, metrics in model_results.items():
        row = {
            'Model': model_name,
            'Accuracy': metrics.get('Accuracy', 0),
            'Precision': metrics.get('Precision', 0),
            'Recall': metrics.get('Recall', 0),
            'F1': metrics.get('F1', 0)
        }
        if 'AUC_ROC' in metrics:
            row['AUC_ROC'] = metrics['AUC_ROC']
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('F1', ascending=False)
    
    # Plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1']
    comparison_df.plot(x='Model', y=metrics_to_plot, kind='bar', ax=axes[0])
    axes[0].set_title('Performance Metrics')
    axes[0].set_ylim(0, 1)
    axes[0].legend(loc='lower right')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Plot F1 scores
    comparison_df.plot(x='Model', y='F1', kind='bar', ax=axes[1], color='green')
    axes[1].set_title('F1 Score Comparison')
    axes[1].set_ylim(0, 1)
    axes[1].tick_params(axis='x', rotation=45)
    
    if 'AUC_ROC' in comparison_df.columns:
        comparison_df.plot(x='Model', y='AUC_ROC', kind='bar', ax=axes[2], color='orange')
        axes[2].set_title('AUC-ROC Comparison')
        axes[2].set_ylim(0, 1)
        axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(RESULTS_DIR / 'model_comparison.png', dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return comparison_df