from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
import joblib
from src.config import MODELS_DIR


def train_logistic(preprocessor, X_train, y_train, **kwargs):
    """
    Train a logistic regression model.
    
    Args:
        preprocessor: Preprocessor pipeline
        X_train: Training features
        y_train: Training target
        **kwargs: Additional arguments for LogisticRegression
    
    Returns:
        Pipeline with preprocessor and classifier
    """
    default_params = {
        "max_iter": 1000,
        "random_state": 42,
        "class_weight": "balanced"
    }
    default_params.update(kwargs)
    
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(**default_params))
    ])
    
    model.fit(X_train, y_train)
    return model


def train_tree(preprocessor, X_train, y_train, **kwargs):
    """
    Train a decision tree model.
    
    Args:
        preprocessor: Preprocessor pipeline
        X_train: Training features
        y_train: Training target
        **kwargs: Additional arguments for DecisionTreeClassifier
    
    Returns:
        Pipeline with preprocessor and classifier
    """
    default_params = {
        "random_state": 42,
        "class_weight": "balanced",
        "max_depth": 10,
        "min_samples_split": 10
    }
    default_params.update(kwargs)
    
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", DecisionTreeClassifier(**default_params))
    ])
    
    model.fit(X_train, y_train)
    return model


def train_forest(preprocessor, X_train, y_train, **kwargs):
    """
    Train a random forest model.
    
    Args:
        preprocessor: Preprocessor pipeline
        X_train: Training features
        y_train: Training target
        **kwargs: Additional arguments for RandomForestClassifier
    
    Returns:
        Pipeline with preprocessor and classifier
    """
    default_params = {
        "n_estimators": 300,
        "random_state": 42,
        "class_weight": "balanced",
        "max_depth": 20,
        "min_samples_split": 5
    }
    default_params.update(kwargs)
    
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(**default_params))
    ])
    
    model.fit(X_train, y_train)
    return model


def train_with_cv(preprocessor, X_train, y_train, model_type="forest", cv_folds=5, **kwargs):
    """
    Train a model with cross-validation.
    
    Args:
        preprocessor: Preprocessor pipeline
        X_train: Training features
        y_train: Training target
        model_type: Type of model ('logistic', 'tree', 'forest')
        cv_folds: Number of cross-validation folds
        **kwargs: Additional arguments for the model
    
    Returns:
        Tuple of (trained_model, cross_val_scores)
    """
    if model_type == "logistic":
        model = train_logistic(preprocessor, X_train, y_train, **kwargs)
    elif model_type == "tree":
        model = train_tree(preprocessor, X_train, y_train, **kwargs)
    elif model_type == "forest":
        model = train_forest(preprocessor, X_train, y_train, **kwargs)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='f1')
    
    return model, cv_scores


def save_model(model, model_name="best_model.pkl"):
    """
    Save trained model to disk.
    
    Args:
        model: Trained model pipeline
        model_name: Name of the model file
    """
    model_path = MODELS_DIR / model_name
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    return model_path


def load_model(model_name="best_model.pkl"):
    """
    Load a saved model from disk.
    
    Args:
        model_name: Name of the model file
    
    Returns:
        Loaded model
    """
    model_path = MODELS_DIR / model_name
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    return joblib.load(model_path)