import pandas as pd
import numpy as np
from datetime import datetime
import logging
import sys
import json
from pathlib import Path
from src.config import RESULTS_DIR, MODELS_DIR


def setup_logging(log_file=None, log_level=logging.INFO):
    """
    Set up logging configuration.
    
    Args:
        log_file: Optional path to log file
        log_level: Logging level
    
    Returns:
        Logger instance
    """
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=handlers
    )
    
    return logging.getLogger(__name__)


def save_results_to_csv(results_dict, filename):
    """
    Save results dictionary to CSV file.
    
    Args:
        results_dict: Dictionary of results
        filename: Output filename (without .csv extension)
    
    Returns:
        Path to saved file
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / f"{filename}.csv"
    
    # Convert any non-serializable values to strings
    serializable_dict = {}
    for key, value in results_dict.items():
        if isinstance(value, (list, dict, np.ndarray)):
            serializable_dict[key] = str(value)
        else:
            serializable_dict[key] = value
    
    df = pd.DataFrame([serializable_dict])
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")
    return output_path


def save_results_to_json(results_dict, filename):
    """
    Save results dictionary to JSON file.
    
    Args:
        results_dict: Dictionary of results
        filename: Output filename (without .json extension)
    
    Returns:
        Path to saved file
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / f"{filename}.json"
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif isinstance(obj, pd.Series):
            return obj.tolist()
        return obj
    
    serializable_dict = {}
    for key, value in results_dict.items():
        serializable_dict[key] = convert_to_serializable(value)
    
    with open(output_path, 'w') as f:
        json.dump(serializable_dict, f, indent=2, default=str)
    
    print(f"Results saved to {output_path}")
    return output_path


def load_results_from_json(filename):
    """
    Load results from JSON file.
    
    Args:
        filename: Name of the JSON file (without .json extension)
    
    Returns:
        Dictionary of results
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    input_path = RESULTS_DIR / f"{filename}.json"
    
    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")
    
    with open(input_path, 'r') as f:
        return json.load(f)


def print_model_summary(model_name, metrics):
    """
    Print a formatted summary of model performance.
    
    Args:
        model_name: Name of the model
        metrics: Dictionary of metrics
    """
    print("=" * 60)
    print(f"Model: {model_name}")
    print("=" * 60)
    
    for metric_name, value in metrics.items():
        if metric_name not in ['Confusion_Matrix', 'Classification_Report']:
            if isinstance(value, float):
                print(f"{metric_name:20s}: {value:.4f}")
            else:
                print(f"{metric_name:20s}: {value}")
    
    if 'Confusion_Matrix' in metrics:
        print("\nConfusion Matrix:")
        print(metrics['Confusion_Matrix'])
    
    if 'Classification_Report' in metrics:
        print("\nClassification Report:")
        print(metrics['Classification_Report'])
    
    print("=" * 60)


def calculate_class_weights(y_train):
    """
    Calculate class weights for imbalanced datasets.
    
    Args:
        y_train: Training target labels
    
    Returns:
        Dictionary of class weights
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    return dict(zip(classes, weights))


def create_submission_file(model, X_test, output_path=None, id_column=None):
    """
    Create a submission file for predictions.
    
    Args:
        model: Trained model
        X_test: Test features
        output_path: Output file path
        id_column: Column name for IDs (if None, use index)
    
    Returns:
        DataFrame with predictions
    """
    predictions = model.predict(X_test)
    
    # Get probabilities if available
    try:
        probabilities = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    except:
        probabilities = None
    
    # Create submission DataFrame
    if id_column is not None and id_column in X_test.columns:
        ids = X_test[id_column]
    else:
        ids = np.arange(len(predictions))
    
    submission_df = pd.DataFrame({
        'id': ids,
        'prediction': predictions
    })
    
    if probabilities is not None:
        submission_df['probability'] = probabilities
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        submission_df.to_csv(output_path, index=False)
        print(f"Submission saved to {output_path}")
    
    return submission_df


def get_timestamp():
    """Get current timestamp as string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_directory_exists(path):
    """Ensure a directory exists, creating it if necessary."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_model_metadata(model_info, model_name="best_model"):
    """
    Save model metadata to JSON file.
    
    Args:
        model_info: Dictionary of model information
        model_name: Name of the model
    
    Returns:
        Path to saved file
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = get_timestamp()
    output_path = MODELS_DIR / f"{model_name}_metadata_{timestamp}.json"
    
    model_info['timestamp'] = timestamp
    model_info['model_name'] = model_name
    
    with open(output_path, 'w') as f:
        json.dump(model_info, f, indent=2, default=str)
    
    print(f"Model metadata saved to {output_path}")
    return output_path