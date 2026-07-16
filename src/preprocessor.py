from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd
import numpy as np


def build_preprocessor(numeric_features=None, categorical_features=None, 
                       numeric_strategy="median", categorical_strategy="constant",
                       categorical_fill_value="Not Asked"):
    """
    Build a preprocessor pipeline for numeric and categorical features.
    
    Args:
        numeric_features: List of numeric column names
        categorical_features: List of categorical column names
        numeric_strategy: Imputation strategy for numeric features
        categorical_strategy: Imputation strategy for categorical features
        categorical_fill_value: Fill value for categorical features
    
    Returns:
        ColumnTransformer preprocessor
    """
    # Default to empty lists if None
    numeric_features = numeric_features or []
    categorical_features = categorical_features or []
    
    # Numeric pipeline: impute and scale
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy=numeric_strategy)),
        ("scaler", StandardScaler())
    ])
    
    # Categorical pipeline: impute and one-hot encode
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy=categorical_strategy, fill_value=categorical_fill_value)),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    # Combine pipelines
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ])
    
    return preprocessor


def separate_features_target(df, target_column="Depression"):
    """
    Separate features and target from dataframe.
    
    Args:
        df: Input dataframe
        target_column: Name of target column
    
    Returns:
        X: Features dataframe
        y: Target series
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataframe")
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y


def get_feature_types(df, target_column="Depression", exclude_columns=None):
    """
    Identify numeric and categorical features automatically.
    
    Args:
        df: Input dataframe
        target_column: Name of target column
        exclude_columns: List of columns to exclude
    
    Returns:
        numeric_features: List of numeric column names
        categorical_features: List of categorical column names
    """
    exclude_columns = exclude_columns or []
    exclude_columns.append(target_column)
    
    # Get all feature columns (excluding target)
    features_df = df.drop(columns=exclude_columns, errors='ignore')
    
    # Identify numeric and categorical columns
    numeric_features = features_df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
    categorical_features = features_df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    
    # Also treat columns with few unique values as categorical
    for col in features_df.columns:
        if col not in numeric_features and col not in categorical_features:
            unique_ratio = len(features_df[col].unique()) / len(features_df)
            if unique_ratio < 0.05:  # Less than 5% unique values
                categorical_features.append(col)
            else:
                # Try to convert to numeric
                try:
                    pd.to_numeric(features_df[col])
                    numeric_features.append(col)
                except:
                    categorical_features.append(col)
    
    return numeric_features, categorical_features


def create_preprocessor_from_df(df, target_column="Depression", 
                                numeric_features=None, categorical_features=None):
    """
    Create a preprocessor automatically from a dataframe.
    
    Args:
        df: Input dataframe
        target_column: Name of target column
        numeric_features: Optional list of numeric features (auto-detect if None)
        categorical_features: Optional list of categorical features (auto-detect if None)
    
    Returns:
        ColumnTransformer preprocessor
    """
    if numeric_features is None or categorical_features is None:
        num_features, cat_features = get_feature_types(df, target_column)
        numeric_features = numeric_features or num_features
        categorical_features = categorical_features or cat_features
    
    return build_preprocessor(numeric_features, categorical_features)