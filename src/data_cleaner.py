"""
Data cleaning utilities for mental health survey data
"""
import pandas as pd
import numpy as np
import re


class DataCleaner:
    """Clean and preprocess mental health survey data"""
    
    def __init__(self, df):
        self.df = df.copy()
        self.cleaning_log = []
    
    def clean_numeric_columns(self):
        """Clean and convert numeric columns"""
        # Age column
        if 'age' in self.df.columns:
            # Remove 'age' prefix if present
            self.df['age'] = self.df['age'].astype(str).str.extract(r'(\d+)')[0]
            self.df['age'] = pd.to_numeric(self.df['age'], errors='coerce')
            # Remove unrealistic ages (e.g., negative or > 100)
            self.df.loc[(self.df['age'] < 0) | (self.df['age'] > 100), 'age'] = np.nan
            self.cleaning_log.append(f"Cleaned age column: {self.df['age'].isna().sum()} invalid values removed")
        
        return self
    
    def clean_categorical_columns(self):
        """Standardize categorical values"""
        # Standardize binary columns
        binary_columns = ['self_employed', 'tech_company', 'family_history', 
                         'currently_disordered', 'diagnosed_disorder', 
                         'past_disorder', 'sought_treatment']
        
        for col in binary_columns:
            if col in self.df.columns:
                # Convert to string first
                self.df[col] = self.df[col].astype(str).str.lower()
                # Map to 0/1
                self.df[col] = self.df[col].map({
                    'yes': 1, 'y': 1, 'true': 1, '1': 1, '1.0': 1,
                    'no': 0, 'n': 0, 'false': 0, '0': 0, '0.0': 0,
                    'nan': np.nan, 'none': np.nan, 'na': np.nan
                })
                # Convert any remaining non-numeric to NaN
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                self.cleaning_log.append(f"Cleaned {col}: {self.df[col].isna().sum()} invalid values converted")
        
        return self
    
    def handle_missing_values(self, strategy='drop', threshold=0.5):
        """
        Handle missing values.
        
        Args:
            strategy: 'drop', 'fill', or 'impute'
            threshold: For 'drop' strategy, drop columns with > threshold missing
        """
        if strategy == 'drop':
            # Drop rows with more than 50% missing
            self.df = self.df.dropna(thresh=int(len(self.df.columns) * 0.5))
            # Drop columns with more than threshold missing
            missing_ratio = self.df.isna().mean()
            columns_to_drop = missing_ratio[missing_ratio > threshold].index.tolist()
            if columns_to_drop:
                self.df = self.df.drop(columns=columns_to_drop)
                self.cleaning_log.append(f"Dropped columns: {columns_to_drop}")
        
        elif strategy == 'fill':
            # Fill numeric with median, categorical with mode
            for col in self.df.columns:
                if self.df[col].dtype in ['int64', 'float64']:
                    self.df[col].fillna(self.df[col].median(), inplace=True)
                else:
                    mode_val = self.df[col].mode()
                    if not mode_val.empty:
                        self.df[col].fillna(mode_val[0], inplace=True)
                    else:
                        self.df[col].fillna('Unknown', inplace=True)
            self.cleaning_log.append("Filled all missing values")
        
        elif strategy == 'impute':
            # Use sklearn imputation (will be handled in preprocessing)
            pass
        
        return self
    
    def filter_response_counts(self, min_responses=10):
        """Filter columns with at least min_responses non-null values"""
        dropped_cols = []
        for col in self.df.columns:
            if self.df[col].count() < min_responses:
                dropped_cols.append(col)
        
        if dropped_cols:
            self.df = self.df.drop(columns=dropped_cols)
            self.cleaning_log.append(f"Dropped columns with < {min_responses} responses: {dropped_cols}")
        
        return self
    
    def remove_duplicates(self):
        """Remove duplicate rows"""
        duplicates = len(self.df) - len(self.df.drop_duplicates())
        if duplicates > 0:
            self.df = self.df.drop_duplicates()
            self.cleaning_log.append(f"Removed {duplicates} duplicate rows")
        return self
    
    def standardize_text_columns(self):
        """Standardize text columns (strip whitespace, etc.)"""
        for col in self.df.select_dtypes(include=['object']).columns:
            # Strip whitespace
            self.df[col] = self.df[col].astype(str).str.strip()
            # Replace empty strings with NaN
            self.df[col] = self.df[col].replace('', np.nan)
            # Standardize case for categorical columns
            if self.df[col].nunique() < 20:  # Likely categorical
                self.df[col] = self.df[col].str.lower()
        
        return self
    
    def get_clean_data(self):
        """Return the cleaned DataFrame"""
        return self.df
    
    def get_cleaning_report(self):
        """Return a report of cleaning operations performed"""
        return {
            'original_shape': self.df.shape,
            'cleaning_log': self.cleaning_log,
            'missing_values': self.df.isna().sum().to_dict(),
            'dtypes': self.df.dtypes.astype(str).to_dict()
        }
    
    def clean_all(self, strategy='drop', threshold=0.5):
        """Run all cleaning steps"""
        self.clean_numeric_columns()
        self.clean_categorical_columns()
        self.standardize_text_columns()
        self.remove_duplicates()
        self.filter_response_counts()
        self.handle_missing_values(strategy=strategy, threshold=threshold)
        return self