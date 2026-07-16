"""
Data cleaning utilities for mental health survey data
"""
import pandas as pd
import numpy as np
import re
from pathlib import Path


class DataCleaner:
    """Clean and preprocess mental health survey data"""
    
    def __init__(self, df):
        self.df = df.copy()
        self.cleaning_log = []
    
    def clean_numeric_columns(self):
        """Clean and convert numeric columns"""
        if 'age' in self.df.columns:
            self.df['age'] = self.df['age'].astype(str).str.extract(r'(\d+)')[0]
            self.df['age'] = pd.to_numeric(self.df['age'], errors='coerce')
            self.df.loc[(self.df['age'] < 0) | (self.df['age'] > 100), 'age'] = np.nan
            self.cleaning_log.append(f"Cleaned age column: {self.df['age'].isna().sum()} invalid values removed")
        
        return self
    
    def clean_categorical_columns(self):
        """Standardize categorical values"""
        binary_columns = ['self_employed', 'tech_company', 'family_history', 
                         'currently_disordered', 'diagnosed_disorder', 
                         'past_disorder', 'sought_treatment']
        
        for col in binary_columns:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(str).str.lower()
                self.df[col] = self.df[col].map({
                    'yes': 1, 'y': 1, 'true': 1, '1': 1, '1.0': 1,
                    'no': 0, 'n': 0, 'false': 0, '0': 0, '0.0': 0,
                    'nan': np.nan, 'none': np.nan, 'na': np.nan
                })
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                self.cleaning_log.append(f"Cleaned {col}: {self.df[col].isna().sum()} invalid values converted")
        
        return self
    
    def handle_missing_values(self, strategy='drop', threshold=0.5):
        """Handle missing values."""
        if strategy == 'drop':
            self.df = self.df.dropna(thresh=int(len(self.df.columns) * 0.5))
            missing_ratio = self.df.isna().mean()
            columns_to_drop = missing_ratio[missing_ratio > threshold].index.tolist()
            if columns_to_drop:
                self.df = self.df.drop(columns=columns_to_drop)
                self.cleaning_log.append(f"Dropped columns: {columns_to_drop}")
        
        elif strategy == 'fill':
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
        
        return self
    
    def filter_response_counts(self, min_responses=10):
        """Filter columns with at least min_responses non-null values."""
        dropped_cols = []
        for col in self.df.columns:
            if self.df[col].count() < min_responses:
                dropped_cols.append(col)
        
        if dropped_cols:
            self.df = self.df.drop(columns=dropped_cols)
            self.cleaning_log.append(f"Dropped columns with < {min_responses} responses: {dropped_cols}")
        
        return self
    
    def remove_duplicates(self):
        """Remove duplicate rows."""
        duplicates = len(self.df) - len(self.df.drop_duplicates())
        if duplicates > 0:
            self.df = self.df.drop_duplicates()
            self.cleaning_log.append(f"Removed {duplicates} duplicate rows")
        return self
    
    def standardize_text_columns(self):
        """Standardize text columns."""
        for col in self.df.select_dtypes(include=['object']).columns:
            self.df[col] = self.df[col].astype(str).str.strip()
            self.df[col] = self.df[col].replace('', np.nan)
            if self.df[col].nunique() < 20:
                self.df[col] = self.df[col].str.lower()
        
        return self
    
    def get_clean_data(self):
        """Return the cleaned DataFrame."""
        return self.df
    
    def get_cleaning_report(self):
        """Return a report of cleaning operations performed."""
        return {
            'original_shape': self.df.shape,
            'cleaning_log': self.cleaning_log,
            'missing_values': self.df.isna().sum().to_dict(),
            'dtypes': self.df.dtypes.astype(str).to_dict()
        }
    
    def clean_all(self, strategy='drop', threshold=0.5):
        """Run all cleaning steps."""
        self.clean_numeric_columns()
        self.clean_categorical_columns()
        self.standardize_text_columns()
        self.remove_duplicates()
        self.filter_response_counts()
        self.handle_missing_values(strategy=strategy, threshold=threshold)
        return self
    
    # ✅ NEW: Save method added
    def save_clean_data(self, output_path):
        """
        Save the cleaned DataFrame to a file.
        
        Args:
            output_path: Path to save the cleaned data (CSV file)
        
        Returns:
            Path to saved file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(output_path, index=False)
        print(f"✅ Cleaned data saved to: {output_path}")
        return output_path