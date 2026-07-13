"""
Data cleaning utilities for mental health survey data
"""
import pandas as pd
import numpy as np


class DataCleaner:
    """Clean and preprocess mental health survey data"""
    
    def __init__(self, df):
        self.df = df.copy()
    
    def clean_numeric_columns(self):
        """Clean and convert numeric columns"""
        # Age column
        if 'age' in self.df.columns:
            self.df['age'] = pd.to_numeric(self.df['age'], errors='coerce')
            # Remove unrealistic ages (e.g., negative or > 100)
            self.df.loc[(self.df['age'] < 0) | (self.df['age'] > 100), 'age'] = np.nan
        
        return self
    
    def clean_categorical_columns(self):
        """Standardize categorical values"""
        # Standardize 'self_employed' values
        if 'self_employed' in self.df.columns:
            self.df['self_employed'] = self.df['self_employed'].replace({
                'Yes': 1, 'yes': 1, 'TRUE': 1, 'True': 1, True: 1,
                'No': 0, 'no': 0, 'FALSE': 0, 'False': 0, False: 0
            })
        
        # Standardize 'tech_company' values
        if 'tech_company' in self.df.columns:
            self.df['tech_company'] = self.df['tech_company'].replace({
                'Yes': 1, 'yes': 1, 'TRUE': 1, 'True': 1, True: 1,
                'No': 0, 'no': 0, 'FALSE': 0, 'False': 0, False: 0
            })
        
        # Standardize 'family_history' values
        if 'family_history' in self.df.columns:
            self.df['family_history'] = self.df['family_history'].replace({
                'Yes': 1, 'yes': 1, 'TRUE': 1, 'True': 1, True: 1,
                'No': 0, 'no': 0, 'FALSE': 0, 'False': 0, False: 0
            })
        
        return self
    
    def handle_missing_values(self, strategy='drop'):
        """Handle missing values"""
        if strategy == 'drop':
            # Drop rows where all columns are NA
            self.df = self.df.dropna(how='all')
            # Drop columns that are all NA
            self.df = self.df.dropna(axis=1, how='all')
        elif strategy == 'fill':
            # Fill numeric with median, categorical with mode
            for col in self.df.columns:
                if self.df[col].dtype in ['int64', 'float64']:
                    self.df[col].fillna(self.df[col].median(), inplace=True)
                else:
                    self.df[col].fillna(self.df[col].mode()[0] if not self.df[col].mode().empty else 'Unknown', inplace=True)
        
        return self
    
    def filter_response_counts(self, min_responses=10):
        """Filter columns with at least min_responses non-null values"""
        for col in self.df.columns:
            if self.df[col].count() < min_responses:
                self.df = self.df.drop(columns=[col])
        
        return self
    
    def get_clean_data(self):
        """Return the cleaned DataFrame"""
        return self.df