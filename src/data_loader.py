"""
Data loader for mental health survey datasets
"""
import re
from pathlib import Path
import pandas as pd
from .schema import COLUMN_MAP, UNIFIED_SCHEMA, DATASET_PATTERNS
from src.config import (
    MODELING_DATASET,
    ENGINEERED_DATASET
)

def load_ml_dataset():
    return pd.read_csv(MODELING_DATASET)


def load_engineered_dataset():
    return pd.read_csv(ENGINEERED_DATASET)

class DataLoader:
    """Load and combine mental health survey datasets"""
    
    def __init__(self, raw_dir="data/raw"):
        self.raw_dir = Path(raw_dir)
        self.datasets = {}
        self.master_df = None
        self.schema = UNIFIED_SCHEMA
        self.column_map = COLUMN_MAP
        
    def load_all_datasets(self):
        """Load all CSV files from subdirectories"""
        print(f"Loading datasets from: {self.raw_dir}")
        
        if not self.raw_dir.exists():
            raise FileNotFoundError(f"Directory not found: {self.raw_dir}")
        
        for folder in self.raw_dir.iterdir():
            if folder.is_dir():
                csv_files = list(folder.glob("*.csv"))
                if csv_files:
                    try:
                        # Load the first CSV file in the folder
                        file_path = csv_files[0]
                        dataset_name = folder.name
                        df = pd.read_csv(file_path)
                        self.datasets[dataset_name] = {
                            'df': df,
                            'file_path': file_path,
                            'rows': len(df),
                            'columns': len(df.columns)
                        }
                        print(f"✅ Loaded '{dataset_name}': {len(df)} rows, {len(df.columns)} columns")
                    except Exception as e:
                        print(f"❌ Error loading {folder.name}: {e}")
        
        print(f"\n📊 Loaded {len(self.datasets)} datasets")
        return self.datasets
    
    def clean_column_name(self, col):
        """Remove special characters from column names"""
        col = col.strip()
        col = re.sub(r'[\*<].*?[\*>]', '', col).strip()
        return col
    
    def map_columns(self, df):
        """Map DataFrame columns to unified schema"""
        # Clean column names
        df.columns = [self.clean_column_name(col) for col in df.columns]
        
        # Map columns
        mapped_data = {}
        for col in df.columns:
            if col in self.column_map:
                mapped_data[self.column_map[col]] = df[col]
        
        return pd.DataFrame(mapped_data)
    
    def combine_datasets(self, source_info=True):
        """Combine all loaded datasets into a master DataFrame"""
        if not self.datasets:
            raise ValueError("No datasets loaded. Call load_all_datasets() first.")
        
        master_dfs = []
        
        for dataset_name, info in self.datasets.items():
            df = info['df']
            
            # Map columns to unified schema
            mapped_df = self.map_columns(df)
            
            # Add source information
            if source_info:
                mapped_df['source_file'] = dataset_name
                # Try to extract year from dataset name
                year_match = re.search(r'(\d{4})', dataset_name)
                if year_match:
                    mapped_df['survey_year'] = int(year_match.group(1))
                else:
                    mapped_df['survey_year'] = None
            
            master_dfs.append(mapped_df)
        
        # Combine all DataFrames
        self.master_df = pd.concat(master_dfs, ignore_index=True, sort=False)
        
        # Ensure all schema columns exist
        for col in UNIFIED_SCHEMA:
            if col not in self.master_df.columns:
                self.master_df[col] = None
        
        # Reorder columns
        final_columns = []
        if source_info:
            final_columns.extend(['source_file', 'survey_year'])
        final_columns.extend(UNIFIED_SCHEMA)
        
        # Only include columns that exist in the DataFrame
        final_columns = [col for col in final_columns if col in self.master_df.columns]
        
        self.master_df = self.master_df[final_columns]
        
        print(f"\n✅ Combined dataset: {len(self.master_df)} rows, {len(self.master_df.columns)} columns")
        
        return self.master_df
    
    def save_master(self, output_path="data/processed/unified_osmi_data.csv"):
        """Save the master DataFrame to CSV"""
        if self.master_df is None:
            raise ValueError("No master DataFrame to save. Run combine_datasets() first.")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.master_df.to_csv(output_path, index=False)
        print(f"💾 Saved unified dataset to: {output_path}")
        return output_path
    
    def get_dataset_info(self):
        """Return information about loaded datasets"""
        info = {}
        for name, data in self.datasets.items():
            info[name] = {
                'rows': data['rows'],
                'columns': data['columns'],
                'file': data['file_path'].name
            }
        return info
    
    def print_summary(self):
        """Print a summary of loaded datasets"""
        print("\n" + "="*60)
        print("📊 DATASET SUMMARY")
        print("="*60)
        
        for name, info in self.datasets.items():
            print(f"\n📁 {name}")
            print(f"   File: {info['file_path'].name}")
            print(f"   Rows: {info['rows']:,}")
            print(f"   Columns: {info['columns']}")
        
        if self.master_df is not None:
            print(f"\n📋 Combined Dataset")
            print(f"   Total Rows: {len(self.master_df):,}")
            print(f"   Total Columns: {len(self.master_df.columns)}")
            print(f"   Source Files: {len(self.datasets)}")
            
            # Show sample columns
            print(f"\n   Sample Columns:")
            for col in self.master_df.columns[:10]:
                print(f"     - {col}")
            if len(self.master_df.columns) > 10:
                print(f"     ... and {len(self.master_df.columns) - 10} more")