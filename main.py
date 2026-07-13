#!/usr/bin/env python
"""
Mental Health Survey Analysis - Main Entry Point
"""
from pathlib import Path
import sys
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.data_loader import DataLoader
from src.data_cleaner import DataCleaner


def main():
    """Main execution function"""
    print("="*60)
    print("🧠 MENTAL HEALTH SURVEY ANALYSIS")
    print("="*60)
    
    # 1. Load data
    print("\n📂 Loading datasets...")
    loader = DataLoader(raw_dir="data/raw")
    loader.load_all_datasets()
    
    # 2. Combine datasets
    print("\n🔗 Combining datasets...")
    master_df = loader.combine_datasets(source_info=True)
    
    # 3. Show summary
    loader.print_summary()
    
    # 4. Clean data
    print("\n🧹 Cleaning data...")
    cleaner = DataCleaner(master_df)
    cleaner.clean_numeric_columns()
    cleaner.clean_categorical_columns()
    cleaner.handle_missing_values(strategy='drop')
    
    clean_df = cleaner.get_clean_data()
    
    # 5. Save combined data
    output_path = loader.save_master("data/processed/unified_osmi_data.csv")
    
    # 6. Show basic statistics
    print("\n📊 Basic Statistics:")
    print(f"   Total responses: {len(clean_df):,}")
    print(f"   Columns: {len(clean_df.columns)}")
    
    if 'age' in clean_df.columns:
        print(f"   Age range: {clean_df['age'].min():.0f} - {clean_df['age'].max():.0f}")
        print(f"   Average age: {clean_df['age'].mean():.1f}")
    
    if 'survey_year' in clean_df.columns:
        print(f"   Years covered: {clean_df['survey_year'].min():.0f} - {clean_df['survey_year'].max():.0f}")
    
    # 7. Sample data
    print(f"\n📝 Sample data (first 5 rows):")
    print(clean_df.head())
    
    print(f"\n✅ Analysis complete!")
    print(f"📁 Clean data saved to: {output_path}")


if __name__ == "__main__":
    main()