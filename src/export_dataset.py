"""
Export Gold Price Dataset for sharing with lecturer.
Fetches GLD ETF data from Yahoo Finance and saves as CSV files.
"""

import yfinance as yf
import pandas as pd
from src.data_module import fetch_gold_data, preprocess_data
from src.feature_engineering import add_technical_indicators


def export_raw_data():
    """Export raw OHLCV data from Yahoo Finance."""
    print("=" * 60)
    print("Option 1: Raw GLD ETF Data (OHLCV)")
    print("=" * 60)

    gold_data = yf.download("GLD", start="2015-01-01")

    filename = "GLD_raw_data.csv"
    gold_data.to_csv(filename)

    print(f"\nSaved to: {filename}")
    print(f"Shape: {gold_data.shape}")
    print(f"Date range: {gold_data.index[0]} to {gold_data.index[-1]}")

    # Handle multi-level columns (yfinance returns tuples)
    if isinstance(gold_data.columns, pd.MultiIndex):
        print(f"Columns: {', '.join([str(c) for c in gold_data.columns])}")
    else:
        print(f"Columns: {', '.join(gold_data.columns.astype(str))}")

    print("\nFirst 5 rows:")
    print(gold_data.head())
    print("\nLast 5 rows:")
    print(gold_data.tail())

    return filename


def export_processed_data():
    """Export data after preprocessing (cleaned)."""
    print("\n" + "=" * 60)
    print("Option 2: Processed GLD Data (Cleaned)")
    print("=" * 60)

    raw_data = fetch_gold_data(start="2015-01-01")
    clean_data = preprocess_data(raw_data)

    filename = "GLD_processed.csv"
    clean_data.to_csv(filename)

    print(f"\nSaved to: {filename}")
    print(f"Shape: {clean_data.shape}")
    print(f"Date range: {clean_data.index[0]} to {clean_data.index[-1]}")
    print("\nFirst 5 rows:")
    print(clean_data.head())

    return filename


def export_featured_data():
    """Export data with all 31 engineered features."""
    print("\n" + "=" * 60)
    print("Option 3: Full Dataset with 31 Technical Features")
    print("=" * 60)

    raw_data = fetch_gold_data(start="2015-01-01")
    clean_data = preprocess_data(raw_data)
    featured_data = add_technical_indicators(clean_data)

    filename = "GLD_with_features.csv"
    featured_data.to_csv(filename)

    print(f"\nSaved to: {filename}")
    print(f"Shape: {featured_data.shape}")
    print(f"Date range: {featured_data.index[0]} to {featured_data.index[-1]}")
    print(f"Number of features: {featured_data.shape[1]}")
    print("\nColumns:")
    for i, col in enumerate(featured_data.columns, 1):
        print(f"  {i}. {col}")
    print("\nFirst 5 rows:")
    print(featured_data.head())

    return filename


if __name__ == "__main__":
    print("\n📊 Gold Price Dataset Export Tool")
    print("=" * 60)
    print("\nThis script will create 3 CSV files:")
    print("  1. GLD_raw_data.csv - Raw OHLCV data from Yahoo Finance")
    print("  2. GLD_processed.csv - Cleaned data (no missing values)")
    print("  3. GLD_with_features.csv - Full dataset with 31 features")
    print("\nWhich version does your lecturer need?")
    print("(Usually Option 1 or Option 3 is sufficient)")
    print("\n" + "=" * 60)
    print("\nStarting export...\n")

    # Export all three versions
    file1 = export_raw_data()
    file2 = export_processed_data()
    file3 = export_featured_data()

    print("\n" + "=" * 60)
    print("✅ EXPORT COMPLETE")
    print("=" * 60)
    print("\nFiles created:")
    print(f"  1. {file1}")
    print(f"  2. {file2}")
    print(f"  3. {file3}")
    print("\nYou can now send these files to your lecturer.")
    print("Recommended: GLD_with_features.csv (most complete dataset)")
