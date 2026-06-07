"""
Data collection and preprocessing module for gold price prediction.
"""
import yfinance as yf
import pandas as pd
from typing import Optional


def fetch_gold_data(
    ticker: str = "GLD",
    start: str = "2015-01-01",
    end: Optional[str] = None,
    interval: str = "1d"
) -> pd.DataFrame:
    """
    Fetch gold ETF (GLD) data using yfinance.
    
    Args:
        ticker: ETF ticker symbol (default GLD)
        start: Start date for historical data
        end: End date (default today)
        interval: Data interval
        
    Returns:
        DataFrame with OHLCV data
    """
    print(f"Fetching {ticker} data from {start} to {end or 'now'}...")
    gold = yf.Ticker(ticker)
    df = gold.history(start=start, end=end, interval=interval)
    
    # Flatten column names if multi-index
    df.columns = [col.lower() if isinstance(col, str) else '_'.join(col).lower() 
                  for col in df.columns]
    
    print(f"Retrieved {len(df)} records.")
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess the raw data.
    
    Args:
        df: Raw DataFrame from yfinance
        
    Returns:
        Cleaned DataFrame
    """
    df = df.copy()

    # Handle missing values
    df = df.ffill()
    df = df.bfill()
    
    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # Sort by date
    df = df.sort_index()
    
    # Remove duplicates
    df = df[~df.index.duplicated(keep='first')]
    
    print(f"Data shape after preprocessing: {df.shape}")
    return df


def split_data(
    df: pd.DataFrame, 
    train_ratio: float = 0.8
) -> tuple:
    """
    Split data into train and test sets (time-based split).
    
    Args:
        df: Preprocessed DataFrame
        train_ratio: Fraction for training
        
    Returns:
        Tuple of (train_df, test_df)
    """
    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
    return train_df, test_df
