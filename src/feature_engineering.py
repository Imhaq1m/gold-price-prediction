"""
Feature engineering module for gold price prediction.
Creates technical indicators, lag features, and prepares sequences for LSTM.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators to the dataset.

    Args:
        df: DataFrame with OHLCV data

    Returns:
        DataFrame with additional technical indicator columns
    """
    df = df.copy()

    # Moving Averages
    df["sma_10"] = df["close"].rolling(window=10).mean()
    df["sma_20"] = df["close"].rolling(window=20).mean()
    df["sma_50"] = df["close"].rolling(window=50).mean()
    df["ema_10"] = df["close"].ewm(span=10, adjust=False).mean()
    df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()

    # Moving Average Crossovers
    df["sma_cross_10_20"] = df["sma_10"] - df["sma_20"]
    df["sma_cross_20_50"] = df["sma_20"] - df["sma_50"]

    # Price returns
    df["returns"] = df["close"].pct_change()
    df["log_returns"] = np.log(df["close"] / df["close"].shift(1))

    # Volatility (rolling standard deviation of returns)
    df["volatility_10"] = df["returns"].rolling(window=10).std()
    df["volatility_20"] = df["returns"].rolling(window=20).std()

    # RSI (Relative Strength Index)
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # MACD (Moving Average Convergence Divergence)
    df["macd"] = df["ema_12"] = (
        df["close"].ewm(span=12, adjust=False).mean()
        - df["close"].ewm(span=26, adjust=False).mean()
    )
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # Bollinger Bands
    df["bb_middle"] = df["close"].rolling(window=20).mean()
    bb_std = df["close"].rolling(window=20).std()
    df["bb_upper"] = df["bb_middle"] + (bb_std * 2)
    df["bb_lower"] = df["bb_middle"] - (bb_std * 2)
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]

    # Price position relative to Bollinger Bands
    df["bb_position"] = (df["close"] - df["bb_lower"]) / (
        df["bb_upper"] - df["bb_lower"]
    )

    # High-Low range
    df["hl_range"] = (df["high"] - df["low"]) / df["close"]

    # Open-Close range
    df["oc_range"] = (df["close"] - df["open"]) / df["open"]

    # Volume features
    if "volume" in df.columns:
        df["volume_sma_10"] = df["volume"].rolling(window=10).mean()
        df["volume_ratio"] = df["volume"] / df["volume_sma_10"]

    # Lag features (previous days' prices and returns)
    for lag in [1, 2, 3, 5, 10]:
        df[f"close_lag_{lag}"] = df["close"].shift(lag)
        df[f"returns_lag_{lag}"] = df["returns"].shift(lag)

    # Rolling statistics
    for window in [5, 10, 20]:
        df[f"close_mean_{window}"] = df["close"].rolling(window=window).mean()
        df[f"close_std_{window}"] = df["close"].rolling(window=window).std()
        df[f"close_min_{window}"] = df["close"].rolling(window=window).min()
        df[f"close_max_{window}"] = df["close"].rolling(window=window).max()

    # Time-based features
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    df["quarter"] = df.index.quarter

    # Target variable: next day's close price
    df["target"] = df["close"].shift(-1)

    # Drop NaN values created by rolling calculations
    df = df.dropna()

    print(f"Features added. Shape: {df.shape}")
    return df


def prepare_features(
    df: pd.DataFrame, feature_columns: List[str] = None
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Select and prepare feature columns.

    Args:
        df: DataFrame with all features
        feature_columns: List of columns to use as features

    Returns:
        Tuple of (feature_df, feature_names)
    """
    if feature_columns is None:
        # Default feature set
        feature_columns = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "sma_10",
            "sma_20",
            "ema_10",
            "ema_20",
            "returns",
            "volatility_10",
            "volatility_20",
            "rsi_14",
            "macd",
            "macd_signal",
            "macd_hist",
            "bb_width",
            "bb_position",
            "hl_range",
            "oc_range",
            "volume_ratio",
            "close_lag_1",
            "close_lag_2",
            "close_lag_3",
            "returns_lag_1",
            "returns_lag_2",
            "close_mean_5",
            "close_mean_10",
            "close_std_10",
            "day_of_week",
            "month",
        ]
        # Filter to only existing columns
        feature_columns = [col for col in feature_columns if col in df.columns]

    return df[feature_columns], feature_columns


def create_sequences(
    data: np.ndarray,
    target: np.ndarray,
    seq_length: int = 60,
    forecast_horizon: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for LSTM input. Supports direct multi-step forecasting.

    Args:
        data: Feature array, shape (n_samples, n_features)
        target: Target array, shape (n_samples,) — 1D returns
        seq_length: Length of each input sequence
        forecast_horizon: Number of future steps to predict per sample

    Returns:
        Tuple of (X_sequences, y_values)
        X shape: (num_sequences, seq_length, n_features)
        y shape: (num_sequences, forecast_horizon)
    """
    X, y = [], []
    for i in range(len(data) - seq_length - forecast_horizon + 1):
        X.append(data[i : i + seq_length])
        y.append(target[i + seq_length : i + seq_length + forecast_horizon])

    return np.array(X), np.array(y)


def scale_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: List[str],
    target_column: str = "target",
) -> Tuple:
    """
    Scale features and target using MinMaxScaler.

    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        feature_columns: List of feature column names
        target_column: Target column name

    Returns:
        Tuple of (X_train, y_train, X_test, y_test, feature_scaler, target_scaler)
    """
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler = MinMaxScaler(feature_range=(0, 1))

    # Fit and transform training data
    train_features = train_df[feature_columns].values
    train_target = train_df[[target_column]].values

    train_features_scaled = feature_scaler.fit_transform(train_features)
    train_target_scaled = target_scaler.fit_transform(train_target).ravel()

    # Transform test data
    test_features = test_df[feature_columns].values
    test_target = test_df[[target_column]].values

    test_features_scaled = feature_scaler.transform(test_features)
    test_target_scaled = target_scaler.transform(test_target).ravel()

    print(
        f"Features scaled. Shape: train={train_features_scaled.shape}, test={test_features_scaled.shape}"
    )

    return (
        train_features_scaled,
        train_target_scaled,
        test_features_scaled,
        test_target_scaled,
        feature_scaler,
        target_scaler,
    )
