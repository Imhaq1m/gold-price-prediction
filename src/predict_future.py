"""
Predict next trading day's gold price using a trained LSTM-Attention model.

Usage:
    python -m src.predict_future
    python -m src.predict_future --model models/cv_fold_1.pt
    python -m src.predict_future --days 5

Requires trained artifacts saved by main.py:
    - models/best_lstm_attention.pt  (or other checkpoint)
    - models/feature_scaler.pkl
    - models/target_scaler.pkl
    - models/bias_correction.txt
"""

import argparse
import datetime
import warnings

import joblib
import numpy as np
import pandas as pd
import torch

from src.data_module import fetch_gold_data, preprocess_data
from src.feature_engineering import add_technical_indicators, prepare_features
from src.lstm_model import LSTMAttentionModel

warnings.filterwarnings("ignore")

SEQ_LENGTH = 30

_FEATURE_COLUMNS = [
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


def load_trained_model(model_path: str) -> LSTMAttentionModel:
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    output_size = checkpoint.get("output_size", 1)
    model = LSTMAttentionModel(
        input_size=checkpoint["input_size"], output_size=output_size
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def prepare_recent_data() -> tuple:
    raw = fetch_gold_data(ticker="GLD", start="2024-01-01", interval="1d")
    raw = preprocess_data(raw)
    actual_last_date = raw.index[-1]
    actual_last_close = raw["close"].iloc[-1]

    df = add_technical_indicators(raw)
    feature_df, feature_columns = prepare_features(df)
    if "close" not in feature_columns:
        feature_columns = ["close"] + feature_columns
        feature_df = df[feature_columns]
    return df, feature_df, feature_columns, actual_last_date, actual_last_close


def next_trading_day(d: datetime.date) -> datetime.date:
    next_day = d + datetime.timedelta(days=1)
    while next_day.weekday() >= 5:
        next_day += datetime.timedelta(days=1)
    return next_day


def _recompute_features(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["sma_10"] = df["close"].rolling(10).mean()
    df["sma_20"] = df["close"].rolling(20).mean()
    df["sma_50"] = df["close"].rolling(50).mean()
    df["ema_10"] = df["close"].ewm(span=10, adjust=False).mean()
    df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["sma_cross_10_20"] = df["sma_10"] - df["sma_20"]
    df["sma_cross_20_50"] = df["sma_20"] - df["sma_50"]
    df["returns"] = df["close"].pct_change()
    df["log_returns"] = np.log(df["close"] / df["close"].shift(1))
    df["volatility_10"] = df["returns"].rolling(10).std()
    df["volatility_20"] = df["returns"].rolling(20).std()
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df["rsi_14"] = 100 - (100 / (1 + rs))
    ema_12 = df["close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema_12 - ema_26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    df["bb_middle"] = df["close"].rolling(20).mean()
    bb_std = df["close"].rolling(20).std()
    df["bb_upper"] = df["bb_middle"] + (bb_std * 2)
    df["bb_lower"] = df["bb_middle"] - (bb_std * 2)
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
    df["bb_position"] = (df["close"] - df["bb_lower"]) / (
        df["bb_upper"] - df["bb_lower"]
    )
    df["hl_range"] = (df["high"] - df["low"]) / df["close"]
    df["oc_range"] = (df["close"] - df["open"]) / df["open"]
    if "volume" in df.columns:
        df["volume_sma_10"] = df["volume"].rolling(10).mean()
        df["volume_ratio"] = df["volume"] / df["volume_sma_10"]
    for lag in [1, 2, 3, 5, 10]:
        df[f"close_lag_{lag}"] = df["close"].shift(lag)
        df[f"returns_lag_{lag}"] = df["returns"].shift(lag)
    for window in [5, 10, 20]:
        df[f"close_mean_{window}"] = df["close"].rolling(window).mean()
        df[f"close_std_{window}"] = df["close"].rolling(window).std()
        df[f"close_min_{window}"] = df["close"].rolling(window).min()
        df[f"close_max_{window}"] = df["close"].rolling(window).max()
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    df["quarter"] = df.index.quarter
    valid = df.iloc[50:].copy()
    cols = [c for c in _FEATURE_COLUMNS if c in valid.columns]
    return valid[cols]


def _build_synthetic_row(
    prev_close: float,
    predicted_close: float,
    prev_date: datetime.date,
    last_volume: float,
) -> pd.DataFrame:
    next_date = next_trading_day(prev_date)
    open_val = prev_close
    high = max(open_val, predicted_close)
    low = min(open_val, predicted_close)
    row = pd.DataFrame(
        {
            "open": open_val,
            "high": high,
            "low": low,
            "close": predicted_close,
            "volume": last_volume,
        },
        index=[pd.Timestamp(next_date)],
    )
    for c in row.columns:
        row[c] = row[c].astype(float)
    return row


def predict_next_day(
    model: LSTMAttentionModel,
    feature_scaler,
    target_scaler,
    df: pd.DataFrame,
    feature_df: pd.DataFrame,
    actual_last_date: datetime.date,
    actual_last_close: float,
    mean_error: float = 0.0,
) -> dict:
    features = feature_df.values[-SEQ_LENGTH:]
    features_scaled = feature_scaler.transform(features)
    X = features_scaled.reshape(1, SEQ_LENGTH, -1)

    pred_scaled = model(torch.FloatTensor(X))  # (1, H)
    pred_return = float(
        target_scaler.inverse_transform(
            pred_scaled.detach().numpy()[:, :1].reshape(-1, 1)
        )[0, 0]
    )
    raw_pred_price = actual_last_close * (1 + pred_return)
    pred_price = raw_pred_price + mean_error

    return {
        "last_date": actual_last_date,
        "last_close": actual_last_close,
        "predicted_date": next_trading_day(actual_last_date),
        "predicted_return_pct": pred_return * 100,
        "raw_predicted_price": raw_pred_price,
        "predicted_price": pred_price,
        "bias_correction": mean_error,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Predict next gold price using trained LSTM-Attention model"
    )
    parser.add_argument(
        "--model",
        default="models/best_lstm_attention.pt",
        help="Path to model checkpoint (default: models/best_lstm_attention.pt)",
    )
    parser.add_argument(
        "--feature-scaler",
        default="models/feature_scaler.pkl",
        help="Path to feature scaler (default: models/feature_scaler.pkl)",
    )
    parser.add_argument(
        "--target-scaler",
        default="models/target_scaler.pkl",
        help="Path to target scaler (default: models/target_scaler.pkl)",
    )
    parser.add_argument(
        "--bias-correction",
        default="models/bias_correction.txt",
        help="Path to bias correction file (default: models/bias_correction.txt)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=1,
        help="Number of future trading days to predict (default: 1). "
        "WARNING: multi-day is recursive and compounds errors.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("GOLD PRICE PREDICTION — FUTURE INFERENCE")
    print("=" * 60)

    print("\nLoading model...")
    model = load_trained_model(args.model)

    print("Loading scalers...")
    feature_scaler = joblib.load(args.feature_scaler)
    target_scaler = joblib.load(args.target_scaler)

    mean_error = 0.0
    try:
        with open(args.bias_correction) as f:
            mean_error = float(f.read().strip())
        print(f"Loaded bias correction: {mean_error:.4f}")
    except FileNotFoundError:
        print("Warning: bias correction file not found, using 0.0")

    print("Fetching and preparing recent market data...")
    df, feature_df, _, actual_last_date, actual_last_close = prepare_recent_data()
    last_date = (
        actual_last_date.date()
        if hasattr(actual_last_date, "date")
        else actual_last_date
    )
    print(f"  Last trading date: {last_date} ({last_date.strftime('%A')})")
    print(f"  Last close price: ${actual_last_close:.2f}")
    print(f"  Data points used: {len(df)}")

    if args.days == 1:
        result = predict_next_day(
            model,
            feature_scaler,
            target_scaler,
            df,
            feature_df,
            actual_last_date,
            actual_last_close,
            mean_error,
        )
        pred_date = result["predicted_date"]
        pred_date = pred_date.date() if hasattr(pred_date, "date") else pred_date
        print(f"\n{'=' * 60}")
        print(f"PREDICTION FOR NEXT TRADING DAY")
        print(f"{'=' * 60}")
        print(f"  Last trading day : {last_date} ({last_date.strftime('%A')})")
        print(f"  Predicted for    : {pred_date} ({pred_date.strftime('%A')})")
        print(f"  Last close       : ${result['last_close']:.2f}")
        print(f"  Predicted return : {result['predicted_return_pct']:+.4f}%")
        print(f"  Predicted price  : ${result['predicted_price']:.2f}")
        if result["bias_correction"] != 0.0:
            print(f"  Bias correction  : ${result['bias_correction']:.2f}")
        print(f"{'=' * 60}")
    else:
        print(f"\nPredicting next {args.days} trading days (direct multi-step)\n")

        import yfinance as yf

        raw_full = yf.Ticker("GLD").history(start="2024-01-01", interval="1d")
        raw_full.columns = [
            c.lower() if isinstance(c, str) else "_".join(c).lower()
            for c in raw_full.columns
        ]
        raw_full = raw_full[["open", "high", "low", "close", "volume"]]
        raw_full = raw_full.ffill().bfill().sort_index()

        # Direct multi-step prediction — one forward pass for all H returns
        feat_df = _recompute_features(raw_full)
        seq = feat_df.values[-SEQ_LENGTH:]
        seq_scaled = feature_scaler.transform(seq)
        X = seq_scaled.reshape(1, SEQ_LENGTH, -1)

        pred_scaled = model(torch.FloatTensor(X))  # (1, H)
        pred_returns = target_scaler.inverse_transform(
            pred_scaled.detach().numpy().reshape(-1, 1)
        ).ravel()

        n_days = min(args.days, len(pred_returns))

        print(f"{'Day':<6} {'Date':>14} {'Return':>10} {'Price':>12}")
        print(f"{'-' * 46}")
        print(
            f"{'0':<6} {actual_last_date:%Y-%m-%d} {'—':>10} ${actual_last_close:>8.2f}"
        )

        current_date = actual_last_date
        results = []
        for i in range(n_days):
            cum_return = np.prod(1 + pred_returns[: i + 1]) - 1
            price = actual_last_close * (1 + cum_return) + mean_error
            next_date = next_trading_day(current_date)
            results.append({"return_pct": pred_returns[i] * 100, "price": price})
            print(
                f"{i + 1:<6} {next_date:%Y-%m-%d} "
                f"{pred_returns[i] * 100:>+9.4f}% ${price:>8.2f}"
            )
            current_date = next_date

        final = results[-1]
        total_return = (final["price"] - mean_error) / actual_last_close - 1
        print(f"\n{'=' * 60}")
        print(f"  {n_days}-DAY PROJECTION SUMMARY (Direct Multi-Step)")
        print(f"{'=' * 60}")
        print(
            f"  Starting date  : {actual_last_date:%Y-%m-%d} ({actual_last_date.strftime('%A')})"
        )
        print(f"  Starting price : ${actual_last_close:.2f}")
        print(
            f"  Final date     : {current_date:%Y-%m-%d} ({current_date.strftime('%A')})"
        )
        print(f"  Final price    : ${final['price']:.2f}")
        print(f"  Total return   : {total_return * 100:+.2f}%")
        print(f"  Bias correction: ${mean_error:.2f} (applied per step)")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
