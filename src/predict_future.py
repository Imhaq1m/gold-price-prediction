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
from src.lstm_model import LSTMAttentionModel, load_cv_models, ensemble_predict

warnings.filterwarnings("ignore")

SEQ_LENGTH = 30


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
    models: list,
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

    pred_scaled = ensemble_predict(models, X)  # (1, H)
    pred_return = float(
        target_scaler.inverse_transform(pred_scaled[:, :1].reshape(-1, 1))[0, 0]
    )
    raw_pred_price = actual_last_close * (1 + pred_return)
    pred_price = raw_pred_price / (1 - mean_error)

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
        help="Path to bias correction file (percentage, default: models/bias_correction.txt)",
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

    print("\nLoading model(s)...")
    if args.model == "models/best_lstm_attention.pt":
        models = load_cv_models()
        print(f"  Loaded {len(models)} CV models for ensemble prediction")
    else:
        models = [load_trained_model(args.model)]
        print("  Loaded single model")

    print("Loading scalers...")
    feature_scaler = joblib.load(args.feature_scaler)
    target_scaler = joblib.load(args.target_scaler)

    mean_error = 0.0
    try:
        with open(args.bias_correction) as f:
            mean_error = float(f.read().strip())
        print(f"Loaded bias correction: {mean_error * 100:.2f}%")
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
            models,
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
            print(f"  Bias correction  : {result['bias_correction'] * 100:.2f}%")
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
        feat_df = add_technical_indicators(raw_full)
        feat_df, _ = prepare_features(feat_df)
        seq = feat_df.values[-SEQ_LENGTH:]
        seq_scaled = feature_scaler.transform(seq)
        X = seq_scaled.reshape(1, SEQ_LENGTH, -1)

        pred_scaled = ensemble_predict(models, X)  # (1, H)
        pred_returns = target_scaler.inverse_transform(
            pred_scaled.reshape(-1, 1)
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
            price = actual_last_close * (1 + cum_return) / (1 - mean_error)
            next_date = next_trading_day(current_date)
            results.append({"return_pct": pred_returns[i] * 100, "price": price})
            print(
                f"{i + 1:<6} {next_date:%Y-%m-%d} "
                f"{pred_returns[i] * 100:>+9.4f}% ${price:>8.2f}"
            )
            current_date = next_date

        final = results[-1]
        total_return = (final["price"] * (1 - mean_error)) / actual_last_close - 1
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
        print(f"  Bias correction: {mean_error * 100:.2f}% (applied per step)")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
