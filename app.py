"""
Flask web dashboard for Gold Price Prediction using LSTM-Attention.
"""

import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
import torch
import yfinance as yf
from flask import Flask, jsonify, render_template

sys.path.insert(0, os.path.dirname(__file__))
from src.lstm_model import LSTMAttentionModel

warnings.filterwarnings("ignore")

app = Flask(__name__)

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


def _load_artifacts():
    ckpt = torch.load(
        "models/best_lstm_attention.pt", map_location="cpu", weights_only=False
    )
    output_size = ckpt.get("output_size", 1)
    model = LSTMAttentionModel(input_size=ckpt["input_size"], output_size=output_size)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    feature_scaler = joblib.load("models/feature_scaler.pkl")
    target_scaler = joblib.load("models/target_scaler.pkl")

    mean_error = 0.0
    try:
        with open("models/bias_correction.txt") as f:
            mean_error = float(f.read().strip())
    except FileNotFoundError:
        pass

    return model, feature_scaler, target_scaler, mean_error


def _compute_features(raw):
    df = raw.copy()
    df["sma_10"] = df["close"].rolling(10).mean()
    df["sma_20"] = df["close"].rolling(20).mean()
    df["sma_50"] = df["close"].rolling(50).mean()
    df["ema_10"] = df["close"].ewm(span=10, adjust=False).mean()
    df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["sma_cross_10_20"] = df["sma_10"] - df["sma_20"]
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
    valid = df.iloc[50:].copy()
    cols = [c for c in _FEATURE_COLUMNS if c in valid.columns]
    return valid[cols]


def _next_trading_day(d):
    d = d + pd.Timedelta(days=1)
    while d.weekday() >= 5:
        d += pd.Timedelta(days=1)
    return d


def _get_prediction_data(model, feature_scaler, target_scaler, mean_error, days=1):
    ticker = yf.Ticker("GLD")
    raw = ticker.history(start="2024-01-01", interval="1d")
    raw.columns = [
        c.lower() if isinstance(c, str) else "_".join(c).lower() for c in raw.columns
    ]
    raw = raw[["open", "high", "low", "close", "volume"]]
    raw = raw.ffill().bfill().sort_index()

    last_date = raw.index[-1]
    last_close = float(raw["close"].iloc[-1])

    feat = _compute_features(raw)
    seq = feat.values[-SEQ_LENGTH:]
    X = feature_scaler.transform(seq).reshape(1, SEQ_LENGTH, -1)

    # Direct multi-step: model predicts H future returns in one forward pass
    pred_scaled = model(torch.FloatTensor(X))  # (1, H)
    pred_returns = target_scaler.inverse_transform(
        pred_scaled.detach().numpy().reshape(-1, 1)
    ).ravel()

    n_days = min(days, len(pred_returns))

    result = {
        "last_date": str(last_date.date()),
        "last_close": round(float(last_close), 2),
        "predicted_return_pct": round(float(pred_returns[0]) * 100, 4),
        "predicted_price": round(
            float(last_close) * (1 + float(pred_returns[0])) + float(mean_error), 2
        ),
        "bias_correction": round(float(mean_error), 2),
        "forecast": [],
    }

    current_date = last_date
    for i in range(n_days):
        cum_return = float(np.prod(1 + pred_returns[: i + 1]) - 1)
        nxt_date = _next_trading_day(current_date)
        result["forecast"].append(
            {
                "day": i + 1,
                "date": str(nxt_date.date()),
                "return_pct": round(float(pred_returns[i]) * 100, 4),
                "price": round(
                    float(last_close) * (1 + cum_return) + float(mean_error), 2
                ),
            }
        )
        current_date = nxt_date

    return result


model, feature_scaler, target_scaler, mean_error = _load_artifacts()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/predict")
def api_predict():
    data = _get_prediction_data(
        model, feature_scaler, target_scaler, mean_error, days=1
    )
    return jsonify(data)


@app.route("/api/predict/<int:days>")
def api_predict_days(days):
    data = _get_prediction_data(
        model, feature_scaler, target_scaler, mean_error, days=days
    )
    return jsonify(data)


if __name__ == "__main__":
    app.run(debug=True)
