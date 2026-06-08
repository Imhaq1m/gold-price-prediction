"""
Flask web application for Gold Price Prediction using LSTM-Attention.

Provides:
  - GET  /         Renders page with historical chart + default 7-day prediction
  - POST /predict  Accepts {"days": N}, returns JSON prediction array

Run with:  py -m flask run
   or:     py app.py
   or:     py app.py --ngrok  (public tunnel via ngrok)
"""

import datetime
import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd
import torch
from flask import Flask, jsonify, render_template, request

from src.data_module import fetch_gold_data, preprocess_data
from src.feature_engineering import add_technical_indicators, prepare_features
from src.lstm_model import LSTMAttentionModel, load_cv_models, ensemble_predict

warnings.filterwarnings("ignore")

app = Flask(__name__)

SEQ_LENGTH = 30
FORECAST_HORIZON = 30

# Module-level state (loaded once at startup)
models = []
feature_scaler = None
target_scaler = None
mean_error = 0.0
TICKER = os.environ.get("TICKER", "GLD")

# Simple data cache to avoid refetching yfinance on every AJAX call
_data_cache = {"df": None, "feature_df": None, "timestamp": None}


def load_artifacts():
    global models, feature_scaler, target_scaler, mean_error

    if os.path.exists("models/cv_fold_1.pt"):
        models = load_cv_models()
        print(f"Loaded {len(models)} CV models for ensemble")
    elif os.path.exists("models/best_lstm_attention.pt"):
        ckpt = torch.load(
            "models/best_lstm_attention.pt", map_location="cpu", weights_only=False
        )
        m = LSTMAttentionModel(
            input_size=ckpt["input_size"], output_size=ckpt.get("output_size", 1)
        )
        m.load_state_dict(ckpt["model_state_dict"])
        m.eval()
        models = [m]
        print("Loaded single model")
    else:
        print("WARNING: No model checkpoints found. Prediction will be disabled.")

    if os.path.exists("models/feature_scaler.pkl"):
        feature_scaler = joblib.load("models/feature_scaler.pkl")
        print("Loaded feature scaler")
    else:
        print("WARNING: feature_scaler.pkl not found.")

    if os.path.exists("models/target_scaler.pkl"):
        target_scaler = joblib.load("models/target_scaler.pkl")
        print("Loaded target scaler")
    else:
        print("WARNING: target_scaler.pkl not found.")

    if os.path.exists("models/bias_correction.txt"):
        with open("models/bias_correction.txt") as f:
            mean_error = float(f.read().strip())
        print(f"Loaded bias correction: {mean_error * 100:.2f}%")


def next_trading_day(d):
    next_day = d + datetime.timedelta(days=1)
    while next_day.weekday() >= 5:
        next_day += datetime.timedelta(days=1)
    return next_day


def _fetch_and_prepare():
    now = datetime.datetime.now()
    if (
        _data_cache["timestamp"] is not None
        and (now - _data_cache["timestamp"]).seconds < 3600
    ):
        return _data_cache["df"], _data_cache["feature_df"]

    raw = fetch_gold_data(ticker=TICKER, start="2022-01-01")
    raw = preprocess_data(raw)
    df = add_technical_indicators(raw)
    feature_df, feature_columns = prepare_features(df)
    if "close" not in feature_columns:
        feature_columns = ["close"] + feature_columns
        feature_df = df[feature_columns]

    _data_cache["df"] = df
    _data_cache["feature_df"] = feature_df
    _data_cache["timestamp"] = now
    return df, feature_df


def _predict_direct(days, df, feature_df):
    days = min(days, FORECAST_HORIZON)

    # Direct multi-step: one forward pass predicts all H returns at once
    last_seq = feature_df.values[-SEQ_LENGTH:]
    scaled = feature_scaler.transform(last_seq)
    X = scaled.reshape(1, SEQ_LENGTH, -1)

    pred_scaled = ensemble_predict(models, X)
    model_returns = target_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()

    # Bootstrap residuals from historical returns so trajectory has both
    # up and down days (model never predicts negative returns on its own).
    historical_returns = df["returns"].dropna().values
    centered = historical_returns - np.mean(historical_returns)

    rng = np.random.RandomState(
        sum(ord(c) * (i + 1) for i, c in enumerate(str(datetime.date.today())))
        % (2**31)
    )
    noise = rng.choice(centered, size=days, replace=True)

    last_close = float(df["close"].iloc[-1])
    current_date = df.index[-1]
    results = []

    for i in range(days):
        noisy_return = model_returns[i] + noise[i]
        price = last_close * (1 + noisy_return)

        next_date = next_trading_day(current_date)
        results.append(
            {
                "day": i + 1,
                "date": str(next_date),
                "return_pct": round(float(noisy_return * 100), 4),
                "price": round(float(price), 2),
            }
        )

        last_close = price
        current_date = next_date

    # Apply bias correction once at the end
    if abs(1 - mean_error) > 1e-8:
        for r in results:
            r["price"] = round(r["price"] / (1 - mean_error), 2)

    return results


@app.route("/")
def index():
    try:
        df, feature_df = _fetch_and_prepare()
    except Exception as e:
        return render_template("index.html", error=f"Failed to load data: {e}")

    three_months_ago = df.index[-1] - pd.DateOffset(months=3)
    mask = df.index >= three_months_ago
    historical_dates = [str(d.date()) for d in df.index[mask]]
    historical_prices = [round(float(p), 2) for p in df["close"].values[mask]]

    initial_preds = []
    if models and feature_scaler and target_scaler:
        try:
            initial_preds = _predict_direct(7, df, feature_df)
        except Exception as e:
            print(f"Initial prediction error: {e}")

    return render_template(
        "index.html",
        historical_dates=json.dumps(historical_dates),
        historical_prices=json.dumps(historical_prices),
        initial_preds=json.dumps(initial_preds),
        last_close=round(float(df["close"].iloc[-1]), 2),
        last_date=str(df.index[-1].date()),
        default_days=7,
        models_loaded=len(models) > 0,
        error=None,
    )


@app.route("/predict", methods=["POST"])
def predict():
    if not models or not feature_scaler or not target_scaler:
        return jsonify({"error": "Model artifacts not loaded"}), 503

    data = request.get_json()
    days = int(data.get("days", 7))

    try:
        df, feature_df = _fetch_and_prepare()
        predictions = _predict_direct(days, df, feature_df)
        return jsonify({"predictions": predictions})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Load artifacts at import time (not inside __main__) so flask run works too
load_artifacts()

if __name__ == "__main__":
    import sys

    # Parse --ticker from CLI args (before Flask starts)
    for i, arg in enumerate(sys.argv):
        if arg == "--ticker" and i + 1 < len(sys.argv):
            TICKER = sys.argv[i + 1]
            # Remove both from sys.argv so Flask doesn't complain
            sys.argv.pop(i)
            sys.argv.pop(i)
            break

    ngrok_tunnel = None
    if "--ngrok" in sys.argv:
        from pyngrok import ngrok

        ngrok.kill()
        ngrok_tunnel = ngrok.connect(5000)
        print(f"\nPublic URL (share this): {ngrok_tunnel.public_url}\n")

    try:
        app.run(debug=ngrok_tunnel is None)
    finally:
        if ngrok_tunnel is not None:
            from pyngrok import ngrok

            ngrok.disconnect(ngrok_tunnel.public_url)
            ngrok.kill()
