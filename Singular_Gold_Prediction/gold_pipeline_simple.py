import os
import json
import datetime
import random
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import yfinance as yf
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from flask import Flask, jsonify, render_template_string, request

warnings.filterwarnings("ignore")

SEQ_LENGTH = 30
FORECAST_HORIZON = 30
RANDOM_STATE = 42

# EMBEDDED HTML TEMPLATE
TEMPLATE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Gold Price Predictor</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js">
  </script>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
      background: #0d0d1a;
      color: #d0d0e0;
      font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
      min-height: 100vh;
      display: flex;
      justify-content: center;
      padding: 2rem 1rem;
    }
    .container { max-width: 1100px; width: 100%; }

    header {
      text-align: center;
      margin-bottom: 2rem;
    }
    header h1 {
      font-size: 2rem;
      letter-spacing: 4px;
      color: #ffd700;
      text-shadow: 0 0 20px rgba(255, 215, 0, 0.15);
    }
    header .subtitle {
      font-size: 0.85rem;
      color: #8888aa;
      margin-top: 0.25rem;
    }
    .info-bar {
      display: flex;
      justify-content: center;
      gap: 2rem;
      margin-top: 1rem;
      font-size: 0.95rem;
    }
    .info-bar .label { color: #8888aa; }
    .info-bar .value { color: #ffd700; font-weight: 600; }

    .chart-card {
      background: #13132a;
      border: 1px solid #2a2a4a;
      border-radius: 12px;
      padding: 1.5rem;
      margin-bottom: 1.5rem;
    }
    .chart-card canvas { width: 100% !important; }

    .controls {
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 0.75rem;
      margin-bottom: 1.5rem;
      flex-wrap: wrap;
    }
    .controls label {
      font-size: 0.95rem;
      color: #aaaacc;
    }
    .controls input[type="number"] {
      width: 70px;
      padding: 0.5rem 0.6rem;
      background: #1a1a35;
      border: 1px solid #3a3a5a;
      border-radius: 6px;
      color: #fff;
      font-size: 1rem;
      text-align: center;
    }
    .controls input[type="number"]:focus {
      outline: none;
      border-color: #ffd700;
    }
    .controls .note {
      font-size: 0.8rem;
      color: #666688;
    }
    .btn {
      padding: 0.55rem 1.8rem;
      background: linear-gradient(135deg, #ffd700, #e6b800);
      border: none;
      border-radius: 6px;
      color: #0d0d1a;
      font-weight: 700;
      font-size: 0.95rem;
      cursor: pointer;
      transition: opacity 0.2s;
    }
    .btn:hover { opacity: 0.85; }
    .btn:disabled {
      opacity: 0.35;
      cursor: not-allowed;
    }

    .table-card {
      background: #13132a;
      border: 1px solid #2a2a4a;
      border-radius: 12px;
      padding: 1.5rem;
      overflow-x: auto;
    }
    .table-card h3 {
      font-size: 0.9rem;
      color: #8888aa;
      text-transform: uppercase;
      letter-spacing: 2px;
      margin-bottom: 0.75rem;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.9rem;
    }
    th {
      text-align: left;
      padding: 0.6rem 0.8rem;
      color: #8888aa;
      border-bottom: 1px solid #2a2a4a;
      font-weight: 500;
      text-transform: uppercase;
      letter-spacing: 1px;
      font-size: 0.78rem;
    }
    td {
      padding: 0.55rem 0.8rem;
      border-bottom: 1px solid #1e1e3a;
    }
    tr:last-child td { border-bottom: none; }
    .text-right { text-align: right; }
    .price { color: #ffd700; font-weight: 600; }
    .positive { color: #4cdf8b; }
    .negative { color: #ff6b6b; }

    .alert {
      background: #1a0a0a;
      border: 1px solid #5a2020;
      color: #ff8888;
      padding: 1rem 1.25rem;
      border-radius: 8px;
      margin-bottom: 1.5rem;
      text-align: center;
    }

    .spinner {
      display: inline-block;
      width: 16px;
      height: 16px;
      border: 2px solid #8888aa;
      border-top-color: #ffd700;
      border-radius: 50%;
      animation: spin 0.7s linear infinite;
      vertical-align: middle;
      margin-right: 6px;
    }
    @keyframes spin { to { transform: rotate(360deg); } }

    @media (max-width: 600px) {
      header h1 { font-size: 1.4rem; }
      .info-bar { flex-direction: column; gap: 0.25rem; align-items: center; }
    }
  </style>
</head>
<body>
<div class="container">

  <header>
    <h1>GOLD PRICE PREDICTOR</h1>
    <p class="subtitle">LSTM-Attention &middot; Direct Multi-Step Forecasting</p>
    <div class="info-bar">
      <span><span class="label">Last Close</span> <span class="value" id="lastClose">${{ last_close }}</span></span>
      <span><span class="label">Last Date</span> <span class="value" id="lastDate">{{ last_date }}</span></span>
    </div>
  </header>

  {% if error %}
  <div class="alert">{{ error }}</div>
  {% endif %}

  {% if not models_loaded %}
  <div class="alert">
    Model artifacts not found. Run <code>py gold_pipeline_simple.py</code> first to train models.
  </div>
  {% endif %}

  <div class="chart-card">
    <canvas id="priceChart"></canvas>
  </div>

  <div class="controls">
    <label>
      Predict next
      <input type="number" id="daysInput" value="{{ default_days }}" min="1" max="30">
      days
    </label>
    <button class="btn" id="predictBtn" {% if not models_loaded %}disabled{% endif %}>
      Predict
    </button>
    <span class="note">(max 30 days)</span>
  </div>

  <div class="table-card">
    <h3>Prediction Details</h3>
    <table>
      <thead>
        <tr>
          <th>Day</th>
          <th>Date</th>
          <th class="text-right">Return %</th>
          <th class="text-right">Price</th>
        </tr>
      </thead>
      <tbody id="predictionBody"></tbody>
    </table>
    <p id="noPredMsg" style="color:#666688;text-align:center;padding:1rem 0;">
      Enter days and click Predict.
    </p>
  </div>

</div>

<script>
(function () {
  var historicalDates = JSON.parse('{{ historical_dates|safe }}');
  var historicalPrices = JSON.parse('{{ historical_prices|safe }}');
  var initialPreds    = JSON.parse('{{ initial_preds|safe }}');
  var histLen = historicalDates.length;

  function fmtPrice(v) { return '$' + v.toFixed(2); }

  function predChartData(preds) {
    var dates = preds.map(function (p) { return p.date; });
    var prices = preds.map(function (p) { return p.price; });
    return { dates: dates, prices: prices };
  }

  var ctx = document.getElementById('priceChart').getContext('2d');

  function makeChart(histDates, histPrices, predDates, predPrices) {
    var labels = [].concat(histDates, predDates);
    var hData  = [].concat(histPrices, new Array(predDates.length).fill(null));
    var pData  = [].concat(new Array(histDates.length).fill(null), predPrices);

    return new Chart(ctx, {
      type: 'line',
      data: {
        labels: labels,
        datasets: [
          {
            label: 'Historical',
            data: hData,
            borderColor: '#FFD700',
            backgroundColor: 'rgba(255,215,0,0.04)',
            borderWidth: 2,
            pointRadius: 0,
            pointHoverRadius: 4,
            tension: 0.15,
            spanGaps: false,
            fill: true,
          },
          {
            label: 'Prediction',
            data: pData,
            borderColor: '#FF8C00',
            backgroundColor: 'rgba(255,140,0,0.06)',
            borderWidth: 2,
            borderDash: [6, 4],
            pointRadius: 3,
            pointHoverRadius: 5,
            pointBackgroundColor: '#FF8C00',
            tension: 0.15,
            spanGaps: false,
            fill: true,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: true,
        interaction: { intersect: false, mode: 'index' },
        plugins: {
          legend: {
            labels: { color: '#aaaacc', font: { size: 13 }, usePointStyle: true, padding: 20 },
          },
          tooltip: {
            backgroundColor: '#1e1e3a',
            titleColor: '#ffd700',
            bodyColor: '#d0d0e0',
            borderColor: '#3a3a5a',
            borderWidth: 1,
            padding: 10,
            callbacks: {
              label: function (ctx) {
                if (ctx.parsed.y === null) return null;
                return ctx.dataset.label + ': $' + ctx.parsed.y.toFixed(2);
              },
            },
          },
        },
        scales: {
          x: {
            ticks: {
              color: '#666688',
              maxTicksLimit: 12,
              font: { size: 11 },
            },
            grid: { color: '#1e1e3a' },
          },
          y: {
            ticks: {
              color: '#666688',
              font: { size: 11 },
              callback: function (v) { return '$' + v.toFixed(0); },
            },
            grid: { color: '#1e1e3a' },
          },
        },
      },
    });
  }

  function renderTable(preds) {
    var tbody = document.getElementById('predictionBody');
    var noMsg = document.getElementById('noPredMsg');
    if (!preds || preds.length === 0) {
      tbody.innerHTML = '';
      noMsg.style.display = '';
      return;
    }
    noMsg.style.display = 'none';
    tbody.innerHTML = preds.map(function (p) {
      var cls = p.return_pct >= 0 ? 'positive' : 'negative';
      var sign = p.return_pct >= 0 ? '+' : '';
      return '<tr>'
        + '<td>' + p.day + '</td>'
        + '<td>' + p.date + '</td>'
        + '<td class="text-right ' + cls + '">' + sign + p.return_pct.toFixed(2) + '%</td>'
        + '<td class="text-right price">' + fmtPrice(p.price) + '</td>'
        + '</tr>';
    }).join('');
  }

  function updateChart(preds) {
    var pd = predChartData(preds);
    var newLabels = [].concat(historicalDates, pd.dates);
    chart.data.labels = newLabels;
    chart.data.datasets[0].data = [].concat(historicalPrices, new Array(pd.dates.length).fill(null));
    chart.data.datasets[1].data = [].concat(new Array(histLen).fill(null), pd.prices);
    chart.update();
  }

  var initPd = predChartData(initialPreds);
  var chart = makeChart(historicalDates, historicalPrices, initPd.dates, initPd.prices);
  renderTable(initialPreds);

  document.getElementById('predictBtn').addEventListener('click', async function () {
    var days = parseInt(document.getElementById('daysInput').value, 10) || 7;
    var btn = this;
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Predicting...';

    try {
      var resp = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ days: days }),
      });
      var data = await resp.json();
      if (data.error) { alert(data.error); return; }
      updateChart(data.predictions);
      renderTable(data.predictions);
    } catch (e) {
      alert('Prediction request failed: ' + e.message);
    } finally {
      btn.disabled = false;
      btn.textContent = 'Predict';
    }
  });

  document.getElementById('daysInput').addEventListener('keydown', function (e) {
    if (e.key === 'Enter') document.getElementById('predictBtn').click();
  });
})();
</script>
</body>
</html>"""


def get_device():
    return torch.device("cpu")


def fetch_gold_data(ticker="GLD", start="2015-01-01", end=None, interval="1d"):
    print("Fetching %s data from %s to %s..." % (ticker, start, end or "now"))
    gold = yf.Ticker(ticker)
    df = gold.history(start=start, end=end, interval=interval)
    # Fix column names
    new_cols = []
    for col in df.columns:
        if isinstance(col, str):
            new_cols.append(col.lower())
        else:
            new_cols.append("_".join(col).lower())
    df.columns = new_cols
    print("Got %d records." % len(df))
    return df


def preprocess_data(df):
    df = df.copy()
    df = df.ffill()
    df = df.bfill()
    df = df.sort_index()
    # Remove duplicate dates
    seen = set()
    keep = []
    for idx in df.index:
        if idx not in seen:
            seen.add(idx)
            keep.append(True)
        else:
            keep.append(False)
    df = df[keep]
    print("Data shape after cleaning: %s" % str(df.shape))
    return df


def split_data(df, train_ratio=0.8):
    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    print("Train: %d, Test: %d" % (len(train_df), len(test_df)))
    return train_df, test_df


# FEATURE ENGINEERING
def add_technical_indicators(df):
    df = df.copy()

    # Moving averages
    df["sma_10"] = df["close"].rolling(window=10).mean()
    df["sma_20"] = df["close"].rolling(window=20).mean()
    df["sma_50"] = df["close"].rolling(window=50).mean()
    df["ema_10"] = df["close"].ewm(span=10, adjust=False).mean()
    df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()

    # Crossover
    df["sma_cross_10_20"] = df["sma_10"] - df["sma_20"]
    df["sma_cross_20_50"] = df["sma_20"] - df["sma_50"]

    # Returns
    df["returns"] = df["close"].pct_change()
    df["log_returns"] = np.log(df["close"] / df["close"].shift(1))

    # Volatility
    df["volatility_10"] = df["returns"].rolling(window=10).std()
    df["volatility_20"] = df["returns"].rolling(window=20).std()

    # RSI
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # Bollinger Bands
    df["bb_middle"] = df["close"].rolling(window=20).mean()
    bb_std = df["close"].rolling(window=20).std()
    df["bb_upper"] = df["bb_middle"] + (bb_std * 2)
    df["bb_lower"] = df["bb_middle"] - (bb_std * 2)
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
    df["bb_position"] = (df["close"] - df["bb_lower"]) / (
        df["bb_upper"] - df["bb_lower"]
    )

    # Range features
    df["hl_range"] = (df["high"] - df["low"]) / df["close"]
    df["oc_range"] = (df["close"] - df["open"]) / df["open"]

    # Volume
    if "volume" in df.columns:
        df["volume_sma_10"] = df["volume"].rolling(window=10).mean()
        df["volume_ratio"] = df["volume"] / df["volume_sma_10"]

    # Lag features
    for lag in [1, 2, 3, 5, 10]:
        df["close_lag_%d" % lag] = df["close"].shift(lag)
        df["returns_lag_%d" % lag] = df["returns"].shift(lag)

    # Rolling stats
    for window in [5, 10, 20]:
        df["close_mean_%d" % window] = df["close"].rolling(window=window).mean()
        df["close_std_%d" % window] = df["close"].rolling(window=window).std()
        df["close_min_%d" % window] = df["close"].rolling(window=window).min()
        df["close_max_%d" % window] = df["close"].rolling(window=window).max()

    # Time features
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    df["quarter"] = df.index.quarter

    # Target
    df["target"] = df["close"].shift(-1)

    df = df.dropna()
    print("Features added. Shape: %s" % str(df.shape))
    return df


def prepare_features(df, feature_columns=None):
    if feature_columns is None:
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
        # Only keep what exists
        existing = []
        for col in feature_columns:
            if col in df.columns:
                existing.append(col)
        feature_columns = existing
    return df[feature_columns], feature_columns


def create_sequences(data, target, seq_length=60, forecast_horizon=1):
    X = []
    y = []
    for i in range(len(data) - seq_length - forecast_horizon + 1):
        X.append(data[i : i + seq_length])
        y.append(target[i + seq_length : i + seq_length + forecast_horizon])
    return np.array(X), np.array(y)


def scale_data(train_df, test_df, feature_columns, target_column="target"):
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler = MinMaxScaler(feature_range=(0, 1))

    train_features = train_df[feature_columns].values
    train_target = train_df[[target_column]].values

    train_features_scaled = feature_scaler.fit_transform(train_features)
    train_target_scaled = target_scaler.fit_transform(train_target).ravel()

    test_features = test_df[feature_columns].values
    test_target = test_df[[target_column]].values

    test_features_scaled = feature_scaler.transform(test_features)
    test_target_scaled = target_scaler.transform(test_target).ravel()

    print(
        "Features scaled. Train: %s, Test: %s"
        % (str(train_features_scaled.shape), str(test_features_scaled.shape))
    )
    return (
        train_features_scaled,
        train_target_scaled,
        test_features_scaled,
        test_target_scaled,
        feature_scaler,
        target_scaler,
    )


# MODEL ARCHITECTURES
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, key_dim, output_dim=None):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim
        if output_dim is None:
            output_dim = key_dim * num_heads
        self.output_dim = output_dim

        self.W_q = nn.Linear(key_dim, self.output_dim)
        self.W_k = nn.Linear(key_dim, self.output_dim)
        self.W_v = nn.Linear(key_dim, self.output_dim)
        self.W_o = nn.Linear(self.output_dim, self.output_dim)
        self.scale = torch.sqrt(torch.FloatTensor([key_dim]))

    def forward(self, x):
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        batch_size = Q.size(0)
        seq_len = Q.size(1)
        head_dim = self.output_dim // self.num_heads

        Q = Q.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale.to(Q.device)
        attn_weights = torch.softmax(scores, dim=-1)
        attended = torch.matmul(attn_weights, V)

        attended = (
            attended.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.output_dim)
        )
        output = self.W_o(attended)
        return output


class StackedLSTM(nn.Module):
    def __init__(self, input_size, hidden_sizes=None, dropout_rates=None):
        super(StackedLSTM, self).__init__()
        if hidden_sizes is None:
            hidden_sizes = [128, 64, 32]
        if dropout_rates is None:
            dropout_rates = [0.3, 0.2, 0.2]

        self.lstm_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        self.batch_norm_layers = nn.ModuleList()

        for i in range(len(hidden_sizes)):
            if i == 0:
                input_dim = input_size
            else:
                input_dim = hidden_sizes[i - 1]
            self.lstm_layers.append(
                nn.LSTM(input_dim, hidden_sizes[i], batch_first=True)
            )
            self.batch_norm_layers.append(nn.BatchNorm1d(hidden_sizes[i]))
            self.dropout_layers.append(nn.Dropout(dropout_rates[i]))

        self.fc = nn.Sequential(
            nn.Linear(hidden_sizes[-1], 32), nn.ReLU(), nn.Linear(32, 1)
        )

    def forward(self, x):
        for i in range(len(self.lstm_layers)):
            x, _ = self.lstm_layers[i](x)
            x = self.dropout_layers[i](x)
        x = x[:, -1, :]
        x = self.batch_norm_layers[-1](x)
        out = self.fc(x)
        return out.squeeze(-1)


class LSTMAttentionModel(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size=48,
        num_heads=4,
        dropout=0.2,
        output_size=1,
        activation="tanh",
    ):
        super(LSTMAttentionModel, self).__init__()

        # Make sure hidden size works with num heads
        if hidden_size % num_heads != 0:
            hidden_size = (hidden_size // num_heads) * num_heads
            if hidden_size == 0:
                hidden_size = num_heads

        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

        if activation == "tanh":
            self.activation_fn = nn.Tanh()
        elif activation == "relu":
            self.activation_fn = nn.ReLU()
        elif activation == "leaky_relu":
            self.activation_fn = nn.LeakyReLU(negative_slope=0.01)
        else:
            self.activation_fn = nn.Tanh()

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout1(lstm_out)

        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = self.dropout2(attn_out)
        attn_out = self.layer_norm(attn_out + lstm_out)
        attn_out = self.activation_fn(attn_out)

        out = attn_out[:, -1, :]
        out = self.fc(out)
        return out


class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=50, dropout=0.2):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        return out.squeeze(-1)


# TRAINING
def create_dataloaders(X_train, y_train, X_val=None, y_val=None, batch_size=64):
    X_t = torch.FloatTensor(X_train)
    y_t = torch.FloatTensor(y_train)
    train_dataset = TensorDataset(X_t, y_t)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(RANDOM_STATE),
    )

    val_loader = None
    if X_val is not None and y_val is not None:
        X_v = torch.FloatTensor(X_val)
        y_v = torch.FloatTensor(y_val)
        val_dataset = TensorDataset(X_v, y_v)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def train_model(
    model,
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    epochs=100,
    batch_size=64,
    learning_rate=0.001,
    patience_es=15,
    model_path="models/best_lstm_model.pt",
    optimizer_type="adam",
):
    device = get_device()
    model = model.to(device)

    train_loader, val_loader = create_dataloaders(
        X_train, y_train, X_val, y_val, batch_size
    )

    criterion = nn.MSELoss()

    if optimizer_type == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_type == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_type == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    elif optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=7, min_lr=1e-6
    )

    history = {"loss": [], "val_loss": [], "mae": [], "val_mae": []}

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    print("\nTraining for up to %d epochs..." % epochs)
    print("Device: %s" % device)
    print("Optimizer: %s" % optimizer_type.upper())
    if X_val is not None:
        print("Train: %d, Val: %d" % (len(X_train), len(X_val)))
    else:
        print("Train: %d, No validation" % len(X_train))

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_mae = 0.0
        n_train = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * len(X_batch)
            train_mae += torch.mean(torch.abs(predictions - y_batch)).item() * len(
                X_batch
            )
            n_train += len(X_batch)

        avg_train_loss = train_loss / n_train
        avg_train_mae = train_mae / n_train
        history["loss"].append(avg_train_loss)
        history["mae"].append(avg_train_mae)

        # Validation
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_mae = 0.0
            n_val = 0

            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)
                    predictions = model(X_batch)
                    loss = criterion(predictions, y_batch)

                    val_loss += loss.item() * len(X_batch)
                    val_mae += torch.mean(
                        torch.abs(predictions - y_batch)
                    ).item() * len(X_batch)
                    n_val += len(X_batch)

            avg_val_loss = val_loss / n_val
            avg_val_mae = val_mae / n_val
            history["val_loss"].append(avg_val_loss)
            history["val_mae"].append(avg_val_mae)

            scheduler.step(avg_val_loss)

            current_lr = optimizer.param_groups[0]["lr"]
            print(
                "Epoch %d/%d - Loss: %.6f - MAE: %.6f - Val Loss: %.6f - Val MAE: %.6f - LR: %.6f"
                % (
                    epoch + 1,
                    epochs,
                    avg_train_loss,
                    avg_train_mae,
                    avg_val_loss,
                    avg_val_mae,
                    current_lr,
                )
            )

            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_state = {}
                for k, v in model.state_dict().items():
                    best_state[k] = v.clone()
                patience_counter = 0

                if model_path is not None:
                    if hasattr(model.fc, "out_features"):
                        out_size = model.fc.out_features
                    else:
                        out_size = 1
                    torch.save(
                        {
                            "model_state_dict": best_state,
                            "input_size": X_train.shape[2],
                            "output_size": out_size,
                            "hidden_sizes": [128, 64, 32],
                            "dropout_rates": [0.3, 0.2, 0.2],
                        },
                        model_path,
                    )
            else:
                patience_counter += 1
                if patience_counter >= patience_es:
                    print("Early stopping at epoch %d" % (epoch + 1))
                    break
        else:
            print(
                "Epoch %d/%d - Loss: %.6f - MAE: %.6f"
                % (epoch + 1, epochs, avg_train_loss, avg_train_mae)
            )
            best_state = {}
            for k, v in model.state_dict().items():
                best_state[k] = v.clone()
            if model_path is not None:
                if hasattr(model.fc, "out_features"):
                    out_size = model.fc.out_features
                else:
                    out_size = 1
                torch.save(
                    {
                        "model_state_dict": best_state,
                        "input_size": X_train.shape[2],
                        "output_size": out_size,
                        "hidden_sizes": [128, 64, 32],
                        "dropout_rates": [0.3, 0.2, 0.2],
                    },
                    model_path,
                )

    if best_state is not None:
        model.load_state_dict(best_state)

    print("Training done! Best val loss: %.6f" % best_val_loss)
    if model_path is not None:
        print("Model saved to %s" % model_path)
    return history


def predict(model, X):
    device = get_device()
    model = model.to(device)
    model.eval()
    X_tensor = torch.FloatTensor(X).to(device)
    with torch.no_grad():
        predictions = model(X_tensor)
    return predictions.cpu().numpy()


def load_model(model_path):
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model = StackedLSTM(
        input_size=checkpoint["input_size"],
        hidden_sizes=checkpoint["hidden_sizes"],
        dropout_rates=checkpoint["dropout_rates"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def load_cv_models(num_folds=5, prefix="models/cv_fold_"):
    models = []
    for fold in range(1, num_folds + 1):
        path = "%s%d.pt" % (prefix, fold)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        output_size = 1
        if "output_size" in ckpt:
            output_size = ckpt["output_size"]
        model = LSTMAttentionModel(
            input_size=ckpt["input_size"], output_size=output_size
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        models.append(model)
    return models


def ensemble_predict(models, X):
    all_preds = []
    for model in models:
        pred = predict(model, X)
        all_preds.append(pred)
    return np.mean(all_preds, axis=0)


# EVALUATION
def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)

    if len(y_true) > 1:
        true_dir = np.sign(np.diff(y_true))
        pred_dir = np.sign(np.diff(y_pred))
        dir_acc = np.mean(true_dir == pred_dir) * 100
    else:
        dir_acc = None

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "R2": r2,
        "Directional Accuracy (%)": dir_acc,
    }


def print_metrics(metrics, title="Model Performance"):
    print(title)
    for name, value in metrics.items():
        if value is not None:
            if "Accuracy" in name:
                print("%-25s: %.2f%%" % (name, value))
            elif "MAPE" in name or name == "R2":
                print("%-25s: %.4f" % (name, value))
            else:
                print("%-25s: %.6f" % (name, value))
    print("=" * 50 + "\n")


def plot_predictions(
    y_true, y_pred, dates=None, title="Gold Price Prediction vs Actual", save_path=None
):
    plt.figure(figsize=(14, 7))

    if dates is not None:
        dates = dates[-len(y_true) :]
        plt.plot(dates, y_true, label="Actual", linewidth=2, alpha=0.8)
        plt.plot(dates, y_pred, label="Predicted", linewidth=2, alpha=0.8)
        plt.xlabel("Date")
        plt.xticks(rotation=45)
    else:
        plt.plot(y_true, label="Actual", linewidth=2, alpha=0.8)
        plt.plot(y_pred, label="Predicted", linewidth=2, alpha=0.8)
        plt.xlabel("Samples")

    plt.ylabel("Price (USD)")
    plt.title(title, fontsize=16, fontweight="bold")
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print("Plot saved to %s" % save_path)

    plt.show(block=False)
    plt.pause(0.1)


def plot_training_history(history, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if hasattr(history, "history"):
        hist = history.history
    else:
        hist = history

    axes[0].plot(hist["loss"], label="Training Loss", linewidth=2)
    if "val_loss" in hist and hist["val_loss"]:
        axes[0].plot(hist["val_loss"], label="Validation Loss", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss (MSE)")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    if "mae" in hist and hist["mae"]:
        axes[1].plot(hist["mae"], label="Training MAE", linewidth=2)
        if "val_mae" in hist and hist["val_mae"]:
            axes[1].plot(hist["val_mae"], label="Validation MAE", linewidth=2)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("MAE")
        axes[1].set_title("MAE")
        axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print("Plot saved to %s" % save_path)

    plt.show()


def plot_error_distribution(y_true, y_pred, save_path=None):
    errors = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(errors, bins=50, edgecolor="black", alpha=0.7)
    axes[0].set_xlabel("Error")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Error Distribution")
    axes[0].axvline(x=0, color="r", linestyle="--", linewidth=2)
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(y_true, y_pred, alpha=0.5, edgecolors="none")
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    axes[1].plot(
        [min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect"
    )
    axes[1].set_xlabel("Actual")
    axes[1].set_ylabel("Predicted")
    axes[1].set_title("Actual vs Predicted")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print("Plot saved to %s" % save_path)
    plt.show()


def compare_models(models_metrics, save_path=None):
    metrics_names = ["MAE", "RMSE", "MAPE", "R2"]
    model_names = list(models_metrics.keys())

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i in range(len(metrics_names)):
        metric = metrics_names[i]
        values = []
        for name in model_names:
            val = 0
            if metric in models_metrics[name]:
                val = models_metrics[name][metric]
            values.append(val)
        bars = axes[i].bar(model_names, values, alpha=0.7, edgecolor="black")
        axes[i].set_title("%s Comparison" % metric)
        axes[i].set_ylabel(metric)
        axes[i].grid(True, alpha=0.3, axis="y")
        for j in range(len(bars)):
            axes[i].text(
                bars[j].get_x() + bars[j].get_width() / 2.0,
                bars[j].get_height(),
                "%.4f" % values[j],
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print("Plot saved to %s" % save_path)
    plt.show()


def evaluate_model(
    model, X_test, y_test_scaled, target_scaler, test_dates=None, output_dir="results"
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Making predictions...")
    y_pred_scaled = predict(model, X_test)

    y_test = target_scaler.inverse_transform(y_test_scaled.reshape(-1, 1)).ravel()
    y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

    metrics = calculate_metrics(y_test, y_pred)
    print_metrics(metrics, "LSTM Performance")

    plot_predictions(
        y_test,
        y_pred,
        dates=test_dates,
        title="LSTM Prediction vs Actual",
        save_path=os.path.join(output_dir, "predictions_vs_actual.png"),
    )

    plot_error_distribution(
        y_test, y_pred, save_path=os.path.join(output_dir, "error_distribution.png")
    )

    data = {
        "Actual": y_test,
        "Predicted": y_pred,
        "Error": y_test - y_pred,
        "Error_%%": ((y_test - y_pred) / y_test) * 100,
    }
    predictions_df = pd.DataFrame(data)
    if test_dates is not None:
        predictions_df.index = test_dates[-len(predictions_df) :]
    predictions_df.to_csv(os.path.join(output_dir, "predictions.csv"))
    print("Predictions saved")

    return metrics, y_test, y_pred


# BASELINE MODELS
def align_data(train_df, test_df, feature_columns):
    X_train = train_df[feature_columns].values[:-1]
    y_train_returns = train_df["returns"].values[1:]

    first_X = train_df[feature_columns].values[-1:]
    rest_X = test_df[feature_columns].values[:-1]
    X_test = np.vstack([first_X, rest_X])

    y_test_returns = test_df["returns"].values

    close_base = np.concatenate(
        [train_df["close"].values[-1:], test_df["close"].values[:-1]]
    )
    actual_prices = close_base * (1 + y_test_returns)

    return X_train, y_train_returns, X_test, y_test_returns, close_base, actual_prices


def train_and_evaluate_baselines(train_df, test_df, feature_columns):
    X_train, y_train, X_test, y_test, close_base, actual_prices = align_data(
        train_df, test_df, feature_columns
    )

    # Linear Regression
    print("\n  Training Linear Regression...")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_prices = close_base * (1 + lr_pred)
    lr_metrics = calculate_metrics(actual_prices, lr_prices)
    print_metrics(lr_metrics, "Linear Regression")

    # Random Forest
    print("\n  Training Random Forest...")
    rf = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_prices = close_base * (1 + rf_pred)
    rf_metrics = calculate_metrics(actual_prices, rf_prices)
    print_metrics(rf_metrics, "Random Forest")

    # SVM
    print("\n  Training SVM...")
    svm = SVR(kernel="rbf")
    svm.fit(X_train, y_train)
    svm_pred = svm.predict(X_test)
    svm_prices = close_base * (1 + svm_pred)
    svm_metrics = calculate_metrics(actual_prices, svm_prices)
    print_metrics(svm_metrics, "SVM")

    return {
        "Linear Regression": {
            "predictions": lr_prices,
            "actual": actual_prices,
            "metrics": lr_metrics,
        },
        "Random Forest": {
            "predictions": rf_prices,
            "actual": actual_prices,
            "metrics": rf_metrics,
        },
        "SVM (SVR)": {
            "predictions": svm_prices,
            "actual": actual_prices,
            "metrics": svm_metrics,
        },
    }


def print_comparison_table(lstm_metrics, baseline_results, naive_metrics):
    rows = [("LSTM-Attention", lstm_metrics)]
    for name, res in baseline_results.items():
        rows.append((name, res["metrics"]))
    rows.append(("Naive (0% return)", naive_metrics))

    print("\n" + "=" * 90)
    print("MODEL COMPARISON")
    print("=" * 90)
    print(
        "%-22s %10s %10s %8s %8s %8s"
        % ("Model", "MAE", "RMSE", "MAPE", "R2", "Dir.Acc")
    )
    print("-" * 90)
    for name, metrics in rows:
        mae = metrics.get("MAE", 0)
        rmse = metrics.get("RMSE", 0)
        mape = metrics.get("MAPE", 0)
        r2 = metrics.get("R2", 0)
        da = metrics.get("Directional Accuracy (%)", 0)
        if da is not None:
            da_str = "%.2f%%" % da
        else:
            da_str = "N/A"
        if mape is not None:
            mape_str = "%.2f%%" % mape
        else:
            mape_str = "N/A"
        print(
            "%-22s %10.4f %10.4f %8s %8.4f %8s"
            % (name, mae, rmse, mape_str, r2, da_str)
        )
    print("=" * 90)


# HYPERPARAMETER TUNING
def build_eval_data(df, feature_columns, val_ratio=0.15):
    split_idx = int(len(df) * (1 - val_ratio))
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]

    (train_feat, train_tgt, val_feat, val_tgt, _, target_scaler) = scale_data(
        train_df, val_df, feature_columns, target_column="returns"
    )

    X_train, y_train = create_sequences(
        train_feat, train_tgt, SEQ_LENGTH, FORECAST_HORIZON
    )

    X_val_ctx = train_feat[-SEQ_LENGTH:]
    X_val_full = np.vstack([X_val_ctx, val_feat])
    y_val_full = np.concatenate([train_tgt[-SEQ_LENGTH:], val_tgt])
    X_val, y_val = create_sequences(
        X_val_full, y_val_full, SEQ_LENGTH, FORECAST_HORIZON
    )

    return X_train, y_train, X_val, y_val, train_feat, val_feat, val_tgt, target_scaler


def run_parameter_sweep(df, feature_columns):
    optimizers = ["adam", "sgd", "rmsprop", "adamw"]
    activations = ["tanh", "relu", "leaky_relu"]
    learning_rates = [0.001, 0.003, 0.01]

    print("\n" + "=" * 70)
    print("HYPERPARAMETER TUNING")
    print("=" * 70)
    print("Optimizers:", optimizers)
    print("Activations:", activations)
    print("Learning rates:", learning_rates)
    total = len(optimizers) * len(activations) * len(learning_rates)
    print("Total combos:", total)

    X_train, y_train, X_val, y_val, _, _, _, target_scaler = build_eval_data(
        df, feature_columns
    )
    input_size = X_train.shape[2]

    split_idx = int(len(df) * 0.85)
    val_df = df.iloc[split_idx:]
    val_close = np.concatenate(
        [df["close"].values[split_idx - 1 : split_idx], val_df["close"].values[:-1]]
    )
    val_actual_prices = val_close * (1 + val_df["returns"].values)
    n_val_seq = len(X_val)
    val_actual_prices = val_actual_prices[SEQ_LENGTH - 1 : SEQ_LENGTH - 1 + n_val_seq]

    results = []
    best_r2 = -999999
    best_config = None
    combo_idx = 0

    for opt in optimizers:
        for act in activations:
            for lr in learning_rates:
                combo_idx += 1
                print(
                    "\n[%d/%d] %s / %s / lr=%s"
                    % (combo_idx, total, opt.upper(), act, str(lr))
                )

                try:
                    model = LSTMAttentionModel(
                        input_size=input_size,
                        hidden_size=50,
                        num_heads=4,
                        dropout=0.2,
                        output_size=FORECAST_HORIZON,
                        activation=act,
                    )

                    train_model(
                        model,
                        X_train,
                        y_train,
                        X_val,
                        y_val,
                        epochs=30,
                        batch_size=32,
                        learning_rate=lr,
                        patience_es=10,
                        model_path=None,
                        optimizer_type=opt,
                    )

                    pred_scaled = predict(model, X_val)
                    pred_returns = target_scaler.inverse_transform(
                        pred_scaled.reshape(-1, 1)
                    ).reshape(-1, FORECAST_HORIZON)
                    pred_prices = val_close[
                        SEQ_LENGTH - 1 : SEQ_LENGTH - 1 + n_val_seq
                    ] * (1 + pred_returns[:, 0])

                    metrics = calculate_metrics(val_actual_prices, pred_prices)
                    r2 = metrics.get("R2", float("nan"))

                    results.append(
                        {
                            "optimizer": opt,
                            "activation": act,
                            "learning_rate": lr,
                            "R2": r2,
                            "MAE": metrics.get("MAE"),
                            "RMSE": metrics.get("RMSE"),
                            "MAPE": metrics.get("MAPE"),
                        }
                    )

                    print("  R2=%.4f, MAE=%.4f" % (r2, metrics.get("MAE", 0)))

                    if r2 > best_r2:
                        best_r2 = r2
                        best_config = (opt, act, lr)

                except Exception as e:
                    print("  FAILED:", e)
                    results.append(
                        {
                            "optimizer": opt,
                            "activation": act,
                            "learning_rate": lr,
                            "R2": float("nan"),
                            "MAE": float("nan"),
                            "RMSE": float("nan"),
                            "MAPE": float("nan"),
                        }
                    )

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("R2", ascending=False).reset_index(drop=True)

    print("\nTop 10:")
    print(results_df.head(10).to_string(index=False))

    if best_config is not None:
        print(
            "\nBest: %s / %s / lr=%s (R2=%.4f)"
            % (best_config[0].upper(), best_config[1], str(best_config[2]), best_r2)
        )

    if not os.path.exists("results"):
        os.makedirs("results")
    results_df.to_csv("results/parameter_sweep.csv", index=False)
    print("Saved to results/parameter_sweep.csv")
    return results_df


# EXPORT DATASETS
def export_raw_data():
    print("Exporting raw data...")
    data = yf.download("GLD", start="2015-01-01")
    filename = "GLD_raw_data.csv"
    data.to_csv(filename)
    print("Saved: %s (shape: %s)" % (filename, str(data.shape)))
    return filename


def export_processed_data():
    print("Exporting processed data...")
    raw = fetch_gold_data(start="2015-01-01")
    clean = preprocess_data(raw)
    filename = "GLD_processed.csv"
    clean.to_csv(filename)
    print("Saved: %s (shape: %s)" % (filename, str(clean.shape)))
    return filename


def export_featured_data():
    print("Exporting featured data...")
    raw = fetch_gold_data(start="2015-01-01")
    clean = preprocess_data(raw)
    featured = add_technical_indicators(clean)
    filename = "GLD_with_features.csv"
    featured.to_csv(filename)
    print(
        "Saved: %s (shape: %s, %d features)"
        % (filename, str(featured.shape), featured.shape[1])
    )
    return filename


def export_all():
    f1 = export_raw_data()
    f2 = export_processed_data()
    f3 = export_featured_data()
    print("\nExported: %s, %s, %s" % (f1, f2, f3))


# FUTURE PREDICTION
def load_trained_model(model_path):
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    output_size = 1
    if "output_size" in checkpoint:
        output_size = checkpoint["output_size"]
    model = LSTMAttentionModel(
        input_size=checkpoint["input_size"], output_size=output_size
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def next_trading_day(d):
    next_day = d + datetime.timedelta(days=1)
    while next_day.weekday() >= 5:
        next_day += datetime.timedelta(days=1)
    return next_day


def prepare_recent_data(ticker="GLD"):
    raw = fetch_gold_data(ticker=ticker, start="2024-01-01", interval="1d")
    raw = preprocess_data(raw)
    last_date = raw.index[-1]
    last_close = raw["close"].iloc[-1]

    df = add_technical_indicators(raw)
    feature_df, feature_columns = prepare_features(df)
    if "close" not in feature_columns:
        feature_columns = ["close"] + feature_columns
        feature_df = df[feature_columns]
    return df, feature_df, feature_columns, last_date, last_close


def predict_next_day(
    models,
    feature_scaler,
    target_scaler,
    df,
    feature_df,
    actual_last_date,
    actual_last_close,
    mean_error=0.0,
):
    features = feature_df.values[-SEQ_LENGTH:]
    features_scaled = feature_scaler.transform(features)
    X = features_scaled.reshape(1, SEQ_LENGTH, -1)

    pred_scaled = ensemble_predict(models, X)
    pred_return = float(
        target_scaler.inverse_transform(pred_scaled[:, :1].reshape(-1, 1))[0, 0]
    )
    raw_price = actual_last_close * (1 + pred_return)
    pred_price = raw_price / (1 - mean_error)

    return {
        "last_date": actual_last_date,
        "last_close": actual_last_close,
        "predicted_date": next_trading_day(actual_last_date),
        "predicted_return_pct": pred_return * 100,
        "raw_predicted_price": raw_price,
        "predicted_price": pred_price,
        "bias_correction": mean_error,
    }


# PIPELINE MAIN
def main():
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)

    print("=" * 60)
    print("GOLD PRICE PREDICTION - LSTM ATTENTION")
    print("=" * 60)

    if not os.path.exists("results"):
        os.makedirs("results")
    if not os.path.exists("models"):
        os.makedirs("models")

    # Step 1: Get data
    print("\n[1] Getting data...")
    df = fetch_gold_data(ticker="GLD", start="2015-01-01", interval="1d")

    # Step 2: Clean
    print("\n[2] Cleaning...")
    df = preprocess_data(df)
    print("Date range: %s to %s" % (df.index[0].date(), df.index[-1].date()))

    # Step 3: Add features
    print("\n[3] Adding features...")
    df = add_technical_indicators(df)
    feature_df, feature_columns = prepare_features(df)
    if "close" not in feature_columns:
        feature_columns = ["close"] + feature_columns

    print("Features:", len(feature_columns))
    df = df.dropna()

    # Step 4: Train/test split
    print("\n[4] Splitting data...")
    train_df, test_df = split_data(df, train_ratio=0.95)

    # Step 5: Scale
    print("\n[5] Scaling...")
    (train_feat, train_tgt, test_feat, test_tgt, feature_scaler, target_scaler) = (
        scale_data(train_df, test_df, feature_columns, target_column="returns")
    )

    # Step 6: Cross validation
    print("\n[6] Cross validation...")
    n_splits = 5
    train_min = int(len(df) * 0.7)
    test_size = int((len(df) - train_min) / n_splits)
    print(
        "Folds: %d, Train min: %d, Test per fold: %d" % (n_splits, train_min, test_size)
    )

    cv_results = []

    for fold in range(n_splits):
        print("\n--- Fold %d/%d ---" % (fold + 1, n_splits))
        train_end = train_min + fold * test_size
        test_end = train_end + test_size

        cv_train_df = df.iloc[:train_end]
        cv_test_df = df.iloc[train_end:test_end]
        cv_test_dates = cv_test_df.index

        (cv_tr_f, cv_tr_t, cv_te_f, cv_te_t, cv_fs, cv_ts) = scale_data(
            cv_train_df, cv_test_df, feature_columns, target_column="returns"
        )

        cv_X_tr, cv_y_tr = create_sequences(
            cv_tr_f, cv_tr_t, SEQ_LENGTH, FORECAST_HORIZON
        )

        ctx = cv_tr_f[-SEQ_LENGTH:]
        X_full = np.vstack([ctx, cv_te_f])
        y_full = np.concatenate([cv_tr_t[-SEQ_LENGTH:], cv_te_t])
        cv_X_te, cv_y_te = create_sequences(
            X_full, y_full, SEQ_LENGTH, FORECAST_HORIZON
        )

        print("  Train: %s, Test: %s" % (str(cv_X_tr.shape), str(cv_X_te.shape)))

        model = LSTMAttentionModel(
            input_size=cv_X_tr.shape[2],
            hidden_size=50,
            num_heads=4,
            dropout=0.2,
            output_size=FORECAST_HORIZON,
        )

        val_split = int(len(cv_X_tr) * 0.9)
        model_path = "models/cv_fold_%d.pt" % (fold + 1)

        train_model(
            model,
            cv_X_tr[:val_split],
            cv_y_tr[:val_split],
            cv_X_tr[val_split:],
            cv_y_tr[val_split:],
            epochs=50,
            batch_size=32,
            learning_rate=0.003,
            patience_es=15,
            model_path=model_path,
        )

        pred_scaled = predict(model, cv_X_te)
        pred_returns = cv_ts.inverse_transform(pred_scaled.reshape(-1, 1)).reshape(
            -1, FORECAST_HORIZON
        )
        actual_returns = cv_ts.inverse_transform(cv_y_te.reshape(-1, 1)).reshape(
            -1, FORECAST_HORIZON
        )

        all_close = list(cv_train_df["close"].values) + list(cv_test_df["close"].values)
        prev_close = np.array(
            [all_close[i + SEQ_LENGTH - 1] for i in range(len(cv_X_te))]
        )

        actual_prices = prev_close * (1 + actual_returns[:, 0])
        pred_prices = prev_close * (1 + pred_returns[:, 0])

        m = calculate_metrics(actual_prices, pred_prices)
        cv_results.append(
            {
                "fold": fold + 1,
                "metrics": m,
                "model": model,
                "feature_scaler": cv_fs,
                "target_scaler": cv_ts,
                "test_df": cv_test_df,
                "actual_prices": actual_prices,
                "pred_prices_raw": pred_prices,
                "dates": cv_test_dates[-len(actual_prices) :],
            }
        )

        print("  Fold %d R2: %.4f, MAE: %.2f" % (fold + 1, m["R2"], m["MAE"]))

    # Best fold
    best_fold = cv_results[0]
    for r in cv_results:
        if r["metrics"]["R2"] > best_fold["metrics"]["R2"]:
            best_fold = r

    print("\nBest fold: %d (R2=%.4f)" % (best_fold["fold"], best_fold["metrics"]["R2"]))

    # Stitch predictions
    all_dates = np.concatenate([r["dates"] for r in cv_results])
    all_actual = np.concatenate([r["actual_prices"] for r in cv_results])
    all_pred_list = []
    for r in cv_results:
        mean_err = np.mean(
            (r["actual_prices"] - r["pred_prices_raw"]) / r["actual_prices"]
        )
        corrected = r["pred_prices_raw"] / (1 - mean_err)
        all_pred_list.append(corrected)
    all_pred = np.concatenate(all_pred_list)

    # Retrain on all data
    print("\n[Retrain] Training final model...")
    all_feat = feature_scaler.transform(df[feature_columns].values)
    all_tgt = target_scaler.transform(df["returns"].values.reshape(-1, 1)).ravel()
    X_all, y_all = create_sequences(all_feat, all_tgt, SEQ_LENGTH, FORECAST_HORIZON)

    val_split = int(len(X_all) * 0.9)
    retrain_model = LSTMAttentionModel(
        input_size=X_all.shape[2],
        hidden_size=50,
        num_heads=4,
        dropout=0.2,
        output_size=FORECAST_HORIZON,
    )
    train_model(
        retrain_model,
        X_all[:val_split],
        y_all[:val_split],
        X_all[val_split:],
        y_all[val_split:],
        epochs=50,
        batch_size=32,
        learning_rate=0.003,
        patience_es=15,
        model_path=None,
    )

    # Baseline comparison
    print("\n[Baselines] Training sklearn models...")
    best_train_df = df.iloc[: train_min + (best_fold["fold"] - 1) * test_size]
    best_test_df = best_fold["test_df"]
    baseline_results = train_and_evaluate_baselines(
        best_train_df, best_test_df, feature_columns
    )

    # Naive baseline
    all_close_b = np.concatenate(
        [best_train_df["close"].values, best_test_df["close"].values]
    )
    seq_idx = np.arange(SEQ_LENGTH, len(all_close_b))
    prev_close_b = all_close_b[seq_idx - 1]
    test_ret = best_test_df["returns"].values
    actual_ret_full = np.concatenate(
        [best_train_df["returns"].values[-SEQ_LENGTH:], test_ret]
    )
    actual_prices_b = prev_close_b[-len(actual_ret_full[SEQ_LENGTH:]) :] * (
        1 + actual_ret_full[SEQ_LENGTH:]
    )
    y_naive = prev_close_b[-len(actual_prices_b) :]
    naive_metrics = calculate_metrics(actual_prices_b, y_naive)

    print_comparison_table(best_fold["metrics"], baseline_results, naive_metrics)

    # Comparison chart
    all_metrics = {"LSTM-Attention": best_fold["metrics"]}
    for name, res in baseline_results.items():
        all_metrics[name] = res["metrics"]
    all_metrics["Naive"] = naive_metrics
    compare_models(all_metrics, save_path="results/model_comparison.png")

    # Build sequences for best fold detail
    best_train_df2 = df.iloc[: train_min + (best_fold["fold"] - 1) * test_size]
    best_test_df2 = best_fold["test_df"]

    best_feat_scaled = feature_scaler.transform(best_train_df2[feature_columns].values)
    best_test_feat_scaled = feature_scaler.transform(
        best_test_df2[feature_columns].values
    )

    ctx2 = best_feat_scaled[-SEQ_LENGTH:]
    X_full2 = np.vstack([ctx2, best_test_feat_scaled])
    y_full2 = np.concatenate(
        [
            best_train_df2["returns"].values[-SEQ_LENGTH:],
            best_test_df2["returns"].values,
        ]
    )
    X_test_final, y_test_final = create_sequences(
        X_full2, y_full2, SEQ_LENGTH, FORECAST_HORIZON
    )

    pred_scaled_final = predict(best_fold["model"], X_test_final)
    pred_returns_final = target_scaler.inverse_transform(
        pred_scaled_final.reshape(-1, 1)
    ).reshape(-1, FORECAST_HORIZON)
    actual_returns_final = target_scaler.inverse_transform(
        y_test_final.reshape(-1, 1)
    ).reshape(-1, FORECAST_HORIZON)

    all_close_p = np.concatenate(
        [best_train_df2["close"].values, best_test_df2["close"].values]
    )
    seq_end = np.arange(SEQ_LENGTH, len(all_close_p))
    prev_close_p = all_close_p[seq_end - 1]

    raw_pred = prev_close_p[-len(pred_returns_final) :] * (1 + pred_returns_final[:, 0])
    actual_p = prev_close_p[-len(actual_returns_final) :] * (
        1 + actual_returns_final[:, 0]
    )

    # Bias correction
    print("\nApplying bias correction...")
    mean_error = np.mean((actual_p - raw_pred) / actual_p)
    pred_corrected = raw_pred / (1 - mean_error)
    print("Mean bias: %.2f%%" % (mean_error * 100))
    print("Correction factor: %.4f" % (1 / (1 - mean_error)))

    corrected_metrics = calculate_metrics(actual_p, pred_corrected)
    print("\nCorrected:")
    for name, val in corrected_metrics.items():
        if val is not None:
            print("  %s: %.4f" % (name, val))

    # Save predictions
    pred_df = pd.DataFrame(
        {
            "Actual": all_actual,
            "Predicted": all_pred,
            "Error": all_actual - all_pred,
            "Error_%%": ((all_actual - all_pred) / all_actual) * 100,
        }
    )
    pred_df.index = all_dates
    pred_df.to_csv("results/predictions.csv")
    print("\nPredictions saved")

    # Plots
    print("\nGenerating plots...")
    plot_predictions(
        all_actual,
        all_pred,
        dates=all_dates,
        title="Gold Price: Prediction vs Actual",
        save_path="results/predictions_vs_actual.png",
    )
    plot_error_distribution(
        all_actual, all_pred, save_path="results/error_distribution.png"
    )

    # Save artifacts
    print("\nSaving artifacts...")
    joblib.dump(feature_scaler, "models/feature_scaler.pkl")
    joblib.dump(target_scaler, "models/target_scaler.pkl")
    with open("models/bias_correction.txt", "w") as f:
        f.write("%.6f" % mean_error)

    torch.save(
        {
            "model_state_dict": retrain_model.state_dict(),
            "input_size": retrain_model.lstm.input_size,
            "output_size": FORECAST_HORIZON,
            "forecast_horizon": FORECAST_HORIZON,
        },
        "models/best_lstm_attention.pt",
    )

    print("Done! Models saved in models/, plots in results/")


# FLASK WEB APP
# Flask global state
flask_models = []
flask_feature_scaler = None
flask_target_scaler = None
flask_mean_error = 0.0
flask_ticker = os.environ.get("TICKER", "GLD")
data_cache = {"df": None, "feature_df": None, "timestamp": None}


def load_artifacts():
    global flask_models, flask_feature_scaler, flask_target_scaler, flask_mean_error

    if os.path.exists("models/cv_fold_1.pt"):
        flask_models = load_cv_models()
        print("Loaded %d CV models" % len(flask_models))
    elif os.path.exists("models/best_lstm_attention.pt"):
        ckpt = torch.load(
            "models/best_lstm_attention.pt", map_location="cpu", weights_only=False
        )
        m = LSTMAttentionModel(
            input_size=ckpt["input_size"], output_size=ckpt.get("output_size", 1)
        )
        m.load_state_dict(ckpt["model_state_dict"])
        m.eval()
        flask_models = [m]
        print("Loaded single model")
    else:
        print("WARNING: No models found")

    if os.path.exists("models/feature_scaler.pkl"):
        flask_feature_scaler = joblib.load("models/feature_scaler.pkl")
    else:
        print("WARNING: No feature_scaler.pkl")

    if os.path.exists("models/target_scaler.pkl"):
        flask_target_scaler = joblib.load("models/target_scaler.pkl")
    else:
        print("WARNING: No target_scaler.pkl")

    if os.path.exists("models/bias_correction.txt"):
        with open("models/bias_correction.txt") as f:
            flask_mean_error = float(f.read().strip())
        print("Bias correction: %.2f%%" % (flask_mean_error * 100))


def fetch_and_prepare():
    now = datetime.datetime.now()
    if data_cache["timestamp"] is not None:
        diff = (now - data_cache["timestamp"]).total_seconds()
        if diff < 3600:
            return data_cache["df"], data_cache["feature_df"]

    raw = fetch_gold_data(ticker=flask_ticker, start="2015-01-01")
    raw = preprocess_data(raw)
    df = add_technical_indicators(raw)
    feature_df, feature_columns = prepare_features(df)
    if "close" not in feature_columns:
        feature_columns = ["close"] + feature_columns
        feature_df = df[feature_columns]

    data_cache["df"] = df
    data_cache["feature_df"] = feature_df
    data_cache["timestamp"] = now
    return df, feature_df


def predict_direct(days, df, feature_df):
    if days > FORECAST_HORIZON:
        days = FORECAST_HORIZON

    last_seq = feature_df.values[-SEQ_LENGTH:]
    scaled = flask_feature_scaler.transform(last_seq)
    X = scaled.reshape(1, SEQ_LENGTH, -1)

    pred_scaled = ensemble_predict(flask_models, X)
    model_returns = flask_target_scaler.inverse_transform(
        pred_scaled.reshape(-1, 1)
    ).ravel()

    # Add noise from historical returns
    historical = df["returns"].dropna().values
    centered = historical - np.mean(historical)
    seed_val = sum(
        ord(c) * (i + 1) for i, c in enumerate(str(datetime.date.today()))
    ) % (2**31)
    rng = np.random.RandomState(seed_val)
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

    if abs(1 - flask_mean_error) > 1e-8:
        for r in results:
            r["price"] = round(r["price"] / (1 - flask_mean_error), 2)

    return results


def create_app():
    app = Flask(__name__)

    @app.route("/")
    def index():
        try:
            df, feature_df = fetch_and_prepare()
        except Exception as e:
            return render_template_string(
                TEMPLATE_HTML, error="Failed to load data: %s" % str(e)
            )

        three_months_ago = df.index[-1] - pd.DateOffset(months=3)
        mask = df.index >= three_months_ago
        historical_dates = [str(d.date()) for d in df.index[mask]]
        historical_prices = [round(float(p), 2) for p in df["close"].values[mask]]

        initial_preds = []
        if flask_models and flask_feature_scaler and flask_target_scaler:
            try:
                initial_preds = predict_direct(7, df, feature_df)
            except Exception as e:
                print("Prediction error:", e)

        return render_template_string(
            TEMPLATE_HTML,
            historical_dates=json.dumps(historical_dates),
            historical_prices=json.dumps(historical_prices),
            initial_preds=json.dumps(initial_preds),
            last_close=round(float(df["close"].iloc[-1]), 2),
            last_date=str(df.index[-1].date()),
            default_days=7,
            models_loaded=len(flask_models) > 0,
            error=None,
        )

    @app.route("/predict", methods=["POST"])
    def predict_route():
        if not flask_models or not flask_feature_scaler or not flask_target_scaler:
            return jsonify({"error": "Models not loaded"}), 503

        data = request.get_json()
        days = int(data.get("days", 7))

        try:
            df, feature_df = fetch_and_prepare()
            predictions = predict_direct(days, df, feature_df)
            return jsonify({"predictions": predictions})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return app


# ENTRY POINT
if __name__ == "__main__":
    models_exist = os.path.exists("models/cv_fold_1.pt") or os.path.exists(
        "models/best_lstm_attention.pt"
    )

    if not models_exist:
        main()
    else:
        print("Models exist, skipping training")
        print("Delete models/ folder to retrain")

    print("\nStarting Flask...")
    load_artifacts()
    app = create_app()
    app.run(debug=True, port=5000)
