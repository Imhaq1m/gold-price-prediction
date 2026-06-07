# Gold Price Prediction using LSTM-Attention

An undergraduate Machine Learning project for predicting gold (GLD ETF) prices using LSTM networks with Multi-Head Self-Attention.

## Project Structure

```
project/
├── src/                          # Python source code
│   ├── main.py                   # Pipeline orchestrator (training + evaluation)
│   ├── predict_future.py         # Predict future prices outside the dataset
│   ├── data_module.py            # Data collection (yfinance) & preprocessing
│   ├── feature_engineering.py    # 31 technical indicators & sequence creation
│   ├── lstm_model.py             # LSTM-Attention, StackedLSTM, SimpleLSTM
│   ├── evaluation.py             # Metrics (MAE, RMSE, R²) & visualizations
│   ├── export_dataset.py         # Dataset export utility
│   └── __init__.py
├── notebooks/
│   └── gold_prediction_colab.ipynb  # Google Colab variant
├── data/                         # Raw & processed CSV datasets
├── docs/                         # Documentation
│   ├── README.md                 # This file
│   ├── QUICKSTART.md
│   ├── QWEN.md
│   └── TESTING_REPORT.md
├── models/                       # Trained model checkpoints & scalers
├── results/                      # Predictions CSV & plots
├── logs/                         # Run logs
├── requirements.txt
└── .gitignore
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Train & Evaluate

```bash
py -m src.main
```

This runs the full pipeline:
1. Fetch GLD data from Yahoo Finance (2015–present)
2. Compute 31 technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, etc.)
3. Train LSTM-Attention model with 5-fold walk-forward cross-validation
4. Evaluate on held-out data and generate plots

### 2. Predict Future Prices

After training, predict the next trading day's price:

```bash
py -m src.predict_future
```

Predict 30 days ahead (recursive, feature-aware):

```bash
py -m src.predict_future --days 30
```

Use a specific CV fold:

```bash
py -m src.predict_future --model models/cv_fold_3.pt
```

---

## Model Architecture

### LSTM-Attention (best performer)

```
Input (30 days × 31 features)
  → LSTM(50 hidden)
    → Dropout(0.2)
      → Multi-Head Self-Attention (4 heads, key_dim=50)
        → Residual + LayerNorm
          → Dense(1 output)
```

### Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| **Target** | Returns (stationary) | Avoids covariate shift from non-stationary prices |
| **Validation** | Walk-forward CV (5 folds) | Simulates real deployment — trains only on past data |
| **Lookback** | 30 trading days | ~6 weeks of market context |
| **Bias correction** | Post-hoc shift by mean error | Fixes systematic overprediction |
| **Features** | 31 (trend, momentum, volatility, price action, lags, cyclical) | Comprehensive without overfitting |

## Performance

| Metric | LSTM-Attention | Paper Baseline |
|---|---|---|
| **R²** | ~0.96 | 0.92 |
| **MAE** | ~$0.61 | $14.46 |
| **RMSE** | ~$0.81 | — |
| **MAPE** | ~0.56% | — |

## Features

| Category | Indicators |
|---|---|
| **Trend** | SMA(10/20/50), EMA(10/20), MA crossovers |
| **Momentum** | RSI(14), MACD, MACD Signal, MACD Histogram |
| **Volatility** | Bollinger Bands (width, position), rolling std |
| **Price Action** | Log returns, HL range, OC range |
| **Lag** | Close/returns at t-1, t-2, t-3, t-5, t-10 |
| **Cyclical** | Day of week, month, quarter |

## Requirements

- Python 3.8+
- PyTorch 2.0+
- yfinance, scikit-learn, pandas, numpy
- matplotlib, seaborn

## Research Basis

This project implements LSTM-Attention architectures from financial time series forecasting literature. The multi-head self-attention mechanism helps the model focus on relevant historical patterns when making predictions.

## Notes

- The model predicts **returns**, not raw prices — returns are stationary, prices are not
- Multi-day predictions beyond 5 days are illustrative; error compounds recursively

