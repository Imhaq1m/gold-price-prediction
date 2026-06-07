# Gold Price Prediction using LSTM-Attention

An undergraduate Machine Learning project for predicting gold (GLD ETF) prices using LSTM networks with Multi-Head Self-Attention.

## Project Structure

```
project/
├── src/                          # Python source package
│   ├── __init__.py
│   ├── main.py                   # Pipeline orchestrator (training + evaluation)
│   ├── predict_future.py         # CLI inference (direct multi-step, H=30)
│   ├── data_module.py            # Data collection (yfinance) & preprocessing
│   ├── feature_engineering.py    # 31 technical indicators & sequence creation
│   ├── lstm_model.py             # LSTM-Attention, StackedLSTM, SimpleLSTM
│   ├── evaluation.py             # Metrics (MAE, RMSE, R²) & visualizations
│   └── export_dataset.py         # CSV dataset export for sharing
├── templates/                    # Flask HTML template (Chart.js)
│   └── index.html                # Dark-themed dashboard with line chart
├── app.py                        # Flask web dashboard (GET / + POST /predict)
├── docs/                         # Documentation
│   ├── README.md                 # This file
│   ├── QUICKSTART.md
│   ├── QWEN.md
│   ├── pipeline.txt
│   └── TESTING_REPORT.md
├── data/                         # Raw & processed CSV datasets
├── models/                       # Trained model checkpoints & scalers
├── results/                      # Predictions CSV & plots
├── logs/                         # Run logs (gitignored)
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

Runs the full pipeline:
1. Fetch GLD data from Yahoo Finance (2015–present)
2. Compute 31 technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, etc.)
3. Train LSTM-Attention model with 5-fold walk-forward cross-validation
4. Retrain on all data, evaluate, generate plots

### 2. CLI Prediction

```bash
py -m src.predict_future                          # single day (ensemble)
py -m src.predict_future --days 30                # multi-day (direct multi-step)
py -m src.predict_future --model models/cv_fold_3.pt   # single model
```

### 3. Web Dashboard

```bash
py app.py
# → http://127.0.0.1:5000
```

Opens an interactive Flask dashboard with:
- Historical GLD price chart (last 3 months)
- Auto-run 7-day LSTM prediction on load
- Adjustable prediction horizon (1–30 days)
- Bootstrap noise from historical returns for realistic day-to-day variation
- Results table with day-by-day returns and prices

---

## Model Architecture

### LSTM-Attention (best performer)

```
Input (30 days × 31 features)
  → LSTM(50 hidden)
    → Dropout(0.2)
      → Multi-Head Self-Attention (4 heads, key_dim=50)
        → Residual + LayerNorm
          → Dense(H=30 outputs)  ← direct multi-step
```

Output shape: (batch, H=30) — predicts 30 future daily returns in one forward pass.

### Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| **Target** | Returns (stationary) | Avoids covariate shift from non-stationary prices |
| **Validation** | Walk-forward CV (5 folds) | Simulates real deployment — trains only on past data |
| **Lookback** | 30 trading days | ~6 weeks of market context |
| **Forecast horizon** | 30 days (direct multi-step) | Predicts all 30 future returns at once; avoids error compounding of recursive |
| **Bias correction** | Post-hoc shift by mean error | Fixes systematic over/under-prediction |
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
- flask (for web dashboard)
- matplotlib, seaborn

## Research Basis

This project implements LSTM-Attention architectures from financial time series forecasting literature. The multi-head self-attention mechanism helps the model focus on relevant historical patterns when making predictions.

## Notes

- The model predicts **returns**, not raw prices — returns are stationary, prices are not
- Multi-day predictions use direct multi-step (one forward pass for H=30 outputs); bootstrap noise adds realistic day-to-day volatility
- The web dashboard applies bootstrap residuals sampled from historical GLD returns to produce jagged, realistic-looking price trajectories
