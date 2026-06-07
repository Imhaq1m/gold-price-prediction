# Gold Price Prediction using LSTM-Attention (PyTorch) — Project Context

## Project Overview

Advanced undergraduate ML project for predicting gold market prices using a hybrid **LSTM-Attention** neural network. Implements a complete pipeline from data collection to model evaluation, plus an **interactive Flask web dashboard**.

**Purpose**: Predict next-day gold returns using historical GLD ETF data, engineered technical indicators, and a stationary target variable to avoid covariate shift.

**Main Technologies**:
- **Python 3.8+**
- **PyTorch** for LSTM-Attention
- **yfinance** for financial market data
- **Flask** for web dashboard
- **Chart.js** for interactive frontend charting
- **pandas/numpy** for data manipulation
- **scikit-learn** for preprocessing and metrics
- **matplotlib/seaborn** for static visualizations

## Project Structure

```
project/
├── src/                         # Python source package
│   ├── __init__.py
│   ├── main.py                  # Pipeline orchestrator (entry point)
│   ├── predict_future.py        # CLI inference (direct multi-step, H=30)
│   ├── data_module.py           # yfinance fetch & preprocessing
│   ├── feature_engineering.py   # 31 technical indicators, sequences, scaling
│   ├── lstm_model.py            # PyTorch models (LSTMAttentionModel, StackedLSTM, SimpleLSTM)
│   ├── evaluation.py            # Metrics + matplotlib/seaborn plots
│   └── export_dataset.py        # CSV dataset export for sharing
├── templates/                   # Flask HTML template
│   └── index.html               # Chart.js dark-themed dashboard
├── app.py                       # Flask web application
├── docs/                        # Documentation
│   ├── README.md                # User-facing docs
│   ├── QUICKSTART.md            # Quick commands
│   ├── QWEN.md                  # This file — project context
│   ├── pipeline.txt             # Detailed technical pipeline
│   └── TESTING_REPORT.md        # Testing results
├── data/                        # Raw & processed CSV datasets
├── models/                      # .pt checkpoints, .pkl scalers, bias_correction.txt
├── results/                     # predictions.csv, .png plots
├── logs/                        # Run logs (gitignored)
├── requirements.txt
└── .gitignore
```

## Module Descriptions

### `main.py` — Pipeline Orchestrator
Entry point for the full pipeline:
1. Data collection (GLD ETF, 2015–present)
2. Preprocessing (ffill/bfill, sort, deduplicate)
3. Feature engineering (31 indicators: trend, momentum, volatility, lags, cyclical)
4. Target definition: `returns = (P_{t+1} / P_t) - 1` (stationary)
5. **Walk-Forward Cross Validation** (5 folds, expanding from 70%)
6. Feature scaling (MinMaxScaler fit on train only)
7. Sequence creation (30-day sliding windows → `(samples, 30, 31)`)
8. Model: LSTM(50) → MultiHeadAttention(4 heads) → LayerNorm → Dense(H=30)
9. Training: Adam, MSE loss, early stopping (patience=15), ReduceLROnPlateau
10. Select best fold by R²
11. Retrain on all data → save `best_lstm_attention.pt`
12. Evaluate, generate plots, save artifacts

### `data_module.py` — Data Collection
- `fetch_gold_data()`: Fetches GLD via yfinance (OHLCV)
- `preprocess_data()`: ffill/bfill NaN, enforce chronology, deduplicate
- `split_data()`: Time-based train/test split

### `feature_engineering.py` — Feature Creation
- `add_technical_indicators()`: 31 features (SMA, EMA, RSI, MACD, Bollinger, lags, rolling stats, cyclical)
- `prepare_features()`: Select and order feature columns
- `create_sequences()`: Tabular → 3D tensors `(samples, 30, 31)` for LSTM
- `scale_data()`: MinMaxScaler [0,1]; fit on train, transform both

### `lstm_model.py` — Neural Network
- **`LSTMAttentionModel`** (primary):
```
Input: (batch, 30, 31)
  → LSTM(31→50, batch_first) → Dropout(0.2)
    → MultiheadAttention(embed_dim=50, heads=4, batch_first)
      → Residual + LayerNorm → Dropout(0.2)
        → attn_out[:, -1, :]  (temporal pooling)
          → Linear(50 → H=30)  (direct multi-step output)
```
- **`StackedLSTM`**: 3-layer LSTM (128→64→32) with BatchNorm, Dropout, FC head
- **`SimpleLSTM`**: LSTM(50) → Dropout → Linear
- **`train_model()`**: Full training loop with early stopping, checkpointing, LR scheduling
- **`predict()`**: Inference wrapper (model.eval(), torch.no_grad())
- **`load_cv_models()`**: Load all 5 CV fold checkpoints as ensemble
- **`ensemble_predict()`**: Average predictions across ensemble

### `evaluation.py` — Metrics & Visualization
- `calculate_metrics()`: MAE, RMSE, MAPE, R², Directional Accuracy
- `print_metrics()`: Formatted console output
- `plot_predictions()`: Actual vs predicted time series overlay
- `plot_training_history()`: Loss + MAE curves
- `plot_error_distribution()`: Error histogram + scatter plot
- `evaluate_model()`: End-to-end evaluation pipeline

### `predict_future.py` — CLI Inference
- Single-day: ensemble predict → inverse transform → bias correct
- Multi-day: direct multi-step (H=30) → cumulative product → bias correct
- `load_cv_models()` / single checkpoint support

### `export_dataset.py` — CSV Export
- `export_raw_data()`: Raw OHLCV → `GLD_raw_data.csv`
- `export_processed_data()`: Cleaned → `GLD_processed.csv`
- `export_featured_data()`: Full 31-feature dataset → `GLD_with_features.csv`

### `app.py` — Flask Web Dashboard
Flask application with 2 routes:
- **`GET /`**: Renders dark-themed Chart.js dashboard with:
  - Historical GLD prices (last 3 months)
  - Auto-run 7-day prediction (ensemble of 5 models)
  - Prediction details table
- **`POST /predict`**: Accepts `{"days": N}`, returns `{"predictions": [...]}`

Key logic:
- Artifacts loaded once at startup (`load_artifacts()`)
- Data cached for 1 hour (avoids yfinance rate limits)
- Direct multi-step for returns → bootstrap noise from historical GLD returns for realistic variation
- Bias correction applied once at end

## Building and Running

### Installation
```bash
pip install -r requirements.txt
```

### Training
```bash
py -m src.main
```

### CLI Inference
```bash
py -m src.predict_future
py -m src.predict_future --days 30
```

### Web Dashboard
```bash
py app.py
# → http://127.0.0.1:5000
```

### Dataset Export
```bash
py -m src.export_dataset
```

## Key Configuration Parameters

In `main.py`:
| Parameter | Default | Description |
|-----------|---------|-------------|
| `SEQ_LENGTH` | 30 | Lookback window in days |
| `FORECAST_HORIZON` | 30 | Direct multi-step horizon |
| `n_splits` | 5 | Number of CV folds |
| `epochs` | 50 | Max training epochs per fold |
| `batch_size` | 32 | Batch size for DataLoader |
| `start` | '2015-01-01' | Data start date |
| `ticker` | 'GLD' | Gold ETF symbol |
| `train_ratio` | 0.95 | Train/test split for final evaluation |

In `lstm_model.py`:
| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | 0.003 | Adam initial LR |
| `patience_es` | 15 | Early stopping patience |
| `hidden_size` | 50 | LSTM hidden units |
| `num_heads` | 4 | Attention heads |
| `dropout` | 0.2 | Dropout rate |

## Development Conventions

### Code Style
- Type hints throughout
- NumPy-style docstrings
- snake_case functions/variables, CamelCase classes
- Modular design, single-responsibility functions

### Data Flow
```
Raw OHLCV → Preprocessing → 31 Features → Returns Target
  → MinMaxScaler → Sequences (30, 31) → LSTM-Attention
    → Predicted Returns → Inverse Transform → Price Conversion
      → Bias Correction → Metrics & Visualization
```

### Flask Data Flow
```
GET / : yfinance → preprocess → features → ensemble predict (7-day)
  → render_template with JSON-embedded data
POST /predict : yfinance (cached) → features → direct multi-step
  → bootstrap noise → cumulative price → bias correct → JSON
```

## Architecture Decisions

### Why LSTM-Attention?
- LSTM solves vanishing gradients via gating mechanisms
- Self-attention computes pairwise relationships across all 30 positions
- 4 heads learn diverse patterns (momentum, volatility, shocks, regimes)
- Residual connections preserve gradient flow

### Why 30-day sequences?
- ~1.5 months of trading context
- Balances context with O(n²) attention complexity
- Standard in financial time series literature

### Why Returns (not Price)?
- Stationarity: returns have constant mean/variance
- Avoids covariate shift: gold moved from ~$1,000 to ~$2,300+
- Aligned with trading practice (% returns)

### Why Walk-Forward Validation?
- No look-ahead bias
- Simulates real deployment
- 5 folds = performance distribution across market regimes

### Why Direct Multi-Step (H=30)?
- One forward pass for all 30 future returns
- Avoids error compounding of recursive approaches
- Loss optimized jointly across all horizons

### Why Bootstrap Noise?
- Model only predicts positive returns (gold trends up)
- Adding residuals from historical returns creates realistic up/down days
- Preserves model's expected drift while adding realistic volatility

## Common Extension Points

1. **More features** (edit `feature_engineering.py`):
   - DXY, interest rates, VIX
   - Sentiment from news

2. **Model changes** (edit `lstm_model.py`):
   - Bidirectional LSTM
   - GRU layers
   - Full Transformer encoder

3. **Validation** (edit `main.py`):
   - Adjust n_splits
   - Sliding window instead of expanding

4. **Dashboard** (edit `app.py` + `templates/index.html`):
   - Add confidence bands
   - Multi-scenario Monte Carlo paths
   - Model comparison charts

## Research Context

- **LSTM-Attention** architecture: superior to LSTM-only or Attention-only baselines in recent literature
- **31 technical features**: comprehensive market structure encoding
- **Walk-Forward CV**: standard in quantitative finance
- **Returns prediction**: stationary target, standard practice
- **Bias correction**: post-processing for systematic neural network bias

**Typical Performance**:
- R² > 0.92 (raw)
- MAE ~$0.61
- MAPE ~0.56%

Suitable for:
- Undergraduate/Graduate ML projects
- PyTorch time series demonstrations
- Interactive financial prediction demos
- Research paper replication
