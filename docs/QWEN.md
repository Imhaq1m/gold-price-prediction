# Gold Price Prediction using LSTM-Attention (PyTorch) - Project Context

## Project Overview

This is an advanced undergraduate-level Machine Learning project for predicting gold market prices using a hybrid **LSTM-Attention** neural network. The project implements a complete ML pipeline from data collection to model evaluation, utilizing **PyTorch** and Walk-Forward Cross Validation for robust performance.

**Purpose**: Predict next-day gold returns using historical GLD ETF data, engineered technical indicators, and a stationary target variable to avoid covariate shift across price regimes.

**Main Technologies**:
- **Python 3.8+** as the programming language
- **PyTorch** for LSTM-Attention neural network implementation
- **yfinance** for fetching financial market data
- **pandas/numpy** for data manipulation
- **scikit-learn** for preprocessing (MinMaxScaler) and metrics
- **matplotlib/seaborn** for visualization

## Project Structure

```
project/
├── main.py                    # Main orchestrator script (entry point)
├── data_module.py             # Data collection from yfinance & preprocessing
├── feature_engineering.py     # Technical indicators & sequence creation
├── lstm_model.py              # LSTM-Attention model architecture & training
├── evaluation.py              # Metrics calculation & visualization
├── pipeline.txt               # Detailed technical documentation
├── requirements.txt           # Python dependencies
├── QWEN.md                    # This file - project context
├── README.md                  # User-facing documentation
├── docs/                      # Supplementary documentation
├── logs/                      # Historical execution logs
├── models/                    # Saved trained models (created at runtime)
└── results/                   # Output plots & predictions (created at runtime)
```

## Module Descriptions

### `main.py` - Pipeline Orchestrator
The main entry point that runs the complete pipeline:
1. Data collection (GLD ETF from Yahoo Finance, 2015-present)
2. Preprocessing (forward/backward fill for missing values, chronological sort, deduplication)
3. Feature engineering (31 technical indicators across trend, momentum, volatility, lag, cyclical categories)
4. Target definition (Convert prices to stationary returns: `(P_{t+1} / P_t) - 1`)
5. **Walk-Forward Cross Validation** (5 folds, expanding window starting at 70% of data)
6. Feature scaling (MinMaxScaler fit on train, applied to test to prevent leakage)
7. Sequence creation (**30-day sliding windows** → shape `(samples, 30, 31)`)
8. Model building (LSTM-Attention: LSTM(50) → MultiHeadAttention(4 heads) → LayerNorm+Residual → Dropout(0.2) → Dense(1))
9. Training (Adam optimizer, MSE loss, early stopping patience=15, ReduceLROnPlateau, gradient clipping max_norm=1.0)
10. Fold selection (best R² across 5 folds)
11. Returns-to-price conversion: `Predicted_Price = Previous_Close * (1 + Predicted_Return)`
12. **Bias Correction** (post-processing shift by mean error)
13. Evaluation (R², MAE, RMSE, MAPE, Directional Accuracy) and visualization generation

### `data_module.py` - Data Collection
- **`fetch_gold_data()`**: Fetches GLD ETF data via yfinance (OHLCV, dividends, splits)
- **`preprocess_data()`**: Handles missing values (ffill → bfill), enforces chronological order, deduplicates timestamps, sets DatetimeIndex
- **`split_data()`**: Time-based train/test split (default 95/5 for final evaluation)

### `feature_engineering.py` - Feature Creation
- **`add_technical_indicators()`**: Creates **31 features** across 5 categories:
  - **Trend**: SMA(10/20/50), EMA(10/20), crossovers
  - **Momentum**: RSI(14), MACD, MACD Signal, MACD Histogram
  - **Volatility**: Bollinger Bands (width, position), rolling std dev
  - **Price Action**: Log returns, high-low range, open-close range
  - **Lag Features**: Close/Returns at t-1, t-2, t-3, t-5, t-10
  - **Cyclical**: Day of week, month, quarter
- **`prepare_features()`**: Selects and orders feature columns
- **`create_sequences()`**: Converts tabular data to 3D tensors `(samples, seq_len=30, features=31)` for LSTM input
- **`scale_data()`**: MinMaxScaler normalization to [0, 1]; fit on train, transform on both train and test

### `lstm_model.py` - Neural Network
- **`LSTMAttentionModel`**: Primary architecture
  ```
  Input: (batch, 30, 31)
    ↓
  LSTM(input=31, hidden=50, batch_first=True)
    → Output: (batch, 30, 50), Dropout(0.2)
    ↓
  MultiheadAttention(embed_dim=50, num_heads=4, batch_first=True)
    → Attention: softmax(QK^T / sqrt(d_k)) @ V
    → Residual: attn_out + lstm_out
    → LayerNorm(hidden=50), Dropout(0.2)
    ↓
  Temporal Pooling: attn_out[:, -1, :] → (batch, 50)
    ↓
  Linear(50 → 1) → squeeze() → (batch,)
  ```
- **`StackedLSTM`**: Alternative baseline (3-layer LSTM: 128→64→32 with batch norm)
- **`SimpleLSTM`**: Lightweight variant (LSTM(50) → Dropout → Dense)
- **`MultiHeadAttention`**: Custom standalone implementation (not used in current pipeline; PyTorch built-in `nn.MultiheadAttention` is used instead)
- **`train_model()`**: Full training loop with Adam optimizer, MSE loss, early stopping, ReduceLROnPlateau scheduler, gradient clipping, and model checkpointing
- **`predict()`**: Inference function; returns numpy array of predictions

### `evaluation.py` - Metrics & Visualization
- **`calculate_metrics()`**: Computes MAE, RMSE, MAPE, R², Directional Accuracy
- **`print_metrics()`**: Formatted console output for metrics
- **`plot_predictions()`**: Time series overlay of actual vs predicted prices
- **`plot_training_history()`**: Dual-axis plot of train/val loss and MAE across epochs
- **`plot_error_distribution()`**: Histogram of prediction errors + scatter plot (actual vs predicted)
- **`evaluate_model()`**: Orchestrates metric calculation and plot generation

## Building and Running

### Installation
```bash
pip install -r requirements.txt
```

### Running the Pipeline
```bash
py main.py
```

This will:
- Download ~11 years of GLD daily data (~2,800 trading days)
- Engineer 31 technical features from OHLCV
- Run **5-fold Walk-Forward Cross Validation** (expanding window)
- Select best fold by R² score
- Convert predicted returns to prices and apply **Bias Correction**
- Generate evaluation plots in `results/`
- Save trained model weights in `models/`

### Expected Output Files
After running:
- `results/predictions.csv` - Date-indexed DataFrame with Actual, Predicted, Error, Error_%
- `results/predictions_vs_actual.png` - Time series visualization
- `results/error_distribution.png` - Error histogram + scatter plot
- `results/training_history.png` - Loss and MAE curves for best fold
- `models/cv_fold_1.pt` through `cv_fold_5.pt` - Model checkpoints for each fold

## Key Configuration Parameters

In `main.py`:
| Parameter | Default | Description |
|-----------|---------|-------------|
| `SEQ_LENGTH` | 30 | Lookback window in days |
| `n_splits` | 5 | Number of CV folds |
| `epochs` | 50 | Max training epochs per fold |
| `batch_size` | 32 | Batch size for DataLoader |
| `start` | '2015-01-01' | Data start date |
| `ticker` | 'GLD' | Gold ETF symbol |
| `train_ratio` | 0.95 | Train/test split for final evaluation |

In `lstm_model.py`:
| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | 0.003 | Adam optimizer initial LR |
| `patience_es` | 15 | Early stopping patience |
| `hidden_size` | 50 | LSTM hidden units |
| `num_heads` | 4 | Attention heads |
| `dropout` | 0.2 | Dropout rate |

## Development Conventions

### Code Style
- **Type hints**: Used throughout (e.g., `pd.DataFrame`, `np.ndarray`, `torch.Tensor`)
- **Docstrings**: NumPy-style with Args/Returns sections
- **Naming**: snake_case for functions/variables, CamelCase for classes
- **Structure**: Modular design with single-responsibility functions; each pipeline step is isolated

### Data Flow
```
Raw OHLCV → Preprocessing → Feature Engineering (31 features)
  → Returns Calculation → MinMaxScaler → Sequence Creation (30-day windows)
    → LSTM-Attention → Predicted Returns → Inverse Transform → Price Conversion
      → Bias Correction → Metrics & Visualization
```

Key design principles:
- Scalers are fit on train only and applied to test (prevents data leakage)
- Sequences include context from training data to ensure smooth transitions
- All transformations are reversible (scalers preserved for inverse_transform)

### Error Handling
- Warnings suppressed (`warnings.filterwarnings('ignore')`) for cleaner output
- Missing values handled via forward fill → backward fill
- NaN rows dropped after rolling window calculations in feature engineering
- Gradient clipping (`max_norm=1.0`) prevents exploding gradients in LSTM backprop

## Architecture Decisions

### Why LSTM-Attention over alternatives?
- **LSTM vs vanilla RNN**: LSTMs solve vanishing gradient problem via gating mechanisms (forget, input, output gates) and cell state, enabling learning of long-range dependencies in financial time series.
- **Attention vs LSTM-only**: Standard LSTM treats all time steps through sequential hidden state updates. Self-attention computes pairwise relationships across all 30 positions, allowing the model to dynamically weigh important days (e.g., price shocks, volume spikes) rather than relying solely on recency bias.
- **Multi-head (4 heads)**: Each head learns a different representation subspace, similar to diverse filters in CNNs. This enables parallel learning of short-term shocks, medium-term momentum, and long-term regime stability.
- **Residual connections**: Enable gradient flow through the attention layer and allow the model to fall back to LSTM representation if attention is uninformative for a given input.

### Why 30-day sequences?
- Represents ~1.5 months of trading data, capturing short-term momentum and mean-reversion patterns.
- Balances contextual information with computational efficiency (longer sequences increase memory quadratically for attention).
- Standard in financial time series literature for daily prediction horizons.

### Why Returns Prediction (not Price)?
- **Stationarity**: Price series are non-stationary (unit root, trending mean/variance). Returns are approximately stationary with mean ~0 and bounded variance, satisfying assumptions of most ML models.
- **Avoids covariate shift**: Gold moved from ~$1,000 (2015) to ~$2,300 (2024). A model trained on absolute prices learns price levels specific to the training period and fails to generalize.
- **Aligned with trading practice**: Quantitative strategies operate on % returns, not absolute levels.

### Why Walk-Forward Validation?
- **No look-ahead bias**: Each fold only uses data available up to that point in time, simulating real deployment.
- **Expanding window**: Leverages all available historical data for training while maintaining temporal integrity.
- **Robust performance estimate**: 5 folds provide distribution of metrics across different market regimes (bull, bear, sideways).

## Common Extension Points

1. **Add more features** (edit `feature_engineering.py`):
   - External macro data (DXY, Treasury yields, inflation expectations)
   - Alternative data (sentiment from news, search trends)
   - Additional technical indicators (Stochastic, ADX, ATR)

2. **Change model architecture** (edit `lstm_model.py`):
   - Bidirectional LSTM (captures patterns from both directions, though causality is a concern for time series)
   - GRU layers (fewer parameters, faster training)
   - Transformer encoder (replace LSTM entirely with positional encoding + multi-head attention)
   - Add more attention heads or increase hidden dimensions

3. **Different validation schemes** (edit `main.py`):
   - Adjust `n_splits` for more/fewer folds
   - Switch to sliding window (constant train size) instead of expanding window
   - Add purged cross-validation to handle overlapping sequences

4. **Compare models** (edit `evaluation.py` or add new modules):
   - Statistical baselines: ARIMA, GARCH
   - Tree-based: XGBoost, LightGBM with lag features
   - Temporal: Prophet, N-BEATS, Temporal Fusion Transformer

## Troubleshooting Notes

- **PyTorch Warnings**: "UserWarning: Named tensors..." is a known issue with certain PyTorch versions and can be safely ignored in this context.
- **Long training time**: Reduce `epochs` (e.g., 30), `n_splits` (e.g., 3), or `batch_size` for faster iteration during development.
- **Poor predictions**: Verify that Bias Correction is active; without it, systematic overprediction (~$10) degrades visual alignment. Also check that scalers are fit on train only.
- **Memory issues**: Reduce `batch_size` or `SEQ_LENGTH`. Attention complexity is O(n²) in sequence length.
- **Overfitting**: Increase dropout rate, reduce hidden units, or add L2 regularization (weight_decay in Adam optimizer).
- **Model not converging**: Check learning rate (try 0.001), verify data is properly scaled [0, 1], and ensure no NaN values in sequences.

## Research Context

This project implements an approach found in recent financial time series prediction literature (2020-2024):
- **LSTM-Attention combined architecture**: Superior to LSTM-only or Attention-only baselines in multiple studies.
- **Technical analysis feature engineering**: 31 features encoding market structure (trend, momentum, volatility, seasonality).
- **Walk-Forward Cross Validation**: Standard in quantitative finance for robust out-of-sample evaluation.
- **Returns-based prediction**: Stationary target avoids covariate shift and aligns with trading practice.
- **Post-processing bias correction**: Addresses systematic prediction bias observed in neural network outputs.

**Typical Performance**:
- R² > 0.92 (raw predictions)
- R² > 0.99 (after bias correction)
- MAE ~$0.34-$14.46 (varies by fold and market regime)

Suitable for:
- Undergraduate/Graduate ML projects
- Understanding PyTorch for Time Series
- Financial prediction demonstrations
- Quantitative finance research
- Portfolio strategy prototyping
