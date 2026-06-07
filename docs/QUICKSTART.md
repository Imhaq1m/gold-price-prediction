# Quick Start Guide

## Train & Evaluate

```bash
py -m src.main
```

## Predict Next Trading Day (CLI)

```bash
py -m src.predict_future
```

## Predict 30 Days (CLI)

```bash
py -m src.predict_future --days 30
```

## Launch Web Dashboard

```bash
py app.py
# → http://127.0.0.1:5000
```

The dashboard shows:
- Historical GLD price chart (last 3 months)
- Auto-run 7-day LSTM prediction with realistic day-to-day variation
- Enter any number 1–30 and click Predict to update
- Results table with per-day return % and price

## Export Dataset

```bash
py -m src.export_dataset
```

Creates 3 CSV files: raw OHLCV, cleaned, and full 31-feature dataset.

## Project Layout

```
project/
├── src/             # Source code (main.py, predict_future.py, feature_engineering.py, etc.)
├── templates/       # Flask HTML template (index.html with Chart.js)
├── app.py           # Flask web dashboard
├── models/          # Trained checkpoints & scalers
├── results/         # Predictions & plots
├── data/            # Raw CSV datasets
└── docs/            # Documentation
```

## Dependencies

```bash
pip install -r requirements.txt
```

## Expected Performance

- **R² Score**: ~0.92–0.96
- **MAE**: ~$0.61
- **MAPE**: ~0.56%
- **Training Time**: 5–15 minutes on CPU

## Quick Links

| Task | Command |
|---|---|
| Train model | `py -m src.main` |
| Predict (CLI) | `py -m src.predict_future` |
| Web dashboard | `py app.py` |
| Export CSV | `py -m src.export_dataset` |
| Lint code | `py -m ruff check src/ app.py` |
