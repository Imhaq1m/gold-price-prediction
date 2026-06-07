# Quick Start Guide

## Train & Evaluate

```bash
py -m src.main
```

## Predict Next Trading Day

```bash
py -m src.predict_future
```

## Predict 30 Days

```bash
py -m src.predict_future --days 30
```

## Launch Web Dashboard

```bash
py app.py
# → http://127.0.0.1:5000
```

## Project Layout

```
project/
├── src/             # Source code (main.py, predict_future.py, etc.)
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
- **MAE**: ~$0.61–1.40
- **MAPE**: ~0.56%
- **Training Time**: 5–15 minutes on CPU
