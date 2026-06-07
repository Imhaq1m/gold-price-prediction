# AGENTS.md — Gold Price Prediction (LSTM-Attention)

## Commands

### Install
```bash
pip install -r requirements.txt
```

### Train pipeline
```bash
py -m src.main
```

### Inference (CLI)
```bash
py -m src.predict_future                  # single day (ensemble)
py -m src.predict_future --days 30        # multi-day (direct multi-step)
py -m src.predict_future --model models/cv_fold_3.pt   # single model
```

### Web Dashboard
```bash
py app.py                                  # Flask dashboard → http://127.0.0.1:5000

```

### Lint (ruff)
```bash
py -m ruff check .
py -m ruff check src/ app.py
```

### Dataset export
```bash
py -m src.export_dataset
```

### Entry points (modules that can run directly)
- `src/main.py` — full pipeline (data → features → CV → retrain → evaluate → save)
- `src/predict_future.py` — CLI inference tool
- `src/export_dataset.py` — export CSV datasets for sharing
- `app.py` — Flask web dashboard

### Note on `py` vs `python`
Use `py` (not `python`) as the interpreter command. Run modules with `py -m <module>`. For modules inside `src/`, the working directory must be the project root so that `sys.path.insert(0, ...)` or relative imports resolve correctly.

---

## Project Structure

```
project/
├── src/                         # Python source package
│   ├── __init__.py
│   ├── main.py                  # Pipeline orchestrator
│   ├── predict_future.py        # CLI inference
│   ├── data_module.py           # yfinance fetch & preprocessing
│   ├── feature_engineering.py   # 31 technical indicators, sequences, scaling
│   ├── lstm_model.py            # PyTorch models (LSTMAttentionModel, StackedLSTM, SimpleLSTM)
│   ├── evaluation.py            # Metrics + matplotlib/seaborn plots
│   └── export_dataset.py        # CSV export for lecturer
├── templates/                   # Flask HTML template
│   └── index.html               # Chart.js dark-themed dashboard
├── app.py                       # Flask web dashboard
├── docs/                        # README, QUICKSTART, TESTING_REPORT
├── data/                        # Raw & processed CSVs
├── models/                      # .pt checkpoints, .pkl scalers, bias_correction.txt
├── results/                     # predictions.csv, .png plots
├── logs/                        # Run logs (gitignored)
├── requirements.txt
└── .gitignore
```

---

## Code Style Guidelines

### Imports
Group and order: standard library → third-party → local. Separate groups by a blank line.

```python
import os
import warnings

import joblib
import numpy as np
import pandas as pd
import torch
import yfinance as yf
from flask import Flask, jsonify, render_template
from sklearn.preprocessing import MinMaxScaler

from src.feature_engineering import add_technical_indicators, prepare_features
from src.lstm_model import LSTMAttentionModel, load_cv_models, ensemble_predict
```

- Use multi-line imports with parentheses when importing many names from one module.
- Use `from typing import Optional, Tuple, ...` for type annotations at the top of the file.
- Prefer importing modules (not individual functions) from PyTorch (`import torch.nn as nn`).

### Formatting
- Indent: 4 spaces (no tabs).
- Line length: ~100 chars max.
- String quotes: double quotes (`"..."`) consistently. Single quotes only inside f-strings.
- No semicolons. No trailing whitespace.
- Blank line after module docstring. Two blank lines before top-level definitions.
- Use f-strings for formatting (`f"{value:.4f}"`), not `%` or `.format()`.
- One blank line between method definitions inside classes.

### Type Annotations
- Always annotate function parameters and return types with the `typing` module.
- Use `Optional[X]` instead of `X | None` (Python 3.8+ compatibility).
- Use `Tuple`, `List`, `Dict` from `typing` (not `tuple[...]`, `list[...]`).
- Annotate `np.ndarray` for array parameters, `pd.DataFrame` / `pd.DatetimeIndex` for pandas types, `nn.Module` for PyTorch models.
- Class constructors need `-> None` return type on `__init__`.

```python
def create_sequences(
    data: np.ndarray,
    target: np.ndarray,
    seq_length: int = 60,
    forecast_horizon: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
```

### Docstrings
- Google-style (Args / Returns sections).
- Every module needs a module-level docstring.
- Every public function/class needs a docstring.
- `Args:` and `Returns:` sections are indented 4 spaces. Type after colon is optional in text.
- Use imperative mood ("Create", "Calculate", "Load"), not "Creates", "Calculates".

```python
def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate regression metrics.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        Dictionary of metrics
    """
```

### Naming Conventions
- `snake_case` for functions, methods, variables, file names
- `PascalCase` for classes (including nn.Module subclasses)
- `UPPER_CASE` for module-level constants (e.g., `SEQ_LENGTH = 30`, `_FEATURE_COLUMNS`)
- Prefix private functions with `_` (e.g., `_load_artifacts`, `_recompute_features`)

### Error Handling
- Use `warnings.filterwarnings("ignore")` at the top of entry-point modules to suppress expected warnings.
- For file loading that may fail (e.g., bias correction file), use `try/except FileNotFoundError: pass`.
- For Flask API routes, wrap the handler body in `try/except Exception` and return `jsonify({"error": str(e)}), 500`.
- Use `print()` for user-facing progress output (no logging module).
- Use `torch.no_grad()` context manager during inference.
- Use `os.makedirs(..., exist_ok=True)` for output directories.
- Do NOT use assertions for input validation.

### Model Config & Constants
- Define model hyperparameters as module-level constants (e.g., `SEQ_LENGTH = 30`, `FORECAST_HORIZON = 30`).
- Feature column lists should be defined once in `feature_engineering.py`; do NOT duplicate them in other files.
- Use `joblib.dump`/`load` for scalers. Use `torch.save`/`load` with `weights_only=False` for model checkpoints.

### PyTorch / ML Conventions
- All models subclass `nn.Module` and call `super().__init__()`.
- `forward()` accepts `torch.Tensor` and returns `torch.Tensor`.
- Use `batch_first=True` for all `nn.LSTM` and `nn.MultiheadAttention` layers.
- Use `nn.MSELoss()` for regression training.
- Training loop: `optimizer.zero_grad()` → `loss.backward()` → `torch.nn.utils.clip_grad_norm_()` → `optimizer.step()`.
- Use `ReduceLROnPlateau` scheduler with `mode="min"`.
- Use `model.eval()` during validation/testing, `model.train()` during training.
- Wrap inference in `with torch.no_grad():`.
- Move tensors to device with `.to(device)`.
- All scalers are `MinMaxScaler(feature_range=(0, 1))` from scikit-learn.
