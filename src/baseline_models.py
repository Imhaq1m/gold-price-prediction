"""
Baseline models for comparative analysis.
Implements Linear Regression, Random Forest, and SVM (SVR)
using the same features as the LSTM-Attention model.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from typing import Dict, Tuple

from src.evaluation import calculate_metrics, print_metrics


def _align_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Align features and targets for sklearn models.

    Each sample uses features[t] (last known indicators) to predict return[t+1].
    The first test prediction uses the last train row's features.

    Returns:
        X_train, y_train_returns, X_test, y_test_returns,
        prev_close_test, actual_test_prices
    """
    X_train = train_df[feature_columns].values[:-1]
    y_train_returns = train_df["returns"].values[1:]

    first_test_X = train_df[feature_columns].values[-1:]
    rest_test_X = test_df[feature_columns].values[:-1]
    X_test = np.vstack([first_test_X, rest_test_X])

    y_test_returns = test_df["returns"].values

    close_for_returns = np.concatenate(
        [
            train_df["close"].values[-1:],
            test_df["close"].values[:-1],
        ]
    )

    actual_test_prices = close_for_returns * (1 + y_test_returns)

    return (
        X_train,
        y_train_returns,
        X_test,
        y_test_returns,
        close_for_returns,
        actual_test_prices,
    )


def train_and_evaluate_baselines(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list,
) -> Dict[str, Dict]:
    """
    Train and evaluate 3 baseline models on the same train/test split.

    Args:
        train_df: Training DataFrame (from best CV fold)
        test_df: Test DataFrame (from best CV fold)
        feature_columns: List of feature column names

    Returns:
        Dictionary of {model_name: {"predictions": np.array, "metrics": dict}}
    """
    X_train, y_train, X_test, y_test, close_base, actual_prices = _align_data(
        train_df, test_df, feature_columns
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
        "SVM (SVR)": SVR(kernel="rbf"),
    }

    results = {}
    for name, model in models.items():
        print(f"\n  Training {name}...")
        model.fit(X_train, y_train)
        y_pred_returns = model.predict(X_test)
        pred_prices = close_base * (1 + y_pred_returns)
        metrics = calculate_metrics(actual_prices, pred_prices)
        print_metrics(metrics, name)
        results[name] = {
            "predictions": pred_prices,
            "actual": actual_prices,
            "metrics": metrics,
        }

    return results


def print_comparison_table(
    lstm_metrics: Dict[str, float],
    baseline_results: Dict[str, Dict],
    naive_metrics: Dict[str, float],
):
    """
    Print a formatted comparison table of all models.

    Args:
        lstm_metrics: Metrics dict for LSTM-Attention
        baseline_results: Output from train_and_evaluate_baselines
        naive_metrics: Metrics dict for Naive baseline
    """
    rows = [("LSTM-Attention", lstm_metrics)]
    for name, res in baseline_results.items():
        rows.append((name, res["metrics"]))
    rows.append(("Naive (0% return)", naive_metrics))

    print("\n" + "=" * 90)
    print("MODEL COMPARISON TABLE")
    print("=" * 90)
    header = (
        f"{'Model':<22} {'MAE':>10} {'RMSE':>10} {'MAPE':>8} {'R²':>8} {'Dir.Acc':>8}"
    )
    print(header)
    print("-" * 90)
    for name, metrics in rows:
        mae = metrics.get("MAE", 0)
        rmse = metrics.get("RMSE", 0)
        mape = metrics.get("MAPE", 0)
        r2 = metrics.get("R²", 0)
        da = metrics.get("Directional Accuracy (%)", 0)
        da_str = f"{da:.2f}%" if da is not None else "N/A"
        mape_str = f"{mape:.2f}%" if mape is not None else "N/A"
        print(
            f"{name:<22} {mae:>10.4f} {rmse:>10.4f} {mape_str:>8} {r2:>8.4f} {da_str:>8}"
        )
    print("=" * 90)
