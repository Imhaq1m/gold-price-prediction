"""
Hyperparameter tuning for LSTM-Attention model.
Grid search over optimizers, transfer functions (activations), and learning rates.
"""

import os
import warnings
from itertools import product
from typing import List, Tuple

import numpy as np
import pandas as pd

from src.evaluation import calculate_metrics
from src.feature_engineering import create_sequences, scale_data
from src.lstm_model import LSTMAttentionModel, train_model, predict

warnings.filterwarnings("ignore")

OPTIMIZERS = ["adam", "sgd", "rmsprop", "adamw"]
ACTIVATIONS = ["tanh", "relu", "leaky_relu"]
LEARNING_RATES = [0.001, 0.003, 0.01]

SEQ_LENGTH = 30
FORECAST_HORIZON = 30
EPOCHS = 30
BATCH_SIZE = 32
PATIENCE_ES = 10


def _build_eval_data(
    df: pd.DataFrame, feature_columns: List[str], val_ratio: float = 0.15
) -> Tuple:
    """
    Build train/val splits for hyperparameter evaluation.

    Uses a chronological split (last val_ratio of data for validation).

    Returns:
        X_train, y_train, X_val, y_val, train_feat_scaled, val_feat_scaled,
        val_target_scaled, target_scaler
    """
    split_idx = int(len(df) * (1 - val_ratio))
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]

    (
        train_feat_scaled,
        train_target_scaled,
        val_feat_scaled,
        val_target_scaled,
        _,
        target_scaler,
    ) = scale_data(train_df, val_df, feature_columns, target_column="returns")

    X_train, y_train = create_sequences(
        train_feat_scaled, train_target_scaled, SEQ_LENGTH, FORECAST_HORIZON
    )

    X_val_ctx = train_feat_scaled[-SEQ_LENGTH:]
    X_val_full = np.vstack([X_val_ctx, val_feat_scaled])
    y_val_full = np.concatenate([train_target_scaled[-SEQ_LENGTH:], val_target_scaled])
    X_val, y_val = create_sequences(
        X_val_full, y_val_full, SEQ_LENGTH, FORECAST_HORIZON
    )

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        train_feat_scaled,
        val_feat_scaled,
        val_target_scaled,
        target_scaler,
    )


def run_parameter_sweep(
    df: pd.DataFrame,
    feature_columns: List[str],
    optimizers: List[str] = None,
    activations: List[str] = None,
    learning_rates: List[float] = None,
    val_ratio: float = 0.15,
) -> pd.DataFrame:
    """
    Run grid search over hyperparameter combinations.

    Args:
        df: Full DataFrame with features
        feature_columns: List of feature column names
        optimizers: List of optimizers to test
        activations: List of activations to test
        learning_rates: List of learning rates to test
        val_ratio: Validation split ratio

    Returns:
        DataFrame with results for all combinations
    """
    if optimizers is None:
        optimizers = OPTIMIZERS
    if activations is None:
        activations = ACTIVATIONS
    if learning_rates is None:
        learning_rates = LEARNING_RATES

    print("\n" + "=" * 70)
    print("HYPERPARAMETER TUNING — Grid Search")
    print("=" * 70)
    print(f"Optimizers ({len(optimizers)}): {optimizers}")
    print(f"Activations ({len(activations)}): {activations}")
    print(f"Learning rates ({len(learning_rates)}): {learning_rates}")
    total = len(optimizers) * len(activations) * len(learning_rates)
    print(f"Total configurations: {total}")

    # Build evaluation data once
    (
        X_train,
        y_train,
        X_val,
        y_val,
        train_feat_scaled,
        val_feat_scaled,
        val_target_scaled,
        target_scaler,
    ) = _build_eval_data(df, feature_columns, val_ratio)

    input_size = X_train.shape[2]

    # Build actual prices for validation metrics
    split_idx = int(len(df) * (1 - val_ratio))
    val_df = df.iloc[split_idx:]
    val_returns = val_df["returns"].values
    val_close = np.concatenate(
        [
            df["close"].values[split_idx - 1 : split_idx],
            val_df["close"].values[:-1],
        ]
    )
    val_actual_prices = val_close * (1 + val_returns)

    # Align with sequences
    n_val_seq = len(X_val)
    val_actual_prices = val_actual_prices[SEQ_LENGTH - 1 : SEQ_LENGTH - 1 + n_val_seq]

    results = []
    best_r2 = -float("inf")
    best_config = None

    for i, (opt, act, lr) in enumerate(
        product(optimizers, activations, learning_rates)
    ):
        config_name = f"{opt.upper()} / {act} / lr={lr}"
        print(f"\n[{i + 1}/{total}] Config: {config_name}")

        try:
            model = LSTMAttentionModel(
                input_size=input_size,
                hidden_size=50,
                num_heads=4,
                dropout=0.2,
                output_size=FORECAST_HORIZON,
                activation=act,
            )

            _ = train_model(
                model=model,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                learning_rate=lr,
                patience_es=PATIENCE_ES,
                model_path=None,
                optimizer_type=opt,
            )

            pred_scaled = predict(model, X_val)
            pred_returns = target_scaler.inverse_transform(
                pred_scaled.reshape(-1, 1)
            ).reshape(-1, FORECAST_HORIZON)

            pred_prices = val_close[SEQ_LENGTH - 1 : SEQ_LENGTH - 1 + n_val_seq] * (
                1 + pred_returns[:, 0]
            )
            actual_prices = val_actual_prices

            metrics = calculate_metrics(actual_prices, pred_prices)
            r2 = metrics.get("R²", float("nan"))

            result = {
                "optimizer": opt,
                "activation": act,
                "learning_rate": lr,
                "R²": r2,
                "MAE": metrics.get("MAE", float("nan")),
                "RMSE": metrics.get("RMSE", float("nan")),
                "MAPE": metrics.get("MAPE", float("nan")),
            }
            results.append(result)

            print(f"  R²={r2:.4f}, MAE={result['MAE']:.4f}, RMSE={result['RMSE']:.4f}")

            if r2 > best_r2:
                best_r2 = r2
                best_config = result

        except Exception as e:
            print(f"  FAILED: {e}")
            results.append(
                {
                    "optimizer": opt,
                    "activation": act,
                    "learning_rate": lr,
                    "R²": float("nan"),
                    "MAE": float("nan"),
                    "RMSE": float("nan"),
                    "MAPE": float("nan"),
                }
            )

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("R²", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 70)
    print("PARAMETER SWEEP RESULTS (Top 10)")
    print("=" * 70)
    print(results_df.head(10).to_string(index=False))

    if best_config:
        print(
            f"\nBest config: {best_config['optimizer'].upper()} / {best_config['activation']} / lr={best_config['learning_rate']}"
        )
        print(f"Best R²: {best_config['R²']:.4f}")

    os.makedirs("results", exist_ok=True)
    results_df.to_csv("results/parameter_sweep.csv", index=False)
    print("\nFull results saved to results/parameter_sweep.csv")

    return results_df
