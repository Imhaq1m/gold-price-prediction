"""
Evaluation and visualization module for gold price prediction.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict
import os


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate regression metrics.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        Dictionary of metrics
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)

    # Directional accuracy
    if len(y_true) > 1:
        true_direction = np.sign(np.diff(y_true))
        pred_direction = np.sign(np.diff(y_pred))
        directional_accuracy = np.mean(true_direction == pred_direction) * 100
    else:
        directional_accuracy = None

    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "R²": r2,
        "Directional Accuracy (%)": directional_accuracy,
    }

    return metrics


def print_metrics(metrics: Dict[str, float], title: str = "Model Performance"):
    """
    Print metrics in a formatted way.

    Args:
        metrics: Dictionary of metrics
        title: Title for the output
    """
    print(f"\n{'=' * 50}")
    print(f"{title}")
    print(f"{'=' * 50}")
    for metric_name, value in metrics.items():
        if value is not None:
            if "Accuracy" in metric_name:
                print(f"{metric_name:25s}: {value:.2f}%")
            elif "MAPE" in metric_name or "R²" in metric_name:
                print(f"{metric_name:25s}: {value:.4f}")
            else:
                print(f"{metric_name:25s}: {value:.6f}")
    print(f"{'=' * 50}\n")


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dates: pd.DatetimeIndex = None,
    title: str = "Gold Price Prediction vs Actual",
    save_path: str = None,
):
    """
    Plot predicted vs actual values.

    Args:
        y_true: True values
        y_pred: Predicted values
        dates: Date index for x-axis
        title: Plot title
        save_path: Path to save the plot
    """
    plt.figure(figsize=(14, 7))

    if dates is not None:
        # Align dates with predictions
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

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    plt.show(block=False)
    plt.pause(0.1)


def plot_training_history(history, save_path: str = None):
    """
    Plot training loss and metrics over epochs.

    Args:
        history: Keras History object or dict with 'loss', 'val_loss', 'mae', 'val_mae'
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Handle both Keras History and dict
    if hasattr(history, "history"):
        hist = history.history
    else:
        hist = history

    # Loss plot
    axes[0].plot(hist["loss"], label="Training Loss", linewidth=2)
    if "val_loss" in hist and hist["val_loss"]:
        axes[0].plot(hist["val_loss"], label="Validation Loss", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss (MSE)")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # MAE plot
    if "mae" in hist and hist["mae"]:
        axes[1].plot(hist["mae"], label="Training MAE", linewidth=2)
        if "val_mae" in hist and hist["val_mae"]:
            axes[1].plot(hist["val_mae"], label="Validation MAE", linewidth=2)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("MAE")
        axes[1].set_title("Training & Validation MAE")
        axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Training history plot saved to {save_path}")

    plt.show()


def plot_error_distribution(
    y_true: np.ndarray, y_pred: np.ndarray, save_path: str = None
):
    """
    Plot distribution of prediction errors.

    Args:
        y_true: True values
        y_pred: Predicted values
        save_path: Path to save the plot
    """
    errors = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    axes[0].hist(errors, bins=50, edgecolor="black", alpha=0.7)
    axes[0].set_xlabel("Prediction Error")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Error Distribution")
    axes[0].axvline(x=0, color="r", linestyle="--", linewidth=2)
    axes[0].grid(True, alpha=0.3)

    # Scatter plot: Actual vs Predicted
    axes[1].scatter(y_true, y_pred, alpha=0.5, edgecolors="none")
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    axes[1].plot(
        [min_val, max_val],
        [min_val, max_val],
        "r--",
        linewidth=2,
        label="Perfect Prediction",
    )
    axes[1].set_xlabel("Actual Price")
    axes[1].set_ylabel("Predicted Price")
    axes[1].set_title("Actual vs Predicted")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Error distribution plot saved to {save_path}")

    plt.show()


def compare_models(models_metrics: Dict[str, Dict[str, float]], save_path: str = None):
    """
    Compare multiple models using a bar chart.

    Args:
        models_metrics: Dictionary of {model_name: metrics_dict}
        save_path: Path to save the plot
    """
    metrics_to_compare = ["MAE", "RMSE", "MAPE", "R²"]
    model_names = list(models_metrics.keys())

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, metric in enumerate(metrics_to_compare):
        values = [models_metrics[model].get(metric, 0) for model in model_names]
        bars = axes[i].bar(model_names, values, alpha=0.7, edgecolor="black")
        axes[i].set_title(f"{metric} Comparison")
        axes[i].set_ylabel(metric)
        axes[i].grid(True, alpha=0.3, axis="y")

        # Add value labels on bars
        for bar, value in zip(bars, values):
            axes[i].text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Model comparison plot saved to {save_path}")

    plt.show()


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test_scaled: np.ndarray,
    target_scaler,
    test_dates: pd.DatetimeIndex = None,
    output_dir: str = "results",
) -> Dict[str, float]:
    """
    Complete model evaluation pipeline (PyTorch compatible).

    Args:
        model: Trained PyTorch model
        X_test: Test features
        y_test_scaled: Scaled test targets
        target_scaler: Scaler for inverse transform
        test_dates: Test set dates
        output_dir: Directory to save results

    Returns:
        Dictionary of metrics
    """
    os.makedirs(output_dir, exist_ok=True)

    # Make predictions (PyTorch)
    print("Making predictions on test set...")
    from lstm_model import predict as torch_predict

    y_pred_scaled = torch_predict(model, X_test)

    # Inverse transform to original scale
    y_test = target_scaler.inverse_transform(y_test_scaled.reshape(-1, 1)).ravel()
    y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred)
    print_metrics(metrics, "LSTM Model Performance on Test Set")

    # Plot predictions
    plot_predictions(
        y_test,
        y_pred,
        dates=test_dates,
        title="Gold Price: LSTM Prediction vs Actual",
        save_path=os.path.join(output_dir, "predictions_vs_actual.png"),
    )

    # Plot error distribution
    plot_error_distribution(
        y_test, y_pred, save_path=os.path.join(output_dir, "error_distribution.png")
    )

    # Save predictions to CSV
    predictions_df = pd.DataFrame(
        {
            "Actual": y_test,
            "Predicted": y_pred,
            "Error": y_test - y_pred,
            "Error_%": ((y_test - y_pred) / y_test) * 100,
        }
    )

    if test_dates is not None:
        predictions_df.index = test_dates[-len(predictions_df) :]

    predictions_df.to_csv(os.path.join(output_dir, "predictions.csv"))
    print(f"Predictions saved to {os.path.join(output_dir, 'predictions.csv')}")

    return metrics, y_test, y_pred


def print_metric_justifications():
    """
    Print justification for each evaluation metric used.

    Explains why each metric is appropriate for gold price prediction.
    """
    print("\n" + "=" * 70)
    print("METRIC JUSTIFICATIONS")
    print("=" * 70)
    print("""
  MAE (Mean Absolute Error):
    Measures average absolute error in dollars. Interpretable directly
    as "the model is off by $X per prediction." Less sensitive to outliers
    than RMSE, which is desirable because gold price spikes (e.g., market
    shocks) are rare events we don't want to over-penalize.

  RMSE (Root Mean Squared Error):
    Penalizes large errors more heavily than MAE via squaring. Useful
    for identifying models with extreme prediction failures. The gap
    between MAE and RMSE indicates error variance - a small gap means
    consistent errors, a large gap means occasional big misses.

  MAPE (Mean Absolute Percentage Error):
    Expresses error as a percentage of the actual price, making it
    scale-independent. A MAPE of 0.56% means predictions are within
    ~0.5% of the true price on average. Useful for comparing accuracy
    across different price regimes.

  R2 (R-squared / Coefficient of Determination):
    Proportion of variance in actual prices explained by the model.
    R2 = 0.96 means the model explains 96% of price variance, indicating
    excellent fit. Values near 1 are expected for price prediction since
    prices exhibit strong autocorrelation (today ~ yesterday).

  Directional Accuracy (%):
    Percentage of correct direction predictions (up/down). Critical for
    trading applications -- a model can have low MAE but wrong direction
    and still lose money. This metric validates that the model captures
    trend changes, not just price levels.
""")
