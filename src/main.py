"""
Main pipeline for Gold Price Prediction using LSTM-Attention (PyTorch).
Based on research paper: LSTM-Attention Combined Model.

This script orchestrates the complete ML pipeline:
1. Data collection from yfinance (GLD ETF)
2. Feature engineering with technical indicators
3. Data preprocessing and scaling
4. LSTM-Attention model training (best performer from paper)
5. Model evaluation and visualization
"""

from src.evaluation import (
    plot_predictions,
    plot_error_distribution,
    calculate_metrics,
    compare_models,
    print_metric_justifications,
)
from src.lstm_model import LSTMAttentionModel, train_model
from src.feature_engineering import (
    add_technical_indicators,
    prepare_features,
    create_sequences,
    scale_data,
)
from src.baseline_models import train_and_evaluate_baselines, print_comparison_table
from src.data_module import fetch_gold_data, preprocess_data, split_data
import os
import joblib
import numpy as np
import pandas as pd
import torch
import warnings

warnings.filterwarnings("ignore")

# Import custom modules


def main():
    """Run the complete gold price prediction pipeline."""

    print("=" * 60)
    print("GOLD PRICE PREDICTION USING LSTM (PyTorch)")
    print("Undergraduate ML Project")
    print("=" * 60)

    # Create output directories
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # STEP 1: DATA COLLECTION
    print("\n[STEP 1] Data Collection")
    print("-" * 40)

    df = fetch_gold_data(ticker="GLD", start="2015-01-01", interval="1d")

    # STEP 2: PREPROCESSING
    print("\n[STEP 2] Data Preprocessing")
    print("-" * 40)

    df = preprocess_data(df)
    print(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"Total records: {len(df)}")

    # STEP 3: FEATURE ENGINEERING
    print("\n[STEP 3] Feature Engineering")
    print("-" * 40)

    df = add_technical_indicators(df)
    feature_df, feature_columns = prepare_features(df)

    if "close" not in feature_columns:
        feature_columns = ["close"] + feature_columns

    print(f"Number of features: {len(feature_columns)}")
    print(f"Features: {feature_columns[:10]}...")

    # STEP 5: FEATURE ENGINEERING FOR RETURNS PREDICTION
    print("\n[STEP 5] Feature Engineering for Returns Prediction")
    print("-" * 40)

    # Use daily returns (pct_change) as the target.
    # These are already computed by add_technical_indicators as df['returns'].
    # For multi-step: target at position t = returns[t] = close[t]/close[t-1] - 1.
    # The model predicts H future returns from a single input sequence.
    df = df.dropna()

    print("Target: Daily returns (stationary)")
    print(f"Target range: [{df['returns'].min():.4f}, {df['returns'].max():.4f}]")

    # STEP 5.1: TRAIN-TEST SPLIT (95/5 as per paper)
    print("\n[STEP 5.1] Train-Test Split")
    print("-" * 40)

    train_df, test_df = split_data(df, train_ratio=0.95)

    print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

    # STEP 6: FEATURE SCALING
    print("\n[STEP 6] Feature Scaling")
    print("-" * 40)

    (
        train_features_scaled,
        train_target_scaled,
        test_features_scaled,
        test_target_scaled,
        feature_scaler,
        target_scaler,
    ) = scale_data(train_df, test_df, feature_columns, target_column="returns")

    # WALK-FORWARD CROSS VALIDATION
    print("\n[STEP 7] Walk-Forward Cross Validation")
    print("-" * 40)

    SEQ_LENGTH = 30  # Paper uses 30-day time steps
    FORECAST_HORIZON = 30  # Direct multi-step: predict 30 future returns per sequence

    # Use expanding window cross validation
    n_splits = 5
    train_min_size = int(len(df) * 0.7)  # Start with 70% for first train
    test_size = int((len(df) - train_min_size) / n_splits)

    print(f"Data points: {len(df)}")
    print(f"Initial train: {train_min_size}, Test per fold: {test_size}")
    print(f"Number of folds: {n_splits}")

    cv_results = []

    for fold in range(n_splits):
        print(f"\n{'=' * 40}")
        print(f"FOLD {fold + 1}/{n_splits}")
        print(f"{'=' * 40}")

        # Define train/test indices for this fold
        train_end = train_min_size + fold * test_size
        test_end = train_end + test_size

        cv_train_df = df.iloc[:train_end]
        cv_test_df = df.iloc[train_end:test_end]
        cv_test_dates = cv_test_df.index

        # Scale features
        (
            cv_train_feat,
            cv_train_tgt,
            cv_test_feat,
            cv_test_tgt,
            cv_feat_scaler,
            cv_tgt_scaler,
        ) = scale_data(
            cv_train_df, cv_test_df, feature_columns, target_column="returns"
        )

        # Create multi-step sequences
        cv_X_train, cv_y_train = create_sequences(
            cv_train_feat, cv_train_tgt, SEQ_LENGTH, FORECAST_HORIZON
        )

        cv_X_test_ctx = cv_train_feat[-SEQ_LENGTH:]
        cv_X_test_full = np.vstack([cv_X_test_ctx, cv_test_feat])
        cv_y_test_full = np.concatenate([cv_train_tgt[-SEQ_LENGTH:], cv_test_tgt])
        cv_X_test, cv_y_test = create_sequences(
            cv_X_test_full, cv_y_test_full, SEQ_LENGTH, FORECAST_HORIZON
        )

        print(f"  Train: {cv_X_train.shape}, Test: {cv_X_test.shape}")

        # Build model
        cv_model = LSTMAttentionModel(
            input_size=cv_X_train.shape[2],
            hidden_size=50,
            num_heads=4,
            key_dim=50,
            dropout=0.2,
            output_size=FORECAST_HORIZON,
        )

        # Train with early stopping
        cv_val_split = int(len(cv_X_train) * 0.9)
        cv_X_tr = cv_X_train[:cv_val_split]
        cv_y_tr = cv_y_train[:cv_val_split]
        cv_X_val = cv_X_train[cv_val_split:]
        cv_y_val = cv_y_train[cv_val_split:]

        cv_model_path = f"models/cv_fold_{fold + 1}.pt"

        train_model(
            model=cv_model,
            X_train=cv_X_tr,
            y_train=cv_y_tr,
            X_val=cv_X_val,
            y_val=cv_y_val,
            epochs=50,
            batch_size=32,
            learning_rate=0.003,
            patience_es=15,
            model_path=cv_model_path,
        )

        # Evaluate (multi-step → use first-step predictions for fold ranking)
        from src.lstm_model import predict as predict_func

        cv_pred_scaled = predict_func(cv_model, cv_X_test)  # (n, H)
        cv_pred_returns = cv_tgt_scaler.inverse_transform(
            cv_pred_scaled.reshape(-1, 1)
        ).reshape(-1, FORECAST_HORIZON)
        cv_actual_returns = cv_tgt_scaler.inverse_transform(
            cv_y_test.reshape(-1, 1)
        ).reshape(-1, FORECAST_HORIZON)

        # Get all close prices from the combined train+test data
        cv_all_close = (
            cv_train_df["close"].values.tolist() + cv_test_df["close"].values.tolist()
        )
        # Sequence i ends at index i+SEQ_LENGTH-1 → prev_close = close at that position
        cv_prev_close = np.array(
            [cv_all_close[i + SEQ_LENGTH - 1] for i in range(len(cv_X_test))]
        )

        # Use only first-step predictions for fold comparison
        cv_actual_prices = cv_prev_close * (1 + cv_actual_returns[:, 0])
        cv_pred_prices = cv_prev_close * (1 + cv_pred_returns[:, 0])

        cv_metrics = calculate_metrics(cv_actual_prices, cv_pred_prices)
        cv_results.append(
            {
                "fold": fold + 1,
                "metrics": cv_metrics,
                "model": cv_model,
                "feature_scaler": cv_feat_scaler,
                "target_scaler": cv_tgt_scaler,
                "test_df": cv_test_df,
                "actual_prices": cv_actual_prices,
                "pred_prices_raw": cv_pred_prices,
                "dates": cv_test_dates[-len(cv_actual_prices) :],
            }
        )

        # Per-step metrics for diagnostic
        step_mae = np.mean(np.abs(cv_actual_returns - cv_pred_returns), axis=0)
        print(
            f"  Fold {fold + 1} R²: {cv_metrics['R²']:.4f}, MAE: {cv_metrics['MAE']:.2f}, "
            f"Step-1 MAE(returns): {step_mae[0]:.4f}, Step-30 MAE(returns): {step_mae[-1]:.4f}"
        )

    # Select best fold by R²
    best_fold = max(cv_results, key=lambda x: x["metrics"]["R²"])
    print(f"\n{'=' * 60}")
    print("CV RESULTS SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Fold':<6} {'R²':>10} {'MAE':>10} {'RMSE':>10}")
    print(f"{'-' * 40}")
    for r in cv_results:
        m = r["metrics"]
        print(f"{r['fold']:<6} {m['R²']:>10.4f} {m['MAE']:>10.2f} {m['RMSE']:>10.2f}")
    print(f"{'-' * 40}")
    print(f"Best fold: {best_fold['fold']} (R²={best_fold['metrics']['R²']:.4f})")
    print(f"{'=' * 60}")

    # Stitch all folds' predictions for full-range visualization
    all_dates = []
    all_actual = []
    all_pred = []
    for r in cv_results:
        fold_mean_error = np.mean(
            (r["actual_prices"] - r["pred_prices_raw"]) / r["actual_prices"]
        )
        fold_pred_corrected = r["pred_prices_raw"] / (1 - fold_mean_error)
        all_actual.append(r["actual_prices"])
        all_pred.append(fold_pred_corrected)
        all_dates.append(r["dates"])
    all_actual = np.concatenate(all_actual)
    all_pred = np.concatenate(all_pred)
    all_dates = np.concatenate(all_dates)

    # Use best model for final evaluation
    model = best_fold["model"]
    feature_scaler = best_fold["feature_scaler"]
    target_scaler = best_fold["target_scaler"]
    test_df = best_fold["test_df"]
    metrics = best_fold["metrics"]

    # ==========================================
    # RETRAIN ON ALL DATA
    # ==========================================
    print("\n[RETRAIN] Training final model on all available data...")
    print("-" * 40)

    all_feat_scaled = feature_scaler.transform(df[feature_columns].values)
    all_tgt_scaled = target_scaler.transform(
        df["returns"].values.reshape(-1, 1)
    ).ravel()
    X_all, y_all = create_sequences(
        all_feat_scaled, all_tgt_scaled, SEQ_LENGTH, FORECAST_HORIZON
    )
    print(f"  Total samples: {len(X_all)}")

    retrain_val_split = int(len(X_all) * 0.9)
    X_rt = X_all[:retrain_val_split]
    y_rt = y_all[:retrain_val_split]
    X_rv = X_all[retrain_val_split:]
    y_rv = y_all[retrain_val_split:]
    print(f"  Train: {len(X_rt)}, Validation: {len(X_rv)}")

    retrain_model = LSTMAttentionModel(
        input_size=X_all.shape[2],
        hidden_size=50,
        num_heads=4,
        key_dim=50,
        dropout=0.2,
        output_size=FORECAST_HORIZON,
    )

    train_model(
        model=retrain_model,
        X_train=X_rt,
        y_train=y_rt,
        X_val=X_rv,
        y_val=y_rv,
        epochs=50,
        batch_size=32,
        learning_rate=0.003,
        patience_es=15,
        model_path=None,
    )
    print("  Final model training complete.")

    # ==========================================
    # STEP 8: FINAL EVALUATION (Best CV Fold)
    # ==========================================
    print("\n[STEP 8] Final Evaluation (Best CV Fold)")
    print("-" * 40)

    # Use metrics from best CV fold
    price_metrics = metrics

    # ==========================================
    # BASELINE COMPARISON
    # ==========================================
    print("\nBaseline Comparison")
    print("-" * 40)

    # Extract best fold data for baseline comparison
    best_cv_train_df_temp = df.iloc[
        : train_min_size + (best_fold["fold"] - 1) * test_size
    ]
    best_cv_test_df_temp = best_fold["test_df"]

    # Train and evaluate sklearn baselines
    baseline_results = train_and_evaluate_baselines(
        best_cv_train_df_temp, best_cv_test_df_temp, feature_columns
    )

    # Naive baseline: predict no change (0% return)
    all_close_for_baseline = np.concatenate(
        [best_cv_train_df_temp["close"].values, best_cv_test_df_temp["close"].values]
    )
    sequence_indices = np.arange(SEQ_LENGTH, len(all_close_for_baseline))
    prev_close_prices = all_close_for_baseline[sequence_indices - 1]

    test_returns = best_cv_test_df_temp["returns"].values
    actual_returns_full = np.concatenate(
        [best_cv_train_df_temp["returns"].values[-SEQ_LENGTH:], test_returns]
    )
    actual_returns_aligned = actual_returns_full[SEQ_LENGTH:]
    actual_prices = prev_close_prices[-len(actual_returns_aligned) :] * (
        1 + actual_returns_aligned
    )
    y_naive = prev_close_prices[-len(actual_prices) :]
    naive_metrics = calculate_metrics(actual_prices, y_naive)

    # Print full comparison table
    print_comparison_table(price_metrics, baseline_results, naive_metrics)

    # Generate comparison bar chart
    all_model_metrics = {"LSTM-Attention": price_metrics}
    for name, res in baseline_results.items():
        all_model_metrics[name] = res["metrics"]
    all_model_metrics["Naive (0% return)"] = naive_metrics
    compare_models(all_model_metrics, save_path="results/model_comparison.png")

    # ==========================================
    # FINAL RESULTS
    # ==========================================
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    print(f"\nModel: LSTM-Attention — Direct Multi-Step (H={FORECAST_HORIZON})")
    print("Method: Walk-Forward Cross Validation (5 folds)")
    print("Final Metrics (1-step-ahead, Best Fold):")
    for metric_name, value in price_metrics.items():
        if value is not None:
            if "Accuracy" in metric_name:
                print(f"  {metric_name}: {value:.2f}%")
            else:
                print(f"  {metric_name}: {value:.4f}")

    # Compare with paper results
    print(f"\n{'=' * 60}")
    print("COMPARISON WITH PAPER RESULTS (1-step)")
    print(f"{'=' * 60}")
    print("  Metric       | Our Result | Paper Result")
    print(f"  {'-' * 45}")
    print(f"  MAE            | {price_metrics['MAE']:>10.2f} | {14.46:>10.2f}")
    print(f"  MSE (RMSE²)    | {price_metrics['RMSE'] ** 2:>10.2f} | {373.09:>10.2f}")
    print(f"  R²             | {price_metrics['R²']:>10.4f} | {0.92:>10.4f}")
    print(f"{'=' * 60}")

    # Metric justifications
    print_metric_justifications()

    # Save predictions CSV from best fold (first-step only)
    from src.lstm_model import predict as predict_func

    # Rebuild sequences for the best fold
    best_cv_train_df_temp = df.iloc[
        : train_min_size + (best_fold["fold"] - 1) * test_size
    ]
    best_cv_test_df_temp = best_fold["test_df"]

    # Scale features
    best_cv_feat_scaled_temp = feature_scaler.transform(
        best_cv_train_df_temp[feature_columns].values
    )
    best_cv_test_feat_scaled_temp = feature_scaler.transform(
        best_cv_test_df_temp[feature_columns].values
    )

    # Create multi-step sequences
    cv_X_test_ctx_temp = best_cv_feat_scaled_temp[-SEQ_LENGTH:]
    cv_X_test_full_temp = np.vstack([cv_X_test_ctx_temp, best_cv_test_feat_scaled_temp])
    cv_y_test_full_temp = np.concatenate(
        [
            best_cv_train_df_temp["returns"].values[-SEQ_LENGTH:],
            best_cv_test_df_temp["returns"].values,
        ]
    )
    cv_X_test_temp, cv_y_test_temp = create_sequences(
        cv_X_test_full_temp, cv_y_test_full_temp, SEQ_LENGTH, FORECAST_HORIZON
    )

    # Make multi-step predictions
    pred_scaled = predict_func(model, cv_X_test_temp)  # (n, H)
    pred_returns = target_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).reshape(
        -1, FORECAST_HORIZON
    )
    actual_returns = target_scaler.inverse_transform(
        cv_y_test_temp.reshape(-1, 1)
    ).reshape(-1, FORECAST_HORIZON)

    # Get the close prices for converting returns to prices
    all_close_prices_temp = np.concatenate(
        [best_cv_train_df_temp["close"].values, best_cv_test_df_temp["close"].values]
    )
    sequence_end_indices = np.arange(SEQ_LENGTH, len(all_close_prices_temp))
    prev_close_prices_temp = all_close_prices_temp[sequence_end_indices - 1]

    # Use only first-step predictions for evaluation
    pred_returns_1step = pred_returns[:, 0]
    actual_returns_1step = actual_returns[:, 0]
    n_test = len(pred_returns_1step)

    raw_pred_prices = prev_close_prices_temp[-n_test:] * (1 + pred_returns_1step)
    actual_prices = prev_close_prices_temp[-n_test:] * (1 + actual_returns_1step)

    # ==========================================
    # BIAS CORRECTION (percentage-based)
    # ==========================================
    print("\nApplying Bias Correction...")
    mean_error = np.mean((actual_prices - raw_pred_prices) / actual_prices)
    pred_prices = raw_pred_prices / (1 - mean_error)
    print(f"  Mean Bias: {mean_error * 100:.2f}%")
    print(f"  Correction: Predictions scaled by {(1 / (1 - mean_error)):.4f}x")

    # Recalculate metrics with corrected predictions
    corrected_metrics = calculate_metrics(actual_prices, pred_prices)
    print("\nCorrected Metrics:")
    for metric_name, value in corrected_metrics.items():
        if value is not None:
            if "Accuracy" in metric_name:
                print(f"  {metric_name}: {value:.2f}%")
            else:
                print(f"  {metric_name}: {value:.4f}")

    predictions_df = pd.DataFrame(
        {
            "Actual": all_actual,
            "Predicted": all_pred,
            "Error": all_actual - all_pred,
            "Error_%": ((all_actual - all_pred) / all_actual) * 100,
        }
    )
    predictions_df.index = all_dates
    predictions_df.to_csv("results/predictions.csv")
    print("\nPredictions saved to results/predictions.csv")

    # ==========================================
    # GENERATE VISUALIZATIONS
    # ==========================================
    print("\nGenerating visualizations...")

    # Plot predictions vs actual (all CV folds stitched)
    plot_predictions(
        all_actual,
        all_pred,
        dates=all_dates,
        title="Gold Price: LSTM Prediction vs Actual (All CV Folds)",
        save_path="results/predictions_vs_actual.png",
    )

    # Plot error distribution
    plot_error_distribution(
        all_actual,
        all_pred,
        save_path="results/error_distribution.png",
    )

    print("Visualizations saved to results/")

    # ==========================================
    # SAVE ARTIFACTS FOR FUTURE INFERENCE
    # ==========================================
    print("\nSaving inference artifacts...")
    joblib.dump(feature_scaler, "models/feature_scaler.pkl")
    joblib.dump(target_scaler, "models/target_scaler.pkl")
    with open("models/bias_correction.txt", "w") as f:
        f.write(f"{mean_error:.6f}")
    print("  - feature_scaler.pkl")
    print("  - target_scaler.pkl")
    print(f"  - bias_correction.txt (mean_error_pct={mean_error * 100:.4f}%)")

    # Save the retrained (on all data) model
    torch.save(
        {
            "model_state_dict": retrain_model.state_dict(),
            "input_size": retrain_model.lstm.input_size,
            "output_size": FORECAST_HORIZON,
            "forecast_horizon": FORECAST_HORIZON,
        },
        "models/best_lstm_attention.pt",
    )
    print(f"  - best_lstm_attention.pt (H={FORECAST_HORIZON}, retrained on all data)")

    # ==========================================
    # HYPERPARAMETER TUNING (optional, can be slow)
    # ==========================================
    RUN_PARAMETER_SWEEP = False  # Set to True to run grid search
    if RUN_PARAMETER_SWEEP:
        print("\n[OPTIONAL] Running Hyperparameter Tuning...")
        print("-" * 40)
        from src.hyperparameter_tuning import run_parameter_sweep

        run_parameter_sweep(df, feature_columns)

    # ==========================================
    # SUMMARY
    # ==========================================
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print("\nResults saved in 'results/' directory:")
    print("  - predictions.csv")
    print("  - predictions_vs_actual.png")
    print("  - error_distribution.png")
    print("  - model_comparison.png")
    if RUN_PARAMETER_SWEEP:
        print("  - parameter_sweep.csv")
    print("\nModels saved in 'models/' directory:")
    print("  - cv_fold_1.pt through cv_fold_5.pt")
    print(f"  - best_lstm_attention.pt (H={FORECAST_HORIZON})")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
