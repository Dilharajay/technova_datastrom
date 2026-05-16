# Evaluation metrics and diagnostics for TechNova outlet potential forecasting model.
# Streamlined for LightGBM regression on outlet maximum monthly liters.

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)


def evaluate_regression_arrays(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    name: str = "Volume"
) -> dict[str, float]:
    """Compute comprehensive regression evaluation metrics."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_clean = y_true[valid_mask]
    y_pred_clean = y_pred[valid_mask]
    
    if len(y_true_clean) == 0:
        raise ValueError("No valid samples after removing NaN values")
    
    mae = mean_absolute_error(y_true_clean, y_pred_clean)
    rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
    mape = mean_absolute_percentage_error(y_true_clean, y_pred_clean)
    r2 = r2_score(y_true_clean, y_pred_clean)
    
    y_range = np.max(y_true_clean) - np.min(y_true_clean)
    nrmse = rmse / y_range if y_range > 0 else 0.0
    
    naive_error = np.mean(np.abs(np.diff(y_true_clean)))
    mase = mae / naive_error if naive_error > 0 else 0.0
    
    bias = np.mean(y_pred_clean - y_true_clean)
    median_ae = np.median(np.abs(y_pred_clean - y_true_clean))
    
    return {
        "mean_absolute_error": float(mae),
        "root_mean_squared_error": float(rmse),
        "mean_absolute_percentage_error": float(mape),
        "normalized_rmse": float(nrmse),
        "r2_score": float(r2),
        "mean_absolute_scaled_error": float(mase),
        "bias": float(bias),
        "median_absolute_error": float(median_ae),
        "min_value": float(np.min(y_true_clean)),
        "max_value": float(np.max(y_true_clean)),
        "mean_value": float(np.mean(y_true_clean)),
        "n_samples": int(len(y_true_clean)),
    }


def print_in_sample_metrics(metrics: dict[str, Any]) -> None:
    """Print formatted in-sample evaluation metrics."""
    n_samples = metrics.get("n_samples", "?")
    
    print("\n" + "="*60)
    print("IN-SAMPLE EVALUATION METRICS")
    print("="*60)
    print(f"Samples: {n_samples}")
    print(f"Target Range: [{metrics.get('min_value', '?'):.2f}, {metrics.get('max_value', '?'):.2f}]")
    print(f"Target Mean: {metrics.get('mean_value', '?'):.2f}")
    print("-" * 60)
    print(f"MAE:             {metrics.get('mean_absolute_error', 0):.4f}")
    print(f"RMSE:            {metrics.get('root_mean_squared_error', 0):.4f}")
    print(f"MAPE:            {metrics.get('mean_absolute_percentage_error', 0):.4%}")
    print(f"Normalized RMSE: {metrics.get('normalized_rmse', 0):.4f}")
    print(f"R2 Score:        {metrics.get('r2_score', 0):.4f}")
    print(f"MASE:            {metrics.get('mean_absolute_scaled_error', 0):.4f}")
    print(f"Bias:            {metrics.get('bias', 0):.4f}")
    print(f"Median AE:       {metrics.get('median_absolute_error', 0):.4f}")
    print("="*60)


def build_evaluation_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    quantiles: list[float] = None,
) -> pd.DataFrame:
    """Build a detailed evaluation matrix with performance by quantile ranges."""
    if quantiles is None:
        quantiles = [0.25, 0.5, 0.75]
    
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_clean = y_true[valid_mask]
    y_pred_clean = y_pred[valid_mask]
    
    quantile_values = [0] + list(np.quantile(y_true_clean, quantiles)) + [np.inf]
    segments = []
    
    for i in range(len(quantile_values) - 1):
        lower = quantile_values[i]
        upper = quantile_values[i + 1]
        
        mask = (y_true_clean >= lower) & (y_true_clean < upper)
        if mask.sum() == 0:
            continue
        
        y_t = y_true_clean[mask]
        y_p = y_pred_clean[mask]
        
        segment_name = f"[{lower:.1f}, {upper:.1f})" if upper != np.inf else f"[{lower:.1f}, inf)"
        
        segments.append({
            "Segment": segment_name,
            "Count": int(mask.sum()),
            "Pct": f"{100 * mask.sum() / len(y_true_clean):.1f}%",
            "Actual_Mean": float(np.mean(y_t)),
            "Pred_Mean": float(np.mean(y_p)),
            "MAE": float(mean_absolute_error(y_t, y_p)),
            "RMSE": float(np.sqrt(mean_squared_error(y_t, y_p))),
            "MAPE": f"{mean_absolute_percentage_error(y_t, y_p):.2%}",
            "R2": float(r2_score(y_t, y_p)),
        })
    
    return pd.DataFrame(segments)
