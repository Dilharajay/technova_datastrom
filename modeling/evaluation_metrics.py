
# Evaluation metrics and diagnostics for TechNova forecasting models.
# Includes regression metrics, constraint classification analysis, and reporting utilities.


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
    """
    Compute comprehensive regression evaluation metrics.
    
    Parameters
    ----------
    y_true : np.ndarray
        Actual values
    y_pred : np.ndarray
        Predicted values
    name : str
        Name of the target variable for reporting
        
    Returns
    -------
    dict[str, float]
        Dictionary containing MAE, MAPE, RMSE, NRMSE, R², and other metrics
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    # Handle NaN values
    valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_clean = y_true[valid_mask]
    y_pred_clean = y_pred[valid_mask]
    
    if len(y_true_clean) == 0:
        raise ValueError("No valid samples after removing NaN values")
    
    mae = mean_absolute_error(y_true_clean, y_pred_clean)
    rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
    mape = mean_absolute_percentage_error(y_true_clean, y_pred_clean)
    r2 = r2_score(y_true_clean, y_pred_clean)
    
    # Normalized RMSE
    y_range = np.max(y_true_clean) - np.min(y_true_clean)
    nrmse = rmse / y_range if y_range > 0 else 0.0
    
    # Mean Absolute Scaled Error (MASE)
    naive_error = np.mean(np.abs(np.diff(y_true_clean)))
    mase = mae / naive_error if naive_error > 0 else 0.0
    
    # Mean Bias
    bias = np.mean(y_pred_clean - y_true_clean)
    
    # Median Absolute Error
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


def evaluate_constraint_model(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
) -> dict[str, Any]:
    """
    Evaluate constraint risk prediction model (regression on constraint share).
    
    Parameters
    ----------
    y_true : np.ndarray
        True constraint share values (0 to 1)
    y_pred_proba : np.ndarray
        Predicted constraint risk values
        
    Returns
    -------
    dict[str, Any]
        Evaluation metrics including calibration analysis
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred_proba = np.asarray(y_pred_proba, dtype=float)
    
    # Clip predictions to [0, 1]
    y_pred_clipped = np.clip(y_pred_proba, 0.0, 1.0)
    
    metrics = evaluate_regression_arrays(y_true, y_pred_clipped, name="Constraint_Share")
    
    # Calibration analysis: compare predicted vs actual by probability bins
    n_bins = 10
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_pred_clipped, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    calibration = {}
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_pred_mean = np.mean(y_pred_clipped[mask])
            bin_true_mean = np.mean(y_true[mask])
            calibration[f"bin_{i}"] = {
                "pred_mean": float(bin_pred_mean),
                "true_mean": float(bin_true_mean),
                "count": int(mask.sum()),
                "calibration_error": float(abs(bin_pred_mean - bin_true_mean)),
            }
    
    metrics["calibration_analysis"] = calibration
    
    return metrics


def print_in_sample_metrics(metrics: dict[str, Any]) -> None:
    """
    Print formatted in-sample evaluation metrics.
    
    Parameters
    ----------
    metrics : dict[str, Any]
        Dictionary of evaluation metrics from evaluate_regression_arrays or similar
    """
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
    print(f"R² Score:        {metrics.get('r2_score', 0):.4f}")
    print(f"MASE:            {metrics.get('mean_absolute_scaled_error', 0):.4f}")
    print(f"Bias:            {metrics.get('bias', 0):.4f}")
    print(f"Median AE:       {metrics.get('median_absolute_error', 0):.4f}")
    print("="*60)


def build_evaluation_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    quantiles: list[float] = None,
) -> pd.DataFrame:
    """
    Build a detailed evaluation matrix with performance by quantile ranges.
    
    Parameters
    ----------
    y_true : np.ndarray
        Actual values
    y_pred : np.ndarray
        Predicted values
    quantiles : list[float], optional
        Quantile boundaries for segmentation (default: [0.25, 0.5, 0.75])
        
    Returns
    -------
    pd.DataFrame
        Evaluation matrix with metrics by segment
    """
    if quantiles is None:
        quantiles = [0.25, 0.5, 0.75]
    
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_clean = y_true[valid_mask]
    y_pred_clean = y_pred[valid_mask]
    
    # Create segments based on actual values
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
        
        segment_name = f"[{lower:.1f}, {upper:.1f})" if upper != np.inf else f"[{lower:.1f}, ∞)"
        
        segments.append({
            "Segment": segment_name,
            "Count": int(mask.sum()),
            "Pct": f"{100 * mask.sum() / len(y_true_clean):.1f}%",
            "Actual_Mean": float(np.mean(y_t)),
            "Pred_Mean": float(np.mean(y_p)),
            "MAE": float(mean_absolute_error(y_t, y_p)),
            "RMSE": float(np.sqrt(mean_squared_error(y_t, y_p))),
            "MAPE": f"{mean_absolute_percentage_error(y_t, y_p):.2%}",
            "R²": float(r2_score(y_t, y_p)),
        })
    
    return pd.DataFrame(segments)


def build_constraint_matrix(
    constraint_share: np.ndarray,
    predicted_risk: np.ndarray,
) -> pd.DataFrame:
    """
    Build evaluation matrix for constraint risk model by constraint level.
    
    Parameters
    ----------
    constraint_share : np.ndarray
        True constraint share (0-1)
    predicted_risk : np.ndarray
        Predicted constraint risk (0-1)
        
    Returns
    -------
    pd.DataFrame
        Evaluation matrix by constraint level
    """
    constraint_share = np.asarray(constraint_share, dtype=float)
    predicted_risk = np.clip(np.asarray(predicted_risk, dtype=float), 0, 1)
    
    valid_mask = ~(np.isnan(constraint_share) | np.isnan(predicted_risk))
    constraint_share_clean = constraint_share[valid_mask]
    predicted_risk_clean = predicted_risk[valid_mask]
    
    segments = []
    for category, (lower, upper, label) in [
        ("unconstrained", (0, 0.2, "Unconstrained (0%-20%)")),
        ("moderate", (0.2, 0.5, "Moderate (20%-50%)")),
        ("high", (0.5, 0.8, "High (50%-80%)")),
        ("severe", (0.8, 1.0, "Severe (80%-100%)")),
    ]:
        mask = (constraint_share_clean >= lower) & (constraint_share_clean <= upper)
        if mask.sum() == 0:
            continue
        
        true_vals = constraint_share_clean[mask]
        pred_vals = predicted_risk_clean[mask]
        
        segments.append({
            "Constraint_Level": label,
            "Count": int(mask.sum()),
            "Pct": f"{100 * mask.sum() / len(constraint_share_clean):.1f}%",
            "Actual_Mean": float(np.mean(true_vals)),
            "Pred_Mean": float(np.mean(pred_vals)),
            "MAE": float(mean_absolute_error(true_vals, pred_vals)),
            "RMSE": float(np.sqrt(mean_squared_error(true_vals, pred_vals))),
            "Calibration_Gap": float(np.mean(pred_vals) - np.mean(true_vals)),
        })
    
    return pd.DataFrame(segments)


def export_evaluation_report(
    eval_dict: dict[str, Any],
    output_path: str | None = None,
) -> str:
    """
    Export evaluation metrics as formatted text report.
    
    Parameters
    ----------
    eval_dict : dict[str, Any]
        Evaluation metrics dictionary
    output_path : str, optional
        Path to save report. If None, returns as string only.
        
    Returns
    -------
    str
        Formatted report text
    """
    lines = [
        "="*70,
        "MODEL EVALUATION REPORT",
        "="*70,
        "",
        "REGRESSION METRICS",
        "-"*70,
    ]
    
    for key, value in eval_dict.items():
        if key != "calibration_analysis" and not isinstance(value, dict):
            if isinstance(value, float):
                lines.append(f"{key:.<50} {value:>15.6f}")
            else:
                lines.append(f"{key:.<50} {str(value):>15}")
    
    if "calibration_analysis" in eval_dict:
        lines.append("")
        lines.append("CALIBRATION ANALYSIS (by Probability Bin)")
        lines.append("-"*70)
        for bin_name, bin_stats in eval_dict["calibration_analysis"].items():
            lines.append(f"  {bin_name}: pred={bin_stats['pred_mean']:.3f}, "
                        f"actual={bin_stats['true_mean']:.3f}, "
                        f"error={bin_stats['calibration_error']:.3f}, "
                        f"n={bin_stats['count']}")
    
    report = "\n".join(lines)
    
    if output_path:
        with open(output_path, "w") as f:
            f.write(report)
    
    return report
