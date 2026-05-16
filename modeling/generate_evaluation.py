
# Generate comprehensive evaluation matrices for constraint and frontier models.
# Produces detailed performance analysis across segments and saves reports.


from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from evaluation_metrics import (
    build_constraint_matrix,
    build_evaluation_matrix,
    evaluate_constraint_model,
    evaluate_regression_arrays,
    export_evaluation_report,
    print_in_sample_metrics,
)
from model import (
    add_peer_features,
    add_time_and_lag_features,
    build_outlet_month_frame,
)


ROOT = Path(__file__).resolve().parents[1]
GOLD_PATH = ROOT / "lakehouse" / "gold" / "gold_with_peers.parquet"
OUTPUT_DIR = ROOT / "output" / "evaluation"
MODELS_DIR = OUTPUT_DIR / "models"


def save_evaluation_summary(
    summary: dict,
    filename: str,
) -> Path:
    """Save evaluation summary as JSON."""
    output_path = OUTPUT_DIR / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types to Python natives for JSON serialization
    def convert_to_native(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        return obj
    
    summary_native = convert_to_native(summary)
    with open(output_path, "w") as f:
        json.dump(summary_native, f, indent=2)
    return output_path


def generate_full_evaluation() -> None:
    """
    Load data, train models, and generate comprehensive evaluation matrices.
    """
    print("Loading data...")
    raw = pd.read_parquet(GOLD_PATH)
    
    # Prepare data
    monthly = build_outlet_month_frame(raw)
    monthly = add_time_and_lag_features(monthly)
    monthly = add_peer_features(monthly)
    
    # Categorical encoding
    categorical_cols = ["Distributor_ID", "Outlet_Type", "Outlet_Size", "Peer_Group_Clean"]
    for col in categorical_cols:
        monthly[col] = monthly[col].astype("category")
    
    # Feature columns
    feature_cols = [
        "Year",
        "Month",
        "year_offset",
        "month_sin",
        "month_cos",
        "lag_1",
        "lag_3",
        "lag_6",
        "rolling_3",
        "rolling_6",
        "Cooler_Count",
        "Latitude",
        "Longitude",
        "Holiday_Count",
        "poi_bus_stop_1500m",
        "poi_hospital_1500m",
        "poi_market_1500m",
        "poi_school_1500m",
        "poi_supermarket_1500m",
        "poi_tourism_1500m",
        "peer_month_median",
        "peer_month_p75",
        "peer_month_p90",
        "peer_overall_median",
        "peer_overall_p90",
        "Distributor_ID",
        "Outlet_Type",
        "Outlet_Size",
        "Peer_Group_Clean",
    ]
    
    # Fill sparse features
    for col in [
        "lag_1",
        "lag_3",
        "lag_6",
        "rolling_3",
        "rolling_6",
        "peer_month_median",
        "peer_month_p75",
        "peer_month_p90",
        "peer_overall_median",
        "peer_overall_p90",
    ]:
        monthly[col] = monthly[col].fillna(monthly["Volume_Liters"].median())
    
    train_df = monthly.loc[monthly["Year"] <= 2025].copy()
    
    print("\n" + "="*70)
    print("CONSTRAINT MODEL EVALUATION")
    print("="*70)
    
    # Train constraint model
    constraint_model = LGBMRegressor(
        objective="regression",
        n_estimators=500,
        learning_rate=0.04,
        num_leaves=48,
        min_child_samples=80,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
    )
    
    fit_df_constraint = train_df.dropna(subset=feature_cols + ["constrained_share"]).copy()
    constraint_model.fit(
        fit_df_constraint[feature_cols],
        fit_df_constraint["constrained_share"],
        categorical_feature=[c for c in categorical_cols if c in feature_cols],
    )
    
    # Predict on training data
    y_constraint_pred = np.clip(
        np.asarray(constraint_model.predict(train_df[feature_cols]), dtype=float),
        0.0,
        1.0,
    )
    
    # Evaluate constraint model
    constraint_metrics = evaluate_constraint_model(
        train_df["constrained_share"].to_numpy(),
        y_constraint_pred,
    )
    print_in_sample_metrics(constraint_metrics)
    
    # Constraint evaluation matrix
    constraint_matrix = build_constraint_matrix(
        train_df["constrained_share"].to_numpy(),
        y_constraint_pred,
    )
    print("\nConstraint Model Performance by Level:")
    print(constraint_matrix.to_string(index=False))
    
    # Save constraint results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    constraint_matrix.to_csv(OUTPUT_DIR / "constraint_evaluation_matrix.csv", index=False)
    save_evaluation_summary(constraint_metrics, "constraint_model_metrics.json")
    
    print("\n" + "="*70)
    print("FRONTIER MODEL EVALUATION")
    print("="*70)
    
    # Train frontier model
    monthly["constraint_risk_hat"] = np.clip(y_constraint_pred, 0.0, 1.0)
    train_df = monthly.loc[monthly["Year"] <= 2025].copy()
    
    frontier_model = LGBMRegressor(
        objective="quantile",
        alpha=0.92,
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=64,
        min_child_samples=80,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
    )
    
    fit_df_frontier = train_df.dropna(subset=feature_cols + ["Volume_Liters", "constraint_risk_hat"]).copy()
    fit_df_frontier["target_log1p"] = np.log1p(fit_df_frontier["Volume_Liters"].clip(lower=0.0))
    weights = (1.0 - fit_df_frontier["constraint_risk_hat"].clip(0.0, 1.0)).clip(lower=0.1)
    
    frontier_model.fit(
        fit_df_frontier[feature_cols],
        fit_df_frontier["target_log1p"],
        sample_weight=weights,
        categorical_feature=[c for c in categorical_cols if c in feature_cols],
    )
    
    # Predict on training data (transform back from log space)
    y_frontier_log = np.asarray(frontier_model.predict(train_df[feature_cols]), dtype=float)
    y_frontier_pred = np.expm1(y_frontier_log)
    
    # Evaluate frontier model
    frontier_metrics = evaluate_regression_arrays(
        train_df["Volume_Liters"].to_numpy(),
        y_frontier_pred,
        name="Frontier Volume",
    )
    print_in_sample_metrics(frontier_metrics)
    
    # Frontier evaluation matrix
    frontier_matrix = build_evaluation_matrix(
        train_df["Volume_Liters"].to_numpy(),
        y_frontier_pred,
    )
    print("\nFrontier Model Performance by Volume Segment:")
    print(frontier_matrix.to_string(index=False))
    
    # Save frontier results
    frontier_matrix.to_csv(OUTPUT_DIR / "frontier_evaluation_matrix.csv", index=False)
    save_evaluation_summary(frontier_metrics, "frontier_model_metrics.json")
    
    print("\n" + "="*70)
    print("MODEL COMPARISON SUMMARY")
    print("="*70)
    
    comparison = pd.DataFrame({
        "Metric": ["MAE", "RMSE", "MAPE", "R² Score", "Samples"],
        "Constraint Model": [
            f"{constraint_metrics['mean_absolute_error']:.4f}",
            f"{constraint_metrics['root_mean_squared_error']:.4f}",
            f"{constraint_metrics['mean_absolute_percentage_error']:.2%}",
            f"{constraint_metrics['r2_score']:.4f}",
            f"{constraint_metrics['n_samples']:,}",
        ],
        "Frontier Model": [
            f"{frontier_metrics['mean_absolute_error']:.4f}",
            f"{frontier_metrics['root_mean_squared_error']:.4f}",
            f"{frontier_metrics['mean_absolute_percentage_error']:.2%}",
            f"{frontier_metrics['r2_score']:.4f}",
            f"{frontier_metrics['n_samples']:,}",
        ],
    })
    
    print(comparison.to_string(index=False))
    comparison.to_csv(OUTPUT_DIR / "model_comparison_summary.csv", index=False)
    
    # Feature importance
    print("\n" + "="*70)
    print("FEATURE IMPORTANCE")
    print("="*70)
    
    constraint_importance = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": constraint_model.feature_importances_,
    }).sort_values("Importance", ascending=False).head(15)
    
    print("\nTop 15 Features - Constraint Model:")
    print(constraint_importance.to_string(index=False))
    constraint_importance.to_csv(OUTPUT_DIR / "constraint_feature_importance.csv", index=False)
    
    frontier_importance = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": frontier_model.feature_importances_,
    }).sort_values("Importance", ascending=False).head(15)
    
    print("\nTop 15 Features - Frontier Model:")
    print(frontier_importance.to_string(index=False))
    frontier_importance.to_csv(OUTPUT_DIR / "frontier_feature_importance.csv", index=False)
    
    # Residual analysis
    print("\n" + "="*70)
    print("RESIDUAL ANALYSIS")
    print("="*70)
    
    frontier_residuals = train_df["Volume_Liters"].to_numpy() - y_frontier_pred
    constraint_residuals = train_df["constrained_share"].to_numpy() - y_constraint_pred
    
    residual_stats = pd.DataFrame({
        "Model": ["Frontier", "Constraint"],
        "Mean Residual": [float(np.mean(frontier_residuals)), float(np.mean(constraint_residuals))],
        "Std Residual": [float(np.std(frontier_residuals)), float(np.std(constraint_residuals))],
        "Min Residual": [float(np.min(frontier_residuals)), float(np.min(constraint_residuals))],
        "Max Residual": [float(np.max(frontier_residuals)), float(np.max(constraint_residuals))],
        "Median Abs Residual": [float(np.median(np.abs(frontier_residuals))), float(np.median(np.abs(constraint_residuals)))],
    })
    
    print(residual_stats.to_string(index=False))
    residual_stats.to_csv(OUTPUT_DIR / "residual_analysis.csv", index=False)
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    print(f"Results saved to: {OUTPUT_DIR}")
    print("\nOutput files:")
    print("  • constraint_evaluation_matrix.csv")
    print("  • frontier_evaluation_matrix.csv")
    print("  • model_comparison_summary.csv")
    print("  • constraint_feature_importance.csv")
    print("  • frontier_feature_importance.csv")
    print("  • residual_analysis.csv")
    print("  • constraint_model_metrics.json")
    print("  • frontier_model_metrics.json")


if __name__ == "__main__":
    generate_full_evaluation()
