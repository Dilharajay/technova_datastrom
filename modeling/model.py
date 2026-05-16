from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm import LGBMRegressor

from evaluation_metrics import evaluate_regression_arrays, print_in_sample_metrics

ROOT = Path(__file__).resolve().parents[1]
GOLD_PATH = ROOT / "lakehouse" / "gold" / "gold_with_peers.parquet"
OUTPUT_DIR = ROOT / "output"
OUTPUT_PATH = OUTPUT_DIR / "TechNova.csv"


def _normalize_text(series: pd.Series, fallback: str = "UNKNOWN") -> pd.Series:
	normalized = series.astype("string").str.strip().fillna(fallback)
	return normalized.replace("", fallback)


def build_outlet_month_frame(df: pd.DataFrame) -> pd.DataFrame:
	df = df.copy()

	df["Outlet_ID"] = _normalize_text(df["Outlet_ID"])
	df["Distributor_ID"] = _normalize_text(df["Distributor_ID"])
	df["Outlet_Type"] = _normalize_text(df["Outlet_Type"])
	df["Outlet_Size"] = _normalize_text(df["Outlet_Size"])

	# Existing Peer_Group is malformed in this dataset; rebuild a clean peer signature.
	df["Peer_Group_Clean"] = (
		df["Outlet_Type"] + "|" + df["Outlet_Size"] + "|" + df["Distributor_ID"]
	)

	grouped = (
		df.groupby(["Outlet_ID", "Year", "Month"], as_index=False)
		.agg(
			Volume_Liters=("Volume_Liters", "sum"),
			constrained_share=("is_constrained", "mean"),
			Distributor_ID=("Distributor_ID", "first"),
			Outlet_Type=("Outlet_Type", "first"),
			Outlet_Size=("Outlet_Size", "first"),
			Cooler_Count=("Cooler_Count", "mean"),
			Latitude=("Latitude", "mean"),
			Longitude=("Longitude", "mean"),
			Holiday_Count=("Holiday_Count", "mean"),
			poi_bus_stop_1500m=("poi_bus_stop_1500m", "mean"),
			poi_hospital_1500m=("poi_hospital_1500m", "mean"),
			poi_market_1500m=("poi_market_1500m", "mean"),
			poi_school_1500m=("poi_school_1500m", "mean"),
			poi_supermarket_1500m=("poi_supermarket_1500m", "mean"),
			poi_tourism_1500m=("poi_tourism_1500m", "mean"),
			Peer_Group_Clean=("Peer_Group_Clean", "first"),
		)
		.sort_values(["Outlet_ID", "Year", "Month"])
	)

	grouped["ds"] = pd.to_datetime(
		grouped["Year"].astype(int).astype(str)
		+ "-"
		+ grouped["Month"].astype(int).astype(str).str.zfill(2)
		+ "-01"
	)

	return grouped


def add_time_and_lag_features(df: pd.DataFrame) -> pd.DataFrame:
	df = df.copy()

	month_angle = 2.0 * np.pi * (df["Month"].astype(float) / 12.0)
	df["month_sin"] = np.sin(month_angle)
	df["month_cos"] = np.cos(month_angle)
	df["year_offset"] = df["Year"].astype(int) - int(df["Year"].min())

	df = df.sort_values(["Outlet_ID", "ds"]).copy()
	by_outlet = df.groupby("Outlet_ID", observed=True)["Volume_Liters"]
	df["lag_1"] = by_outlet.shift(1)
	df["lag_3"] = by_outlet.shift(3)
	df["lag_6"] = by_outlet.shift(6)
	df["rolling_3"] = by_outlet.transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
	df["rolling_6"] = by_outlet.transform(lambda s: s.shift(1).rolling(6, min_periods=1).mean())

	return df


def add_peer_features(df: pd.DataFrame) -> pd.DataFrame:
	df = df.copy()

	# For each month and peer group, use robust distribution summaries.
	peer_month = (
		df.groupby(["Peer_Group_Clean", "Month"], observed=True)["Volume_Liters"]
		.agg(
			peer_month_median="median",
			peer_month_p75=lambda s: s.quantile(0.75),
			peer_month_p90=lambda s: s.quantile(0.90),
		)
		.reset_index()
	)
	df = df.merge(peer_month, on=["Peer_Group_Clean", "Month"], how="left")

	peer_overall = (
		df.groupby("Peer_Group_Clean", observed=True)["Volume_Liters"]
		.agg(peer_overall_median="median", peer_overall_p90=lambda s: s.quantile(0.90))
		.reset_index()
	)
	df = df.merge(peer_overall, on="Peer_Group_Clean", how="left")
	df["peer_rel_perf"] = df["Volume_Liters"] / df["peer_month_median"].clip(lower=1.0)

	return df


def train_constraint_model(
	train_df: pd.DataFrame,
	feature_cols: list[str],
	categorical_cols: list[str],
) -> LGBMRegressor:
	fit_df = train_df.dropna(subset=feature_cols + ["constrained_share"]).copy()

	model = LGBMRegressor(
		objective="regression",
		n_estimators=500,
		learning_rate=0.04,
		num_leaves=48,
		min_child_samples=80,
		subsample=0.9,
		colsample_bytree=0.9,
		random_state=42,
	)

	model.fit(
		fit_df[feature_cols],
		fit_df["constrained_share"],
		categorical_feature=[c for c in categorical_cols if c in feature_cols],
	)

	return model


def train_frontier_model(
	train_df: pd.DataFrame,
	feature_cols: list[str],
	categorical_cols: list[str],
) -> LGBMRegressor:
	fit_df = train_df.dropna(subset=feature_cols + ["Volume_Liters", "constraint_risk_hat"]).copy()
	fit_df["target_log1p"] = np.log1p(fit_df["Volume_Liters"].clip(lower=0.0))

	# Down-weight observations likely to be censored by constraints.
	weights = (1.0 - fit_df["constraint_risk_hat"].clip(0.0, 1.0)).clip(lower=0.1)

	model = LGBMRegressor(
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

	model.fit(
		fit_df[feature_cols],
		fit_df["target_log1p"],
		sample_weight=weights,
		categorical_feature=[c for c in categorical_cols if c in feature_cols],
	)

	return model


def build_jan_2026_frame(history: pd.DataFrame) -> pd.DataFrame:
	base_cols = [
		"Outlet_ID",
		"Year",
		"Month",
		"ds",
		"Volume_Liters",
		"Distributor_ID",
		"Outlet_Type",
		"Outlet_Size",
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
		"Peer_Group_Clean",
	]
	history_base = history[base_cols].copy()

	outlets = history_base["Outlet_ID"].drop_duplicates().to_frame()
	latest = (
		history_base.sort_values("ds")
		.groupby("Outlet_ID", observed=True)
		.tail(1)
		.set_index("Outlet_ID")
	)

	jan = outlets.copy()
	jan["Year"] = 2026
	jan["Month"] = 1
	jan["ds"] = pd.Timestamp("2026-01-01")

	cols_from_latest = [
		"Distributor_ID",
		"Outlet_Type",
		"Outlet_Size",
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
		"Peer_Group_Clean",
	]

	jan = jan.merge(latest[cols_from_latest], left_on="Outlet_ID", right_index=True, how="left")

	jan = add_time_and_lag_features(pd.concat([history_base, jan], ignore_index=True, sort=False))
	jan = jan[jan["Year"] == 2026].copy()
	jan = add_peer_features(pd.concat([history_base, jan], ignore_index=True, sort=False))
	jan = jan[jan["Year"] == 2026].copy()

	# Fill lag features with outlet historical stats when short history exists.
	hist_stats = (
		history_base.groupby("Outlet_ID", observed=True)["Volume_Liters"]
		.agg(hist_median="median", hist_max="max")
		.reset_index()
	)
	jan = jan.merge(hist_stats, on="Outlet_ID", how="left")

	for lag_col in ["lag_1", "lag_3", "lag_6", "rolling_3", "rolling_6"]:
		jan[lag_col] = jan[lag_col].fillna(jan["hist_median"])

	for col in ["peer_month_median", "peer_month_p75", "peer_overall_median", "peer_overall_p90"]:
		jan[col] = jan[col].fillna(jan["hist_median"])
	jan["peer_month_p90"] = jan["peer_month_p90"].fillna(jan["hist_median"])

	return jan


def build_outlet_strength(history: pd.DataFrame) -> pd.DataFrame:
	stable = history[history["constrained_share"] <= 0.2].copy()
	outlet_strength = (
		stable.groupby("Outlet_ID", observed=True)["peer_rel_perf"]
		.median()
		.clip(lower=0.7, upper=1.7)
		.rename("outlet_strength")
		.reset_index()
	)
	return outlet_strength


def predict_potential(
	frontier_model: LGBMRegressor,
	jan_df: pd.DataFrame,
	feature_cols: list[str],
) -> pd.DataFrame:
	pred_log = np.asarray(frontier_model.predict(jan_df[feature_cols]), dtype=float)
	pred_frontier = np.expm1(pred_log)
	out = jan_df[["Outlet_ID", "hist_max", "peer_month_p90", "outlet_strength", "constraint_risk_hat"]].copy()

	peer_frontier = out["peer_month_p90"].to_numpy(dtype=float) * out["outlet_strength"].to_numpy(dtype=float)

	# Latent uncapping rule:
	# potential = max(frontier_model, peer_frontier, historical_max) * (1 + lambda * risk)
	# where lambda controls additional upside when constraints are likely.
	base = np.column_stack(
		[
			np.asarray(pred_frontier, dtype=float),
			out["hist_max"].to_numpy(dtype=float),
			np.asarray(peer_frontier, dtype=float),
		]
	)
	latent_base = np.max(base, axis=1)
	risk = out["constraint_risk_hat"].clip(lower=0.0, upper=1.0).to_numpy(dtype=float)
	risk_uplift = 1.0 + (0.35 * risk)

	out["Maximum_Monthly_Liters (Potential)"] = np.clip(latent_base * risk_uplift, a_min=0.0, a_max=None)

	return out[["Outlet_ID", "Maximum_Monthly_Liters (Potential)"]]


def save_feature_importance_plot(
	model: LGBMRegressor,
	output_path: Path,
	max_num_features: int = 20,
) -> None:
	"""Save feature-importance plot without failing when model has no useful splits."""
	try:
		ax = lgb.plot_importance(
			model,
			max_num_features=max_num_features,
			ignore_zero=False,
		)
		ax.figure.savefig(output_path, bbox_inches="tight")
		ax.figure.clf()
	except ValueError:
		print(f"Skipping feature importance plot for {output_path.name}: no feature importances available.")


def main() -> None:
	print("Loading data and building features...")
	if not GOLD_PATH.exists():
		raise FileNotFoundError(f"Dataset not found: {GOLD_PATH}")
	else:
		print(f"Found dataset at: {GOLD_PATH}")

	raw = pd.read_parquet(GOLD_PATH)
	monthly = build_outlet_month_frame(raw)
	monthly = add_time_and_lag_features(monthly)
	monthly = add_peer_features(monthly)

	categorical_cols = ["Distributor_ID", "Outlet_Type", "Outlet_Size", "Peer_Group_Clean"]
	for col in categorical_cols:
		monthly[col] = monthly[col].astype("category")

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

	# Fill sparse lag/peer fields for early periods while preserving distribution shape.
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

	print("Training constraint model...")

	constraint_model = train_constraint_model(
		train_df=train_df,
		feature_cols=feature_cols,
		categorical_cols=categorical_cols,
	)
	monthly_risk = np.asarray(constraint_model.predict(monthly[feature_cols]), dtype=float)
	monthly["constraint_risk_hat"] = np.clip(monthly_risk, 0.0, 1.0)
	train_df = monthly.loc[monthly["Year"] <= 2025].copy()

	print("Training frontier model...")

	frontier_model = train_frontier_model(
		train_df=train_df,
		feature_cols=feature_cols,
		categorical_cols=categorical_cols,
	)

	# In-sample metrics on liters scale for quick training diagnostics.
	y_train_liters = train_df["Volume_Liters"].to_numpy(dtype=float)
	y_train_pred_liters = np.expm1(
		np.asarray(frontier_model.predict(train_df[feature_cols]), dtype=float)
	)
	train_metrics = evaluate_regression_arrays(y_train_liters, y_train_pred_liters)
	print_in_sample_metrics(train_metrics)

	jan_2026 = build_jan_2026_frame(monthly)
	jan_risk = np.asarray(constraint_model.predict(jan_2026[feature_cols]), dtype=float)
	jan_2026["constraint_risk_hat"] = np.clip(jan_risk, 0.0, 1.0)
	outlet_strength = build_outlet_strength(monthly)
	jan_2026 = jan_2026.merge(outlet_strength, on="Outlet_ID", how="left")
	jan_2026["outlet_strength"] = jan_2026["outlet_strength"].fillna(1.0)

	for col in categorical_cols:
		jan_2026[col] = jan_2026[col].astype("category")

	predictions = predict_potential(
		frontier_model=frontier_model,
		jan_df=jan_2026,
		feature_cols=feature_cols,
	)

	predictions.to_csv(OUTPUT_PATH, index=False)
	print("Prediction complete.")
	print(f"Saved predictions to: {OUTPUT_PATH}")
	print(f"Rows: {len(predictions):,}")


if __name__ == "__main__":
	main()
