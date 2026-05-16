import lightgbm as lgb
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import json
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from evaluation_metrics import evaluate_regression_arrays, build_evaluation_matrix, print_in_sample_metrics

ROOT = Path(__file__).resolve().parents[1]

df = pd.read_parquet(ROOT / 'lakehouse/gold/master_features.parquet')

aggregation_rules = {
    'Volume_Liters': 'sum',    
    'Total_Bill_Value': 'sum',  
    'Cooler_Count': 'first',     
    'Outlet_Size': 'first',       
    'Outlet_Type': 'first',        
    'Latitude': 'first',            
    'Longitude': 'first',            
    'Seasonality_Index': 'first',     
    'Holiday_Count': 'first',         
    'poi_bus_stop_1500m': 'first',    
    'poi_hospital_1500m': 'first',    
    'poi_market_1500m': 'first',      
    'poi_school_1500m': 'first',      
    'poi_supermarket_1500m': 'first', 
    'poi_tourism_1500m': 'first'
}

df_monthly = df.groupby(['Outlet_ID', 'Year', 'Month']).agg(aggregation_rules).reset_index()

#calculate hist max
historical_max = df_monthly.groupby('Outlet_ID')['Volume_Liters'].max().reset_index()

historical_max.rename(columns={'Volume_Liters': 'Historical_Max_Volume'}, inplace=True)
df_monthly = df_monthly.merge(historical_max, on='Outlet_ID', how='left')

outlet_agg = df_monthly.groupby('Outlet_ID').agg(
    cv        = ('Volume_Liters', lambda x: x.std() / x.mean() if x.mean() > 0 else 0),
    repeat_r  = ('Volume_Liters', lambda x: 1 - x.nunique() / len(x)),
    round_r   = ('Volume_Liters', lambda x: (x % 10 == 0).mean()),
).reset_index()

outlet_agg['is_censored'] = (
    (outlet_agg['cv'] < 0.15).astype(int) +
    (outlet_agg['repeat_r'] > 0.5).astype(int) +
    (outlet_agg['round_r'] > 0.7).astype(int)
) >= 2

df_monthly = df_monthly.merge(outlet_agg[['Outlet_ID', 'is_censored']], on='Outlet_ID')

cohort_p90 = df_monthly.groupby(['Outlet_Type', 'Outlet_Size'])['Volume_Liters'] \
    .quantile(0.90).reset_index().rename(columns={'Volume_Liters': 'cohort_p90'})

df_monthly = df_monthly.merge(cohort_p90, on=['Outlet_Type', 'Outlet_Size'], how='left')

outlet_level = df_monthly.groupby('Outlet_ID').agg(
    Cooler_Count         = ('Cooler_Count', 'first'),
    Latitude             = ('Latitude', 'first'),
    Longitude            = ('Longitude', 'first'),
    Outlet_Size          = ('Outlet_Size', 'first'),
    Outlet_Type          = ('Outlet_Type', 'first'),
    poi_bus_stop_1500m   = ('poi_bus_stop_1500m', 'first'),
    poi_hospital_1500m   = ('poi_hospital_1500m', 'first'),
    poi_market_1500m     = ('poi_market_1500m', 'first'),
    poi_school_1500m     = ('poi_school_1500m', 'first'),
    poi_supermarket_1500m= ('poi_supermarket_1500m', 'first'),
    poi_tourism_1500m    = ('poi_tourism_1500m', 'first'),
    cohort_p90           = ('cohort_p90', 'first'),
    is_censored          = ('is_censored', 'first'),
    target               = ('Volume_Liters', 'max'),
).reset_index()

outlet_level['target'] = np.where(
    outlet_level['is_censored'],
    outlet_level['cohort_p90'],
    outlet_level['target']
)

# make it numerical cause feature importance without this is 9
size_order = {'Small': 1, 'Medium': 2, 'Large': 3, 'Extra Large': 4}
outlet_level['Outlet_Size_Ord'] = outlet_level['Outlet_Size'].astype(str).map(size_order).fillna(1)

coords = outlet_level[['Latitude', 'Longitude']].values

# find natural clusters in your actual outlet distribution
kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
outlet_level['cluster'] = kmeans.fit_predict(coords)
centers = kmeans.cluster_centers_

# distance to nearest cluster center = urbanness proxy
dists = cdist(coords, centers, metric='euclidean')
outlet_level['dist_to_nearest_center'] = dists.min(axis=1)

# density of each cluster = how many outlets share this center
cluster_density = outlet_level['cluster'].value_counts().rename('cluster_density')
outlet_level = outlet_level.merge(cluster_density, left_on='cluster', right_index=True)

cap = outlet_level['target'].quantile(0.99)
outlet_level['target'] = outlet_level['target'].clip(upper=cap)

features = [
    'Cooler_Count',
    'poi_bus_stop_1500m', 'poi_hospital_1500m', 'poi_market_1500m',
    'poi_school_1500m', 'poi_supermarket_1500m', 'poi_tourism_1500m',
    'Outlet_Size_Ord', 'Outlet_Type', 'cohort_p90', 'dist_to_nearest_center','cluster_density', 'cluster'
]
cat_cols = ['Outlet_Type']

X_train = outlet_level[features]
y_train = outlet_level['target']   # already the ceiling

# Drop quantile objective — you've already built the ceiling into y_train
# Use regular regression to predict it cleanly
model = lgb.LGBMRegressor(
    objective='regression',
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    num_leaves=31,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train, categorical_feature=cat_cols)

outlets_2026 = outlet_level[['Outlet_ID'] + features].copy()
# build Outlet -> Distributor mapping from latest available transaction per outlet

tx = pd.read_parquet(ROOT / 'lakehouse/silver/transactions_history.parquet')

outlet_dist = (
    tx.sort_values(['Year', 'Month'])
      .groupby('Outlet_ID', as_index=False)
      .tail(1)[['Outlet_ID', 'Distributor_ID']]
      .drop_duplicates('Outlet_ID')
)

# January seasonality per distributor (latest year)
distributor_seasonality_details = pd.read_parquet(ROOT / 'lakehouse/silver/distributor_seasonality_details.parquet')
jan_season = (
    distributor_seasonality_details[distributor_seasonality_details['Month'] == 1]
    .sort_values('Year')
    .drop_duplicates('Distributor_ID', keep='last')[['Distributor_ID', 'Seasonality_Index']]
)

outlets_2026 = outlets_2026.merge(outlet_dist, on='Outlet_ID', how='left')
outlets_2026 = outlets_2026.merge(jan_season, on='Distributor_ID', how='left')

season_map = {'Favorable': 1.2, 'Moderate': 1.0, 'Un-Favorable': 0.8}
outlets_2026['season_mult'] = outlets_2026['Seasonality_Index'].astype(str).map(season_map).fillna(1.0)

outlets_2026['Maximum_Monthly_Liters'] = (
    model.predict(outlets_2026[features]) * outlets_2026['season_mult']
)

outlets_2026['Maximum_Monthly_Liters'] = np.maximum(
    outlets_2026['Maximum_Monthly_Liters'],
    outlet_level.set_index('Outlet_ID')['target'].reindex(outlets_2026['Outlet_ID']).values
)

outlets_2026['Maximum_Monthly_Liters'] = outlets_2026['Maximum_Monthly_Liters'].clip(lower=0).round(2)

# ============================================================
# EVALUATION METRICS & DIAGNOSTICS
# ============================================================

# Generate training predictions for evaluation
y_train_true = outlet_level['target'].values
y_train_pred = model.predict(outlet_level[features])

# Compute in-sample metrics
train_metrics = evaluate_regression_arrays(y_train_true, y_train_pred, name="Maximum_Monthly_Liters")
print_in_sample_metrics(train_metrics)

# Build evaluation matrix by volume segment
eval_matrix = build_evaluation_matrix(y_train_true, y_train_pred, quantiles=[0.25, 0.5, 0.75])

# Compute residuals for analysis
residuals = y_train_pred - y_train_true
abs_errors = np.abs(residuals)
ape = np.abs((y_train_pred - y_train_true) / (y_train_true + 1e-6))

# Create comprehensive assessment summary
assessment = {
    'Model': 'LightGBM Regression',
    'Target': 'Maximum Monthly Liters (Outlet Potential)',
    'Training_Samples': int(len(outlet_level)),
    'Hyperparameters': {
        'Objective': 'regression',
        'Learning_Rate': 0.05,
        'N_Estimators': 500,
        'Max_Depth': 6,
        'Num_Leaves': 31,
        'Random_State': 42,
    },
    'Features_Used': len(features),
    'Performance_Metrics': {
        'MAE_Liters': float(train_metrics['mean_absolute_error']),
        'RMSE_Liters': float(train_metrics['root_mean_squared_error']),
        'MAPE_Percent': f"{train_metrics['mean_absolute_percentage_error']*100:.2f}%",
        'R2_Score': float(train_metrics['r2_score']),
        'Normalized_RMSE': float(train_metrics['normalized_rmse']),
        'Bias_Liters': float(train_metrics['bias']),
        'Median_AE_Liters': float(train_metrics['median_absolute_error']),
    },
    'Target_Distribution': {
        'Min_Liters': float(train_metrics['min_value']),
        'Max_Liters': float(train_metrics['max_value']),
        'Mean_Liters': float(train_metrics['mean_value']),
        'Std_Dev': float(np.std(y_train_true)),
    },
    'Residuals_Summary': {
        'Mean': float(np.mean(residuals)),
        'Std': float(np.std(residuals)),
        'Min': float(np.min(residuals)),
        'Max': float(np.max(residuals)),
        'Median': float(np.median(residuals)),
    },
}

# Save evaluation outputs
output_eval_dir = ROOT / 'output' / 'evaluation'
output_eval_dir.mkdir(parents=True, exist_ok=True)

# 1. Save evaluation matrix by volume segment
eval_matrix.to_csv(output_eval_dir / 'model_evaluation_matrix.csv', index=False)

# 2. Save model assessment as JSON
with open(output_eval_dir / 'model_metrics.json', 'w') as f:
    json.dump(assessment, f, indent=2)

# 3. Save feature importance
feature_importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_,
}).sort_values('Importance', ascending=False)
feature_importance_df.to_csv(output_eval_dir / 'feature_importance.csv', index=False)

# 4. Save residual analysis (detailed)
residual_df = pd.DataFrame({
    'Outlet_ID': outlet_level['Outlet_ID'],
    'Outlet_Type': outlet_level['Outlet_Type'],
    'Outlet_Size': outlet_level['Outlet_Size'],
    'Actual_Liters': y_train_true,
    'Predicted_Liters': y_train_pred,
    'Residual_Liters': residuals,
    'Abs_Error': abs_errors,
    'APE_Percent': ape * 100,
}).sort_values('Abs_Error', ascending=False)
residual_df.to_csv(output_eval_dir / 'residual_analysis.csv', index=False)

print(f"\nEvaluation files saved to: {output_eval_dir}/")
print(f"  [OK] model_evaluation_matrix.csv")
print(f"  [OK] model_metrics.json")
print(f"  [OK] feature_importance.csv")
print(f"  [OK] residual_analysis.csv")

# Save feature importance visualization
lgb.plot_importance(model, max_num_features=15)
plt.savefig(ROOT / 'output' / 'feature_importance.png', bbox_inches='tight')
plt.close()
print(f"  [OK] feature_importance.png")