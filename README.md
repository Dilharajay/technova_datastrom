# TechNova Datastrom Competition Pipeline

A complete data science pipeline for demand forecasting and outlet sales analysis, featuring data ingestion, exploratory analysis, constraint detection, peer grouping, and machine learning model training.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Setup & Environment Configuration](#setup--environment-configuration)
- [Data Requirements](#data-requirements)
- [Quick Start](#quick-start)
- [Pipeline Overview](#pipeline-overview)
- [Running the Pipeline End-to-End](#running-the-pipeline-end-to-end)
- [Project Structure](#project-structure)
- [Outputs](#outputs)

## Prerequisites

- **Python** >= 3.14 ([Download](https://www.python.org/))
- **uv** (for fast dependency management) — [Installation Guide](https://docs.astral.sh/uv/#getting-started)
- **Git** (for version control)
- **Jupyter Lab/Notebook** (installed via dependencies)
- **Disk space** ~5-10 GB (for data lakehouse and models)
- **Internet connection** (for downloading OpenStreetMap data)

## Setup & Environment Configuration

### 1. Clone the Repository
```bash
git clone <repository-url>
cd technova_datastrom
```

### 2. Install Dependencies
```bash
# This creates a virtual environment and installs all dependencies
uv sync
```

### 3. Activate the Virtual Environment
**On Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
```

**On Windows (CMD):**
```cmd
.venv\Scripts\activate.bat
```

**On Linux/macOS:**
```bash
source .venv/bin/activate
```

Verify the environment is activated (you should see `(.venv)` in your terminal).

## Data Requirements

### Input CSV Files

Create a `data/` folder in the project root and place the following CSV files:

| File | Description |
|------|-------------|
| `transactions_history_final.csv` | Historical transaction data with outlet volumes |
| `outlet_master.csv` | Outlet metadata (type, size, location) |
| `outlet_coordinates.csv` | Outlet geographical coordinates |
| `holiday_list.csv` | Public holidays for the region |
| `distributor_seasonality_details.csv` | Seasonal patterns by distributor |

```bash
data/
├── transactions_history_final.csv
├── outlet_master.csv
├── outlet_coordinates.csv
├── holiday_list.csv
└── distributor_seasonality_details.csv
```

### OpenStreetMap Data (Optional but Recommended)

Download Sri Lanka map data for geospatial analysis:

```bash
# Using wget (Linux/macOS)
wget https://download.geofabrik.de/asia/sri-lanka-latest.osm.pbf -O sri-lanka-latest.osm.pbf

# Using curl (alternative)
curl -o sri-lanka-latest.osm.pbf https://download.geofabrik.de/asia/sri-lanka-latest.osm.pbf

# Using PowerShell (Windows)
Invoke-WebRequest -Uri "https://download.geofabrik.de/asia/sri-lanka-latest.osm.pbf" -OutFile "sri-lanka-latest.osm.pbf"
```

**Note:** The `main.ipynb` expects the filename to match the `PBF_FILE` variable. Update this variable if needed.

## Quick Start

For a quick smoke test of the pipeline:

```bash
# 1. Ensure you're in the project root and environment is activated
cd /path/to/technova_datastrom
source .venv/bin/activate  # or activate.bat on Windows

# 2. Start Jupyter Lab
jupyter lab

# 3. Open main.ipynb and run cells sequentially
```

## Pipeline Overview

The pipeline consists of multiple interconnected stages:

```
Raw CSV Data
    ↓
[main.ipynb] Data Ingestion & Lakehouse Setup
    ├─→ Bronze Layer (Raw Parquet)
    ├─→ Silver Layer (Cleaned & Standardized)
    └─→ Gold Layer (Feature-Engineered)
    ↓
[modeling/eda.ipynb] Exploratory Data Analysis
    ↓
[modeling/constraint_detection.ipynb] Constraint Detection
    ↓
[modeling/peer_grouping.ipynb] Peer Group Analysis
    ↓
[modeling/model.py] Model Training & Predictions
    ├─→ Constraint Model (LightGBM)
    └─→ Frontier Model (LightGBM)
    ↓
[modeling/generate_evaluation.py] Evaluation & Metrics
    ↓
Output (output/TechNova.csv + evaluation metrics)
```

## Running the Pipeline End-to-End

### Option 1: Using Jupyter Notebooks (Recommended for Development)

This is the most flexible and interactive approach.

#### Step 1: Launch Jupyter Lab
```bash
jupyter lab
```

The browser will open at `http://localhost:8888/`.

#### Step 2: Run the Main Pipeline (main.ipynb)
1. Open `main.ipynb`
2. Run cells in sequence (Shift + Enter or use "Run > Run All Cells")
3. This performs:
   - Data loading and validation
   - Lakehouse ingestion (Bronze → Silver → Gold)
   - Feature engineering
   - Data quality checks (invalid records moved to quarantine)

**Expected output:**
- `lakehouse/bronze/`: Raw data in Parquet format
- `lakehouse/silver/`: Cleaned data
- `lakehouse/gold/`: Feature-engineered data ready for modeling

#### Step 3: Exploratory Data Analysis (Optional)
```
Open: modeling/eda.ipynb
Run all cells to generate visualizations and statistical summaries
```

#### Step 4: Constraint Detection Analysis
```
Open: modeling/constraint_detection.ipynb
Run all cells to identify constrained outlets and demand patterns
```

#### Step 5: Peer Grouping Analysis (Optional)
```
Open: modeling/peer_grouping.ipynb
Run all cells to analyze outlet peer groups and benchmarks
```

#### Step 6: Model Training & Prediction
In Jupyter, create a new cell and run:
```python
import subprocess
subprocess.run(["python", "modeling/model.py"])
```

Or run directly in terminal:
```bash
cd modeling
python model.py
```

This trains two models:
- **Constraint Model**: Predicts which outlets are supply-constrained
- **Frontier Model**: Forecasts maximum monthly demand (potential)

**Expected output:**
- `output/TechNova.csv` - Final predictions with one row per outlet for Jan 2026
- `output/evaluation/*` - Model evaluation metrics and feature importance

#### Step 7: Generate Evaluation Reports
In terminal:
```bash
cd modeling
python generate_evaluation.py
```

**Expected output:**
- `output/evaluation/constraint_evaluation_matrix.csv`
- `output/evaluation/frontier_evaluation_matrix.csv`
- `output/evaluation/model_comparison_summary.csv`
- `output/evaluation/residual_analysis.csv`
- `output/evaluation/[constraint|frontier]_feature_importance.csv`
- `output/evaluation/[constraint|frontier]_model_metrics.json`

### Option 2: Full Automation (Batch Mode)

For production or automated runs:

```bash
# 1. Activate environment
source .venv/bin/activate

# 2. Run the main notebook programmatically
jupyter nbconvert --to notebook --execute main.ipynb

# 3. Train models
python modeling/model.py

# 4. Generate evaluations
python modeling/generate_evaluation.py

# 5. Check outputs
ls -la output/
```

### Option 3: Command-Line Quick Reference

```bash
# Setup
git clone <repo-url> && cd technova_datastrom
uv sync
source .venv/bin/activate  # Windows: .venv\Scripts\activate.bat

# Run
jupyter lab                          # Interactive mode
python modeling/model.py             # Train models only
python modeling/generate_evaluation.py  # Generate metrics

# Check outputs
cat output/TechNova.csv              # Predictions
cat output/evaluation/*.json         # Metrics
```

## Project Structure

```
technova_datastrom/
├── main.ipynb                    # Main data pipeline & ingestion
├── pyproject.toml                # Project dependencies & config
├── README.md                      # This file
│
├── data/                          # Input CSV files
│   ├── transactions_history_final.csv
│   ├── outlet_master.csv
│   ├── outlet_coordinates.csv
│   ├── holiday_list.csv
│   └── distributor_seasonality_details.csv
│
├── lakehouse/                     # Data staging layers
│   ├── bronze/                    # Raw parquet data
│   ├── silver/                    # Cleaned & validated data
│   ├── gold/                      # Feature-engineered data
│   └── quarantine/                # Failed validation records
│
├── modeling/                      # ML pipeline
│   ├── model.py                   # Model training (Constraint + Frontier)
│   ├── evaluation_metrics.py      # Metric calculations
│   ├── generate_evaluation.py     # Evaluation report generation
│   ├── eda.ipynb                  # Exploratory data analysis
│   ├── constraint_detection.ipynb # Constrained outlet detection
│   └── peer_grouping.ipynb        # Peer analysis notebook
│
├── utils/                         # Utilities
│   ├── __init__.py
│   ├── preprocessing.py           # Data preprocessing functions
│   ├── helpers.py                 # General helper functions
│   └── __pycache__/
│
├── output/                        # Final outputs
│   ├── TechNova.csv               # Predictions (main deliverable)
│   └── evaluation/                # Model evaluation artifacts
│       ├── constraint_evaluation_matrix.csv
│       ├── frontier_evaluation_matrix.csv
│       ├── model_comparison_summary.csv
│       ├── constraint_feature_importance.csv
│       ├── frontier_feature_importance.csv
│       ├── constraint_model_metrics.json
│       ├── frontier_model_metrics.json
│       └── residual_analysis.csv
│
└── docs/                          # Documentation (if exists)
```

## Outputs

### Primary Deliverable: `output/TechNova.csv`

Contains predictions for January 2026:

| Column | Description |
|--------|-------------|
| `Outlet_ID` | Unique outlet identifier |
| `Maximum_Monthly_Liters` | Predicted maximum monthly demand (potential) |
| `is_constrained` | Boolean indicating if outlet is supply-constrained |
| Other features | Relevant outlet attributes |

**Schema Example:**
```
Outlet_ID, Year, Month, Output_Prediction, is_constrained, ...
OUT001, 2026, 1, 5234.45, True
OUT002, 2026, 1, 3821.20, False
OUT003, 2026, 1, 7102.10, True
```

### Evaluation Artifacts: `output/evaluation/`

- **Model Metrics** (JSON): Precision, Recall, F1, MSE, RMSE, R²
- **Feature Importance** (CSV): Top 10 influential features per model
- **Evaluation Matrix** (CSV): Per-outlet predictions vs actual (test set)
- **Residual Analysis** (CSV): Error distribution and outlier analysis
- **Model Comparison** (CSV): Side-by-side constraint vs frontier metrics

## Troubleshooting

### Issue: Missing Data Files
**Solution:** Ensure `data/` folder contains all 5 CSV files
```bash
ls data/  # Check all files are present
```

### Issue: Python Version Mismatch
**Solution:** Verify Python >= 3.14
```bash
python --version
```

### Issue: Jupyter Won't Start
**Solution:** Reinstall dependencies
```bash
uv sync --reinstall
jupyter lab --version
```

### Issue: Memory Error During Training
**Solution:** Reduce batch size in `modeling/model.py` or close other applications

### Issue: Quarantine Folder Issues
**Solution:** Data quality warnings are logged in `lakehouse/quarantine/`. Review these files to understand data issues.

## Performance Notes

- **Data Ingestion**: ~2-5 minutes (depends on CSV size)
- **EDA & Analysis**: ~5-10 minutes
- **Model Training**: ~10-20 minutes (200k+ records)
- **Full Pipeline**: ~30-45 minutes end-to-end

Timing varies by hardware and dataset size.

## Contact & Support

For issues or questions:
1. Check the troubleshooting section above
2. Review error messages in Jupyter cell outputs
3. Check `lakehouse/quarantine/` for data quality issues
4. Consult repository documentation in `docs/`
