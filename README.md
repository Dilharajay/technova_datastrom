# Technova

## Prerequisites

- [Python](https://www.python.org/) (>= 3.14)
- [uv](https://github.com/astral-sh/uv) (for dependency management)

## Getting Started

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd technova_datastrom
    ```

2.  **Initialize the environment:**
    ```bash
    uv sync
    ```

3.  **Prepare the Data:**
    Create a folder named `data` in the project root and place the following CSV files inside it:
    - `transactions_history_final.csv`
    - `outlet_master.csv`
    - `outlet_coordinates.csv`
    - `holiday_list.csv`
    - `distributor_seasonality_details.csv`

4.  **Download Map Data:**
    Download the OpenStreetMap data for Sri Lanka in `.pbf` format. You can get the latest data from Geofabrik:
    ```bash
    wget https://download.geofabrik.de/asia/sri-lanka-latest.osm.pbf -O sri-lanka-latest.osm.pbf
    ```
    *Note: The analysis notebook may expect the file to be named `sri-lanka-260515.osm.pbf`. Ensure the filename matches the `PBF_FILE` variable in `main.ipynb`.*

## Usage

The main analysis is performed within the `main.ipynb` Jupyter notebook. 

1.  Activate the environment:
    ```bash
    source .venv/bin/activate  # On Linux/macOS
    # or
    .venv\Scripts\activate     # On Windows
    ```
2.  Launch Jupyter Lab or Notebook:
    ```bash
    jupyter lab
    ```
3.  Open and run `main.ipynb` to execute the data pipeline.

## Project Structure

- `data/`: Input CSV data files.
- `lakehouse/`: Lakehouse data storage:
    - `bronze/`: Raw data converted to Parquet format.
    - `silver/`: Cleaned and standardized data.
    - `gold/`: Aggregated and feature-engineered data for analysis.
    - `quarantine/`: Data that failed validation checks.
- `main.ipynb`: The primary analysis and processing notebook.
- `pyproject.toml`: Project configuration and dependencies.
