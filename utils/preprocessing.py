import pandas as pd
import numpy as np
from pathlib import Path

def qurantine(df: pd.DataFrame, dataset_name: str, reason: str):
    if df.empty:
        return
    QURANTINE_PATH = "lakehouse/quarantine"
    df = df.copy()
    df['_failure_reason'] = reason
    df['_source_dataset'] = dataset_name
    path = Path(f"{QURANTINE_PATH}/{dataset_name}_{reason}.parquet")
    df.to_parquet(path, compression="zstd", index=False)

# !TODO: add success msg via print or python loggin


# check duplicates
def check_duplicates(df: pd.DataFrame, columns, dataset_name) -> pd.DataFrame:
    duplicates = df[df.duplicated(subset=columns, keep=False)]
    qurantine(duplicates, dataset_name, "duplicated_record")
    return df.drop_duplicates(subset=columns, keep='first')

# null check
def check_nulls(df: pd.DataFrame, columns, dataset_name) -> pd.DataFrame:
    mask = df[columns].isnull().any(axis=1)
    qurantine(df[mask], dataset_name, "null_in_mandatory_field")
    return df[~mask]

# key integrity check
def check_referential_integrity(df: pd.DataFrame, fk_col, ref_df: pd.DataFrame, ref_col, dataset_name):
    valid_keys = set(ref_df[ref_col])
    mask = ~df[fk_col].isin(valid_keys)
    qurantine(df[mask], dataset_name, f"invalid_fk_{fk_col}")
    return df[~mask]


def check_value_range(df: pd.DataFrame, col, min_val: int|float, max_val: int|float, dataset_name):
    mask = (df[col] < min_val) | (df[col] > max_val)
    qurantine(df[mask], dataset_name, f"{col}_out_of_range")
    return df[~mask]


def check_format(df: pd.DataFrame, col, dtype: np.dtype, dataset_name):
    try:
        df[col] = df[col].astype(dtype)
        return df
    except Exception:
        mask = pd.to_numeric(df[col], errors='coerce').isnull()
        qurantine(df[mask], dataset_name, f"{col}_type_mismatch")
        return df[~mask]


def check_geo_bounds(df, latitude, longitude, dataset_name) -> pd.DataFrame:

    lat = df[latitude]
    lon = df[longitude]

    lat_valid = (lat >= -90) & (lat <= 90)
    lon_valid = (lon >= -180) & (lon <= 180)

    not_zero_zero = ~((lat == 0) & (lon == 0))

    valid_mask = lat_valid & lon_valid & not_zero_zero

    invalid_mask = ~valid_mask

    if invalid_mask.any():
        qurantine(df[invalid_mask], dataset_name, f"{latitude}_{longitude}_geo_out_of_bounds")

    return df[valid_mask]


def check_datetime_format(df, column, dataset_name, date_format=None) -> pd.DataFrame:
    try:
        if date_format:
            df[column] = pd.to_datetime(df[column], format=date_format)
        else:
            df[column] = pd.to_datetime(df[column])
        return df
    except Exception:
        if date_format:
            converted = pd.to_datetime(df[column], format=date_format, errors='coerce')
        else:
            converted = pd.to_datetime(df[column], errors='coerce')

        invalid_mask = converted.isna()
        qurantine(df[invalid_mask], dataset_name, f"{column}_type_mismatch")

        valid_df = df[~invalid_mask].copy()
        valid_df[column] = converted[~invalid_mask]
        return valid_df


def fix_outlet_type(df, col='Outlet_Type', dataset_name=None, quarantine_unknown=False):
    corrections = {
        'Bakry': 'Bakery',
        'Grocry': 'Grocery',
        ' Eatery': 'Eatery',
        'eatery': 'Eatery',
        'bakery': 'Bakery',
        'grocery': 'Grocery',
        'hotel': 'Hotel',
        'kiosk': 'Kiosk',
        'pharmacy': 'Pharmacy',
        'smmt': 'SMMT',
    }

    valid_types = {'Hotel', 'Grocery', 'SMMT', 'Pharmacy', 'Kiosk', 'Bakery', 'Eatery'}

    df = df.copy()
    df[col] = df[col].astype(str).str.strip()
    df[col] = df[col].replace(corrections)

    if quarantine_unknown and dataset_name:
        unknown_mask = ~df[col].isin(valid_types)
        if unknown_mask.any():
            qurantine(df[unknown_mask], dataset_name, f"{col}_unknown_value")

        return df[~unknown_mask]
    else:
        return df