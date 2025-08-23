# src/python/data_prep.py

from pathlib import Path
import pandas as pd

from .config import TARGET_COL, FLAG_COL, COMBINED_CSV

def load_combined_csv(path: Path = COMBINED_CSV) -> pd.DataFrame:
    """Load the combined dataset and normalize date column name to 'DATE'.""" 
    df = pd.read_csv(path)
    # normalize date column name, 'DATE' will be used as the common key to merge the datasets
    date_cols = [c for c in df.columns if "date" in c.lower()]
    if not date_cols:
        raise ValueError("No date-like column found in combined CSV.")
    date_col = date_cols[0]
    df = df.rename(columns={date_col: "DATE"})
    return df

def make_default_flag(df: pd.DataFrame, target: str = TARGET_COL) -> pd.DataFrame:
    """Create a binary flag by splitting target at its median."""
    thr = df[target].median()
    df[FLAG_COL] = (df[target] > thr).astype(int) # for "DRCCLACBS", threshold = 0.5, splits to 0 and 1
    return df

def split_X_y(df: pd.DataFrame, target: str = TARGET_COL, flag: str = FLAG_COL):
    """Return X (features) and y (binary flag / label). Drops DATE, target, and flag."""
    drop_cols = [c for c in ["DATE", target, flag] if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df[flag]
    return X, y 