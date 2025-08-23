# src/python/config.py

from pathlib import Path

# Paths (relative to repo root)
DATA_DIR = Path("data")
COMBINED_CSV = DATA_DIR / "processed" / "combined_data_quarterly.csv"

MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
FIGURES_DIR = REPORTS_DIR / "figures"

# Target & label
TARGET_COL = "DRCCLACBS"       # delinquency rate series
FLAG_COL = "DefaultFlag"       # binary label column name

# Resampling (kept for reference if you generate combined CSV again)
TARGET_FREQ = "Q" #quaterly 
AGG = "mean"

# Train/test split
TEST_SIZE = 0.25
RANDOM_STATE = 42

# Logistic Regression
MAX_ITER = 5000
PENALTY = "l2"    # 'l1', 'l2', 'elasticnet', or 'none' (with saga solver)
SOLVER = "lbfgs"  # works with l2