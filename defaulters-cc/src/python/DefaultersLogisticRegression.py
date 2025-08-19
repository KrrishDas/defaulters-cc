import pandas as pd
import numpy as np
import glob, os

DATA_DIR = "data"
TARGET_FREQ = "Q"       # 'Q' for quarterly (quarter-end). Use 'M' for monthly if you prefer.
AGG = "mean"            # how to aggregate when downsampling: 'mean' or 'last'

def read_and_resample(path, target=TARGET_FREQ, agg=AGG):
    # Read
    df = pd.read_csv(path)

    # Find the date column (handles 'observation_date', 'DATE', etc.)
    date_cols = [c for c in df.columns if "date" in c.lower()]
    if not date_cols:
        raise ValueError(f"No date column found in {path}")
    date_col = date_cols[0]

    # Find the value column (assumes exactly one non-date column)
    value_cols = [c for c in df.columns if c != date_col]
    if len(value_cols) != 1:
        raise ValueError(f"Expected 1 value column in {path}, found {value_cols}")
    value_col = value_cols[0]

    # Parse & index by date
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()

    # Ensure numeric
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

    # Resample to target frequency
    if agg == "mean":
        df = df.resample(target).mean()
    elif agg == "last":
        df = df.resample(target).last()
    else:
        raise ValueError("AGG must be 'mean' or 'last'")

    # Fill gaps sensibly for modeling (time interpolation + forward/back fill)
    df[value_col] = df[value_col].interpolate(method="time").ffill().bfill()

    # Rename column to file stem (e.g., 'GDP.csv' -> 'GDP')
    series_name = os.path.splitext(os.path.basename(path))[0]
    df = df.rename(columns={value_col: series_name})

    return df

# # Build list of per-file resampled dataframes
# dfs = []
# for fp in glob.glob(os.path.join(DATA_DIR, "*.csv")):
#     try:
#         dfs.append(read_and_resample(fp))
#     except Exception as e:
#         print(f"⚠️ Skipped {os.path.basename(fp)}: {e}")

# # Combine on the date index (outer join keeps full coverage)
# combined = pd.concat(dfs, axis=1).sort_index()

# # Optionally: keep only rows where delinquency data exists
# if "DRCCLACBS" in combined.columns:
#     combined = combined[combined["DRCCLACBS"].notna()]

# combined = combined.reset_index().rename(columns={"index": "DATE"})
# combined.to_csv("combined_data_quarterly.csv", index=False)


df = pd.read_csv("combined_data_quarterly.csv")

# print(df.info())       # see datatypes and missing values
# print(df.describe())   # quick stats


threshold = df["DRCCLACBS"].median()
df["DefaultFlag"] = (df["DRCCLACBS"] > threshold).astype(int)

X = df.drop(columns=["observation_date", "DRCCLACBS", "DefaultFlag"])
y = df["DefaultFlag"]

X = X.fillna(X.mean())   # simplest: fill with column mean


#---------------------- CREATING THE MODEL--------------------------------- 

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# 1. Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # learns scale from training
X_test_scaled = scaler.transform(X_test)        # applies same scale to test

# 2. Initialize and fit the model
model = LogisticRegression(max_iter=5000)  # plenty of iterations to converge
model.fit(X_train_scaled, y_train)

# 3. Predictions
y_pred = model.predict(X_test_scaled)           # class labels (0 or 1)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]  # probability of 1

# 4. Evaluate
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nROC-AUC Score:", roc_auc_score(y_test, y_pred_proba))

coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_[0]
}).sort_values(by="Coefficient", ascending=False)
print(coefficients)


# plot the loss curve, r^2 - vanilla model is what we did
# penalize = loss of regression - put a penalty ; lorso, reach, elastic net? 
# time series data, think about how moving average, auto regression - giving a window range
