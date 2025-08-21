# src/python/permutation_importance.py
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

# --- paths ---
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed" / "combined_data_quarterly.csv"
FIGS = ROOT / "reports" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

BEST_CFG = ROOT / "reports" / "best_regularization.json"
OUT_JSON_TOP15 = ROOT / "reports" / "feature_importance_top15.json"
OUT_CSV_FULL = ROOT / "reports" / "feature_importance_full.csv"

# --- data ---
df = pd.read_csv(DATA, parse_dates=["observation_date"])
threshold = df["DRCCLACBS"].median()
df["DefaultFlag"] = (df["DRCCLACBS"] > threshold).astype(int)

X = df.drop(columns=["observation_date", "DRCCLACBS", "DefaultFlag"])
y = df["DefaultFlag"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# --- choose configuration: load best from experiments if available ---
penalty = "elasticnet"
C = 1.0
l1_ratio = 0.5
solver = "saga"  # supports l1 and elasticnet

if BEST_CFG.exists():
    with open(BEST_CFG, "r") as f:
        best = json.load(f)
    model_key = best.get("model", "elasticnet")
    C = float(best.get("C", 1.0))
    l1_ratio = best.get("l1_ratio", None)
    if model_key == "l1":
        penalty = "l1"
        l1_ratio = None
    elif model_key == "l2":
        penalty = "l2"
        l1_ratio = None
    else:
        penalty = "elasticnet"
        # ensure float if present
        l1_ratio = None if l1_ratio is None else float(l1_ratio)

# build pipeline with chosen config
clf = LogisticRegression(
    solver=solver,
    penalty=penalty,
    C=C,
    l1_ratio=l1_ratio,
    max_iter=5000,
    random_state=42
)
pipe = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("clf", clf)
])
pipe.fit(X_train, y_train)

proba = pipe.predict_proba(X_test)[:,1]
print("Test ROC-AUC:", roc_auc_score(y_test, proba))

# --- permutation importance ---
r = permutation_importance(
    pipe, X_test, y_test,
    scoring="roc_auc", n_repeats=30, random_state=42
)
importances = pd.Series(r.importances_mean, index=X.columns)\
                .sort_values(ascending=True)

# Save full ranking as CSV
importances_full_df = importances.sort_values(ascending=False).to_frame(name="importance")
importances_full_df.to_csv(OUT_CSV_FULL, index=True)

# Save top-15 as JSON (list of {feature, importance})
top15 = importances.sort_values(ascending=False).head(15)
top15_list = [{"feature": k, "importance": float(v)} for k, v in top15.items()]
with open(OUT_JSON_TOP15, "w") as f:
    json.dump({
        "config_used": {
            "penalty": penalty,
            "C": C,
            "l1_ratio": l1_ratio,
            "solver": solver,
            "source": "best_regularization.json" if BEST_CFG.exists() else "default_fallback"
        },
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "top15": top15_list
    }, f, indent=2)

# Plot (horizontal bar)
plt.figure(figsize=(8,10))
importances.plot(kind="barh")
plt.title("Permutation Importance (scoring=ROC-AUC) – Test set")
plt.xlabel("Mean importance (decrease in ROC-AUC)")
plt.tight_layout()
plt.savefig(FIGS / "permutation_importance_roc_auc.png", bbox_inches="tight")
plt.show()

print("\nTop features by permutation importance:")
print(importances.sort_values(ascending=False).head(15))
print(f"\nSaved top-15 JSON to: {OUT_JSON_TOP15}")
print(f"Saved full CSV to: {OUT_CSV_FULL}")
if BEST_CFG.exists():
    print(f"Used best configuration from: {BEST_CFG}")
else:
    print("No best_regularization.json found. Used default elasticnet C=1.0, l1_ratio=0.5.")