# src/python/experiments_regularization.py
from pathlib import Path
import json
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    average_precision_score, precision_recall_curve
)

# --- project paths ---
ROOT = Path(__file__).resolve().parents[2]        # .../defaulters-cc/
DATA = ROOT / "data" / "processed" / "combined_data_quarterly.csv"
FIGS = ROOT / "reports" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)
OUT_JSON = ROOT / "reports" / "regularization_summary.json"

# --- load & prepare data ---
df = pd.read_csv(DATA, parse_dates=["observation_date"])
# create label as you did before (median threshold on DRCCLACBS)
threshold = df["DRCCLACBS"].median()
df["DefaultFlag"] = (df["DRCCLACBS"] > threshold).astype(int)

X = df.drop(columns=["observation_date", "DRCCLACBS", "DefaultFlag"])
y = df["DefaultFlag"]

# consistent split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

feature_names = X.columns.tolist()

# --- model grid ---
penalties = {
    "l2":   dict(penalty="l2",        l1_ratio=None),
    "l1":   dict(penalty="l1",        l1_ratio=None),
    "enet": dict(penalty="elasticnet", l1_ratio=[0.2, 0.5, 0.8]),
}
C_grid = [0.01, 0.1, 1.0, 10.0]

results = []

def build_pipe(penalty, C, l1_ratio=None):
    clf = LogisticRegression(
        solver="saga",        # supports l1 & elasticnet
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
    return pipe

def plot_roc_pr(y_true, proba, label, prefix):
    # ROC
    fpr, tpr, _ = roc_curve(y_true, proba)
    auc = roc_auc_score(y_true, proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f"{label} (AUC={auc:.3f})")
    plt.plot([0,1],[0,1],"--",color="gray")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC – {label}")
    plt.legend()
    plt.savefig(FIGS / f"{prefix}_roc.png", bbox_inches="tight")
    plt.close()

    # PR
    pr, rc, _ = precision_recall_curve(y_true, proba)
    ap = average_precision_score(y_true, proba)
    plt.figure()
    plt.plot(rc, pr, label=f"{label} (AP={ap:.3f})")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR – {label}")
    plt.legend()
    plt.savefig(FIGS / f"{prefix}_pr.png", bbox_inches="tight")
    plt.close()

def plot_coefficients(coefs, names, title, prefix, top_k=15):
    s = pd.Series(coefs, index=names).sort_values()
    # show top negative & positive
    s_plot = pd.concat([s.head(top_k), s.tail(top_k)])
    plt.figure(figsize=(8,6))
    s_plot.plot(kind="barh")
    plt.title(title)
    plt.xlabel("Coefficient")
    plt.tight_layout()
    plt.savefig(FIGS / f"{prefix}_coefs.png", bbox_inches="tight")
    plt.close()

# --- run the experiments ---
for key, spec in penalties.items():
    if key == "enet":
        for C, l1r in itertools.product(C_grid, spec["l1_ratio"]):
            pipe = build_pipe("elasticnet", C, l1_ratio=l1r)
            pipe.fit(X_train, y_train)
            proba = pipe.predict_proba(X_test)[:,1]

            auc = roc_auc_score(y_test, proba)
            ap  = average_precision_score(y_test, proba)

            # plots
            label = f"ElasticNet (C={C}, l1={l1r})"
            prefix = f"enet_C{C}_l1{l1r}".replace(".","p")
            plot_roc_pr(y_test, proba, label, prefix)

            # coefficients (after scaling)
            clf = pipe.named_steps["clf"]
            coefs = clf.coef_.ravel()
            plot_coefficients(coefs, feature_names, label, prefix)

            results.append(dict(model="elasticnet", C=C, l1_ratio=l1r,
                                roc_auc=auc, avg_precision=ap))
    else:
        for C in C_grid:
            pipe = build_pipe(spec["penalty"], C, l1_ratio=None)
            pipe.fit(X_train, y_train)
            proba = pipe.predict_proba(X_test)[:,1]

            auc = roc_auc_score(y_test, proba)
            ap  = average_precision_score(y_test, proba)

            label = f"{key.upper()} (C={C})"
            prefix = f"{key}_C{C}".replace(".","p")
            plot_roc_pr(y_test, proba, label, prefix)

            clf = pipe.named_steps["clf"]
            coefs = clf.coef_.ravel()
            plot_coefficients(coefs, feature_names, label, prefix)

            results.append(dict(model=key, C=C, l1_ratio=None,
                                roc_auc=auc, avg_precision=ap))

# save summary table
with open(OUT_JSON, "w") as f:
    json.dump({
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "results": results
    }, f, indent=2)

print(f"Done. Wrote figures to {FIGS} and summary to {OUT_JSON}")