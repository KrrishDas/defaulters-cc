# src/python/train.py

from pathlib import Path
import json
import pandas as pd
from joblib import dump

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from .config import (
    MODELS_DIR, REPORTS_DIR,
    TEST_SIZE, RANDOM_STATE,
    MAX_ITER, PENALTY, SOLVER,
)

def train_logreg(X: pd.DataFrame, y: pd.Series):
    """
    Impute → scale → logistic regression (pipeline).
    Saves model, scaler, coefficients, and metrics.
    Returns fitted pipeline and a dict of metrics.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # split first (so imputer/scaler learn only from training data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Build pipeline
    pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(
            max_iter=MAX_ITER, penalty=PENALTY, solver=SOLVER
        )),
    ])

    # Fit
    pipe.fit(X_train, y_train)

    # Evaluate
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    # Also get TRAIN probabilities for a train ROC curve
    y_proba_train = pipe.predict_proba(X_train)[:, 1]

    # Pack a small eval payload for plotting
    eval_payload = {
        "X_train": X_train,          # shapes only used for info
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_proba_test": y_proba,
        "y_proba_train": y_proba_train,
    }

    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, output_dict=True)
    roc_auc = float(roc_auc_score(y_test, y_proba))

    metrics = {
        "confusion_matrix": cm,
        "classification_report": report,
        "roc_auc": roc_auc,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

    # Save metrics
    with open(REPORTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save pipeline (imputer+scaler+model) together
    dump(pipe, MODELS_DIR / "logreg_pipeline.joblib")

    # Save coefficients with feature names
    # (pipeline step name is 'logreg'; access coef_ on underlying estimator)
    logreg = pipe.named_steps["logreg"]
    coefs = pd.DataFrame({
        "Feature": X.columns,
        "Coefficient": logreg.coef_[0]
    }).sort_values(by="Coefficient", ascending=False)
    coefs.to_csv(MODELS_DIR / "logreg_coefficients.csv", index=False)

    return pipe, metrics, eval_payload