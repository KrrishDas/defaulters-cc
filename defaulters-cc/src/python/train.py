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
    log_loss
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
    y_proba = pipe.predict_proba(X_test)[:, 1] #probability of it being 1(positive)

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


def train_logreg_with_loss(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    epochs: int = 200,
    C: float = 1.0,
    class_weight=None,
):
    """
    Same model family (sklearn LogisticRegression) but trained in 'warm-start' steps
    so we can record loss/AUC across epochs. Preprocessing (impute+scale) is learned
    on the training split only to avoid leakage.

    Returns:
        pipe_final: fitted Pipeline (imputer+scaler+logreg) at the final epoch
        metrics: standard test set metrics (same schema as train_logreg)
        curves: dict with per-epoch lists: train_loss, val_loss, train_auc, val_auc
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # split first
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Preprocess (fit on train only)
    pre = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
    ])
    X_train_t = pre.fit_transform(X_train)
    X_test_t   = pre.transform(X_test)

    # Warm-start LogisticRegression: take 1 optimizer step per epoch
    # Allows training over gradual epochs, step-by-step, almost behaving like a neural network
    logreg = LogisticRegression(
        penalty=PENALTY,
        solver=SOLVER,
        C=C,
        class_weight=class_weight,
        max_iter=1,
        warm_start=True,
        random_state=RANDOM_STATE,
    )

    # first call initializes internal state
    logreg.fit(X_train_t, y_train)

    train_losses, test_losses = [], []
    train_aucs,   test_aucs   = [], []

    for _ in range(epochs):
        logreg.fit(X_train_t, y_train)
        p_tr = logreg.predict_proba(X_train_t)[:, 1]
        p_va = logreg.predict_proba(X_test_t)[:, 1]
        train_losses.append(log_loss(y_train, p_tr, labels=[0, 1])) # tells me how wrong the probability is 
        test_losses.append(log_loss(y_test,   p_va, labels=[0, 1]))
        train_aucs.append(roc_auc_score(y_train, p_tr))
        test_aucs.append(roc_auc_score(y_test,   p_va))

    # Build a final pipeline object for consistent downstream use
    pipe_final = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(
            penalty=PENALTY,
            solver=SOLVER,
            C=C,
            class_weight=class_weight,
            max_iter=1,
            warm_start=True,
            random_state=RANDOM_STATE,
        )),
    ])
    # Fit the preprocessing, then set the learned logreg into the pipeline
    pipe_final.named_steps["imputer"].fit(X_train)
    X_train_imp = pipe_final.named_steps["imputer"].transform(X_train)
    pipe_final.named_steps["scaler"].fit(X_train_imp)
    # Drop the freshly-initialized estimator and insert our trained one
    pipe_final.named_steps["logreg"] = logreg

    # Standard test-set evaluation to mirror train_logreg
    y_pred = logreg.predict(X_test_t)
    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, output_dict=True)
    roc_auc = float(roc_auc_score(y_test, logreg.predict_proba(X_test_t)[:, 1]))

    metrics = {
        "confusion_matrix": cm,
        "classification_report": report,
        "roc_auc": roc_auc,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

    curves = {
        "train_loss": train_losses,
        "test_loss":   test_losses,
        "train_auc":  train_aucs,
        "test_auc":    test_aucs,
    }

    # Optionally persist curves for plotting elsewhere
    with open(REPORTS_DIR / "loss_curves.json", "w") as f:
        json.dump(curves, f, indent=2)

    return pipe_final, metrics, curves