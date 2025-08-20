# src/python/DefaultersLogisticRegression.py

from .data_prep import load_combined_csv, make_default_flag, split_X_y
from .config import TARGET_COL, REPORTS_DIR, FIGURES_DIR
from .train import train_logreg, train_logreg_with_loss
from .plots import (
    plot_roc_train_test,
    plot_pr_train_test,
    plot_confusion_matrix_basic,
    plot_loss_and_auc_curves,
)

USE_LOSS_CURVE = False  # set True when you want curves

def main():
    # 1) Load data
    df = load_combined_csv()

    # 2) Make binary flag
    df = make_default_flag(df, target=TARGET_COL)

    # 3) Split features/label
    X, y = split_X_y(df, target=TARGET_COL)

    # 4) Train & save artifacts
    if USE_LOSS_CURVE:
        pipe, metrics, curves = train_logreg_with_loss(X, y, epochs=200, C=1.0, class_weight="balanced")
        # Plot loss/auc curves
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        plot_loss_and_auc_curves(curves, outdir=FIGURES_DIR)

    else:
        pipe, metrics, ev = train_logreg(X, y)
        # Plot ROC/PR/CM using ev payload
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        plot_roc_train_test(
            ev["y_train"], ev["y_proba_train"],
            ev["y_test"],  ev["y_proba_test"],
            outpath=FIGURES_DIR / "roc_train_test.png"
        )
        plot_pr_train_test(
            ev["y_train"], ev["y_proba_train"],
            ev["y_test"],  ev["y_proba_test"],
            outpath=FIGURES_DIR / "pr_train_test.png"
        )
        plot_confusion_matrix_basic(
            ev["y_test"], ev["y_pred"],
            class_names=("Non-Default","Default"),
            outpath=FIGURES_DIR / "confusion_matrix.png"
        )


if __name__ == "__main__":
    main()