# src/python/DefaultersLogisticRegression.py

from .data_prep import load_combined_csv, make_default_flag, split_X_y
from .config import TARGET_COL, REPORTS_DIR
from .train import train_logreg
from .plots import plot_roc_train_test, plot_pr_train_test, plot_confusion_matrix_basic

def main():
    # 1) Load data
    df = load_combined_csv()

    # 2) Make binary flag
    df = make_default_flag(df, target=TARGET_COL)

    # 3) Split features/label
    X, y = split_X_y(df, target=TARGET_COL)

    # 4) Train & save artifacts
    pipe, metrics, ev = train_logreg(X, y)

    # 5) Quick console summary
    print("Training complete.")
    print(f"ROC-AUC: {metrics['roc_auc']:.3f}")
    print(f"Confusion Matrix: {metrics['confusion_matrix']}")

    # Plot & save figures
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_roc_train_test(
        ev["y_train"], ev["y_proba_train"],
        ev["y_test"],  ev["y_proba_test"],
        outpath=REPORTS_DIR / "roc_train_test.png"
    )
    plot_pr_train_test(
        ev["y_train"], ev["y_proba_train"],
        ev["y_test"],  ev["y_proba_test"],
        outpath=REPORTS_DIR / "pr_train_test.png"
    )
    plot_confusion_matrix_basic(
        ev["y_test"], ev["y_pred"],
        class_names=("Non-Default","Default"),
        outpath=REPORTS_DIR / "confusion_matrix.png"
    )


if __name__ == "__main__":
    main()