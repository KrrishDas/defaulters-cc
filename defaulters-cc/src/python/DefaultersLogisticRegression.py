# src/python/DefaultersLogisticRegression.py

from .data_prep import load_combined_csv, make_default_flag, split_X_y
from .config import TARGET_COL
from .train import train_logreg

def main():
    # 1) Load data
    df = load_combined_csv()

    # 2) Make binary flag
    df = make_default_flag(df, target=TARGET_COL)

    # 3) Split features/label
    X, y = split_X_y(df, target=TARGET_COL)

    # 4) Train & save artifacts
    pipe, metrics = train_logreg(X, y)

    # 5) Quick console summary
    print("Training complete.")
    print(f"ROC-AUC: {metrics['roc_auc']:.3f}")
    print(f"Confusion Matrix: {metrics['confusion_matrix']}")

if __name__ == "__main__":
    main()