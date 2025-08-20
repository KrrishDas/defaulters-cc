# src/python/plots.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix

def plot_roc_train_test(y_train, p_train, y_test, p_test, outpath=None):
    fpr_tr, tpr_tr, _ = roc_curve(y_train, p_train)
    fpr_te, tpr_te, _ = roc_curve(y_test,  p_test)
    auc_tr = auc(fpr_tr, tpr_tr)
    auc_te = auc(fpr_te, tpr_te)

    plt.figure()
    plt.plot(fpr_tr, tpr_tr, label=f"Train ROC (AUC={auc_tr:.3f})")
    plt.plot(fpr_te, tpr_te, label=f"Test  ROC (AUC={auc_te:.3f})")
    plt.plot([0,1],[0,1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves (Train vs Test)")
    plt.legend(loc="lower right")
    if outpath:
        plt.savefig(outpath, bbox_inches="tight")
    plt.show()

def plot_pr_train_test(y_train, p_train, y_test, p_test, outpath=None):
    pr_tr, rc_tr, _ = precision_recall_curve(y_train, p_train)
    pr_te, rc_te, _ = precision_recall_curve(y_test,  p_test)

    plt.figure()
    plt.plot(rc_tr, pr_tr, label="Train PR")
    plt.plot(rc_te, pr_te, label="Test PR")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curves (Train vs Test)")
    plt.legend(loc="lower left")
    if outpath:
        plt.savefig(outpath, bbox_inches="tight")
    plt.show()

def plot_confusion_matrix_basic(y_true, y_pred, class_names=("Non-Default","Default"), outpath=None):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title("Confusion Matrix (Test)")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(class_names); ax.set_yticklabels(class_names)

    # annotate cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if outpath:
        plt.savefig(outpath, bbox_inches="tight")
    plt.show()

# Plot loss and ROC-AUC vs epochs for logistic regression (warm start)
def plot_loss_and_auc_curves(curves: dict, outdir=None):
    """
    Plot loss (log loss) and ROC-AUC vs epochs using the `curves` dict returned by
    train_logreg_with_loss(). Expects keys:
      - 'train_loss', 'val_loss', 'train_auc', 'val_auc'
    If `outdir` is provided (Path or str), saves two PNGs into that directory.
    """
    # Safely extract series
    train_loss = curves.get("train_loss", [])
    val_loss   = curves.get("val_loss", [])
    train_auc  = curves.get("train_auc", [])
    val_auc    = curves.get("val_auc", [])

    # X-axis (epochs)
    xs_loss = range(1, len(train_loss) + 1)

    # Loss curve
    plt.figure()
    plt.plot(xs_loss, train_loss, label="Train Loss")
    plt.plot(xs_loss, val_loss,   label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Log Loss")
    plt.title("LogisticRegression (warm-start): Loss Curve")
    plt.legend(loc="best")
    if outdir is not None:
        from pathlib import Path
        Path(outdir).mkdir(parents=True, exist_ok=True)
        plt.savefig(Path(outdir) / "loss_curve_logreg.png", bbox_inches="tight")
    plt.show()

    # AUC curve
    xs_auc = range(1, len(train_auc) + 1)
    plt.figure()
    plt.plot(xs_auc, train_auc, label="Train ROC-AUC")
    plt.plot(xs_auc, val_auc,   label="Validation ROC-AUC")
    plt.xlabel("Epoch")
    plt.ylabel("ROC-AUC")
    plt.title("LogisticRegression (warm-start): ROC-AUC vs Epochs")
    plt.legend(loc="best")
    if outdir is not None:
        from pathlib import Path
        Path(outdir).mkdir(parents=True, exist_ok=True)
        plt.savefig(Path(outdir) / "auc_curve_logreg.png", bbox_inches="tight")
    plt.show()