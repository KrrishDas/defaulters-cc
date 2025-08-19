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