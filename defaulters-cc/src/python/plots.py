# src/python/plots.py

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix

def plot_roc_train_test(y_train, p_train, y_test, p_test, outpath=None):
    '''Plotting ROC curves for training and testing datasets'''
    fpr_tr, tpr_tr, _ = roc_curve(y_train, p_train) # FPR : False Positive Rate, TPR : True Positive Rate 
    fpr_te, tpr_te, _ = roc_curve(y_test,  p_test)
    auc_tr = auc(fpr_tr, tpr_tr) 
    auc_te = auc(fpr_te, tpr_te)

    plt.figure()
    plt.plot(fpr_tr, tpr_tr, label=f"Train ROC (AUC={auc_tr:.3f})")
    plt.plot(fpr_te, tpr_te, label=f"Test  ROC (AUC={auc_te:.3f})")

    # Adds a diagonal line, represents a random classifier (AUC = 0.5) 
    # Baseline to compare your model against
    plt.plot([0,1],[0,1], linestyle="--")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves (Train vs Test)")
    plt.legend(loc="lower right")
    if outpath:
        plt.savefig(outpath, bbox_inches="tight") # 'tight' trims whitespaces
    plt.show()

def plot_pr_train_test(y_train, p_train, y_test, p_test, outpath=None):
    '''Plotting the Precision-Recall curves for training and testing datasets'''
    pr_tr, rc_tr, _ = precision_recall_curve(y_train, p_train) # PR : Precision values, RC : Recall values
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
    '''Plotting the confusion matrix (TN,FN,TP,FP)'''
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots() # ax (axes) for plotting

    # Displays the confusion matrix as a heatmap image (im) on the axes
    im = ax.imshow(cm, interpolation="nearest") # 'nearest' means each cell is a solid color

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

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04) # visual scale
    if outpath:
        plt.savefig(outpath, bbox_inches="tight")
    plt.show()


def plot_loss_and_auc_curves(curves: dict, outdir=None):
    """
    Plot loss (log loss) and ROC-AUC vs epochs using the `curves` dict returned by train_logreg_with_loss().
    """
    # Safely extract series
    train_loss = curves.get("train_loss", [])
    val_loss   = curves.get("test_loss", [])
    train_auc  = curves.get("train_auc", [])
    val_auc    = curves.get("test_auc", [])

    # X-axis : epoch numbers starting at 1, ending at number of recorded losses.
    xs_loss = range(1, len(train_loss) + 1) 

    # Loss curve
    plt.figure()

    # Plotting both train and test curves on the same plot
    plt.plot(xs_loss, train_loss, label="Train Loss")
    plt.plot(xs_loss, val_loss,   label="Test Loss")

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
    xs_auc = range(1, len(train_auc) + 1) # Same as before, but for the AUC curves.
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