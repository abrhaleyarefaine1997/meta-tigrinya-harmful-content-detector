import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    precision_recall_curve,
    auc
)


class ModelEvaluator:
    def __init__(self, save_dir="reports"):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def plot_confusion_matrix(self, y_true, y_pred, name="confusion_matrix"):
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)

        fig, ax = plt.subplots(figsize=(5, 5))
        disp.plot(ax=ax, values_format="d", colorbar=False)
        ax.set_title("Confusion Matrix")

        path = os.path.join(self.save_dir, f"{name}.png")
        plt.savefig(path, bbox_inches="tight", dpi=300)
        plt.close(fig)

        return path

    def plot_roc_curve(self, y_true, y_proba, name="roc_curve"):
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)

        fig = plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
        plt.plot([0, 1], [0, 1], linestyle="--")

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()

        path = os.path.join(self.save_dir, f"{name}.png")
        plt.savefig(path, bbox_inches="tight", dpi=300)
        plt.close(fig)

        return path, roc_auc

    def plot_pr_curve(self, y_true, y_proba, name="pr_curve"):
        precision, recall, _ = precision_recall_curve(y_true, y_proba)

        fig = plt.figure(figsize=(6, 5))
        plt.plot(recall, precision)

        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")

        path = os.path.join(self.save_dir, f"{name}.png")
        plt.savefig(path, bbox_inches="tight", dpi=300)
        plt.close(fig)

        return path