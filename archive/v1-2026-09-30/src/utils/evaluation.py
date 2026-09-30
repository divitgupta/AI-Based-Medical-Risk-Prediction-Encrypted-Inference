"""
Evaluation utilities shared by every model.

Functions:
    evaluate_model()        → dict of metric scores
    print_report()          → formatted console output
    plot_confusion_matrix() → saves a confusion-matrix heatmap
    plot_roc_curve()        → saves a ROC curve
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for saving PNGs
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, auc,
    classification_report,
)

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import PLOTS_DIR


def evaluate_model(y_true, y_pred, y_prob=None):
    """
    Compute all evaluation metrics from the synopsis.

    Parameters
    ----------
    y_true : array-like of int
    y_pred : array-like of int
    y_prob : array-like of float, optional   (probability of positive class)

    Returns
    -------
    metrics : dict
    """
    metrics = {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
    }
    if y_prob is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
    return metrics


def print_report(name, metrics):
    """Pretty-print a metrics dictionary."""
    print(f"\n{'─'*50}")
    print(f"  {name}")
    print(f"{'─'*50}")
    for k, v in metrics.items():
        print(f"  {k:<12s}: {v:.4f}")
    print()


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix", filename=None):
    """Save a confusion-matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Negative", "Positive"],
                yticklabels=["Negative", "Positive"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    plt.tight_layout()
    if filename:
        path = PLOTS_DIR / filename
        fig.savefig(path, dpi=150)
        print(f"  Saved → {path}")
    plt.close(fig)


def plot_roc_curve(y_true, y_prob, title="ROC Curve", filename=None):
    """Save a ROC curve plot."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc_val = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc_val:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    plt.tight_layout()
    if filename:
        path = PLOTS_DIR / filename
        fig.savefig(path, dpi=150)
        print(f"  Saved → {path}")
    plt.close(fig)


def full_evaluation(y_true, y_pred, y_prob, model_name, dataset_name, save_plots=True):
    """
    Run the complete evaluation suite: metrics + confusion matrix + ROC curve.
    Returns the metrics dict.
    """
    metrics = evaluate_model(y_true, y_pred, y_prob)
    print_report(f"{model_name} on {dataset_name}", metrics)

    if save_plots:
        safe_name = f"{dataset_name}_{model_name}".replace(" ", "_").lower()
        plot_confusion_matrix(y_true, y_pred,
                              title=f"{model_name} – {dataset_name}",
                              filename=f"cm_{safe_name}.png")
        plot_roc_curve(y_true, y_prob,
                       title=f"{model_name} – {dataset_name}",
                       filename=f"roc_{safe_name}.png")
    return metrics
