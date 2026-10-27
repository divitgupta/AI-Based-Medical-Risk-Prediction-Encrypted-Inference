"""
Metrics for diagnostic classification (synopsis §7). Positive class = 1 =
Disease, so recall is sensitivity to disease.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, fbeta_score, precision_score,
    recall_score, roc_auc_score, roc_curve,
)

METRIC_KEYS = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "f2": float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def roc_points(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return {"fpr": [round(float(v), 5) for v in fpr],
            "tpr": [round(float(v), 5) for v in tpr]}


def summarize_folds(per_fold):
    """Mean and standard deviation of each metric across folds."""
    out = {}
    for k in METRIC_KEYS:
        vals = np.array([m[k] for m in per_fold])
        out[k] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=0))}
    return out
