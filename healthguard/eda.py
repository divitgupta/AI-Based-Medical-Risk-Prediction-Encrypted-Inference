"""
Exploratory data analysis (synopsis §4). Produces plain JSON that the web app
draws as SVG charts, so every number on the Dataset page is computed here.
"""

import numpy as np
import pandas as pd

from . import config
from .data import load_breast, load_heart


def _histogram(values, labels, bins=12):
    values = np.asarray(values, dtype=float)
    ok = ~np.isnan(values)
    edges = np.histogram_bin_edges(values[ok], bins=bins)
    return {
        "type": "continuous",
        "edges": [round(float(e), 3) for e in edges],
        "by_class": {
            str(c): np.histogram(values[ok & (labels == c)], bins=edges)[0].tolist()
            for c in (0, 1)
        },
    }


def _category_counts(values, labels):
    values = np.asarray(values, dtype=float)
    cats = sorted(int(v) for v in np.unique(values[~np.isnan(values)]))
    return {
        "type": "categorical",
        "categories": cats,
        "by_class": {
            str(c): [int(np.sum((values == v) & (labels == c))) for v in cats]
            for c in (0, 1)
        },
        "missing": int(np.isnan(values).sum()),
    }


def heart_eda():
    X, y = load_heart()
    raw = pd.read_csv(config.HEART_CSV)

    features = {}
    for col in X.columns:
        if col in config.HD_CONTINUOUS:
            features[col] = _histogram(X[col], y)
        else:
            features[col] = _category_counts(X[col], y)
        features[col]["group"] = (
            "continuous" if col in config.HD_CONTINUOUS
            else "binary" if col in config.HD_BINARY else "nominal"
        )
        features[col].update(config.HD_FEATURE_INFO[col])

    # Correlations on the cleaned features plus the corrected disease label.
    corr_df = X.assign(disease=y).corr(numeric_only=True).round(3)

    # Evidence for the label correction, computed from the untouched CSV.
    evidence = []
    for col, expect in [("exang", "higher"), ("oldpeak", "higher"), ("ca", "higher"),
                        ("thalach", "lower"), ("age", "higher"), ("sex", "higher")]:
        m = raw.groupby("target")[col].mean()
        evidence.append({"feature": col, "label": config.HD_FEATURE_INFO[col]["label"],
                         "csv_target_0": round(float(m[0]), 2),
                         "csv_target_1": round(float(m[1]), 2),
                         "expected_in_disease": expect})

    return {
        "name": "Heart Disease (Cleveland)",
        "role": "primary",
        "source": "UCI Cleveland via Kaggle (ronitf/heart-disease-uci)",
        "n_samples": int(len(y)),
        "n_attributes": int(raw.shape[1]),
        "n_features": int(X.shape[1]),
        "class_names": {"0": "No Disease", "1": "Disease"},
        "class_counts": {"0": int((y == 0).sum()), "1": int((y == 1).sum())},
        "groups": {"continuous": config.HD_CONTINUOUS, "binary": config.HD_BINARY,
                   "nominal": config.HD_NOMINAL},
        "invalid_codes": {col: {"codes": bad, "rows": int(raw[col].isin(bad).sum())}
                          for col, bad in config.HD_INVALID_CODES.items()},
        "label_evidence": evidence,
        "features": features,
        "correlation": {"labels": list(corr_df.columns),
                        "matrix": corr_df.values.tolist()},
    }


def breast_eda():
    X, y = load_breast()
    return {
        "name": "Breast Cancer Wisconsin (Diagnostic)",
        "role": "secondary",
        "source": "UCI via Kaggle (uciml/breast-cancer-wisconsin-data)",
        "n_samples": int(len(y)),
        "n_features": int(X.shape[1]),
        "class_names": {"0": "Benign", "1": "Malignant"},
        "class_counts": {"0": int((y == 0).sum()), "1": int((y == 1).sum())},
        "missing_values": int(X.isna().sum().sum()),
        "status": "Dataset available · full experiment in Phase 2",
    }


def build():
    return {"heart": heart_eda(), "breast": breast_eda()}
