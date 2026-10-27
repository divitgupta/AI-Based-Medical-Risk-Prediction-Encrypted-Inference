"""
Exploratory data analysis (synopsis §4). Produces plain JSON that the web app
draws as SVG charts, so every number on the Dataset page is computed here.
"""

import numpy as np
import pandas as pd

from . import config
from .data import load_base_paper, load_heart


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


def base_paper_eda():
    """How the base paper's 1,025-row file relates to our 303-row file."""
    X, y, group = load_base_paper()
    ours = pd.read_csv(config.HEART_CSV).drop_duplicates()
    theirs = pd.read_csv(config.BASE_PAPER_CSV)
    distinct = theirs.drop_duplicates()
    shared = distinct.merge(ours, how="inner", on=list(ours.columns))
    copies = np.bincount(np.bincount(group))
    return {
        "name": "Heart Disease (base-paper copy)",
        "source": "Kaggle johnsmith88/heart-disease-dataset, as used by Naresh & Reddi (2025)",
        "n_rows": int(len(theirs)),
        "n_distinct": int(len(distinct)),
        "n_duplicate_rows": int(len(theirs) - len(distinct)),
        "n_shared_with_ours": int(len(shared)),
        "n_ours_distinct": int(len(ours)),
        "copies_per_patient": {str(k): int(v) for k, v in enumerate(copies) if k and v},
        "class_counts_rows": {"0": int((y == 0).sum()), "1": int((y == 1).sum())},
        "class_counts_distinct": {
            "0": int((1 - distinct["target"] == 0).sum()),
            "1": int((1 - distinct["target"] == 1).sum()),
        },
        "class_names": {"0": "No Disease", "1": "Disease"},
    }


def build():
    return {"heart": heart_eda(), "base_paper": base_paper_eda()}
