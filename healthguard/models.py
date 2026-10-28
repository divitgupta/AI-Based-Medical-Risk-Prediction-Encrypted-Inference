"""
Preprocessing, the three model pipelines, and how each one's complexity is
measured.

Every model is a Pipeline(preprocessor, model). Passing the *whole* pipeline to
cross-validation means the scaler, imputer and one-hot encoder are refitted on
each training fold only, so no statistic ever comes from held-out rows.
"""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from . import config
from .mlp import MLPClassifier

# F2 weights recall twice as heavily as precision. It is our Phase-1
# model-selection metric, chosen to match the synopsis' emphasis on
# sensitivity (a missed disease case is the costlier error).
F2_SCORER = make_scorer(fbeta_score, beta=2)


def build_preprocessor():
    """Continuous → scale; binary → as-is; nominal → impute + one-hot."""
    nominal = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", StandardScaler(), config.HD_CONTINUOUS),
        ("bin", "passthrough", config.HD_BINARY),
        ("nom", nominal, config.HD_NOMINAL),
    ])


def build_logreg(y_train=None):
    model = LogisticRegression(class_weight="balanced", max_iter=5000,
                               random_state=config.SEED)
    return Pipeline([("prep", build_preprocessor()), ("model", model)]), config.LOGREG_GRID


def build_xgb(y_train):
    # Class weighting for trees: up-weight positives by the training fold's
    # negative/positive ratio (computed on training rows only).
    n_pos = int(np.sum(y_train == 1))
    spw = (len(y_train) - n_pos) / max(n_pos, 1)
    model = XGBClassifier(
        scale_pos_weight=spw, eval_metric="logloss",
        random_state=config.SEED, n_jobs=1, verbosity=0,
    )
    return Pipeline([("prep", build_preprocessor()), ("model", model)]), config.XGB_GRID


def build_mlp(y_train=None):
    model = MLPClassifier(random_state=config.SEED, **config.MLP_CONFIG)
    return Pipeline([("prep", build_preprocessor()), ("model", model)]), None


BUILDERS = {"logreg": build_logreg, "xgboost": build_xgb, "mlp": build_mlp}


def fit_params(pipeline, groups):
    """Extra fit() arguments for a pipeline: the MLP takes patient ids so its
    early-stopping split never separates copies of one patient."""
    if groups is not None and isinstance(pipeline.named_steps["model"], MLPClassifier):
        return {"model__groups": groups}
    return {}


def complexity(key, pipeline):
    """
    Model-appropriate complexity measures. A single "parameter count" means
    different things for a linear model, a tree ensemble and a network, so
    trees are described by their structure instead.
    """
    model = pipeline.named_steps["model"]
    n_features = int(len(pipeline.named_steps["prep"].get_feature_names_out()))

    if key == "logreg":
        return {
            "kind": "parameters",
            "n_params": int(model.coef_.size + model.intercept_.size),
            "n_features": n_features,
            "summary": f"{n_features} weights + 1 bias",
        }

    if key == "xgboost":
        trees = model.get_booster().trees_to_dataframe()
        n_trees = int(trees["Tree"].nunique())
        n_leaves = int((trees["Feature"] == "Leaf").sum())
        return {
            "kind": "trees",
            "n_trees": n_trees,
            "max_depth": int(model.get_params()["max_depth"]),
            "total_nodes": int(len(trees)),
            "n_leaves": n_leaves,
            "n_features": n_features,
            "summary": f"{n_trees} trees, max depth {model.get_params()['max_depth']}",
        }

    if key == "mlp":
        sizes = model.layer_sizes_
        return {
            "kind": "parameters",
            "n_params": model.n_params_,
            "layers": sizes,
            "n_features": n_features,
            "summary": " → ".join(str(s) for s in sizes),
        }

    raise ValueError(key)
