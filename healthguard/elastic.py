"""
Elastic Learner: an elastic-net stacking meta-learner over the three models,
plus a leaderboard that names the best model per metric and overall.

    python -m healthguard.elastic

Stacking: a second-level model learns how much to trust each base model. Its
inputs are the three base models' predicted probabilities (as logits); its
output is the final probability of disease. The meta-learner is a logistic
regression with an elastic-net penalty (L1 + L2): the L1 part can set a base
model's weight to exactly zero, so the learner can drop models that add
nothing, and the L2 part keeps the weights stable when the base models agree.

Leakage-free, per outer fold (the same 5 folds as the main experiment):
  1. the fold's already-tuned base pipelines (results/models/) are refitted on
     inner splits of the *training* rows, giving out-of-fold base predictions
     for every training patient;
  2. the meta-learner (C and l1_ratio tuned by inner CV on F2) is fitted on
     those predictions;
  3. the test patients are scored by the base models trained on the full
     training rows, then combined by the meta-learner.
No test patient influences any step that scores them.
"""

import json
import time
from datetime import datetime

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from . import config
from .data import folds, load_heart
from .evaluate import metrics, roc_points, summarize_folds
from .models import F2_SCORER

META_GRID = {"C": [0.01, 0.1, 1.0, 10.0], "l1_ratio": [0.0, 0.25, 0.5, 0.75, 1.0]}

# The metric each "best model" verdict is decided on, with how to read it.
LEADERBOARD = [
    ("f2", "F2 (selection metric)", "higher"),
    ("recall", "Recall", "higher"),
    ("accuracy", "Accuracy", "higher"),
    ("precision", "Precision", "higher"),
    ("roc_auc", "ROC-AUC", "higher"),
    ("encrypted_ms", "Encrypted time / patient", "lower"),
]
# Two models whose F2 differ by less than this are treated as tied: on ~61
# test patients per fold, one patient moves F2 by more than a point.
F2_TIE = 0.01


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def inner_oof(pipeline, X, y):
    """Out-of-fold probabilities for every row of X, from clones of `pipeline`
    refitted on inner stratified folds (hyperparameters kept as tuned)."""
    out = np.zeros(len(y))
    cv = StratifiedKFold(config.N_INNER_FOLDS, shuffle=True, random_state=config.SEED)
    for tr, te in cv.split(X, y):
        model = clone(pipeline).fit(X.iloc[tr], y[tr])
        out[te] = model.predict_proba(X.iloc[te])[:, 1]
    return out


def _meta_learner():
    base = LogisticRegression(penalty="elasticnet", solver="saga", class_weight="balanced",
                              max_iter=20000, random_state=config.SEED)
    return GridSearchCV(base, META_GRID, scoring=F2_SCORER, refit=True,
                        cv=StratifiedKFold(config.N_INNER_FOLDS, shuffle=True, random_state=config.SEED))


def run():
    t0 = time.perf_counter()
    X, y = load_heart()
    fold_list = folds(y)
    preds = json.loads((config.RESULTS / "predictions.json").read_text())
    fhe = json.loads((config.RESULTS / "metrics.json").read_text())["fhe"]
    keys = config.MODEL_KEYS

    oof = np.zeros(len(y))
    per_fold, fold_info = [], []
    for k, (tr, te) in enumerate(fold_list):
        X_tr, X_te, y_tr = X.iloc[tr], X.iloc[te], y[tr]
        pipes = {m: joblib.load(config.MODELS_DIR / m / f"fold{k}.joblib") for m in keys}

        Z_tr = np.column_stack([_logit(inner_oof(pipes[m], X_tr, y_tr)) for m in keys])
        Z_te = np.column_stack([_logit(pipes[m].predict_proba(X_te)[:, 1]) for m in keys])

        search = _meta_learner().fit(Z_tr, y_tr)
        meta = search.best_estimator_
        oof[te] = meta.predict_proba(Z_te)[:, 1]
        per_fold.append(metrics(y[te], oof[te]))
        weights = {m: float(w) for m, w in zip(keys, meta.coef_[0])}
        fold_info.append({"fold": k, "C": search.best_params_["C"],
                          "l1_ratio": search.best_params_["l1_ratio"],
                          "weights": weights, "intercept": float(meta.intercept_[0]),
                          "dropped": [m for m, w in weights.items() if abs(w) < 1e-6]})
        print(f"   fold {k + 1}: acc {per_fold[-1]['accuracy']:.3f}  recall {per_fold[-1]['recall']:.3f}  "
              f"C={search.best_params_['C']} l1_ratio={search.best_params_['l1_ratio']}  "
              + "  ".join(f"{config.MODEL_NAMES[m]} {w:+.2f}" for m, w in weights.items()))

    # ── Leaderboard: base models + Elastic Learner, all pooled out-of-fold ──
    rows = {}
    for m in keys:
        prob = np.array(preds["prob"][m])
        rows[m] = {"name": config.MODEL_NAMES[m], "pooled": metrics(y, prob),
                   "encrypted_ms": fhe[m]["real_exec_ms_per_sample"]}
    rows["elastic"] = {"name": "Elastic Learner", "pooled": metrics(y, oof),
                       # Encrypting the stack means running all three circuits.
                       "encrypted_ms": sum(fhe[m]["real_exec_ms_per_sample"] for m in keys)}

    def value(key, metric):
        return rows[key]["encrypted_ms"] if metric == "encrypted_ms" else rows[key]["pooled"][metric]

    best = {}
    for metric, label, direction in LEADERBOARD:
        pick = (min if direction == "lower" else max)(rows, key=lambda r: value(r, metric))
        best[metric] = {"label": label, "direction": direction, "model": pick,
                        "value": value(pick, metric)}

    top_f2 = max(value(r, "f2") for r in rows)
    tied = [r for r in rows if top_f2 - value(r, "f2") < F2_TIE]
    recommended = min(tied, key=lambda r: rows[r]["encrypted_ms"])
    out = {
        "order": [*keys, "elastic"],
        "rows": rows,
        "elastic": {"pooled": metrics(y, oof), "summary": summarize_folds(per_fold),
                    "folds": fold_info, "roc": roc_points(y, oof), "grid": META_GRID},
        "best": best,
        "recommendation": {
            "model": recommended, "tied_on_f2": tied, "f2_tie_margin": F2_TIE,
            "rule": (f"Highest pooled F2 (recall-weighted, the project's selection metric). Models within "
                     f"{F2_TIE:.2f} of the top F2 count as tied; among them, pick the cheapest to run encrypted."),
        },
        "meta": {"generated": datetime.now().isoformat(timespec="seconds"),
                 "runtime_seconds": time.perf_counter() - t0, "seed": config.SEED},
    }
    with open(config.RESULTS / "elastic.json", "w") as f:
        json.dump(out, f, indent=1)
    with open(config.RESULTS / "predictions.json", "w") as f:
        preds["prob_elastic"] = [round(float(v), 6) for v in oof]
        json.dump(preds, f, indent=1)

    print("\n══ Leaderboard · pooled out-of-fold, 303 patients ══")
    for r in out["order"]:
        p = rows[r]["pooled"]
        print(f"   {rows[r]['name']:20s} F2 {p['f2']:.4f}  recall {p['recall']:.4f}  acc {p['accuracy']:.4f}  "
              f"auc {p['roc_auc']:.4f}  encrypted {rows[r]['encrypted_ms'] / 1000:.2f} s")
    print(f"   Tied on F2: {', '.join(rows[r]['name'] for r in tied)} → recommended: {rows[recommended]['name']}")
    return out


if __name__ == "__main__":
    run()
