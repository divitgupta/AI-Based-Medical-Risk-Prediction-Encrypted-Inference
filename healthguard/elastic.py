"""
Elastic Learner: an elastic-net stacking meta-learner over the three models,
plus a leaderboard that names the best model per metric and overall.

    python -m healthguard.elastic [--dataset cleveland|base1025_patient|base1025_row]

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

Encrypted use: the server runs the three base models' FHE circuits; the
client decrypts the three probabilities and applies the (tiny, plaintext)
meta-learner itself, so the server still never sees a feature value. The
fitted meta-learner of each fold is saved to results/models/elastic/.
"""

import argparse
import json
import time
from datetime import datetime

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from . import config
from .data import folds, inner_cv, load_dataset
from .evaluate import metrics, roc_points, summarize_folds
from .models import F2_SCORER, fit_params

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


def inner_oof(pipeline, X, y, groups=None):
    """Out-of-fold probabilities for every row of X, from clones of `pipeline`
    refitted on inner stratified folds (hyperparameters kept as tuned). With
    `groups`, copies of a patient never sit on both sides of an inner fold."""
    out = np.zeros(len(y))
    for tr, te in inner_cv(groups).split(X, y, groups):
        g = None if groups is None else groups[tr]
        model = clone(pipeline).fit(X.iloc[tr], y[tr], **fit_params(pipeline, g))
        out[te] = model.predict_proba(X.iloc[te])[:, 1]
    return out


def _meta_learner(groups=None):
    base = LogisticRegression(penalty="elasticnet", solver="saga", class_weight="balanced",
                              max_iter=20000, random_state=config.SEED)
    return GridSearchCV(base, META_GRID, scoring=F2_SCORER, refit=True, cv=inner_cv(groups))


def leaderboard(rows):
    """Best model per LEADERBOARD metric, plus the recommendation: highest F2,
    models within F2_TIE count as tied, the cheapest to encrypt wins.
    rows: {key: {"pooled": metrics, "encrypted_ms": float}}."""
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
    return best, tied, recommended


RULE = (f"Highest pooled F2 (recall-weighted, the project's selection metric). Models within "
        f"{F2_TIE:.2f} of the top F2 count as tied; among them, pick the cheapest to run encrypted.")


def combine(meta, probs):
    """Apply a fitted meta-learner to base-model probabilities.

    probs : array of shape (n, 3), columns in config.MODEL_KEYS order.
    Returns (p(disease), per-model contribution weight × logit)."""
    Z = _logit(probs).reshape(-1, len(config.MODEL_KEYS))
    return meta.predict_proba(Z)[:, 1], Z * meta.coef_[0]


def run(dataset=config.DEFAULT_DATASET):
    t0 = time.perf_counter()
    config.use_dataset(dataset)
    X, y, groups = load_dataset(dataset)
    fold_list = folds(y, groups)
    preds = json.loads((config.RESULTS / "predictions.json").read_text())
    fhe = json.loads((config.RESULTS / "metrics.json").read_text())["fhe"]
    keys = config.MODEL_KEYS

    oof, oof_enc = np.zeros(len(y)), np.zeros(len(y))
    per_fold, fold_info = [], []
    (config.MODELS_DIR / "elastic").mkdir(parents=True, exist_ok=True)
    for k, (tr, te) in enumerate(fold_list):
        X_tr, X_te, y_tr = X.iloc[tr], X.iloc[te], y[tr]
        g_tr = None if groups is None else groups[tr]
        pipes = {m: joblib.load(config.MODELS_DIR / m / f"fold{k}.joblib") for m in keys}

        Z_tr = np.column_stack([_logit(inner_oof(pipes[m], X_tr, y_tr, g_tr)) for m in keys])
        Z_te = np.column_stack([_logit(pipes[m].predict_proba(X_te)[:, 1]) for m in keys])

        search = _meta_learner(g_tr).fit(Z_tr, y_tr, groups=g_tr)
        meta = search.best_estimator_
        oof[te] = meta.predict_proba(Z_te)[:, 1]
        joblib.dump(meta, config.MODELS_DIR / "elastic" / f"fold{k}.joblib")
        # The same meta-learner applied to the base models' *encrypted* outputs.
        enc_probs = np.column_stack([np.array(preds["prob_fhe"][m])[te] for m in keys])
        oof_enc[te] = combine(meta, enc_probs)[0]
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

    best, tied, recommended = leaderboard(rows)
    out = {
        "order": [*keys, "elastic"],
        "rows": rows,
        "elastic": {"pooled": metrics(y, oof), "summary": summarize_folds(per_fold),
                    "folds": fold_info, "roc": roc_points(y, oof), "grid": META_GRID},
        "fhe": {
            "pooled": metrics(y, oof_enc),
            "agreement": float(np.mean((oof_enc >= 0.5) == (oof >= 0.5))),
            "n_disagree": int(np.sum((oof_enc >= 0.5) != (oof >= 0.5))),
            "max_abs_prob_diff": float(np.max(np.abs(oof_enc - oof))),
            "real_exec_ms_per_sample": sum(fhe[m]["real_exec_ms_per_sample"] for m in keys),
            "settings": None,
            "cost_mean": {f: sum(fhe[m]["cost_mean"][f] for m in keys)
                          for f in ("pbs_count", "evaluation_key_bytes")},
        },
        "best": best,
        "recommendation": {
            "model": recommended, "tied_on_f2": tied, "f2_tie_margin": F2_TIE, "rule": RULE,
        },
        "meta": {"generated": datetime.now().isoformat(timespec="seconds"),
                 "runtime_seconds": time.perf_counter() - t0, "seed": config.SEED},
    }
    with open(config.RESULTS / "elastic.json", "w") as f:
        json.dump(out, f, indent=1)
    with open(config.RESULTS / "predictions.json", "w") as f:
        preds["prob_elastic"] = [round(float(v), 6) for v in oof]
        preds["prob_elastic_fhe"] = [round(float(v), 6) for v in oof_enc]
        json.dump(preds, f, indent=1)

    print(f"\n══ Leaderboard · pooled out-of-fold, {len(y)} rows ({dataset}) ══")
    for r in out["order"]:
        p = rows[r]["pooled"]
        print(f"   {rows[r]['name']:20s} F2 {p['f2']:.4f}  recall {p['recall']:.4f}  acc {p['accuracy']:.4f}  "
              f"auc {p['roc_auc']:.4f}  encrypted {rows[r]['encrypted_ms'] / 1000:.2f} s")
    print(f"   Tied on F2: {', '.join(rows[r]['name'] for r in tied)} → recommended: {rows[recommended]['name']}")
    print(f"   Encrypted Elastic Learner: agreement {out['fhe']['agreement'] * 100:.2f}% with plaintext, "
          f"accuracy {out['fhe']['pooled']['accuracy']:.4f}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(config.DATASETS), default=config.DEFAULT_DATASET)
    run(dataset=ap.parse_args().dataset)
