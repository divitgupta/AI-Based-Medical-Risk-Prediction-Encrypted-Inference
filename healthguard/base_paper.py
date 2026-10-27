"""
Phase 2: our pipeline on the base paper's own data.

    python -m healthguard.base_paper [--quick]

Naresh & Reddi (2025) report 85% accuracy for plaintext logistic regression
and 84% for their CKKS-encrypted version, on the Kaggle file
johnsmith88/heart-disease-dataset. That file has 1,025 rows but only 302
distinct patient records (see data.load_base_paper), so a split made over
*rows* can put copies of the same patient in both training and test sets.

We run the same three models and the encrypted logistic regression twice:

    row split      stratified 5-fold over the 1,025 rows (copies leak across
                   train and test, as in a plain random split);
    patient split  stratified *grouped* 5-fold, every copy of a patient kept
                   on the same side (no leakage).

The gap between the two is how much duplicate leakage inflates the scores.
"""

import argparse
import json
import time
from datetime import datetime

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from . import config, fhe
from .data import load_base_paper
from .evaluate import metrics
from .experiment import _tune_and_fit

# What the base paper reports for its heart dataset (Naresh & Reddi 2025,
# Journal of Big Data 12:52, results section). Accuracy is the only metric
# it gives as numbers.
BASE_PAPER_REPORTED = {
    "citation": "Naresh & Reddi (2025), Journal of Big Data 12:52",
    "scheme": "CKKS (TenSEAL)",
    "lr_plaintext_accuracy": 0.85,
    "lr_encrypted_accuracy": {"poly_modulus_4096": 0.84, "poly_modulus_8192": 0.84},
    "svm_encrypted_score": {"poly_modulus_4096": 0.67, "poly_modulus_8192": 0.79},
    "lr_encrypted_score_vs_svm": {"poly_modulus_4096": 0.817, "poly_modulus_8192": 0.85},
    "split": "not described in the paper",
}


def _splits(kind, y, group):
    if kind == "row":
        cv = StratifiedKFold(config.N_FOLDS, shuffle=True, random_state=config.SEED)
        return list(cv.split(np.zeros(len(y)), y))
    cv = StratifiedGroupKFold(config.N_FOLDS, shuffle=True, random_state=config.SEED)
    return list(cv.split(np.zeros(len(y)), y, group))


def _run_split(kind, X, y, group, quick):
    fold_list = _splits(kind, y, group)
    oof = {k: np.zeros(len(y)) for k in config.MODEL_KEYS}
    oof_fhe = np.zeros(len(y))
    leaked = 0
    for k, (tr, te) in enumerate(fold_list):
        leaked += int(np.isin(group[te], group[tr]).sum())
        X_tr, X_te, y_tr = X.iloc[tr], X.iloc[te], y[tr]
        for key in config.MODEL_KEYS:
            best, *_ = _tune_and_fit(key, X_tr, y_tr, quick)
            oof[key][te] = best.predict_proba(X_te)[:, 1]
            if key == "logreg":
                model, _ = fhe.compile_model("logreg", best, X_tr)
                X_te_enc = best.named_steps["prep"].transform(X_te)
                oof_fhe[te] = model.predict_proba(X_te_enc, fhe="simulate")[:, 1]
        print(f"   {kind:7s} fold {k + 1}: "
              + "  ".join(f"{key} {metrics(y[te], oof[key][te])['accuracy']:.3f}"
                          for key in config.MODEL_KEYS))
    return {
        "test_rows_with_copy_in_training": leaked,
        "test_rows_with_copy_in_training_pct": leaked / len(y),
        "models": {k: metrics(y, oof[k]) for k in config.MODEL_KEYS},
        "fhe_logreg": {
            "pooled": metrics(y, oof_fhe),
            "agreement": float(np.mean((oof_fhe >= 0.5) == (oof["logreg"] >= 0.5))),
        },
    }


def run(quick=False):
    t0 = time.perf_counter()
    X, y, group = load_base_paper()
    print(f"Base-paper file: {len(y)} rows, {len(np.unique(group))} distinct patients")
    out = {
        "dataset": {"rows": int(len(y)), "distinct_patients": int(len(np.unique(group))),
                    "positive_rows": int(y.sum())},
        "base_paper_reported": BASE_PAPER_REPORTED,
        "row_split": _run_split("row", X, y, group, quick),
        "patient_split": _run_split("patient", X, y, group, quick),
        "quick": quick,
        "meta": {"generated": datetime.now().isoformat(timespec="seconds"),
                 "runtime_seconds": time.perf_counter() - t0, "seed": config.SEED},
    }
    with open(config.RESULTS / "base_paper.json", "w") as f:
        json.dump(out, f, indent=1)

    print("\n══ Base-paper data · pooled accuracy ══")
    print(f"   base paper (reported)    LR {BASE_PAPER_REPORTED['lr_plaintext_accuracy']:.3f}  "
          f"encrypted LR {BASE_PAPER_REPORTED['lr_encrypted_accuracy']['poly_modulus_8192']:.3f}")
    for kind in ("row_split", "patient_split"):
        r = out[kind]
        print(f"   ours, {kind:15s}   " + "  ".join(
            f"{config.MODEL_NAMES[k]} {r['models'][k]['accuracy']:.3f}" for k in config.MODEL_KEYS)
            + f"  encrypted LR {r['fhe_logreg']['pooled']['accuracy']:.3f}"
            + f"  ({r['test_rows_with_copy_in_training_pct'] * 100:.0f}% of test rows also in training)")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tiny grids, fewer epochs")
    run(quick=ap.parse_args().quick)
