"""
Phase 2: the base paper's own dataset, split two ways.

    python -m healthguard.experiment --dataset base1025_patient
    python -m healthguard.experiment --dataset base1025_row
    python -m healthguard.base_paper            # this summary

Naresh & Reddi (2025) report 85% accuracy for plaintext logistic regression
and 84% for their CKKS-encrypted version, on the Kaggle file
johnsmith88/heart-disease-dataset. That file has 1,025 rows but only 302
distinct patient records (see data.load_base_paper), so a split made over
*rows* can put copies of the same patient in both training and test sets.

The full pipeline (three models, FHE, Elastic Learner) is retrained on the
file twice, each run in its own results folder:

    row split      our usual stratified 5-fold, applied to the 1,025 rows
                   (copies leak across train and test, as in a plain random
                   split);
    patient split  stratified *grouped* 5-fold, every copy of a patient kept
                   on the same side - also inside tuning and early stopping.

This script puts the two side by side with what the paper reports. The gap
between them is how much duplicate leakage inflates the scores.
"""

import json
from datetime import datetime

from . import config

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

SPLITS = {"row_split": "base1025_row", "patient_split": "base1025_patient"}


def _from_run(dataset):
    path = config.DATASETS[dataset]["results"] / "metrics.json"
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run python -m healthguard.experiment --dataset {dataset} first.")
    m = json.loads(path.read_text())
    return {
        "dataset": dataset,
        "test_rows_with_copy_in_training": m["test_rows_with_copy_in_training"],
        "test_rows_with_copy_in_training_pct": m["test_rows_with_copy_in_training_pct"],
        "models": {k: m["models"][k]["pooled"] for k in config.MODEL_KEYS},
        "fhe_logreg": {"pooled": m["fhe"]["logreg"]["pooled"], "agreement": m["fhe"]["logreg"]["agreement"]},
        "fhe": {k: {"pooled": m["fhe"][k]["pooled"], "agreement": m["fhe"][k]["agreement"]}
                for k in config.MODEL_KEYS},
    }, m


def run():
    out = {"base_paper_reported": BASE_PAPER_REPORTED}
    for key, dataset in SPLITS.items():
        out[key], m = _from_run(dataset)
    out["dataset"] = {"rows": m["n_samples"], "distinct_patients": m["n_distinct_patients"],
                      "positive_rows": m["n_positive"]}
    out["meta"] = {"generated": datetime.now().isoformat(timespec="seconds"), "seed": config.SEED}
    with open(config.DATASETS[config.DEFAULT_DATASET]["results"] / "base_paper.json", "w") as f:
        json.dump(out, f, indent=1)

    print("\n══ Base paper's file · pooled accuracy (retrained) ══")
    print(f"   base paper (reported)    LR {BASE_PAPER_REPORTED['lr_plaintext_accuracy']:.3f}  "
          f"encrypted LR {BASE_PAPER_REPORTED['lr_encrypted_accuracy']['poly_modulus_8192']:.3f}")
    for kind in SPLITS:
        r = out[kind]
        print(f"   {kind:15s}   " + "  ".join(
            f"{config.MODEL_NAMES[k]} {r['models'][k]['accuracy']:.3f}" for k in config.MODEL_KEYS)
            + f"  encrypted LR {r['fhe_logreg']['pooled']['accuracy']:.3f}"
            + f"  ({r['test_rows_with_copy_in_training_pct'] * 100:.0f}% of test rows also in training)")
    return out


if __name__ == "__main__":
    run()
