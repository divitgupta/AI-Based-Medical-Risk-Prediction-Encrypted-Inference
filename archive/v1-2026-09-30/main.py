#!/usr/bin/env python3
"""
main.py – End-to-end pipeline for the AIML EL project.

Phases:
    1. EDA  (plots saved to results/plots/)
    2. Train & evaluate all 3 models on both datasets
    3. FHE encrypted-inference comparison
    4. Summary table

Run from project root:
    source venv/bin/activate
    python main.py
"""

import sys, os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_predict

# Ensure src is importable
sys.path.insert(0, os.path.dirname(__file__))

from src.config import RANDOM_SEED, CV_FOLDS, RESULTS_DIR
from src.data.loader import load_breast_cancer, load_heart_disease
from src.eda import eda_breast_cancer, eda_heart_disease
from src.models.trainers import (
    train_logistic_regression,
    train_xgboost,
    train_mlp,
)
from src.fhe.encrypted_inference import compile_and_evaluate_fhe
from src.utils.evaluation import full_evaluation


def run_breast_cancer_pipeline():
    """Full pipeline for the Breast Cancer dataset."""
    print("\n" + "=" * 70)
    print("  BREAST CANCER WISCONSIN (DIAGNOSTIC)")
    print("=" * 70)

    X_train, X_test, y_train, y_test, feat_names, preprocessor = load_breast_cancer()

    results = {}

    # 1. Logistic Regression
    lr_model, lr_metrics = train_logistic_regression(
        X_train, y_train, X_test, y_test, "Breast Cancer"
    )
    results["LogisticRegression"] = lr_metrics

    # 2. XGBoost
    xgb_model, xgb_metrics = train_xgboost(
        X_train, y_train, X_test, y_test, "Breast Cancer"
    )
    results["XGBoost"] = xgb_metrics

    # 3. MLP
    mlp_model, mlp_metrics = train_mlp(
        X_train, y_train, X_test, y_test, "Breast Cancer"
    )
    results["MLP"] = mlp_metrics

    # 4. FHE comparison (LogReg and XGBoost only)
    fhe_results = {}
    for mtype in ["logreg", "xgboost"]:
        plain, enc, timing = compile_and_evaluate_fhe(
            mtype, X_train, X_test, y_train, y_test, "Breast Cancer"
        )
        if plain is not None:
            fhe_results[mtype] = {
                "plaintext": plain,
                "encrypted": enc,
                "timing": timing,
            }
    results["FHE"] = fhe_results

    return results


def run_heart_disease_pipeline():
    """
    Full pipeline for the Heart Disease dataset.

    The loader performs the stratified 80/20 split and fits the preprocessor on
    the training rows only, so no test-set statistics reach the models.
    """
    print("\n" + "=" * 70)
    print("  HEART DISEASE (CLEVELAND)")
    print("=" * 70)

    X_train, X_test, y_train, y_test, feat_names, preprocessor = load_heart_disease()

    results = {}

    # 1. Logistic Regression
    lr_model, lr_metrics = train_logistic_regression(
        X_train, y_train, X_test, y_test, "Heart Disease"
    )
    results["LogisticRegression"] = lr_metrics

    # 2. XGBoost
    xgb_model, xgb_metrics = train_xgboost(
        X_train, y_train, X_test, y_test, "Heart Disease"
    )
    results["XGBoost"] = xgb_metrics

    # 3. MLP
    mlp_model, mlp_metrics = train_mlp(
        X_train, y_train, X_test, y_test, "Heart Disease"
    )
    results["MLP"] = mlp_metrics

    # 4. FHE comparison
    fhe_results = {}
    for mtype in ["logreg", "xgboost"]:
        plain, enc, timing = compile_and_evaluate_fhe(
            mtype, X_train, X_test, y_train, y_test, "Heart Disease"
        )
        if plain is not None:
            fhe_results[mtype] = {
                "plaintext": plain,
                "encrypted": enc,
                "timing": timing,
            }
    results["FHE"] = fhe_results

    return results


def print_summary_table(bc_results, hd_results):
    """Print a consolidated comparison table."""
    print("\n" + "=" * 70)
    print("  SUMMARY OF ALL RESULTS")
    print("=" * 70)

    rows = []
    for dataset, res in [("Breast Cancer", bc_results), ("Heart Disease", hd_results)]:
        for model_name in ["LogisticRegression", "XGBoost", "MLP"]:
            m = res.get(model_name, {})
            rows.append({
                "Dataset": dataset,
                "Model": model_name,
                "Accuracy": m.get("accuracy", None),
                "Precision": m.get("precision", None),
                "Recall": m.get("recall", None),
                "F1": m.get("f1", None),
                "ROC-AUC": m.get("roc_auc", None),
            })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format="%.4f"))

    # Save to CSV
    csv_path = RESULTS_DIR / "summary_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved → {csv_path}")

    # FHE comparison
    fhe_rows = []
    for dataset, res in [("Breast Cancer", bc_results), ("Heart Disease", hd_results)]:
        for mtype, data in res.get("FHE", {}).items():
            if not data:
                continue
            t = data.get("timing") or {}
            fhe_rows.append({
                "Dataset": dataset,
                "Model": mtype,
                "Plaintext Acc": (data.get("plaintext") or {}).get("accuracy"),
                "Encrypted Acc": (data.get("encrypted") or {}).get("accuracy"),
                "Agreement": t.get("agreement_with_plaintext"),
                "Compile (s)": t.get("compile_time_s"),
                "Sim ms/sample": t.get("per_sample_ms"),
                "Real FHE ms/sample": t.get("real_fhe_per_sample_ms"),
            })
    if fhe_rows:
        fhe_df = pd.DataFrame(fhe_rows)
        fhe_csv = RESULTS_DIR / "fhe_comparison.csv"
        fhe_df.to_csv(fhe_csv, index=False)
        print(f"\n  Saved → {fhe_csv}")

    for dataset, res in [("Breast Cancer", bc_results), ("Heart Disease", hd_results)]:
        fhe = res.get("FHE", {})
        if fhe:
            print(f"\n  FHE Comparison – {dataset}:")
            for mtype, data in fhe.items():
                if data:
                    print(f"    {mtype}:")
                    print(f"      Plaintext accuracy : {data['plaintext'].get('accuracy', 'N/A') if data['plaintext'] else 'N/A'}")
                    print(f"      Encrypted accuracy : {data['encrypted'].get('accuracy', 'N/A') if data.get('encrypted') else 'N/A'}")
                    if data.get('timing'):
                        print(f"      Compile time       : {data['timing']['compile_time_s']:.2f}s")
                        print(f"      Per-sample latency : {data['timing']['per_sample_ms']:.2f}ms")


def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  AI Based Medical Risk Prediction & Encrypted Inference ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    # Phase 1: EDA
    print("▶ Phase 1: Exploratory Data Analysis")
    eda_breast_cancer()
    eda_heart_disease()

    # Phase 2-3: Training & Evaluation
    print("\n▶ Phase 2-3: Model Training & Evaluation")
    bc_results = run_breast_cancer_pipeline()
    hd_results = run_heart_disease_pipeline()

    # Phase 4: Summary
    print_summary_table(bc_results, hd_results)

    # Save full results as JSON (FHE block included so the dashboard can
    # show the plaintext vs. encrypted comparison without re-running).
    all_results = {
        "breast_cancer": bc_results,
        "heart_disease": hd_results,
    }
    json_path = RESULTS_DIR / "all_metrics.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Full metrics saved → {json_path}")

    print("\n✅ Pipeline complete!")


if __name__ == "__main__":
    main()
