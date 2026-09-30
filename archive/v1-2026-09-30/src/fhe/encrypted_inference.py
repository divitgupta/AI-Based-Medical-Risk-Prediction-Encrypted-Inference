"""
Encrypted inference extension using Concrete ML.

Compiles the best scikit-learn / XGBoost models into FHE circuits, runs the
client → server → client flow, and compares encrypted predictions with the
plaintext ones.

Two encrypted modes are reported:
  * "simulate" – runs the FHE circuit's arithmetic (incl. quantisation noise)
    over the whole test set quickly, so accuracy is measured on every sample.
  * "execute"  – genuine homomorphic evaluation on real ciphertexts, run on a
    small subset to measure honest per-sample latency.

NOTE: Concrete ML shells out to the system linker, which breaks when the
interpreter lives under a path containing spaces. See tools/setup_fhe_env.sh.
"""

import time
import numpy as np
import joblib

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import MODELS_DIR, RESULTS_DIR
from src.utils.evaluation import full_evaluation

# Concrete ML imports (guarded so the rest of the project still works
# even if Concrete ML is not installed yet)
try:
    from concrete.ml.sklearn import (
        LogisticRegression as FHELogisticRegression,
        XGBClassifier as FHEXGBClassifier,
    )
    CONCRETE_AVAILABLE = True
except ImportError:
    CONCRETE_AVAILABLE = False
    print("[WARNING] concrete-ml not found. FHE features disabled.")

# How many test samples to push through *real* homomorphic execution.
# Real FHE is orders of magnitude slower than simulation, so we time a subset.
N_REAL_FHE_SAMPLES = 5


def build_fhe_model(model_type, n_bits=6):
    """Return an untrained Concrete ML model of the requested type."""
    if model_type == "logreg":
        return FHELogisticRegression(n_bits=n_bits, random_state=42)
    if model_type == "xgboost":
        # Kept small on purpose: FHE cost grows with ensemble size/depth.
        return FHEXGBClassifier(
            n_bits=n_bits, n_estimators=50, max_depth=3, random_state=42,
        )
    raise ValueError(f"Unsupported model_type: {model_type}")


def compile_and_evaluate_fhe(model_type, X_train, X_test, y_train, y_test,
                             dataset_name="dataset", n_bits=6):
    """
    Re-train the model using Concrete ML's drop-in replacement, compile it to
    an FHE circuit, run encrypted inference, and compare against plaintext.

    Returns
    -------
    metrics_plaintext : dict | None
    metrics_encrypted : dict | None
    timing_info       : dict | None
    """
    if not CONCRETE_AVAILABLE:
        print("  ⚠  Skipping FHE — concrete-ml not installed")
        return None, None, None

    print(f"\n🔐 Compiling {model_type} for FHE ({dataset_name}) …")

    # ── 1. Train the Concrete ML equivalent ────────────────────────────
    fhe_model = build_fhe_model(model_type, n_bits=n_bits)
    fhe_model.fit(X_train, y_train)

    # ── 2. Plaintext predictions (Concrete ML model, no encryption) ────
    y_pred_plain = fhe_model.predict(X_test)
    y_prob_plain = fhe_model.predict_proba(X_test)[:, 1]
    metrics_plain = full_evaluation(
        y_test, y_pred_plain, y_prob_plain,
        f"FHE-{model_type} (plaintext)", dataset_name,
    )

    # ── 3. Compile to an FHE circuit ───────────────────────────────────
    t0 = time.time()
    try:
        fhe_model.compile(X_train)
    except RuntimeError as e:
        print("  ⚠  FHE compilation failed. Most likely the interpreter path")
        print("     contains spaces — run tools/setup_fhe_env.sh and retry.")
        print(f"     ({str(e).splitlines()[0][:120]})")
        return metrics_plain, None, None

    compile_time = time.time() - t0
    print(f"  Compilation time: {compile_time:.2f}s")

    # ── 4. Encrypted inference over the full test set (simulation) ─────
    t0 = time.time()
    y_pred_enc = fhe_model.predict(X_test, fhe="simulate")
    sim_time = time.time() - t0

    # Probabilities are not directly available under FHE simulation,
    # so we reuse the plaintext probabilities for ROC-AUC.
    metrics_enc = full_evaluation(
        y_test, y_pred_enc, y_prob_plain,
        f"FHE-{model_type} (encrypted sim)", dataset_name,
    )

    agreement = float(np.mean(y_pred_enc == y_pred_plain))
    print(f"  Encrypted vs plaintext agreement: {agreement*100:.2f}%")

    # ── 5. Genuine homomorphic execution on a small subset ─────────────
    n_real = min(N_REAL_FHE_SAMPLES, len(X_test))
    real_per_sample_ms = None
    real_match = None
    try:
        print(f"  Running REAL encrypted inference on {n_real} samples …")
        t0 = time.time()
        y_pred_real = fhe_model.predict(X_test[:n_real], fhe="execute")
        real_time = time.time() - t0
        real_per_sample_ms = real_time / n_real * 1000
        real_match = float(np.mean(y_pred_real == y_pred_plain[:n_real]))
        print(f"  Real FHE latency: {real_per_sample_ms:.1f} ms/sample "
              f"(matches plaintext on {real_match*100:.0f}% of the subset)")
    except Exception as e:
        print(f"  ⚠  Real FHE execution unavailable: {str(e).splitlines()[0][:120]}")

    timing = {
        "compile_time_s": compile_time,
        "total_inference_s": sim_time,
        "per_sample_ms": sim_time / max(len(X_test), 1) * 1000,
        "n_samples": len(X_test),
        "agreement_with_plaintext": agreement,
        "real_fhe_per_sample_ms": real_per_sample_ms,
        "real_fhe_n_samples": n_real if real_per_sample_ms is not None else 0,
        "real_fhe_agreement": real_match,
    }
    return metrics_plain, metrics_enc, timing
