"""
app.py – Demo web server for the AIML EL project.

Serves a dashboard that runs the same patient record through two paths:

  1. Plaintext inference  – the classical scikit-learn model.
  2. Encrypted inference  – a Concrete ML circuit driven through the real
     client → server → client FHE flow. The client quantises and encrypts the
     record, the "hospital server" evaluates the model on ciphertext only, and
     the client decrypts the result. The server never sees a plaintext feature.

Run with the space-free interpreter (see tools/setup_fhe_env.sh):
    ~/aimlel-venv/bin/python app.py
"""

import os
import sys
import json
import time
import shutil
import traceback

import numpy as np
import pandas as pd
import joblib
from flask import Flask, jsonify, request, render_template, send_from_directory
from flask_cors import CORS
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(__file__))

from src.config import (
    BREAST_CANCER_CSV, HEART_DISEASE_CSV,
    RANDOM_SEED, TEST_SIZE, CV_FOLDS, MODELS_DIR, RESULTS_DIR, PLOTS_DIR,
    LOGREG_PARAMS, XGB_PARAMS, MLP_PARAMS, SCORING_METRIC,
)
from src.data.loader import load_breast_cancer, load_heart_disease
from src.utils.evaluation import evaluate_model

app = Flask(__name__)
CORS(app)

# ── Configuration ────────────────────────────────────────────────────────
FHE_N_BITS = 8                       # quantisation width for the FHE circuit
FHE_DEPLOY_DIR = RESULTS_DIR / "fhe_deployment"
DATASETS = ("breast_cancer", "heart_disease")

LABELS = {
    "breast_cancer": {0: "Malignant", 1: "Benign", "positive_is_bad": False},
    "heart_disease": {0: "No Disease", 1: "Disease", "positive_is_bad": True},
}

# ── Global state ─────────────────────────────────────────────────────────
MODELS = {}          # dataset -> fitted sklearn LogisticRegression
PREPROCESSORS = {}   # dataset -> fitted ColumnTransformer
RAW_TEST_DATA = {}   # dataset -> DataFrame of raw features + _true_label
SPLITS = {}          # dataset -> dict of preprocessed arrays
METRICS = {}         # contents of results/all_metrics.json
FHE = {}             # dataset -> dict(client, server, evk, meta)
FHE_ERROR = None     # populated if the FHE stack could not be brought up
FEATURE_NAMES = {}   # dataset -> list[str] of post-preprocessing feature names
EDA = {}             # dataset -> descriptive statistics for the Data view
MODEL_CARDS = {}     # dataset -> per-model hyperparameters and internals


# ═════════════════════════════════════════════════════════════════════════
#  Start-up
# ═════════════════════════════════════════════════════════════════════════

def _load_data_and_models():
    """Rebuild the exact train/test splits used by main.py and load models."""
    print("Loading datasets, preprocessors and models …")

    # ── Breast Cancer ────────────────────────────────────────────────────
    X_tr_bc, X_te_bc, y_tr_bc, y_te_bc, _, prep_bc = load_breast_cancer()
    df_bc = pd.read_csv(BREAST_CANCER_CSV)
    X_raw_bc = df_bc.drop(columns=["target"])
    y_raw_bc = df_bc["target"].values
    _, X_test_raw_bc, _, _ = train_test_split(
        X_raw_bc, y_raw_bc, test_size=TEST_SIZE,
        random_state=RANDOM_SEED, stratify=y_raw_bc,
    )

    # ── Heart Disease ────────────────────────────────────────────────────
    X_tr_hd, X_te_hd, y_tr_hd, y_te_hd, _, prep_hd = load_heart_disease()
    df_hd = pd.read_csv(HEART_DISEASE_CSV)
    df_hd.replace("?", np.nan, inplace=True)
    df_hd.dropna(inplace=True)
    X_raw_hd = df_hd.drop(columns=["target"])
    y_raw_hd = df_hd["target"].astype(int).values
    # Same parameters as the loader's split, so the raw rows recovered here line
    # up row-for-row with X_te_hd.
    _, X_test_raw_hd, _, y_test_raw_hd = train_test_split(
        X_raw_hd, y_raw_hd, test_size=TEST_SIZE,
        random_state=RANDOM_SEED, stratify=y_raw_hd,
    )

    PREPROCESSORS["breast_cancer"] = prep_bc
    PREPROCESSORS["heart_disease"] = prep_hd
    FEATURE_NAMES["breast_cancer"] = list(prep_bc.get_feature_names_out())
    FEATURE_NAMES["heart_disease"] = list(prep_hd.get_feature_names_out())

    SPLITS["breast_cancer"] = dict(
        X_train=X_tr_bc, y_train=y_tr_bc, X_test=X_te_bc, y_test=y_te_bc)
    SPLITS["heart_disease"] = dict(
        X_train=X_tr_hd, y_train=y_tr_hd, X_test=X_te_hd, y_test=y_te_hd)

    # Keep the raw (human-readable) test rows next to their true labels so the
    # UI can show a realistic patient record rather than scaled numbers.
    X_test_raw_bc = X_test_raw_bc.copy()
    X_test_raw_hd = X_test_raw_hd.copy()
    X_test_raw_bc["_true_label"] = y_te_bc
    X_test_raw_hd["_true_label"] = y_test_raw_hd
    RAW_TEST_DATA["breast_cancer"] = X_test_raw_bc
    RAW_TEST_DATA["heart_disease"] = X_test_raw_hd

    MODELS["breast_cancer"] = joblib.load(MODELS_DIR / "logreg_breast_cancer.joblib")
    MODELS["heart_disease"] = joblib.load(MODELS_DIR / "logreg_heart_disease.joblib")

    global METRICS
    try:
        with open(RESULTS_DIR / "all_metrics.json") as f:
            METRICS = json.load(f)
        print(f"  Loaded metrics for: {', '.join(METRICS.keys())}")
    except Exception as e:
        print(f"  [warn] could not read all_metrics.json: {e}")
        METRICS = {}


def _clean_feature_name(name):
    """Strip the ColumnTransformer prefix: 'scale__mean radius' -> 'mean radius'."""
    return name.split("__", 1)[-1] if "__" in name else name


def _build_eda():
    """Descriptive statistics for the Data view, computed from the raw CSVs."""
    # ── Breast Cancer ────────────────────────────────────────────────────
    df_bc = pd.read_csv(BREAST_CANCER_CSV)
    counts_bc = df_bc["target"].value_counts().to_dict()
    EDA["breast_cancer"] = {
        "title": "Breast Cancer Wisconsin (Diagnostic)",
        "source": "UCI ML Repository · digitised fine-needle aspirate images",
        "n_samples": int(df_bc.shape[0]),
        "n_features": int(df_bc.shape[1] - 1),
        "n_missing": int(df_bc.isna().sum().sum()),
        "feature_types": "30 continuous (10 measurements × mean / SE / worst)",
        "class_balance": [
            {"label": "Malignant", "count": int(counts_bc.get(0, 0))},
            {"label": "Benign", "count": int(counts_bc.get(1, 0))},
        ],
        "split": f"Stratified {int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)} train-test",
        "preprocessing": "StandardScaler on all 30 features",
        "plots": [
            {"file": "eda_bc_class_dist.png", "caption": "Class distribution"},
            {"file": "eda_bc_corr_matrix.png", "caption": "Feature correlation matrix"},
            {"file": "eda_bc_feat_dists.png", "caption": "Feature distributions"},
            {"file": "eda_bc_top5_boxplots.png", "caption": "Top-5 discriminative features"},
        ],
    }

    # ── Heart Disease ────────────────────────────────────────────────────
    df_hd = pd.read_csv(HEART_DISEASE_CSV)
    raw_rows = int(df_hd.shape[0])
    df_hd_clean = df_hd.replace("?", np.nan).dropna()
    counts_hd = df_hd_clean["target"].astype(int).value_counts().to_dict()
    EDA["heart_disease"] = {
        "title": "Heart Disease (Cleveland)",
        "source": "UCI ML Repository · Cleveland Clinic Foundation",
        "n_samples": int(df_hd_clean.shape[0]),
        "n_features": int(df_hd_clean.shape[1] - 1),
        "n_missing": raw_rows - int(df_hd_clean.shape[0]),
        "feature_types": "5 continuous + 8 categorical → 30 after one-hot encoding",
        "class_balance": [
            {"label": "No Disease", "count": int(counts_hd.get(0, 0))},
            {"label": "Disease", "count": int(counts_hd.get(1, 0))},
        ],
        "split": "Stratified 80/20 train-test",
        "preprocessing": "StandardScaler on continuous + OneHotEncoder on categorical",
        "plots": [
            {"file": "eda_hd_class_dist.png", "caption": "Class distribution"},
            {"file": "eda_hd_corr_matrix.png", "caption": "Feature correlation matrix"},
            {"file": "eda_hd_cont_dists.png", "caption": "Continuous feature distributions"},
            {"file": "eda_hd_cat_dists.png", "caption": "Categorical feature distributions"},
        ],
    }


def _build_model_cards():
    """
    Hyperparameters actually selected by grid search, plus the learned
    coefficients / feature importances, for the Models view.
    """
    slug = {"breast_cancer": "breast_cancer", "heart_disease": "heart_disease"}
    for ds in DATASETS:
        names = FEATURE_NAMES[ds]
        cards = []

        # ── Logistic Regression ──────────────────────────────────────────
        lr = MODELS[ds]
        coefs = lr.coef_[0]
        order = np.argsort(np.abs(coefs))[::-1][:10]
        cards.append({
            "name": "Logistic Regression",
            "family": "Linear model",
            "selected": {
                "C": lr.C,
                "solver": lr.solver,
                "class_weight": str(lr.class_weight),
                "max_iter": lr.max_iter,
            },
            "searched": {"C": LOGREG_PARAMS["C"]},
            "notes": "Linear decision boundary; compiles to a single encrypted "
                     "dot product, which is why it is the FHE workhorse here.",
            "weights": [
                {"feature": _clean_feature_name(names[i]),
                 "value": float(coefs[i])} for i in order
            ],
            "weights_label": "Largest coefficients (signed)",
        })

        # ── XGBoost ──────────────────────────────────────────────────────
        try:
            xgb = joblib.load(MODELS_DIR / f"xgb_{slug[ds]}.joblib")
            imps = np.asarray(xgb.feature_importances_)
            order = np.argsort(imps)[::-1][:10]
            params = xgb.get_params()
            cards.append({
                "name": "XGBoost",
                "family": "Gradient-boosted trees",
                "selected": {
                    "n_estimators": params.get("n_estimators"),
                    "max_depth": params.get("max_depth"),
                    "learning_rate": params.get("learning_rate"),
                    "scale_pos_weight": round(float(params.get("scale_pos_weight") or 0), 3),
                },
                "searched": {k: XGB_PARAMS[k] for k in
                             ("n_estimators", "max_depth", "learning_rate")},
                "notes": "Captures non-linear feature interactions. Roughly 40× more "
                         "expensive than the linear model under FHE.",
                "weights": [
                    {"feature": _clean_feature_name(names[i]),
                     "value": float(imps[i])} for i in order
                ],
                "weights_label": "Feature importance (gain)",
            })
        except Exception as e:
            print(f"  [warn] could not read XGBoost model for {ds}: {e}")

        # ── MLP ──────────────────────────────────────────────────────────
        h = MLP_PARAMS["hidden_sizes"]
        n_in = len(names)
        cards.append({
            "name": "MLP",
            "family": "Neural network (PyTorch)",
            "selected": {
                "architecture": f"{n_in} → {h[0]} → {h[1]} → 1",
                "dropout": MLP_PARAMS["dropout"],
                "learning_rate": MLP_PARAMS["lr"],
                "batch_size": MLP_PARAMS["batch_size"],
                "max_epochs": MLP_PARAMS["epochs"],
                "early_stopping_patience": MLP_PARAMS["patience"],
            },
            "searched": {},
            "notes": "Trained with early stopping on a validation split. Not yet "
                     "FHE-compiled — that needs quantisation-aware training.",
            "weights": [],
            "weights_label": "",
        })

        MODEL_CARDS[ds] = {
            "models": cards,
            "protocol": {
                "cv": f"{CV_FOLDS}-fold stratified cross-validation",
                "scoring": SCORING_METRIC,
                "scoring_rationale": "Recall is the primary metric: a missed "
                                     "diagnosis costs far more than a false alarm.",
                "seed": RANDOM_SEED,
                "class_imbalance": "class_weight='balanced' (LogReg) / "
                                   "scale_pos_weight (XGBoost)",
            },
        }


def _setup_fhe():
    """
    Train + compile a Concrete ML logistic regression per dataset and wire up
    the client/server deployment objects used by /api/predict.
    """
    global FHE_ERROR
    try:
        from concrete.ml.sklearn import LogisticRegression as FHELogisticRegression
        from concrete.ml.deployment import FHEModelDev, FHEModelClient, FHEModelServer
    except ImportError as e:
        FHE_ERROR = f"concrete-ml not installed ({e})"
        print(f"  [warn] {FHE_ERROR}")
        return

    for ds in DATASETS:
        split = SPLITS[ds]
        print(f"\n🔐 Preparing FHE circuit for {ds} …")
        try:
            model = FHELogisticRegression(n_bits=FHE_N_BITS, random_state=RANDOM_SEED)
            model.fit(split["X_train"], split["y_train"])

            t0 = time.time()
            model.compile(split["X_train"])
            compile_s = time.time() - t0
            print(f"   compiled in {compile_s:.2f}s")

            # Quality of the quantised model, measured on the same test set.
            y_pred = model.predict(split["X_test"])
            y_prob = model.predict_proba(split["X_test"])[:, 1]
            fhe_metrics = evaluate_model(split["y_test"], y_pred, y_prob)

            # Deployment artefacts: dev bundle (server side) + client keys.
            dev_dir = FHE_DEPLOY_DIR / ds / "dev"
            key_dir = FHE_DEPLOY_DIR / ds / "keys"
            if dev_dir.exists():
                shutil.rmtree(dev_dir)
            dev_dir.parent.mkdir(parents=True, exist_ok=True)
            FHEModelDev(path_dir=str(dev_dir), model=model).save()

            client = FHEModelClient(path_dir=str(dev_dir), key_dir=str(key_dir))
            t0 = time.time()
            evk = client.get_serialized_evaluation_keys()
            keygen_s = time.time() - t0

            server = FHEModelServer(path_dir=str(dev_dir))
            server.load()

            FHE[ds] = {
                "model": model,
                "client": client,
                "server": server,
                "evk": evk,
                "meta": {
                    "n_bits": FHE_N_BITS,
                    "compile_time_s": round(compile_s, 3),
                    "keygen_time_s": round(keygen_s, 3),
                    "evaluation_key_bytes": len(evk),
                    "n_features": int(split["X_train"].shape[1]),
                    "metrics": fhe_metrics,
                },
            }
            print(f"   ready — eval key {len(evk)/1024:.1f} KB, "
                  f"quantised accuracy {fhe_metrics['accuracy']*100:.2f}%")
        except Exception as e:
            FHE_ERROR = str(e).splitlines()[0][:200]
            print(f"   [warn] FHE setup failed for {ds}: {FHE_ERROR}")
            if "ConcretelangRuntime" in str(e) or "Can't emit artifacts" in str(e):
                print("   → The interpreter path contains spaces. "
                      "Run tools/setup_fhe_env.sh and use ~/aimlel-venv/bin/python.")
            traceback.print_exc(limit=1)


def init_app():
    _load_data_and_models()
    _build_eda()
    _build_model_cards()
    _setup_fhe()
    print("\n✅ Server ready.\n")


# ═════════════════════════════════════════════════════════════════════════
#  Helpers
# ═════════════════════════════════════════════════════════════════════════

def _label_text(dataset, raw):
    return LABELS[dataset][int(raw)]


def _is_alarming(dataset, raw):
    """True when the prediction is the clinically bad outcome."""
    if dataset == "breast_cancer":
        return int(raw) == 0          # malignant
    return int(raw) == 1              # disease present


def _preprocess(dataset, features_dict):
    df_in = pd.DataFrame([features_dict])
    return PREPROCESSORS[dataset].transform(df_in)


# ═════════════════════════════════════════════════════════════════════════
#  Routes
# ═════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def status():
    """Everything the UI needs to describe the current backend state."""
    return jsonify({
        "fhe_available": {ds: ds in FHE for ds in DATASETS},
        "fhe_error": FHE_ERROR,
        "fhe_meta": {ds: FHE[ds]["meta"] for ds in FHE},
        "datasets": {
            ds: {
                "n_test": int(len(RAW_TEST_DATA[ds])),
                "n_features_raw": int(RAW_TEST_DATA[ds].shape[1] - 1),
                "labels": {"0": LABELS[ds][0], "1": LABELS[ds][1]},
            } for ds in DATASETS
        },
    })


@app.route("/api/metrics")
def get_metrics():
    return jsonify(METRICS)


@app.route("/api/eda/<dataset>")
def get_eda(dataset):
    if dataset not in EDA:
        return jsonify({"error": "Unknown dataset"}), 404
    return jsonify(EDA[dataset])


@app.route("/api/models/<dataset>")
def get_models(dataset):
    if dataset not in MODEL_CARDS:
        return jsonify({"error": "Unknown dataset"}), 404
    return jsonify(MODEL_CARDS[dataset])


@app.route("/api/evaluation/<dataset>")
def get_evaluation(dataset):
    """Confusion-matrix and ROC plot filenames for every model on this dataset."""
    if dataset not in DATASETS:
        return jsonify({"error": "Unknown dataset"}), 404

    variants = [
        ("logistic_regression", "Logistic Regression", False),
        ("xgboost", "XGBoost", False),
        ("mlp", "MLP", False),
        ("fhe-logreg_(plaintext)", "FHE LogReg — plaintext", True),
        ("fhe-logreg_(encrypted_sim)", "FHE LogReg — encrypted", True),
        ("fhe-xgboost_(plaintext)", "FHE XGBoost — plaintext", True),
        ("fhe-xgboost_(encrypted_sim)", "FHE XGBoost — encrypted", True),
    ]
    out = []
    for slug, label, is_fhe in variants:
        cm = f"cm_{dataset}_{slug}.png"
        roc = f"roc_{dataset}_{slug}.png"
        if (PLOTS_DIR / cm).exists() and (PLOTS_DIR / roc).exists():
            out.append({"label": label, "cm": cm, "roc": roc, "fhe": is_fhe})
    return jsonify({"variants": out})


@app.route("/plots/<path:filename>")
def serve_plot(filename):
    return send_from_directory(PLOTS_DIR, filename)


@app.route("/api/random_patient/<dataset>")
def random_patient(dataset):
    if dataset not in RAW_TEST_DATA:
        return jsonify({"error": "Unknown dataset"}), 404

    row = RAW_TEST_DATA[dataset].sample(1)
    true_label = int(row["_true_label"].iloc[0])
    features = row.drop(columns=["_true_label"]).iloc[0].to_dict()
    features = {k: (float(v) if isinstance(v, (int, float, np.number)) else v)
                for k, v in features.items()}

    return jsonify({
        "features": features,
        "true_label_raw": true_label,
        "true_label_text": _label_text(dataset, true_label),
        "true_label_alarming": _is_alarming(dataset, true_label),
    })


@app.route("/api/predict/<dataset>", methods=["POST"])
def predict(dataset):
    """
    Run one patient record through the plaintext model, the encrypted circuit,
    or both. Body: {"features": {...}, "mode": "plaintext"|"encrypted"|"both"}
    """
    if dataset not in MODELS:
        return jsonify({"error": "Unknown dataset"}), 404

    payload = request.get_json(silent=True) or {}
    if "features" not in payload:
        return jsonify({"error": "Missing features"}), 400

    mode = payload.get("mode", "both")
    try:
        X = _preprocess(dataset, payload["features"])
    except Exception as e:
        return jsonify({"error": f"Preprocessing failed: {e}"}), 400

    out = {"dataset": dataset, "mode": mode}

    # ── Plaintext path ───────────────────────────────────────────────────
    # To make the comparison meaningful this runs the *same* quantised model
    # that the FHE circuit was compiled from, just without encryption. Any
    # difference against the encrypted card is therefore caused by FHE alone,
    # not by comparing two differently-trained models.
    if mode in ("plaintext", "both"):
        if dataset in FHE:
            model = FHE[dataset]["model"]
            source = f"Concrete ML logistic regression ({FHE_N_BITS}-bit), unencrypted"
        else:
            model = MODELS[dataset]
            source = "scikit-learn logistic regression"

        t0 = time.perf_counter()
        probs = model.predict_proba(X)[0]
        elapsed_ms = (time.perf_counter() - t0) * 1000
        pred = int(np.argmax(probs))
        prob = float(probs[1]) if len(probs) > 1 else float(probs[0])
        out["plaintext"] = {
            "prediction_raw": pred,
            "prediction_text": _label_text(dataset, pred),
            "alarming": _is_alarming(dataset, pred),
            "prob_positive": prob,
            "confidence": float(np.max(probs)),
            "latency_ms": round(elapsed_ms, 3),
            "source": source,
        }

        # scikit-learn baseline, shown as a footnote so the quantised model can
        # be sanity-checked against the model reported in summary_results.csv.
        base = MODELS[dataset]
        base_probs = base.predict_proba(X)[0]
        base_pred = int(np.argmax(base_probs))
        out["baseline"] = {
            "prediction_raw": base_pred,
            "prediction_text": _label_text(dataset, base_pred),
            "confidence": float(np.max(base_probs)),
        }

        out["explanation"] = _explain(dataset, X)

    # ── Encrypted path ───────────────────────────────────────────────────
    if mode in ("encrypted", "both"):
        if dataset not in FHE:
            out["encrypted"] = {"error": FHE_ERROR or "FHE circuit unavailable"}
        else:
            try:
                out["encrypted"] = _run_encrypted(dataset, X)
            except Exception as e:
                out["encrypted"] = {"error": str(e).splitlines()[0][:200]}

    return jsonify(out)


def _explain(dataset, X, top_k=6):
    """
    Per-feature contribution to the log-odds for this one patient.

    For a linear model the decision function is w·x + b, so the signed product
    w_i * x_i is exactly how much feature i pushed this prediction, in log-odds.
    That makes the live prediction inspectable rather than a black box.
    """
    model = MODELS[dataset]
    coefs = model.coef_[0]
    x = np.asarray(X).reshape(-1)
    contributions = coefs * x

    order = np.argsort(np.abs(contributions))[::-1][:top_k]
    names = FEATURE_NAMES[dataset]
    total = float(np.sum(np.abs(contributions))) or 1.0

    return {
        "intercept": float(model.intercept_[0]),
        "log_odds": float(np.dot(coefs, x) + model.intercept_[0]),
        "positive_class": LABELS[dataset][1],
        "negative_class": LABELS[dataset][0],
        "top_features": [
            {
                "feature": _clean_feature_name(names[i]),
                "contribution": float(contributions[i]),
                "share": float(abs(contributions[i]) / total),
                "scaled_value": float(x[i]),
                "pushes_toward": LABELS[dataset][1] if contributions[i] > 0
                                 else LABELS[dataset][0],
            }
            for i in order
        ],
    }


def _run_encrypted(dataset, X):
    """
    The full client → server → client flow, timed step by step.

    The `server` object only ever touches `ct` (serialized ciphertext) and the
    public evaluation key: at no point does it hold a plaintext feature value.
    """
    bundle = FHE[dataset]
    client, server, evk = bundle["client"], bundle["server"], bundle["evk"]

    t0 = time.perf_counter()
    ct = client.quantize_encrypt_serialize(X)
    t_encrypt = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    ct_result = server.run(ct, evk)
    t_server = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    decrypted = client.deserialize_decrypt_dequantize(ct_result)
    t_decrypt = (time.perf_counter() - t0) * 1000

    probs = np.asarray(decrypted).reshape(-1)
    pred = int(np.argmax(probs))
    prob_pos = float(probs[1]) if probs.size > 1 else float(probs[0])

    return {
        "prediction_raw": pred,
        "prediction_text": _label_text(dataset, pred),
        "alarming": _is_alarming(dataset, pred),
        "prob_positive": prob_pos,
        "confidence": float(np.max(probs)),
        "timings_ms": {
            "encrypt": round(t_encrypt, 2),
            "server_compute": round(t_server, 2),
            "decrypt": round(t_decrypt, 2),
            "total": round(t_encrypt + t_server + t_decrypt, 2),
        },
        "ciphertext_bytes": len(ct),
        "ciphertext_preview": ct[:32].hex(),
        "result_ciphertext_bytes": len(ct_result),
        "evaluation_key_bytes": len(evk),
        "n_bits": bundle["meta"]["n_bits"],
    }


if __name__ == "__main__":
    init_app()
    # debug=False: the reloader would re-run the (slow) FHE compilation twice.
    app.run(host="0.0.0.0", port=5001, debug=False)
