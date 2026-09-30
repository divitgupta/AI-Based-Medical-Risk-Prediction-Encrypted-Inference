"""
Central configuration for the project.
All paths, hyperparameters, random seeds, and column definitions live here
so every other module imports from one place.
"""

import os
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
MODELS_DIR = RESULTS_DIR / "models"

for _d in [DATA_PROCESSED, RESULTS_DIR, PLOTS_DIR, MODELS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Reproducibility ─────────────────────────────────────────────────────
RANDOM_SEED = 42

# ── Dataset Definitions ─────────────────────────────────────────────────
BREAST_CANCER_CSV = DATA_RAW / "breast_cancer.csv"
HEART_DISEASE_CSV = DATA_RAW / "heart_disease.csv"

# Heart Disease column metadata (Cleveland dataset)
HEART_COLS = {
    "target": "target",
    "categorical": ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"],
    "continuous": ["age", "trestbps", "chol", "thalach", "oldpeak"],
}

# Breast Cancer: all 30 features are continuous; target is last column
BREAST_CANCER_COLS = {
    "target": "target",
    # The ten base measurements, each with mean / se / worst
    "base_features": [
        "radius", "texture", "perimeter", "area", "smoothness",
        "compactness", "concavity", "concave points", "symmetry",
        "fractal dimension",
    ],
}

# ── Train / Test Split ──────────────────────────────────────────────────
TEST_SIZE = 0.20          # 80/20 for Breast Cancer
CV_FOLDS = 5              # 5-fold stratified CV for Heart Disease

# ── Model Hyperparameter Search Spaces ──────────────────────────────────
LOGREG_PARAMS = {
    "C": [0.01, 0.1, 1.0, 10.0],
    "max_iter": [1000],
    "solver": ["lbfgs"],
}

RF_PARAMS = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 8, None],
    "min_samples_split": [2, 5],
}

XGB_PARAMS = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 8],
    "learning_rate": [0.05, 0.1, 0.2],
    "use_label_encoder": [False],
    "eval_metric": ["logloss"],
}

MLP_PARAMS = {
    "hidden_sizes": [16, 8],     # 30 → 16 → 8 → 1  (breast cancer)
    "lr": 1e-3,
    "dropout": 0.3,
    "epochs": 200,
    "patience": 20,              # early stopping patience
    "batch_size": 32,
}

# ── Evaluation ──────────────────────────────────────────────────────────
SCORING_METRIC = "recall"        # primary metric for a diagnostic task
EVAL_METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]
