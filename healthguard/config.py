"""
Single source of configuration: paths, seeds, feature groups, search spaces.
Every other module imports from here.
"""

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
HEART_CSV = DATA_RAW / "heart_disease.csv"
# The 1,025-row Kaggle file the base paper used (johnsmith88/heart-disease-dataset).
BASE_PAPER_CSV = DATA_RAW / "heart_disease_base_paper_1025.csv"

# ── Datasets ─────────────────────────────────────────────────────────────
# The full pipeline (three models, FHE, Elastic Learner) can be trained on
# three datasets; each writes its own results folder and the web app has a
# switch between them.
#   cleveland         our 303 patients, stratified 5-fold.
#   base1025_patient  the base paper's 1,025-row file (only 302 distinct
#                     patients). Every split - outer folds, inner tuning
#                     folds, the MLP's early-stopping slice - keeps all copies
#                     of a patient on one side, so no model is tested on a
#                     patient it trained on.
#   base1025_row      the same file with our usual split applied to rows, as
#                     a plain random split would. Copies of a test patient sit
#                     in training, so its scores are inflated (shown for
#                     comparison).
RESULTS_ROOT = ROOT / "results"
DATASETS = {
    "cleveland": {"label": "Cleveland · 303 patients", "short": "Cleveland (303)",
                  "results": RESULTS_ROOT, "split": "stratified 5-fold"},
    "base1025_patient": {"label": "Base paper · 1,025 rows, split by patient",
                         "short": "Base paper · by patient",
                         "results": RESULTS_ROOT / "base1025_patient",
                         "split": "stratified 5-fold, grouped by patient"},
    "base1025_row": {"label": "Base paper · 1,025 rows, split by row",
                     "short": "Base paper · by row",
                     "results": RESULTS_ROOT / "base1025_row",
                     "split": "stratified 5-fold over rows (copies leak)"},
}
DEFAULT_DATASET = "cleveland"

DATASET = DEFAULT_DATASET
RESULTS = DATASETS[DATASET]["results"]
MODELS_DIR = RESULTS / "models"
FHE_DIR = RESULTS / "fhe"


def use_dataset(name):
    """Point RESULTS / MODELS_DIR / FHE_DIR at one dataset's results folder."""
    global DATASET, RESULTS, MODELS_DIR, FHE_DIR
    if name not in DATASETS:
        raise ValueError(f"unknown dataset '{name}' (choose from {', '.join(DATASETS)})")
    DATASET = name
    RESULTS = DATASETS[name]["results"]
    MODELS_DIR = RESULTS / "models"
    FHE_DIR = RESULTS / "fhe"


# ── Reproducibility & protocol ───────────────────────────────────────────
SEED = 42
N_FOLDS = 5          # outer stratified folds (synopsis §5, §6.4)
N_INNER_FOLDS = 5    # hyperparameter search inside each training fold

# ── Heart Disease feature groups (synopsis §5) ───────────────────────────
HD_CONTINUOUS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
HD_BINARY = ["sex", "fbs", "exang"]                   # already 0/1: passthrough
HD_NOMINAL = ["cp", "restecg", "slope", "ca", "thal"]  # one-hot, not ordinal

# Codes that are missing values in the original UCI Cleveland data but were
# written as real categories in this Kaggle export.
HD_INVALID_CODES = {"ca": [4], "thal": [0]}

# Human-readable descriptions shown in the web app. Nominal codes are left as
# raw codes: this Kaggle export re-numbers them and we don't guess meanings.
HD_FEATURE_INFO = {
    "age":      {"label": "Age", "unit": "years"},
    "sex":      {"label": "Sex", "values": {0: "Female", 1: "Male"}},
    "cp":       {"label": "Chest pain type", "unit": "code"},
    "trestbps": {"label": "Resting blood pressure", "unit": "mmHg"},
    "chol":     {"label": "Serum cholesterol", "unit": "mg/dl"},
    "fbs":      {"label": "Fasting blood sugar > 120 mg/dl", "values": {0: "No", 1: "Yes"}},
    "restecg":  {"label": "Resting ECG", "unit": "code"},
    "thalach":  {"label": "Max heart rate achieved", "unit": "bpm"},
    "exang":    {"label": "Exercise-induced angina", "values": {0: "No", 1: "Yes"}},
    "oldpeak":  {"label": "ST depression (exercise vs rest)", "unit": "mm"},
    "slope":    {"label": "Slope of peak exercise ST", "unit": "code"},
    "ca":       {"label": "Major vessels coloured", "unit": "code"},
    "thal":     {"label": "Thalassemia", "unit": "code"},
}

# ── Models (order = complexity ladder) ───────────────────────────────────
MODEL_KEYS = ["logreg", "xgboost", "mlp"]
MODEL_NAMES = {"logreg": "Logistic Regression", "xgboost": "XGBoost", "mlp": "MLP"}

# Search spaces follow synopsis §6.1 / §6.2. The MLP uses the fixed §6.3 design.
LOGREG_GRID = {"model__C": [0.01, 0.1, 1.0, 10.0]}
XGB_GRID = {
    "model__n_estimators": [100, 300],
    "model__max_depth": [3, 5, 8],
    "model__learning_rate": [0.05, 0.2],
}
MLP_CONFIG = {
    "hidden": (16, 8),
    "dropout": 0.3,
    "lr": 1e-3,
    "batch_size": 32,
    "max_epochs": 300,
    "patience": 25,
    "val_fraction": 0.15,
}

# ── FHE ──────────────────────────────────────────────────────────────────
# Quantisation per model. Each is the cheapest setting in the quantisation
# study (FHE_QUANT_STUDY) that keeps every plaintext decision unchanged.
FHE_SETTINGS = {
    "logreg":  {"n_bits": 8},
    "xgboost": {"n_bits": 7},
    "mlp":     {"n_bits": 7, "rounding_threshold_bits": 6},
}
# Rows per fold timed with genuine encrypted execution. Every row is also run
# through FHE simulation (the circuit's exact integer arithmetic), which is
# how the 303 encrypted predictions are obtained in reasonable time.
FHE_REAL_EXEC_SAMPLES = {"logreg": 3, "xgboost": 1, "mlp": 1}

# Settings compared in the quantisation study (simulation only).
FHE_QUANT_STUDY = {
    "xgboost": [{"n_bits": 5}, {"n_bits": 6}, {"n_bits": 7}],
    "mlp": [{"n_bits": 6}, {"n_bits": 7, "rounding_threshold_bits": 6},
            {"n_bits": 8, "rounding_threshold_bits": 6}],
}

# ── Demo scenarios (Encrypted Inference page) ────────────────────────────
# Each scenario loads a *real* patient, chosen by the logistic regression's
# out-of-fold p(disease): the patient in the band closest to `target` whose
# encrypted and plaintext decisions agree. No records are invented.
SCENARIOS = [
    {"key": "no_disease", "title": "No disease", "tone": "healthy",
     "band": (0.0, 0.15), "target": 0.05,
     "blurb": "Model is confident the patient is healthy"},
    {"key": "probable", "title": "Probable disease", "tone": "probable",
     "band": (0.55, 0.75), "target": 0.65,
     "blurb": "Model leans towards disease but is not certain"},
    {"key": "disease", "title": "Disease", "tone": "disease",
     "band": (0.85, 1.0), "target": 0.95,
     "blurb": "Model is confident the patient has disease"},
]
