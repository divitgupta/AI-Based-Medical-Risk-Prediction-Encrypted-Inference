"""
Single source of configuration: paths, seeds, feature groups, search spaces.
Every other module imports from here.
"""

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
HEART_CSV = DATA_RAW / "heart_disease.csv"
BREAST_CSV = DATA_RAW / "breast_cancer.csv"

RESULTS = ROOT / "results"
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
FHE_N_BITS = 8
FHE_REAL_EXEC_SAMPLES = 3     # rows per fold timed with genuine encrypted execution
ENABLE_XGB_FHE = False        # P3: only switched on if it compiles first try

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
