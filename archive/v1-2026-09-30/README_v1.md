# AI Based Medical Risk Prediction & Encrypted Inference

Privacy-preserving diagnostic ML: models are trained normally, then compiled to
Fully Homomorphic Encryption circuits so the inference server can produce a
diagnosis **without ever decrypting the patient record**.

👉 **Presenting this? Read [DEMO.md](DEMO.md).**

## Quick Start

```bash
# One-time: build the FHE-capable interpreter (see "Known issue" below)
./tools/setup_fhe_env.sh

# Run the full pipeline (EDA → Training → FHE benchmark → Summary), ~3 min
./run_pipeline.sh

# Launch the demo dashboard at http://localhost:5001
./run_demo.sh
```

## Known issue: spaces in the project path

Concrete ML invokes the system linker to build each FHE circuit, and that
command is assembled without quoting. Because this project lives under
`/Users/…/3rd year/AIML EL`, the space truncates the library search path and
compilation fails with `ld: library 'ConcretelangRuntime' not found`.

`tools/setup_fhe_env.sh` mirrors the project venv to `~/aimlel-venv` (a
space-free path) and verifies that a circuit compiles and executes. The project
directory itself can stay where it is. The two run scripts pick that
interpreter up automatically and fall back to plaintext-only if it is missing.

## Project Structure

```
AIML EL/
├── app.py                   # Flask demo dashboard (plaintext vs encrypted)
├── main.py                  # End-to-end pipeline entry point
├── run_demo.sh              # Launch the dashboard
├── run_pipeline.sh          # Re-run training + FHE benchmarks
├── tools/setup_fhe_env.sh   # Build the space-free venv
├── DEMO.md                  # Presentation runbook
├── data/raw/                # breast_cancer.csv, heart_disease.csv
├── src/
│   ├── config.py            # Paths, hyperparams, constants
│   ├── eda.py               # Exploratory Data Analysis
│   ├── data/loader.py       # Loading & preprocessing
│   ├── models/trainers.py   # LogReg, XGBoost, MLP training
│   ├── fhe/encrypted_inference.py   # Concrete ML compilation + benchmark
│   └── utils/evaluation.py  # Metrics, confusion matrices, ROC curves
├── results/
│   ├── plots/               # 36 plots (EDA, confusion matrices, ROC)
│   ├── models/              # Saved model files
│   ├── fhe_deployment/      # Compiled circuits + client keys (generated)
│   ├── summary_results.csv  # Plaintext comparison table
│   ├── fhe_comparison.csv   # Plaintext vs encrypted accuracy & latency
│   └── all_metrics.json     # Full metrics dump (incl. FHE block)
├── templates/, static/      # Dashboard frontend
└── docs/                    # Reference papers, synopsis
```

## Datasets
- **Breast Cancer Wisconsin (Diagnostic):** 569 samples, 30 features
- **Heart Disease UCI (Cleveland):** 303 samples, 13 features → 30 after encoding

## Models
1. **Logistic Regression** – linear baseline, grid-searched `C`
2. **XGBoost** – gradient-boosted tree ensemble
3. **MLP** – 2-hidden-layer PyTorch network with early stopping

Grid search optimises **recall**, since a false negative is the expensive error
in a diagnostic setting.

## Results

Plaintext (`results/summary_results.csv`):

| Dataset | Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| Breast Cancer | LogisticRegression | 94.74% | 94.59% | 97.22% | 95.89% | 0.9947 |
| Breast Cancer | XGBoost | 94.74% | 94.59% | 97.22% | 95.89% | 0.9907 |
| Breast Cancer | MLP | 95.61% | 98.55% | 94.44% | 96.45% | 0.9921 |
| Heart Disease | LogisticRegression | 83.61% | 82.86% | 87.88% | 85.29% | 0.8994 |
| Heart Disease | XGBoost | 77.05% | 75.68% | 84.85% | 80.00% | 0.8680 |
| Heart Disease | MLP | 80.33% | 78.38% | 87.88% | 82.86% | 0.8831 |

All runs are seeded (`RANDOM_SEED = 42`), including PyTorch weight
initialisation, dropout and DataLoader shuffling, so these numbers reproduce
exactly on re-run.

Both loaders split **before** fitting the preprocessor, so the scaler's
mean/std and the encoder's category set are learned from training rows only —
no test-set statistics leak into training.

Encrypted (`results/fhe_comparison.csv`) — **encrypted predictions agreed with
plaintext on 100% of test samples in all four configurations**:

| Dataset | Model | Plaintext acc | Encrypted acc | Agreement | Real FHE latency |
|---|---|---|---|---|---|
| Breast Cancer | logreg | 97.37% | 97.37% | 100% | 109 ms/patient |
| Breast Cancer | xgboost | 93.86% | 93.86% | 100% | 4.87 s/patient |
| Heart Disease | logreg | 86.89% | 86.89% | 100% | 124 ms/patient |
| Heart Disease | xgboost | 78.69% | 78.69% | 100% | 7.04 s/patient |

(Latencies are wall-clock on an M-series MacBook Air and vary by ±20% run to
run; the accuracy and agreement figures are exact.)

## The dashboard

`run_demo.sh` serves a four-stage walkthrough of the whole pipeline:

1. **Data & EDA** — sample counts, class balance, preprocessing pipeline, and
   the generated EDA plots (distributions, correlation matrices).
2. **Models & Training** — the cross-validation protocol, the hyperparameters
   grid search actually selected for each model, and the learned coefficients /
   feature importances.
3. **Evaluation** — a metric comparison chart, the full benchmark table, and
   confusion matrices plus ROC curves for all seven model variants.
4. **Encrypted Inference** — the live client → server → client FHE flow, with a
   per-prediction feature-attribution breakdown.

## How the encrypted path works

`app.py` drives the real Concrete ML client/server deployment flow:

1. **Client** quantises and encrypts the record → ~984 bytes of ciphertext.
2. **Server** evaluates the compiled circuit on that ciphertext, holding only
   the public evaluation key. It never sees a plaintext feature.
3. **Client** decrypts the returned ciphertext into class probabilities.

The dashboard times each stage separately and shows the raw ciphertext the
server receives.

## Evaluation Metrics
Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix.

## Next steps
- Encrypted MLP inference via quantisation-aware training (Brevitas).
- Split client and server into separate processes over a network boundary.
- Characterise FHE latency scaling against feature count and model depth.
