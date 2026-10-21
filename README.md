# AI-Based Medical Risk Prediction & Encrypted Inference

**Working Prototype · Phase 1.** Can a model diagnose heart disease without the
server ever seeing the patient's data?

This project trains three models of increasing complexity on the Heart Disease
(Cleveland) dataset, compares them with rigorous cross-validation, and serves
predictions through **Fully Homomorphic Encryption (FHE)**: the server computes
on encrypted data and returns an encrypted answer that only the patient can
decrypt.

| | |
|---|---|
| Project specification | [`docs/synopsis/`](docs/synopsis/) |
| Presentation (Phase 1) | [`docs/presentation/Phase1_Presentation.pptx`](docs/presentation/Phase1_Presentation.pptx) |
| Demo walkthrough + viva Q&A | [DEMO.md](DEMO.md) |
| Papers | [`docs/papers/`](docs/papers/) |

## Research questions

1. **Does a more complex model diagnose better?** Logistic regression, an MLP
   and XGBoost are compared on identical folds, scored with recall-weighted
   metrics because a missed disease case is the costly error.
2. **Can a server predict without seeing the data?** The trained logistic
   regression is compiled to an FHE circuit with Concrete ML and served
   client → server → client.

## Base paper

Naresh, V. S., Reddi, S. *Exploring the future of privacy-preserving heart
disease prediction: a fully homomorphic encryption-driven logistic regression
approach.* Journal of Big Data 12:52, 2025.
[doi:10.1186/s40537-025-01098-6](https://doi.org/10.1186/s40537-025-01098-6)

They show encrypted logistic regression (CKKS, TenSEAL) stays within 1–3% of
its plaintext accuracy on heart disease data. This project extends that in
three ways:

| Base paper | This project |
|---|---|
| CKKS scheme via TenSEAL | TFHE via Concrete ML, with 8-bit quantisation |
| Logistic regression vs. SVM | Three models of increasing complexity: LR, MLP, XGBoost |
| Reports the accuracy gap | Checks every patient: 302 / 303 identical decisions, and explains the one that differs |

## Architecture

```
 ① TRAINING  (offline, run once: ./scripts/run_experiments.sh)

  Data ──▶ Clean ──▶ Preprocess ──▶ 5-fold CV ──▶ 3 Models
  303 pts  fix labels  scale+one-hot  nested tuning  LR · MLP · XGBoost
                                                         │
                                          ┌──────────────┴──────────────┐
                                          ▼                             ▼
                                     Evaluate metrics            Compile LR → FHE
                                          └──────────────┬──────────────┘
                                                         ▼
                                                   results/ (saved)
 ─────────────────────────────────────────────────────────┼──────────────
 ② WEB APP  (live: ./scripts/run_app.sh)                  ▼

  Dashboard ──patient──▶ Client ──ciphertext──▶ Server
  (browser) ◀──result─── encrypt/decrypt ◀─encrypted─ computes on ciphertext
                         holds secret key   answer    no secret key
```

The web app only loads `results/`. It never trains, compiles or re-implements
preprocessing. In this prototype the client and server roles run in the same
Flask process as separate objects (`FHEModelClient` / `FHEModelServer`); the
server object only ever receives ciphertext and the evaluation key.

## Getting started

Requires **Python 3.11** (macOS or Linux).

```bash
# 1. Create the environment and install dependencies
python3.11 -m venv venv
./venv/bin/pip install -r requirements.txt

# 2. Create the FHE interpreter at ~/aimlel-venv (see note below)
./scripts/setup_fhe_env.sh

# 3. Train, evaluate and compile the FHE circuits (~15 s).
#    Optional: results/ is already committed.
./scripts/run_experiments.sh

# 4. Start the dashboard on http://localhost:5001
./scripts/run_app.sh

# Tests (16)
~/aimlel-venv/bin/python -m pytest tests/
```

**Why `~/aimlel-venv`?** Concrete ML calls the system linker without quoting
its arguments, so FHE compilation breaks when Python lives under a path that
contains spaces. `setup_fhe_env.sh` copies the venv to a space-free path and
checks that a circuit compiles and runs encrypted.

If port 5001 is already in use: `lsof -ti:5001 | xargs kill -9`

## Method

| Stage | Implementation |
|---|---|
| Data | 303 patients, 13 clinical features + target. Labels corrected (see below). |
| Preprocessing | continuous → `StandardScaler`; binary → passthrough; nominal → impute + `OneHotEncoder` (13 → 25 features) |
| Protocol | stratified **5-fold CV**; hyperparameters tuned by grid search **inside each training fold** (inner 5-fold) |
| Models | Logistic Regression · XGBoost · MLP (`25 → 16 → 8 → 1`, dropout, early stopping) |
| Selection | **F2**, which weights recall twice as heavily as precision |
| Evaluation | accuracy, precision, **recall**, F1, ROC-AUC, confusion matrix, ROC curve; pooled out-of-fold and mean ± std |
| FHE | each fold's trained LR compiled with Concrete ML (8-bit) |

Every transformer sits inside a scikit-learn `Pipeline`, so it is refitted on
each training fold and no statistic comes from held-out patients. Every
patient is predicted exactly once, by the fold model that never saw them.

## Results

Pooled out-of-fold, 303 patients, positive class = Disease.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Size |
|---|---|---|---|---|---|---|
| Logistic Regression | **85.15%** ± 6.0 | **85.50%** | 81.16% | **83.27%** | **0.915** | 26 parameters |
| MLP | 83.50% ± 5.5 | 81.88% | **81.88%** | 81.88% | 0.901 | 561 parameters |
| XGBoost | 80.20% ± 3.5 | 80.00% | 75.36% | 77.61% | 0.873 | up to 6,610 tree nodes |

± is the standard deviation of accuracy across the 5 folds. **Added
complexity does not buy accuracy on 303 patients.** Logistic regression is
best or tied on every metric except recall, where the MLP leads by one
patient. Fold-to-fold spread (±3–6 points) is larger than the gaps between
models.

**Encrypted logistic regression**

| | |
|---|---|
| Identical decisions to plaintext | **302 / 303** (99.7%) |
| Encrypted accuracy | 85.48% (plaintext 85.15%) |
| Largest probability shift | 0.058 |
| Time per patient (encrypted, after warm-up) | ~15 ms |
| Ciphertext sent to the server | 864 bytes |

The one disagreement is patient #85: plaintext p = 0.533, encrypted p = 0.483.
8-bit quantisation moved this borderline case across the 0.5 threshold. The
encrypted model happens to be right on this patient, so the 0.3-point
accuracy gain is chance, not an effect of encryption.

Results are reproducible: all runs are seeded, and two runs give identical
metrics and predictions apart from timings.

## Data corrections

1. **Inverted labels.** In this Kaggle export (`ronitf/heart-disease-uci`),
   `target = 1` means *no* disease. The `target = 0` group has more exercise
   angina (55% vs 14%), more ST depression, more blocked vessels and a lower
   max heart rate. 165 rows with `target = 1` also match the 164 *absence*
   cases in the original UCI data. We use `disease = 1 − target`, so recall
   means sensitivity to disease.
2. **Invalid codes.** `ca = 4` (5 rows) and `thal = 0` (2 rows) don't exist in
   the UCI codebook. They are treated as missing and imputed inside each
   training fold, so all 303 records are kept.

## Project layout

```
healthguard/     ML package: config, data, models, mlp, evaluate, eda, fhe, experiment
webapp/          Flask server + single-page dashboard (vanilla JS, SVG charts, no CDN)
results/         generated: metrics.json, predictions.json, eda.json,
                 training_curves.json, models/, fhe/
scripts/         setup_fhe_env.sh, run_experiments.sh, run_app.sh
tests/           pipeline-correctness and API tests
docs/            synopsis, papers, presentation
archive/v1-…/    the Phase-0 implementation, preserved unchanged
```

## Phase 2

- Encrypt XGBoost and the MLP. For the MLP, the x² activation from
  CryptoNets is under feasibility testing.
- Measure memory, key size and programmable-bootstrapping cost per model.
- Combine both into the complexity vs. accuracy vs. encrypted-cost trade-off
  described in the synopsis.

## Tech stack

Python 3.11 · scikit-learn · XGBoost · PyTorch · Concrete ML (Zama) · Flask ·
pandas · NumPy · pytest

## References

1. Detrano, R., et al. "Heart Disease." UCI Machine Learning Repository, 1988.
2. Naresh, V. S., Reddi, S. Journal of Big Data 12:52, 2025. *(base paper)*
3. Dziri, M., et al. "Privacy-Preserving Machine Learning for Heart Disease Detection Using Fully Homomorphic Encryption." TechRxiv preprint, 2025.
4. Čepová, L., et al. Scientific Reports 14:20218, 2024.
5. Dowlin, N., et al. "CryptoNets." ICML, 2016.
6. Chou, E., et al. "Faster CryptoNets." arXiv, 2018.
7. Mishra, P., et al. "Delphi." USENIX Security, 2020.
8. Chen, T., Guestrin, C. "XGBoost." KDD, 2016.
9. Pedregosa, F., et al. "Scikit-learn." JMLR, 2011.
10. Zama. Concrete ML documentation. https://docs.zama.ai/concrete-ml
