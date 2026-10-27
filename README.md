# AI-Based Medical Risk Prediction & Encrypted Inference

**Working Prototype · Phase 1 + Phase 2.** Can a model diagnose heart disease
without the server ever seeing the patient's data, and what does that privacy
cost as the model gets more complex?

This project trains three models of increasing complexity on the Heart Disease
(Cleveland) dataset, compares them with rigorous cross-validation, and serves
predictions from **all three** through **Fully Homomorphic Encryption (FHE)**:
the server computes on encrypted data and returns an encrypted answer that only
the patient can decrypt. Phase 2 measures what encryption costs per model and
re-evaluates the base paper on its own dataset.

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
2. **Can a server predict without seeing the data?** Each trained model is
   compiled to an FHE circuit with Concrete ML and served client → server →
   client.
3. **What does encryption cost as complexity grows?** (Phase 2) Accuracy kept,
   bootstraps, key and ciphertext sizes, time and memory, per model.

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
                                     Evaluate metrics         Compile all 3 → FHE
                                                              + cost studies
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
server object only ever receives ciphertext and the evaluation key. LR's five
fold circuits load at start-up; XGBoost and the MLP (about 1 GB and 2.6 GB of
evaluation keys per fold) load one fold at a time, on first use.

## Getting started

Requires **Python 3.11** (macOS or Linux).

```bash
# 1. Create the environment and install dependencies
python3.11 -m venv venv
./venv/bin/pip install -r requirements.txt

# 2. Create the FHE interpreter at ~/aimlel-venv (see note below)
./scripts/setup_fhe_env.sh

# 3. Train, evaluate, compile all three models to FHE, run the Phase-2
#    studies and the base-paper comparison (~10 min, mostly genuine encrypted
#    XGBoost runs). Optional: results/ is already committed.
./scripts/run_experiments.sh

# 4. Start the dashboard on http://localhost:5001
./scripts/run_app.sh

# Tests (23, ~20 s; one runs genuine encrypted MLP inference)
~/aimlel-venv/bin/python -m pytest tests/ -p no:warnings
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
| FHE | each fold's trained model compiled with Concrete ML (TFHE): LR 8-bit, XGBoost 7-bit, MLP 7-bit with 6-bit rounding |

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

### Encrypted models (Phase 1: LR · Phase 2: XGBoost, MLP)

| Model | Quantisation | Encrypted accuracy | Same decision as plaintext | Max prob. shift |
|---|---|---|---|---|
| Logistic Regression | 8-bit | 85.48% (plain 85.15%) | **302 / 303** | 0.058 |
| XGBoost | 7-bit | 80.20% (plain 80.20%) | **303 / 303** | 0.240 |
| MLP | 7-bit, rounding 6 | 83.50% (plain 83.50%) | **303 / 303** | 0.084 |

The one LR disagreement is patient #85: plaintext p = 0.533, encrypted
p = 0.483. Quantisation moved this borderline case across the 0.5 threshold;
the encrypted model happens to be right on this patient, so the 0.3-point
accuracy gain is chance, not an effect of encryption. Encrypted predictions
for all 303 patients come from FHE simulation (the circuit's exact integer
arithmetic); genuine encryption was run on sample patients in every fold and
matched the simulation exactly.

### What encryption costs (Phase 2)

Mean over the 5 fold circuits; genuine encrypted execution on an Apple-silicon
laptop CPU.

| Model | Bootstraps (PBS) / patient | Evaluation keys | Ciphertext in → out | Key generation | **Encrypted time / patient** | Peak extra memory |
|---|---|---|---|---|---|---|
| Logistic Regression | 0 | none | 600 B → 10 kB | < 1 ms | **12 ms** | < 1 MB |
| MLP | 246 | 2.57 GB | 600 B → 66 kB | 8.4 s | **5.0 s** | 4.6 GB |
| XGBoost | 37,820 (20,100–66,300) | 961 MB | 600 B → 3.6 MB | 0.8 s | **49.8 s** | 1.3 GB |

**Encryption cost grows by orders of magnitude with complexity, while
accuracy does not.** Logistic regression needs no bootstrapping: only its
weighted sum runs encrypted, and the client applies the sigmoid after
decrypting. Every ReLU of the MLP and every comparison in XGBoost's trees is a
programmable bootstrap, which is where the time goes. The MLP has fewer
bootstraps than XGBoost, but at 16 bits each one needs much larger keys.

**Quantisation study** (simulation, 303 patients). Fewer bits are cheaper but
move probabilities further from plaintext:

| Model | Setting | Same decision | Max prob. shift | PBS | Eval. keys |
|---|---|---|---|---|---|
| XGBoost | 5-bit | 296 / 303 | 0.544 | 30,300 | 1.08 GB |
| XGBoost | 6-bit | 297 / 303 | 0.340 | 34,060 | 0.96 GB |
| XGBoost | **7-bit (used)** | 303 / 303 | 0.240 | 37,820 | 0.96 GB |
| MLP | 6-bit | 303 / 303 | 0.314 | 25 | 1.12 GB |
| MLP | **7-bit, rounding 6 (used)** | 303 / 303 | 0.084 | 246 | 2.57 GB |
| MLP | 8-bit, rounding 6 | 302 / 303 | 0.063 | 296 | 4.59 GB |

**Activation study: ReLU vs x².** CKKS-style schemes (CryptoNets, the base
paper) can only add and multiply, so they need a polynomial activation such as
x². TFHE evaluates ReLU exactly as a table lookup, so the swap isn't needed
here, and it doesn't make the circuit cheaper:

| MLP activation | Accuracy | Recall | AUC | Encrypted accuracy | PBS |
|---|---|---|---|---|---|
| ReLU (used) | 83.5% | 81.9% | 0.901 | 83.5% | 246 |
| x² | 82.5% | 82.6% | 0.883 | 80.2% | 250 |

### Elastic Learner: combining the three models

A stacking meta-learner (`python -m healthguard.elastic`): a logistic
regression with an **elastic-net** penalty takes the three models'
predictions (as logits) and learns how much to trust each. The L1 part can
drop a model entirely; the L2 part keeps weights stable. It is leakage-free:
inside each outer fold, the base models are refitted on inner folds of the
training rows so the meta-learner only ever learns from out-of-fold
predictions, and its C and L1 ratio are tuned by inner CV on F2.

| Model | Accuracy | Recall | F2 | ROC-AUC | Encrypted time |
|---|---|---|---|---|---|
| **Logistic Regression** 👑 | **85.1%** | 81.2% | **0.820** | **0.915** | **12 ms** |
| MLP | 83.5% | **81.9%** | 0.819 | 0.901 | 5.0 s |
| Elastic Learner | 83.5% | 81.2% | 0.814 | 0.906 | 54.9 s (all three circuits) |
| XGBoost | 80.2% | 75.4% | 0.762 | 0.873 | 49.8 s |

**Recommendation rule:** highest F2; models within 0.01 F2 count as tied, and
the cheapest to run encrypted wins. LR, the MLP and the Elastic Learner tie,
so **logistic regression is recommended**. Combining the models doesn't beat
the best single one, and the learned weights swing from fold to fold. That's
the same conclusion from a third direction: on 303 patients, complexity
doesn't pay.

### The base paper on its own dataset (Phase 2)

The base paper used Kaggle `johnsmith88/heart-disease-dataset`
([`data/raw/heart_disease_base_paper_1025.csv`](data/raw/heart_disease_base_paper_1025.csv),
downloaded from Kaggle). It has 1,025 rows but only **302 distinct patients**,
and those are exactly the records of our Cleveland file (same values, same
inverted labels), each repeated 1–4 times. Running our full pipeline on it
(`python -m healthguard.base_paper`):

| Accuracy | Logistic Reg. | Encrypted LR | XGBoost | MLP | Test rows also in training |
|---|---|---|---|---|---|
| Base paper, as reported (CKKS) | 85% | 84% | — | — | split not described |
| Ours, split by **row** | 85.9% | 85.9% | 99.2% | 96.2% | 99% |
| Ours, split by **patient** | 83.5% | 83.5% | 78.1% | 81.0% | 0% |

- Our logistic regression matches the base paper's 85%, and our encrypted LR
  keeps every decision (100% agreement) where theirs lost about 1 point.
- With a row split, 99% of test rows have an identical copy in training, so
  flexible models memorise their test patients: XGBoost reaches 99.2%. Kept
  apart by patient, it falls to 78.1%. Any random row split on this file
  inflates accuracy, especially for complex models.
- Split correctly, the base paper's data gives the same answer as our main
  study: added complexity does not help.

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
healthguard/     ML package: config, data, models, mlp, evaluate, eda, fhe,
                 experiment (main study + Phase-2 FHE studies), elastic (stacking
                 + leaderboard), base_paper
webapp/          Flask server + single-page dashboard (vanilla JS, SVG charts, no CDN)
data/raw/        heart_disease.csv (303, main) · heart_disease_base_paper_1025.csv
results/         generated: metrics.json, predictions.json, eda.json, base_paper.json, elastic.json,
                 training_curves.json, models/, fhe/{logreg,xgboost,mlp}/
scripts/         setup_fhe_env.sh, run_experiments.sh, run_app.sh
tests/           pipeline-correctness and API tests
docs/            synopsis, papers, presentation
archive/v1-…/    the Phase-0 implementation, preserved unchanged
```

## Phase 2 · status

Done: XGBoost and the MLP compiled to FHE and served live; cost per model
(bootstraps, key and ciphertext sizes, time, memory); quantisation and
activation studies; the base paper re-evaluated on its own data.

Limits worth stating: timings are from one laptop CPU; encrypted predictions
for all 303 patients use FHE simulation, with genuine encryption checked on
sample patients per fold; client and server share one process in the demo.

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
