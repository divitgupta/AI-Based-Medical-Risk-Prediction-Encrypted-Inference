# AI-Based Medical Risk Prediction & Encrypted Inference

**Working Prototype · Phase 1**: diagnostic classification on the Heart
Disease (Cleveland) dataset. Three models of increasing complexity, rigorous
cross-validation, and a working **Fully Homomorphic Encryption (FHE)** path:
the server returns a diagnosis without ever seeing the patient's data.

The project specification is in [`docs/synopsis/`](docs/synopsis/).
[DEMO.md](DEMO.md) is the presentation walkthrough.

## Run it

```bash
./scripts/run_experiments.sh   # train + evaluate + compile FHE circuits (~15 s)
./scripts/run_app.sh           # dashboard at http://localhost:5001
~/aimlel-venv/bin/python -m pytest tests/   # 16 tests
```

Both scripts use `~/aimlel-venv/bin/python`. Concrete ML calls the system
linker without quoting its arguments, which breaks under this project's path
(`…/3rd year/AIML EL`). [`scripts/setup_fhe_env.sh`](scripts/setup_fhe_env.sh)
creates that space-free interpreter.

## What Phase 1 covers

| Stage | Implementation |
|---|---|
| Data | 303 patients, 13 clinical features + target. Labels corrected (below). |
| Preprocessing | continuous → `StandardScaler`, binary → passthrough, nominal → impute + `OneHotEncoder` (13 → 25 features) |
| Protocol | stratified **5-fold CV**. Hyperparameters tuned by grid search **inside each training fold** (inner 5-fold). |
| Models | Logistic Regression · XGBoost · MLP (`25 → 16 → 8 → 1`) |
| Selection | **F2**, which weights recall twice as heavily as precision. It is our Phase-1 model-selection metric, consistent with the synopsis' emphasis on sensitivity. |
| Evaluation | accuracy, precision, **recall**, F1, ROC-AUC, confusion matrix, ROC curve. Reported both pooled out-of-fold and as mean ± std. |
| FHE | the trained LR of each fold is compiled with Concrete ML (8-bit) and served client → server → client |

Every transformer sits inside a scikit-learn `Pipeline`, so it is refitted on
each training fold. No statistic ever comes from held-out patients. Every
patient is predicted exactly once, by the fold model that never saw them.

## Results (pooled out-of-fold, 303 patients, positive = Disease)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Complexity |
|---|---|---|---|---|---|---|
| Logistic Regression | **85.15%** ± 6.0 | **85.50%** | 81.16% | **83.27%** | **0.915** | 26 parameters |
| XGBoost | 80.20% ± 3.5 | 80.00% | 75.36% | 77.61% | 0.873 | 100–300 trees, depth 3–5, 1,896–6,610 nodes |
| MLP | 83.50% ± 5.5 | 81.88% | **81.88%** | 81.88% | 0.901 | 561 parameters |

(± = std of accuracy across the 5 folds.) On 303 patients, added model
complexity does not buy accuracy. The linear model is best or tied on every
metric except recall, where the MLP is +0.7 points (one patient). Fold-to-fold
spread (±3–6 points) is larger than the gaps between models.

**Encrypted logistic regression:**
- **302 / 303** decisions identical to plaintext.
- The one disagreement (patient #85, p = 0.533 → 0.483) is a borderline case
  where 8-bit quantisation crosses the 0.5 threshold.
- Largest probability shift 0.058; ~15 ms per patient of genuine encrypted
  execution after warm-up.
- Encrypted accuracy is 85.48%. The flipped patient happened to become
  correct, which is chance, not a gain from encryption.

Results are exactly reproducible: all runs are seeded, and two runs give
byte-identical metrics and predictions apart from timings.

## Data corrections

1. **Inverted labels.** In this Kaggle export (`ronitf/heart-disease-uci`),
   `target = 1` means *no* disease. The `target = 0` group has more
   exercise angina, more ST depression, more blocked vessels, a lower max heart
   rate, and is older and more male. 165 ones also match the 164 *absence* cases
   of the original UCI data. We use `disease = 1 − target`, so recall means
   sensitivity to disease.
2. **Invalid codes.** `ca = 4` (5 rows) and `thal = 0` (2 rows) don't exist in
   the UCI codebook; they encode missing values. They become NaN and are imputed
   inside each training fold, so all 303 records are kept.

## Layout

```
healthguard/     ML package: config, data, models, mlp, evaluate, eda, fhe, experiment
webapp/          Flask server + single-page dashboard (vanilla JS, SVG charts, no CDN)
results/         generated: metrics.json, predictions.json, eda.json,
                 training_curves.json, models/, fhe/
scripts/         run_experiments.sh, run_app.sh, setup_fhe_env.sh
tests/           pipeline correctness + API tests
archive/v1-…/    the Phase-0 implementation, preserved unchanged
```

The web app only loads artifacts. It never trains, compiles or re-implements
preprocessing.

## Phase 2

- Breast Cancer Wisconsin full experiment (dataset already integrated).
- XGBoost and MLP under FHE. For the MLP, the x² activation per CryptoNets is
  under feasibility testing.
- Memory, key size and programmable-bootstrap cost per model. This leads to
  the complexity vs accuracy vs encrypted-cost trade-off analysis in the
  synopsis (§8).
