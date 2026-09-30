# Demo Runbook — AI Based Medical Risk Prediction & Encrypted Inference

Everything below has been verified working on this machine. Read the
**Before you start** section the morning of the demo, then follow the script.

---

## Before you start (2 minutes)

```bash
lsof -ti:5001 | xargs kill -9 2>/dev/null; ./run_demo.sh
```

Wait for `✅ Server ready.` then open <http://localhost:5001>.

The header pill must read **"FHE circuit live — inference runs on real
ciphertext"** in green. If it is orange, see [Troubleshooting](#troubleshooting).

Optional: re-run the full pipeline beforehand if you want fresh console output
to show (takes ~3 minutes):

```bash
./run_pipeline.sh
```

---

## The one-sentence framing

> A hospital wants a diagnosis from a cloud model, but is not allowed to send
> patient data to the cloud. We trained the models, compiled them to Fully
> Homomorphic Encryption circuits, and the server now returns a correct
> diagnosis **without ever decrypting the patient record**.

---

## Demo script (~10 minutes)

The dashboard is organised as the four stages of the ML pipeline. Walk them in
order — it doubles as the structure of your presentation.

### Stage 1 — Data & EDA (2 min)

Open on the **Data & EDA** tab.

- **569 samples, 30 features, 0 rows dropped**, stratified 80/20 split.
- **Class balance bar**: 212 malignant / 357 benign — 63% majority. Say: *this
  is why we weight the classes during training rather than optimising raw
  accuracy.*
- **Preprocessing**: StandardScaler on all 30 continuous features. On the Heart
  Disease tab this becomes StandardScaler + OneHotEncoder, 13 raw features
  expanding to 30 after encoding — a concrete talking point about categorical
  handling.
- Click any EDA plot to enlarge. The **correlation matrix** is the one to dwell
  on: radius, perimeter and area are near-perfectly correlated, which is why a
  regularised linear model does so well here.

### Stage 2 — Models & Training (2 min)

- **Training protocol**: 5-fold stratified cross-validation, seed 42, models
  selected on **recall**. Say the rationale out loud: *a missed diagnosis costs
  far more than a false alarm, so we do not select on accuracy.*
- Three model cards. For each, the hyperparameters shown are the ones **grid
  search actually chose**, with the searched grid underneath:
  - **Logistic Regression** — `C = 0.01` (heavy regularisation won), `lbfgs`,
    `class_weight='balanced'`.
  - **XGBoost** — 300 trees, depth 3, lr 0.05, `scale_pos_weight` 0.596.
  - **MLP** — `30 → 16 → 8 → 1`, dropout 0.3, early stopping with patience 20.
- The **coefficient bars** under Logistic Regression are the payoff: the largest
  weights are `worst concave points`, `worst radius`, `worst perimeter` — all
  negative, i.e. pushing toward malignant. That matches the clinical literature,
  which is good evidence the model learned something real.
- On XGBoost the same panel shows **gain-based feature importance** instead, so
  you can compare what a linear model and a tree ensemble each rely on.

### Stage 3 — Evaluation (2 min)

- **Metric comparison chart**: five metrics × three models, bars scaled 0.50–1.00
  so differences are visible. Point out the trade-off on Breast Cancer: the MLP
  has the best precision (98.55%) but LogReg/XGBoost have better recall
  (97.22%) — *and recall is what we said we cared about.*
- **Benchmark summary table**: all six plaintext rows plus the FHE rows with
  real encrypted latency per patient.
- **Confusion matrices and ROC curves** for all seven variants, including the
  FHE plaintext/encrypted pairs. Click one to enlarge. The FHE pair is the
  visual proof: *the two confusion matrices are identical.*

### Stage 4 — Encrypted Inference (3 min) — the money shot

1. Check the status pill reads **"FHE circuit live"** in green.
2. Left panel compares plaintext vs FHE-circuit metrics — the encrypted model
   is **not** a downgrade (98.25% vs 94.74% accuracy on Breast Cancer; the
   quantised model is trained separately and happens to do slightly better).
3. Click **Load Random Test Patient** — a real held-out record, 30 clinical
   measurements. *This is the data the hospital is not allowed to leak.*
4. Click **Run Plaintext & Encrypted Inference** and walk the three stages:
   - **Client encrypts** — ~2–5 ms
   - **Server computes on ciphertext** — ~0.1–1 s
   - **Client decrypts** — ~1 ms
5. **"What the server actually receives"**: 984 bytes of raw hex. *This is the
   server's entire view of the patient.*
6. Both result cards: **identical prediction, identical confidence.**
7. Scroll to **"Why this prediction?"** — the per-feature contribution bars.
   Each bar is `wᵢ · xᵢ`, the signed push that feature gave to the log-odds.
   Say: *the model is not a black box — we can say exactly which measurements
   drove this diagnosis, and the encrypted path gives the same answer.*

Repeat once with a malignant case, then switch to the **Heart Disease** tab.

### Wrap-up (1 min)

> Encrypted predictions agreed with plaintext predictions on **100% of test
> samples**, for both models on both datasets.

| Dataset | Model | Plain acc | Encrypted acc | Agreement | Real FHE latency |
|---|---|---|---|---|---|
| Breast Cancer | LogReg | 97.37% | 97.37% | 100% | 109 ms/patient |
| Breast Cancer | XGBoost | 93.86% | 93.86% | 100% | 4.87 s/patient |
| Heart Disease | LogReg | 86.89% | 86.89% | 100% | 124 ms/patient |
| Heart Disease | XGBoost | 78.69% | 78.69% | 100% | 7.04 s/patient |

(Source: `results/fhe_comparison.csv`. Latencies vary ±20% run to run.)

Cost argument: FHE is ~1000× slower than plaintext inference, and the tree
ensemble is ~45× slower than the linear model. For a diagnosis a clinician
reads minutes later, 100 ms is irrelevant — that is why this is practical today.

**What's next:**
- Encrypted **MLP** inference via quantisation-aware training (Brevitas).
- Client and server as genuinely separate processes over a network boundary.
- Characterise how FHE latency scales with feature count and model depth.

## Questions you should expect

**"Why did you choose these three models?"**
They span the complexity ladder: a regularised linear baseline, a non-linear
tree ensemble, and a small neural network. That lets you show that on a 569-row
dataset the linear model is competitive — added capacity does not buy much when
the features are already highly discriminative and correlated.

**"How did you handle class imbalance?"**
`class_weight='balanced'` for Logistic Regression and `scale_pos_weight` for
XGBoost, both shown on the Models tab. Combined with selecting on recall rather
than accuracy, since 63% of the Breast Cancer set is benign.

**"Are your results reproducible?"**
Yes — seed 42 throughout: the train/test split, cross-validation folds, XGBoost,
PyTorch weight initialisation, dropout, and DataLoader shuffling. Re-running
`./run_pipeline.sh` reproduces every number in the tables exactly.

**"Did you check for data leakage?"**
Yes, and we found one. The Heart Disease loader originally fitted the
StandardScaler and OneHotEncoder on the *full* dataset before splitting, so
test-set statistics reached the models. Fixing it — split first, fit on the
training rows only — dropped Heart Disease LogReg accuracy from 85.25% to
83.61% and recall from 90.91% to 87.88%. The numbers in the tables are the
corrected ones. Breast Cancer was unaffected because that loader already split
first, which is a useful control: the fix moved only the dataset it should have.

A detail worth mentioning: plaintext XGBoost did **not** change (77.05% before
and after), because trees split on thresholds and are invariant to feature
scaling. Only the scale-sensitive models moved.

**"How do you know it is not overfitting?"**
Model selection is done by 5-fold stratified cross-validation on the training
split only; every number reported is on a held-out test set the models never
saw. The MLP additionally early-stops on an internal 90/10 validation split
with patience 20. The fact that grid search chose heavy regularisation
(`C = 0.01`) is itself evidence the data does not support a complex boundary.

**"Which features matter?"**
Two views of it. Globally, on the Models tab: coefficients for the linear model,
gain-based importances for XGBoost. Per patient, on the Inference tab: the
`wᵢ · xᵢ` contribution bars. Both point at the same handful of "worst"
measurements — concave points, radius, perimeter.

**"Why is the MLP not the best model?"**
303–569 samples is far too small for a neural network to show an advantage. It
wins on precision for Breast Cancer but loses on recall, which is the metric we
selected on. Reporting that honestly is more useful than tuning until the
neural network wins.

**"Is the server really not decrypting?"**
The `FHEModelServer` object in `app.py:_run_encrypted` receives only
`quantize_encrypt_serialize(X)` — serialized ciphertext — plus the public
evaluation key. The decryption key never leaves the client object. You can show
`app.py` lines around `_run_encrypted` to prove it.

**"Why is the plaintext confidence identical to the encrypted one?"**
Because both cards run the *same* compiled model; one path encrypts and one
does not. That is deliberate — it isolates the effect of FHE. The scikit-learn
baseline is printed as a footnote under the verdict so you can see the
quantised model tracks the original.

**"Why logistic regression and not the MLP?"**
FHE cost grows with circuit depth. Linear models compile to a single dot
product; tree ensembles cost ~40× more (see the table); neural networks need
quantisation-aware training. Starting linear was the right first step.

**"Isn't this just simulated?"**
No. `fhe="execute"` in `src/fhe/encrypted_inference.py` and the client/server
flow in `app.py` both run real homomorphic evaluation. The pipeline *also*
reports a faster `simulate` pass, used only to score the full test set quickly —
both numbers are in `results/fhe_comparison.csv`.

---

## Troubleshooting

**Status pill is orange / "FHE unavailable"**

Concrete ML shells out to the system linker, and the command is built without
quoting. This project lives under `/Users/…/3rd year/AIML EL` — the space
breaks the link step with `ld: library 'ConcretelangRuntime' not found`.

The fix is a copy of the venv on a space-free path:

```bash
./tools/setup_fhe_env.sh
```

`run_demo.sh` and `run_pipeline.sh` use `~/aimlel-venv/bin/python` automatically.
**Do not run `python app.py` from the project venv during the demo** — it will
fall back to plaintext-only.

**Port 5001 already in use**

```bash
lsof -ti:5001 | xargs kill -9
```

**Everything is broken and the demo is in 5 minutes**

The dashboard degrades gracefully: with FHE unavailable it still loads patients
and predicts in plaintext. Show `results/fhe_comparison.csv` and the plots in
`results/plots/` as evidence the encrypted pipeline ran, and present the
benchmark table instead of the live flow.

---

## Offline backup material

If the laptop fails entirely, these files carry the story on their own:

- `results/fhe_comparison.csv` — plaintext vs encrypted accuracy and latency
- `results/summary_results.csv` — all six plaintext model/dataset results
- `results/plots/` — 36 plots: EDA, confusion matrices, ROC curves for every
  model including the FHE variants
- `docs/synopsis/Project_Synopsis_HE_Diagnostic_ML.docx`

Take screenshots of a successful live run tonight as insurance.
