# Demo walkthrough: Working Prototype · Phase 1

## Before you start (1 minute)

```bash
lsof -ti:5001 | xargs kill -9 2>/dev/null; ./scripts/run_app.sh
```

Wait for these lines in the terminal:

```
Loaded 5 precompiled FHE bundles (no compilation) in 0.8s
Warmed up 5 FHE circuits in …
Serving on http://localhost:5001
```

Then open <http://localhost:5001>. Everything runs offline: no CDN, no web
fonts.

**Fallback:** if anything breaks, the Phase-0 version still runs:
`cd archive/v1-2026-09-30 && ~/aimlel-venv/bin/python app.py`.

---

## Opening line

> The project asks two questions from the synopsis. How does diagnostic
> performance change as model complexity increases? And can a prediction be
> made without the server ever seeing the patient's data? Phase 1 answers
> both on the Heart Disease dataset.

## 1 · Overview (1 min)

- Stat cards: **303** patients, **14** attributes (13 features + target),
  **3** models, **5** CV folds.
- Walk the pipeline strip left to right: Data → Clean → Preprocess → 5-fold CV
  → Models → Evaluate → FHE.
- Headline results: logistic regression is best on accuracy (85.1%) and AUC
  (0.915).
- Point at the **Project status** card. It is explicit about what is done and
  what is Phase 2, which shows the scope is under control.

## 2 · Dataset (2 min)

- **Class balance:** 138 disease / 165 no disease, mildly imbalanced. Hence
  stratified folds and class weighting.
- **Feature types:** 5 continuous (scaled), 3 binary (passthrough),
  5 nominal (one-hot). Say: *"chest pain type is a category, not a number,
  so it's one-hot encoded rather than treated as ordinal."* This is synopsis
  §5.
- **Histogram:** it defaults to *max heart rate*. The disease group sits
  visibly lower, as it should clinically.
- **Correlation matrix:** the bottom row is correlation with the disease
  label. Exercise angina (+0.44), ST depression (+0.43), vessels (+0.46),
  chest pain type (−0.43).
- **Data-quality corrections (lead with this, it's the strongest viva point):**
  > *"The Kaggle CSV has its labels inverted: target = 1 means no disease.
  > Every clinical sign in this table agrees, and 165 ones matches the 164
  > healthy patients in the original UCI data. Without this fix, 'recall'
  > would measure how well we find healthy people."*
- **Breast Cancer tab:** the dataset is integrated, with the full experiment
  in Phase 2. No model numbers are shown for it, on purpose.

## 3 · Model Comparison (3 min), the core of the demo

- **Metrics table:** recall is highlighted because a missed disease case is
  the costly error (synopsis §7).
  - LR: 85.15% accuracy, 81.2% recall, 0.915 AUC
  - XGBoost: 80.2% / 75.4% / 0.873
  - MLP: 83.5% / 81.9% / 0.901
- Flip the toggle to **Mean ± std**. Say: *"accuracy varies ±3–6 points
  across folds, more than the gap between the models. With 61 patients per
  fold, one patient moves accuracy by 1.6 points."* That's the honest
  reading of small-data results.
- **ROC curves:** LR and MLP nearly overlap; XGBoost sits below.
- **Confusion matrices:** read the *missed disease* cell. LR 26, MLP 25,
  XGBoost 34.
- **Complexity ladder:** LR has 26 parameters, the MLP 561, and XGBoost up to
  6,610 tree nodes.
  > *"More complexity bought nothing on 303 patients. The simplest model is
  > the best or tied."*

  This is the answer to the synopsis' central research question, for this
  dataset.
- **Hyperparameters per fold:** each fold tuned its own model, on its own
  training rows only. LR chose `C = 0.1` in all five folds, which is a stable
  choice.
- **MLP training curve:** validation loss flattens and early stopping kicks
  in. The weights from the best epoch are restored.

## 4 · Encrypted Inference (3 min), the "wow" moment

Use the three **Quick scenarios** chips in the Patient card. Each one loads a
real patient (chosen by the model's own out-of-fold probability) and runs the
encrypted inference immediately.

1. **No disease** (patient #156, p = 0.05). Walk the three steps:
   - **Client** encodes and encrypts (~3–7 ms).
   - **Server** computes on ciphertext (~10–20 ms).
   - **Client** decrypts (~1 ms).

   The dark box shows ~864 bytes of ciphertext, which is everything the server
   receives. The fingerprint changes on every run because encryption is
   randomised. Plaintext and FHE agree: ✓ No Disease.
2. **Disease** (patient #211, p = 0.95). Same flow, with the opposite,
   confident answer.
3. **Probable disease** (patient #210, p = 0.65). This is the one to talk
   about. Read the amber note:
   > *"Of the 42 patients the model places between 0.55 and 0.75, 76 %
   > actually have disease. So when the model says 'probable', it means about
   > 3 in 4. Its probabilities are honest, not just its yes/no decisions."*
4. Bottom card: **302 / 303** identical decisions across every patient.
   Click **patient #85** and run it:
   > *"This is the one disagreement. Plaintext says 0.533, just over the
   > threshold; 8-bit quantisation gives 0.483. It's a borderline case, and we
   > report it rather than claim 100%."*
5. Switch the model to **MLP**. The FHE card reads *Phase 2 · Not measured*.
   We never show numbers we haven't measured.

## Closing line

> Phase 1 delivers the complete ML comparison with a working encrypted path
> for the linear model. Phase 2 extends encryption to XGBoost and the MLP,
> measures memory and bootstrapping cost, and runs the same study on Breast
> Cancer.

---

## Viva Q&A

**Why is recall starred?**
Synopsis §7: a false negative (missed disease) costs more than a false alarm.
Recall = the fraction of actual disease cases we catch.

**What's F2 and why use it?**
F2 is the F-score with recall weighted twice as heavily as precision. It's
our Phase-1 model-selection metric, chosen to match the synopsis' emphasis on
sensitivity. The synopsis doesn't mandate F2 specifically. Selecting on raw
recall alone can pick degenerate models that flag everyone.

**How do you know there's no data leakage?**
Every step (scaler, imputer, one-hot encoder) sits inside a scikit-learn
`Pipeline` and is refitted on each training fold. Hyperparameter search runs
inside the training fold too. A test asserts that the fitted scaler's mean
equals the training rows' mean, not the full dataset's.

**Why 5-fold CV and not a train/test split?**
Synopsis §5 and §6.4. With 303 patients, a single 61-patient test set would
make results depend on which rows landed there. CV gives every patient one
out-of-sample prediction, and shows the spread.

**How did you find the label problem?**
By checking the data against clinical knowledge. Disease should mean more
angina, more ST depression, a lower max heart rate. The CSV's `target = 0`
group had all of those. The class counts matched the original UCI absence
counts.

**Why is the MLP not encrypted yet?**
The prototype first validates the complete encrypted pipeline with the linear
model. MLP encryption needs feasibility work on the non-linear activation
(the synopsis proposes x², per CryptoNets) and on circuit compilation. That's
Phase 2.

**Why isn't the MLP the best model, as the synopsis expected?**
303 patients is too few for a network to show an advantage. Reporting that
honestly is a result, not a failure. It's exactly the complexity-vs-performance
question the project set out to answer.

**Why did the encrypted model disagree once?**
FHE works on integers, so inputs and weights are quantised to 8 bits.
Patient #85 has plaintext p = 0.533, and quantisation shifts it to 0.483,
across the threshold. The largest shift over all 303 patients is 0.058.

**Is the encrypted accuracy (85.5%) better than plaintext (85.1%)?**
No. The one flipped patient happened to land on the correct side. That's
chance, not an effect of encryption.

**Are the scenario patients cherry-picked?**
They are chosen by a fixed rule, not by hand. For each band (below 0.15,
0.55–0.75, above 0.85) it takes the real patient whose out-of-fold probability
is closest to the band's centre and whose encrypted result matches plaintext
(the rule is in `config.SCENARIOS`). Ground truth is never used to pick them.

**Are results reproducible?**
Yes. Everything is seeded. Two runs give identical metrics and predictions,
apart from timings. Run `./scripts/run_experiments.sh` and see.
