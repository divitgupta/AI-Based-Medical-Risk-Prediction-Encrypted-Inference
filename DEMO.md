# Demo walkthrough: Working Prototype · Phase 1 + Phase 2

## Before you start (1 minute)

```bash
lsof -ti:5001 | xargs kill -9 2>/dev/null; ./scripts/run_app.sh
```

Wait for these lines in the terminal:

```
[cleveland] loaded 3 models × 5 folds (303 rows); artifacts consistent with predictions.json
[base1025_patient] loaded 3 models × 5 folds (1025 rows); …
[base1025_row] loaded 3 models × 5 folds (1025 rows); …
[base paper table] our Cleveland models on all 1025 rows (not retrained, nothing saved) …
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
> made without the server ever seeing the patient's data? The site has four
> tabs, in the order of this demo.

The **Dataset** switch at the top right flips every tab between three
datasets: our Cleveland file (303 patients), and the base paper's own file
(1,025 rows) with every model **retrained** on it, split *by patient*
(honest) or *by row* (copies leak, inflated).
Demo on Cleveland; switch at the end of tab 4 (see below). Grey
**More details** bars at the bottom of each tab hold the extra charts. Leave
them closed unless someone asks.

## 1 · Overview & Data (2 min)

- Stat cards: **303** patients, **14** attributes (13 features + target),
  **3** models, **5** CV folds.
- Walk the pipeline strip left to right: Data → Clean → Preprocess → 5-fold CV
  → Models → Evaluate → FHE.
- **Class balance:** 138 disease / 165 no disease, mildly imbalanced. Hence
  stratified folds and class weighting.
- **Labels are inverted (strongest viva point):**
  > *"The Kaggle CSV has its labels inverted: target = 1 means no disease.
  > Every clinical sign in this table agrees, and 165 ones matches the 164
  > healthy patients in the original UCI data. Without this fix, 'recall'
  > would measure how well we find healthy people."*
- *If asked:* **More data details** has the feature histograms, feature types
  (one-hot for chest pain type, a category not a number), the correlation
  matrix and the invalid codes.

## 2 · Results (3 min), the core of the demo

- **Leaderboard:** all four models side by side, with encrypted time.
  - LR: 85.1% accuracy, 81.2% recall, 0.915 AUC, **12 ms** encrypted
  - MLP: 83.5% / 81.9% / 0.901, 5 s
  - XGBoost: 80.2% / 75.4% / 0.873, 50 s
  - Elastic Learner (combines the three): 83.5% / 81.2% / 0.906, 55 s
- **Recommended model card:** LR, MLP and the Elastic Learner tie on F2
  (within 0.01), so the cheapest to encrypt wins: **logistic regression**.
- **How the combined model works:** an elastic-net logistic regression learns
  how much to trust each model; the L1 part can switch one off.
  > *"Even a learner built to combine all three can't beat the simplest model."*
- **ROC curves:** LR and MLP nearly overlap; XGBoost sits below.
- **Confusion matrices:** read the *missed disease* cell. LR 26, MLP 25,
  XGBoost 34.
- **Complexity ladder:** LR has 26 parameters, the MLP 561, XGBoost up to
  6,610 tree nodes.
  > *"More complexity bought nothing on 303 patients. The simplest model is
  > the best or tied."* This answers the synopsis' central question.
- *If asked:* **More details** has mean ± std per fold (±3–6 points, bigger
  than the gaps between models), the hyperparameters chosen in each fold, the
  MLP training curve and the Elastic Learner's weights per fold.

## 3 · Encrypted Demo (3 min), the "wow" moment

Use the three **Quick scenarios** chips in the Patient card. Each one loads a
real patient (chosen by the selected model's own out-of-fold probability) and
runs the encrypted inference immediately. The chips change with the model:
every model, including the Elastic Learner, has its own *No disease /
Probable disease / Disease* patients. The numbers below are for Logistic
Regression.

1. **No disease** (patient #156, p = 0.05). Walk the three steps:
   - **Client** encodes and encrypts (~3–7 ms).
   - **Server** computes on ciphertext (~10–20 ms).
   - **Client** decrypts (~1 ms).

   The dark box shows ~864 bytes of ciphertext, which is everything the server
   receives. The fingerprint changes on every run because encryption is
   randomised. Plaintext and FHE agree: ✓ No Disease.
2. **Disease** (patient #211, p = 0.95). Same flow, with the opposite,
   confident answer.
3. **Probable disease** (patient #210, p = 0.65). Read the amber note:
   > *"Of the 42 patients the model places between 0.55 and 0.75, 76 %
   > actually have disease. So 'probable' means about 3 in 4."*
4. Bottom card: **302 / 303** identical decisions. Click **patient #85**:
   > *"The one disagreement. Plaintext says 0.533, just over the threshold;
   > 8-bit quantisation gives 0.483. A borderline case, and we report it."*
5. Switch the model to **MLP** (~5 s, plus ~8 s the first time) and click
   its *Probable disease* chip (patient #273, p = 0.65: of the 25 patients the
   MLP puts there, only 56% have disease, less certain than LR). Then
   **XGBoost** (~30–60 s; start it and keep talking).
   > *"Encryption works for all three models. What changes is the price."*
   **Elastic Learner** runs all three encrypted circuits (~40–55 s) and shows
   how each model's decrypted answer was combined on the client.

## 4 · Cost & Base Paper (3 min), Phase 2

- **Accuracy table:** every model keeps its accuracy encrypted.
- **Cost table and the two log-scale charts:** LR 12 ms and no keys; MLP
  5 s and 2.6 GB of keys; XGBoost 50 s and 38,000 bootstraps per patient.
  > *"Accuracy didn't grow with complexity, but encryption cost grew by more
  > than three orders of magnitude. On this data the simple model wins twice."*
- **Base paper table:**
  > *"The base paper's file has 1,025 rows but only 302 patients, ours,
  > repeated. We retrained everything on it twice. Split by row, 99% of test
  > rows have a copy in training and XGBoost scores 99%. Split by patient it
  > drops to 79%. Duplicates make complex models look far better than they
  > are."*
- **Flip the Dataset switch to "Base paper · by patient"** and click back
  through tabs 2 and 3:
  > *"Retrained on the paper's own data, our LR gets 85.7% against their 85%,
  > and the encrypted version keeps every decision, where theirs lost a point.
  > The simplest model still wins."*
- **Then "Base paper · by row"**: the amber banner warns the scores are
  inflated. On tab 3, pick any row: the note under the patient says how many
  copies of that patient were in the model's training data.
- *If asked:* **More encryption details** has the quantisation study (fewer
  bits are cheaper but drift further) and ReLU vs x² (what CKKS needs; no
  cheaper under TFHE).

## Closing line

> All three models now run under encryption with no loss of accuracy. The
> cost of that privacy rises from milliseconds to nearly a minute per patient
> as models get more complex, while accuracy does not rise at all. For this
> problem, the simplest model is both the most accurate and the cheapest to
> keep private.

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

**Why didn't you use x² for the MLP, as the synopsis planned?**
x² is needed by schemes like CKKS that can only add and multiply. Concrete ML
uses TFHE, which computes ReLU exactly as a table lookup during bootstrapping.
We tested x² anyway: it costs the same (250 vs 246 bootstraps) and is slightly
less accurate (82.5% vs 83.5%), so ReLU stays.

**Why does logistic regression need zero bootstraps?**
Only its weighted sum is computed encrypted, which needs additions and
multiplications by plaintext weights, no bootstrapping. The client applies
the sigmoid after decrypting the score.

**Why is XGBoost so slow encrypted?**
Every comparison in every tree is a programmable bootstrap: about 38,000 per
patient on average (20,000–66,000 depending on the fold's tree count). Each
takes about a millisecond on a CPU.

**Are the 303 encrypted predictions real encryption?**
They come from FHE simulation, which runs the circuit's exact integer
arithmetic without the cryptography. Genuine encrypted runs on sample patients
in every fold matched the simulation exactly, and the website runs genuine
encryption live.

**What is the Elastic Learner, and is it leakage-free?**
Stacking with an elastic-net logistic regression as the meta-learner. Inside
each outer fold, the base models are refitted on inner folds of the training
rows, so the meta-learner learns only from predictions on patients those
models didn't see. Test patients are never used to fit anything that scores them.

**How did you get the base paper's dataset?**
Downloaded from Kaggle (`johnsmith88/heart-disease-dataset`), the source the
paper cites. All 1,025 rows are copies of our 302 distinct Cleveland records.

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
