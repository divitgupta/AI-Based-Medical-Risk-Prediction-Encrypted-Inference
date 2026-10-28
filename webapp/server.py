"""
Web app for the prototype (Phase 1 + Phase 2).

Loads what the experiment produced — metrics, out-of-fold predictions, the
fitted fold pipelines and the precompiled FHE bundles — and never trains or
re-implements preprocessing. Run from the project root:

    ~/aimlel-venv/bin/python -m webapp.server        (or scripts/run_app.sh)

Every API call takes ?ds=<name> for one of config.DATASETS (default
cleveland). Each dataset has its own results folder: trained fold models,
FHE circuits, metrics and predictions.

  cleveland         our 303 patients.
  base1025_patient  models retrained on the base paper's 1,025-row file,
                    every split grouped by patient.
  base1025_row      the same file, split by row (copies leak; for comparison).

At start-up the app also runs our Cleveland models over all 1,025 rows of
the base paper's file (each row by the fold model that never saw that
patient) for one row of the base-paper comparison table. Nothing is saved.
"""

import hashlib
import json
import sys
import threading
import time
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, abort, jsonify, render_template, request

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from healthguard import config  # noqa: E402
from healthguard.data import load_base_paper, load_dataset, match_to_cleveland  # noqa: E402

app = Flask(__name__)
app.json.sort_keys = False   # keep the complexity-ladder order LR → XGB → MLP

# Logistic regression's keys are tiny, so all its fold circuits load at start-up.
# XGBoost and the MLP need ~1 GB and ~2.6 GB of evaluation keys per fold, so
# only one circuit of each (across all datasets) is kept in memory, loaded on
# first use.
EAGER_FHE = {"logreg"}
_FHE_LOCKS = {k: threading.Lock() for k in config.MODEL_KEYS}
_FHE_CACHE = {k: {} for k in config.MODEL_KEYS}     # key -> {(dataset, fold): EncryptedModel}

DATASETS = {}            # name -> Dataset, only those whose results exist
BASE_PAPER = {"json": None, "our_models": None}


class Dataset:
    """Everything the app serves for one dataset."""

    def __init__(self, name):
        self.name = name
        self.info = config.DATASETS[name]
        self.root = self.info["results"]
        self.models_dir = self.root / "models"
        self.fhe_dir = self.root / "fhe"
        self.metrics = self.predictions = self.eda = self.curves = None
        self.elastic = self.elastic_meta = None
        self.X = self.y = self.fold = self.patient = None
        self.pipelines = {}      # key -> [fold pipeline, ...]
        self.fhe_status = {}     # key -> {"ready": bool, "reason": str}
        self.scenarios = []

    def available(self):
        return all((self.root / f).exists() for f in
                   ("metrics.json", "predictions.json", "eda.json", "training_curves.json"))

    def bundle(self, key, k):
        return self.fhe_dir / key / f"fold{k}"

    @property
    def n(self):
        return len(self.y)


# ═════════════════════════════════════════════════════════════════════════
#  Start-up
# ═════════════════════════════════════════════════════════════════════════

def _read(path):
    return json.loads(path.read_text())


def load_state():
    DATASETS.clear()
    for cache in _FHE_CACHE.values():
        cache.clear()
    bp = config.RESULTS_ROOT / "base_paper.json"
    BASE_PAPER["json"] = _read(bp) if bp.exists() else None
    for name in config.DATASETS:
        S = Dataset(name)
        if not S.available():
            if name == config.DEFAULT_DATASET:
                sys.exit(f"Missing results in {S.root}. Run scripts/run_experiments.sh first.")
            print(f"  [info] {name}: no results in {S.root} — run "
                  f"python -m healthguard.experiment --dataset {name}")
            continue
        _load_dataset(S)
        DATASETS[name] = S
    BASE_PAPER["our_models"] = _our_models_on_base_file(DATASETS[config.DEFAULT_DATASET])


def _load_dataset(S):
    S.metrics = _read(S.root / "metrics.json")
    S.predictions = _read(S.root / "predictions.json")
    S.eda = _read(S.root / "eda.json")
    S.curves = _read(S.root / "training_curves.json")
    el = S.root / "elastic.json"
    S.elastic = _read(el) if el.exists() else None
    metas = [S.models_dir / "elastic" / f"fold{k}.joblib" for k in range(config.N_FOLDS)]
    if S.elastic is not None and all(m.exists() for m in metas):
        S.elastic_meta = [joblib.load(m) for m in metas]

    X, y, _ = load_dataset(S.name)
    preds = S.predictions
    if preds["y_true"] != y.tolist():
        sys.exit(f"{S.name}: predictions.json does not match the dataset — re-run the experiment.")
    S.X, S.y, S.fold = X, y, np.array(preds["fold"])
    S.patient = np.array(preds["patient"]) if "patient" in preds else None

    for key in config.MODEL_KEYS:
        S.pipelines[key] = [joblib.load(S.models_dir / key / f"fold{k}.joblib")
                            for k in range(config.N_FOLDS)]

    # Consistency guard: the loaded fold models must reproduce the stored
    # out-of-fold probabilities, otherwise the artifacts are stale.
    for key in config.MODEL_KEYS:
        for i in range(0, S.n, 37):
            live, stored = _plain_prob(S, key, i), preds["prob"][key][i]
            if abs(live - stored) > 1e-4:
                sys.exit(f"Stale artifacts ({S.name}): {key} patient {i} gives {live:.4f}, "
                         f"stored {stored:.4f}. Re-run the experiment.")
    if S.elastic_meta is not None and "prob_elastic" in preds:
        from healthguard.elastic import combine
        for i in range(0, S.n, 37):
            base = np.array([_plain_prob(S, m, i) for m in config.MODEL_KEYS])
            live = float(combine(S.elastic_meta[int(S.fold[i])], base)[0][0])
            if abs(live - preds["prob_elastic"][i]) > 1e-4:
                sys.exit(f"Stale Elastic Learner ({S.name}): patient {i} gives {live:.4f}, stored "
                         f"{preds['prob_elastic'][i]:.4f}. Re-run python -m healthguard.elastic "
                         f"--dataset {S.name}.")
    print(f"[{S.name}] loaded {len(config.MODEL_KEYS)} models × {config.N_FOLDS} folds ({S.n} rows); "
          f"artifacts consistent with predictions.json")

    S.scenarios = _build_scenarios(S)
    _load_fhe(S)


def _our_models_on_base_file(cl):
    """
    One row of the base-paper comparison table: our Cleveland models, not
    retrained, run over every row of the base paper's file. Row i is a copy
    of Cleveland patient pid[i], so it is scored by the fold model that did
    not train on that patient. Logistic regression also runs with genuine
    encryption on every row. Computed live; nothing is saved.
    """
    from healthguard.evaluate import metrics

    t0 = time.perf_counter()
    X, y, _ = load_base_paper()
    pid = match_to_cleveland()
    assert np.array_equal(cl.y[pid], y)
    fold, n = cl.fold[pid], len(y)
    prob = {k: np.zeros(n) for k in config.MODEL_KEYS}
    for k in config.MODEL_KEYS:
        for f in range(config.N_FOLDS):
            rows = np.flatnonzero(fold == f)
            prob[k][rows] = cl.pipelines[k][f].predict_proba(X.iloc[rows])[:, 1]
        # the same models, so each row gets its Cleveland patient's probability
        assert np.allclose(prob[k], np.array(cl.predictions["prob"][k])[pid], atol=1e-5)

    enc_lr = None
    if cl.fhe_status["logreg"]["ready"]:
        enc_lr = np.zeros(n)
        for f in range(config.N_FOLDS):
            rows = np.flatnonzero(fold == f)
            Xe = cl.pipelines["logreg"][f].named_steps["prep"].transform(X.iloc[rows])
            model = _encrypted_model(cl, "logreg", f)[0]
            for j, i in enumerate(rows):
                enc_lr[i] = model.predict(Xe[j:j + 1])[0]
    print(f"[base paper table] our Cleveland models on all {n} rows (not retrained, nothing saved), "
          f"incl. genuine encrypted LR, in {time.perf_counter() - t0:.1f}s")
    return {"n_rows": n,
            "models": {k: metrics(y, prob[k]) for k in config.MODEL_KEYS},
            "encrypted_logreg": None if enc_lr is None else metrics(y, enc_lr)}


def _build_scenarios(S):
    """Demo presets for every model (see config.SCENARIOS): {model: [scenario, ...]}."""
    preds = S.predictions
    sources = {k: (preds["prob"][k], preds["prob_fhe"].get(k)) for k in config.MODEL_KEYS}
    if "prob_elastic" in preds:
        sources["elastic"] = (preds["prob_elastic"], preds.get("prob_elastic_fhe"))
    return {k: _scenarios_for(S, plain, enc) for k, (plain, enc) in sources.items()}


def _scenarios_for(S, plain, enc):
    """
    One real patient per scenario band, picked by this model's own
    out-of-fold p(disease): the patient in the band closest to the band's
    target. Where the model's encrypted predictions are known, only patients
    whose encrypted and plaintext decisions agree are eligible.
    """
    p = np.asarray(plain)
    agree = np.ones(len(p), bool) if enc is None else (p >= 0.5) == (np.asarray(enc) >= 0.5)
    out = []
    for sc in config.SCENARIOS:
        lo, hi = sc["band"]
        in_band = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        candidates = np.flatnonzero(in_band & agree)
        if not len(candidates):
            continue
        i = int(candidates[np.argsort(np.abs(p[candidates] - sc["target"]), kind="stable")[0]])
        out.append({
            "key": sc["key"], "title": sc["title"], "tone": sc["tone"], "blurb": sc["blurb"],
            "band": [lo, hi], "patient_id": i, "fold": int(S.fold[i]) + 1,
            "p_disease": float(p[i]),
            # How often patients in this band truly have disease — tells the
            # audience what "probable" actually means for this model.
            "band_size": int(in_band.sum()),
            "band_disease_rate": float(S.y[in_band].mean()),
        })
    return out


def _load_fhe(S):
    for key in config.MODEL_KEYS:
        if not all((S.bundle(key, k) / "server.zip").exists() for k in range(config.N_FOLDS)):
            S.fhe_status[key] = {"ready": False,
                                 "reason": "FHE bundles missing — run scripts/run_experiments.sh"}
            print(f"  [warn] {S.name}/{key}: FHE bundles missing; encrypted inference disabled")
            continue
        S.fhe_status[key] = {"ready": True, "reason": ""}
        if key not in EAGER_FHE:
            continue

        t0 = time.perf_counter()
        try:
            for k in range(config.N_FOLDS):
                _encrypted_model(S, key, k)
        except Exception as e:  # never compile here — see note in healthguard/fhe.py
            S.fhe_status[key] = {"ready": False, "reason": f"FHE bundle failed to load: {e}"}
            print(f"  [warn] {S.name}/{key}: FHE bundle load failed: {e}")
            continue
        # Warm-up: the first encrypted call on each circuit pays a one-off
        # runtime initialisation (~0.5 s). Run it once per fold at start-up so
        # the timings shown during the demo are steady-state.
        for k in range(config.N_FOLDS):
            i = int(np.flatnonzero(S.fold == k)[0])
            prep = S.pipelines[key][k].named_steps["prep"]
            _FHE_CACHE[key][(S.name, k)].predict(prep.transform(S.X.iloc[[i]]))
        print(f"[{S.name}] loaded + warmed up {config.N_FOLDS} precompiled {key} FHE bundles "
              f"(no compilation) in {time.perf_counter() - t0:.2f}s")
    lazy = [k for k in config.MODEL_KEYS if k not in EAGER_FHE and S.fhe_status[k]["ready"]]
    if lazy:
        print(f"[{S.name}] FHE bundles for {', '.join(lazy)} load on first use (large evaluation keys)")


def _encrypted_model(S, key, k):
    """The EncryptedModel for one dataset's fold, loading it (and evicting
    other large ones) if needed. Returns (model, setup seconds spent now)."""
    from healthguard.fhe import OUTPUT_KIND, EncryptedModel

    cache = _FHE_CACHE[key]
    if (S.name, k) in cache:
        return cache[(S.name, k)], 0.0
    if key not in EAGER_FHE:
        cache.clear()        # free the previous circuit's keys first
    t0 = time.perf_counter()
    cache[(S.name, k)] = EncryptedModel(S.bundle(key, k), OUTPUT_KIND[key])
    return cache[(S.name, k)], time.perf_counter() - t0


# ═════════════════════════════════════════════════════════════════════════
#  Helpers
# ═════════════════════════════════════════════════════════════════════════

def _ds():
    name = request.args.get("ds", config.DEFAULT_DATASET)
    if name not in DATASETS:
        abort(404, f"unknown dataset '{name}'")
    return DATASETS[name]


def _plain_prob(S, key, i, k=None):
    pipe = S.pipelines[key][int(S.fold[i]) if k is None else k]
    return float(pipe.predict_proba(S.X.iloc[[i]])[0, 1])


def _disagreements(S, plain, enc):
    if enc is None:
        return []
    plain, enc = np.asarray(plain), np.asarray(enc)
    ids = np.flatnonzero((plain >= 0.5) != (enc >= 0.5))
    return [{"id": int(i), "fold": int(S.fold[i]) + 1,
             "plain": float(plain[i]), "enc": float(enc[i])} for i in ids]


def _elastic_overview(S):
    """Elastic Learner entry for the inference page, shaped like a base model's."""
    if S.elastic_meta is None or S.elastic is None:
        return None
    preds = S.predictions
    return {"name": "Elastic Learner", "fhe": S.elastic["fhe"], "status": _status(S, "elastic"),
            "disagreements": _disagreements(S, preds["prob_elastic"], preds.get("prob_elastic_fhe"))}


def _dataset_info(S):
    m = S.metrics
    return {"key": S.name, "label": S.info["label"], "short": S.info["short"], "split": S.info["split"],
            "n_rows": S.n, "n_distinct_patients": int(m.get("n_distinct_patients", S.n)),
            "base": S.patient is not None,                  # the base paper's file
            "grouped": S.name == "base1025_patient",        # split keeps a patient's copies together
            "leak_pct": m.get("test_rows_with_copy_in_training_pct")}


def _label(v):
    return "Disease" if v == 1 else "No Disease"


def _patient_record(S, i):
    row = S.X.iloc[i]
    record = []
    for col in S.X.columns:
        info = config.HD_FEATURE_INFO[col]
        v = row[col]
        if np.isnan(v):
            shown = "missing (imputed)"
        elif "values" in info:
            shown = info["values"].get(int(v), str(int(v)))
        elif info.get("unit") == "code":
            shown = f"code {int(v)}"
        else:
            num = int(v) if float(v).is_integer() else round(float(v), 1)
            shown = f"{num} {info['unit']}"
        record.append({"feature": col, "label": info["label"], "value": shown})
    return record


def _patient_id(S):
    body = request.get_json(silent=True) or {}
    try:
        i = int(body.get("patient_id"))
    except (TypeError, ValueError):
        abort(400, "patient_id must be an integer")
    if not 0 <= i < S.n:
        abort(400, f"patient_id must be between 0 and {S.n - 1}")
    return i


# ═════════════════════════════════════════════════════════════════════════
#  Routes
# ═════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/datasets")
def datasets():
    return jsonify({"default": config.DEFAULT_DATASET,
                    "datasets": [{**_dataset_info(S), "available": True} for S in DATASETS.values()]})


@app.route("/api/overview")
def overview():
    S = _ds()
    m, e = S.metrics, S.eda["heart"]
    return jsonify({
        "dataset": _dataset_info(S),
        "n_samples": S.n,
        "n_attributes": e["n_attributes"],
        "n_features": e["n_features"],
        "n_models": len(config.MODEL_KEYS),
        "n_folds": m["n_folds"],
        "model_order": config.MODEL_KEYS,
        "models": {k: {"name": v["name"], "pooled": v["pooled"], "summary": v["summary"]}
                   for k, v in m["models"].items()},
        "fhe": m["fhe"],
        "fhe_disagreements": {k: _disagreements(S, S.predictions["prob"][k], S.predictions["prob_fhe"].get(k))
                              for k in config.MODEL_KEYS},
        "elastic": _elastic_overview(S),
        "fhe_status": S.fhe_status,
        "selection_metric": m["selection_metric"],
        "meta": m["meta"],
    })


@app.route("/api/eda/<name>")
def eda(name):
    S = _ds()
    if name not in S.eda:
        abort(404)
    return jsonify(S.eda[name])


@app.route("/api/results")
def results():
    S = _ds()
    return jsonify({"metrics": S.metrics, "curves": S.curves, "model_order": config.MODEL_KEYS})


@app.route("/api/phase2")
def phase2():
    S = _ds()
    m = S.metrics
    return jsonify({"dataset": _dataset_info(S), "n_samples": S.n,
                    "study_n": m["fhe_study"].get("n_samples", S.n),
                    "model_order": config.MODEL_KEYS,
                    "models": {k: {"name": v["name"], "pooled": v["pooled"],
                                   "complexity": [f["complexity"] for f in v["fold_info"]]}
                               for k, v in m["models"].items()},
                    "fhe": m["fhe"], "study": m["fhe_study"],
                    "base_paper": BASE_PAPER["json"],
                    # our Cleveland models, not retrained, on all rows of the base paper's file
                    "our_models_on_base_file": BASE_PAPER["our_models"],
                    "base_paper_data": S.eda.get("base_paper")})


@app.route("/api/elastic")
def elastic():
    S = _ds()
    if S.elastic is None:
        abort(404, "Run python -m healthguard.elastic to build the Elastic Learner")
    return jsonify(S.elastic)


@app.route("/api/patients")
def patients():
    S = _ds()
    if S.patient is None:
        return jsonify([{"id": i, "fold": int(f) + 1} for i, f in enumerate(S.fold)])
    # base1025: which of our patients each row is a copy of.
    return jsonify([{"id": i, "fold": int(f) + 1, "patient": int(S.patient[i])}
                    for i, f in enumerate(S.fold)])


@app.route("/api/scenarios")
def scenarios():
    return jsonify(_ds().scenarios)


@app.route("/api/patient/<int:i>")
def patient(i):
    S = _ds()
    if not 0 <= i < S.n:
        abort(404)
    copies = in_training = None
    if S.patient is not None:
        same = np.flatnonzero(S.patient == S.patient[i])
        copies = [int(j) for j in same]
        # copies of this patient that sat in the training rows of the model scoring row i
        in_training = int(np.sum(S.fold[same] != S.fold[i]))
    return jsonify({"id": i, "fold": int(S.fold[i]) + 1, "record": _patient_record(S, i),
                    "patient": None if S.patient is None else int(S.patient[i]),
                    "copies": copies, "copies_in_training": in_training})


def _encrypted_run(S, key, i, k):
    """Client encodes + encrypts, server computes on ciphertext, client decrypts."""
    pipe = S.pipelines[key][k]
    t_prep = time.perf_counter()
    x_encoded = pipe.named_steps["prep"].transform(S.X.iloc[[i]])
    prep_ms = (time.perf_counter() - t_prep) * 1000
    with _FHE_LOCKS[key]:
        enc_model, setup_s = _encrypted_model(S, key, k)
        p_enc, timings, ct, ct_out = enc_model.predict(x_encoded)
    return {"prob": p_enc, "timings_ms": {"preprocess": prep_ms, **timings},
            "setup_ms": setup_s * 1000, "ct": ct, "ct_out": ct_out}


def _ciphertext_fields(ct, ct_out):
    body = ct[len(ct) // 2: len(ct) // 2 + 24]
    return {"ciphertext_bytes": len(ct), "result_ciphertext_bytes": len(ct_out),
            "ciphertext_sha256": hashlib.sha256(ct).hexdigest()[:16],
            "ciphertext_sample": body.hex()}


def _status(S, key):
    if key == "elastic":
        if S.elastic_meta is None:
            return {"ready": False, "reason": "Run python -m healthguard.elastic first"}
        if not all(S.fhe_status[m]["ready"] for m in config.MODEL_KEYS):
            return {"ready": False, "reason": "needs all three base models' FHE bundles"}
        return {"ready": True, "reason": ""}
    return S.fhe_status[key]


@app.route("/api/infer/<key>", methods=["POST"])
def infer(key):
    S = _ds()
    if key not in (*config.MODEL_KEYS, "elastic"):
        abort(400, f"unknown model '{key}'")
    if key == "elastic" and S.elastic_meta is None:
        abort(400, "Elastic Learner not built — run python -m healthguard.elastic")
    i = _patient_id(S)
    k = int(S.fold[i])
    from healthguard.elastic import combine

    t0 = time.perf_counter()
    if key == "elastic":
        base_plain = [_plain_prob(S, m, i, k) for m in config.MODEL_KEYS]
        p_plain = float(combine(S.elastic_meta[k], np.array(base_plain))[0][0])
    else:
        p_plain = _plain_prob(S, key, i, k)
    plain_ms = (time.perf_counter() - t0) * 1000

    status = _status(S, key)
    out = {
        "dataset": S.name, "patient_id": i, "fold": k + 1, "model": key,
        "model_name": "Elastic Learner" if key == "elastic" else config.MODEL_NAMES[key],
        "plaintext": {"prob": p_plain, "label": _label(int(p_plain >= 0.5)),
                      "latency_ms": plain_ms},
        "encrypted": {"available": status["ready"], "reason": status["reason"]},
    }

    # ?encrypted=0 skips the FHE path (used by tests that only check plaintext).
    if request.args.get("encrypted", "1") == "0":
        out["encrypted"] = {"available": False, "reason": "skipped by request"}
    elif status["ready"] and key != "elastic":
        r = _encrypted_run(S, key, i, k)
        p_enc = r["prob"]
        out["encrypted"].update({
            "prob": p_enc, "label": _label(int(p_enc >= 0.5)),
            "timings_ms": r["timings_ms"], "total_ms": sum(r["timings_ms"].values()),
            "setup_ms": r["setup_ms"], **_ciphertext_fields(r["ct"], r["ct_out"]),
            "agree": (p_enc >= 0.5) == (p_plain >= 0.5),
        })
    elif status["ready"]:
        # Elastic Learner: the server runs all three base circuits on the
        # patient's ciphertext; the client decrypts the three probabilities
        # and applies the small plaintext meta-learner itself.
        runs = {m: _encrypted_run(S, m, i, k) for m in config.MODEL_KEYS}
        meta = S.elastic_meta[k]
        base_enc = np.array([runs[m]["prob"] for m in config.MODEL_KEYS])
        t_c = time.perf_counter()
        p_arr, contrib = combine(meta, base_enc)
        combine_ms = (time.perf_counter() - t_c) * 1000
        p_enc = float(p_arr[0])
        timings = {
            "preprocess": sum(r["timings_ms"]["preprocess"] for r in runs.values()),
            "encrypt_ms": sum(r["timings_ms"]["encrypt_ms"] for r in runs.values()),
            "server_ms": sum(r["timings_ms"]["server_ms"] for r in runs.values()),
            "decrypt_ms": sum(r["timings_ms"]["decrypt_ms"] for r in runs.values()) + combine_ms,
        }
        ct = b"".join(runs[m]["ct"] for m in config.MODEL_KEYS)
        ct_out = b"".join(runs[m]["ct_out"] for m in config.MODEL_KEYS)
        out["encrypted"].update({
            "prob": p_enc, "label": _label(int(p_enc >= 0.5)),
            "timings_ms": timings, "total_ms": sum(timings.values()),
            "setup_ms": sum(r["setup_ms"] for r in runs.values()),
            **_ciphertext_fields(ct, ct_out),
            "agree": (p_enc >= 0.5) == (p_plain >= 0.5),
        })
        out["elastic"] = {
            "intercept": float(meta.intercept_[0]),
            "base": [{"model": m, "name": config.MODEL_NAMES[m],
                      "plain_prob": base_plain[j], "enc_prob": float(base_enc[j]),
                      "weight": float(meta.coef_[0][j]), "contribution": float(contrib[0][j]),
                      "server_ms": runs[m]["timings_ms"]["server_ms"]}
                     for j, m in enumerate(config.MODEL_KEYS)],
        }

    if key == "elastic" and "elastic" not in out:
        meta = S.elastic_meta[k]
        contrib = combine(meta, np.array(base_plain))[1][0]
        out["elastic"] = {"intercept": float(meta.intercept_[0]), "base": [
            {"model": m, "name": config.MODEL_NAMES[m], "plain_prob": base_plain[j],
             "weight": float(meta.coef_[0][j]), "contribution": float(contrib[j])}
            for j, m in enumerate(config.MODEL_KEYS)]}

    out["truth"] = {"label": _label(int(S.y[i])), "value": int(S.y[i])}
    return jsonify(out)


@app.errorhandler(400)
@app.errorhandler(404)
def bad_request(err):
    return jsonify({"error": getattr(err, "description", str(err))}), err.code


def main():
    load_state()
    print("Serving on http://localhost:5001")
    app.run(host="127.0.0.1", port=5001, debug=False)


if __name__ == "__main__":
    main()
