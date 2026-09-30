"""
Web app for the Phase-1 prototype.

Loads what the experiment produced — metrics, out-of-fold predictions, the
fitted fold pipelines and the precompiled FHE bundles — and never trains or
re-implements preprocessing. Run from the project root:

    ~/aimlel-venv/bin/python -m webapp.server        (or scripts/run_app.sh)
"""

import hashlib
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, abort, jsonify, render_template, request

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from healthguard import config  # noqa: E402
from healthguard.data import load_heart  # noqa: E402

app = Flask(__name__)
app.json.sort_keys = False   # keep the complexity-ladder order LR → XGB → MLP

STATE = {
    "metrics": None, "predictions": None, "eda": None, "curves": None,
    "X": None, "y": None, "fold": None,
    "pipelines": {},     # key -> [fold pipeline, ...]
    "encrypted": {},     # key -> [EncryptedModel per fold]
    "fhe_status": {},    # key -> {"ready": bool, "reason": str}
    "scenarios": [],     # demo presets, one real patient per probability band
}


# ═════════════════════════════════════════════════════════════════════════
#  Start-up: load artifacts
# ═════════════════════════════════════════════════════════════════════════

def _read(name):
    path = config.RESULTS / name
    if not path.exists():
        sys.exit(f"Missing {path}. Run scripts/run_experiments.sh first.")
    return json.loads(path.read_text())


def load_state():
    STATE["metrics"] = _read("metrics.json")
    STATE["predictions"] = _read("predictions.json")
    STATE["eda"] = _read("eda.json")
    STATE["curves"] = _read("training_curves.json")

    X, y = load_heart()
    preds = STATE["predictions"]
    if preds["y_true"] != y.tolist():
        sys.exit("predictions.json does not match the dataset — re-run the experiment.")
    STATE["X"], STATE["y"], STATE["fold"] = X, y, np.array(preds["fold"])

    for key in config.MODEL_KEYS:
        STATE["pipelines"][key] = [
            joblib.load(config.MODELS_DIR / key / f"fold{k}.joblib")
            for k in range(config.N_FOLDS)
        ]

    # Consistency guard: the loaded fold models must reproduce the stored
    # out-of-fold probabilities, otherwise the artifacts are stale.
    for key in config.MODEL_KEYS:
        for i in range(0, len(y), 37):
            live = _plaintext_prob(key, i)
            stored = preds["prob"][key][i]
            if abs(live - stored) > 1e-4:
                sys.exit(f"Stale artifacts: {key} patient {i} gives {live:.4f}, "
                         f"stored {stored:.4f}. Re-run the experiment.")
    print(f"Loaded {len(config.MODEL_KEYS)} models × {config.N_FOLDS} folds; "
          f"artifacts consistent with predictions.json")

    STATE["scenarios"] = _build_scenarios()
    _load_fhe()


def _build_scenarios():
    """Pick one real patient per scenario band (see config.SCENARIOS)."""
    preds = STATE["predictions"]
    p = np.array(preds["prob"]["logreg"])
    enc = np.array(preds["prob_fhe"]["logreg"])
    y = STATE["y"]
    agree = (p >= 0.5) == (enc >= 0.5)
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
            "band": [lo, hi], "patient_id": i, "fold": int(STATE["fold"][i]) + 1,
            "p_disease": float(p[i]),
            # How often patients in this band truly have disease — tells the
            # audience what "probable" actually means for this model.
            "band_size": int(in_band.sum()),
            "band_disease_rate": float(y[in_band].mean()),
        })
    return out


def _load_fhe():
    from healthguard.fhe import EncryptedModel

    for key in config.MODEL_KEYS:
        STATE["fhe_status"][key] = {"ready": False,
                                    "reason": "Phase 2 · not compiled in current prototype"}

    bundle_root = config.FHE_DIR / "logreg"
    bundles = [bundle_root / f"fold{k}" for k in range(config.N_FOLDS)]
    if not all((b / "server.zip").exists() for b in bundles):
        STATE["fhe_status"]["logreg"]["reason"] = (
            "FHE bundles missing — run scripts/run_experiments.sh")
        print("  [warn] FHE bundles missing; encrypted inference disabled")
        return

    t0 = time.perf_counter()
    try:
        STATE["encrypted"]["logreg"] = [
            EncryptedModel(b, config.FHE_DIR / "keys" / "logreg" / b.name) for b in bundles
        ]
    except Exception as e:  # never compile here — see note in healthguard/fhe.py
        STATE["fhe_status"]["logreg"]["reason"] = f"FHE bundle failed to load: {e}"
        print(f"  [warn] FHE bundle load failed: {e}")
        return
    STATE["fhe_status"]["logreg"] = {"ready": True, "reason": ""}
    print(f"Loaded {len(bundles)} precompiled FHE bundles (no compilation) "
          f"in {time.perf_counter() - t0:.2f}s")

    # Warm-up: the first encrypted call on each circuit pays a one-off runtime
    # initialisation (~0.5 s). Run it once per fold at start-up so the timings
    # shown during the demo are steady-state rather than first-call outliers.
    t0 = time.perf_counter()
    for k, enc in enumerate(STATE["encrypted"]["logreg"]):
        i = int(np.flatnonzero(STATE["fold"] == k)[0])
        pipe = STATE["pipelines"]["logreg"][k]
        enc.predict(pipe.named_steps["prep"].transform(STATE["X"].iloc[[i]]))
    print(f"Warmed up {len(bundles)} FHE circuits in {time.perf_counter() - t0:.2f}s")


# ═════════════════════════════════════════════════════════════════════════
#  Helpers
# ═════════════════════════════════════════════════════════════════════════

def _plaintext_prob(key, i):
    pipe = STATE["pipelines"][key][STATE["fold"][i]]
    return float(pipe.predict_proba(STATE["X"].iloc[[i]])[0, 1])


def _fhe_disagreements():
    """Patients whose encrypted and plaintext OOF decisions differ."""
    plain = np.array(STATE["predictions"]["prob"]["logreg"])
    enc = np.array(STATE["predictions"]["prob_fhe"]["logreg"])
    ids = np.flatnonzero((plain >= 0.5) != (enc >= 0.5))
    return [{"id": int(i), "fold": int(STATE["fold"][i]) + 1,
             "plain": float(plain[i]), "enc": float(enc[i])} for i in ids]


def _label(v):
    return "Disease" if v == 1 else "No Disease"


def _patient_record(i):
    row = STATE["X"].iloc[i]
    record = []
    for col in STATE["X"].columns:
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


def _patient_id():
    body = request.get_json(silent=True) or {}
    try:
        i = int(body.get("patient_id"))
    except (TypeError, ValueError):
        abort(400, "patient_id must be an integer")
    if not 0 <= i < len(STATE["y"]):
        abort(400, f"patient_id must be between 0 and {len(STATE['y']) - 1}")
    return i


# ═════════════════════════════════════════════════════════════════════════
#  Routes
# ═════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/overview")
def overview():
    m, e = STATE["metrics"], STATE["eda"]["heart"]
    return jsonify({
        "n_samples": e["n_samples"],
        "n_attributes": e["n_attributes"],
        "n_features": e["n_features"],
        "n_models": len(config.MODEL_KEYS),
        "n_folds": m["n_folds"],
        "model_order": config.MODEL_KEYS,
        "models": {k: {"name": v["name"], "pooled": v["pooled"], "summary": v["summary"]}
                   for k, v in m["models"].items()},
        "fhe": m["fhe"]["logreg"],
        "fhe_disagreements": _fhe_disagreements(),
        "fhe_status": STATE["fhe_status"],
        "selection_metric": m["selection_metric"],
        "meta": m["meta"],
    })


@app.route("/api/eda/<ds>")
def eda(ds):
    if ds not in STATE["eda"]:
        abort(404)
    return jsonify(STATE["eda"][ds])


@app.route("/api/results")
def results():
    return jsonify({"metrics": STATE["metrics"], "curves": STATE["curves"],
                    "model_order": config.MODEL_KEYS})


@app.route("/api/patients")
def patients():
    return jsonify([{"id": i, "fold": int(f) + 1} for i, f in enumerate(STATE["fold"])])


@app.route("/api/scenarios")
def scenarios():
    return jsonify(STATE["scenarios"])


@app.route("/api/patient/<int:i>")
def patient(i):
    if not 0 <= i < len(STATE["y"]):
        abort(404)
    return jsonify({"id": i, "fold": int(STATE["fold"][i]) + 1, "record": _patient_record(i)})


@app.route("/api/infer/<key>", methods=["POST"])
def infer(key):
    if key not in config.MODEL_KEYS:
        abort(400, f"unknown model '{key}'")
    i = _patient_id()
    k = int(STATE["fold"][i])
    pipe = STATE["pipelines"][key][k]
    x_raw = STATE["X"].iloc[[i]]

    t0 = time.perf_counter()
    p_plain = float(pipe.predict_proba(x_raw)[0, 1])
    plain_ms = (time.perf_counter() - t0) * 1000

    out = {
        "patient_id": i, "fold": k + 1, "model": key,
        "model_name": config.MODEL_NAMES[key],
        "plaintext": {"prob": p_plain, "label": _label(int(p_plain >= 0.5)),
                      "latency_ms": plain_ms},
        "encrypted": {"available": STATE["fhe_status"][key]["ready"],
                      "reason": STATE["fhe_status"][key]["reason"]},
    }

    if STATE["fhe_status"][key]["ready"]:
        # Client side: the *saved* fitted preprocessor encodes the record.
        t_prep = time.perf_counter()
        x_encoded = pipe.named_steps["prep"].transform(x_raw)
        prep_ms = (time.perf_counter() - t_prep) * 1000

        p_enc, timings, ct, ct_out = STATE["encrypted"][key][k].predict(x_encoded)
        body = ct[len(ct) // 2: len(ct) // 2 + 24]
        out["encrypted"].update({
            "prob": p_enc,
            "label": _label(int(p_enc >= 0.5)),
            "timings_ms": {"preprocess": prep_ms, **timings},
            "total_ms": prep_ms + sum(timings.values()),
            "ciphertext_bytes": len(ct),
            "result_ciphertext_bytes": len(ct_out),
            "ciphertext_sha256": hashlib.sha256(ct).hexdigest()[:16],
            "ciphertext_sample": body.hex(),
            "agree": (p_enc >= 0.5) == (p_plain >= 0.5),
        })

    out["truth"] = {"label": _label(int(STATE["y"][i])), "value": int(STATE["y"][i])}
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
