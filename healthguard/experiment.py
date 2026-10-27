"""
Main experiment: Heart Disease, stratified 5-fold CV, three models, each one
also compiled to FHE (Phase 1: logistic regression; Phase 2: XGBoost, MLP).

    python -m healthguard.experiment [--quick]

For every outer fold:
  1. each model pipeline is tuned with GridSearchCV on that fold's *training*
     rows only (inner 5-fold, refit on F2), then scored once on the fold's
     held-out rows;
  2. the fold's trained model is compiled to an FHE circuit; its encrypted
     predictions for the held-out rows, the circuit's cost (bootstraps, key
     and ciphertext sizes) and genuine encrypted run time and memory are
     recorded;
  3. Phase-2 studies: the same models compiled at other quantisation
     settings, and an MLP with an x² activation instead of ReLU.

Pooling the held-out predictions gives one out-of-fold (OOF) prediction per
patient; metrics are reported both pooled and as mean ± std across folds.
"""

import argparse
import json
import platform
import shutil
import time
from datetime import datetime

import joblib
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from . import config, eda, fhe
from .data import fold_of_each_sample, folds, load_heart
from .evaluate import metrics, roc_points, summarize_folds
from .models import BUILDERS, F2_SCORER, build_mlp, complexity


def _tune_and_fit(key, X_tr, y_tr, quick):
    pipe, grid = BUILDERS[key](y_tr)
    if key == "mlp" and quick:
        pipe.set_params(model__max_epochs=60)
    t0 = time.perf_counter()
    if grid:
        if quick:
            grid = {k: v[:1] for k, v in grid.items()}
        search = GridSearchCV(
            pipe, grid, scoring=F2_SCORER, refit=True, n_jobs=-1,
            cv=StratifiedKFold(config.N_INNER_FOLDS, shuffle=True, random_state=config.SEED),
        )
        search.fit(X_tr, y_tr)
        best = search.best_estimator_
        params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
        inner_f2 = float(search.best_score_)
    else:
        best = pipe.fit(X_tr, y_tr)
        m = best.named_steps["model"]
        params = {"architecture": " → ".join(map(str, m.layer_sizes_)),
                  "dropout": m.dropout, "lr": m.lr,
                  "best_epoch": m.best_epoch_ + 1, "epochs_run": m.epochs_run_}
        inner_f2 = None
    return best, params, inner_f2, time.perf_counter() - t0


def _fhe_fold(key, best, X_tr, X_te, prob, k, quick):
    """Compile one fold's model, predict its test rows encrypted, measure cost."""
    fhe_model, compile_s = fhe.compile_model(key, best, X_tr)
    X_te_enc = best.named_steps["prep"].transform(X_te)
    n_real = 1 if quick else config.FHE_REAL_EXEC_SAMPLES[key]
    enc_prob, exec_info = fhe.evaluate_fold(fhe_model, X_te_enc, n_real, warmup=(key == "logreg"))
    fhe.save_bundle(fhe_model, config.FHE_DIR / key / f"fold{k}")
    agree = float(np.mean((enc_prob >= 0.5) == (prob >= 0.5)))
    info = {"fold": k, "compile_seconds": compile_s, "agreement": agree,
            "cost": fhe.circuit_cost(fhe_model), **exec_info}
    print(f"   {'  └ FHE circuit':20s} compiled {compile_s:.1f}s  agreement {agree * 100:.1f}%  "
          f"{info['cost']['pbs_count']} PBS  real {exec_info['real_exec_ms_per_sample'] / 1000:.2f} s/patient")
    return enc_prob, info


def _quant_study(key, best, X_tr, X_te, prob):
    """Simulation-only accuracy and cost at each setting in FHE_QUANT_STUDY."""
    X_te_enc = best.named_steps["prep"].transform(X_te)
    out = []
    for setting in config.FHE_QUANT_STUDY[key]:
        model, _ = fhe.compile_model(key, best, X_tr, setting)
        p = model.predict_proba(X_te_enc, fhe="simulate")[:, 1]
        out.append({"setting": setting,
                    "n_agree": int(np.sum((p >= 0.5) == (prob >= 0.5))),
                    "max_abs_prob_diff": float(np.max(np.abs(p - prob))),
                    "cost": fhe.circuit_cost(model)})
    return out


def _activation_study(X_tr, X_te, y_tr, y_te, quick):
    """MLP with x² instead of ReLU: plaintext quality and its FHE circuit."""
    pipe, _ = build_mlp(y_tr)
    pipe.set_params(model__activation="square")
    if quick:
        pipe.set_params(model__max_epochs=60)
    pipe.fit(X_tr, y_tr)
    prob = pipe.predict_proba(X_te)[:, 1]
    model, _ = fhe.compile_model("mlp", pipe, X_tr)
    p_enc = model.predict_proba(pipe.named_steps["prep"].transform(X_te), fhe="simulate")[:, 1]
    return prob, p_enc, fhe.circuit_cost(model)


def _versions():
    import concrete.ml, sklearn, torch, xgboost
    return {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
            "xgboost": xgboost.__version__, "torch": torch.__version__,
            "concrete-ml": concrete.ml.__version__}


def run(quick=False):
    t_start = time.perf_counter()
    config.RESULTS.mkdir(parents=True, exist_ok=True)
    for d in (config.MODELS_DIR, config.FHE_DIR):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    X, y = load_heart()
    fold_list = folds(y)
    owner = fold_of_each_sample(fold_list, len(y))
    print(f"Heart Disease: {len(y)} patients, {int(y.sum())} with disease | "
          f"{config.N_FOLDS}-fold stratified CV{' (quick)' if quick else ''}")

    oof = {k: np.zeros(len(y)) for k in config.MODEL_KEYS}
    oof_fhe = {k: np.zeros(len(y)) for k in config.MODEL_KEYS}
    per_fold = {k: [] for k in config.MODEL_KEYS}
    fold_info = {k: [] for k in config.MODEL_KEYS}
    fhe_folds = {k: [] for k in config.MODEL_KEYS}
    quant = {k: [] for k in config.FHE_QUANT_STUDY}
    sq_prob, sq_enc, sq_cost = np.zeros(len(y)), np.zeros(len(y)), []
    curves = []

    for k, (tr, te) in enumerate(fold_list):
        print(f"\n── Fold {k + 1}/{config.N_FOLDS}  (train {len(tr)}, test {len(te)})")
        X_tr, X_te, y_tr, y_te = X.iloc[tr], X.iloc[te], y[tr], y[te]

        for key in config.MODEL_KEYS:
            best, params, inner_f2, fit_s = _tune_and_fit(key, X_tr, y_tr, quick)
            prob = best.predict_proba(X_te)[:, 1]
            oof[key][te] = prob
            m = metrics(y_te, prob)
            per_fold[key].append(m)
            fold_info[key].append({"fold": k, "best_params": params,
                                   "inner_cv_f2": inner_f2, "fit_seconds": fit_s,
                                   "complexity": complexity(key, best)})
            (config.MODELS_DIR / key).mkdir(exist_ok=True)
            joblib.dump(best, config.MODELS_DIR / key / f"fold{k}.joblib")
            print(f"   {config.MODEL_NAMES[key]:20s} acc {m['accuracy']:.3f}  "
                  f"recall {m['recall']:.3f}  auc {m['roc_auc']:.3f}  {params}")

            if key == "mlp":
                mlp = best.named_steps["model"]
                curves.append({"fold": k, "best_epoch": mlp.best_epoch_,
                               "train_loss": [round(v, 5) for v in mlp.history_["train_loss"]],
                               "val_loss": [round(v, 5) for v in mlp.history_["val_loss"]]})

            oof_fhe[key][te], info = _fhe_fold(key, best, X_tr, X_te, prob, k, quick)
            fhe_folds[key].append(info)
            if key in config.FHE_QUANT_STUDY:
                quant[key].append(_quant_study(key, best, X_tr, X_te, prob))

        sq_prob[te], sq_enc[te], cost = _activation_study(X_tr, X_te, y_tr, y_te, quick)
        sq_cost.append(cost)
        print(f"   {'MLP with x²':20s} acc {metrics(y_te, sq_prob[te])['accuracy']:.3f}  "
              f"{cost['pbs_count']} PBS")

    # ── Aggregate ────────────────────────────────────────────────────────
    results = {
        "dataset": "heart",
        "n_samples": int(len(y)),
        "n_positive": int(y.sum()),
        "n_folds": config.N_FOLDS,
        "positive_class": "Disease",
        "selection_metric": "F2 (recall weighted 2× precision)",
        "quick": quick,
        "models": {},
    }
    for key in config.MODEL_KEYS:
        results["models"][key] = {
            "name": config.MODEL_NAMES[key],
            "pooled": metrics(y, oof[key]),
            "folds": per_fold[key],
            "summary": summarize_folds(per_fold[key]),
            "fold_info": fold_info[key],
            "roc": roc_points(y, oof[key]),
        }

    def mean_of(rows, field):
        return float(np.mean([r[field] for r in rows]))

    results["fhe"] = {}
    for key in config.MODEL_KEYS:
        rows, plain, enc = fhe_folds[key], oof[key], oof_fhe[key]
        costs = [r["cost"] for r in rows]
        results["fhe"][key] = {
            "settings": config.FHE_SETTINGS[key],
            "n_bits": config.FHE_SETTINGS[key]["n_bits"],
            "pooled": metrics(y, enc),
            "agreement": float(np.mean((enc >= 0.5) == (plain >= 0.5))),
            "n_disagree": int(np.sum((enc >= 0.5) != (plain >= 0.5))),
            "max_abs_prob_diff": float(np.max(np.abs(enc - plain))),
            "folds": rows,
            "real_exec_ms_per_sample": mean_of(rows, "real_exec_ms_per_sample"),
            "compile_seconds_mean": mean_of(rows, "compile_seconds"),
            "keygen_seconds_mean": mean_of(rows, "keygen_seconds"),
            "peak_extra_memory_mb_max": float(max(r["peak_extra_memory_mb"] for r in rows)),
            "real_vs_sim_agreement": mean_of(rows, "real_vs_sim_agreement"),
            "cost_mean": {f: mean_of(costs, f) for f in costs[0]},
            "cost_range": {f: [min(c[f] for c in costs), max(c[f] for c in costs)] for f in costs[0]},
        }

    study = {}
    for key, per_fold_rows in quant.items():
        study[key] = []
        for i, setting in enumerate(config.FHE_QUANT_STUDY[key]):
            cells = [fold_rows[i] for fold_rows in per_fold_rows]
            study[key].append({
                "setting": setting,
                "chosen": setting == config.FHE_SETTINGS[key],
                "n_agree": int(sum(c["n_agree"] for c in cells)),
                "max_abs_prob_diff": float(max(c["max_abs_prob_diff"] for c in cells)),
                "pbs_mean": float(np.mean([c["cost"]["pbs_count"] for c in cells])),
                "max_bit_width": int(max(c["cost"]["max_bit_width"] for c in cells)),
                "evaluation_key_bytes_mean": float(np.mean([c["cost"]["evaluation_key_bytes"] for c in cells])),
            })
    results["fhe_study"] = {
        "quantisation": study,
        "activation": {
            "relu": {"pooled": results["models"]["mlp"]["pooled"],
                     "encrypted": results["fhe"]["mlp"]["pooled"],
                     "pbs_mean": results["fhe"]["mlp"]["cost_mean"]["pbs_count"],
                     "max_bit_width": int(results["fhe"]["mlp"]["cost_range"]["max_bit_width"][1])},
            "square": {"pooled": metrics(y, sq_prob),
                       "encrypted": metrics(y, sq_enc),
                       "agreement": float(np.mean((sq_enc >= 0.5) == (sq_prob >= 0.5))),
                       "pbs_mean": float(np.mean([c["pbs_count"] for c in sq_cost])),
                       "max_bit_width": int(max(c["max_bit_width"] for c in sq_cost))},
        },
    }
    results["meta"] = {"generated": datetime.now().isoformat(timespec="seconds"),
                       "runtime_seconds": time.perf_counter() - t_start,
                       "versions": _versions(), "seed": config.SEED}

    predictions = {
        "y_true": y.tolist(),
        "fold": owner.tolist(),
        "prob": {k: [round(float(v), 6) for v in oof[k]] for k in config.MODEL_KEYS},
        "prob_fhe": {k: [round(float(v), 6) for v in oof_fhe[k]] for k in config.MODEL_KEYS},
    }

    def dump(name, obj):
        with open(config.RESULTS / name, "w") as f:
            json.dump(obj, f, indent=1)

    dump("metrics.json", results)
    dump("predictions.json", predictions)
    dump("training_curves.json", {"mlp": curves})
    dump("eda.json", eda.build())

    print("\n══ Pooled out-of-fold results (303 patients, positive = Disease) ══")
    for key in config.MODEL_KEYS:
        p, s = results["models"][key]["pooled"], results["models"][key]["summary"]
        print(f"   {config.MODEL_NAMES[key]:20s} acc {p['accuracy']:.4f} "
              f"(fold mean {s['accuracy']['mean']:.4f} ± {s['accuracy']['std']:.4f})  "
              f"recall {p['recall']:.4f}  f1 {p['f1']:.4f}  auc {p['roc_auc']:.4f}")
    print("\n══ Encrypted (FHE) versions ══")
    for key in config.MODEL_KEYS:
        f = results["fhe"][key]
        print(f"   {config.MODEL_NAMES[key]:20s} agreement {f['agreement'] * 100:.2f}%  "
              f"acc {f['pooled']['accuracy']:.4f}  {f['cost_mean']['pbs_count']:.0f} PBS  "
              f"{f['real_exec_ms_per_sample'] / 1000:.2f} s/patient  "
              f"eval keys {f['cost_mean']['evaluation_key_bytes'] / 1e6:.0f} MB")
    print(f"\nDone in {results['meta']['runtime_seconds']:.0f}s → {config.RESULTS}")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tiny grids, fewer epochs")
    run(quick=ap.parse_args().quick)
