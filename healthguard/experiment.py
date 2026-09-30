"""
Phase-1 experiment: Heart Disease, stratified 5-fold CV, three models, and the
encrypted logistic regression.

    python -m healthguard.experiment [--quick]

For every outer fold:
  1. each model pipeline is tuned with GridSearchCV on that fold's *training*
     rows only (inner 5-fold, refit on F2), then scored once on the fold's
     held-out rows;
  2. the fold's logistic regression is compiled to an FHE circuit and its
     encrypted predictions for the held-out rows are recorded.

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
from .models import BUILDERS, F2_SCORER, complexity


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
    oof_fhe = np.zeros(len(y))
    per_fold = {k: [] for k in config.MODEL_KEYS}
    fold_info = {k: [] for k in config.MODEL_KEYS}
    fhe_folds, curves = [], []

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

            if key == "logreg":
                fhe_model, compile_s = fhe.compile_logreg(best, X_tr)
                n_real = 1 if quick else config.FHE_REAL_EXEC_SAMPLES
                enc_prob, exec_info = fhe.evaluate_fold(fhe_model, best, X_te, n_real)
                oof_fhe[te] = enc_prob
                fhe.save_bundle(fhe_model, config.FHE_DIR / "logreg" / f"fold{k}")
                agree = float(np.mean((enc_prob >= 0.5) == (prob >= 0.5)))
                fhe_folds.append({"fold": k, "compile_seconds": compile_s,
                                  "agreement": agree, **exec_info})
                print(f"   {'  └ FHE circuit':20s} compiled {compile_s:.2f}s  "
                      f"agreement {agree * 100:.1f}%  "
                      f"real {exec_info['real_exec_ms_per_sample']:.0f} ms/patient")

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

    agreement = float(np.mean((oof_fhe >= 0.5) == (oof["logreg"] >= 0.5)))
    results["fhe"] = {
        "logreg": {
            "n_bits": config.FHE_N_BITS,
            "pooled": metrics(y, oof_fhe),
            "agreement": agreement,
            "n_disagree": int(np.sum((oof_fhe >= 0.5) != (oof["logreg"] >= 0.5))),
            "max_abs_prob_diff": float(np.max(np.abs(oof_fhe - oof["logreg"]))),
            "folds": fhe_folds,
            "real_exec_ms_per_sample": float(np.mean([f["real_exec_ms_per_sample"] for f in fhe_folds])),
            "compile_seconds_mean": float(np.mean([f["compile_seconds"] for f in fhe_folds])),
            "keygen_seconds_mean": float(np.mean([f["keygen_seconds"] for f in fhe_folds])),
            "real_vs_sim_agreement": float(np.mean([f["real_vs_sim_agreement"] for f in fhe_folds])),
        },
    }
    results["meta"] = {"generated": datetime.now().isoformat(timespec="seconds"),
                       "runtime_seconds": time.perf_counter() - t_start,
                       "versions": _versions(), "seed": config.SEED}

    predictions = {
        "y_true": y.tolist(),
        "fold": owner.tolist(),
        "prob": {k: [round(float(v), 6) for v in oof[k]] for k in config.MODEL_KEYS},
        "prob_fhe": {"logreg": [round(float(v), 6) for v in oof_fhe]},
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
    print(f"   FHE logistic regression: agreement {agreement * 100:.2f}% with plaintext, "
          f"{results['fhe']['logreg']['real_exec_ms_per_sample']:.0f} ms/patient encrypted")
    print(f"\nDone in {results['meta']['runtime_seconds']:.0f}s → {config.RESULTS}")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tiny grids, fewer epochs")
    run(quick=ap.parse_args().quick)
