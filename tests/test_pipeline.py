"""
Correctness tests for the ML pipeline. These guard the specific mistakes the
Phase-0 implementation made: inverted labels, preprocessing leakage, and an
early-stopping restore that silently kept the last epoch.
"""

import numpy as np
import pytest

from healthguard import config
from healthguard.data import (fold_of_each_sample, folds, inner_cv, load_base_paper, load_dataset,
                              load_heart, match_to_cleveland)
from healthguard.models import build_logreg, build_mlp


@pytest.fixture(scope="module")
def heart():
    return load_heart()


def test_labels_are_binary_with_disease_as_one(heart):
    X, y = heart
    assert set(np.unique(y)) == {0, 1}
    assert len(y) == 303
    # The corrected positive class must look like disease clinically.
    assert X.loc[y == 1, "oldpeak"].mean() > X.loc[y == 0, "oldpeak"].mean()
    assert X.loc[y == 1, "exang"].mean() > X.loc[y == 0, "exang"].mean()
    assert X.loc[y == 1, "thalach"].mean() < X.loc[y == 0, "thalach"].mean()


def test_invalid_codes_become_missing(heart):
    X, _ = heart
    assert not (X["ca"] == 4).any()
    assert not (X["thal"] == 0).any()
    assert X["ca"].isna().sum() == 5 and X["thal"].isna().sum() == 2


def test_folds_partition_every_patient_exactly_once(heart):
    _, y = heart
    fl = folds(y)
    assert len(fl) == config.N_FOLDS
    test_ids = np.concatenate([te for _, te in fl])
    assert len(test_ids) == len(y) and len(np.unique(test_ids)) == len(y)
    assert (fold_of_each_sample(fl, len(y)) >= 0).all()
    for tr, te in fl:
        assert not set(tr) & set(te)


def test_scaler_is_fitted_on_training_rows_only(heart):
    X, y = heart
    tr, _ = folds(y)[0]
    pipe, _ = build_logreg()
    pipe.fit(X.iloc[tr], y[tr])
    scaler = pipe.named_steps["prep"].named_transformers_["num"]
    train_mean = X.iloc[tr][config.HD_CONTINUOUS].mean().to_numpy()
    full_mean = X[config.HD_CONTINUOUS].mean().to_numpy()
    assert np.allclose(scaler.mean_, train_mean)
    assert not np.allclose(scaler.mean_, full_mean)


def test_mlp_restores_best_epoch_weights(heart):
    X, y = heart
    tr, _ = folds(y)[0]
    pipe, _ = build_mlp()
    pipe.fit(X.iloc[tr], y[tr])
    mlp = pipe.named_steps["model"]
    Xt = pipe.named_steps["prep"].transform(X.iloc[tr])
    restored = mlp.validation_loss(Xt, y[tr])
    assert restored == pytest.approx(mlp.best_val_loss_, abs=1e-6)
    assert restored == pytest.approx(min(mlp.history_["val_loss"]), abs=1e-6)
    # The test only means something if training ran past the best epoch.
    assert mlp.epochs_run_ > mlp.best_epoch_ + 1


def test_same_seed_gives_identical_predictions(heart):
    X, y = heart
    tr, te = folds(y)[1]
    probs = []
    for _ in range(2):
        pipe, _ = build_mlp()
        pipe.fit(X.iloc[tr], y[tr])
        probs.append(pipe.predict_proba(X.iloc[te])[:, 1])
    assert np.array_equal(probs[0], probs[1])


def test_mlp_input_width_comes_from_the_data(heart):
    X, y = heart
    pipe, _ = build_mlp()
    pipe.fit(X, y)
    n_encoded = len(pipe.named_steps["prep"].get_feature_names_out())
    assert pipe.named_steps["model"].layer_sizes_ == [n_encoded, 16, 8, 1]


def test_base_paper_file_is_our_patients_repeated(heart):
    X, y, group = load_base_paper()
    assert len(y) == 1025 and len(np.unique(group)) == 302
    ours = {tuple(np.nan_to_num(r, nan=-1)) for r in np.column_stack([heart[0].to_numpy(), heart[1]])}
    theirs = {tuple(np.nan_to_num(r, nan=-1)) for r in np.column_stack([X.to_numpy(), y])}
    assert theirs <= ours          # every base-paper record, with its label, is one of ours


def test_patient_split_keeps_every_copy_on_one_side():
    X, y, group = load_dataset("base1025_patient")
    for tr, te in folds(y, group):
        assert not set(group[tr]) & set(group[te])
        # the inner tuning folds inside each training set are grouped too
        for itr, ite in inner_cv(group[tr]).split(X.iloc[tr], y[tr], group[tr]):
            assert not set(group[tr][itr]) & set(group[tr][ite])
    _, y_row, no_groups = load_dataset("base1025_row")
    assert no_groups is None
    leaked = sum(np.isin(group[te], group[tr]).sum() for tr, te in folds(y_row))
    assert leaked > 0.9 * len(y)   # a plain row split leaks almost every test patient


def test_mlp_early_stopping_split_keeps_patients_together():
    X, y, group = load_dataset("base1025_patient")
    tr, _ = folds(y, group)[0]
    pipe, _ = build_mlp()
    pipe.set_params(model__max_epochs=5)
    pipe.fit(X.iloc[tr], y[tr], model__groups=group[tr])
    val, g = pipe.named_steps["model"].val_idx_, group[tr]
    assert not set(g[val]) & set(np.delete(g, val))


def test_elastic_inner_oof_with_groups():
    from healthguard.elastic import inner_oof
    X, y, group = load_dataset("base1025_patient")
    tr, _ = folds(y, group)[0]
    pipe, _ = build_logreg()
    pipe.set_params(model__C=0.1)
    p = inner_oof(pipe, X.iloc[tr], y[tr], group[tr])
    g = group[tr]
    # copies of one patient are scored by the same inner model → identical
    for pid in np.unique(g)[:20]:
        assert np.ptp(p[g == pid]) < 1e-12


def test_cleveland_folds_are_unchanged(heart):
    _, y, groups = load_dataset("cleveland")
    assert groups is None
    assert all(np.array_equal(a[1], b[1]) for a, b in zip(folds(y), folds(heart[1])))


def test_mlp_square_activation_is_supported(heart):
    X, y = heart
    tr, _ = folds(y)[0]
    pipe, _ = build_mlp()
    pipe.set_params(model__activation="square", model__max_epochs=5)
    pipe.fit(X.iloc[tr], y[tr])
    p = pipe.predict_proba(X.iloc[tr])[:, 1]
    assert np.all((p >= 0) & (p <= 1))


def test_elastic_inner_oof_predicts_every_training_row_from_unseen_models(heart):
    from healthguard.elastic import inner_oof
    X, y = heart
    tr, _ = folds(y)[0]
    pipe, _ = build_logreg()
    pipe.set_params(model__C=0.1)
    p = inner_oof(pipe, X.iloc[tr], y[tr])
    assert p.shape == (len(tr),) and np.all((p > 0) & (p < 1))
    # Out-of-fold, so it must differ from the fit-on-everything predictions.
    full = pipe.fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[tr])[:, 1]
    assert not np.allclose(p, full)


def test_every_base_paper_row_matches_one_of_our_patients(heart):
    X, y, _ = load_base_paper()
    pid = match_to_cleveland()
    assert len(pid) == 1025 and len(np.unique(pid)) == 302
    assert np.array_equal(heart[1][pid], y)
    ours = heart[0].to_numpy()[pid]
    assert np.allclose(np.nan_to_num(ours, nan=-1), np.nan_to_num(X.to_numpy(), nan=-1))
