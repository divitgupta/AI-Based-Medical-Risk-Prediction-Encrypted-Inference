"""
Correctness tests for the ML pipeline. These guard the specific mistakes the
Phase-0 implementation made: inverted labels, preprocessing leakage, and an
early-stopping restore that silently kept the last epoch.
"""

import numpy as np
import pytest

from healthguard import config
from healthguard.data import fold_of_each_sample, folds, load_heart
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
