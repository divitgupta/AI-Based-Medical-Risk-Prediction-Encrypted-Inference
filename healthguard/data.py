"""
Dataset loading and the cross-validation scheme.

Label convention everywhere in this project: 1 = the finding we must not miss
(disease present / malignant), 0 = its absence. Recall therefore always means
sensitivity to disease.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from . import config


def load_heart():
    """
    Load the Heart Disease (Cleveland) data with corrected labels.

    Returns
    -------
    X : DataFrame, 303 × 13 raw clinical features (invalid codes set to NaN)
    y : ndarray of int, 1 = Disease, 0 = No Disease

    Two corrections to the Kaggle export are applied here:

    1. Label inversion. In this CSV `target = 1` means *no* disease. Every
       clinical sign agrees: the target=0 group has more exercise angina, more
       ST depression, more blocked vessels, a lower max heart rate, is older
       and more male. Its counts (165 ones) also match the 164 *absence* cases
       of the original UCI data. We therefore use y = 1 - target.

    2. Invalid codes. ca = 4 and thal = 0 do not exist in the UCI codebook;
       they are how this export encoded missing values. They become NaN and
       are imputed inside the model pipeline, so all 303 records are kept
       and no statistic is computed on held-out rows.
    """
    df = pd.read_csv(config.HEART_CSV)
    y = (1 - df["target"]).astype(int).to_numpy()
    X = df.drop(columns=["target"]).astype(float)
    for col, bad in config.HD_INVALID_CODES.items():
        X.loc[X[col].isin(bad), col] = np.nan
    return X, y


def load_breast():
    """
    Load Breast Cancer Wisconsin for the dataset preview (Phase 2 experiment).

    The CSV codes 1 = benign; we flip it so 1 = malignant, matching the
    project-wide convention.
    """
    df = pd.read_csv(config.BREAST_CSV)
    y = (1 - df["target"]).astype(int).to_numpy()
    X = df.drop(columns=["target"])
    return X, y


def folds(y):
    """
    Stratified 5-fold split. Returns a list of (train_idx, test_idx).

    Every sample appears in exactly one test fold, so pooling the test-fold
    predictions gives one out-of-fold (OOF) prediction per patient.
    """
    skf = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True,
                          random_state=config.SEED)
    return list(skf.split(np.zeros(len(y)), y))


def fold_of_each_sample(fold_list, n):
    """Array mapping sample index → the fold in which it was held out."""
    owner = np.full(n, -1, dtype=int)
    for k, (_, test_idx) in enumerate(fold_list):
        owner[test_idx] = k
    return owner
