"""
Data loading, cleaning, and preprocessing pipelines.

Two public functions:
    load_breast_cancer()  → (X_train, X_test, y_train, y_test, feature_names, preprocessor)
    load_heart_disease()  → (X_train, X_test, y_train, y_test, feature_names, preprocessor)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import (
    BREAST_CANCER_CSV, HEART_DISEASE_CSV,
    HEART_COLS, TEST_SIZE, RANDOM_SEED,
)


# ═════════════════════════════════════════════════════════════════════════
#  Breast Cancer Wisconsin (Diagnostic)
# ═════════════════════════════════════════════════════════════════════════

def load_breast_cancer():
    """
    Load, clean, and split the Breast Cancer dataset.

    Returns
    -------
    X_train, X_test : np.ndarray   (scaled)
    y_train, y_test : np.ndarray
    feature_names   : list[str]
    preprocessor    : fitted ColumnTransformer (StandardScaler on all 30 features)
    """
    df = pd.read_csv(BREAST_CANCER_CSV)
    print(f"[Breast Cancer] Loaded {df.shape[0]} samples, {df.shape[1]-1} features")

    # Separate features and target
    target_col = "target"
    X = df.drop(columns=[target_col])
    y = df[target_col].values
    feature_names = list(X.columns)

    # All features are continuous → StandardScaler
    preprocessor = ColumnTransformer(
        transformers=[("scale", StandardScaler(), feature_names)],
        remainder="passthrough",
    )

    # Stratified 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    print(f"  Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")
    print(f"  Class distribution (train): {np.bincount(y_train)}")
    return X_train, X_test, y_train, y_test, feature_names, preprocessor


# ═════════════════════════════════════════════════════════════════════════
#  Heart Disease UCI (Cleveland)
# ═════════════════════════════════════════════════════════════════════════

def load_heart_disease():
    """
    Load, clean, and split the Heart Disease dataset.

    The preprocessor is fitted on the TRAINING split only and then applied to
    the test split. Fitting it on the full dataset first would leak test-set
    statistics (the scaler's mean/std and the encoder's category set) into
    training and inflate the reported scores.

    Returns
    -------
    X_train, X_test : np.ndarray   (preprocessed)
    y_train, y_test : np.ndarray
    feature_names   : list[str]    (raw column names, pre-encoding)
    preprocessor    : ColumnTransformer fitted on the training split
    """
    df = pd.read_csv(HEART_DISEASE_CSV)
    print(f"[Heart Disease] Loaded {df.shape[0]} samples, {df.shape[1]-1} features")

    # Drop any rows with NaN (Cleveland set sometimes has '?' as missing)
    df.replace("?", np.nan, inplace=True)
    df.dropna(inplace=True)

    target_col = HEART_COLS["target"]
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int).values
    feature_names = list(X.columns)

    # Identify categorical vs continuous
    cat_cols = [c for c in HEART_COLS["categorical"] if c in X.columns]
    cont_cols = [c for c in HEART_COLS["continuous"] if c in X.columns]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), cont_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ],
        remainder="passthrough",
    )

    # Split the RAW frame first, then fit only on the training rows.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    print(f"  Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")
    print(f"  Features after encoding: {X_train.shape[1]}")
    print(f"  Class distribution (train): {np.bincount(y_train)}")
    return X_train, X_test, y_train, y_test, feature_names, preprocessor


# ═════════════════════════════════════════════════════════════════════════
#  Quick sanity check
# ═════════════════════════════════════════════════════════════════════════

def get_correlation_matrix(csv_path):
    """Return the Pearson correlation matrix of all numeric columns."""
    df = pd.read_csv(csv_path)
    return df.select_dtypes(include=[np.number]).corr()


if __name__ == "__main__":
    print("=" * 60)
    load_breast_cancer()
    print("=" * 60)
    load_heart_disease()
