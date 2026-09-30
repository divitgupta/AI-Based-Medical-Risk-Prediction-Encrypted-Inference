"""
Training code for the three plaintext models:
    1. Logistic Regression
    2. XGBoost (tree ensemble)
    3. Small MLP (PyTorch)

Each model exposes a `train_and_evaluate(...)` function that returns
(fitted_model, metrics_dict).
"""

import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import (
    RANDOM_SEED, CV_FOLDS, MODELS_DIR,
    LOGREG_PARAMS, XGB_PARAMS, MLP_PARAMS,
)
from src.utils.evaluation import full_evaluation


# ═════════════════════════════════════════════════════════════════════════
#  1. Logistic Regression
# ═════════════════════════════════════════════════════════════════════════

def train_logistic_regression(X_train, y_train, X_test, y_test, dataset_name="dataset"):
    """Grid-search over C; return (best_model, metrics)."""
    print("\n🔹 Training Logistic Regression …")
    lr = LogisticRegression(random_state=RANDOM_SEED, class_weight="balanced")
    gs = GridSearchCV(
        lr, LOGREG_PARAMS,
        cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED),
        scoring="recall", refit=True, n_jobs=-1,
    )
    gs.fit(X_train, y_train)
    best = gs.best_estimator_
    print(f"  Best C = {gs.best_params_['C']}")

    y_pred = best.predict(X_test)
    y_prob = best.predict_proba(X_test)[:, 1]
    metrics = full_evaluation(y_test, y_pred, y_prob,
                              "Logistic Regression", dataset_name)

    # Persist
    path = MODELS_DIR / f"logreg_{dataset_name.lower().replace(' ','_')}.joblib"
    joblib.dump(best, path)
    print(f"  Model saved → {path}")
    return best, metrics


# ═════════════════════════════════════════════════════════════════════════
#  2. XGBoost
# ═════════════════════════════════════════════════════════════════════════

def train_xgboost(X_train, y_train, X_test, y_test, dataset_name="dataset"):
    """Grid-search over XGBoost hyperparameters; return (best_model, metrics)."""
    print("\n🔹 Training XGBoost …")

    # Compute scale_pos_weight for class imbalance
    n_neg = np.sum(y_train == 0)
    n_pos = np.sum(y_train == 1)
    scale_pos = n_neg / max(n_pos, 1)

    xgb = XGBClassifier(
        random_state=RANDOM_SEED,
        scale_pos_weight=scale_pos,
        verbosity=0,
    )
    gs = GridSearchCV(
        xgb, XGB_PARAMS,
        cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED),
        scoring="recall", refit=True, n_jobs=-1,
    )
    gs.fit(X_train, y_train)
    best = gs.best_estimator_
    print(f"  Best params: {gs.best_params_}")

    y_pred = best.predict(X_test)
    y_prob = best.predict_proba(X_test)[:, 1]
    metrics = full_evaluation(y_test, y_pred, y_prob, "XGBoost", dataset_name)

    path = MODELS_DIR / f"xgb_{dataset_name.lower().replace(' ','_')}.joblib"
    joblib.dump(best, path)
    print(f"  Model saved → {path}")
    return best, metrics


# ═════════════════════════════════════════════════════════════════════════
#  3. Small MLP (PyTorch)
# ═════════════════════════════════════════════════════════════════════════

class DiagnosticMLP(nn.Module):
    """2-hidden-layer feedforward network for binary classification."""

    def __init__(self, input_dim, hidden_sizes=(16, 8), dropout=0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp(X_train, y_train, X_test, y_test, dataset_name="dataset"):
    """Train the small MLP with early stopping; return (model, metrics)."""
    print("\n🔹 Training MLP …")
    cfg = MLP_PARAMS
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Seed every source of randomness in the run: weight init, dropout masks and
    # DataLoader shuffling. Without this the MLP scores drift by ~1% between
    # runs, which makes the reported numbers impossible to reproduce.
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Tensors
    Xtr = torch.tensor(X_train, dtype=torch.float32)
    ytr = torch.tensor(y_train, dtype=torch.float32)
    Xte = torch.tensor(X_test, dtype=torch.float32)
    yte = torch.tensor(y_test, dtype=torch.float32)

    # 90/10 internal val split for early stopping
    n_val = max(1, int(0.1 * len(Xtr)))
    perm = torch.randperm(len(Xtr), generator=torch.Generator().manual_seed(RANDOM_SEED))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    train_loader = DataLoader(
        TensorDataset(Xtr[tr_idx], ytr[tr_idx]),
        batch_size=cfg["batch_size"], shuffle=True,
        generator=torch.Generator().manual_seed(RANDOM_SEED),
    )
    val_X, val_y = Xtr[val_idx].to(device), ytr[val_idx].to(device)

    model = DiagnosticMLP(
        input_dim=X_train.shape[1],
        hidden_sizes=cfg["hidden_sizes"],
        dropout=cfg["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    # Class-weighted BCE
    pos_weight = torch.tensor(
        [np.sum(y_train == 0) / max(np.sum(y_train == 1), 1)],
        dtype=torch.float32,
    ).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(cfg["epochs"]):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(val_X), val_y).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= cfg["patience"]:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    # Evaluate
    model.eval()
    with torch.no_grad():
        logits = model(Xte.to(device))
        y_prob = torch.sigmoid(logits).cpu().numpy()
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = full_evaluation(y_test, y_pred, y_prob, "MLP", dataset_name)

    # Save
    path = MODELS_DIR / f"mlp_{dataset_name.lower().replace(' ','_')}.pt"
    torch.save(model.state_dict(), path)
    print(f"  Model saved → {path}")
    return model, metrics
