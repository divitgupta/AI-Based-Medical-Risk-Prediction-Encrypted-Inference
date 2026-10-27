"""
Small multi-layer perceptron (synopsis §6.3) wrapped as a scikit-learn
classifier, so it goes through exactly the same Pipeline / cross-validation
code as the other two models.

Architecture: n_in → 16 → 8 → 1, where n_in is read from the data at fit time
(the width after one-hot encoding), never hardcoded.

The hidden activation is ReLU by default. `activation="square"` swaps it for
x², the polynomial activation CKKS-style schemes need (CryptoNets; Zimerman
et al., 2024); the Phase-2 activation study compares the two.
"""

import copy

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split


class _Square(nn.Module):
    def forward(self, x):
        return x * x


ACTIVATIONS = {"relu": nn.ReLU, "square": _Square}


class _Net(nn.Module):
    def __init__(self, n_in, hidden, dropout, activation="relu"):
        super().__init__()
        layers, prev = [], n_in
        for h in hidden:
            layers += [nn.Linear(prev, h), ACTIVATIONS[activation](), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class MLPClassifier(BaseEstimator, ClassifierMixin):
    """
    Adam + class-weighted BCE + early stopping on a held-out slice of the
    training data. The weights from the best validation epoch are restored at
    the end (a deep copy — a shallow state_dict().copy() would silently keep
    the last epoch's weights).
    """

    def __init__(self, hidden=(16, 8), dropout=0.3, lr=1e-3, batch_size=32,
                 max_epochs=300, patience=25, val_fraction=0.15, activation="relu",
                 random_state=42):
        self.hidden = hidden
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.val_fraction = val_fraction
        self.activation = activation
        self.random_state = random_state

    # ── training ─────────────────────────────────────────────────────────
    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        torch.manual_seed(self.random_state)

        idx = np.arange(len(y))
        tr_idx, val_idx = train_test_split(
            idx, test_size=self.val_fraction, stratify=y,
            random_state=self.random_state,
        )
        self.val_idx_ = val_idx

        Xtr, ytr = torch.from_numpy(X[tr_idx]), torch.from_numpy(y[tr_idx])
        Xva, yva = torch.from_numpy(X[val_idx]), torch.from_numpy(y[val_idx])

        self.n_features_in_ = X.shape[1]
        self.classes_ = np.array([0, 1])
        self.net_ = _Net(self.n_features_in_, self.hidden, self.dropout, self.activation)

        # Class weighting (synopsis §4.3): up-weight the positive class by the
        # negative/positive ratio of the training rows.
        n_pos = float(ytr.sum())
        self.pos_weight_ = (len(ytr) - n_pos) / max(n_pos, 1.0)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.pos_weight_))
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr)
        gen = torch.Generator().manual_seed(self.random_state)

        best_state, best_loss, best_epoch, bad_epochs = None, float("inf"), 0, 0
        self.history_ = {"train_loss": [], "val_loss": []}

        for epoch in range(self.max_epochs):
            self.net_.train()
            perm = torch.randperm(len(ytr), generator=gen)
            running = 0.0
            for start in range(0, len(ytr), self.batch_size):
                b = perm[start:start + self.batch_size]
                opt.zero_grad()
                loss = loss_fn(self.net_(Xtr[b]), ytr[b])
                loss.backward()
                opt.step()
                running += loss.item() * len(b)

            val_loss = self._loss(Xva, yva, loss_fn)
            self.history_["train_loss"].append(running / len(ytr))
            self.history_["val_loss"].append(val_loss)

            if val_loss < best_loss - 1e-6:
                best_loss, best_epoch, bad_epochs = val_loss, epoch, 0
                best_state = copy.deepcopy(self.net_.state_dict())
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    break

        self.net_.load_state_dict(best_state)
        self.net_.eval()
        self.best_epoch_ = best_epoch
        self.best_val_loss_ = best_loss
        self.epochs_run_ = len(self.history_["val_loss"])
        return self

    def _loss(self, X, y, loss_fn):
        self.net_.eval()
        with torch.no_grad():
            return float(loss_fn(self.net_(X), y))

    def validation_loss(self, X, y):
        """Loss of the *current* weights on the internal validation rows."""
        X = torch.from_numpy(np.asarray(X, dtype=np.float32)[self.val_idx_])
        y = torch.from_numpy(np.asarray(y, dtype=np.float32)[self.val_idx_])
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.pos_weight_))
        return self._loss(X, y, loss_fn)

    # ── inference ────────────────────────────────────────────────────────
    def predict_proba(self, X):
        self.net_.eval()
        with torch.no_grad():
            logits = self.net_(torch.from_numpy(np.asarray(X, dtype=np.float32)))
            p = torch.sigmoid(logits).numpy().astype(float)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    @property
    def n_params_(self):
        return int(sum(p.numel() for p in self.net_.parameters()))

    @property
    def layer_sizes_(self):
        return [self.n_features_in_, *self.hidden, 1]
