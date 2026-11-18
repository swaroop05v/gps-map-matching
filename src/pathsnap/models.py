"""Model zoo: candidate classifiers from different algorithm families."""
import os
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
N_JOBS = int(os.environ.get("PATHSNAP_N_JOBS", "-1"))  # thread budget (lets two experiments share the CPU)


class TorchMLP:
    """Small fully-connected network trained with binary cross-entropy (Adam).

    Keeps an epoch-wise history of training and validation loss.
    """

    def __init__(self, hidden=(128, 64), lr=1e-3, epochs=20, batch_size=2048, dropout=0.1,
                 weight_decay=1e-5, patience=4, seed=SEED):
        self.hidden, self.lr, self.epochs, self.batch_size = hidden, lr, epochs, batch_size
        self.dropout, self.weight_decay, self.patience, self.seed = dropout, weight_decay, patience, seed
        self.history = {"train_loss": [], "val_loss": []}

    def _net(self, d):
        import torch  # imported lazily: torch's OpenMP runtime slows LightGBM down ~9x if loaded together
        layers, prev = [], d
        for h in self.hidden:
            layers += [torch.nn.Linear(prev, h), torch.nn.ReLU(), torch.nn.Dropout(self.dropout)]
            prev = h
        layers.append(torch.nn.Linear(prev, 1))
        return torch.nn.Sequential(*layers)

    def fit(self, X, y, X_val=None, y_val=None):
        import torch
        if N_JOBS > 0:
            torch.set_num_threads(N_JOBS)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.scaler = StandardScaler().fit(X)
        Xt = torch.tensor(self.scaler.transform(X), dtype=torch.float32)
        yt = torch.tensor(np.asarray(y), dtype=torch.float32)
        if X_val is not None:
            Xv = torch.tensor(self.scaler.transform(X_val), dtype=torch.float32)
            yv = torch.tensor(np.asarray(y_val), dtype=torch.float32)
        self.net = self._net(X.shape[1])
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        lossf = torch.nn.BCEWithLogitsLoss()
        best, best_state, bad = np.inf, None, 0
        self.history = {"train_loss": [], "val_loss": []}
        for ep in range(self.epochs):
            self.net.train()
            perm = torch.randperm(len(Xt))
            tot = 0.0
            for i in range(0, len(Xt), self.batch_size):
                idx = perm[i:i + self.batch_size]
                opt.zero_grad()
                loss = lossf(self.net(Xt[idx]).squeeze(1), yt[idx])
                loss.backward()
                opt.step()
                tot += loss.item() * len(idx)
            self.history["train_loss"].append(tot / len(Xt))
            if X_val is not None:
                self.net.eval()
                with torch.no_grad():
                    vl = lossf(self.net(Xv).squeeze(1), yv).item()
                self.history["val_loss"].append(vl)
                if vl < best - 1e-5:
                    best, bad = vl, 0
                    best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= self.patience:
                        break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        import torch
        self.net.eval()
        with torch.no_grad():
            z = self.net(torch.tensor(self.scaler.transform(X), dtype=torch.float32)).squeeze(1)
            p = torch.sigmoid(z).numpy()
        return np.column_stack([1 - p, p])


def make_model(name, params=None):
    params = params or {}
    if name == "logreg":
        return Pipeline([("scale", StandardScaler()),
                         ("clf", LogisticRegression(C=params.get("C", 1.0), max_iter=2000, random_state=SEED))])
    if name == "gnb":
        return Pipeline([("scale", StandardScaler()), ("clf", GaussianNB(var_smoothing=params.get("var_smoothing", 1e-9)))])
    if name == "knn":
        return Pipeline([("scale", StandardScaler()),
                         ("clf", KNeighborsClassifier(n_neighbors=params.get("n_neighbors", 25), weights="distance", n_jobs=-1))])
    if name == "rf":
        return RandomForestClassifier(n_estimators=params.get("n_estimators", 150), max_depth=params.get("max_depth", 16),
                                      min_samples_leaf=params.get("min_samples_leaf", 5), max_samples=params.get("max_samples", 0.3),
                                      n_jobs=-1, random_state=SEED)
    if name == "hgb":
        return HistGradientBoostingClassifier(max_iter=params.get("max_iter", 300), learning_rate=params.get("learning_rate", 0.1),
                                              max_leaf_nodes=params.get("max_leaf_nodes", 31), early_stopping=False, random_state=SEED)
    if name == "lgbm":
        import lightgbm as lgb
        defaults = dict(n_estimators=400, learning_rate=0.05, num_leaves=63, max_depth=-1, min_child_samples=20,
                        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=0.0)
        defaults.update(params)
        return lgb.LGBMClassifier(**defaults, random_state=SEED, n_jobs=N_JOBS, verbose=-1)
    if name == "mlp":
        return TorchMLP(**params)
    raise ValueError(name)


MODEL_INFO = {
    "logreg": ("Logistic Regression", "Linear / probabilistic", "StandardScaler, L2, C=1.0"),
    "gnb": ("Gaussian Naive Bayes", "Probabilistic (generative)", "StandardScaler, var_smoothing=1e-9"),
    "knn": ("KNN (distance-weighted)", "Instance-based", "k=25, Euclidean on scaled features, fit on 200k-row subsample"),
    "rf": ("Random Forest", "Ensemble (bagging)", "150 trees, max_depth=16, min_leaf=5, max_samples=0.3"),
    "hgb": ("HistGradientBoosting", "Boosting", "300 iters, lr=0.1, 31 leaves"),
    "lgbm": ("LightGBM", "Boosting", "400 trees, lr=0.05, 63 leaves, subsample=0.8, colsample=0.8"),
    "mlp": ("MLP (PyTorch)", "Neural network", "128-64 ReLU, dropout 0.1, Adam lr=1e-3, BCE, early stop (patience 4)"),
}


def timed_fit(model, X, y, **kw):
    t0 = time.perf_counter()
    model.fit(X, y, **kw)
    return time.perf_counter() - t0
