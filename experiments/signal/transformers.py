"""Small fit-on-train transformers and a resampling wrapper, all safe to place inside a Pipeline."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import KFold
from sklearn.preprocessing import TargetEncoder


class ColumnDropper(TransformerMixin, BaseEstimator):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=[c for c in self.columns if c in X.columns])


class AddInteractions(TransformerMixin, BaseEstimator):
    """Row-wise products of column pairs; nothing is learned."""

    def __init__(self, pairs):
        self.pairs = pairs

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for a, b in self.pairs:
            X[f"{a}*{b}"] = X[a].astype(float) * X[b].astype(float)
        return X


class WoeBinner(TransformerMixin, BaseEstimator):
    """Quantile bins and smoothed weight of evidence per column, learned on the fit rows only."""

    def __init__(self, columns, n_bins=5, smoothing=0.5, keep_original=True):
        self.columns = columns
        self.n_bins = n_bins
        self.smoothing = smoothing
        self.keep_original = keep_original

    def fit(self, X, y):
        y = np.asarray(y)
        self.edges_, self.woe_ = {}, {}
        pos, neg = y.sum(), len(y) - y.sum()
        for c in self.columns:
            edges = np.unique(np.quantile(X[c].astype(float), np.linspace(0, 1, self.n_bins + 1)[1:-1]))
            b = np.searchsorted(edges, X[c].astype(float), side="right")
            woe = []
            for k in range(len(edges) + 1):
                p, n = y[b == k].sum() + self.smoothing, (b == k).sum() - y[b == k].sum() + self.smoothing
                woe.append(np.log((p / pos) / (n / neg)))
            self.edges_[c], self.woe_[c] = edges, np.array(woe)
        return self

    def transform(self, X):
        X = X.copy()
        for c in self.columns:
            b = np.searchsorted(self.edges_[c], X[c].astype(float), side="right")
            X[f"{c}_woe"] = self.woe_[c][b]
            if not self.keep_original:
                X = X.drop(columns=c)
        return X


class Resampled(ClassifierMixin, BaseEstimator):
    """Resample the rows handed to `fit` only; prediction is untouched. Methods: over, under, smote."""

    def __init__(self, estimator, method="over", ratio=1.0, k=5, seed=0):
        self.estimator = estimator
        self.method = method
        self.ratio = ratio
        self.k = k
        self.seed = seed

    def _smote(self, X, n_new, rng):
        k = min(self.k, len(X) - 1)
        nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
        neigh = nn.kneighbors(X, return_distance=False)[:, 1:]
        base = rng.integers(0, len(X), n_new)
        pick = neigh[base, rng.integers(0, k, n_new)]
        return X[base] + rng.random((n_new, 1)) * (X[pick] - X[base])

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        rng = np.random.default_rng(self.seed)
        pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        if self.method == "under":
            keep = rng.choice(neg, int(len(pos) / self.ratio), replace=False)
            idx = np.concatenate([pos, keep])
            Xr, yr = X[idx], y[idx]
        else:
            n_new = int(len(neg) * self.ratio) - len(pos)
            if self.method == "over":
                extra = X[rng.choice(pos, n_new, replace=True)]
            elif self.method == "smote":
                extra = self._smote(X[pos], n_new, rng)
            else:
                raise ValueError(self.method)
            Xr, yr = np.vstack([X, extra]), np.concatenate([y, np.ones(n_new, dtype=y.dtype)])
        self.resampled_positive_share_ = float(yr.mean())
        self.estimator_ = clone(self.estimator).fit(Xr, yr)
        self.classes_ = self.estimator_.classes_
        return self

    def predict_proba(self, X):
        return self.estimator_.predict_proba(np.asarray(X, dtype=float))


class AddTargetEncoding(TransformerMixin, BaseEstimator):
    """Smoothed target mean per category; fit_transform cross-fits so a row never sees its own label."""

    def __init__(self, columns, cv=5, seed=0):
        self.columns = columns
        self.cv = cv
        self.seed = seed

    def _cats(self, X):
        return X[list(self.columns)].astype("object").fillna("NA").astype(str)

    def _names(self):
        return [f"{c}_te" for c in self.columns]

    def fit(self, X, y):
        self.encoder_ = TargetEncoder(target_type="binary", cv=KFold(self.cv, shuffle=True, random_state=self.seed))
        self.encoder_.fit(self._cats(X), y)
        return self

    def fit_transform(self, X, y=None):
        self.encoder_ = TargetEncoder(target_type="binary", cv=KFold(self.cv, shuffle=True, random_state=self.seed))
        enc = self.encoder_.fit_transform(self._cats(X), y)
        return X.assign(**dict(zip(self._names(), enc.T)))

    def transform(self, X):
        return X.assign(**dict(zip(self._names(), self.encoder_.transform(self._cats(X)).T)))
