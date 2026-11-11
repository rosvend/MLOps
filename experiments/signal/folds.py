"""Fold generators and a scorer that returns per-fold metrics plus pooled out-of-fold scores."""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold


def time_folds(vintage: pd.Series, n_val: int = 4):
    """Expanding origin: validate each of the last n_val months on everything strictly earlier."""
    meses = sorted(vintage.unique())
    v = vintage.to_numpy()
    out = []
    for m in meses[-n_val:]:
        out.append((np.where(v < m)[0], np.where(v == m)[0]))
    return out


def stratified_folds(y, n_splits=5, n_repeats=3, seed=0):
    return list(RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed).split(y, y))


def score_model(model, X, y, folds, fit_params=None):
    """Fit a fresh clone per fold; return per-fold PR-AUC/ROC-AUC and out-of-fold scores."""
    rows, oof = [], {}
    for k, (tr, va) in enumerate(folds):
        m = clone(model).fit(X.iloc[tr], y.iloc[tr])
        s = m.predict_proba(X.iloc[va])[:, 1]
        oof[k] = (va, s)
        rows.append(
            {"fold": k, "pr_auc": average_precision_score(y.iloc[va], s), "roc_auc": roc_auc_score(y.iloc[va], s)}
        )
    return pd.DataFrame(rows), oof


def pooled(oof, y):
    """One vector of scores over every validated row (rows repeat across stratified repeats)."""
    idx = np.concatenate([v for v, _ in oof.values()])
    s = np.concatenate([s for _, s in oof.values()])
    return y.to_numpy()[idx], s
