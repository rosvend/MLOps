import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from experiments.signal.transformers import (
    AddInteractions,
    ColumnDropper,
    AddTargetEncoding,
    Resampled,
    WoeBinner,
)


def _frame(n=600, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(2 * x - 3)))).astype(int)
    return pd.DataFrame({"x": x, "z": rng.normal(size=n)}), y


def test_woe_bins_come_from_fit_rows_only():
    X, y = _frame()
    t = WoeBinner(["x"], n_bins=4).fit(X.iloc[:300], y[:300])
    out = t.transform(X.iloc[300:] * 100)
    assert np.isfinite(out["x_woe"]).all() and out["x_woe"].nunique() <= 4


def test_woe_is_monotone_with_signal():
    X, y = _frame()
    t = WoeBinner(["x"], n_bins=5).fit(X, y)
    w = t.transform(X).groupby(pd.qcut(X.x, 5), observed=True)["x_woe"].mean().to_numpy()
    assert w[-1] > w[0]


def test_interactions_and_dropper():
    X, _ = _frame(10)
    out = AddInteractions([("x", "z")]).fit(X).transform(X)
    assert np.allclose(out["x*z"], X.x * X.z)
    assert list(ColumnDropper(["z"]).fit(X).transform(X).columns) == ["x"]


def test_resampled_changes_only_training_and_balances():
    X, y = _frame(2000)
    for method in ("over", "under", "smote"):
        r = Resampled(LogisticRegression(), method=method, seed=0).fit(X, y)
        assert r.resampled_positive_share_ > y.mean() * 2
        assert r.predict_proba(X.iloc[:5]).shape == (5, 2)
        assert len(X) == 2000


def test_target_encoding_is_cross_fitted_on_training_rows():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"c": rng.choice(["a", "b", "c"], 900)})
    y = ((X.c == "a") & (rng.random(900) < 0.5)).astype(int).to_numpy()
    t = AddTargetEncoding(["c"], cv=5, seed=0)
    train = t.fit_transform(X, y)
    test = t.transform(X)
    assert "c_te" in train and not np.allclose(train["c_te"], test["c_te"])  # in-sample rows differ
    assert test.groupby("c")["c_te"].mean().idxmax() == "a"
