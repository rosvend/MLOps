import io
import json

import joblib
import numpy as np
import pandas as pd
import pytest

from src.gcp.gate import Bundle, bootstrap_diff_ci, decide, evaluation_rows, run_gate
from src.gcp.gate_spec import Bootstrap, default_gate_spec
from src.gcp.spec import default_dataset_spec

SPEC = default_gate_spec()
BOOT = Bootstrap(resamples=200, confidence=0.95, seed=0)


class ColumnModel:
    """Scores a row with one of its columns, so the test controls every prediction."""

    def __init__(self, column="a"):
        self.column = column

    def predict_proba(self, X):
        p = X[self.column].to_numpy(dtype=float)
        return np.column_stack([1 - p, p])


def meta(features=("a", "b"), threshold=0.5, share=0.2, **extra):
    base = {"model": "m", "threshold": threshold, "flagged_share_at_fit": share, "features": list(features)}
    return {**base, **extra}


def data(n=4000, seed=1):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.1).astype(int)
    good = np.clip(0.15 + 0.5 * y + rng.normal(0, 0.2, n), 0, 1)   # separates the classes well
    weak = np.clip(0.3 + 0.1 * y + rng.normal(0, 0.2, n), 0, 1)    # barely separates them
    return pd.DataFrame({"a": good, "b": weak}), pd.Series(y)


def test_identical_models_are_kept_and_the_reasons_say_why():
    X, y = data()
    same = Bundle(ColumnModel("a"), meta(threshold=0.45))
    d = decide(same, same, X, y, SPEC)
    assert d["decision"] == "keep"
    assert d["metrics"]["pr_auc_gain"] == 0
    assert any("PR-AUC" in r for r in d["reasons"])
    assert d["checks"]["bootstrap_excludes_zero"]["passed"] is False


def test_clearly_better_candidate_is_promoted():
    X, y = data()
    d = decide(Bundle(ColumnModel("a"), meta(threshold=0.45)), Bundle(ColumnModel("b"), meta()), X, y, SPEC)
    assert d["decision"] == "promote", d["reasons"]
    assert d["metrics"]["pr_auc_gain"] > SPEC.promotion.min_pr_auc_gain
    assert d["metrics"]["diff_ci"][0] > 0


def test_worse_candidate_is_kept():
    X, y = data()
    d = decide(Bundle(ColumnModel("b"), meta()), Bundle(ColumnModel("a"), meta()), X, y, SPEC)
    assert d["decision"] == "keep"
    assert d["checks"]["pr_auc_gain"]["passed"] is False


def test_gain_below_the_margin_is_kept():
    X, y = data()
    strict = SPEC.model_copy(update={"promotion": SPEC.promotion.model_copy(update={"min_pr_auc_gain": 0.9})})
    d = decide(Bundle(ColumnModel("a"), meta(threshold=0.45)), Bundle(ColumnModel("b"), meta()), X, y, strict)
    assert d["decision"] == "keep"
    assert d["checks"]["pr_auc_gain"]["passed"] is False


def test_bootstrap_is_seeded_and_brackets_a_real_difference():
    X, y = data()
    a, b = X["a"].to_numpy(), X["b"].to_numpy()
    assert bootstrap_diff_ci(y, a, b, BOOT) == bootstrap_diff_ci(y, a, b, BOOT)
    lo, hi = bootstrap_diff_ci(y, a, b, BOOT)
    assert 0 < lo < hi
    assert bootstrap_diff_ci(y, a, a, BOOT) == (0.0, 0.0)


def test_small_noisy_sample_does_not_clear_the_bootstrap():
    X, y = data(n=60, seed=3)
    loose = SPEC.model_copy(update={"promotion": SPEC.promotion.model_copy(update={"min_pr_auc_gain": 0.0})})
    d = decide(Bundle(ColumnModel("a"), meta(threshold=0.45)), Bundle(ColumnModel("b"), meta()), X, y, loose)
    lo, _ = d["metrics"]["diff_ci"]
    assert d["checks"]["bootstrap_excludes_zero"]["passed"] == (lo > 0)


@pytest.mark.parametrize(
    "candidate_meta, failed",
    [
        ({"features": ["a", "b"], "flagged_share_at_fit": 0.2}, "meta_has_threshold"),
        (meta(features=("b", "a")), "same_features"),
        (meta(threshold=0.99), "flagged_share_in_band"),
        (meta(threshold=0.0), "flagged_share_in_band"),
    ],
)
def test_sanity_failures_keep_the_champion(candidate_meta, failed):
    X, y = data()
    d = decide(Bundle(ColumnModel("a"), candidate_meta), Bundle(ColumnModel("b"), meta()), X, y, SPEC)
    assert d["decision"] == "keep"
    assert d["checks"][failed]["passed"] is False
    assert any(failed in r for r in d["reasons"])


def test_missing_scores_keep_the_champion():
    X, y = data()
    X.loc[0, "a"] = np.nan
    d = decide(Bundle(ColumnModel("a"), meta(threshold=0.45)), Bundle(ColumnModel("b"), meta()), X, y, SPEC)
    assert d["decision"] == "keep"
    assert d["checks"]["no_missing_scores"]["passed"] is False


def test_evaluation_rows_use_only_test_rows_when_a_split_column_exists():
    ds = default_dataset_spec()
    frame = pd.DataFrame(
        {"a": [1, 2, 3], ds.label_column: [0, 1, 0], ds.split_column: ["TRAIN", ds.test_value, ds.test_value]}
    )
    X, y = evaluation_rows(frame, ds)
    assert list(X.columns) == ["a"] and len(X) == 2 and list(y) == [1, 0]
    X2, _ = evaluation_rows(frame.drop(columns=ds.split_column), ds)
    assert len(X2) == 3


def _fake_fetch(models):
    def fetch(uri):
        base, _, name = uri.rpartition("/")
        model, m = models[base]
        if name.endswith(".joblib"):
            buf = io.BytesIO()
            joblib.dump(model, buf)
            return buf.getvalue()
        return json.dumps(m).encode()

    return fetch


def test_run_gate_end_to_end_with_fakes_states_the_limitation():
    ds = default_dataset_spec()
    X, y = data()
    frame = X.assign(**{ds.label_column: y, ds.split_column: ds.test_value})
    models = {
        "gs://b/run/models/m": (ColumnModel("a"), meta(threshold=0.45)),
        "gs://b/champion": (ColumnModel("b"), meta()),
    }
    d = run_gate("gs://b/run", "m", "gs://b/champion", "file.csv", SPEC, ds, _fake_fetch(models), lambda uri: frame)
    assert d["decision"] == "promote"
    assert d["candidate"] == {"run_dir": "gs://b/run", "name": "m"}
    assert d["evaluation"] == {"uri": "file.csv", "rows": len(X), "positives": int(y.sum())}
    assert "not a fresh holdout" in d["limitation"]

