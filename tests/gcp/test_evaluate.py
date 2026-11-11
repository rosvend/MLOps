import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.gcp.data import SplitData
from src.gcp.evaluate import build_candidate, train_and_evaluate
from src.models.training_spec import Candidate


def _data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = pd.Series((X["a"] + rng.normal(size=n) > 1).astype(int))
    h = n // 2
    return SplitData(X[:h], y[:h], X[h:].reset_index(drop=True), y[h:].reset_index(drop=True))


def test_reports_every_metric_and_timing():
    out = train_and_evaluate("logistic", LogisticRegression(), _data(), flagged_share=0.2)
    for key in ("pr_auc", "roc_auc", "gini", "f1", "precision", "recall", "brier",
                "fit_seconds", "predict_ms_per_1k", "n_train", "n_test"):
        assert key in out
    assert out["name"] == "logistic"
    assert out["n_train"] == 200 and out["n_test"] == 200


def test_threshold_flags_the_configured_share_of_test():
    m = train_and_evaluate("logistic", LogisticRegression(), _data(), flagged_share=0.2)["confusion_matrix"]
    flagged = m["tp"] + m["fp"]
    assert abs(flagged / 200 - 0.2) < 0.02


def test_model_is_fit_on_train_only():
    data = _data()
    model = LogisticRegression()
    train_and_evaluate("m", model, data, flagged_share=0.2)
    assert model.n_features_in_ == 2 and model.coef_.shape == (1, 2)


def test_build_candidate_applies_params_over_fixed_and_scale():
    cand = Candidate(
        estimator="sklearn.linear_model.LogisticRegression", scale=True, fixed={"max_iter": 50}
    )
    pipe = build_candidate(cand, {"C": 0.5}, seed=1)
    model = pipe.named_steps["model"]
    assert model.C == 0.5 and model.max_iter == 50
    assert "scale" in pipe.named_steps
