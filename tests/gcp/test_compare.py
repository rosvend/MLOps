from src.gcp.compare import compare

RES = [
    {"name": "logistic", "pr_auc": 0.13, "gini": 0.33},
    {"name": "xgboost", "pr_auc": 0.11, "gini": 0.30},
]
REF = {"name": "local_champion", "pr_auc": 0.129, "gini": 0.327}


def test_best_is_highest_primary_metric():
    assert compare(RES, REF, "pr_auc")["best"] == "logistic"


def test_deltas_are_against_the_reference():
    out = compare(RES, REF, "pr_auc")
    assert out["delta_vs_reference"]["xgboost"]["pr_auc"] == 0.11 - 0.129


def test_nan_score_cannot_win():
    res = [{"name": "a", "pr_auc": float("nan")}, {"name": "b", "pr_auc": 0.01}]
    assert compare(res, {"name": "r", "pr_auc": 0.0}, "pr_auc")["best"] == "b"
