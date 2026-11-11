import json

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pytest

from src.reports import phase2 as p2
from src.reports import phase2_plots as plots

ENTRIES = [
    {"id": "local_a", "label": "A", "platform": "local", "probabilistic": True},
    {"id": "cloud_a", "label": "A", "platform": "cloud", "probabilistic": True},
    {"id": "automl_v2", "label": "V2", "platform": "automl", "probabilistic": True, "optional": True},
]


@pytest.fixture
def preds():
    rng = np.random.default_rng(0)
    y = (rng.random(600) < 0.1).astype(int)
    s = np.clip(y * 0.3 + rng.random(600) * 0.7, 0, 1)
    return pd.DataFrame({"y": y, "local_a": s, "cloud_a": s})


def test_flag_threshold_flags_the_requested_share():
    s = np.arange(100) / 100
    assert p2.flagged(s, 0.2).sum() == 20


def test_accuracy_at_operating_point():
    y = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    s = np.array([0.9, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    assert p2.accuracy(y, s, 0.2) == pytest.approx(0.9)


def test_nobody_defaults_baseline():
    out = p2.trivial_baseline(np.array([1, 0, 0, 0]))
    assert out["accuracy"] == 0.75 and out["recall"] == 0.0


def test_bootstrap_is_seeded_and_brackets_the_point(preds):
    a = p2.bootstrap(preds.y, preds.local_a, 0.157, n_boot=100, seed=1)
    b = p2.bootstrap(preds.y, preds.local_a, 0.157, n_boot=100, seed=1)
    assert a == b
    lo, hi = a["pr_auc"]
    assert lo < p2.point_metrics(preds.y, preds.local_a, 0.157)["pr_auc"] < hi


def test_paired_difference_of_identical_scores_is_zero(preds):
    lo, hi = p2.paired_diff(preds.y, preds.local_a, preds.cloud_a, "pr_auc", n_boot=50, seed=1)
    assert lo == hi == 0.0


def test_parse_automl_lines_and_fill_unscored():
    lines = [
        json.dumps({"instance": {"row_id": "2"}, "prediction": {"classes": ["0", "1"], "scores": [0.7, 0.3]}}),
        json.dumps({"instance": {"row_id": "0"}, "prediction": {"classes": ["1", "0"], "scores": [0.9, 0.1]}}),
    ]
    s = p2.parse_automl(lines, "row_id", n_rows=4)
    assert s[0] == 0.9 and s[2] == 0.3 and np.isnan(s[1]) and len(s) == 4
    assert p2.fill_unscored(s).tolist() == [0.9, 0.0, 0.3, 0.0]


class FakeModel:
    def predict_proba(self, X):
        p = X["x"].to_numpy()
        return np.c_[1 - p, p]


def test_collect_predictions_scores_only_the_test_rows():
    frame = pd.DataFrame({"x": [0.1, 0.9, 0.5, 0.2], "defaulted": [0, 1, 0, 0], "split": ["TRAIN", "TEST", "TEST", "TEST"]})
    out = p2.collect_predictions(
        frame, label="defaulted", split="split", test_value="TEST",
        scorers={"local_a": lambda X: FakeModel().predict_proba(X)[:, 1]},
        extra={"automl_v1": pd.Series([0.1, np.nan, 0.4])},
    )
    assert out["y"].tolist() == [1, 0, 0]
    assert out["local_a"].tolist() == [0.9, 0.5, 0.2]
    assert np.isnan(out["automl_v1"][1])


def test_metrics_table_marks_missing_optional_entry_pending(preds):
    rows = p2.metrics_table(preds, ENTRIES, share=0.157, n_boot=30, seed=1, reference="local_a")
    by_id = {r["id"]: r for r in rows}
    assert by_id["automl_v2"]["status"] == "pending"
    assert by_id["local_a"]["status"] == "ok"
    assert by_id["local_a"]["accuracy"] == by_id["cloud_a"]["accuracy"]
    assert "pr_auc_ci" in by_id["local_a"] and "brier" in by_id["local_a"]


def test_non_probabilistic_entry_has_no_brier(preds):
    entries = [{"id": "local_a", "label": "H", "platform": "local", "probabilistic": False}]
    row = p2.metrics_table(preds, entries, 0.157, 20, 1, "local_a")[0]
    assert row["brier"] is None


def test_reproducibility_reports_max_abs_diff(preds):
    preds["cloud_a"] = preds["local_a"] + 1e-12
    out = p2.reproducibility(preds, [("a", "local_a", "cloud_a")])
    assert 0 < out["a"] < 1e-9


def test_cv_vs_oot_gap():
    champion = {"comparison": [{"model": "logistic", "cv_mean": 0.16, "cv_std": 0.02, "pr_auc": 0.13}]}
    row = p2.cv_vs_oot(champion)[0]
    assert row["gap"] == pytest.approx(0.03)


def test_time_fold_summary():
    folds = pd.DataFrame({"fold": [0, 1], "pr_auc": [0.1, 0.3], "roc_auc": [0.6, 0.7]})
    out = p2.time_fold_summary(folds)
    assert out["pr_auc"]["min"] == 0.1 and out["pr_auc"]["max"] == 0.3 and out["pr_auc"]["mean"] == pytest.approx(0.2)


def test_render_fills_placeholders_and_rejects_unknown():
    assert p2.render("a {{ x }} b", {"x": "1"}) == "a 1 b"
    with pytest.raises(KeyError):
        p2.render("{{ missing }}", {})


def test_markdown_table_formats_pending_and_values():
    rows = [
        {"label": "A", "platform": "Local", "status": "ok", "pr_auc": 0.1234, "pr_auc_ci": [0.1, 0.15]},
        {"label": "B", "platform": "Cloud", "status": "pending"},
    ]
    md = p2.markdown_table(rows, [("pr_auc", "PR-AUC")])
    assert "0.123 [0.100, 0.150]" in md and "pending" in md.splitlines()[-1]


def test_plots_write_pngs(preds, tmp_path):
    rows = p2.metrics_table(preds, ENTRIES, 0.157, 20, 1, "local_a")
    platforms = {"local": "Local", "cloud": "Cloud", "automl": "AutoML"}
    plots.metric_bars(rows, platforms, tmp_path / "bars.png")
    plots.pr_curves(preds, ENTRIES, platforms, tmp_path / "pr.png")
    plots.accuracy_vs_baseline(rows, 0.9, platforms, tmp_path / "acc.png")
    cv = [{"model": "logistic", "cv_mean": 0.16, "cv_std": 0.02, "pr_auc": 0.13, "gap": 0.03}]
    plots.cv_vs_oot(cv, {"logistic": (0.1, 0.16)}, tmp_path / "cv.png")
    for name in ("bars", "pr", "acc", "cv"):
        assert (tmp_path / f"{name}.png").stat().st_size > 1000
