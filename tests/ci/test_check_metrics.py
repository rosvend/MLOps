import json

import pytest

from src.ci.check_metrics import MetricsCheckError, check, main

MINIMUMS = {"pr_auc": 0.12, "gini": 0.30}


def test_passes_when_metrics_meet_minimums():
    check({"metrics": {"pr_auc": 0.129, "gini": 0.327}}, MINIMUMS)


def test_fails_when_a_metric_is_below_minimum():
    with pytest.raises(MetricsCheckError, match="pr_auc"):
        check({"metrics": {"pr_auc": 0.05, "gini": 0.327}}, MINIMUMS)


def test_reports_every_failing_metric():
    with pytest.raises(MetricsCheckError) as exc:
        check({"metrics": {"pr_auc": 0.05, "gini": 0.1}}, MINIMUMS)
    assert "pr_auc" in str(exc.value) and "gini" in str(exc.value)


@pytest.mark.parametrize(
    "meta",
    [
        {},
        {"metrics": None},
        {"metrics": {"pr_auc": 0.2}},
        {"metrics": {"pr_auc": "x", "gini": 0.4}},
        {"metrics": {"pr_auc": float("nan"), "gini": 0.4}},
    ],
)
def test_malformed_metadata_fails(meta):
    with pytest.raises(MetricsCheckError):
        check(meta, MINIMUMS)


def _write(tmp_path, meta):
    cfg = tmp_path / "thresholds.yaml"
    meta_file = tmp_path / "meta.json"
    meta_file.write_text(json.dumps(meta))
    cfg.write_text(f"meta_path: {meta_file}\nminimums:\n  pr_auc: 0.12\n  gini: 0.30\n")
    return cfg


def test_main_returns_zero_on_pass(tmp_path):
    assert main([str(_write(tmp_path, {"metrics": {"pr_auc": 0.13, "gini": 0.33}}))]) == 0


def test_main_returns_one_on_fail(tmp_path):
    assert main([str(_write(tmp_path, {"metrics": {"pr_auc": 0.01, "gini": 0.33}}))]) == 1


def test_main_returns_one_when_metadata_missing(tmp_path):
    cfg = tmp_path / "t.yaml"
    cfg.write_text(f"meta_path: {tmp_path / 'nope.json'}\nminimums:\n  pr_auc: 0.12\n")
    assert main([str(cfg)]) == 1
