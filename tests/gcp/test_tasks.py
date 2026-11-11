from src.gcp.storage import read_json, write_json
from src.gcp.tasks import metrics_uri, run_compare
from tests.gcp.test_storage import FakeClient


def test_run_compare_reads_candidate_files_and_writes_summary():
    c = FakeClient()
    run = "gs://b/pipeline-runs/r1"
    write_json(metrics_uri(run, "logistic"), {"name": "logistic", "pr_auc": 0.13, "gini": 0.33}, c)
    write_json(metrics_uri(run, "xgboost"), {"name": "xgboost", "pr_auc": 0.10, "gini": 0.30}, c)

    out = run_compare(run, ["logistic", "xgboost"], {"name": "ref", "pr_auc": 0.129}, "pr_auc", client=c)

    assert out["comparison"]["best"] == "logistic"
    assert [r["name"] for r in out["candidates"]] == ["logistic", "xgboost"]
    assert read_json(f"{run}/custom_pipeline_metrics.json", c) == out
