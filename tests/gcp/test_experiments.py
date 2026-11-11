from src.gcp.experiments import log_candidates, metrics_for_logging


class FakeAip:
    def __init__(self):
        self.calls = []

    def init(self, **kw):
        self.calls.append(("init", kw))

    def start_run(self, run):
        self.calls.append(("start_run", run))

    def log_params(self, p):
        self.calls.append(("params", p))

    def log_metrics(self, m):
        self.calls.append(("metrics", m))

    def end_run(self):
        self.calls.append(("end_run",))


def test_metrics_keep_numbers_and_flatten_the_confusion_matrix():
    result = {"name": "x", "pr_auc": 0.1, "n_train": 5, "confusion_matrix": {"tp": 2, "fp": 3}}
    assert metrics_for_logging(result) == {"pr_auc": 0.1, "n_train": 5.0, "cm_tp": 2.0, "cm_fp": 3.0}


def test_one_run_per_candidate_with_params_and_metrics():
    aip = FakeAip()
    results = [{"name": "logistic", "pr_auc": 0.13}, {"name": "xgboost", "pr_auc": 0.1}]
    params = {"logistic": {"C": 0.1}, "xgboost": {"max_depth": 3}}
    log_candidates(aip, "loan-default", "run-1", results, params, "p", "r")

    assert aip.calls[0] == ("init", {"project": "p", "location": "r", "experiment": "loan-default"})
    runs = [c[1] for c in aip.calls if c[0] == "start_run"]
    assert runs == ["logistic-run-1", "xgboost-run-1"]
    assert ("params", {"C": 0.1}) in aip.calls and ("metrics", {"pr_auc": 0.1}) in aip.calls
    assert [c for c in aip.calls if c[0] == "end_run"] == [("end_run",)] * 2
