"""The canary decision, and the metrics it reads from Cloud Monitoring (a fake client, no network)."""

from types import SimpleNamespace as NS

import pytest

from src.ci.canary import Decision, Stats, Thresholds, evaluate, fetch_stats

TH = Thresholds(min_requests=20, max_error_rate=0.02, max_p95_ms=6000, max_p95_ratio=2.0)


def test_a_healthy_candidate_passes():
    decision = evaluate(Stats(100, 0, 800.0), Stats(100, 0, 700.0), TH)

    assert decision.passed and decision.reasons == []


def test_too_few_requests_is_no_evidence_and_fails():
    decision = evaluate(Stats(5, 0, 800.0), Stats(100, 0, 700.0), TH)

    assert not decision.passed and any("requests" in r for r in decision.reasons)


def test_a_high_error_rate_fails():
    decision = evaluate(Stats(100, 5, 800.0), Stats(100, 0, 700.0), TH)

    assert not decision.passed and any("error" in r for r in decision.reasons)


def test_an_error_rate_at_the_limit_passes():
    assert evaluate(Stats(100, 2, 800.0), None, TH).passed


def test_a_slow_candidate_fails_the_absolute_cap():
    decision = evaluate(Stats(100, 0, 7000.0), None, TH)

    assert not decision.passed and any("p95" in r for r in decision.reasons)


def test_a_candidate_much_slower_than_the_previous_revision_fails():
    decision = evaluate(Stats(100, 0, 3000.0), Stats(100, 0, 1000.0), TH)

    assert not decision.passed and any("previous" in r for r in decision.reasons)


def test_no_latency_data_fails():
    assert not evaluate(Stats(100, 0, None), None, TH).passed


def test_without_a_previous_revision_only_the_absolute_checks_apply():
    assert evaluate(Stats(100, 0, 3000.0), None, TH).passed


def test_a_failing_probe_fails_the_canary_even_if_metrics_look_fine():
    decision = evaluate(Stats(100, 0, 800.0), None, TH, probe_failures=1)

    assert not decision.passed and any("probe" in r for r in decision.reasons)


def test_an_override_of_the_latency_cap_can_force_a_failure():
    th = Thresholds(**{**TH.__dict__, "max_p95_ms": 1})

    assert not evaluate(Stats(100, 0, 800.0), None, th).passed


class FakeMonitoring:
    def __init__(self, counts, p95):
        self.counts, self.p95, self.requests = counts, p95, []

    def list_time_series(self, request):
        self.requests.append(request)
        if "request_count" in request["filter"]:
            return [NS(metric=NS(labels={"response_code_class": cls}),
                       points=[NS(value=NS(int64_value=n, double_value=0.0)) for n in pts])
                    for cls, pts in self.counts.items()]
        return [NS(metric=NS(labels={}), points=[NS(value=NS(int64_value=0, double_value=v)) for v in self.p95])]


def test_stats_are_read_from_request_count_and_latency():
    client = FakeMonitoring({"2xx": [30, 12], "5xx": [1]}, [420.0, 910.0])

    stats = fetch_stats(client, "proj", "scoring-api", "scoring-api-00016-abc", start=1000, end=1300)

    assert stats == Stats(requests=43, errors_5xx=1, p95_ms=910.0)


def test_stats_filter_on_the_revision_and_the_window():
    client = FakeMonitoring({"2xx": [10]}, [100.0])

    fetch_stats(client, "proj", "scoring-api", "rev-1", start=1000, end=1300)

    assert all('revision_name="rev-1"' in r["filter"] for r in client.requests)
    assert all(r["name"] == "projects/proj" for r in client.requests)
    assert all(r["interval"] == {"start_time": {"seconds": 1000}, "end_time": {"seconds": 1300}} for r in client.requests)


def test_a_revision_with_no_series_has_zero_requests_and_no_latency():
    assert fetch_stats(FakeMonitoring({}, []), "proj", "svc", "rev", 0, 100) == Stats(0, 0, None)


def test_a_decision_is_printable():
    assert "PASS" in Decision(True, []).summary() and "FAIL" in Decision(False, ["x"]).summary()


def test_the_run_probes_waits_and_decides_from_the_metrics():
    from src.ci.canary import CanarySpec, run_canary

    spec = CanarySpec(project="proj", service="scoring-api",
                      probe={"requests_per_revision": 3, "soak_seconds": 30, "metrics_wait_seconds": 60},
                      thresholds=TH.model_dump())
    sent, slept = [], []
    clock = iter([1000, 1100])

    decision, candidate, previous = run_canary(
        spec, "rev-new", "https://new", "rev-old", "https://old", "tok", FakeMonitoring({"2xx": [40]}, [500.0]),
        {"records": []}, sleep=slept.append, now=lambda: next(clock), send=lambda url, p, t: (sent.append(url), (200, 10.0))[1])

    assert decision.passed and candidate.requests == 40 and previous.requests == 40
    assert sent == ["https://new", "https://old"] * 3
    assert slept == [10.0, 10.0, 10.0, 60]


def test_a_failing_probe_status_fails_the_run():
    from src.ci.canary import CanarySpec, run_canary

    spec = CanarySpec(project="p", service="s", probe={"requests_per_revision": 2, "soak_seconds": 2, "metrics_wait_seconds": 0},
                      thresholds=TH.model_dump())
    clock = iter([1, 2])

    decision, *_ = run_canary(spec, "r", "https://new", None, None, "t", FakeMonitoring({"2xx": [40]}, [100.0]), {},
                              sleep=lambda s: None, now=lambda: next(clock), send=lambda u, p, t: (500, 5.0))

    assert not decision.passed and any("probe" in r for r in decision.reasons)


def test_the_shipped_config_loads():
    import yaml

    from src.ci.canary import DEFAULT_CONFIG, CanarySpec

    spec = CanarySpec(**yaml.safe_load(DEFAULT_CONFIG.read_text()))

    assert spec.service == "scoring-api" and spec.thresholds.min_requests <= spec.probe.requests_per_revision
