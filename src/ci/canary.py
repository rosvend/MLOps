"""Canary analysis: judge the new Cloud Run revision from its own metrics, not only from smoke tests.

While the candidate has 10 % of the traffic, probe requests keep both revisions busy, then the request count, the
5xx share and the 95th percentile latency of each revision are read from Cloud Monitoring and compared with the
thresholds in config/ci/canary.yaml. A failure exits non-zero, which makes the Deploy workflow roll back.

    uv run python -m src.ci.canary --candidate-revision R --candidate-url URL --previous-revision P --previous-url URL
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config" / "ci" / "canary.yaml"
SMOKE_REQUEST = ROOT / "config" / "ci" / "smoke_request.json"


class Thresholds(BaseModel):
    min_requests: int
    max_error_rate: float
    max_p95_ms: float
    max_p95_ratio: float


class ProbeSpec(BaseModel):
    requests_per_revision: int
    soak_seconds: int
    metrics_wait_seconds: int


class CanarySpec(BaseModel):
    project: str
    service: str
    probe: ProbeSpec
    thresholds: Thresholds


@dataclass(frozen=True)
class Stats:
    requests: int
    errors_5xx: int
    p95_ms: float | None


@dataclass(frozen=True)
class Decision:
    passed: bool
    reasons: list[str]

    def summary(self) -> str:
        return "PASS" if self.passed else "FAIL: " + "; ".join(self.reasons)


def evaluate(candidate: Stats, previous: Stats | None, th: Thresholds, probe_failures: int = 0) -> Decision:
    reasons = []
    if candidate.requests < th.min_requests:
        reasons.append(f"only {candidate.requests} requests in the metrics, at least {th.min_requests} needed")
    elif candidate.errors_5xx / candidate.requests > th.max_error_rate:
        reasons.append(f"error rate {candidate.errors_5xx / candidate.requests:.1%} is above {th.max_error_rate:.1%}")

    if candidate.p95_ms is None:
        reasons.append("no latency data for the candidate")
    else:
        if candidate.p95_ms > th.max_p95_ms:
            reasons.append(f"p95 {candidate.p95_ms:.0f} ms is above the cap of {th.max_p95_ms:.0f} ms")
        if previous and previous.p95_ms and candidate.p95_ms > previous.p95_ms * th.max_p95_ratio:
            reasons.append(
                f"p95 {candidate.p95_ms:.0f} ms is more than {th.max_p95_ratio:g}x the previous revision's {previous.p95_ms:.0f} ms"
            )
    if probe_failures:
        reasons.append(f"{probe_failures} probe requests to the candidate failed")
    return Decision(not reasons, reasons)


def fetch_stats(client, project: str, service: str, revision: str, start: int, end: int) -> Stats:
    """Requests, 5xx count and p95 latency (ms) of one revision between two epoch seconds."""
    base = (
        'resource.type="cloud_run_revision" AND '
        f'resource.labels.service_name="{service}" AND resource.labels.revision_name="{revision}"'
    )
    window = {"start_time": {"seconds": start}, "end_time": {"seconds": end}}
    period = {"seconds": max(end - start, 60)}

    counts = client.list_time_series(request={
        "name": f"projects/{project}",
        "filter": f'metric.type="run.googleapis.com/request_count" AND {base}',
        "interval": window,
        "aggregation": {"alignment_period": period, "per_series_aligner": "ALIGN_SUM",
                        "cross_series_reducer": "REDUCE_SUM", "group_by_fields": ["metric.labels.response_code_class"]},
        "view": "FULL",
    })
    total = errors = 0
    for series in counts:
        n = sum(int(p.value.int64_value) for p in series.points)
        total += n
        if series.metric.labels.get("response_code_class") == "5xx":
            errors += n

    latencies = client.list_time_series(request={
        "name": f"projects/{project}",
        "filter": f'metric.type="run.googleapis.com/request_latencies" AND {base}',
        "interval": window,
        "aggregation": {"alignment_period": period, "per_series_aligner": "ALIGN_DELTA",
                        "cross_series_reducer": "REDUCE_PERCENTILE_95"},
        "view": "FULL",
    })
    values = [float(p.value.double_value) for series in latencies for p in series.points]
    return Stats(total, errors, max(values) if values else None)


def http_post(url: str, payload: dict, token: str) -> tuple[int, float]:
    """(HTTP status or 0 on a connection error, elapsed milliseconds)."""
    request = urllib.request.Request(
        f"{url}/predict/batch", data=json.dumps(payload).encode(), method="POST",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
    )
    inicio = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=90) as respuesta:
            status = respuesta.status
    except urllib.error.HTTPError as error:
        status = error.code
    except OSError:
        status = 0
    return status, (time.monotonic() - inicio) * 1000


def run_canary(spec: CanarySpec, candidate_revision: str, candidate_url: str, previous_revision: str | None,
               previous_url: str | None, token: str, client, payload: dict, max_p95_ms: float | None = None,
               sleep=time.sleep, now=time.time, send=http_post):
    """Probe both revisions for the soak time, wait for Cloud Monitoring, then decide."""
    th = spec.thresholds if max_p95_ms is None else spec.thresholds.model_copy(update={"max_p95_ms": max_p95_ms})
    start = int(now())
    n = spec.probe.requests_per_revision
    pause = spec.probe.soak_seconds / max(n, 1)
    failures = 0
    for _ in range(n):
        status, _ms = send(candidate_url, payload, token)
        failures += status != 200
        if previous_url:
            send(previous_url, payload, token)
        sleep(pause)

    sleep(spec.probe.metrics_wait_seconds)
    end = int(now())
    candidate = fetch_stats(client, spec.project, spec.service, candidate_revision, start, end)
    previous = fetch_stats(client, spec.project, spec.service, previous_revision, start, end) if previous_revision else None
    return evaluate(candidate, previous, th, failures), candidate, previous


def _table(candidate: Stats, previous: Stats | None, decision: Decision) -> str:
    def row(name: str, s: Stats | None) -> str:
        if s is None:
            return f"| {name} | n/a | n/a | n/a |"
        rate = f"{s.errors_5xx / s.requests:.1%}" if s.requests else "n/a"
        p95 = f"{s.p95_ms:.0f} ms" if s.p95_ms is not None else "n/a"
        return f"| {name} | {s.requests} | {rate} | {p95} |"

    return "\n".join([
        "| Revision | Requests | 5xx | p95 latency |", "| --- | --- | --- | --- |",
        row("candidate", candidate), row("previous", previous), "", f"Canary analysis: **{decision.summary()}**",
    ])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--candidate-revision", required=True)
    ap.add_argument("--candidate-url", required=True)
    ap.add_argument("--previous-revision")
    ap.add_argument("--previous-url")
    ap.add_argument("--max-p95-ms", type=float, help="override the latency cap (a way to force a failure in a test)")
    args = ap.parse_args(argv)

    from google.cloud import monitoring_v3

    spec = CanarySpec(**yaml.safe_load(Path(args.config).read_text()))
    payload = json.loads(SMOKE_REQUEST.read_text())
    decision, candidate, previous = run_canary(
        spec, args.candidate_revision, args.candidate_url, args.previous_revision, args.previous_url,
        os.environ["ID_TOKEN"], monitoring_v3.MetricServiceClient(), payload, args.max_p95_ms,
    )
    report = _table(candidate, previous, decision)
    print(report)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write(report + "\n")
    return 0 if decision.passed else 1


if __name__ == "__main__":
    sys.exit(main())
