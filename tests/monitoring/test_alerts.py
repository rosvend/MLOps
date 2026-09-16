"""An alert always logs; it also posts to a webhook when one is configured, in a body
that works for either Slack or Discord."""

import logging

import httpx
import pytest

from src.monitoring.alerts import send_alert
from src.monitoring.spec import MonitoringSpec


@pytest.fixture
def spec(tmp_path):
    return MonitoringSpec(
        reference_path="models/reference_data.parquet",
        predictions_db_path="data/monitoring/predictions.db",
        drift_share_threshold=0.3,
        flagged_share_threshold=0.25,
        numeric_sample_size_cutoff=1000,
        numeric_small_sample_stattest="ks",
        numeric_large_sample_stattest="wasserstein",
        categorical_stattest="chisquare",
        alpha=0.05,
        min_current_rows=30,
        webhook_url=None,
        offline_report_html="r.html",
        offline_report_json="r.json",
        live_report_html="r.html",
        live_report_json="r.json",
    )


def test_no_webhook_configured_only_logs(spec, caplog):
    with caplog.at_level(logging.WARNING):
        send_alert("dataset drift detected", {"drift_share": 0.5}, spec)

    assert "dataset drift detected" in caplog.text


def test_a_configured_webhook_receives_a_post(spec, caplog):
    llamadas = []

    def transporte(request: httpx.Request) -> httpx.Response:
        llamadas.append(request)
        return httpx.Response(200)

    spec = spec.model_copy(update={"webhook_url": "https://hooks.example.com/alert"})

    with caplog.at_level(logging.WARNING):
        send_alert(
            "flagged share exceeded",
            {"flagged_share": 0.4},
            spec,
            transport=httpx.MockTransport(transporte),
        )

    assert len(llamadas) == 1
    cuerpo = llamadas[0].content.decode()
    assert "flagged share exceeded" in cuerpo


def test_the_webhook_body_works_for_slack_and_discord(spec):
    import json

    llamadas = []

    def transporte(request: httpx.Request) -> httpx.Response:
        llamadas.append(json.loads(request.content))
        return httpx.Response(200)

    spec = spec.model_copy(update={"webhook_url": "https://hooks.example.com/alert"})
    send_alert("m", {}, spec, transport=httpx.MockTransport(transporte))

    cuerpo = llamadas[0]
    assert cuerpo["text"] == "m"       # Slack incoming webhook key
    assert cuerpo["content"] == "m"    # Discord incoming webhook key


def test_a_webhook_failure_never_raises(spec, caplog):
    """An alert about a broken system must not itself crash the request that triggered it."""

    def transporte(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to host")

    spec = spec.model_copy(update={"webhook_url": "https://unreachable.example.com"})

    with caplog.at_level(logging.WARNING):
        send_alert("m", {}, spec, transport=httpx.MockTransport(transporte))

    assert "m" in caplog.text


def test_a_non_success_webhook_response_is_treated_as_a_failure(spec, caplog):
    """httpx does not raise on 4xx/5xx by itself - without raise_for_status a rejected
    webhook would look delivered."""

    def transporte(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    spec = spec.model_copy(update={"webhook_url": "https://hooks.example.com/alert"})

    with caplog.at_level(logging.WARNING):
        send_alert("m", {}, spec, transport=httpx.MockTransport(transporte))

    assert "no se pudo entregar" in caplog.text
