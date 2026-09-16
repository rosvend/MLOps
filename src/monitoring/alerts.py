"""Alerting: always logged locally, also POSTed to a webhook if one is configured.

The body carries both `text` (Slack incoming webhooks) and `content` (Discord incoming
webhooks) - unknown keys are ignored by both, so one body works for either without
needing to know which service is on the other end.
"""

import logging

import httpx

from src.monitoring.spec import MonitoringSpec

_log = logging.getLogger(__name__)


def send_alert(
    message: str,
    payload: dict,
    spec: MonitoringSpec,
    transport: httpx.BaseTransport | None = None,
) -> None:
    _log.warning("ALERTA: %s | %s", message, payload)
    if not spec.webhook_url:
        return
    cuerpo = {"text": message, "content": message, **payload}
    try:
        with httpx.Client(transport=transport, timeout=5.0) as cliente:
            cliente.post(spec.webhook_url, json=cuerpo)
    except httpx.HTTPError as exc:
        # An alert about a broken system must not itself take down the caller.
        _log.warning("no se pudo entregar el webhook de alerta: %s", exc)
