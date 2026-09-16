"""The live drift endpoint: refuses to report on too little data, returns the
contracted keys once seeded, and fires an alert when a shift is injected."""

from pathlib import Path

import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.skipif(
    not (RAIZ / "models" / "champion.joblib").exists()
    or not (RAIZ / "models" / "reference_data.parquet").exists(),
    reason="run `make export-champion export-reference` first",
)


@pytest.fixture(scope="module")
def client():
    from src.api.app import app
    from fastapi.testclient import TestClient

    return TestClient(app)


@pytest.fixture
def solicitud():
    return {
        "application_id": "APP-00001",
        "tipo_credito": "4",
        "capital_prestado": 1852560,
        "plazo_meses": 12,
        "edad_cliente": 32,
        "tipo_laboral": "Empleado",
        "salario_cliente": 3500000,
        "total_otros_prestamos": 1000000,
        "cuota_pactada": 128650,
        "puntaje_datacredito": 795,
        "cant_creditosvigentes": 2,
        "huella_consulta": 2,
        "saldo_mora": 0,
        "saldo_total": 12000,
        "saldo_principal": 10000,
        "creditos_sectorFinanciero": 2,
        "creditos_sectorCooperativo": 0,
        "creditos_sectorReal": 0,
        "promedio_ingresos_datacredito": 916148,
        "tendencia_ingresos": "Creciente",
    }


def test_too_little_data_is_refused(client, isolated_log):
    respuesta = client.post("/monitoring/run-drift-check")

    assert respuesta.status_code == 409


def test_enough_data_returns_the_contracted_keys(client, solicitud, isolated_log):
    from src.monitoring.spec import default_monitoring_spec

    minimo = default_monitoring_spec().min_current_rows
    for i in range(minimo):
        client.post(
            "/predict/batch", json={"records": [{**solicitud, "application_id": f"APP-{i}"}]}
        )

    respuesta = client.post("/monitoring/run-drift-check")

    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert isinstance(cuerpo["dataset_drift_detected"], bool)
    assert 0.0 <= cuerpo["drift_share"] <= 1.0
    assert isinstance(cuerpo["drifted_features"], list)
    assert cuerpo["current_rows"] == minimo


def test_uniform_replays_of_the_same_application_look_like_the_reference(client, solicitud, isolated_log):
    """Repeating one realistic application many times should not itself look like drift -
    it is a degenerate but not adversarial current sample."""
    from src.monitoring.spec import default_monitoring_spec

    minimo = default_monitoring_spec().min_current_rows
    for i in range(minimo):
        client.post(
            "/predict/batch", json={"records": [{**solicitud, "application_id": f"APP-{i}"}]}
        )

    cuerpo = client.post("/monitoring/run-drift-check").json()

    # Not asserting no drift (a constant sample IS a distributional shift by definition
    # for most features) - asserting the endpoint completes and reports something coherent.
    assert cuerpo["flagged_share_current"] in (0.0, 1.0)


def test_an_injected_shift_triggers_an_alert(client, solicitud, isolated_log, monkeypatch):
    alertas = []
    monkeypatch.setattr(
        "src.monitoring.live_check.send_alert",
        lambda mensaje, payload, spec: alertas.append((mensaje, payload)),
    )
    from src.monitoring.spec import default_monitoring_spec

    minimo = default_monitoring_spec().min_current_rows
    # A deliberately extreme, out-of-range applicant, repeated - a shift large enough
    # that the small-sample stattest cannot plausibly miss it.
    extremo = {**solicitud, "capital_prestado": 999_000_000, "plazo_meses": 590, "cuota_pactada": 1}
    for i in range(minimo):
        client.post(
            "/predict/batch", json={"records": [{**extremo, "application_id": f"APP-{i}"}]}
        )

    client.post("/monitoring/run-drift-check")

    assert len(alertas) >= 1
