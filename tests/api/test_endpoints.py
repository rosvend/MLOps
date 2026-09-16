"""The HTTP surface: validation at the edge, honest health, bounded batches."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

RAIZ = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.skipif(
    not (RAIZ / "models" / "champion.joblib").exists(), reason="run `make export-champion` first"
)


@pytest.fixture(scope="module")
def client():
    from src.api.app import app

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


def test_health_reports_the_loaded_model(client):
    cuerpo = client.get("/health").json()

    assert cuerpo["status"] == "ok"
    assert cuerpo["model_loaded"] is True
    assert cuerpo["threshold"] > 0
    assert cuerpo["n_features"] == 35


def test_health_does_not_claim_an_online_store(client):
    """Stage 6 has none; a health check implying otherwise would be lying."""
    cuerpo = client.get("/health").json()

    assert cuerpo["online_store"] is None
    assert "offline" in cuerpo["feature_store"] or "registry" in cuerpo["feature_store"]


def test_a_batch_is_scored(client, solicitud):
    respuesta = client.post("/predict/batch", json={"records": [solicitud]})

    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert cuerpo["model"] == "logistic"
    pred = cuerpo["predictions"][0]
    assert pred["application_id"] == "APP-00001"
    assert 0.0 <= pred["probability_default"] <= 1.0
    assert pred["review_flag"] == (pred["probability_default"] >= cuerpo["threshold"])


def test_several_records_come_back_in_order(client, solicitud):
    registros = [{**solicitud, "application_id": f"APP-{i:05d}"} for i in range(5)]

    cuerpo = client.post("/predict/batch", json={"records": registros}).json()

    assert [p["application_id"] for p in cuerpo["predictions"]] == [
        r["application_id"] for r in registros
    ]
    assert cuerpo["flagged"] == sum(p["review_flag"] for p in cuerpo["predictions"])


def test_optional_bureau_fields_may_be_absent(client, solicitud):
    """Missing bureau data is information the model handles, not a reason to refuse."""
    sin_bureau = {k: v for k, v in solicitud.items()
                  if k not in ("puntaje_datacredito", "promedio_ingresos_datacredito",
                               "tendencia_ingresos", "saldo_mora", "saldo_total", "saldo_principal")}

    respuesta = client.post("/predict/batch", json={"records": [sin_bureau]})

    assert respuesta.status_code == 200


@pytest.mark.parametrize(
    "cambio",
    [
        {"edad_cliente": 7},                      # below legal majority
        {"tipo_laboral": "Contratista"},          # not in the vocabulary
        {"plazo_meses": 0},                       # a loan with no term
        {"capital_prestado": -1},                 # negative money
        {"campo_inventado": 1},                   # extra="forbid"
    ],
)
def test_invalid_payloads_are_refused_at_the_edge(client, solicitud, cambio):
    respuesta = client.post("/predict/batch", json={"records": [{**solicitud, **cambio}]})

    assert respuesta.status_code == 422


def test_an_empty_batch_is_refused(client):
    assert client.post("/predict/batch", json={"records": []}).status_code == 422


def test_an_oversized_batch_is_refused(client, solicitud):
    from src.models.training_spec import default_serving_spec

    limite = default_serving_spec().max_batch_size
    registros = [{**solicitud, "application_id": f"APP-{i}"} for i in range(limite + 1)]

    assert client.post("/predict/batch", json={"records": registros}).status_code == 413
