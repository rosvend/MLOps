"""Logging is a side effect of scoring: when it fails, the customer still gets the score."""

from pathlib import Path

import pytest

from src.models.training_spec import default_serving_spec

RAIZ = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(
    not (RAIZ / "models" / "champion.joblib").exists(), reason="run `make fetch-champion` first"
)

SAMPLE = {
    "application_id": "APP-1", "tipo_credito": "9", "capital_prestado": 4200000, "plazo_meses": 24,
    "edad_cliente": 23, "tipo_laboral": "Independiente", "salario_cliente": 1200000,
    "total_otros_prestamos": 3500000, "cuota_pactada": 390000, "puntaje_datacredito": 680,
    "cant_creditosvigentes": 7, "huella_consulta": 11, "tendencia_ingresos": "Decreciente",
}


class BrokenLog:
    def append(self, records):
        raise RuntimeError("BigQuery rechazó filas")

    def read_all(self):
        raise NotImplementedError


class RecordingLog:
    def __init__(self):
        self.records = []

    def append(self, records):
        self.records.extend(records)

    def read_all(self):
        raise NotImplementedError


@pytest.fixture(scope="module")
def champion():
    from src.api.service import load_champion

    serving = default_serving_spec()
    return load_champion(serving.model_path, serving.meta_path)


def test_a_failing_log_does_not_fail_the_prediction(champion, caplog):
    from src.api.schemas import ApplicationRecord
    from src.api.service import score

    resultado = score([ApplicationRecord(**SAMPLE).model_dump()], champion, predictions_log=BrokenLog())

    assert len(resultado) == 1 and 0 <= resultado[0]["probability_default"] <= 1
    assert any("registr" in m.lower() or "log" in m.lower() for m in caplog.messages)


def test_an_application_with_missing_optional_fields_logs_cleanly(champion):
    from src.api.service import score

    from src.api.schemas import ApplicationRecord

    incompleto = {k: v for k, v in SAMPLE.items() if k not in ("salario_cliente", "tendencia_ingresos")}
    registro = ApplicationRecord(**incompleto).model_dump()  # what the API passes to score()
    log = RecordingLog()

    resultado = score([registro], champion, predictions_log=log)

    assert len(resultado) == 1 and len(log.records) == 1
