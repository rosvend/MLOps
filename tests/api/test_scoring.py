"""The serving path must agree with the training path, and must not depend on the batch."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.training_spec import default_serving_spec

RAIZ = Path(__file__).resolve().parents[2]
ARTEFACTO = RAIZ / "models" / "champion.joblib"

pytestmark = pytest.mark.skipif(
    not ARTEFACTO.exists(), reason="run `make export-champion` first"
)


@pytest.fixture(scope="module")
def champion():
    from src.api.service import load_champion

    serving = default_serving_spec()
    return load_champion(serving.model_path, serving.meta_path)


@pytest.fixture(scope="module")
def crudo():
    from src.data.csv_source import CsvDataSource

    return CsvDataSource("data/raw/BD_creditos.csv").read()


def _payload(fila: pd.Series, idx: int) -> dict:
    """The fields an originating system would actually send."""
    campos = [
        "tipo_credito", "capital_prestado", "plazo_meses", "edad_cliente", "tipo_laboral",
        "salario_cliente", "total_otros_prestamos", "cuota_pactada", "puntaje_datacredito",
        "cant_creditosvigentes", "huella_consulta", "saldo_mora", "saldo_total",
        "saldo_principal", "creditos_sectorFinanciero", "creditos_sectorCooperativo",
        "creditos_sectorReal", "promedio_ingresos_datacredito", "tendencia_ingresos",
    ]
    registro = {"application_id": f"APP-{idx:05d}",
                "fecha_prestamo": pd.to_datetime(fila["fecha_prestamo"], dayfirst=True)}
    for c in campos:
        valor = fila[c]
        registro[c] = None if pd.isna(valor) else (str(valor) if c == "tipo_credito" else valor)
    return registro


# --- parity with training ------------------------------------------------------


def test_serving_reproduces_the_training_path(champion, crudo):
    """If the serving feature computation had drifted from training, this is where it shows."""
    from src.api.service import score
    from src.features.contract import features
    from src.pipelines.prepare import prepare_labelled

    class Mem:
        def __init__(self, f):
            self._f = f

        def read(self):
            return self._f.copy()

    muestra = crudo.head(50)
    esperado = champion.pipeline.predict_proba(
        features(prepare_labelled(Mem(muestra)), champion.spec)[champion.meta["features"]]
    )[:, 1]

    obtenido = [p["probability_default"] for p in score(
        [_payload(f, i) for i, (_, f) in enumerate(muestra.iterrows())], champion
    )]

    assert np.allclose(obtenido, esperado, atol=1e-9)


# --- the frozen threshold ------------------------------------------------------


def test_a_record_scores_the_same_alone_as_in_a_batch(champion, crudo):
    """No batch-dependence: the property this pipeline has defended at every stage."""
    from src.api.service import score

    registros = [_payload(f, i) for i, (_, f) in enumerate(crudo.head(200).iterrows())]

    sola = score(registros[:1], champion)[0]
    en_lote = score(registros, champion)[0]

    assert sola["probability_default"] == pytest.approx(en_lote["probability_default"])
    assert sola["review_flag"] == en_lote["review_flag"]


def test_the_flag_follows_the_frozen_threshold(champion, crudo):
    from src.api.service import score

    predicciones = score(
        [_payload(f, i) for i, (_, f) in enumerate(crudo.head(100).iterrows())], champion
    )

    for p in predicciones:
        assert p["review_flag"] == (p["probability_default"] >= champion.threshold)
        assert p["threshold"] == champion.threshold


def test_the_threshold_comes_from_training_not_from_the_request(champion, crudo):
    """Scoring only low-risk applications must not lower the bar to keep the share at 15.7%."""
    from src.api.service import score

    predicciones = score(
        [_payload(f, i) for i, (_, f) in enumerate(crudo.head(300).iterrows())], champion
    )
    marcados = sum(p["review_flag"] for p in predicciones)

    assert all(p["threshold"] == champion.threshold for p in predicciones)
    assert marcados != pytest.approx(len(predicciones) * 0.157, abs=1)


def test_every_application_gets_its_own_id_back(champion, crudo):
    from src.api.service import score

    registros = [_payload(f, i) for i, (_, f) in enumerate(crudo.head(10).iterrows())]

    predicciones = score(registros, champion)

    assert [p["application_id"] for p in predicciones] == [r["application_id"] for r in registros]


def test_probabilities_are_probabilities(champion, crudo):
    from src.api.service import score

    predicciones = score(
        [_payload(f, i) for i, (_, f) in enumerate(crudo.head(100).iterrows())], champion
    )

    assert all(0.0 <= p["probability_default"] <= 1.0 for p in predicciones)
