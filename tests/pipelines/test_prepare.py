import pandas as pd

from src.pipelines.prepare import prepare

DERIVED = [
    "dti",
    "pti",
    "monto_sobre_ingreso",
    "ratio_ingreso_declarado_bureau",
    "creditos_por_anio_adulto",
    "tiene_mora_bureau",
    "rango_edad",
    "mes_prestamo",
    "cuota_supera_salario",
]


class InMemorySource:
    def __init__(self, frame: pd.DataFrame):
        self._frame = frame

    def read(self) -> pd.DataFrame:
        return self._frame.copy()


def test_prepare_keeps_every_record(sample_source):
    assert len(prepare(sample_source)) == 14


def test_prepare_adds_the_derived_features(sample_source):
    assert set(DERIVED) <= set(prepare(sample_source).columns)


def test_prepare_works_with_any_source_implementing_the_port(raw):
    from_csv = prepare(InMemorySource(raw))

    assert from_csv.shape == prepare(InMemorySource(raw)).shape


def test_prepare_output_is_validated(sample_source):
    prepared = prepare(sample_source)

    assert "saldo_mora_codeudor" not in prepared.columns
    assert prepared["Pago_atiempo"].dtype == bool
