"""End to end: raw CSV in, a Feast-ready feature table out."""

import pandas as pd
import pytest

from src.features.spec import FeatureSpec, default_spec
from src.pipelines.features import build_feature_table, materialise


@pytest.fixture
def spec():
    return default_spec()


@pytest.fixture
def tabla(sample_source, spec):
    return build_feature_table(sample_source, spec)


def test_one_row_per_loan(tabla, raw):
    assert len(tabla) == len(raw)


def test_it_is_keyed_by_the_entity_and_the_event_timestamp(tabla, spec):
    assert list(tabla.columns[:2]) == [spec.entity_key, spec.event_timestamp]
    assert tabla[spec.entity_key].is_unique
    assert tabla[spec.entity_key].notna().all()
    assert pd.api.types.is_datetime64_any_dtype(tabla[spec.event_timestamp])


def test_the_target_never_reaches_the_feature_store(tabla, spec):
    assert spec.target not in tabla.columns


@pytest.mark.parametrize("columna", ["puntaje", "mes_prestamo"])
def test_withheld_columns_never_reach_the_feature_store(tabla, columna):
    assert columna not in tabla.columns


def test_it_carries_the_engineered_inputs_a_model_needs(tabla):
    for columna in ("dti", "pti", "tiene_mora_bureau", "rango_edad", "falta_salario_cliente"):
        assert columna in tabla.columns


def test_the_table_holds_only_row_independent_values(sample_source, spec, raw):
    """No fitted statistic is materialised, so a subset gives identical values."""

    class Mem:
        def __init__(self, frame):
            self._frame = frame

        def read(self):
            return self._frame.copy()

    completo = build_feature_table(sample_source, spec)
    parcial = build_feature_table(Mem(raw.head(5)), spec)

    compartidas = [c for c in parcial.columns if c != spec.entity_key]
    assert parcial[compartidas].reset_index(drop=True).equals(
        completo.head(5)[compartidas].reset_index(drop=True)
    )


def _variando(spec, **cambios):
    """A spec differing from the default in one group, with its validators re-run."""
    return FeatureSpec(**{**spec.model_dump(), **cambios})


def test_the_injected_spec_drives_cleaning_not_only_the_column_roles(sample_source, spec):
    """An override honoured by feature_names but ignored by clean() builds an incoherent table."""
    plano = _variando(spec, units={**spec.units.model_dump(), "factor": 1})

    escalado = build_feature_table(sample_source, spec)
    sin_escalar = build_feature_table(sample_source, plano)

    assert (sin_escalar["saldo_total"] * spec.units.factor).equals(escalado["saldo_total"])


def test_the_injected_spec_drives_sentinel_nulling(sample_source, spec):
    estrecho = _variando(spec, bounds={**spec.bounds.model_dump(), "edad": [40, 100]})

    tabla = build_feature_table(sample_source, estrecho)

    assert tabla["edad_cliente"].dropna().min() >= 40
    por_defecto = build_feature_table(sample_source, spec)
    assert tabla["falta_edad_cliente"].sum() > por_defecto["falta_edad_cliente"].sum()


def test_materialise_writes_a_readable_parquet(sample_source, spec, tmp_path):
    destino = tmp_path / "features.parquet"

    escrito = materialise(sample_source, spec, destino)

    recargado = pd.read_parquet(destino)
    assert len(recargado) == len(escrito)
    assert list(recargado.columns) == list(escrito.columns)
    assert recargado[spec.entity_key].is_unique


def test_no_column_is_dictionary_encoded(sample_source, spec, tmp_path):
    """Feast has no Categorical type and its file store reads through pyarrow."""
    import pyarrow.parquet as pq

    destino = tmp_path / "features.parquet"
    materialise(sample_source, spec, destino)

    esquema = pq.read_schema(destino)
    dictionarys = [f.name for f in esquema if "dictionary" in str(f.type)]
    assert dictionarys == []


def test_the_category_vocabulary_survives_as_strings(tabla, spec):
    """The vocabulary is enforced by the spec and the schema, not by parquet encoding."""
    valores = set(tabla["tipo_credito"].dropna())

    assert valores <= set(spec.vocabularies.tipos_credito)
    assert str(tabla["tipo_credito"].dtype) == "string"
