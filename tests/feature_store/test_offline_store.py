"""The feature store must retrieve correctly, and must refuse to answer about the future.

These tests read the registry built by `make feast-apply`, so they need it to exist.
"""

from datetime import timedelta
from pathlib import Path

import pandas as pd
import pytest

from src.features.feast_types import parquet_types
from src.features.spec import default_spec
from src.pipelines.feature_store_check import mismatched_values, stale_fields

RAIZ = Path(__file__).resolve().parents[2]
REPO = RAIZ / "feature_repo"
REGISTRY = REPO / "data" / "registry.db"
PARQUET = RAIZ / default_spec().output_path

pytestmark = pytest.mark.skipif(
    not (REGISTRY.exists() and PARQUET.exists()),
    reason="run `make feast-apply` first to build the feature table and the registry",
)


@pytest.fixture(scope="module")
def spec():
    return default_spec()


@pytest.fixture(scope="module")
def store():
    from feast import FeatureStore

    return FeatureStore(repo_path=str(REPO))


@pytest.fixture(scope="module")
def fuente(spec):
    return pd.read_parquet(RAIZ / spec.output_path)


@pytest.fixture(scope="module")
def refs(spec):
    return [f"{vista}:{c}" for vista, cols in spec.feature_views.items() for c in cols]


def _entity_df(fuente, spec, n=25, desplazamiento=timedelta(0)):
    muestra = fuente.head(n)
    return pd.DataFrame(
        {
            spec.entity_key: muestra[spec.entity_key].to_numpy(),
            "event_timestamp": pd.to_datetime(muestra[spec.event_timestamp])
            + pd.Timedelta(desplazamiento),  # already tz-aware from the source
        }
    ).reset_index(drop=True)


# --- the registry matches the pipeline ---------------------------------------


def test_every_feature_view_is_registered(store, spec):
    registradas = {v.name for v in store.list_feature_views()}

    assert registradas == set(spec.feature_views)


def test_the_entity_is_the_configured_join_key(store, spec):
    entidad = store.get_entity(spec.entity_name)

    assert entidad.join_key == spec.entity_key


def test_no_online_store_is_configured(store):
    """The constraint for this stage: offline retrieval only."""
    assert store.config.online_store is None


def test_registered_types_match_the_parquet(store, spec):
    """A dtype change upstream must fail here, not reshape the store silently."""
    esperados = parquet_types(RAIZ / spec.output_path)

    for vista in store.list_feature_views():
        for campo in vista.schema:
            assert campo.dtype == esperados[campo.name], campo.name


def test_the_views_expose_every_feature_and_nothing_reserved(store, spec):
    expuestas = {
        c.name
        for v in store.list_feature_views()
        for c in v.schema
        if c.name not in (spec.entity_key, spec.event_timestamp)
    }

    assert expuestas == set(spec.feature_view_columns)
    assert not expuestas & spec.no_son_features


# --- retrieval ----------------------------------------------------------------


def test_historical_retrieval_returns_one_row_per_entity(store, fuente, spec, refs):
    entity_df = _entity_df(fuente, spec)

    salida = store.get_historical_features(entity_df=entity_df, features=refs).to_df()

    assert len(salida) == len(entity_df)
    assert set(salida[spec.entity_key]) == set(entity_df[spec.entity_key])


def test_retrieved_values_equal_the_source(store, fuente, spec, refs):
    entity_df = _entity_df(fuente, spec)

    salida = store.get_historical_features(entity_df=entity_df, features=refs).to_df()

    unidos = salida.merge(fuente, on=spec.entity_key, suffixes=("_feast", "_src"))
    for columna in ("capital_prestado", "plazo_meses", "puntaje_datacredito", "dti"):
        izq = unidos[f"{columna}_feast"].astype("Float64")
        der = unidos[f"{columna}_src"].astype("Float64")
        assert izq.equals(der), columna


# --- point in time: the test that matters -------------------------------------


def test_asking_before_origination_returns_no_feature_values(store, fuente, spec, refs):
    """A store that answers here is leaking the future into the training set.

    Feast may express "nothing known yet" either as no rows or as null features; the
    property that matters is that no value crosses back over its own event timestamp.
    """
    entity_df = _entity_df(fuente, spec, desplazamiento=-timedelta(days=1))

    salida = store.get_historical_features(entity_df=entity_df, features=refs).to_df()

    columnas = [c for c in salida.columns if c not in (spec.entity_key, "event_timestamp")]
    assert salida.empty or salida[columnas].isna().all().all()


def test_asking_at_origination_returns_the_features(store, fuente, spec, refs):
    entity_df = _entity_df(fuente, spec)

    salida = store.get_historical_features(entity_df=entity_df, features=refs).to_df()

    assert salida["capital_prestado"].notna().all()


def test_a_later_timestamp_still_sees_the_application(store, fuente, spec, refs):
    """Point-in-time joins look backwards, so a later question still finds the row."""
    entity_df = _entity_df(fuente, spec, desplazamiento=timedelta(days=30))

    salida = store.get_historical_features(entity_df=entity_df, features=refs).to_df()

    assert salida["capital_prestado"].notna().all()


def test_the_target_is_not_retrievable(store, spec):
    """The label never entered the store, so it cannot come back out of it."""
    expuestas = {c.name for v in store.list_feature_views() for c in v.schema}

    assert spec.target not in expuestas


def test_nothing_crosses_back_over_its_own_event_timestamp(store, fuente, spec, refs):
    """The leakage check on the whole book, not a sample: every answer is old enough."""
    entity_df = _entity_df(fuente, spec, n=len(fuente))

    salida = store.get_historical_features(entity_df=entity_df, features=refs).to_df()

    unidos = salida.merge(fuente[[spec.entity_key, spec.event_timestamp]], on=spec.entity_key)
    assert (unidos[spec.event_timestamp] <= unidos["event_timestamp"]).all()


# --- the retrieved frame is keyed, not ordered ---------------------------------


def _revuelto(fuente, spec):
    """An entity_df deliberately out of timestamp order."""
    muestra = fuente.iloc[[500, 10, 900, 3, 700]]
    return pd.DataFrame(
        {
            spec.entity_key: muestra[spec.entity_key].to_numpy(),
            "event_timestamp": pd.to_datetime(muestra[spec.event_timestamp]).to_numpy(),
        }
    )


def test_retrieval_is_not_positionally_aligned_with_the_entity_df(store, fuente, spec, refs):
    """Feast sorts by event timestamp and resets the index, so the reorder is invisible.

    A consumer that pairs the result with a label by position gets the wrong label for
    almost every row. The only safe pairing is a join on the entity key.
    """
    entity_df = _revuelto(fuente, spec)

    salida = store.get_historical_features(entity_df=entity_df, features=refs).to_df()

    assert set(salida[spec.entity_key]) == set(entity_df[spec.entity_key])
    assert list(salida[spec.entity_key]) != list(entity_df[spec.entity_key])


def test_joining_on_the_entity_key_recovers_the_right_values(store, fuente, spec, refs):
    """The complement: keyed, every column is exactly the source's for that loan."""
    entity_df = _revuelto(fuente, spec)

    salida = store.get_historical_features(entity_df=entity_df, features=refs).to_df()

    assert mismatched_values(salida, fuente, spec, spec.feature_view_columns) == []


# --- staleness ----------------------------------------------------------------


def test_the_live_registry_describes_the_parquet(store):
    assert stale_fields(store, PARQUET) == []


def test_a_registry_stale_against_the_parquet_is_detected(store, fuente, tmp_path):
    """`make features` alone leaves the registry behind; Feast would serve the drift."""
    desviado = fuente.copy()
    desviado["capital_prestado"] = desviado["capital_prestado"].astype("float64")
    ruta = tmp_path / "features.parquet"
    desviado.to_parquet(ruta, index=False)

    assert stale_fields(store, ruta) == ["capital_prestado"]


def test_a_column_dropped_from_the_parquet_is_detected(store, fuente, tmp_path):
    recortado = fuente.drop(columns=["dti"])
    ruta = tmp_path / "features.parquet"
    recortado.to_parquet(ruta, index=False)

    assert stale_fields(store, ruta) == ["dti"]


# --- edge cases ---------------------------------------------------------------


def test_an_empty_entity_df_returns_no_rows_but_every_column(store, spec, refs):
    entity_df = pd.DataFrame(
        {
            spec.entity_key: pd.Series([], dtype="object"),
            "event_timestamp": pd.Series([], dtype="datetime64[ns, UTC]"),
        }
    )

    salida = store.get_historical_features(entity_df=entity_df, features=refs).to_df()

    assert len(salida) == 0
    assert set(spec.feature_view_columns) <= set(salida.columns)


def test_an_unknown_entity_comes_back_as_a_row_of_nulls(store, spec, refs):
    """A loan the store never saw must not borrow another loan's features."""
    entity_df = pd.DataFrame(
        {
            spec.entity_key: ["CLI-9999999"],
            "event_timestamp": [pd.Timestamp("2025-06-01", tz="UTC")],
        }
    )

    salida = store.get_historical_features(entity_df=entity_df, features=refs).to_df()

    assert len(salida) == 1
    assert salida[spec.feature_view_columns].isna().all().all()


def test_a_single_record_retrieves_its_own_values(store, fuente, spec, refs):
    entity_df = _entity_df(fuente, spec, n=1)

    salida = store.get_historical_features(entity_df=entity_df, features=refs).to_df()

    assert len(salida) == 1
    assert salida["capital_prestado"].iloc[0] == fuente["capital_prestado"].iloc[0]


def test_repeated_entity_rows_collapse(store, fuente, spec, refs):
    """Feast deduplicates identical (entity, timestamp) pairs, so the count shrinks."""
    una = _entity_df(fuente, spec, n=3)
    entity_df = pd.concat([una, una], ignore_index=True)

    salida = store.get_historical_features(entity_df=entity_df, features=refs).to_df()

    assert len(entity_df) == 6
    assert len(salida) == 3


def test_a_timezone_naive_entity_df_is_read_as_utc(store, fuente, spec, refs):
    """The source is localised, not converted, so naive and aware must agree."""
    aware = _entity_df(fuente, spec, n=20)
    naive = aware.assign(event_timestamp=aware["event_timestamp"].dt.tz_localize(None))

    con_zona = store.get_historical_features(entity_df=aware, features=refs).to_df()
    sin_zona = store.get_historical_features(entity_df=naive, features=refs).to_df()

    assert mismatched_values(sin_zona, con_zona, spec, ["capital_prestado"]) == []


def test_a_timestamp_before_the_whole_book_returns_nothing(store, fuente, spec, refs):
    entity_df = pd.DataFrame(
        {
            spec.entity_key: fuente[spec.entity_key].head(5).to_numpy(),
            "event_timestamp": [pd.Timestamp("1990-01-01", tz="UTC")] * 5,
        }
    )

    salida = store.get_historical_features(entity_df=entity_df, features=refs).to_df()

    assert len(salida) == 0
