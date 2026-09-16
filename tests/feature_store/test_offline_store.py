"""The feature store must retrieve correctly, and must refuse to answer about the future.

These tests read the registry built by `make feast-apply`, so they need it to exist.
"""

from datetime import timedelta
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from src.features.spec import default_spec

RAIZ = Path(__file__).resolve().parents[2]
REPO = RAIZ / "feature_repo"
REGISTRY = REPO / "data" / "registry.db"

pytestmark = pytest.mark.skipif(
    not REGISTRY.exists(), reason="run `make feast-apply` first to build the registry"
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
    entidad = store.get_entity("cliente")

    assert entidad.join_key == spec.entity_key


def test_no_online_store_is_configured(store):
    """The constraint for this stage: offline retrieval only."""
    assert store.config.online_store is None


def test_registered_types_match_the_parquet(store, spec):
    """A dtype change upstream must fail here, not reshape the store silently."""
    import sys

    sys.path.insert(0, str(REPO))
    from features import feast_type

    arrow = {f.name: str(f.type) for f in pq.read_schema(RAIZ / spec.output_path)}

    for vista in store.list_feature_views():
        for campo in vista.schema:
            if campo.name in (spec.entity_key, spec.event_timestamp):
                continue
            assert campo.dtype == feast_type(arrow[campo.name]), campo.name


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
