import pytest

from src.features.contract import PROHIBIDAS, TARGET, features
from src.pipelines.prepare import prepare_labelled


@pytest.fixture
def prepared(sample_source):
    return prepare_labelled(sample_source)


def test_the_target_never_reaches_a_model(prepared):
    assert TARGET not in features(prepared).columns


@pytest.mark.parametrize("columna", sorted(PROHIBIDAS))
def test_every_prohibited_column_is_withheld(prepared, columna):
    assert columna in prepared.columns
    assert columna not in features(prepared).columns


def test_the_safe_view_keeps_every_row_and_the_rest_of_the_columns(prepared):
    safe = features(prepared)

    assert len(safe) == len(prepared)
    assert set(safe.columns) == set(prepared.columns) - PROHIBIDAS - {TARGET}


def test_features_is_a_copy_so_a_model_cannot_mutate_the_prepared_frame(prepared):
    features(prepared)["capital_prestado"] = 0

    assert (prepared["capital_prestado"] != 0).all()
