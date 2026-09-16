"""The split must be honest about time, and the label must never be a feature."""

import pytest

from src.features.spec import default_spec
from src.models.dataset import Dataset, split_out_of_time
from src.models.training_spec import Split, default_training_spec


@pytest.fixture(scope="module")
def datos():
    from src.data.csv_source import CsvDataSource
    from src.models.dataset import load_from_feast

    return load_from_feast(CsvDataSource("data/raw/BD_creditos.csv"), default_spec())


def test_features_come_back_from_the_store(datos, spec_fixture=None):
    spec = default_spec()

    assert isinstance(datos, Dataset)
    assert list(datos.X.columns) == spec.feature_view_columns
    assert len(datos.X) == len(datos.y) == len(datos.vintage)


def test_the_label_is_joined_from_outside_the_store(datos):
    assert datos.y.name == "defaulted"
    assert 0.03 < datos.y.mean() < 0.07


def test_no_reserved_column_reaches_the_features(datos):
    assert not set(datos.X.columns) & default_spec().no_son_features


def test_every_test_loan_is_newer_than_every_training_loan(datos):
    partido = split_out_of_time(datos, default_training_spec().split)

    assert partido.train.vintage.max() < partido.test.vintage.min()


def test_both_sides_are_populated(datos):
    partido = split_out_of_time(datos, default_training_spec().split)

    assert len(partido.train) > len(partido.test) > 0
    assert len(partido.train) + len(partido.test) == len(datos)


def test_the_split_honours_the_configured_fraction(datos):
    partido = split_out_of_time(datos, Split(by="mes_prestamo", train_fraction=0.5))

    assert 0.4 < len(partido.train) / len(datos) < 0.7


def test_a_nonsensical_fraction_is_rejected():
    with pytest.raises(ValueError):
        Split(by="mes_prestamo", train_fraction=0.99)
