"""The GCP training file must split by time, carry no identifier or leaked column, and
use the same cutoff as the local champion so both are scored on the same loans."""

import pandas as pd
import pytest

from src.features.spec import default_spec
from src.models.dataset import vintage_cutoff
from src.models.training_spec import GcpDatasetSpec, Split
from src.pipelines.export_gcp_dataset import assign_split, build_dataset
from src.pipelines.prepare import prepare_labelled

GCP = GcpDatasetSpec(
    split_column="split",
    label_column="defaulted",
    train_value="TRAIN",
    validation_value="VALIDATE",
    test_value="TEST",
    validation_fraction=0.2,
    output_path="data/gcp/loans_split.csv",
)
SPLIT = Split(by="mes_prestamo", train_fraction=0.6)


@pytest.fixture
def vintages() -> pd.Series:
    meses = [f"2025-{m:02d}" for m in range(1, 11)]
    return pd.Series([m for m in meses for _ in range(10)])


def test_the_cutoff_is_the_first_month_reaching_the_fraction(vintages):
    assert vintage_cutoff(vintages, 0.6) == "2025-06"


def test_time_orders_the_three_sets(vintages):
    etiquetas = assign_split(vintages, SPLIT, GCP)

    por_conjunto = vintages.groupby(etiquetas).agg(["min", "max"])
    assert por_conjunto.loc["TRAIN", "max"] < por_conjunto.loc["VALIDATE", "min"]
    assert por_conjunto.loc["VALIDATE", "max"] < por_conjunto.loc["TEST", "min"]


def test_the_test_set_matches_the_champions_held_out_window(vintages):
    etiquetas = assign_split(vintages, SPLIT, GCP)

    assert (vintages[etiquetas == "TEST"] >= vintage_cutoff(vintages, SPLIT.train_fraction)).all()
    assert (vintages[etiquetas != "TEST"] < vintage_cutoff(vintages, SPLIT.train_fraction)).all()


def test_all_three_sets_are_populated(vintages):
    assert set(assign_split(vintages, SPLIT, GCP)) == {"TRAIN", "VALIDATE", "TEST"}


@pytest.fixture
def dataset(sample_source) -> pd.DataFrame:
    etiquetado = prepare_labelled(sample_source)
    return build_dataset(etiquetado, default_spec(), Split(by="mes_prestamo", train_fraction=0.6), GCP)


def test_columns_are_the_features_plus_label_and_split(dataset):
    esperado = [*default_spec().feature_view_columns, "defaulted", "split"]
    assert list(dataset.columns) == esperado


def test_no_identifier_timestamp_or_leaky_column_is_exported(dataset):
    prohibidas = {"cliente_id", "fecha_prestamo", "mes_prestamo", "puntaje", "Pago_atiempo"}
    assert not prohibidas & set(dataset.columns)


def test_the_label_is_default_not_on_time_payment(sample_source, dataset):
    etiquetado = prepare_labelled(sample_source)
    assert dataset["defaulted"].tolist() == (~etiquetado["Pago_atiempo"]).astype(int).tolist()


def test_the_config_rejects_fractions_that_empty_a_set():
    with pytest.raises(ValueError):
        GcpDatasetSpec(**{**GCP.model_dump(), "validation_fraction": 0.0})
