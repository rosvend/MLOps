import pandera.errors
import pytest

from src.data.schema import CreditoSchema
from src.features.cleaning import clean
from src.features.derive import add_derived_features


@pytest.fixture
def prepared(raw):
    return add_derived_features(clean(raw))


def test_the_prepared_sample_satisfies_the_contract(prepared):
    assert len(CreditoSchema.validate(prepared)) == len(prepared)


def test_an_impossible_age_is_rejected(prepared):
    prepared.loc[0, "edad_cliente"] = 150

    with pytest.raises(pandera.errors.SchemaError):
        CreditoSchema.validate(prepared)


def test_principal_above_total_balance_is_rejected(prepared):
    prepared.loc[0, "saldo_principal"] = prepared.loc[0, "saldo_total"] + 1

    with pytest.raises(pandera.errors.SchemaError):
        CreditoSchema.validate(prepared)


def test_an_unexpected_column_is_rejected(prepared):
    prepared["columna_inesperada"] = 1

    with pytest.raises(pandera.errors.SchemaErrors):
        CreditoSchema.validate(prepared)
