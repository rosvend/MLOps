import pandera.errors
import pytest

from src.data.schema import CreditoFeaturesSchema, CreditoLabelledSchema
from src.features.cleaning import clean
from src.features.derive import add_derived_features
from src.features.spec import default_spec


@pytest.fixture
def prepared(raw):
    return add_derived_features(clean(raw))


def test_the_prepared_sample_satisfies_the_contract(prepared):
    assert len(CreditoLabelledSchema.validate(prepared)) == len(prepared)


def test_an_impossible_age_is_rejected(prepared):
    prepared.loc[0, "edad_cliente"] = 150

    with pytest.raises(pandera.errors.SchemaError):
        CreditoLabelledSchema.validate(prepared)


def test_principal_above_total_balance_is_rejected(prepared):
    prepared.loc[0, "saldo_principal"] = prepared.loc[0, "saldo_total"] + 1

    with pytest.raises(pandera.errors.SchemaError):
        CreditoLabelledSchema.validate(prepared)


def test_an_unexpected_column_is_rejected(prepared):
    prepared["columna_inesperada"] = 1

    with pytest.raises(pandera.errors.SchemaErrors):
        CreditoLabelledSchema.validate(prepared)


def test_the_feature_contract_does_not_require_the_outcome(prepared):
    sin_label = prepared.drop(columns=["Pago_atiempo"])

    assert len(CreditoFeaturesSchema.validate(sin_label)) == len(prepared)


def test_the_labelled_contract_does_require_the_outcome(prepared):
    with pytest.raises(pandera.errors.SchemaError):
        CreditoLabelledSchema.validate(prepared.drop(columns=["Pago_atiempo"]))


def test_the_declared_indicators_match_the_watched_columns():
    """pandera declares columns as class attributes, so the falta_ fields cannot be
    generated from the config; this is what stops the two lists drifting apart."""
    declarados = {
        c for c in CreditoFeaturesSchema.to_schema().columns if c.startswith("falta_")
    }

    assert declarados == {f"falta_{c}" for c in default_spec().columns.vigiladas}
