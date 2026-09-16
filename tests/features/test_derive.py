import pandas as pd
import pytest

from src.features.cleaning import clean
from src.features.derive import add_derived_features


@pytest.fixture
def derived(raw):
    return add_derived_features(clean(raw))


def test_debt_to_income_divides_other_loans_by_salary(derived):
    assert derived["dti"].iloc[0] == pytest.approx(1_000_000 / 3_500_000)


def test_payment_to_income_divides_instalment_by_salary(derived):
    assert derived["pti"].iloc[0] == pytest.approx(128_650 / 3_500_000)


def test_loan_to_income_divides_capital_by_salary(derived):
    assert derived["monto_sobre_ingreso"].iloc[0] == pytest.approx(1_852_560 / 3_500_000)


def test_income_gap_contrasts_declared_against_bureau(derived):
    assert derived["ratio_ingreso_declarado_bureau"].iloc[0] == pytest.approx(3_500_000 / 916_148)


def test_ratios_are_null_when_the_salary_was_a_sentinel(derived):
    assert derived["dti"].isna().sum() == 2


def test_credit_intensity_is_null_for_eighteen_year_olds(derived):
    assert pd.isna(derived["creditos_por_anio_adulto"].iloc[13])


def test_credit_intensity_spreads_credits_over_adult_years(derived):
    assert derived["creditos_por_anio_adulto"].iloc[0] == pytest.approx(2 / (32 - 18))


def test_bureau_arrears_flag_is_true_only_with_a_positive_balance(derived):
    assert derived["tiene_mora_bureau"].sum() == 1
    assert bool(derived["tiene_mora_bureau"].iloc[12])


def test_age_bands_follow_the_eda_cuts(derived):
    assert derived["rango_edad"].iloc[0] == "26-35"
    assert derived["rango_edad"].iloc[13] == "18-25"
    assert pd.isna(derived["rango_edad"].iloc[1])


def test_vintage_is_the_disbursement_month(derived):
    assert derived["mes_prestamo"].iloc[0] == "2025-01"


def test_deriving_never_drops_rows(raw, derived):
    assert len(derived) == len(raw)
