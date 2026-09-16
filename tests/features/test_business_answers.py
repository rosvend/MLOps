"""The three data-meaning decisions taken with the business, made executable.

Answers recorded 2026-09-16: the bureau pull is origination-time, bureau balances
are in thousands of COP, and a derived flag must distinguish "no" from "unknown".
"""

import pandas as pd
import pytest

from src.features.cleaning import MILES, SALDOS_EN_MILES, clean
from src.features.contract import PROHIBIDAS
from src.features.derive import COLUMNAS_VIGILADAS, add_derived_features
from src.pipelines.prepare import prepare_labelled


@pytest.fixture
def prepared(sample_source):
    return prepare_labelled(sample_source)


# --- bureau pull is origination-time -----------------------------------------


@pytest.mark.parametrize("columna", ["saldo_mora", "tiene_mora_bureau"])
def test_the_bureau_arrears_columns_are_no_longer_withheld(columna):
    """Confirmed observed at origination, so they are features, not leakage."""
    assert columna not in PROHIBIDAS


@pytest.mark.parametrize("columna", ["puntaje", "mes_prestamo"])
def test_the_genuinely_leaky_columns_are_still_withheld(columna):
    assert columna in PROHIBIDAS


# --- bureau balances are thousands of COP ------------------------------------


@pytest.mark.parametrize("columna", SALDOS_EN_MILES)
def test_bureau_balances_are_stored_in_pesos(raw, columna):
    cleaned = clean(raw)

    esperado = raw[columna].dropna() * MILES
    assert cleaned.loc[esperado.index, columna].tolist() == esperado.astype("Int64").tolist()


def test_scaling_leaves_the_arrears_flag_alone(prepared):
    """A threshold at zero is unit-invariant, so the heuristic's scores cannot move."""
    assert prepared["tiene_mora_bureau"].sum() == (prepared["saldo_mora"] > 0).sum()


def test_balances_are_now_plausible_against_salary(prepared):
    ratio = (prepared["saldo_total"] / prepared["salario_cliente"]).median()

    assert ratio > 1.0


# --- unknown is not "no" -----------------------------------------------------


@pytest.mark.parametrize("bandera", ["cuota_supera_salario", "tiene_mora_bureau"])
def test_a_flag_is_unknown_when_its_input_is_unknown(prepared, bandera):
    assert prepared[bandera].dtype == "boolean"


def test_an_unverifiable_salary_does_not_read_as_an_affordable_instalment(raw):
    sin_salario = raw.copy()
    sin_salario.loc[0, "salario_cliente"] = 0  # sentinel: salary not verified

    flag = clean(sin_salario)["cuota_supera_salario"]

    assert pd.isna(flag.iloc[0])


def test_missing_arrears_do_not_read_as_confirmed_no_arrears(raw):
    sin_saldo = raw.copy()
    sin_saldo["saldo_mora"] = pd.NA

    derivado = add_derived_features(clean(sin_saldo))

    assert derivado["tiene_mora_bureau"].isna().all()


@pytest.mark.parametrize("columna", COLUMNAS_VIGILADAS)
def test_every_watched_column_gets_a_missingness_indicator(prepared, columna):
    indicador = f"falta_{columna}"

    assert indicador in prepared.columns
    assert prepared[indicador].dtype == bool
    assert prepared[indicador].tolist() == prepared[columna].isna().tolist()


def test_missingness_indicators_are_offered_to_the_model(prepared):
    from src.features.contract import features

    vista = features(prepared)

    assert any(c.startswith("falta_") for c in vista.columns)
