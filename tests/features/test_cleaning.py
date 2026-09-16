import pandas as pd
import pytest

from src.features.cleaning import (
    clean,
    coerce_types,
    drop_unusable_columns,
    flag_inconsistencies,
    group_rare_credit_types,
    nullify_sentinels,
)


@pytest.fixture
def nullified(raw):
    return nullify_sentinels(raw)


def test_ages_above_the_human_range_become_null(nullified):
    assert nullified["edad_cliente"].isna().sum() == 1
    assert nullified["edad_cliente"].max() == 62


def test_bureau_scores_outside_150_950_become_null(nullified):
    assert nullified["puntaje_datacredito"].isna().sum() == 3


def test_impossible_salaries_become_null(nullified):
    assert nullified["salario_cliente"].isna().sum() == 2


def test_non_positive_bureau_income_becomes_null(nullified):
    assert nullified["promedio_ingresos_datacredito"].isna().sum() == 2


def test_income_trend_keeps_only_its_three_labels(nullified):
    assert set(nullified["tendencia_ingresos"].dropna()) == {"Creciente", "Estable", "Decreciente"}


def test_nullifying_never_drops_rows(raw, nullified):
    assert len(nullified) == len(raw)


def test_constant_codebtor_column_is_dropped(raw):
    assert "saldo_mora_codeudor" not in drop_unusable_columns(raw).columns


def test_types_match_the_meaning_of_each_column(nullified):
    typed = coerce_types(nullified)

    assert typed["fecha_prestamo"].dtype.kind == "M"
    assert typed["puntaje"].iloc[0] == pytest.approx(95.227787)
    assert typed["Pago_atiempo"].dtype == bool
    assert typed["edad_cliente"].dtype == "Int64"
    assert typed["tendencia_ingresos"].cat.ordered


def test_dates_are_read_day_first(nullified):
    assert coerce_types(nullified)["fecha_prestamo"].iloc[0] == pd.Timestamp("2025-01-07 14:40")


def test_residual_credit_types_are_grouped_into_otro(nullified):
    grouped = group_rare_credit_types(coerce_types(nullified))

    assert set(grouped["tipo_credito"]) == {"4", "9", "10", "Otro"}
    assert (grouped["tipo_credito"] == "Otro").sum() == 2


def test_instalment_above_salary_is_flagged_not_removed(nullified):
    flagged = flag_inconsistencies(coerce_types(nullified))

    assert flagged["cuota_supera_salario"].sum() == 1
    assert len(flagged) == len(nullified)


def test_clean_composes_every_step(raw):
    cleaned = clean(raw)

    assert len(cleaned) == len(raw)
    assert "saldo_mora_codeudor" not in cleaned.columns
    assert "cuota_supera_salario" in cleaned.columns
    assert cleaned["puntaje"].dtype == "Float64"


def _con_columna(raw, nombre, valores):
    df = raw.head(len(valores)).copy()
    df[nombre] = valores
    return df


@pytest.mark.parametrize("crudo", [["False", "True", "0"], [0, 1, 0], ["0", "1", "0"]])
def test_the_target_survives_any_boolean_representation(raw, crudo):
    typed = coerce_types(nullify_sentinels(_con_columna(raw, "Pago_atiempo", crudo)))

    assert typed["Pago_atiempo"].tolist() == [False, True, False]


@pytest.mark.parametrize("basura", [[None, 1, 0], ["si", "no", "1"], [2, 1, 0]])
def test_an_unreadable_target_raises_instead_of_defaulting_to_paid(raw, basura):
    with pytest.raises(ValueError, match="Pago_atiempo"):
        coerce_types(nullify_sentinels(_con_columna(raw, "Pago_atiempo", basura)))


@pytest.mark.parametrize("crudo", [["95,2", "90,5", "88,1"], [95.2, 90.5, 88.1]])
def test_the_score_parses_whether_the_source_sends_text_or_numbers(raw, crudo):
    typed = coerce_types(nullify_sentinels(_con_columna(raw, "puntaje", crudo)))

    assert typed["puntaje"].tolist() == pytest.approx([95.2, 90.5, 88.1])


def test_categories_do_not_depend_on_which_rows_are_in_the_batch(raw):
    completo = clean(raw)
    parcial = clean(raw.head(2))

    for columna in ("tipo_credito", "tipo_laboral"):
        assert list(parcial[columna].cat.categories) == list(completo[columna].cat.categories)


def test_ages_below_adulthood_are_nulled_like_any_other_impossible_value(raw):
    nullified = nullify_sentinels(_con_columna(raw, "edad_cliente", [17, 30, 122]))

    assert nullified["edad_cliente"].isna().tolist() == [True, False, True]


@pytest.mark.parametrize("dtype", ["int64", "str", "float64"])
def test_product_codes_group_the_same_whatever_type_the_source_sends(raw, dtype):
    """4, "4" and 4.0 are the same product code; a text source must not become all Otro."""
    df = raw.assign(tipo_credito=raw["tipo_credito"].astype(dtype))

    grouped = clean(df)["tipo_credito"].value_counts().to_dict()

    assert grouped == clean(raw)["tipo_credito"].value_counts().to_dict()
    assert grouped["Otro"] < len(raw)


def test_a_scoring_payload_need_not_carry_the_column_we_only_delete(raw):
    sin_codeudor = raw.drop(columns=["saldo_mora_codeudor"])

    assert "saldo_mora_codeudor" not in drop_unusable_columns(sin_codeudor).columns
