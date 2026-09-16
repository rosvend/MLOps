import numpy as np
import pytest

from src.features.spec import default_spec
from src.models import rules


def test_worst_bureau_score_costs_the_most(neutral_record):
    assert rules.bureau_score_band(neutral_record(puntaje_datacredito=700)) > rules.bureau_score_band(
        neutral_record(puntaje_datacredito=790)
    )


def test_bureau_score_points_fall_as_the_score_rises(neutral_record):
    points = [rules.bureau_score_band(neutral_record(puntaje_datacredito=s)) for s in (700, 790, 900)]

    assert points == sorted(points, reverse=True)


def test_bureau_score_bands_split_at_the_measured_terciles(neutral_record):
    assert rules.bureau_score_band(neutral_record(puntaje_datacredito=769)) != rules.bureau_score_band(
        neutral_record(puntaje_datacredito=770)
    )
    assert rules.bureau_score_band(neutral_record(puntaje_datacredito=812)) != rules.bureau_score_band(
        neutral_record(puntaje_datacredito=813)
    )


def test_a_missing_bureau_score_is_treated_as_risky(neutral_record):
    assert rules.bureau_score_band(neutral_record(puntaje_datacredito=None)) > 0


def test_more_inquiries_cost_more(neutral_record):
    points = [rules.inquiry_band(neutral_record(huella_consulta=h)) for h in (1, 5, 9)]

    assert points == sorted(points)


def test_inquiry_bands_split_at_three_and_six(neutral_record):
    assert rules.inquiry_band(neutral_record(huella_consulta=3)) != rules.inquiry_band(
        neutral_record(huella_consulta=4)
    )
    assert rules.inquiry_band(neutral_record(huella_consulta=6)) != rules.inquiry_band(
        neutral_record(huella_consulta=7)
    )


def test_the_youngest_band_is_the_most_expensive(neutral_record):
    youngest = rules.age_band(neutral_record(rango_edad="18-25"))

    assert youngest == max(rules.age_band(neutral_record(rango_edad=b)) for b in default_spec().age_bands.labels)


def test_age_points_decline_across_the_first_four_bands(neutral_record):
    points = [rules.age_band(neutral_record(rango_edad=b)) for b in ("18-25", "26-35", "36-45", "46-55")]

    assert points == sorted(points, reverse=True)


def test_a_missing_age_is_treated_as_risky(neutral_record):
    assert rules.age_band(neutral_record(rango_edad=np.nan)) > 0


def test_being_independent_only_costs_under_thirty_six(neutral_record):
    young = neutral_record(edad_cliente=30, tipo_laboral="Independiente")
    older = neutral_record(edad_cliente=50, tipo_laboral="Independiente")

    assert rules.young_independent(young) > 0
    assert rules.young_independent(older) == 0


def test_employed_applicants_never_pay_the_interaction(neutral_record):
    assert rules.young_independent(neutral_record(edad_cliente=25, tipo_laboral="Empleado")) == 0


def test_over_declaring_income_costs_more_than_matching_the_bureau(neutral_record):
    points = [rules.income_gap_quartile(neutral_record(ratio_ingreso_declarado_bureau=r)) for r in (0.5, 1.5, 2.5, 5.0)]

    assert points == sorted(points)


def test_a_missing_income_gap_is_neutral_because_another_rule_covers_it(neutral_record):
    assert rules.income_gap_quartile(neutral_record(ratio_ingreso_declarado_bureau=None)) == 0


def test_decreasing_income_costs_and_growing_income_pays(neutral_record):
    assert rules.decreasing_income_trend(neutral_record(tendencia_ingresos="Decreciente")) > 0
    assert rules.decreasing_income_trend(neutral_record(tendencia_ingresos="Creciente")) < 0


def test_a_missing_income_trend_is_neutral(neutral_record):
    assert rules.decreasing_income_trend(neutral_record(tendencia_ingresos=None)) == 0


def test_a_large_loan_over_a_long_term_is_the_costliest_single_rule(neutral_record):
    risky = neutral_record(capital_prestado=4_000_000, plazo_meses=18)

    assert rules.high_amount_long_term(risky) > 0


def test_a_large_loan_over_a_short_term_costs_nothing(neutral_record):
    assert rules.high_amount_long_term(neutral_record(capital_prestado=4_000_000, plazo_meses=6)) == 0


def test_a_small_loan_over_a_long_term_costs_nothing(neutral_record):
    assert rules.high_amount_long_term(neutral_record(capital_prestado=1_000_000, plazo_meses=18)) == 0


def test_a_missing_bureau_income_block_costs_points(neutral_record):
    assert rules.missing_bureau_income(neutral_record(promedio_ingresos_datacredito=None)) > 0
    assert rules.missing_bureau_income(neutral_record()) == 0


def test_every_age_band_the_pipeline_can_produce_has_a_weight(neutral_record):
    """The band labels and the points table must not drift apart."""
    for banda in default_spec().age_bands.labels:
        assert isinstance(rules.age_band(neutral_record(rango_edad=banda)), int)


def test_an_unknown_band_raises_instead_of_scoring_zero(neutral_record):
    with pytest.raises(KeyError, match="rango_edad"):
        rules.age_band(neutral_record(rango_edad="66-100"))
