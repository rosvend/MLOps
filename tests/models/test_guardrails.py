"""Failure modes that must be loud: a credit scorecard should never degrade silently."""

import logging
import math

import pandas as pd
import pytest

from src.models.evaluate import DECILE_MINIMO, auc, decile_rates, evaluate
from src.models.heuristic import score, score_frame
from src.models.rules import REQUIRED_COLUMNS
from src.pipelines.prepare import prepare_labelled


@pytest.fixture
def prepared(sample_source):
    return prepare_labelled(sample_source)


@pytest.mark.parametrize("columna", sorted(REQUIRED_COLUMNS))
def test_a_missing_column_raises_instead_of_shifting_every_score(prepared, columna):
    with pytest.raises(KeyError, match=columna):
        score_frame(prepared.drop(columns=[columna]))


def test_a_missing_column_would_otherwise_have_inflated_risk(prepared):
    """Absence pays points, so dropping a column silently pushes loans over the threshold."""
    base = score_frame(prepared)
    sin_bureau = prepared.drop(columns=["puntaje_datacredito"]).assign(puntaje_datacredito=pd.NA)

    assert (score_frame(sin_bureau) > base).any()


def test_a_blank_application_is_flagged_for_review(prepared):
    """Maximum uncertainty scores above the operating threshold of 4, by design."""
    assert score({}) == 5


def test_one_loan_reports_a_rate_instead_of_raising():
    rates = decile_rates(pd.Series([3]), pd.Series([True]))

    assert rates.tolist() == [1.0]


@pytest.mark.parametrize("n", [1, 2, 5, 9, 11])
def test_small_portfolios_never_raise(n):
    scores = pd.Series(range(n))
    defaulted = pd.Series([i % 2 == 0 for i in range(n)])

    assert len(decile_rates(scores, defaulted)) <= min(10, n)


def test_decile_metrics_are_withheld_below_the_minimum_portfolio():
    scores = pd.Series(range(DECILE_MINIMO - 1))
    defaulted = pd.Series([i % 3 == 0 for i in range(DECILE_MINIMO - 1)])

    metrics = evaluate(scores, defaulted, threshold=0)

    assert math.isnan(metrics["decile_lift"])
    assert math.isnan(metrics["top_decile_rate"])


def test_a_clean_bottom_decile_reports_nan_not_infinity():
    scores = pd.Series(range(DECILE_MINIMO))
    defaulted = pd.Series([i >= DECILE_MINIMO // 2 for i in range(DECILE_MINIMO)])

    metrics = evaluate(scores, defaulted, threshold=0)

    assert not math.isinf(metrics["decile_lift"])
    assert math.isnan(metrics["decile_lift"])


def test_a_single_class_batch_says_why_the_metrics_are_missing(caplog):
    scores = pd.Series(range(10))
    defaulted = pd.Series([False] * 10)

    with caplog.at_level(logging.WARNING):
        resultado = auc(scores, defaulted)

    assert math.isnan(resultado)
    assert "una sola clase" in caplog.text
