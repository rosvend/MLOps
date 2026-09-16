import pandas as pd
import pytest

from src.models.evaluate import auc, evaluate, gini, precision_recall


@pytest.fixture
def defaulted():
    return pd.Series([False] * 50 + [True] * 50)


def test_a_perfect_ranking_scores_one(defaulted):
    scores = pd.Series(list(range(50)) + list(range(100, 150)))

    assert auc(scores, defaulted) == pytest.approx(1.0)


def test_an_inverted_ranking_scores_zero(defaulted):
    scores = pd.Series(list(range(100, 150)) + list(range(50)))

    assert auc(scores, defaulted) == pytest.approx(0.0)


def test_a_constant_score_is_a_coin_flip(defaulted):
    assert auc(pd.Series([1] * 100), defaulted) == pytest.approx(0.5)


def test_gini_rescales_auc_to_minus_one_and_one(defaulted):
    scores = pd.Series(list(range(50)) + list(range(100, 150)))

    assert gini(scores, defaulted) == pytest.approx(1.0)
    assert gini(pd.Series([1] * 100), defaulted) == pytest.approx(0.0)


def test_precision_and_recall_at_a_perfect_threshold(defaulted):
    scores = pd.Series(list(range(50)) + list(range(100, 150)))

    result = precision_recall(scores, defaulted, threshold=100)

    assert result["precision"] == pytest.approx(1.0)
    assert result["recall"] == pytest.approx(1.0)
    assert result["flagged_share"] == pytest.approx(0.5)


def test_flagging_nobody_reports_zero_precision(defaulted):
    result = precision_recall(pd.Series([1] * 100), defaulted, threshold=99)

    assert result["flagged_share"] == 0.0
    assert result["precision"] == 0.0


def test_evaluate_gathers_every_metric(defaulted):
    scores = pd.Series(list(range(50)) + list(range(100, 150)))

    metrics = evaluate(scores, defaulted, threshold=100)

    assert {"auc", "gini", "top_decile_rate", "bottom_decile_rate", "decile_lift"} <= set(metrics)
