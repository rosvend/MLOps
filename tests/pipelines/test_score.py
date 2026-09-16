import pytest

from src.models.scorecard import default_scorecard
from src.pipelines.score import score_portfolio


@pytest.fixture
def con_umbral():
    def build(threshold: int):
        return default_scorecard().model_copy(update={"threshold": threshold})

    return build


def test_scoring_the_portfolio_returns_a_score_per_loan(sample_source, con_umbral, raw):
    scores, _ = score_portfolio(sample_source, con_umbral(0))

    assert len(scores) == len(raw)


def test_scoring_the_portfolio_reports_metrics(sample_source, con_umbral):
    _, metrics = score_portfolio(sample_source, con_umbral(0))

    assert {"auc", "gini", "precision", "recall", "flagged_share"} <= set(metrics)


def test_the_threshold_drives_how_many_loans_are_flagged(sample_source, con_umbral):
    _, permisivo = score_portfolio(sample_source, con_umbral(-99))
    _, estricto = score_portfolio(sample_source, con_umbral(99))

    assert permisivo["flagged_share"] == 1.0
    assert estricto["flagged_share"] == 0.0


def test_the_model_only_ever_sees_the_leakage_safe_view(sample_source, con_umbral):
    """score_portfolio must route through features(), not hand over the labelled frame."""
    from src.features.contract import features
    from src.models.heuristic import score_frame
    from src.pipelines.prepare import prepare_labelled

    scores, _ = score_portfolio(sample_source, con_umbral(4))

    assert scores.equals(score_frame(features(prepare_labelled(sample_source))))
