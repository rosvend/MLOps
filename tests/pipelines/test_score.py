from src.pipelines.score import score_portfolio


def test_scoring_the_portfolio_returns_a_score_per_loan(sample_source):
    scores, _ = score_portfolio(sample_source, threshold=0)

    assert len(scores) == 14


def test_scoring_the_portfolio_reports_metrics(sample_source):
    _, metrics = score_portfolio(sample_source, threshold=0)

    assert {"auc", "gini", "precision", "recall", "flagged_share"} <= set(metrics)


def test_the_threshold_drives_how_many_loans_are_flagged(sample_source):
    _, permisivo = score_portfolio(sample_source, threshold=-99)
    _, estricto = score_portfolio(sample_source, threshold=99)

    assert permisivo["flagged_share"] == 1.0
    assert estricto["flagged_share"] == 0.0
