"""Credit-risk metrics: PR-AUC leads, Gini is always reported."""

import numpy as np
import pytest

from src.models.metrics import METRICAS, summarize_classification


@pytest.fixture
def desbalanceado():
    """1:20, like the book: 50 defaults in 1000."""
    rng = np.random.default_rng(0)
    y = np.zeros(1000, dtype=bool)
    y[rng.choice(1000, 50, replace=False)] = True
    return y


def test_it_reports_every_metric_the_stage_requires(desbalanceado):
    resumen = summarize_classification(desbalanceado, np.linspace(0, 1, 1000))

    assert set(METRICAS) <= set(resumen)
    for nombre in ("roc_auc", "pr_auc", "gini", "f1", "precision", "recall", "brier"):
        assert nombre in resumen


def test_a_perfect_ranking(desbalanceado):
    resumen = summarize_classification(desbalanceado, desbalanceado.astype(float))

    assert resumen["roc_auc"] == pytest.approx(1.0)
    assert resumen["gini"] == pytest.approx(1.0)
    assert resumen["pr_auc"] == pytest.approx(1.0)


def test_an_inverted_ranking(desbalanceado):
    resumen = summarize_classification(desbalanceado, 1.0 - desbalanceado.astype(float))

    assert resumen["roc_auc"] == pytest.approx(0.0)
    assert resumen["gini"] == pytest.approx(-1.0)


def test_a_constant_score_is_a_coin_flip(desbalanceado):
    resumen = summarize_classification(desbalanceado, np.full(len(desbalanceado), 0.5))

    assert resumen["roc_auc"] == pytest.approx(0.5)
    assert resumen["gini"] == pytest.approx(0.0)
    assert resumen["pr_auc"] == pytest.approx(desbalanceado.mean(), abs=1e-6)


def test_pr_auc_is_the_metric_that_notices_imbalance(desbalanceado):
    """A useless model scores 0.5 on ROC but only the base rate on PR."""
    resumen = summarize_classification(desbalanceado, np.full(len(desbalanceado), 0.5))

    assert resumen["roc_auc"] == pytest.approx(0.5)
    assert resumen["pr_auc"] < 0.10


def test_the_confusion_matrix_counts_every_case(desbalanceado):
    resumen = summarize_classification(desbalanceado, desbalanceado.astype(float))

    conteos = resumen["confusion_matrix"]
    assert sum(conteos.values()) == len(desbalanceado)
    assert conteos["tp"] == int(desbalanceado.sum())
    assert conteos["fp"] == 0


def test_brier_rewards_a_calibrated_probability(desbalanceado):
    calibrado = np.where(desbalanceado, 0.9, 0.05)
    descalibrado = np.where(desbalanceado, 0.55, 0.45)

    assert (
        summarize_classification(desbalanceado, calibrado)["brier"]
        < summarize_classification(desbalanceado, descalibrado)["brier"]
    )


def test_the_threshold_moves_precision_and_recall(desbalanceado):
    puntajes = np.linspace(0, 1, len(desbalanceado))

    laxo = summarize_classification(desbalanceado, puntajes, threshold=0.1)
    estricto = summarize_classification(desbalanceado, puntajes, threshold=0.9)

    assert laxo["recall"] > estricto["recall"]


def test_brier_is_withheld_for_a_non_probabilistic_score(desbalanceado):
    """The scorecard emits integer points; scoring them against Brier means nothing."""
    import math

    puntos = np.where(desbalanceado, 9.0, -3.0)

    resumen = summarize_classification(desbalanceado, puntos, probabilistic=False)

    assert math.isnan(resumen["brier"])
    assert resumen["roc_auc"] == pytest.approx(1.0)
