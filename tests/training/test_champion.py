"""Selection must be driven by the configured metric, and must be able to keep the incumbent."""

import pytest

from src.models.champion import CandidateResult, comparison_table, select_champion


def _c(name, pr_auc, gini, cv_std=0.01, tuned=True):
    return CandidateResult(
        name=name,
        cv_mean=pr_auc,
        cv_std=cv_std,
        metrics={"pr_auc": pr_auc, "gini": gini},
        fit_seconds=1.0,
        predict_ms_per_1k=1.0,
        tuned=tuned,
    )


def test_the_primary_metric_decides():
    ganador = select_champion([_c("a", 0.08, 0.40), _c("b", 0.12, 0.20)], "pr_auc")

    assert ganador.name == "b"


def test_a_different_primary_metric_can_pick_a_different_model():
    candidatos = [_c("a", 0.08, 0.40), _c("b", 0.12, 0.20)]

    assert select_champion(candidatos, "pr_auc").name == "b"
    assert select_champion(candidatos, "gini").name == "a"


def test_fold_stability_breaks_a_tie():
    inestable = _c("inestable", 0.10, 0.30, cv_std=0.05)
    estable = _c("estable", 0.10, 0.30, cv_std=0.01)

    assert select_champion([inestable, estable], "pr_auc").name == "estable"


def test_the_incumbent_can_win():
    """If the heuristic is best it stays; the selector is not there to justify a change."""
    candidatos = [_c("logistic", 0.07, 0.24), _c("heuristic", 0.11, 0.29, tuned=False)]

    assert select_champion(candidatos, "pr_auc").name == "heuristic"


def test_a_missing_metric_never_wins():
    sin_metrica = CandidateResult(name="roto", metrics={})

    assert select_champion([sin_metrica, _c("ok", 0.05, 0.1)], "pr_auc").name == "ok"


def test_no_candidates_is_an_error():
    with pytest.raises(ValueError):
        select_champion([], "pr_auc")


def test_the_table_reports_every_candidate_and_metric():
    tabla = comparison_table([_c("a", 0.08, 0.40), _c("b", 0.12, 0.20)], ["pr_auc", "gini"])

    assert list(tabla.index) == ["a", "b"]
    assert {"pr_auc", "gini", "cv_std", "fit_s", "predict_ms_1k"} <= set(tabla.columns)
