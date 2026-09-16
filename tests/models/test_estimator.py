"""The scorecard must behave like any other sklearn classifier, and leak nothing."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone, is_classifier
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from src.features.contract import TARGET
from src.models.estimator import CreditPreparer, HeuristicScorecard
from src.pipelines.prepare import prepare_labelled


@pytest.fixture
def datos(sample_source):
    prepared = prepare_labelled(sample_source)
    return prepared, ~prepared[TARGET]


@pytest.fixture
def crudo(raw):
    return raw, ~raw[TARGET].astype(bool)


# --- sklearn API conformance -------------------------------------------------


def test_it_is_recognised_as_a_classifier():
    assert is_classifier(HeuristicScorecard())


def test_params_round_trip_through_clone():
    original = HeuristicScorecard(threshold=7)

    copia = clone(original)

    assert copia.get_params()["threshold"] == 7
    assert copia is not original


def test_set_params_reaches_the_estimator():
    modelo = HeuristicScorecard().set_params(threshold=9)

    assert modelo.get_params()["threshold"] == 9


def test_the_constructor_stores_its_arguments_untouched():
    """sklearn's check_no_attributes_set_in_init: no validation or derived state in __init__."""
    modelo = HeuristicScorecard(threshold=3)

    publicos = {a for a in vars(modelo) if not a.startswith("_")}
    assert publicos == set(modelo.get_params())
    assert not any(a.endswith("_") for a in publicos)


def test_fit_returns_self(datos):
    X, y = datos
    modelo = HeuristicScorecard()

    assert modelo.fit(X, y) is modelo


def test_fit_records_the_label_space_and_the_feature_space(datos):
    X, y = datos

    modelo = HeuristicScorecard().fit(X, y)

    assert list(modelo.classes_) == [False, True]
    assert modelo.n_features_in_ == X.shape[1]
    assert "puntaje_datacredito" in list(modelo.feature_names_in_)


def test_predicting_before_fitting_raises(datos):
    X, _ = datos

    with pytest.raises(NotFittedError):
        HeuristicScorecard().predict(X)


def test_predict_returns_labels_from_the_declared_class_space(datos):
    X, y = datos

    pred = HeuristicScorecard().fit(X, y).predict(X)

    assert set(np.unique(pred)) <= set(HeuristicScorecard().fit(X, y).classes_)
    assert pred.shape == (len(X),)


# --- the score is a ranking, not a probability -------------------------------


def test_decision_function_is_the_raw_scorecard_score(datos):
    from src.models.heuristic import score_frame

    X, y = datos
    modelo = HeuristicScorecard().fit(X, y)

    assert list(modelo.decision_function(X)) == list(score_frame(X))


def test_decision_function_needs_no_fit(datos):
    """The rules are frozen; only the calibration is learned."""
    X, _ = datos

    assert len(HeuristicScorecard().decision_function(X)) == len(X)


def test_predict_proba_is_unavailable_until_calibrated(datos):
    X, _ = datos

    with pytest.raises(NotFittedError):
        HeuristicScorecard().predict_proba(X)


def test_predict_proba_returns_a_real_distribution(crudo):
    X, y = crudo
    pipe = Pipeline([("prep", CreditPreparer()), ("clf", HeuristicScorecard())]).fit(X, y)

    proba = pipe.predict_proba(X)

    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_calibrated_probabilities_are_monotone_in_the_score(crudo):
    X, y = crudo
    modelo = Pipeline([("prep", CreditPreparer()), ("clf", HeuristicScorecard())]).fit(X, y)

    scores = modelo.decision_function(X)
    pd_ = modelo.predict_proba(X)[:, 1]
    orden = np.argsort(scores)

    assert np.all(np.diff(pd_[orden]) >= -1e-9)


def test_score_reports_ranking_power_not_accuracy(crudo):
    """At a 4.75% base rate, accuracy would rank a constant 'never defaults' model higher."""
    X, y = crudo
    modelo = Pipeline([("prep", CreditPreparer()), ("clf", HeuristicScorecard())]).fit(X, y)

    assert 0.5 < modelo.score(X, y) < 1.0


# --- isolation ---------------------------------------------------------------


def test_the_preparer_never_hands_the_target_to_the_model(crudo):
    X, _ = crudo

    transformado = CreditPreparer().fit_transform(X)

    assert TARGET not in transformado.columns


def test_the_preparer_is_stateless(crudo):
    """Nothing is fitted, so a row transforms identically alone or in a batch."""
    X, y = crudo
    preparer = CreditPreparer().fit(X, y)

    completo = preparer.transform(X)
    parcial = preparer.transform(X.head(5))

    assert parcial.equals(completo.head(5))


def test_the_calibration_is_fitted_on_train_only(crudo):
    X, y = crudo
    X_tr, X_te, y_tr, _ = train_test_split(X, y, test_size=0.4, random_state=0)
    pipe = Pipeline([("prep", CreditPreparer()), ("clf", HeuristicScorecard())])

    entrenado = pipe.fit(X_tr, y_tr)
    sobre_todo = entrenado.predict_proba(X_te)[:, 1]
    sobre_mitad = entrenado.predict_proba(X_te.head(len(X_te) // 2))[:, 1]

    assert np.allclose(sobre_mitad, sobre_todo[: len(sobre_mitad)])


def test_seeing_more_test_rows_never_changes_an_applicants_probability(crudo):
    X, y = crudo
    pipe = Pipeline([("prep", CreditPreparer()), ("clf", HeuristicScorecard())]).fit(X, y)

    sola = pipe.predict_proba(X.head(1))[0, 1]
    en_lote = pipe.predict_proba(X)[0, 1]

    assert sola == pytest.approx(en_lote)


def test_it_survives_cross_validation(crudo):
    X, y = crudo
    pipe = Pipeline([("prep", CreditPreparer()), ("clf", HeuristicScorecard())])

    puntajes = cross_val_score(pipe, X, y, cv=3, scoring="roc_auc")

    # Folds of ~4 loans tie easily; what this pins is that CV runs at all - the real
    # ranking power is measured on the full portfolio by make score.
    assert len(puntajes) == 3
    assert np.isfinite(puntajes).all()
    assert puntajes.mean() >= 0.5


# --- sklearn's own conformance checks ----------------------------------------


@pytest.mark.parametrize(
    "check",
    [
        "check_no_attributes_set_in_init",
        "check_get_params_invariance",
        "check_set_params",
        "check_estimator_repr",
    ],
)
def test_sklearns_api_checks_pass(check):
    """The API half of check_estimator, run explicitly.

    Full check_estimator reports success only because the DataFrame-only tag makes it
    skip every data check - the rules address columns by name, so a bare ndarray is not
    valid input. Those skipped behaviours are covered by the hand-written tests above.
    """
    from sklearn.utils import estimator_checks

    getattr(estimator_checks, check)("HeuristicScorecard", HeuristicScorecard())


def test_the_preparer_also_clones(crudo):
    X, y = crudo
    preparer = clone(CreditPreparer())

    assert preparer.fit(X, y).transform(X).shape[0] == len(X)
