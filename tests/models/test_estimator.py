"""The scorecard must behave like any other sklearn classifier, and leak nothing."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone, is_classifier
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from src.features.contract import TARGET
from src.models.estimator import (
    CreditPreparer,
    HeuristicModel,
    calibrated_model,
    credit_pipeline,
)
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
    assert is_classifier(HeuristicModel())


def test_params_round_trip_through_clone():
    original = HeuristicModel(threshold=7)

    copia = clone(original)

    assert copia.get_params()["threshold"] == 7
    assert copia is not original


def test_set_params_reaches_the_estimator():
    modelo = HeuristicModel().set_params(threshold=9)

    assert modelo.get_params()["threshold"] == 9


def test_the_constructor_stores_its_arguments_untouched():
    """sklearn's check_no_attributes_set_in_init: no validation or derived state in __init__."""
    modelo = HeuristicModel(threshold=3)

    publicos = {a for a in vars(modelo) if not a.startswith("_")}
    assert publicos == set(modelo.get_params())
    assert not any(a.endswith("_") for a in publicos)


def test_fit_returns_self(datos):
    X, y = datos
    modelo = HeuristicModel()

    assert modelo.fit(X, y) is modelo


def test_fit_records_the_label_space_and_the_feature_space(datos):
    X, y = datos

    modelo = HeuristicModel().fit(X, y)

    assert list(modelo.classes_) == [False, True]
    assert modelo.n_features_in_ == X.shape[1]
    assert "puntaje_datacredito" in list(modelo.feature_names_in_)


def test_predicting_before_fitting_raises(datos):
    X, _ = datos

    with pytest.raises(NotFittedError):
        HeuristicModel().predict(X)


def test_predict_returns_labels_from_the_declared_class_space(datos):
    X, y = datos

    pred = HeuristicModel().fit(X, y).predict(X)

    assert set(np.unique(pred)) <= set(HeuristicModel().fit(X, y).classes_)
    assert pred.shape == (len(X),)


# --- the score is a ranking, not a probability -------------------------------


def test_decision_function_is_the_raw_scorecard_score(datos):
    from src.models.heuristic import score_frame

    X, y = datos
    modelo = HeuristicModel().fit(X, y)

    assert list(modelo.decision_function(X)) == list(score_frame(X))


def test_decision_function_follows_the_fitted_contract(datos):
    """The rules need no fitting, but the estimator still honours sklearn's contract."""
    X, _ = datos

    with pytest.raises(NotFittedError):
        HeuristicModel().decision_function(X)


def test_the_rules_are_still_usable_without_an_estimator(datos):
    """score_frame stays the function-level API for scoring without fitting anything."""
    from src.models.heuristic import score_frame

    X, _ = datos

    assert len(score_frame(X)) == len(X)


def test_predicting_on_a_different_feature_space_is_caught(datos):
    X, y = datos
    modelo = HeuristicModel().fit(X, y)

    with pytest.raises(ValueError, match="feature names should match"):
        modelo.decision_function(X.drop(columns=["dti"]))


def test_the_bare_scorecard_offers_no_probability(datos):
    """The points are not a probability; nothing pretends otherwise."""
    X, y = datos

    assert not hasattr(HeuristicModel().fit(X, y), "predict_proba")


def test_predict_proba_returns_a_real_distribution(crudo):
    X, y = crudo
    pipe = credit_pipeline(cv=3).fit(X, y)

    proba = pipe.predict_proba(X)

    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_calibrated_probabilities_are_monotone_in_the_score(crudo):
    """Isotonic is monotone by construction, so calibration never reorders applicants."""
    X, y = crudo
    modelo = credit_pipeline(cv=3).fit(X, y)

    preparado = CreditPreparer().fit_transform(X)
    scores = HeuristicModel().fit(preparado, y).decision_function(preparado)
    pd_ = modelo.predict_proba(X)[:, 1]
    orden = np.argsort(scores, kind="stable")

    assert np.all(np.diff(pd_[orden]) >= -1e-9)


def test_no_applicant_is_declared_a_certain_default(crudo):
    """Isotonic fitted in-sample hands whole bands a probability of exactly 1.0."""
    X, y = crudo

    pd_ = credit_pipeline(cv=3).fit(X, y).predict_proba(X)[:, 1]

    assert pd_.max() < 1.0
    assert pd_.min() > 0.0


def test_the_calibration_never_sees_the_rows_it_scores(crudo):
    """Out-of-fold calibration: train and test agreement should be close.

    Fitting isotonic on the rows it then reports on made train log-loss look better
    than it is; cross-fitting removes most of that gap.
    """
    from sklearn.metrics import log_loss

    X, y = crudo
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    pipe = credit_pipeline(cv=3).fit(X_tr, y_tr)

    en_train = log_loss(y_tr, pipe.predict_proba(X_tr)[:, 1])
    en_test = log_loss(y_te, pipe.predict_proba(X_te)[:, 1])

    assert en_test - en_train < 0.02


def test_score_reports_ranking_power_not_accuracy(crudo):
    """At a 4.75% base rate, accuracy would rank a constant 'never defaults' model higher."""
    X, y = crudo
    modelo = Pipeline([("prep", CreditPreparer()), ("clf", HeuristicModel())]).fit(X, y)

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
    pipe = credit_pipeline(cv=3)

    entrenado = pipe.fit(X_tr, y_tr)
    sobre_todo = entrenado.predict_proba(X_te)[:, 1]
    sobre_mitad = entrenado.predict_proba(X_te.head(len(X_te) // 2))[:, 1]

    assert np.allclose(sobre_mitad, sobre_todo[: len(sobre_mitad)])


def test_seeing_more_test_rows_never_changes_an_applicants_probability(crudo):
    X, y = crudo
    pipe = credit_pipeline(cv=3).fit(X, y)

    sola = pipe.predict_proba(X.head(1))[0, 1]
    en_lote = pipe.predict_proba(X)[0, 1]

    assert sola == pytest.approx(en_lote)


def test_it_survives_cross_validation(crudo):
    X, y = crudo
    pipe = Pipeline([("prep", CreditPreparer()), ("clf", HeuristicModel())])

    puntajes = cross_val_score(pipe, X, y, cv=3, scoring="roc_auc")

    # Folds of ~4 loans tie easily; what this pins is that CV runs at all - the real
    # ranking power is measured on the full portfolio by make score.
    assert len(puntajes) == 3
    assert np.isfinite(puntajes).all()
    assert puntajes.mean() >= 0.5


# --- sklearn's own conformance checks ----------------------------------------
#
# check_estimator() reports success for both classes, but that result is empty: the
# DataFrame-only input tag makes it skip its entire suite (1 "check" runs). So the
# applicable checks are enumerated and run directly instead. Both lists were derived
# empirically - every check here genuinely executes against these estimators.

CHECKS_CLASIFICADOR = [
    "check_classifiers_one_label_sample_weights",
    "check_dataframe_column_names_consistency",
    "check_decision_proba_consistency",
    "check_do_not_raise_errors_in_init_or_set_params",
    "check_dont_overwrite_parameters",
    "check_estimator_cloneable",
    "check_estimator_repr",
    "check_estimator_tags_renamed",
    "check_estimators_fit_returns_self",
    "check_estimators_overwrite_params",
    "check_estimators_unfitted",
    "check_fit2d_1feature",
    "check_fit2d_1sample",
    "check_fit_check_is_fitted",
    "check_get_params_invariance",
    "check_mixin_order",
    "check_n_features_in",
    "check_n_features_in_after_fitting",
    "check_no_attributes_set_in_init",
    "check_parameters_default_constructible",
    "check_readonly_memmap_input",
    "check_requires_y_none",
    "check_set_params",
    "check_valid_tag_types",
]

CHECKS_TRANSFORMADOR = [
    "check_dataframe_column_names_consistency",
    "check_do_not_raise_errors_in_init_or_set_params",
    "check_estimator_cloneable",
    "check_estimator_repr",
    "check_estimator_tags_renamed",
    "check_estimators_unfitted",
    "check_get_feature_names_out_error",
    "check_get_params_invariance",
    "check_mixin_order",
    "check_n_features_in_after_fitting",
    "check_no_attributes_set_in_init",
    "check_param_validation",
    "check_parameters_default_constructible",
    "check_set_params",
    "check_transformers_unfitted",
    "check_valid_tag_types",
]


def _run(nombre: str, estimator) -> None:
    from sklearn.utils import estimator_checks

    getattr(estimator_checks, nombre)(type(estimator).__name__, estimator)


@pytest.mark.parametrize("check", CHECKS_CLASIFICADOR)
def test_sklearn_checks_on_the_scorecard(check):
    _run(check, HeuristicModel())


@pytest.mark.parametrize("check", CHECKS_TRANSFORMADOR)
def test_sklearn_checks_on_the_preparer(check):
    _run(check, CreditPreparer())


def test_the_check_suite_has_not_silently_collapsed():
    """check_estimator skips everything under a DataFrame-only tag; guard the real count."""
    assert len(CHECKS_CLASIFICADOR) >= 24
    assert len(CHECKS_TRANSFORMADOR) >= 16


def test_the_preparer_also_clones(crudo):
    X, y = crudo
    preparer = clone(CreditPreparer())

    assert preparer.fit(X, y).transform(X).shape[0] == len(X)


def test_fit_alone_establishes_everything_fit_earns(crudo):
    """get_feature_names_out must not depend on having gone through fit_transform."""
    X, y = crudo

    solo_fit = CreditPreparer().fit(X, y)

    assert list(solo_fit.get_feature_names_out()) == list(
        CreditPreparer().fit_transform(X, y).columns
    )
    assert solo_fit.n_features_in_ == X.shape[1]
    assert list(solo_fit.feature_names_in_) == list(X.columns)
