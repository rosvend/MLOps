"""The pipeline exists to make leakage impossible, so that is what these test."""

import numpy as np
import pytest
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from src.features.contract import features
from src.features.spec import default_spec
from src.models.pipelines import build_model
from src.pipelines.prepare import prepare_labelled


@pytest.fixture(scope="module")
def datos():
    from src.data.csv_source import CsvDataSource

    p = prepare_labelled(CsvDataSource("data/raw/BD_creditos.csv"))
    return features(p), ~p[default_spec().target]


def test_the_estimator_is_the_last_step():
    pipe = build_model("sklearn.linear_model.LogisticRegression", {"max_iter": 100})

    assert isinstance(pipe, Pipeline)
    assert pipe.steps[-1][0] == "model"
    assert pipe.steps[0][0] == "features"


def test_the_scaler_is_added_only_when_configured():
    con = build_model("sklearn.linear_model.LogisticRegression", scale=True)
    sin = build_model("sklearn.linear_model.LogisticRegression", scale=False)

    assert "scale" in dict(con.steps)
    assert "scale" not in dict(sin.steps)


def test_params_reach_the_estimator():
    pipe = build_model("sklearn.linear_model.LogisticRegression", {"C": 0.25})

    assert pipe.named_steps["model"].C == 0.25


def test_the_seed_reaches_estimators_that_take_one():
    pipe = build_model("sklearn.ensemble.RandomForestClassifier", {"n_estimators": 5}, seed=7)

    assert pipe.named_steps["model"].random_state == 7


# --- the leakage guarantee -----------------------------------------------------


def test_transformers_are_refitted_on_every_fold(datos):
    """Direct proof: if the winsor caps were fitted once globally they would be identical."""
    X, y = datos
    pipe = build_model("sklearn.linear_model.LogisticRegression", {"max_iter": 200}, scale=True)

    resultado = cross_validate(
        pipe, X, y, cv=StratifiedKFold(3, shuffle=True, random_state=0), return_estimator=True
    )

    caps = [est.named_steps["features"].caps_["dti"] for est in resultado["estimator"]]
    assert len(set(caps)) == len(caps), f"caps identical across folds: {caps}"


def test_a_fold_never_sees_the_statistics_of_the_whole_book(datos):
    """A fold's cap must come from its own training rows, not from all of them."""
    X, y = datos
    pipe = build_model("sklearn.linear_model.LogisticRegression", {"max_iter": 200})

    resultado = cross_validate(
        pipe, X, y, cv=StratifiedKFold(3, shuffle=True, random_state=0), return_estimator=True
    )
    global_cap = X["dti"].astype("float64").quantile(0.99)

    for est in resultado["estimator"]:
        assert est.named_steps["features"].caps_["dti"] != pytest.approx(global_cap)


def test_the_target_is_never_a_feature(datos):
    X, _ = datos

    assert default_spec().target not in X.columns
    assert not set(X.columns) & default_spec().no_son_features
