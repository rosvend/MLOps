"""Stage 5: the first transforms in this project that learn anything from the data.

Everything here is fitted on the training rows and applied to the test rows. These
tests exist to make that property fail loudly if it is ever broken.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from src.features.cleaning import TIPOS_CREDITO
from src.features.contract import TARGET, features
from src.features.engineering import COLAS_PESADAS, ESCALA_LOGARITMICA, FeatureEngineer
from src.models.estimator import CreditPreparer
from src.pipelines.prepare import prepare_labelled


@pytest.fixture
def X(sample_source):
    return features(prepare_labelled(sample_source))


@pytest.fixture
def real():
    from src.data.csv_source import CsvDataSource

    prepared = prepare_labelled(CsvDataSource("data/raw/BD_creditos.csv"))
    return features(prepared), ~prepared[TARGET]


# --- fitted on train, applied to test ----------------------------------------


def test_the_cap_comes_from_training_rows_only(real):
    X, y = real
    X_tr, X_te = train_test_split(X, test_size=0.3, random_state=0)
    columna = "total_otros_prestamos"

    fe = FeatureEngineer().fit(X_tr)

    assert fe.caps_[columna] == pytest.approx(X_tr[columna].astype("float64").quantile(0.99))
    assert fe.caps_[columna] != pytest.approx(X_te[columna].astype("float64").quantile(0.99))


def test_the_median_comes_from_training_rows_only(real):
    X, _ = real
    X_tr, X_te = train_test_split(X, test_size=0.3, random_state=0)

    fe = FeatureEngineer().fit(X_tr)

    assert fe.medianas_["salario_cliente"] == pytest.approx(
        X_tr["salario_cliente"].astype("float64").median()
    )


def test_a_test_row_above_the_training_cap_is_clipped_to_it(real):
    X, _ = real
    X_tr = X.head(500)
    fe = FeatureEngineer().fit(X_tr)
    extremo = X_tr.head(1).copy()
    extremo["total_otros_prestamos"] = 10**12

    salida = fe.transform(extremo)["total_otros_prestamos"].iloc[0]

    esperado = np.log1p(fe.caps_["total_otros_prestamos"])
    assert salida == pytest.approx(esperado)


def test_refitting_on_more_data_changes_the_caps(real):
    X, _ = real

    pequeno = FeatureEngineer().fit(X.head(500))
    grande = FeatureEngineer().fit(X)

    assert pequeno.caps_ != grande.caps_


# --- one row scores the same alone as in a batch ------------------------------


def test_a_fitted_transformer_is_row_independent(real):
    X, _ = real
    fe = FeatureEngineer().fit(X.head(500))

    completo = fe.transform(X.head(50))
    sola = fe.transform(X.head(1))

    assert np.allclose(sola.to_numpy(), completo.head(1).to_numpy(), equal_nan=True)


def test_the_batch_never_changes_what_a_row_becomes(real):
    X, _ = real
    fe = FeatureEngineer().fit(X.head(500))

    con_extremo = X.head(10).copy()
    con_extremo.loc[con_extremo.index[-1], "salario_cliente"] = 10**12

    assert np.allclose(
        fe.transform(con_extremo).head(9).to_numpy(),
        fe.transform(X.head(9)).to_numpy(),
        equal_nan=True,
    )


# --- output contract ----------------------------------------------------------


def test_the_output_is_numeric_and_complete(X):
    salida = FeatureEngineer().fit_transform(X)

    assert not salida.isna().any().any()
    assert all(str(d).startswith("float") for d in salida.dtypes)


def test_every_fixed_category_gets_a_column_even_when_absent(X):
    """Categories are frozen in cleaning, so encoding cannot drift between batches."""
    salida = FeatureEngineer().fit_transform(X)

    for nivel in TIPOS_CREDITO:
        assert f"tipo_credito_{nivel}" in salida.columns


def test_a_batch_missing_a_category_still_emits_its_column(real):
    X, _ = real
    fe = FeatureEngineer().fit(X)
    solo_uno = X[X["tipo_credito"] == "4"].head(20)

    salida = fe.transform(solo_uno)

    assert list(salida.columns) == list(fe.get_feature_names_out())


def test_heavy_tails_are_compressed(real):
    """Judged on the real portfolio: skew on 14 rows is noise."""
    X, _ = real
    salida = FeatureEngineer().fit_transform(X)

    for columna in ESCALA_LOGARITMICA:
        if columna not in X:
            continue
        antes = X[columna].astype("float64")
        # Skew is scale-invariant on a near-degenerate column, so a monotone transform
        # cannot move it; only judge columns with a real spread.
        if antes.nunique() < 5 or antes.skew() <= 1.0:
            continue
        assert salida[columna].skew() < antes.skew()


def test_missingness_survives_imputation(X):
    """Imputing is safe only because the falta_ flag still carries the information."""
    salida = FeatureEngineer().fit_transform(X)

    assert "falta_salario_cliente" in salida.columns


def test_untouched_columns_keep_their_values(X):
    salida = FeatureEngineer().fit_transform(X)

    assert salida["edad_cliente"].dropna().max() == X["edad_cliente"].dropna().max()


@pytest.mark.parametrize("columna", ["edad_cliente", "puntaje_datacredito", "plazo_meses"])
def test_well_behaved_columns_are_left_alone(columna):
    assert columna not in COLAS_PESADAS
    assert columna not in ESCALA_LOGARITMICA


# --- in a pipeline ------------------------------------------------------------


def test_the_heuristic_cannot_consume_engineered_features(real):
    """Deliberate: the rules read named raw columns, so encoding removes what they need.

    FeatureEngineer sits in front of a learned model, never in front of the scorecard.
    """
    from src.models.estimator import HeuristicModel

    X, y = real
    engineered = FeatureEngineer().fit_transform(X)
    modelo = HeuristicModel().fit(engineered, y)

    with pytest.raises(KeyError, match="rango_edad|tipo_laboral|tendencia_ingresos"):
        modelo.decision_function(engineered)


def test_engineered_features_beat_the_baseline_in_shuffled_cv(real):
    """Shuffled CV only. Out of time the heuristic still wins - see docs/feature-engineering.md."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    X, y = real
    pipe = Pipeline(
        [
            ("feat", FeatureEngineer()),
            ("scale", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )

    # Same CV as the heuristic's published 0.676, so the comparison is like for like.
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    auc = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")

    assert auc.mean() > 0.676  # the heuristic's out-of-fold AUC


def test_scaling_is_left_to_the_model_pipeline(real):
    """Winsorise/log/encode is representation; standardisation is model-specific."""
    X, _ = real
    salida = FeatureEngineer().fit_transform(X)

    assert salida["edad_cliente"].std() > 1.5
