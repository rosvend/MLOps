"""scikit-learn surface for the heuristic model.

The rules stay frozen pure functions. Responsibilities are split the way sklearn
splits them: `HeuristicModel` ranks, `calibrated_model()` turns a rank into a
probability, and it does so with a calibration that never sees the rows it scores.

Not to be confused with `Scorecard` in src/models/scorecard.py, which is the config
object holding the weights and cut-points - `HeuristicModel` is the estimator that
applies them.

`y` is the default indicator: True means the loan defaulted.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.utils.multiclass import type_of_target, unique_labels
from sklearn.utils.validation import check_is_fitted, validate_data

from src.features.contract import features
from src.models.heuristic import explain, score_frame
from src.models.scorecard import Scorecard, default_scorecard
from src.pipelines.prepare import prepare_features_frame

CALIBRATION_CV = 5


class CreditPreparer(TransformerMixin, BaseEstimator):
    """Stateless: read -> clean -> derive -> withhold the target and the leaky columns.

    Nothing is learned, so a row transforms identically alone or inside a batch. The
    target is stripped here, so a model downstream cannot reach it even by accident.
    """

    def _prepare(self, X: pd.DataFrame) -> pd.DataFrame:
        # Same contract as prepare_features: validate, then narrow to the model view.
        return features(prepare_features_frame(X))

    def _align(self, X: pd.DataFrame) -> pd.DataFrame:
        """Raw column order is not meaningful; a missing raw column is."""
        esperadas = list(self.feature_names_in_)
        faltantes = [c for c in esperadas if c not in X.columns]
        if faltantes:
            raise ValueError(f"Faltan columnas de entrada: {', '.join(faltantes)}")
        sobrantes = [c for c in X.columns if c not in set(esperadas)]
        return X[esperadas + sobrantes]

    def fit(self, X: pd.DataFrame, y=None) -> "CreditPreparer":
        # skip_check_array: the rules address columns by name, so X must stay a frame.
        validate_data(self, X=X, skip_check_array=True, reset=True)
        self.feature_names_out_ = np.asarray(self._prepare(X).columns, dtype=object)
        return self

    def fit_transform(self, X: pd.DataFrame, y=None, **kwargs) -> pd.DataFrame:
        validate_data(self, X=X, skip_check_array=True, reset=True)
        salida = self._prepare(X)
        self.feature_names_out_ = np.asarray(salida.columns, dtype=object)
        return salida

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self)
        X = self._align(X)
        validate_data(self, X=X, skip_check_array=True, reset=False)
        # Canonical output order, so a downstream estimator sees a stable feature space.
        return self._prepare(X)[list(self.feature_names_out_)]

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        check_is_fitted(self)
        return self.feature_names_out_

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.two_d_array = False
        tags.no_validation = False
        return tags


class HeuristicModel(ClassifierMixin, BaseEstimator):
    """The heuristic baseline as an sklearn classifier; higher score means higher risk.

    A ranker, not a probability model. `decision_function` is the raw integer score and
    is a pure function of one application - no batch statistic, no learned state. There
    is deliberately no `predict_proba`: the points are not a probability, and rescaling
    them into [0, 1] would only make them look like one. For a calibrated probability of
    default use `calibrated_model()`, which fits the calibration out-of-fold.
    """

    def __init__(self, *, threshold: int | None = None, scorecard: Scorecard | None = None):
        self.threshold = threshold
        self.scorecard = scorecard

    def _spec(self) -> Scorecard:
        return self.scorecard or default_scorecard()

    def fit(self, X: pd.DataFrame, y) -> "HeuristicModel":
        """Learns nothing from the rules' point of view; it fixes the label and feature space."""
        X, y = validate_data(self, X=X, y=y, skip_check_array=True, reset=True)
        # Labels come from y rather than being assumed: astype(bool) would read
        # ["0", "1"] as two defaults and calibrate against nonsense. A loan either
        # defaulted or it did not, so anything but a binary target is a mistake.
        y_type = type_of_target(y, input_name="y", raise_unknown=True)
        if y_type != "binary":
            raise ValueError(
                f"Only binary classification is supported. The type of the target is {y_type}."
            )
        self.classes_ = unique_labels(y)
        self.scorecard_ = self._spec()
        self.threshold_ = self.scorecard_.threshold if self.threshold is None else self.threshold
        return self

    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        """The raw integer score, checked against the feature space fit established.

        The rules themselves need no fitting - `score_frame()` is the function-level API
        for that - but the estimator follows sklearn's contract so it is interchangeable
        with any other classifier.
        """
        check_is_fitted(self)
        X = validate_data(self, X=X, skip_check_array=True, reset=False)
        return score_frame(X, self.scorecard_).to_numpy()

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self)
        return self.classes_[(self.decision_function(X) >= self.threshold_).astype(int)]

    def score(self, X: pd.DataFrame, y) -> float:
        """AUC, not accuracy: at a 4.75 % base rate accuracy rewards never flagging anyone."""
        check_is_fitted(self)
        en_mora = np.asarray(y) == self.classes_[1]
        return float(roc_auc_score(en_mora, self.decision_function(X)))

    def gini(self, X: pd.DataFrame, y) -> float:
        return 2 * self.score(X, y) - 1

    def explain(self, record) -> dict[str, int]:
        """Per-rule points behind one score; sklearn has no reason-code API."""
        return explain(record, self._spec())

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        # The rules address columns by name, so a bare ndarray is not valid input.
        tags.input_tags.two_d_array = False
        tags.target_tags.required = True
        tags.classifier_tags.multi_class = False
        return tags


def calibrated_model(cv: int = CALIBRATION_CV, **kwargs) -> CalibratedClassifierCV:
    """Probability of default, from a calibration that never sees the rows it scores.

    Isotonic fitted on the same rows it then reports on is optimistic and, on a 21-point
    scale, assigns some bands a probability of exactly 1.0 - a claim no credit model can
    make. CalibratedClassifierCV fits one calibrator per fold on the other folds and
    averages them, which removes both problems.
    """
    return CalibratedClassifierCV(HeuristicModel(**kwargs), method="isotonic", cv=cv)


def credit_pipeline(cv: int = CALIBRATION_CV, **kwargs) -> Pipeline:
    """prepare -> score -> calibrate, as one estimator."""
    return Pipeline([("prep", CreditPreparer()), ("clf", calibrated_model(cv, **kwargs))])
