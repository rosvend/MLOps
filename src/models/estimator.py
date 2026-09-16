"""scikit-learn surface for the heuristic scorecard.

The rules stay frozen pure functions; only the score-to-probability calibration is
learned, and it is learned from the training fold alone. That is what makes the
scorecard comparable against a trained model using the same fit/predict calls.

`y` is the default indicator: True means the loan defaulted.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
from sklearn.utils.validation import check_is_fitted

from src.features.cleaning import clean
from src.features.contract import features
from src.features.derive import add_derived_features
from src.models.heuristic import score_frame
from src.models.scorecard import Scorecard, default_scorecard


class CreditPreparer(TransformerMixin, BaseEstimator):
    """Stateless: read -> clean -> derive -> withhold the target and the leaky columns.

    Nothing is fitted, so a row transforms identically alone or inside a batch. The
    target is stripped here, so a model downstream cannot reach it even by accident.
    """

    def fit(self, X: pd.DataFrame, y=None) -> "CreditPreparer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return features(add_derived_features(clean(X)))

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        check_is_fitted(self, "feature_names_out_")
        return np.asarray(self.feature_names_out_, dtype=object)

    def fit_transform(self, X: pd.DataFrame, y=None, **kwargs) -> pd.DataFrame:
        salida = self.fit(X, y).transform(X)
        self.feature_names_out_ = np.asarray(salida.columns, dtype=object)
        return salida


class HeuristicScorecard(ClassifierMixin, BaseEstimator):
    """Frozen additive scorecard; higher score means higher risk.

    `decision_function` is the raw integer score and needs no fitting. `predict_proba`
    is a probability of default only because `fit` learns a monotone calibration over
    that score - the raw points are not a probability and are not treated as one.
    """

    def __init__(self, *, threshold: int | None = None, scorecard: Scorecard | None = None):
        self.threshold = threshold
        self.scorecard = scorecard

    def _spec(self) -> Scorecard:
        return self.scorecard or default_scorecard()

    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        """Frozen rules over one record at a time: no batch statistic, no fitted state."""
        return score_frame(X, self._spec()).to_numpy()

    def fit(self, X: pd.DataFrame, y) -> "HeuristicScorecard":
        spec = self._spec()
        y = np.asarray(y).astype(bool)
        self.classes_ = np.array([False, True])
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.scorecard_ = spec
        self.threshold_ = spec.threshold if self.threshold is None else self.threshold
        # Isotonic only: monotone by construction, so it also repairs the documented
        # score wobble without letting the calibration reorder applicants.
        self.calibrator_ = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(
            self.decision_function(X), y.astype(float)
        )
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, "calibrator_")
        pd_default = self.calibrator_.predict(self.decision_function(X))
        return np.column_stack([1.0 - pd_default, pd_default])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, "threshold_")
        return self.classes_[(self.decision_function(X) >= self.threshold_).astype(int)]

    def score(self, X: pd.DataFrame, y) -> float:
        """AUC, not accuracy: at a 4.75 % base rate accuracy rewards never flagging anyone."""
        check_is_fitted(self, "calibrator_")
        return float(roc_auc_score(np.asarray(y).astype(bool), self.decision_function(X)))

    def gini(self, X: pd.DataFrame, y) -> float:
        return 2 * self.score(X, y) - 1

    def explain(self, record) -> dict[str, int]:
        """Per-rule points behind one score; sklearn has no reason-code API."""
        from src.models.heuristic import explain

        return explain(record, self._spec())

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        # The rules address columns by name, so a bare ndarray is not valid input.
        tags.input_tags.two_d_array = False
        tags.target_tags.required = True
        return tags
