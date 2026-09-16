"""Stage 5: the first transforms in this project that learn from the data.

Everything before this point is a pure function of one row. These are not: a p99 cap
and a median are statistics of a sample. They are therefore fitted on the training
rows and applied to the test rows, which is what keeps the split honest.

Which columns get which treatment comes from config/features/default.yaml, as a frozen
list rather than a rule evaluated at fit time - selecting by measured skew would let the
output schema differ between folds. The measurements are in docs/feature-engineering.md.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data

from src.features.spec import FeatureSpec, default_spec

class FeatureEngineer(TransformerMixin, BaseEstimator):
    """Winsorise, log-compress, impute and encode. Every statistic comes from `fit`.

    Encoding needs no fitted state: `clean()` freezes the category sets, so a batch
    missing a level still emits that level's column. That is what stops the feature
    space drifting between training and serving.
    """

    def __init__(self, *, spec: FeatureSpec | None = None, winsor_quantile: float | None = None):
        self.spec = spec
        self.winsor_quantile = winsor_quantile

    def _spec(self) -> FeatureSpec:
        return self.spec or default_spec()

    def _quantile(self) -> float:
        if self.winsor_quantile is not None:
            return self.winsor_quantile
        return self._spec().engineering.winsor_quantile

    def _categoricas(self, X: pd.DataFrame) -> list[str]:
        """From the spec, not from the dtype: Feast returns strings where pandas returns
        categories, and the same column must be encoded either way."""
        return [c for c in self._spec().columns.categoricas if c in X.columns]

    def _numericas(self, X: pd.DataFrame) -> list[str]:
        categoricas = set(self._categoricas(X))
        return [c for c in X.columns if c not in categoricas]

    @staticmethod
    def _rechazar_columnas_opacas(X: pd.DataFrame) -> None:
        """Coercing anything that is not already a number is never harmless here.

        A timestamp coerces to int64 nanoseconds - the vintage itself, correlation 1.0
        with the loan date - and text coerces to NaN and then to a constant. Both are
        silent. The only valid input is the view built by src/features/contract.py.
        """
        opacas = {
            c: str(X[c].dtype)
            for c in X.columns
            if str(X[c].dtype) != "category" and not pd.api.types.is_numeric_dtype(X[c].dtype)
        }
        if opacas:
            raise ValueError(
                "FeatureEngineer sólo admite columnas numéricas o categóricas; aplica antes "
                f"src.features.contract.features(): {opacas}"
            )

    def _numerico(self, X: pd.DataFrame) -> pd.DataFrame:
        self._rechazar_columnas_opacas(X)
        marco = X[self._numericas(X)].copy()
        for columna in marco.columns:
            # errors="raise": the dtypes are already checked, so a NaN here would be a bug.
            marco[columna] = pd.to_numeric(marco[columna], errors="raise").astype("float64")
        return marco

    def fit(self, X: pd.DataFrame, y=None) -> "FeatureEngineer":
        validate_data(self, X=X, skip_check_array=True, reset=True)
        spec = self._spec()
        self.heavy_tailed_ = list(spec.engineering.heavy_tailed)
        self.log_scale_ = list(spec.engineering.log_scale)
        # ±inf sits on no scale: one of them makes a p99 NaN, which silently disables the
        # cap, and makes a median inf, which then imputes inf. Statistics ignore them.
        numerico = self._numerico(X).replace([np.inf, -np.inf], np.nan)
        self.caps_ = {
            c: float(numerico[c].quantile(self._quantile()))
            for c in self.heavy_tailed_
            if c in numerico and numerico[c].notna().any()
        }
        self.medianas_ = {
            c: (float(numerico[c].median()) if numerico[c].notna().any() else 0.0)
            for c in numerico.columns
        }
        self.feature_names_out_ = np.asarray(self._aplicar(X).columns, dtype=object)
        return self

    def _aplicar(self, X: pd.DataFrame) -> pd.DataFrame:
        numerico = self._numerico(X)
        for columna, tope in self.caps_.items():
            if columna in numerico:
                numerico[columna] = numerico[columna].clip(upper=tope)
        # An unpayable ratio is the extreme of the distribution, so the clip above already
        # mapped +inf onto the cap. What survives had no cap, so treat it as unknown.
        numerico = numerico.replace([np.inf, -np.inf], np.nan)
        for columna, mediana in self.medianas_.items():
            if columna in numerico:
                # Impute before the log, so the filled value is on the same scale as the
                # median it came from. Safe only because falta_<columna> keeps the fact.
                numerico[columna] = numerico[columna].fillna(mediana)
        for columna in self.log_scale_:
            if columna in numerico:
                numerico[columna] = np.log1p(numerico[columna].clip(lower=0))
        # Fixed vocabularies from the spec, so a batch missing a level still emits its column.
        categoricas = self._categoricas(X)
        if not categoricas:
            return numerico
        codificado = pd.get_dummies(
            X[categoricas].astype("object"), prefix_sep="_", dummy_na=False, dtype="float64"
        )
        return pd.concat([numerico, codificado.set_index(numerico.index)], axis=1)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self)
        validate_data(self, X=X, skip_check_array=True, reset=False)
        salida = self._aplicar(X)
        # Canonical order and membership, so a batch cannot change the feature space.
        return salida.reindex(columns=list(self.feature_names_out_), fill_value=0.0)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        check_is_fitted(self)
        return self.feature_names_out_

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.two_d_array = False
        return tags
