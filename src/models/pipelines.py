"""Pipeline construction for the training stage.

Every transformer lives inside the sklearn Pipeline, so cross-validation refits each one
on the training part of each fold. Nothing is fitted globally before splitting - that is
the whole point of building models this way rather than transforming up front.
"""

from importlib import import_module
from typing import Any

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features.engineering import FeatureEngineer
from src.features.spec import FeatureSpec, default_spec


def _resolve(ruta: str) -> type:
    """'xgboost.XGBClassifier' -> the class, so candidates can live in config."""
    modulo, _, nombre = ruta.rpartition(".")
    return getattr(import_module(modulo), nombre)


def build_model(
    estimator: str,
    params: dict[str, Any] | None = None,
    *,
    scale: bool = False,
    spec: FeatureSpec | None = None,
    seed: int | None = None,
) -> Pipeline:
    """A fitted-transforms-inside pipeline: engineer -> [scale] -> estimator."""
    spec = spec or default_spec()
    clase = _resolve(estimator)
    argumentos = dict(params or {})
    if seed is not None and "random_state" in clase().get_params():
        argumentos.setdefault("random_state", seed)

    pasos: list[tuple[str, BaseEstimator]] = [("features", FeatureEngineer(spec=spec))]
    if scale:
        pasos.append(("scale", StandardScaler()))
    pasos.append(("model", clase(**argumentos)))
    return Pipeline(pasos)
