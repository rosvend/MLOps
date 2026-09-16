import pandas as pd

from src.config import load_config
from src.data.factory import build_source
from src.data.schema import CreditoFeaturesSchema, CreditoLabelledSchema
from src.data.source import DataSource
from src.features.cleaning import clean
from src.features.spec import FeatureSpec, default_spec
from src.features.derive import add_derived_features


def prepare_features_frame(df: pd.DataFrame, spec: FeatureSpec | None = None) -> pd.DataFrame:
    """The scoring contract over a frame already in memory.

    Shared with the sklearn transformer so the two routes into the pipeline cannot
    drift into accepting different data. The spec is threaded all the way down: a caller
    that overrides it must get the cleaning and the derivation it asked for, not the
    defaults with its column roles applied on top.
    """
    spec = spec or default_spec()
    derivado = add_derived_features(clean(df, spec), spec)
    return CreditoFeaturesSchema.validate(derivado.drop(columns=[spec.target], errors="ignore"))


def prepare_features(source: DataSource, spec: FeatureSpec | None = None) -> pd.DataFrame:
    """Scoring path: an application whose outcome is not yet known is still preparable."""
    return prepare_features_frame(source.read(), spec)


def prepare_labelled(source: DataSource, spec: FeatureSpec | None = None) -> pd.DataFrame:
    """Training and evaluation path: the outcome is required."""
    spec = spec or default_spec()
    return CreditoLabelledSchema.validate(add_derived_features(clean(source.read(), spec), spec))


def prepare(source: DataSource) -> pd.DataFrame:
    return prepare_labelled(source)


def prepare_from_config(overrides: list[str] | None = None) -> pd.DataFrame:
    return prepare_labelled(build_source(load_config(overrides).data_source))
