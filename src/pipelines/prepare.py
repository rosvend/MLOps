import pandas as pd

from src.config import load_config
from src.data.factory import build_source
from src.data.schema import CreditoFeaturesSchema, CreditoLabelledSchema
from src.data.source import DataSource
from src.features.cleaning import clean
from src.features.contract import TARGET
from src.features.derive import add_derived_features


def prepare_features_frame(df: pd.DataFrame) -> pd.DataFrame:
    """The scoring contract over a frame already in memory.

    Shared with the sklearn transformer so the two routes into the pipeline cannot
    drift into accepting different data.
    """
    derivado = add_derived_features(clean(df))
    return CreditoFeaturesSchema.validate(derivado.drop(columns=[TARGET], errors="ignore"))


def prepare_features(source: DataSource) -> pd.DataFrame:
    """Scoring path: an application whose outcome is not yet known is still preparable."""
    return prepare_features_frame(source.read())


def prepare_labelled(source: DataSource) -> pd.DataFrame:
    """Training and evaluation path: the outcome is required."""
    return CreditoLabelledSchema.validate(add_derived_features(clean(source.read())))


def prepare(source: DataSource) -> pd.DataFrame:
    return prepare_labelled(source)


def prepare_from_config(overrides: list[str] | None = None) -> pd.DataFrame:
    return prepare_labelled(build_source(load_config(overrides).data_source))
