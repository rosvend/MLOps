from pathlib import Path

import pandas as pd

from src.config import DEFAULT_CONFIG_PATH, load_config
from src.data.factory import build_source
from src.data.schema import CreditoFeaturesSchema, CreditoLabelledSchema
from src.data.source import DataSource
from src.features.cleaning import clean
from src.features.contract import TARGET
from src.features.derive import add_derived_features


def prepare_features(source: DataSource) -> pd.DataFrame:
    """Scoring path: an application whose outcome is not yet known is still preparable."""
    df = add_derived_features(clean(source.read()))
    return CreditoFeaturesSchema.validate(df.drop(columns=[TARGET], errors="ignore"))


def prepare_labelled(source: DataSource) -> pd.DataFrame:
    """Training and evaluation path: the outcome is required."""
    return CreditoLabelledSchema.validate(add_derived_features(clean(source.read())))


def prepare(source: DataSource) -> pd.DataFrame:
    return prepare_labelled(source)


def prepare_from_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> pd.DataFrame:
    return prepare_labelled(build_source(load_config(config_path).data_source))
