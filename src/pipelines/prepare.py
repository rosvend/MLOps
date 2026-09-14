from pathlib import Path

import pandas as pd

from src.config import DEFAULT_CONFIG_PATH, load_config
from src.data.factory import build_source
from src.data.schema import CreditoSchema
from src.data.source import DataSource
from src.features.cleaning import clean
from src.features.derive import add_derived_features


def prepare(source: DataSource) -> pd.DataFrame:
    """The seam every later stage attaches to: read, clean, derive, validate."""
    df = source.read()
    df = clean(df)
    df = add_derived_features(df)
    return CreditoSchema.validate(df)


def prepare_from_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> pd.DataFrame:
    return prepare(build_source(load_config(config_path).data_source))
